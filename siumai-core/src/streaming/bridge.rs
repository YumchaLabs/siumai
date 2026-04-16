//! Stream event bridging utilities.
//!
//! This module provides best-effort transformations between provider-specific
//! legacy/custom stream-part carriers and the stronger runtime semantic part
//! lane so that a stream can be re-serialized into another provider's protocol
//! (gateway/proxy use-cases).
//!
//! English-only comments in code as requested.

use std::collections::{HashMap, HashSet};

use crate::streaming::{ChatStreamEvent, LanguageModelV3StreamPart};
use crate::types::{
    ChatStreamPart, ChatStreamReplay, ChatStreamToolCall, ChatStreamToolResult,
    StreamProviderMetadata,
};

/// Bridges Gemini/Anthropic legacy stream-part carriers into OpenAI Responses-
/// compatible runtime events.
///
/// When the incoming custom payload already matches the audited V4-capable
/// stream-part shape, the bridge upgrades it onto `ChatStreamEvent::Part`
/// directly. Tool call/result parts still get `PartWithReplay` so downstream
/// OpenAI Responses serializers keep same-protocol raw-item fidelity.
///
/// Unknown or target-unsupported custom event types are passed through
/// unchanged.
#[derive(Debug, Default, Clone)]
pub struct OpenAiResponsesStreamPartsBridge {
    emitted_tool_call_ids: HashSet<String>,
    tool_input_by_call_id: HashMap<String, String>,
    tool_name_by_call_id: HashMap<String, String>,
}

impl OpenAiResponsesStreamPartsBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a single `ChatStreamEvent` into zero or more events.
    ///
    /// - Standard events are passed through unchanged.
    /// - Provider-specific custom events are mapped into `openai:*` stream parts
    ///   when possible.
    pub fn bridge_event(&mut self, ev: ChatStreamEvent) -> Vec<ChatStreamEvent> {
        match ev {
            ChatStreamEvent::Custom { event_type, data } => {
                self.bridge_custom_event(event_type, data)
            }
            other => vec![other],
        }
    }

    fn bridge_custom_event(
        &mut self,
        event_type: String,
        data: serde_json::Value,
    ) -> Vec<ChatStreamEvent> {
        if let Some(out) = self.bridge_v3_custom_event(&data) {
            return out;
        }

        vec![ChatStreamEvent::Custom { event_type, data }]
    }

    fn bridge_v3_custom_event(&mut self, data: &serde_json::Value) -> Option<Vec<ChatStreamEvent>> {
        let part = LanguageModelV3StreamPart::parse_loose_json(data)?;

        match part {
            LanguageModelV3StreamPart::ToolCall(_) => {
                Some(self.bridge_tool_call_custom_event(data))
            }
            LanguageModelV3StreamPart::ToolResult(_) => {
                Some(self.bridge_tool_result_custom_event(data))
            }
            LanguageModelV3StreamPart::StreamStart { .. }
            | LanguageModelV3StreamPart::ResponseMetadata(_)
            | LanguageModelV3StreamPart::TextStart { .. }
            | LanguageModelV3StreamPart::TextDelta { .. }
            | LanguageModelV3StreamPart::TextEnd { .. }
            | LanguageModelV3StreamPart::ReasoningStart { .. }
            | LanguageModelV3StreamPart::ReasoningDelta { .. }
            | LanguageModelV3StreamPart::ReasoningEnd { .. }
            | LanguageModelV3StreamPart::ToolInputStart { .. }
            | LanguageModelV3StreamPart::ToolInputDelta { .. }
            | LanguageModelV3StreamPart::ToolInputEnd { .. }
            | LanguageModelV3StreamPart::ToolApprovalRequest(_)
            | LanguageModelV3StreamPart::Source(_)
            | LanguageModelV3StreamPart::Finish { .. }
            | LanguageModelV3StreamPart::Error { .. } => Some(vec![part.to_part_event()]),
            LanguageModelV3StreamPart::Raw { .. }
            | LanguageModelV3StreamPart::Custom(_)
            | LanguageModelV3StreamPart::File(_)
            | LanguageModelV3StreamPart::ReasoningFile(_) => None,
        }
    }

    fn bridge_tool_call_custom_event(&mut self, data: &serde_json::Value) -> Vec<ChatStreamEvent> {
        let tool_call_id = data
            .get("toolCallId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tool_name = data
            .get("toolName")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if tool_call_id.is_empty() || tool_name.is_empty() {
            return Vec::new();
        }

        let input_str =
            normalize_json_string(data.get("input")).unwrap_or_else(|| "{}".to_string());

        self.tool_input_by_call_id
            .insert(tool_call_id.clone(), input_str.clone());
        self.tool_name_by_call_id
            .insert(tool_call_id.clone(), tool_name.clone());

        self.emitted_tool_call_ids.insert(tool_call_id.clone());

        let raw_item = openai_provider_tool_call_raw_item(&tool_call_id, &tool_name, &input_str);
        let replay = ChatStreamReplay::openai_responses(None, Some(raw_item)).expect("replay");

        vec![ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id,
                tool_name,
                input: input_str,
                provider_executed: Some(
                    data.get("providerExecuted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true),
                ),
                dynamic: Some(
                    data.get("dynamic")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                ),
                provider_metadata: provider_metadata_from_value(data.get("providerMetadata")),
            }),
            replay,
        }]
    }

    fn bridge_tool_result_custom_event(
        &mut self,
        data: &serde_json::Value,
    ) -> Vec<ChatStreamEvent> {
        let tool_call_id = data
            .get("toolCallId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tool_name = data
            .get("toolName")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if tool_call_id.is_empty() || tool_name.is_empty() {
            return Vec::new();
        }

        let provider_executed = data
            .get("providerExecuted")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let dynamic = data
            .get("dynamic")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Ensure we have a stable input string for the final raw item.
        let input_str = self
            .tool_input_by_call_id
            .get(&tool_call_id)
            .cloned()
            .unwrap_or_else(|| "{}".to_string());
        self.tool_name_by_call_id
            .entry(tool_call_id.clone())
            .or_insert_with(|| tool_name.clone());

        let mut out = Vec::new();

        // If the upstream provider only produced a tool-result (no explicit tool-call),
        // synthesize a tool-call scaffold so downstream clients see output_item.added.
        if !self.emitted_tool_call_ids.contains(&tool_call_id) {
            self.emitted_tool_call_ids.insert(tool_call_id.clone());
            let raw_item =
                openai_provider_tool_call_raw_item(&tool_call_id, &tool_name, &input_str);
            out.push(ChatStreamEvent::PartWithReplay {
                part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    input: input_str.clone(),
                    provider_executed: Some(provider_executed),
                    dynamic: Some(dynamic),
                    provider_metadata: provider_metadata_from_value(data.get("providerMetadata")),
                }),
                replay: ChatStreamReplay::openai_responses(None, Some(raw_item)).expect("replay"),
            });
        }

        let result = data
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let is_error = data
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let raw_item = openai_provider_tool_result_raw_item(
            &tool_call_id,
            &tool_name,
            &input_str,
            &result,
            is_error,
        );
        out.push(ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolResult(ChatStreamToolResult {
                tool_call_id,
                tool_name,
                result,
                is_error: Some(is_error),
                preliminary: None,
                dynamic: Some(dynamic),
                provider_metadata: provider_metadata_from_value(data.get("providerMetadata")),
            }),
            replay: ChatStreamReplay::openai_responses(None, Some(raw_item)).expect("replay"),
        });

        out
    }
}

fn normalize_json_string(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?;
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Null => None,
        other => serde_json::to_string(other).ok(),
    }
}

fn provider_metadata_from_value(
    value: Option<&serde_json::Value>,
) -> Option<StreamProviderMetadata> {
    let serde_json::Value::Object(obj) = value?.clone() else {
        return None;
    };

    Some(obj.into_iter().collect())
}

fn openai_provider_tool_call_raw_item(
    tool_call_id: &str,
    tool_name: &str,
    input_str: &str,
) -> serde_json::Value {
    if let Some(mcp_name) = tool_name.strip_prefix("mcp.") {
        return serde_json::json!({
            "id": tool_call_id,
            "type": "mcp_call",
            "status": "in_progress",
            "name": mcp_name,
            "arguments": input_str,
        });
    }

    serde_json::json!({
        "id": tool_call_id,
        "type": "custom_tool_call",
        "status": "in_progress",
        "name": tool_name,
        "input": input_str,
    })
}

fn openai_provider_tool_result_raw_item(
    tool_call_id: &str,
    tool_name: &str,
    input_str: &str,
    result: &serde_json::Value,
    is_error: bool,
) -> serde_json::Value {
    if let Some(mcp_name) = tool_name.strip_prefix("mcp.") {
        let (name, arguments, output, server_label) = result
            .as_object()
            .map(|obj| {
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(mcp_name)
                    .to_string();
                let arguments = obj
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or(input_str)
                    .to_string();
                let output = obj.get("output").cloned().unwrap_or_else(|| result.clone());
                let server_label = obj
                    .get("serverLabel")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string);
                (name, arguments, output, server_label)
            })
            .unwrap_or_else(|| {
                (
                    mcp_name.to_string(),
                    input_str.to_string(),
                    result.clone(),
                    None,
                )
            });

        let mut raw_item = serde_json::json!({
            "id": tool_call_id,
            "type": "mcp_call",
            "status": "completed",
            "name": name,
            "arguments": arguments,
            "output": output,
        });
        if let Some(server_label) = server_label
            && let Some(obj) = raw_item.as_object_mut()
        {
            obj.insert(
                "server_label".to_string(),
                serde_json::Value::String(server_label),
            );
        }
        return raw_item;
    }

    serde_json::json!({
        "id": tool_call_id,
        "type": "custom_tool_call",
        "status": "completed",
        "name": tool_name,
        "input": input_str,
        "output": result,
        "is_error": is_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay_raw_item(event: &ChatStreamEvent) -> &serde_json::Value {
        event
            .replay_ref()
            .and_then(ChatStreamReplay::openai_responses_ref)
            .and_then(|replay| replay.raw_item.as_ref())
            .expect("openai responses raw item")
    }

    #[test]
    fn bridge_v3_tool_call_from_unknown_prefix_adds_raw_item_and_preserves_provider_metadata() {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();

        let in_event = ChatStreamEvent::Custom {
            event_type: "custom:any".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "tc_1",
                "toolName": "web_search",
                "input": "{\"q\":\"hello\"}",
                "providerMetadata": { "gemini": { "traceId": "t1" } }
            }),
        };

        let out = bridge.bridge_event(in_event);
        assert_eq!(out.len(), 1);

        let ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolCall(call),
            ..
        } = &out[0]
        else {
            panic!("expected tool-call part with replay");
        };
        assert!(replay_raw_item(&out[0]).is_object());
        assert_eq!(
            call.provider_metadata
                .as_ref()
                .and_then(|v| v.get("gemini"))
                .and_then(|v| v.get("traceId"))
                .and_then(|v| v.as_str()),
            Some("t1")
        );
    }

    #[test]
    fn bridge_v3_tool_result_synthesizes_tool_call_when_missing() {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();

        let in_event = ChatStreamEvent::Custom {
            event_type: "custom:any".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "tc_2",
                "toolName": "web_search",
                "result": { "ok": true },
                "providerMetadata": { "anthropic": { "requestId": "r1" } }
            }),
        };

        let out = bridge.bridge_event(in_event);
        assert_eq!(out.len(), 2);

        let ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolCall(call),
            ..
        } = &out[0]
        else {
            panic!("expected tool-call part with replay");
        };
        assert_eq!(
            replay_raw_item(&out[0])
                .get("status")
                .and_then(|v| v.as_str()),
            Some("in_progress")
        );
        assert_eq!(
            call.provider_metadata
                .as_ref()
                .and_then(|v| v.get("anthropic"))
                .and_then(|v| v.get("requestId"))
                .and_then(|v| v.as_str()),
            Some("r1")
        );

        let ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolResult(_),
            ..
        } = &out[1]
        else {
            panic!("expected tool-result part with replay");
        };
        assert_eq!(
            replay_raw_item(&out[1])
                .get("status")
                .and_then(|v| v.as_str()),
            Some("completed")
        );
    }

    #[test]
    fn bridge_v3_tool_input_parts_are_promoted_to_runtime_parts() {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();

        let in_event = ChatStreamEvent::Custom {
            event_type: "custom:any".to_string(),
            data: serde_json::json!({
                "type": "tool-input-start",
                "id": "call_1",
                "toolName": "web_search",
            }),
        };

        let out = bridge.bridge_event(in_event);
        assert_eq!(out.len(), 1);

        let ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart { id, tool_name, .. },
        } = &out[0]
        else {
            panic!("expected stable tool-input-start part");
        };
        assert_eq!(id, "call_1");
        assert_eq!(tool_name, "web_search");
    }

    #[test]
    fn bridge_v3_text_and_source_parts_are_promoted_to_runtime_parts() {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();

        let text_start = ChatStreamEvent::Custom {
            event_type: "anthropic:text-start".to_string(),
            data: serde_json::json!({
                "type": "text-start",
                "id": "0",
                "providerMetadata": { "anthropic": { "contentBlockIndex": 0 } }
            }),
        };
        let source = ChatStreamEvent::Custom {
            event_type: "gemini:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "url",
                "id": "src_1",
                "url": "https://example.com",
                "title": "Example",
                "providerMetadata": { "gemini": { "traceId": "trace_1" } }
            }),
        };

        let bridged_text = bridge.bridge_event(text_start);
        assert_eq!(bridged_text.len(), 1);
        match &bridged_text[0] {
            ChatStreamEvent::Part {
                part:
                    ChatStreamPart::TextStart {
                        id,
                        provider_metadata,
                    },
            } => {
                assert_eq!(id, "0");
                assert_eq!(
                    provider_metadata
                        .as_ref()
                        .and_then(|value| value.get("anthropic"))
                        .and_then(|value| value.get("contentBlockIndex"))
                        .and_then(|value| value.as_u64()),
                    Some(0)
                );
            }
            other => panic!("expected stable text-start part, got {other:?}"),
        }

        let bridged_source = bridge.bridge_event(source);
        assert_eq!(bridged_source.len(), 1);
        match &bridged_source[0] {
            ChatStreamEvent::Part {
                part:
                    ChatStreamPart::Source {
                        id,
                        source: crate::types::SourcePart::Url { url, title },
                        provider_metadata,
                    },
            } => {
                assert_eq!(id, "src_1");
                assert_eq!(url, "https://example.com");
                assert_eq!(title.as_deref(), Some("Example"));
                assert_eq!(
                    provider_metadata
                        .as_ref()
                        .and_then(|value| value.get("gemini"))
                        .and_then(|value| value.get("traceId"))
                        .and_then(|value| value.as_str()),
                    Some("trace_1")
                );
            }
            other => panic!("expected stable source part, got {other:?}"),
        }
    }

    #[test]
    fn bridge_v3_finish_parts_are_promoted_to_runtime_parts() {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();

        let in_event = ChatStreamEvent::Custom {
            event_type: "custom:any".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "usage": {
                    "inputTokens": { "total": 3, "noCache": 3, "cacheRead": 0, "cacheWrite": 0 },
                    "outputTokens": { "total": 5, "text": 5, "reasoning": 0 },
                    "raw": { "native": true }
                },
                "finishReason": { "unified": "stop", "raw": "stop" },
                "providerMetadata": { "anthropic": { "stopSequence": null } }
            }),
        };

        let out = bridge.bridge_event(in_event);
        assert_eq!(out.len(), 1);

        match &out[0] {
            ChatStreamEvent::Part {
                part:
                    ChatStreamPart::Finish {
                        usage,
                        finish_reason,
                        provider_metadata,
                    },
            } => {
                assert_eq!(usage.normalized_input_tokens().total, Some(3));
                assert_eq!(usage.normalized_output_tokens().total, Some(5));
                assert_eq!(finish_reason.unified, crate::types::FinishReason::Stop);
                assert_eq!(finish_reason.raw.as_deref(), Some("stop"));
                assert_eq!(
                    provider_metadata
                        .as_ref()
                        .and_then(|value| value.get("anthropic"))
                        .and_then(|value| value.get("stopSequence")),
                    Some(&serde_json::Value::Null)
                );
            }
            other => panic!("expected stable finish part, got {other:?}"),
        }
    }

    #[test]
    fn bridge_v3_raw_parts_are_not_rewritten() {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();

        let in_event = ChatStreamEvent::Custom {
            event_type: "custom:raw".to_string(),
            data: serde_json::json!({
                "type": "raw",
                "rawValue": { "hello": "world" }
            }),
        };

        let out = bridge.bridge_event(in_event);
        assert_eq!(out.len(), 1);

        let ChatStreamEvent::Custom { event_type, .. } = &out[0] else {
            panic!("expected Custom");
        };
        assert_eq!(event_type, "custom:raw");
    }

    #[test]
    fn bridge_v3_mcp_tool_events_rebuild_openai_mcp_raw_items() {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();

        let call = ChatStreamEvent::Custom {
            event_type: "custom:any".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "mcp_1",
                "toolName": "mcp.web_search_exa",
                "input": "{\"query\":\"nyc mayor\"}",
                "providerExecuted": true,
            }),
        };
        let result = ChatStreamEvent::Custom {
            event_type: "custom:any".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "mcp_1",
                "toolName": "mcp.web_search_exa",
                "providerExecuted": true,
                "result": {
                    "type": "call",
                    "serverLabel": "exa",
                    "name": "web_search_exa",
                    "arguments": "{\"query\":\"nyc mayor\"}",
                    "output": { "hits": 3 }
                }
            }),
        };

        let bridged_call = bridge.bridge_event(call);
        assert_eq!(bridged_call.len(), 1);
        let call_data = replay_raw_item(&bridged_call[0]);
        assert_eq!(
            call_data.pointer("/type").and_then(|v| v.as_str()),
            Some("mcp_call")
        );
        assert_eq!(
            call_data.pointer("/name").and_then(|v| v.as_str()),
            Some("web_search_exa")
        );
        assert_eq!(
            call_data.pointer("/arguments").and_then(|v| v.as_str()),
            Some("{\"query\":\"nyc mayor\"}")
        );

        let bridged_result = bridge.bridge_event(result);
        assert_eq!(bridged_result.len(), 1);
        let result_data = replay_raw_item(&bridged_result[0]);
        assert_eq!(
            result_data.pointer("/type").and_then(|v| v.as_str()),
            Some("mcp_call")
        );
        assert_eq!(
            result_data
                .pointer("/server_label")
                .and_then(|v| v.as_str()),
            Some("exa")
        );
        assert_eq!(
            result_data.pointer("/output/hits").and_then(|v| v.as_u64()),
            Some(3)
        );
    }
}
