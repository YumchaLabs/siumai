//! Stream event bridging utilities.
//!
//! This module provides best-effort transformations between provider-specific
//! Vercel-aligned stream parts (`ChatStreamEvent::Custom`) so that a stream can
//! be re-serialized into another provider's protocol (gateway/proxy use-cases).
//!
//! English-only comments in code as requested.

use std::collections::{HashMap, HashSet};

use crate::streaming::{ChatStreamEvent, LanguageModelV3StreamPart};

/// Bridges Gemini/Anthropic Vercel-aligned stream parts into OpenAI Responses
/// stream parts (`openai:*`) that can be serialized by the OpenAI Responses SSE
/// converter.
///
/// This is intentionally conservative: unknown/unsupported custom event types
/// are passed through unchanged.
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
        match event_type.as_str() {
            // Gemini: multiplexed custom events
            "gemini:tool" => self.bridge_tool_like_custom_event(&data),
            // Legacy Gemini: split tool events (kept for backward compatibility)
            "gemini:tool-call" | "gemini:tool-result" => self.bridge_tool_like_custom_event(&data),
            "gemini:source" => self.rename_custom("openai:source", data),
            "gemini:reasoning" => self.bridge_reasoning_multiplexed_custom_event(&data),

            // Anthropic: already split into multiple custom event types
            "anthropic:stream-start" => self.rename_custom("openai:stream-start", data),
            "anthropic:response-metadata" => self.rename_custom("openai:response-metadata", data),
            "anthropic:text-start" => self.rename_custom("openai:text-start", data),
            "anthropic:text-delta" => self.rename_custom("openai:text-delta", data),
            "anthropic:text-end" => self.rename_custom("openai:text-end", data),
            "anthropic:reasoning-start" => {
                self.bridge_anthropic_reasoning_start_end("openai:reasoning-start", &data)
            }
            "anthropic:reasoning-end" => {
                self.bridge_anthropic_reasoning_start_end("openai:reasoning-end", &data)
            }
            "anthropic:tool-call" => self.bridge_tool_call_custom_event(&data),
            "anthropic:tool-result" => self.bridge_tool_result_custom_event(&data),
            "anthropic:source" => self.rename_custom("openai:source", data),
            "anthropic:finish" => self.rename_custom("openai:finish", data),

            _ => {
                if let Some(out) = self.bridge_v3_custom_event(&data) {
                    return out;
                }
                vec![ChatStreamEvent::Custom { event_type, data }]
            }
        }
    }

    fn rename_custom(&self, new_event_type: &str, data: serde_json::Value) -> Vec<ChatStreamEvent> {
        vec![ChatStreamEvent::Custom {
            event_type: new_event_type.to_string(),
            data,
        }]
    }

    fn bridge_v3_custom_event(&mut self, data: &serde_json::Value) -> Option<Vec<ChatStreamEvent>> {
        let tpe = data.get("type").and_then(|v| v.as_str())?;

        match tpe {
            "stream-start" => Some(self.rename_custom("openai:stream-start", data.clone())),
            "response-metadata" => {
                Some(self.rename_custom("openai:response-metadata", data.clone()))
            }
            "text-start" => Some(self.rename_custom("openai:text-start", data.clone())),
            "text-delta" => Some(self.rename_custom("openai:text-delta", data.clone())),
            "text-end" => Some(self.rename_custom("openai:text-end", data.clone())),
            "reasoning-start" => Some(self.rename_custom("openai:reasoning-start", data.clone())),
            "reasoning-delta" => Some(self.rename_custom("openai:reasoning-delta", data.clone())),
            "reasoning-end" => Some(self.rename_custom("openai:reasoning-end", data.clone())),
            "tool-input-start" => Some(self.rename_custom("openai:tool-input-start", data.clone())),
            "tool-input-delta" => Some(self.rename_custom("openai:tool-input-delta", data.clone())),
            "tool-input-end" => Some(self.rename_custom("openai:tool-input-end", data.clone())),
            "tool-approval-request" => {
                Some(self.rename_custom("openai:tool-approval-request", data.clone()))
            }
            "tool-call" => Some(self.bridge_tool_call_custom_event(data)),
            "tool-result" => Some(self.bridge_tool_result_custom_event(data)),
            "source" => Some(self.rename_custom("openai:source", data.clone())),
            "finish" => Some(self.rename_custom("openai:finish", data.clone())),
            "error" => Some(self.rename_custom("openai:error", data.clone())),
            _ => {
                let _ = LanguageModelV3StreamPart::parse_loose_json(data)?;
                None
            }
        }
    }

    fn bridge_reasoning_multiplexed_custom_event(
        &self,
        data: &serde_json::Value,
    ) -> Vec<ChatStreamEvent> {
        let tpe = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("");

        let (event_type, mut out_data) = match tpe {
            "reasoning-start" => ("openai:reasoning-start", serde_json::json!({ "id": id })),
            "reasoning-delta" => (
                "openai:reasoning-delta",
                serde_json::json!({
                    "id": id,
                    "delta": data.get("delta").cloned().unwrap_or(serde_json::Value::Null),
                }),
            ),
            "reasoning-end" => ("openai:reasoning-end", serde_json::json!({ "id": id })),
            _ => {
                return vec![ChatStreamEvent::Custom {
                    event_type: "gemini:reasoning".to_string(),
                    data: data.clone(),
                }];
            }
        };

        if let Some(pm) = data.get("providerMetadata")
            && let Some(obj) = out_data.as_object_mut()
        {
            obj.insert("providerMetadata".to_string(), pm.clone());
        }

        vec![ChatStreamEvent::Custom {
            event_type: event_type.to_string(),
            data: out_data,
        }]
    }

    fn bridge_anthropic_reasoning_start_end(
        &self,
        new_event_type: &str,
        data: &serde_json::Value,
    ) -> Vec<ChatStreamEvent> {
        let idx = data
            .get("contentBlockIndex")
            .and_then(|v| v.as_u64())
            .map(|v| v.to_string())
            .unwrap_or_default();
        if idx.is_empty() {
            return vec![ChatStreamEvent::Custom {
                event_type: new_event_type.to_string(),
                data: serde_json::json!({}),
            }];
        }
        vec![ChatStreamEvent::Custom {
            event_type: new_event_type.to_string(),
            data: serde_json::json!({ "id": idx }),
        }]
    }

    fn bridge_tool_like_custom_event(&mut self, data: &serde_json::Value) -> Vec<ChatStreamEvent> {
        let tpe = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match tpe {
            "tool-call" => self.bridge_tool_call_custom_event(data),
            "tool-result" => self.bridge_tool_result_custom_event(data),
            _ => vec![ChatStreamEvent::Custom {
                event_type: "gemini:tool".to_string(),
                data: data.clone(),
            }],
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

        let mut payload = serde_json::json!({
            "type": "tool-call",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "input": input_str,
            "providerExecuted": data.get("providerExecuted").and_then(|v| v.as_bool()).unwrap_or(true),
            "dynamic": data.get("dynamic").and_then(|v| v.as_bool()).unwrap_or(false),
            // The OpenAI Responses serializer requires `rawItem` for provider tool results.
            // We also attach a best-effort `rawItem` on tool-call so a downstream gateway
            // can emit output_item.added consistently.
            "rawItem": {
                "id": data.get("toolCallId").cloned().unwrap_or(serde_json::Value::Null),
                "type": "custom_tool_call",
                "status": "in_progress",
                "name": data.get("toolName").cloned().unwrap_or(serde_json::Value::Null),
                "input": normalize_json_string_value(data.get("input")).unwrap_or(serde_json::Value::String("{}".to_string())),
            }
        });
        if let Some(pm) = data.get("providerMetadata")
            && let Some(obj) = payload.as_object_mut()
        {
            obj.insert("providerMetadata".to_string(), pm.clone());
        }
        if let Some(title) = data.get("title")
            && let Some(obj) = payload.as_object_mut()
        {
            obj.insert("title".to_string(), title.clone());
        }

        vec![ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: payload,
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
            let mut tool_call_payload = serde_json::json!({
                "type": "tool-call",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": input_str,
                "providerExecuted": provider_executed,
                "dynamic": dynamic,
                "rawItem": {
                    "id": data.get("toolCallId").cloned().unwrap_or(serde_json::Value::Null),
                    "type": "custom_tool_call",
                    "status": "in_progress",
                    "name": data.get("toolName").cloned().unwrap_or(serde_json::Value::Null),
                    "input": serde_json::Value::String(input_str.clone()),
                }
            });
            if let Some(pm) = data.get("providerMetadata")
                && let Some(obj) = tool_call_payload.as_object_mut()
            {
                obj.insert("providerMetadata".to_string(), pm.clone());
            }
            out.push(ChatStreamEvent::Custom {
                event_type: "openai:tool-call".to_string(),
                data: tool_call_payload,
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

        let mut result_payload = serde_json::json!({
            "type": "tool-result",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "providerExecuted": provider_executed,
            "dynamic": dynamic,
            // The OpenAI Responses serializer currently requires `rawItem` here.
            "rawItem": {
                "id": data.get("toolCallId").cloned().unwrap_or(serde_json::Value::Null),
                "type": "custom_tool_call",
                "status": "completed",
                "name": data.get("toolName").cloned().unwrap_or(serde_json::Value::Null),
                "input": serde_json::Value::String(input_str),
                "output": result,
                "is_error": is_error,
            }
        });
        if let Some(pm) = data.get("providerMetadata")
            && let Some(obj) = result_payload.as_object_mut()
        {
            obj.insert("providerMetadata".to_string(), pm.clone());
        }
        out.push(ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: result_payload,
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

fn normalize_json_string_value(value: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    let value = value?;
    match value {
        serde_json::Value::String(s) => Some(serde_json::Value::String(s.clone())),
        serde_json::Value::Null => None,
        other => serde_json::to_string(other)
            .ok()
            .map(serde_json::Value::String),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let ChatStreamEvent::Custom { event_type, data } = &out[0] else {
            panic!("expected Custom");
        };
        assert_eq!(event_type, "openai:tool-call");
        assert!(data.get("rawItem").is_some());
        assert_eq!(
            data.get("providerMetadata")
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

        let ChatStreamEvent::Custom {
            event_type: call_event_type,
            data: call_data,
        } = &out[0]
        else {
            panic!("expected tool-call Custom");
        };
        assert_eq!(call_event_type, "openai:tool-call");
        assert_eq!(
            call_data
                .get("rawItem")
                .and_then(|v| v.get("status"))
                .and_then(|v| v.as_str()),
            Some("in_progress")
        );
        assert_eq!(
            call_data
                .get("providerMetadata")
                .and_then(|v| v.get("anthropic"))
                .and_then(|v| v.get("requestId"))
                .and_then(|v| v.as_str()),
            Some("r1")
        );

        let ChatStreamEvent::Custom {
            event_type: result_event_type,
            data: result_data,
        } = &out[1]
        else {
            panic!("expected tool-result Custom");
        };
        assert_eq!(result_event_type, "openai:tool-result");
        assert_eq!(
            result_data
                .get("rawItem")
                .and_then(|v| v.get("status"))
                .and_then(|v| v.as_str()),
            Some("completed")
        );
    }

    #[test]
    fn bridge_v3_tool_input_parts_are_renamed() {
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

        let ChatStreamEvent::Custom { event_type, data } = &out[0] else {
            panic!("expected Custom");
        };
        assert_eq!(event_type, "openai:tool-input-start");
        assert_eq!(data.get("id").and_then(|v| v.as_str()), Some("call_1"));
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
}
