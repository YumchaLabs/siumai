//! Anthropic provider-hosted tools helpers (streaming).
//!
//! Anthropic's Messages API can execute certain tools on the provider side (e.g. web search).
//! Siumai now exposes the stable `Part` lane for these semantics, and this helper keeps the
//! historical custom-event shadow only as a compatibility fallback.

use super::request_options::merge_anthropic_provider_option_object;
use crate::provider_options::anthropic::AnthropicToolOptions;
use crate::streaming::ChatStreamEvent;
use crate::types::{Tool, ToolFunction};

/// Anthropic function-tool helpers.
pub trait AnthropicToolExt {
    /// Attach typed Anthropic tool options under `providerOptions.anthropic`.
    ///
    /// For `Tool::ProviderDefined`, this is a no-op because Anthropic tool options currently apply
    /// only to user-defined function tools.
    fn with_anthropic_tool_options(self, options: AnthropicToolOptions) -> Self;
}

impl AnthropicToolExt for ToolFunction {
    fn with_anthropic_tool_options(mut self, options: AnthropicToolOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize AnthropicToolOptions");
        merge_anthropic_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl AnthropicToolExt for Tool {
    fn with_anthropic_tool_options(self, options: AnthropicToolOptions) -> Self {
        match self {
            Self::Function { function } => Self::Function {
                function: function.with_anthropic_tool_options(options),
            },
            other => other,
        }
    }
}

/// Anthropic streaming extension events emitted by Siumai.
///
/// Current events:
/// - stable provider-hosted `tool-call` / `tool-result` parts
/// - stable URL `source` parts for hosted search results
/// - legacy `anthropic:*` custom-event shadows for compatibility
#[derive(Debug, Clone)]
pub enum AnthropicCustomEvent {
    ProviderToolCall(AnthropicProviderToolCallEvent),
    ProviderToolResult(AnthropicProviderToolResultEvent),
    Source(AnthropicSourceEvent),
}

impl AnthropicCustomEvent {
    pub fn from_stream_event(event: &ChatStreamEvent) -> Option<Self> {
        if let Some(part) = event.part_ref() {
            if let Some(tool_call) = AnthropicProviderToolCallEvent::from_part(part) {
                return Some(AnthropicCustomEvent::ProviderToolCall(tool_call));
            }

            if let Some(tool_result) = AnthropicProviderToolResultEvent::from_part(part) {
                return Some(AnthropicCustomEvent::ProviderToolResult(tool_result));
            }

            if let Some(source) = AnthropicSourceEvent::from_part(part) {
                return Some(AnthropicCustomEvent::Source(source));
            }
        }

        let ChatStreamEvent::Custom { event_type, data } = event else {
            return None;
        };

        if event_type == AnthropicProviderToolCallEvent::EVENT_TYPE {
            return AnthropicProviderToolCallEvent::from_custom(event_type, data)
                .map(AnthropicCustomEvent::ProviderToolCall);
        }

        if event_type == AnthropicProviderToolResultEvent::EVENT_TYPE {
            return AnthropicProviderToolResultEvent::from_custom(event_type, data)
                .map(AnthropicCustomEvent::ProviderToolResult);
        }

        if event_type == AnthropicSourceEvent::EVENT_TYPE {
            return AnthropicSourceEvent::from_custom(event_type, data)
                .map(AnthropicCustomEvent::Source);
        }

        None
    }
}

/// Provider-hosted tool call event (`anthropic:tool-call`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicProviderToolCallEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
    pub provider_executed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content_block: Option<serde_json::Value>,
}

impl AnthropicProviderToolCallEvent {
    pub const EVENT_TYPE: &'static str = "anthropic:tool-call";

    pub fn from_custom(event_type: &str, data: &serde_json::Value) -> Option<Self> {
        if event_type != Self::EVENT_TYPE {
            return None;
        }
        serde_json::from_value(data.clone()).ok()
    }

    pub fn from_part(part: &crate::types::ChatStreamPart) -> Option<Self> {
        let crate::types::ChatStreamPart::ToolCall(tool_call) = part else {
            return None;
        };

        if !tool_call.provider_executed.unwrap_or(false) {
            return None;
        }

        Some(Self {
            kind: "tool-call".to_string(),
            tool_call_id: tool_call.tool_call_id.clone(),
            tool_name: tool_call.tool_name.clone(),
            input: serde_json::from_str(&tool_call.input)
                .unwrap_or_else(|_| serde_json::Value::String(tool_call.input.clone())),
            provider_executed: true,
            content_block_index: None,
            raw_content_block: None,
        })
    }
}

/// Provider-hosted tool result event (`anthropic:tool-result`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicProviderToolResultEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub result: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    pub provider_executed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content_block: Option<serde_json::Value>,
}

impl AnthropicProviderToolResultEvent {
    pub const EVENT_TYPE: &'static str = "anthropic:tool-result";

    pub fn from_custom(event_type: &str, data: &serde_json::Value) -> Option<Self> {
        if event_type != Self::EVENT_TYPE {
            return None;
        }
        serde_json::from_value(data.clone()).ok()
    }

    pub fn from_part(part: &crate::types::ChatStreamPart) -> Option<Self> {
        let crate::types::ChatStreamPart::ToolResult(tool_result) = part else {
            return None;
        };

        if !has_anthropic_provider_metadata(tool_result.provider_metadata.as_ref())
            && tool_result.dynamic != Some(true)
        {
            return None;
        }

        Some(Self {
            kind: "tool-result".to_string(),
            tool_call_id: tool_result.tool_call_id.clone(),
            tool_name: tool_result.tool_name.clone(),
            result: tool_result.result.clone(),
            is_error: tool_result.is_error,
            provider_executed: true,
            content_block_index: None,
            raw_content_block: None,
        })
    }
}

/// Source event (`anthropic:source`), typically emitted from web search tool results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicSourceEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub source_type: String,
    pub id: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

impl AnthropicSourceEvent {
    pub const EVENT_TYPE: &'static str = "anthropic:source";

    pub fn from_custom(event_type: &str, data: &serde_json::Value) -> Option<Self> {
        if event_type != Self::EVENT_TYPE {
            return None;
        }
        serde_json::from_value(data.clone()).ok()
    }

    pub fn from_part(part: &crate::types::ChatStreamPart) -> Option<Self> {
        let crate::types::ChatStreamPart::Source {
            id,
            source,
            provider_metadata,
        } = part
        else {
            return None;
        };

        let crate::types::SourcePart::Url { url, title } = source else {
            return None;
        };

        Some(Self {
            kind: "source".to_string(),
            source_type: "url".to_string(),
            id: id.clone(),
            url: url.clone(),
            title: title.clone(),
            tool_call_id: anthropic_source_metadata_str(provider_metadata.as_ref(), "toolCallId"),
            provider_metadata: provider_metadata
                .as_ref()
                .and_then(|metadata| serde_json::to_value(metadata).ok()),
        })
    }
}

fn has_anthropic_provider_metadata(
    provider_metadata: Option<&crate::types::StreamProviderMetadata>,
) -> bool {
    provider_metadata
        .map(|metadata| metadata.contains_key("anthropic"))
        .unwrap_or(false)
}

fn anthropic_source_metadata_str(
    provider_metadata: Option<&crate::types::StreamProviderMetadata>,
    key: &str,
) -> Option<String> {
    provider_metadata
        .and_then(|metadata| metadata.get("anthropic"))
        .and_then(|value| value.as_object())
        .and_then(|metadata| metadata.get(key))
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_anthropic_runtime_part_provider_tool_events() {
        let tool_call = ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolCall(crate::types::ChatStreamToolCall {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                input: r#"{"query":"rust"}"#.to_string(),
                provider_executed: Some(true),
                dynamic: None,
                provider_metadata: None,
            }),
        };

        match AnthropicCustomEvent::from_stream_event(&tool_call).unwrap() {
            AnthropicCustomEvent::ProviderToolCall(ev) => {
                assert_eq!(ev.tool_call_id, "srvtoolu_1");
                assert_eq!(ev.tool_name, "web_search");
                assert!(ev.provider_executed);
                assert_eq!(ev.input["query"], serde_json::json!("rust"));
                assert_eq!(ev.content_block_index, None);
            }
            _ => panic!("expected tool call"),
        }

        let mut tool_result_metadata = std::collections::HashMap::new();
        tool_result_metadata.insert(
            "anthropic".to_string(),
            serde_json::json!({
                "serverToolName": "web_search_20260209",
            }),
        );
        let tool_result = ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolResult(crate::types::ChatStreamToolResult {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                result: serde_json::json!([
                    {
                        "url": "https://www.rust-lang.org",
                        "title": "Rust",
                        "type": "web_search_result"
                    }
                ]),
                is_error: Some(false),
                preliminary: None,
                dynamic: None,
                provider_metadata: Some(tool_result_metadata),
            }),
        };

        match AnthropicCustomEvent::from_stream_event(&tool_result).unwrap() {
            AnthropicCustomEvent::ProviderToolResult(ev) => {
                assert_eq!(ev.tool_call_id, "srvtoolu_1");
                assert_eq!(ev.tool_name, "web_search");
                assert!(ev.provider_executed);
                assert!(ev.result.is_array());
                assert_eq!(ev.content_block_index, None);
            }
            _ => panic!("expected tool result"),
        }
    }

    #[test]
    fn parses_anthropic_runtime_part_source_event() {
        let source = ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Source {
                id: "srvtoolu_1:0".to_string(),
                source: crate::types::SourcePart::Url {
                    url: "https://www.rust-lang.org".to_string(),
                    title: Some("Rust".to_string()),
                },
                provider_metadata: None,
            },
        };

        match AnthropicCustomEvent::from_stream_event(&source).unwrap() {
            AnthropicCustomEvent::Source(ev) => {
                assert_eq!(ev.source_type, "url");
                assert_eq!(ev.url, "https://www.rust-lang.org");
                assert_eq!(ev.tool_call_id, None);
            }
            _ => panic!("expected source"),
        }
    }

    #[test]
    fn parses_anthropic_custom_provider_tool_events() {
        let tool_call = ChatStreamEvent::Custom {
            event_type: "anthropic:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "srvtoolu_1",
                "toolName": "web_search",
                "input": {"query":"rust"},
                "providerExecuted": true,
                "contentBlockIndex": 0,
            }),
        };

        match AnthropicCustomEvent::from_stream_event(&tool_call).unwrap() {
            AnthropicCustomEvent::ProviderToolCall(ev) => {
                assert_eq!(ev.tool_call_id, "srvtoolu_1");
                assert_eq!(ev.tool_name, "web_search");
                assert!(ev.provider_executed);
                assert_eq!(ev.content_block_index, Some(0));
                assert_eq!(ev.input["query"], serde_json::json!("rust"));
            }
            _ => panic!("expected tool call"),
        }

        let tool_result = ChatStreamEvent::Custom {
            event_type: "anthropic:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "srvtoolu_1",
                "toolName": "web_search",
                "result": [{"url":"https://www.rust-lang.org","title":"Rust","type":"web_search_result","encrypted_content":"..."}],
                "providerExecuted": true,
            }),
        };

        match AnthropicCustomEvent::from_stream_event(&tool_result).unwrap() {
            AnthropicCustomEvent::ProviderToolResult(ev) => {
                assert_eq!(ev.tool_call_id, "srvtoolu_1");
                assert_eq!(ev.tool_name, "web_search");
                assert!(ev.provider_executed);
                assert!(ev.result.is_array());
            }
            _ => panic!("expected tool result"),
        }

        let source = ChatStreamEvent::Custom {
            event_type: "anthropic:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "url",
                "id": "srvtoolu_1:0",
                "url": "https://www.rust-lang.org",
                "title": "Rust",
                "toolCallId": "srvtoolu_1",
            }),
        };

        match AnthropicCustomEvent::from_stream_event(&source).unwrap() {
            AnthropicCustomEvent::Source(ev) => {
                assert_eq!(ev.source_type, "url");
                assert_eq!(ev.url, "https://www.rust-lang.org");
                assert_eq!(ev.tool_call_id.as_deref(), Some("srvtoolu_1"));
            }
            _ => panic!("expected source"),
        }
    }

    #[test]
    fn tool_ext_merges_existing_anthropic_provider_options() {
        let tool = Tool::function(
            "weather",
            "Get weather",
            serde_json::json!({ "type": "object", "properties": {} }),
        )
        .with_anthropic_tool_options(
            crate::provider_options::anthropic::AnthropicToolOptions::new()
                .with_defer_loading(true)
                .with_eager_input_streaming(true),
        )
        .with_anthropic_tool_options(
            crate::provider_options::anthropic::AnthropicToolOptions::new().with_allowed_callers([
                crate::provider_options::anthropic::AnthropicToolAllowedCaller::Direct,
            ]),
        );

        let function = tool.function_ref().expect("function tool");
        let options = function
            .provider_options_map
            .get("anthropic")
            .and_then(|value| value.as_object())
            .expect("anthropic provider options");

        assert_eq!(options.get("deferLoading"), Some(&serde_json::json!(true)));
        assert_eq!(
            options.get("eagerInputStreaming"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            options.get("allowedCallers"),
            Some(&serde_json::json!(["direct"]))
        );
    }
}
