//! Anthropic provider-hosted tools helpers (streaming).
//!
//! Anthropic's Messages API can execute certain tools on the provider side (e.g. web search).
//! Siumai surfaces these as `ChatStreamEvent::Custom` events so the unified streaming enum
//! remains stable.

use crate::streaming::ChatStreamEvent;

/// Anthropic streaming custom events emitted by Siumai.
///
/// Current events (beta.5 guidance):
/// - `anthropic:tool-call` — provider-hosted tool call started (`content_block_start.server_tool_use`)
/// - `anthropic:tool-result` — provider-hosted tool result emitted (`content_block_start.*_tool_result`)
#[derive(Debug, Clone)]
pub enum AnthropicCustomEvent {
    ProviderToolCall(AnthropicProviderToolCallEvent),
    ProviderToolResult(AnthropicProviderToolResultEvent),
    Source(AnthropicSourceEvent),
}

impl AnthropicCustomEvent {
    pub fn from_stream_event(event: &ChatStreamEvent) -> Option<Self> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
