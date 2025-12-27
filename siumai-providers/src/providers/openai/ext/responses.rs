//! OpenAI Responses API extension helpers.
//!
//! This module provides a small, explicit API surface for interacting with the
//! Responses API through the existing unified chat pipeline.

use crate::error::LlmError;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, OpenAiOptions, ResponsesApiConfig};

/// Execute a chat request via OpenAI Responses API (explicit extension API).
///
/// Notes:
/// - This uses the existing `ChatCapability::chat_request` pipeline; it does not introduce
///   a new client type.
/// - You can still use `ChatRequest::with_openai_options(...)` directly; this helper exists
///   to make the “provider extension” intent obvious in user code.
pub async fn chat_via_responses_api<C>(
    client: &C,
    mut request: ChatRequest,
    config: ResponsesApiConfig,
) -> Result<ChatResponse, LlmError>
where
    C: crate::traits::ChatCapability + ?Sized,
{
    request = request.with_openai_options(OpenAiOptions::new().with_responses_api(config));
    client.chat_request(request).await
}

/// OpenAI Responses streaming custom events emitted by Siumai.
///
/// These are emitted as `ChatStreamEvent::Custom` so we can extend streaming semantics without
/// breaking the unified `ChatStreamEvent` enum.
///
/// Current events (beta.5 guidance):
/// - `openai:tool-call` — provider-hosted tool call started (from `response.output_item.added`)
/// - `openai:tool-result` — provider-hosted tool call completed (from `response.output_item.done`)
#[derive(Debug, Clone)]
pub enum OpenAiResponsesCustomEvent {
    ProviderToolCall(OpenAiProviderToolCallEvent),
    ProviderToolResult(OpenAiProviderToolResultEvent),
    Source(OpenAiSourceEvent),
}

impl OpenAiResponsesCustomEvent {
    pub fn from_stream_event(event: &ChatStreamEvent) -> Option<Self> {
        let ChatStreamEvent::Custom { event_type, data } = event else {
            return None;
        };

        if event_type == OpenAiProviderToolCallEvent::EVENT_TYPE {
            return OpenAiProviderToolCallEvent::from_custom(event_type, data)
                .map(OpenAiResponsesCustomEvent::ProviderToolCall);
        }

        if event_type == OpenAiProviderToolResultEvent::EVENT_TYPE {
            return OpenAiProviderToolResultEvent::from_custom(event_type, data)
                .map(OpenAiResponsesCustomEvent::ProviderToolResult);
        }

        if event_type == OpenAiSourceEvent::EVENT_TYPE {
            return OpenAiSourceEvent::from_custom(event_type, data)
                .map(OpenAiResponsesCustomEvent::Source);
        }

        None
    }
}

/// Provider-hosted tool call event (`openai:tool-call`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiProviderToolCallEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: String,
    pub provider_executed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_item: Option<serde_json::Value>,
}

impl OpenAiProviderToolCallEvent {
    pub const EVENT_TYPE: &'static str = "openai:tool-call";

    pub fn from_custom(event_type: &str, data: &serde_json::Value) -> Option<Self> {
        if event_type != Self::EVENT_TYPE {
            return None;
        }
        serde_json::from_value(data.clone()).ok()
    }
}

/// Provider-hosted tool result event (`openai:tool-result`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiProviderToolResultEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub result: serde_json::Value,
    pub provider_executed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_item: Option<serde_json::Value>,
}

impl OpenAiProviderToolResultEvent {
    pub const EVENT_TYPE: &'static str = "openai:tool-result";

    pub fn from_custom(event_type: &str, data: &serde_json::Value) -> Option<Self> {
        if event_type != Self::EVENT_TYPE {
            return None;
        }
        serde_json::from_value(data.clone()).ok()
    }
}

/// Source event (`openai:source`), typically emitted from web search tool results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiSourceEvent {
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
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

impl OpenAiSourceEvent {
    pub const EVENT_TYPE: &'static str = "openai:source";

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
    fn parses_openai_custom_provider_tool_events() {
        let tool_call = ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "ws_1",
                "toolName": "web_search",
                "input": "{}",
                "providerExecuted": true,
                "outputIndex": 1,
            }),
        };

        match OpenAiResponsesCustomEvent::from_stream_event(&tool_call).unwrap() {
            OpenAiResponsesCustomEvent::ProviderToolCall(ev) => {
                assert_eq!(ev.tool_call_id, "ws_1");
                assert_eq!(ev.tool_name, "web_search");
                assert_eq!(ev.input, "{}");
                assert!(ev.provider_executed);
                assert_eq!(ev.output_index, Some(1));
            }
            _ => panic!("expected tool call"),
        }

        let tool_result = ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "ws_1",
                "toolName": "web_search",
                "result": {"status":"completed"},
                "providerExecuted": true,
            }),
        };

        match OpenAiResponsesCustomEvent::from_stream_event(&tool_result).unwrap() {
            OpenAiResponsesCustomEvent::ProviderToolResult(ev) => {
                assert_eq!(ev.tool_call_id, "ws_1");
                assert_eq!(ev.tool_name, "web_search");
                assert!(ev.provider_executed);
                assert_eq!(ev.result["status"], serde_json::json!("completed"));
            }
            _ => panic!("expected tool result"),
        }

        let source = ChatStreamEvent::Custom {
            event_type: "openai:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "url",
                "id": "ws_1:0",
                "url": "https://www.rust-lang.org",
                "title": "Rust",
                "toolCallId": "ws_1",
            }),
        };

        match OpenAiResponsesCustomEvent::from_stream_event(&source).unwrap() {
            OpenAiResponsesCustomEvent::Source(ev) => {
                assert_eq!(ev.source_type, "url");
                assert_eq!(ev.url, "https://www.rust-lang.org");
                assert_eq!(ev.tool_call_id.as_deref(), Some("ws_1"));
            }
            _ => panic!("expected source"),
        }

        let doc_source = ChatStreamEvent::Custom {
            event_type: "openai:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "document",
                "id": "ann:doc:10",
                "url": "file_123",
                "title": "Document",
                "mediaType": "text/plain",
                "filename": "notes.txt",
                "providerMetadata": { "openai": { "fileId": "file_123" } }
            }),
        };

        match OpenAiResponsesCustomEvent::from_stream_event(&doc_source).unwrap() {
            OpenAiResponsesCustomEvent::Source(ev) => {
                assert_eq!(ev.source_type, "document");
                assert_eq!(ev.url, "file_123");
                assert_eq!(ev.media_type.as_deref(), Some("text/plain"));
                assert_eq!(ev.filename.as_deref(), Some("notes.txt"));
                assert!(ev.provider_metadata.is_some());
            }
            _ => panic!("expected source"),
        }
    }
}
