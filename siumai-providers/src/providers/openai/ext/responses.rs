//! OpenAI Responses API extension helpers.
//!
//! This module provides a small, explicit API surface for interacting with the
//! Responses API through the existing unified chat pipeline.

use crate::error::LlmError;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, OpenAiOptions, ResponsesApiConfig};

/// Parameters for `GET /responses/{response_id}/input_items`.
#[derive(Debug, Clone, Default)]
pub struct OpenAiResponsesInputItemsParams {
    pub limit: Option<u32>,
    pub order: Option<OpenAiResponsesInputItemsOrder>,
    pub after: Option<String>,
    pub include: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiResponsesInputItemsOrder {
    Asc,
    Desc,
}

impl OpenAiResponsesInputItemsOrder {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asc => "asc",
            Self::Desc => "desc",
        }
    }
}

/// Response type for `GET /responses/{response_id}/input_items`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAiResponsesInputItemsPage {
    pub object: String,
    pub data: Vec<serde_json::Value>,
    pub has_more: bool,
    pub first_id: String,
    pub last_id: String,
}

/// Request body for `POST /responses/compact`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAiResponsesCompactRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl OpenAiResponsesCompactRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: None,
            previous_response_id: None,
            instructions: None,
        }
    }

    pub fn with_input(mut self, input: serde_json::Value) -> Self {
        self.input = Some(input);
        self
    }

    pub fn with_previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.previous_response_id = Some(id.into());
        self
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Response type for `POST /responses/compact`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAiResponsesCompaction {
    pub id: String,
    pub object: String,
    pub output: Vec<serde_json::Value>,
    pub created_at: u64,
    pub usage: serde_json::Value,
}

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

/// Retrieve a stored response by ID (`GET /responses/{response_id}`).
pub async fn retrieve(
    client: &crate::providers::openai::OpenAiClient,
    response_id: &str,
    include: Option<Vec<String>>,
) -> Result<serde_json::Value, LlmError> {
    client.responses_retrieve(response_id, include).await
}

/// Delete a stored response (`DELETE /responses/{response_id}`).
pub async fn delete(
    client: &crate::providers::openai::OpenAiClient,
    response_id: &str,
) -> Result<serde_json::Value, LlmError> {
    client.responses_delete(response_id).await
}

/// Cancel an in-progress background response (`POST /responses/{response_id}/cancel`).
pub async fn cancel(
    client: &crate::providers::openai::OpenAiClient,
    response_id: &str,
) -> Result<serde_json::Value, LlmError> {
    client.responses_cancel(response_id).await
}

/// List response input items (`GET /responses/{response_id}/input_items`).
pub async fn list_input_items(
    client: &crate::providers::openai::OpenAiClient,
    response_id: &str,
    params: OpenAiResponsesInputItemsParams,
) -> Result<OpenAiResponsesInputItemsPage, LlmError> {
    client.responses_list_input_items(response_id, params).await
}

/// Compact a conversation (`POST /responses/compact`).
pub async fn compact(
    client: &crate::providers::openai::OpenAiClient,
    request: OpenAiResponsesCompactRequest,
) -> Result<OpenAiResponsesCompaction, LlmError> {
    client.responses_compact(request).await
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
    use crate::providers::openai::{OpenAiClient, OpenAiConfig};
    use wiremock::matchers::{body_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    #[tokio::test]
    async fn responses_admin_endpoints_work() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/responses/resp_1"))
            .and(query_param("include", "file_search_call.results"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_1",
                "object": "response",
                "status": "completed"
            })))
            .mount(&server)
            .await;

        Mock::given(method("DELETE"))
            .and(path("/v1/responses/resp_1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_1",
                "object": "response",
                "deleted": true
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/responses/resp_1/cancel"))
            .and(body_json(serde_json::json!({})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_1",
                "object": "response",
                "status": "cancelled"
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/v1/responses/resp_1/input_items"))
            .and(query_param("limit", "10"))
            .and(query_param("order", "asc"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"id":"msg_1","type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
                "first_id": "msg_1",
                "last_id": "msg_1",
                "has_more": false
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .and(body_json(serde_json::json!({
                "model": "gpt-4o-mini",
                "previous_response_id": "resp_1"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmp_1",
                "object": "response.compaction",
                "created_at": 1,
                "output": [],
                "usage": {"total_tokens": 1}
            })))
            .mount(&server)
            .await;

        let cfg = OpenAiConfig::new("KEY").with_base_url(format!("{}/v1", server.uri()));
        let client = OpenAiClient::new(cfg, reqwest::Client::new());

        let got = retrieve(
            &client,
            "resp_1",
            Some(vec!["file_search_call.results".to_string()]),
        )
        .await
        .unwrap();
        assert_eq!(got["id"], serde_json::json!("resp_1"));

        let deleted = delete(&client, "resp_1").await.unwrap();
        assert_eq!(deleted["deleted"], serde_json::json!(true));

        let cancelled = cancel(&client, "resp_1").await.unwrap();
        assert_eq!(cancelled["status"], serde_json::json!("cancelled"));

        let page = list_input_items(
            &client,
            "resp_1",
            OpenAiResponsesInputItemsParams {
                limit: Some(10),
                order: Some(OpenAiResponsesInputItemsOrder::Asc),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(page.object, "list");
        assert_eq!(page.data.len(), 1);
        assert!(!page.has_more);

        let compaction = compact(
            &client,
            OpenAiResponsesCompactRequest::new("gpt-4o-mini").with_previous_response_id("resp_1"),
        )
        .await
        .unwrap();
        assert_eq!(compaction.object, "response.compaction");
        assert_eq!(compaction.id, "cmp_1");
    }
}
