//! Cerebras structured-output response normalization.

use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::middleware::{GenerateAsyncFn, LanguageModelMiddleware, StreamAsyncFn};
use crate::streaming::{ChatStream, ChatStreamEvent};
use crate::types::{
    ChatRequest, ChatResponse, ChatStreamPart, ContentPart, FinishReason, MessageContent,
};
use futures_util::StreamExt;

#[derive(Debug, Default)]
pub(crate) struct OpenAiCompatibleCerebrasStructuredOutputMiddleware;

#[derive(Debug, Default)]
struct CerebrasStreamState {
    has_text: bool,
}

impl OpenAiCompatibleCerebrasStructuredOutputMiddleware {
    pub(crate) const fn new() -> Self {
        Self
    }

    fn request_uses_json_response_format(req: &ChatRequest) -> bool {
        matches!(
            req.response_format,
            Some(
                crate::types::chat::ResponseFormat::Json { .. }
                    | crate::types::chat::ResponseFormat::JsonObject { .. }
            )
        )
    }

    fn raw_finish_reason_is_tool_calls(response: &ChatResponse) -> bool {
        response.raw_finish_reason.as_deref() == Some("tool_calls")
    }

    fn content_has_non_empty_text(content: &MessageContent) -> bool {
        match content {
            MessageContent::Text(text) => !text.is_empty(),
            MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(ContentPart::as_text)
                .any(|text| !text.is_empty()),
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(_) => false,
        }
    }

    fn message_content_without_tool_calls(content: MessageContent) -> MessageContent {
        let MessageContent::MultiModal(parts) = content else {
            return content;
        };

        let parts: Vec<ContentPart> = parts
            .into_iter()
            .filter(|part| !part.is_tool_call())
            .collect();

        match parts.as_slice() {
            [
                ContentPart::Text {
                    text,
                    provider_options,
                    provider_metadata: None,
                },
            ] if provider_options.is_empty() => MessageContent::Text(text.clone()),
            [] => MessageContent::Text(String::new()),
            _ => MessageContent::MultiModal(parts),
        }
    }

    fn normalize_response(mut response: ChatResponse) -> ChatResponse {
        if !Self::raw_finish_reason_is_tool_calls(&response)
            || !Self::content_has_non_empty_text(&response.content)
        {
            return response;
        }

        if response.has_tool_calls() {
            response.content = Self::message_content_without_tool_calls(response.content);
        }
        response.finish_reason = Some(FinishReason::Stop);
        response
    }

    fn normalize_stream_part(
        state: &mut CerebrasStreamState,
        mut part: ChatStreamPart,
    ) -> Option<ChatStreamPart> {
        match &mut part {
            ChatStreamPart::TextDelta { delta, .. } => {
                if !delta.is_empty() {
                    state.has_text = true;
                }
                Some(part)
            }
            ChatStreamPart::ToolInputStart { .. }
            | ChatStreamPart::ToolInputDelta { .. }
            | ChatStreamPart::ToolInputEnd { .. }
            | ChatStreamPart::ToolCall(_)
                if state.has_text =>
            {
                None
            }
            ChatStreamPart::Finish { finish_reason, .. }
                if state.has_text && finish_reason.raw.as_deref() == Some("tool_calls") =>
            {
                finish_reason.unified = FinishReason::Stop;
                Some(part)
            }
            _ => Some(part),
        }
    }

    fn normalize_stream_event(
        state: &mut CerebrasStreamState,
        event: ChatStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        match event {
            ChatStreamEvent::Part { part } => Self::normalize_stream_part(state, part)
                .map(|part| ChatStreamEvent::Part { part })
                .into_iter()
                .collect(),
            ChatStreamEvent::PartWithReplay { part, replay } => {
                Self::normalize_stream_part(state, part)
                    .map(|part| ChatStreamEvent::PartWithReplay { part, replay })
                    .into_iter()
                    .collect()
            }
            ChatStreamEvent::StreamEnd { response } => {
                let response = if state.has_text && Self::raw_finish_reason_is_tool_calls(&response)
                {
                    let mut response = response;
                    response.finish_reason = Some(FinishReason::Stop);
                    response
                } else {
                    response
                };
                vec![ChatStreamEvent::StreamEnd { response }]
            }
            other => vec![other],
        }
    }

    fn wrap_stream(stream: ChatStream) -> ChatStream {
        let mut state = CerebrasStreamState::default();
        Box::pin(stream.flat_map(move |item| {
            let out: Vec<Result<ChatStreamEvent, LlmError>> = match item {
                Ok(event) => Self::normalize_stream_event(&mut state, event)
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(err) => vec![Err(err)],
            };
            futures_util::stream::iter(out)
        }))
    }
}

impl LanguageModelMiddleware for OpenAiCompatibleCerebrasStructuredOutputMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        if Self::request_uses_json_response_format(req) {
            Ok(Self::normalize_response(response))
        } else {
            Ok(response)
        }
    }

    fn wrap_stream_async(&self, next: Arc<StreamAsyncFn>) -> Arc<StreamAsyncFn> {
        Arc::new(move |req: ChatRequest| {
            let next = next.clone();
            Box::pin(async move {
                let should_normalize = Self::request_uses_json_response_format(&req);
                let stream = next(req).await?;
                if should_normalize {
                    Ok(Self::wrap_stream(stream))
                } else {
                    Ok(stream)
                }
            })
        })
    }

    fn wrap_generate_async(&self, next: Arc<GenerateAsyncFn>) -> Arc<GenerateAsyncFn> {
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatStreamFinishInfo, ProviderOptionsMap};

    #[test]
    fn post_generate_drops_repeated_tool_call_for_structured_text() {
        let req = ChatRequest::builder()
            .response_format(crate::types::chat::ResponseFormat::json_object())
            .build();
        let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::Text {
                text: "{\"result\":\"2026\"}".to_string(),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            },
            ContentPart::tool_call("call_1", "lookup", serde_json::json!({}), None),
        ]));
        response.finish_reason = Some(FinishReason::ToolCalls);
        response.raw_finish_reason = Some("tool_calls".to_string());

        let response = OpenAiCompatibleCerebrasStructuredOutputMiddleware::new()
            .post_generate(&req, response)
            .expect("post generate");

        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.raw_finish_reason.as_deref(), Some("tool_calls"));
        assert!(!response.has_tool_calls());
        assert_eq!(response.content.all_text(), "{\"result\":\"2026\"}");
    }

    #[test]
    fn post_generate_normalizes_structured_text_without_tool_call_parts() {
        let req = ChatRequest::builder()
            .response_format(crate::types::chat::ResponseFormat::json_object())
            .build();
        let mut response = ChatResponse::new(MessageContent::Text("{\"result\":\"2026\"}".into()));
        response.finish_reason = Some(FinishReason::ToolCalls);
        response.raw_finish_reason = Some("tool_calls".to_string());

        let response = OpenAiCompatibleCerebrasStructuredOutputMiddleware::new()
            .post_generate(&req, response)
            .expect("post generate");

        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.raw_finish_reason.as_deref(), Some("tool_calls"));
        assert!(!response.has_tool_calls());
        assert_eq!(response.content.all_text(), "{\"result\":\"2026\"}");
    }

    #[test]
    fn stream_finish_is_normalized_after_text() {
        let mut state = CerebrasStreamState::default();
        let events = vec![
            ChatStreamEvent::Part {
                part: ChatStreamPart::TextDelta {
                    id: "text_0".to_string(),
                    delta: "{\"result\":\"2026\"}".to_string(),
                    provider_metadata: None,
                },
            },
            ChatStreamEvent::Part {
                part: ChatStreamPart::ToolCall(crate::types::ChatStreamToolCall {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "lookup".to_string(),
                    input: "{}".to_string(),
                    provider_executed: None,
                    dynamic: None,
                    provider_metadata: None,
                }),
            },
            ChatStreamEvent::Part {
                part: ChatStreamPart::Finish {
                    usage: Default::default(),
                    finish_reason: ChatStreamFinishInfo {
                        unified: FinishReason::ToolCalls,
                        raw: Some("tool_calls".to_string()),
                    },
                    provider_metadata: None,
                },
            },
        ];

        let normalized: Vec<_> = events
            .into_iter()
            .flat_map(|event| {
                OpenAiCompatibleCerebrasStructuredOutputMiddleware::normalize_stream_event(
                    &mut state, event,
                )
            })
            .collect();

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().all(|event| !matches!(
            event,
            ChatStreamEvent::Part {
                part: ChatStreamPart::ToolCall(_)
            }
        )));
        let Some(ChatStreamEvent::Part {
            part: ChatStreamPart::Finish { finish_reason, .. },
        }) = normalized.last()
        else {
            panic!("expected finish part");
        };
        assert_eq!(finish_reason.unified, FinishReason::Stop);
        assert_eq!(finish_reason.raw.as_deref(), Some("tool_calls"));
    }
}
