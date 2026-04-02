//! Compatibility warning middleware for compat structured-output downgrades.

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Warning};

#[derive(Debug, Default)]
pub(crate) struct OpenAiCompatibleStructuredOutputsWarningMiddleware;

impl OpenAiCompatibleStructuredOutputsWarningMiddleware {
    pub(crate) const fn new() -> Self {
        Self
    }

    fn should_warn(req: &ChatRequest) -> bool {
        matches!(
            req.response_format,
            Some(crate::types::chat::ResponseFormat::Json { .. })
        )
    }

    fn warnings(req: &ChatRequest) -> Vec<Warning> {
        if Self::should_warn(req) {
            vec![Warning::unsupported(
                "responseFormat",
                Some("JSON response format schema is only supported with structuredOutputs"),
            )]
        } else {
            Vec::new()
        }
    }

    fn merge_warnings(mut resp: ChatResponse, warnings: Vec<Warning>) -> ChatResponse {
        if warnings.is_empty() {
            return resp;
        }

        let mut merged = resp.warnings.take().unwrap_or_default();
        merged.extend(warnings);
        resp.warnings = Some(merged);
        resp
    }
}

impl LanguageModelMiddleware for OpenAiCompatibleStructuredOutputsWarningMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        Ok(Self::merge_warnings(resp, Self::warnings(req)))
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let response = Self::merge_warnings(response, Self::warnings(req));
                Ok(vec![ChatStreamEvent::StreamEnd { response }])
            }
            other => Ok(vec![other]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, MessageContent};

    #[test]
    fn post_generate_emits_warning_when_schema_outputs_are_disabled() {
        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({ "type": "object", "properties": {} }),
            ))
            .build();

        let resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
        let resp = OpenAiCompatibleStructuredOutputsWarningMiddleware::new()
            .post_generate(&req, resp)
            .expect("post-generate");

        assert_eq!(
            resp.warnings,
            Some(vec![Warning::Unsupported {
                feature: "responseFormat".to_string(),
                details: Some(
                    "JSON response format schema is only supported with structuredOutputs"
                        .to_string(),
                ),
            }])
        );
    }
}
