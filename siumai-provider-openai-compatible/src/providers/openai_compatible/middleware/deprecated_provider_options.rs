//! Compatibility warning middleware for deprecated openai-compatible providerOptions keys.

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Warning};

const DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING: &str =
    "The 'openai-compatible' key in providerOptions is deprecated. Use 'openaiCompatible' instead.";

#[derive(Debug, Default)]
pub(crate) struct OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware;

impl OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware {
    pub(crate) const fn new() -> Self {
        Self
    }

    fn should_warn(req: &ChatRequest) -> bool {
        req.provider_options_map.get("openai-compatible").is_some()
    }

    fn warnings(req: &ChatRequest) -> Vec<Warning> {
        if Self::should_warn(req) {
            vec![Warning::other(DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING)]
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

impl LanguageModelMiddleware for OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware {
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
    fn post_generate_emits_warning_when_deprecated_provider_options_key_is_used() {
        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .build()
            .with_provider_option(
                "openai-compatible",
                serde_json::json!({ "user": "compat-user" }),
            );

        let resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
        let resp = OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware::new()
            .post_generate(&req, resp)
            .expect("post-generate");

        assert_eq!(
            resp.warnings,
            Some(vec![Warning::Other {
                message: DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING.to_string(),
            }])
        );
    }
}
