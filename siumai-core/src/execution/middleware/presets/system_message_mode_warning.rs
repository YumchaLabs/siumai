//! Vercel-aligned warning middleware for `systemMessageMode`.
//!
//! Vercel AI SDK emits a warning when `systemMessageMode: "remove"` is used and
//! the prompt contains system messages. The OpenAI Chat Completions converter
//! in this repository applies the same removal behavior, but warnings are
//! surfaced at the middleware layer (post-generate / stream-end).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, MessageRole, Warning};

#[derive(Debug, Default)]
pub struct SystemMessageModeWarningMiddleware;

impl SystemMessageModeWarningMiddleware {
    pub const fn new() -> Self {
        Self
    }

    fn system_message_mode(req: &ChatRequest) -> Option<&str> {
        req.provider_option("openai")
            .or_else(|| req.provider_option("azure"))
            .and_then(|v| v.as_object())
            .and_then(|obj| {
                obj.get("systemMessageMode")
                    .or_else(|| obj.get("system_message_mode"))
                    .and_then(|v| v.as_str())
            })
    }

    fn has_system_messages(req: &ChatRequest) -> bool {
        req.messages
            .iter()
            .any(|m| matches!(m.role, MessageRole::System))
    }

    fn compute_warnings(req: &ChatRequest) -> Vec<Warning> {
        if Self::system_message_mode(req) == Some("remove") && Self::has_system_messages(req) {
            vec![Warning::other("system messages are removed for this model")]
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

impl LanguageModelMiddleware for SystemMessageModeWarningMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        Ok(Self::merge_warnings(resp, Self::compute_warnings(req)))
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let response = Self::merge_warnings(response, Self::compute_warnings(req));
                Ok(vec![ChatStreamEvent::StreamEnd { response }])
            }
            other => Ok(vec![other]),
        }
    }
}
