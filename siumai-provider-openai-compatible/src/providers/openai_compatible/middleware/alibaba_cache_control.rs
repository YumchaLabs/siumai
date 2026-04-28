//! Alibaba/Qwen prompt cache-control warning parity middleware.

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Warning};

#[derive(Debug, Clone)]
pub(crate) struct OpenAiCompatibleAlibabaCacheControlWarningMiddleware {
    provider_id: String,
}

impl OpenAiCompatibleAlibabaCacheControlWarningMiddleware {
    pub(crate) fn new(provider_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
        }
    }

    fn compute_warnings(&self, req: &ChatRequest) -> Vec<Warning> {
        crate::standards::openai::compat::alibaba_cache_control::cache_control_warnings(
            &self.provider_id,
            req,
        )
    }

    fn merge_warnings(mut resp: ChatResponse, additional: Vec<Warning>) -> ChatResponse {
        if additional.is_empty() {
            return resp;
        }

        match resp.warnings.as_mut() {
            Some(existing) => existing.extend(additional),
            None => resp.warnings = Some(additional),
        }

        resp
    }
}

impl LanguageModelMiddleware for OpenAiCompatibleAlibabaCacheControlWarningMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        Ok(Self::merge_warnings(resp, self.compute_warnings(req)))
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let response = Self::merge_warnings(response, self.compute_warnings(req));
                Ok(vec![ChatStreamEvent::StreamEnd { response }])
            }
            other => Ok(vec![other]),
        }
    }
}
