//! xAI Responses API response transformer presets.
//!
//! This module wraps the OpenAI Responses response transformer with xAI-specific defaults
//! to keep call sites and tests free of provider-specific mode toggles.

use crate::execution::transformers::response::ResponseTransformer;

/// xAI-aligned Responses API response transformer.
#[derive(Clone)]
pub struct XaiResponsesResponseTransformer {
    inner: crate::standards::openai::transformers::OpenAiResponsesResponseTransformer,
}

impl XaiResponsesResponseTransformer {
    pub fn new() -> Self {
        Self {
            inner: crate::standards::openai::transformers::OpenAiResponsesResponseTransformer::new(
            )
            .with_style(crate::standards::openai::transformers::ResponsesTransformStyle::Xai),
        }
    }
}

impl Default for XaiResponsesResponseTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseTransformer for XaiResponsesResponseTransformer {
    fn provider_id(&self) -> &str {
        "xai_responses"
    }

    fn transform_chat_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ChatResponse, crate::error::LlmError> {
        self.inner.transform_chat_response(raw)
    }
}
