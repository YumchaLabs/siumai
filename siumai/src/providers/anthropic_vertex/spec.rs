//! Vertex Anthropic Provider Spec
//!
//! Defines the ProviderSpec implementation for Anthropic on Vertex AI.

use std::sync::Arc;

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;

/// Vertex Anthropic provider specification
#[derive(Clone)]
pub struct VertexAnthropicSpec {
    pub base_url: String,
    pub model: String,
    pub extra_headers: std::collections::HashMap<String, String>,
}

impl VertexAnthropicSpec {
    pub fn new(
        base_url: String,
        model: String,
        extra_headers: std::collections::HashMap<String, String>,
    ) -> Self {
        Self {
            base_url,
            model,
            extra_headers,
        }
    }
}

impl ProviderSpec for VertexAnthropicSpec {
    fn id(&self) -> &'static str {
        "anthropic-vertex"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
    }

    fn build_headers(
        &self,
        _ctx: &ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        crate::utils::http_headers::ProviderHeaders::vertex_bearer(&self.extra_headers)
    }

    fn chat_url(&self, stream: bool, _req: &ChatRequest, _ctx: &ProviderContext) -> String {
        if stream {
            crate::utils::url::join_url(
                &self.base_url,
                &format!("models/{}:streamRawPredict?alt=sse", self.model),
            )
        } else {
            crate::utils::url::join_url(
                &self.base_url,
                &format!("models/{}:rawPredict", self.model),
            )
        }
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let stream_transformer = if req.stream {
            let converter = crate::providers::anthropic::streaming::AnthropicEventConverter::new(
                crate::params::AnthropicParams::default(),
            );
            let stream_tx =
                crate::providers::anthropic::transformers::AnthropicStreamChunkTransformer {
                    provider_id: "anthropic-vertex".to_string(),
                    inner: converter,
                };
            Some(Arc::new(stream_tx)
                as Arc<
                    dyn crate::transformers::stream::StreamChunkTransformer,
                >)
        } else {
            None
        };

        ChatTransformers {
            request: Arc::new(
                crate::providers::anthropic::transformers::AnthropicRequestTransformer::new(None),
            ),
            response: Arc::new(
                crate::providers::anthropic::transformers::AnthropicResponseTransformer,
            ),
            stream: stream_transformer,
            json: None,
        }
    }
}
