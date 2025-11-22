//! Vertex Anthropic Provider Spec
//!
//! Defines the ProviderSpec implementation for Anthropic on Vertex AI.
//!
//! This spec reuses the Anthropic standard (`siumai-std-anthropic`) and the
//! core execution pipeline, while keeping Vertex-specific details (base URL,
//! headers) local to this module.

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
        // Mirror Anthropic's core capabilities; Vertex exposes the same
        // Messages API semantics over a different transport/auth layer.
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(
        &self,
        _ctx: &ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        crate::execution::http::headers::ProviderHeaders::vertex_bearer(&self.extra_headers)
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
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        #[cfg(feature = "std-anthropic-external")]
        {
            use crate::core::provider_spec::{
                anthropic_like_chat_request_to_core_input, anthropic_like_map_core_stream_event,
                bridge_core_chat_transformers,
            };
            use crate::std_anthropic::anthropic::chat::{
                AnthropicChatStandard, AnthropicDefaultChatAdapter,
            };
            use siumai_core::provider_spec::CoreChatTransformers;

            // Reuse the Anthropic Messages standard to build core-level transformers,
            // then bridge them into the aggregator's ChatTransformers bundle.
            let standard =
                AnthropicChatStandard::with_adapter(Arc::new(AnthropicDefaultChatAdapter));

            let core_txs: CoreChatTransformers = CoreChatTransformers {
                request: standard.create_request_transformer(&ctx.provider_id),
                response: standard.create_response_transformer(&ctx.provider_id),
                stream: Some(standard.create_stream_converter(&ctx.provider_id)),
            };

            bridge_core_chat_transformers(
                core_txs,
                anthropic_like_chat_request_to_core_input,
                |evt| anthropic_like_map_core_stream_event("anthropic-vertex", evt),
            )
        }

        // Fallback path when the external Anthropic standard is not available.
        // This retains the legacy in-crate transformers as a safety net,
        // although `feature = "anthropic"` normally implies `std-anthropic-external`.
        #[cfg(not(feature = "std-anthropic-external"))]
        {
            let stream_transformer = if req.stream {
                let converter =
                    crate::providers::anthropic::streaming::AnthropicEventConverter::new(
                        crate::params::AnthropicParams::default(),
                    );
                let stream_tx =
                    crate::providers::anthropic::transformers::AnthropicStreamChunkTransformer {
                        provider_id: "anthropic-vertex".to_string(),
                        inner: converter,
                    };
                Some(Arc::new(stream_tx)
                    as Arc<
                        dyn crate::execution::transformers::stream::StreamChunkTransformer,
                    >)
            } else {
                None
            };

            ChatTransformers {
                request: Arc::new(
                    crate::providers::anthropic::transformers::AnthropicRequestTransformer::new(
                        None,
                    ),
                ),
                response: Arc::new(
                    crate::providers::anthropic::transformers::AnthropicResponseTransformer,
                ),
                stream: stream_transformer,
                json: None,
            }
        }
    }
}
