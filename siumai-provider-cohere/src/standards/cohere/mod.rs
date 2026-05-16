//! Cohere native standards.

pub mod chat;
pub mod embedding;
pub mod errors;
pub mod rerank;
mod shared;

use crate::core::{
    ChatTransformers, EmbeddingTransformers, ProviderContext, ProviderSpec, RerankTransformers,
};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, EmbeddingRequest, RerankRequest};
use reqwest::header::HeaderMap;

#[derive(Clone, Default)]
pub struct CohereSpec;

impl CohereSpec {
    pub fn new() -> Self {
        Self
    }
}

impl ProviderSpec for CohereSpec {
    fn id(&self) -> &'static str {
        "cohere"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_rerank()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        shared::build_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        headers: &HeaderMap,
    ) -> Option<LlmError> {
        errors::classify_cohere_http_error(self.id(), status, body_text, headers)
    }

    fn try_chat_url(
        &self,
        _stream: bool,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(crate::utils::url::join_url(&ctx.base_url, "/chat"))
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        chat::CohereChatStandard::new().create_transformers(&ctx.provider_id, req)
    }

    fn try_embedding_url(
        &self,
        _req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(crate::utils::url::join_url(&ctx.base_url, "/embed"))
    }

    fn choose_embedding_transformers(
        &self,
        req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        embedding::CohereEmbeddingStandard::new().create_transformers(&ctx.provider_id, req)
    }

    fn try_rerank_url(
        &self,
        _req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(crate::utils::url::join_url(&ctx.base_url, "/rerank"))
    }

    fn choose_rerank_transformers(
        &self,
        _req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> RerankTransformers {
        rerank::CohereRerankStandard::new().create_transformers(&ctx.provider_id)
    }
}
