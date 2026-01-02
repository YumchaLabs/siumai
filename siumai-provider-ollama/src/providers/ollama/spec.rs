use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use reqwest::header::HeaderMap;
use std::sync::Arc;

use super::config::OllamaParams;

/// Ollama ProviderSpec implementation
#[derive(Clone, Default)]
pub struct OllamaSpec {
    params: OllamaParams,
}

impl OllamaSpec {
    pub fn new(params: OllamaParams) -> Self {
        Self { params }
    }
}

impl ProviderSpec for OllamaSpec {
    fn id(&self) -> &'static str {
        "ollama"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        // Ollama typically has no auth; pass through extra headers and JSON content-type
        crate::standards::ollama::utils::build_headers(&ctx.http_extra_headers)
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!("{}/api/chat", ctx.base_url.trim_end_matches('/'))
    }

    fn models_url(&self, ctx: &ProviderContext) -> String {
        format!("{}/api/tags", ctx.base_url.trim_end_matches('/'))
    }

    fn embedding_url(
        &self,
        _req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!("{}/api/embed", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let req_tx = crate::providers::ollama::transformers::OllamaRequestTransformer {
            params: self.params.clone(),
        };
        let resp_tx = crate::providers::ollama::transformers::OllamaResponseTransformer;
        ChatTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
            stream: None,
            json: Some(Arc::new(
                crate::providers::ollama::streaming::OllamaEventConverter::new(),
            )),
        }
    }
}

/// Ollama ProviderSpec carrying embedding defaults.
///
/// Use this when building embedding executors to avoid duplicating transformer wiring
/// and to ensure request.model falls back to the configured default.
#[derive(Clone)]
pub struct OllamaSpecWithConfig {
    params: OllamaParams,
    default_embedding_model: String,
}

impl OllamaSpecWithConfig {
    pub fn new(params: OllamaParams, default_embedding_model: String) -> Self {
        Self {
            params,
            default_embedding_model,
        }
    }
}

impl ProviderSpec for OllamaSpecWithConfig {
    fn id(&self) -> &'static str {
        "ollama"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        OllamaSpec {
            params: self.params.clone(),
        }
        .capabilities()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        crate::standards::ollama::utils::build_headers(&ctx.http_extra_headers)
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!("{}/api/chat", ctx.base_url.trim_end_matches('/'))
    }

    fn models_url(&self, ctx: &ProviderContext) -> String {
        OllamaSpec {
            params: self.params.clone(),
        }
        .models_url(ctx)
    }

    fn embedding_url(
        &self,
        _req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!("{}/api/embed", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        OllamaSpec {
            params: self.params.clone(),
        }
        .choose_chat_transformers(req, ctx)
    }

    fn choose_embedding_transformers(
        &self,
        _req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> crate::core::EmbeddingTransformers {
        let req_tx = crate::providers::ollama::transformers::OllamaEmbeddingRequestTransformer {
            default_model: self.default_embedding_model.clone(),
            params: self.params.clone(),
        };
        let resp_tx = crate::providers::ollama::transformers::OllamaEmbeddingResponseTransformer;
        crate::core::EmbeddingTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
        }
    }
}
