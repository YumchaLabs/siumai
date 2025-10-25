use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::utils::http_headers::ProviderHeaders;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Ollama ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct OllamaSpec;

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
        ProviderHeaders::ollama(&ctx.http_extra_headers)
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!("{}/api/chat", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Not used by current client wiring (client provides transformers with full params).
        // Provide a functional default to satisfy trait contract.
        let req_tx = crate::providers::ollama::transformers::OllamaRequestTransformer {
            params: crate::providers::ollama::config::OllamaParams::default(),
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
