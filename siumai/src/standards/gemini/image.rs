//! Gemini Image Generation Standard
//!
//! Standard + adapter wrapper for image generation via Gemini generateContent.

use crate::core::{ImageTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use std::sync::Arc;

/// Adapter trait for Gemini image generation
pub trait GeminiImageAdapter: Send + Sync {
    /// Override image generation endpoint
    fn image_endpoint(&self, model: &str) -> String {
        format!("/models/{}:generateContent", model)
    }
    /// Allow custom header injection
    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct GeminiImageStandard {
    adapter: Option<Arc<dyn GeminiImageAdapter>>,
}

impl GeminiImageStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }
    pub fn with_adapter(adapter: Arc<dyn GeminiImageAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }
    pub fn create_spec(&self, provider_id: &'static str) -> GeminiImageSpec {
        GeminiImageSpec {
            provider_id,
            adapter: self.adapter.clone(),
        }
    }
    pub fn create_transformers(&self) -> ImageTransformers {
        let cfg = crate::providers::gemini::types::GeminiConfig::default();
        let req_tx = crate::providers::gemini::transformers::GeminiRequestTransformer {
            config: cfg.clone(),
        };
        let resp_tx =
            crate::providers::gemini::transformers::GeminiResponseTransformer { config: cfg };
        ImageTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
        }
    }
}

impl Default for GeminiImageStandard {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GeminiImageSpec {
    provider_id: &'static str,
    adapter: Option<Arc<dyn GeminiImageAdapter>>,
}

impl ProviderSpec for GeminiImageSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_custom_feature("image_generation", true)
    }
    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> String {
        unreachable!("chat_url not supported by GeminiImageSpec")
    }
    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> crate::core::ChatTransformers {
        unreachable!("choose_chat_transformers not supported by GeminiImageSpec")
    }
    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        let mut headers = ProviderHeaders::gemini(api_key, &ctx.http_extra_headers)?;
        if let Some(adapter) = &self.adapter {
            adapter.build_headers(api_key, &mut headers)?;
        }
        Ok(headers)
    }
    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = req.model.as_deref().unwrap_or("");
        if let Some(adapter) = &self.adapter {
            format!("{}{}", base, adapter.image_endpoint(model))
        } else {
            format!("{}/models/{}:generateContent", base, model)
        }
    }
    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        GeminiImageStandard {
            adapter: self.adapter.clone(),
        }
        .create_transformers()
    }
}
