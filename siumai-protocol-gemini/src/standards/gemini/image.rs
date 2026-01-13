//! Gemini Image Generation Standard
//!
//! Standard + adapter wrapper for image generation via Gemini generateContent.

use crate::core::{ImageTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::standards::gemini::headers::build_gemini_headers;
use crate::standards::gemini::types::GeminiConfig;
use std::sync::Arc;

/// Adapter trait for Gemini image generation
pub trait GeminiImageAdapter: Send + Sync {
    /// Override image generation endpoint
    fn image_endpoint(&self, model: &str) -> String {
        // Gemini image generation models use generateContent, Imagen uses predict.
        if model.trim().starts_with("imagen-") {
            format!("/models/{}:predict", model)
        } else {
            format!("/models/{}:generateContent", model)
        }
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
        self.create_transformers_with_model(None)
    }

    pub fn create_transformers_with_model(&self, model: Option<&str>) -> ImageTransformers {
        let mut cfg = GeminiConfig::default();
        if let Some(m) = model
            && !m.is_empty()
        {
            cfg.model = m.to_string();
            cfg.common_params.model = m.to_string();
        }
        Self::transformers_from_config(cfg)
    }

    fn transformers_from_config(cfg: GeminiConfig) -> ImageTransformers {
        let req_tx = crate::standards::gemini::transformers::GeminiRequestTransformer {
            config: cfg.clone(),
        };
        let resp_tx =
            crate::standards::gemini::transformers::GeminiResponseTransformer { config: cfg };
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
        crate::traits::ProviderCapabilities::new().with_image_generation()
    }
    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        let mut headers = build_gemini_headers(api_key, &ctx.http_extra_headers)?;
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
        let model = super::normalize_gemini_model_id(req.model.as_deref().unwrap_or(""));
        if let Some(adapter) = &self.adapter {
            format!("{}{}", base, adapter.image_endpoint(&model))
        } else if model.trim().starts_with("imagen-") {
            format!("{}/models/{}:predict", base, model)
        } else {
            format!("{}/models/{}:generateContent", base, model)
        }
    }
    fn choose_image_transformers(
        &self,
        req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        GeminiImageStandard {
            adapter: self.adapter.clone(),
        }
        .create_transformers_with_model(req.model.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderContext;

    #[test]
    fn image_url_accepts_vertex_resource_style_model_ids() {
        let spec = GeminiImageStandard::new().create_spec("gemini");
        let ctx = ProviderContext::new(
            "gemini",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google".to_string(),
            Some("".to_string()),
            std::collections::HashMap::new(),
        );

        let req = crate::types::ImageGenerationRequest {
            prompt: "a cat".to_string(),
            count: 1,
            model: Some("publishers/google/models/gemini-2.0-flash".to_string()),
            ..Default::default()
        };

        assert_eq!(
            spec.image_url(&req, &ctx),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"
        );
    }

    #[test]
    fn image_url_uses_predict_for_imagen_models() {
        let spec = GeminiImageStandard::new().create_spec("gemini");
        let ctx = ProviderContext::new(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta".to_string(),
            Some("".to_string()),
            std::collections::HashMap::new(),
        );

        let req = crate::types::ImageGenerationRequest {
            prompt: "a cat".to_string(),
            count: 1,
            model: Some("imagen-4.0-generate-001".to_string()),
            ..Default::default()
        };

        assert_eq!(
            spec.image_url(&req, &ctx),
            "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict"
        );
    }
}
