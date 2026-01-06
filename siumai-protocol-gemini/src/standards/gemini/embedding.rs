//! Gemini Embedding API Standard
//!
//! Standard + adapter wrapper for Gemini Embeddings API.

use crate::core::{EmbeddingTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::standards::gemini::headers::build_gemini_headers;
use crate::standards::gemini::types::GeminiConfig;
use std::sync::Arc;

/// Adapter trait for Gemini embeddings
pub trait GeminiEmbeddingAdapter: Send + Sync {
    /// Override embedding endpoint for single input
    fn embed_endpoint(&self, model: &str) -> String {
        format!("/models/{}:embedContent", model)
    }
    /// Override embedding endpoint for batch inputs
    fn batch_embed_endpoint(&self, model: &str) -> String {
        format!("/models/{}:batchEmbedContents", model)
    }
    /// Allow custom headers if needed
    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct GeminiEmbeddingStandard {
    adapter: Option<Arc<dyn GeminiEmbeddingAdapter>>,
}

impl GeminiEmbeddingStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }
    pub fn with_adapter(adapter: Arc<dyn GeminiEmbeddingAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }
    pub fn create_spec(&self, provider_id: &'static str) -> GeminiEmbeddingSpec {
        GeminiEmbeddingSpec {
            provider_id,
            adapter: self.adapter.clone(),
        }
    }
    pub fn create_transformers(&self) -> EmbeddingTransformers {
        self.create_transformers_with_model(None)
    }

    pub fn create_transformers_with_model(&self, model: Option<&str>) -> EmbeddingTransformers {
        let mut cfg = GeminiConfig::default();
        if let Some(m) = model
            && !m.is_empty()
        {
            cfg.model = m.to_string();
            cfg.common_params.model = m.to_string();
        }
        Self::transformers_from_config(cfg)
    }

    fn transformers_from_config(cfg: GeminiConfig) -> EmbeddingTransformers {
        let req_tx = crate::standards::gemini::transformers::GeminiRequestTransformer {
            config: cfg.clone(),
        };
        let resp_tx =
            crate::standards::gemini::transformers::GeminiResponseTransformer { config: cfg };
        EmbeddingTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
        }
    }
}

impl Default for GeminiEmbeddingStandard {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GeminiEmbeddingSpec {
    provider_id: &'static str,
    adapter: Option<Arc<dyn GeminiEmbeddingAdapter>>,
}

impl ProviderSpec for GeminiEmbeddingSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_embedding()
    }
    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        let mut headers = build_gemini_headers(api_key, &ctx.http_extra_headers)?;
        if let Some(adapter) = &self.adapter {
            adapter.build_headers(api_key, &mut headers)?;
        }
        Ok(headers)
    }
    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = super::normalize_gemini_model_id(req.model.as_deref().unwrap_or(""));
        if let Some(adapter) = &self.adapter {
            if req.input.len() == 1 {
                format!("{}{}", base, adapter.embed_endpoint(&model))
            } else {
                format!("{}{}", base, adapter.batch_embed_endpoint(&model))
            }
        } else if req.input.len() == 1 {
            format!("{}/models/{}:embedContent", base, model)
        } else {
            format!("{}/models/{}:batchEmbedContents", base, model)
        }
    }
    fn choose_embedding_transformers(
        &self,
        req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        GeminiEmbeddingStandard {
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
    fn embedding_url_accepts_vertex_resource_style_model_ids() {
        let spec = GeminiEmbeddingStandard::new().create_spec("gemini");
        let ctx = ProviderContext::new(
            "gemini",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google".to_string(),
            Some("".to_string()),
            std::collections::HashMap::new(),
        );

        let req = crate::types::EmbeddingRequest::single("hi")
            .with_model("publishers/google/models/text-embedding-004");
        assert_eq!(
            spec.embedding_url(&req, &ctx),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/text-embedding-004:embedContent"
        );
    }
}
