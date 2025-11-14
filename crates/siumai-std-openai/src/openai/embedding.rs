//! OpenAI Embeddings API Standard (external)
//!
//! Implements request/response transformers for OpenAI's Embeddings API,
//! using core's minimal embedding traits/types.

use siumai_core::error::LlmError;
use siumai_core::execution::embedding::{
    EmbeddingInput, EmbeddingRequestTransformer, EmbeddingResponseTransformer, EmbeddingResult,
    EmbeddingUsage,
};
use std::sync::Arc;

/// OpenAI Embedding API Standard
#[derive(Clone)]
pub struct OpenAiEmbeddingStandard {
    adapter: Option<Arc<dyn OpenAiEmbeddingAdapter>>,
}

impl OpenAiEmbeddingStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }
    pub fn with_adapter(adapter: Arc<dyn OpenAiEmbeddingAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    pub fn create_request_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn EmbeddingRequestTransformer> {
        Arc::new(OpenAiEmbeddingRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn EmbeddingResponseTransformer> {
        Arc::new(OpenAiEmbeddingResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
}

impl Default for OpenAiEmbeddingStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter trait for provider-specific differences in OpenAI Embedding API
pub trait OpenAiEmbeddingAdapter: Send + Sync {
    fn transform_request(
        &self,
        _req: &EmbeddingInput,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }
    fn embedding_endpoint(&self) -> &str {
        "/embeddings"
    }
}

#[derive(Clone)]
struct OpenAiEmbeddingRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiEmbeddingAdapter>>,
}

impl EmbeddingRequestTransformer for OpenAiEmbeddingRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding(&self, req: &EmbeddingInput) -> Result<serde_json::Value, LlmError> {
        // Base OpenAI embedding body
        let mut body = serde_json::json!({
            "input": req.input,
        });

        if let Some(model) = &req.model {
            body["model"] = serde_json::json!(model);
        }
        if let Some(dim) = req.dimensions {
            body["dimensions"] = serde_json::json!(dim);
        }
        if let Some(fmt) = &req.encoding_format {
            body["encoding_format"] = serde_json::json!(fmt);
        }
        if let Some(user) = &req.user {
            body["user"] = serde_json::json!(user);
        }

        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }
        Ok(body)
    }
}

#[derive(Clone)]
struct OpenAiEmbeddingResponseTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiEmbeddingAdapter>>,
}

impl EmbeddingResponseTransformer for OpenAiEmbeddingResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResult, LlmError> {
        let mut raw = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut raw)?;
        }

        // Minimal OpenAI embedding response mapping
        #[derive(serde::Deserialize)]
        struct DataItem {
            embedding: Vec<f32>,
        }
        #[derive(serde::Deserialize)]
        struct Usage {
            prompt_tokens: Option<u32>,
            total_tokens: Option<u32>,
        }
        #[derive(serde::Deserialize)]
        struct Resp {
            data: Vec<DataItem>,
            model: String,
            #[serde(default)]
            usage: Option<Usage>,
        }

        let r: Resp = serde_json::from_value(raw)
            .map_err(|e| LlmError::ParseError(format!("Invalid embedding response: {}", e)))?;
        let vectors = r.data.into_iter().map(|d| d.embedding).collect::<Vec<_>>();
        let usage = r
            .usage
            .and_then(|u| match (u.prompt_tokens, u.total_tokens) {
                (Some(p), Some(t)) => Some(EmbeddingUsage {
                    prompt_tokens: p,
                    total_tokens: t,
                }),
                _ => None,
            });
        Ok(EmbeddingResult {
            embeddings: vectors,
            model: r.model,
            usage,
            metadata: Default::default(),
        })
    }
}
