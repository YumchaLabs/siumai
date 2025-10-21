//! OpenAI Embeddings API Standard
//!
//! This module implements the OpenAI Embeddings API format.

use crate::error::LlmError;
use crate::transformers::request::RequestTransformer;
use crate::transformers::response::ResponseTransformer;
use crate::types::{EmbeddingRequest, EmbeddingResponse};
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

    pub fn create_request_transformer(&self, provider_id: &str) -> Arc<dyn RequestTransformer> {
        Arc::new(OpenAiEmbeddingRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }

    pub fn create_response_transformer(&self, provider_id: &str) -> Arc<dyn ResponseTransformer> {
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
        _req: &EmbeddingRequest,
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

impl RequestTransformer for OpenAiEmbeddingRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Embedding transformer does not support chat".to_string(),
        ))
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        let openai_tx = crate::providers::openai::transformers::request::OpenAiRequestTransformer;
        let mut body = openai_tx.transform_embedding(req)?;

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

impl ResponseTransformer for OpenAiEmbeddingResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        let mut raw = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut raw)?;
        }

        let openai_tx = crate::providers::openai::transformers::response::OpenAiResponseTransformer;
        openai_tx.transform_embedding_response(&raw)
    }
}
