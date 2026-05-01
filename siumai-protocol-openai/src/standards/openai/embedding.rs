//! OpenAI Embeddings API Standard
//!
//! This module implements the OpenAI Embeddings API format.

use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::{EmbeddingRequest, EmbeddingResponse, HttpResponseInfo};
use crate::{core::ProviderContext, core::ProviderSpec};
use std::collections::HashMap;
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

    /// Create a ProviderSpec wrapper for this standard.
    pub fn create_spec(&self, provider_id: &'static str) -> OpenAiEmbeddingSpec {
        OpenAiEmbeddingSpec {
            provider_id,
            adapter: self.adapter.clone(),
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

    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// ProviderSpec implementation for OpenAI Embedding Standard.
pub struct OpenAiEmbeddingSpec {
    provider_id: &'static str,
    adapter: Option<Arc<dyn OpenAiEmbeddingAdapter>>,
}

impl ProviderSpec for OpenAiEmbeddingSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_embedding()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers =
            crate::standards::openai::headers::build_openai_compatible_json_headers(ctx)?;

        if let Some(adapter) = &self.adapter {
            adapter.build_headers(ctx.api_key.as_deref().unwrap_or(""), &mut headers)?;
        }

        Ok(headers)
    }

    fn choose_embedding_transformers(
        &self,
        _req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> crate::core::EmbeddingTransformers {
        let standard = OpenAiEmbeddingStandard {
            adapter: self.adapter.clone(),
        };
        crate::core::EmbeddingTransformers {
            request: standard.create_request_transformer(&ctx.provider_id),
            response: standard.create_response_transformer(&ctx.provider_id),
        }
    }

    fn embedding_url(&self, _req: &EmbeddingRequest, ctx: &ProviderContext) -> String {
        let endpoint = self
            .adapter
            .as_ref()
            .map(|a| a.embedding_endpoint())
            .unwrap_or("/embeddings");
        crate::utils::url::join_url(&ctx.base_url, endpoint)
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
        let openai_tx = crate::standards::openai::transformers::OpenAiRequestTransformer;
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
        let raw_body = raw.clone();
        let mut raw = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut raw)?;
        }

        let openai_tx = crate::standards::openai::transformers::OpenAiResponseTransformer;
        let mut response = openai_tx.transform_embedding_response(&raw)?;
        response.response = Some(HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: raw_body
                .get("model")
                .or_else(|| raw.get("model"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| Some(response.model.clone())),
            headers: HashMap::new(),
            body: Some(raw_body),
        });
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct NormalizingEmbeddingAdapter;

    impl OpenAiEmbeddingAdapter for NormalizingEmbeddingAdapter {
        fn transform_response(&self, resp: &mut serde_json::Value) -> Result<(), LlmError> {
            let provider_model = resp
                .get("provider_model")
                .and_then(|v| v.as_str())
                .unwrap_or("normalized-model");
            let vectors = resp
                .get("vectors")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            *resp = json!({
                "object": "list",
                "model": provider_model,
                "data": vectors
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| {
                        json!({
                            "object": "embedding",
                            "index": index,
                            "embedding": embedding,
                        })
                    })
                    .collect::<Vec<_>>(),
                "usage": {
                    "prompt_tokens": 3,
                    "total_tokens": 3
                }
            });
            Ok(())
        }
    }

    #[test]
    fn embedding_response_preserves_adapter_raw_response_body() {
        let tx = OpenAiEmbeddingStandard::with_adapter(Arc::new(NormalizingEmbeddingAdapter))
            .create_response_transformer("test-provider");
        let raw = json!({
            "provider_model": "provider-embedding-model",
            "vectors": [[0.3, 0.7]],
            "trace": {
                "id": "trace-123"
            }
        });

        let response = tx
            .transform_embedding_response(&raw)
            .expect("embedding response");
        let response_info = response.response.expect("response info");
        let body = response_info.body.expect("raw response body");

        assert_eq!(response.model, "provider-embedding-model");
        assert_eq!(
            response_info.model_id.as_deref(),
            Some("provider-embedding-model")
        );
        assert_eq!(body["trace"]["id"], "trace-123");
        assert!(body.get("data").is_none());
    }
}
