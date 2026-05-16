//! Cohere Rerank Standard (Vercel-aligned).
//!
//! Vercel reference: `repo-ref/ai/packages/cohere/src/reranking/cohere-reranking-model.ts`
//! API docs: <https://docs.cohere.com/v2/reference/rerank>

use super::shared;
use crate::core::{ProviderContext, ProviderSpec, RerankTransformers};
use crate::error::LlmError;
use crate::execution::transformers::rerank_request::RerankRequestTransformer;
use crate::execution::transformers::rerank_response::RerankResponseTransformer;
use crate::provider_options::CohereRerankOptions;
use crate::types::{
    HttpResponseInfo, RerankDocuments, RerankRankingEntry, RerankRequest, RerankResponse,
    RerankTokenUsage,
};
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct CohereRerankStandard;

impl CohereRerankStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(&self, provider_id: &str) -> RerankTransformers {
        RerankTransformers {
            request: Arc::new(CohereRerankRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(CohereRerankResponseTransformer {
                provider_id: provider_id.to_string(),
            }),
        }
    }

    pub fn create_spec(&self, provider_id: &'static str) -> CohereRerankSpec {
        CohereRerankSpec { provider_id }
    }
}

/// ProviderSpec implementation for Cohere reranking.
pub struct CohereRerankSpec {
    provider_id: &'static str,
}

impl ProviderSpec for CohereRerankSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_rerank()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        shared::build_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::cohere::errors::classify_cohere_http_error(
            self.provider_id,
            status,
            body_text,
            headers,
        )
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
        CohereRerankStandard::new().create_transformers(&ctx.provider_id)
    }
}

struct CohereRerankRequestTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

impl CohereRerankRequestTransformer {
    fn cohere_options(req: &RerankRequest) -> Result<Option<CohereRerankOptions>, LlmError> {
        shared::cohere_provider_options(&req.provider_options_map)
    }
}

impl RerankRequestTransformer for CohereRerankRequestTransformer {
    fn transform(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        let documents = match &req.documents {
            RerankDocuments::Text(values) => values.clone(),
            RerankDocuments::Object(_) => req.documents.to_strings_lossy(),
        };

        let mut body = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": documents,
        });

        if let Some(top_n) = req.top_n {
            body["top_n"] = serde_json::json!(top_n);
        }

        if let Some(opts) = Self::cohere_options(req)? {
            if let Some(max) = opts.max_tokens_per_doc {
                body["max_tokens_per_doc"] = serde_json::json!(max);
            }
            if let Some(priority) = opts.priority {
                body["priority"] = serde_json::json!(priority);
            }
        }

        Ok(body)
    }
}

struct CohereRerankResponseTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

impl CohereRerankResponseTransformer {
    fn response_info(raw: &serde_json::Value) -> HttpResponseInfo {
        HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: None,
            headers: HashMap::new(),
            body: Some(raw.clone()),
        }
    }
}

impl RerankResponseTransformer for CohereRerankResponseTransformer {
    fn transform(&self, raw: serde_json::Value) -> Result<RerankResponse, LlmError> {
        let id = raw
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let results = raw
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LlmError::ParseError("Missing 'results' field".into()))?
            .iter()
            .map(|item| {
                let index = item
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| LlmError::ParseError("Missing 'index' field".into()))?
                    as u32;

                let relevance_score = item
                    .get("relevance_score")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| {
                        LlmError::ParseError("Missing 'relevance_score' field".into())
                    })?;

                Ok(RerankRankingEntry {
                    document: None,
                    index,
                    relevance_score,
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;

        // Cohere v2 uses billed units (search_units) instead of explicit token counts.
        // We keep a best-effort mapping into `input_tokens` for parity with existing APIs.
        let input_tokens = raw
            .get("meta")
            .and_then(|m| m.get("billed_units"))
            .and_then(|b| b.get("search_units"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        Ok(RerankResponse {
            id,
            results,
            tokens: RerankTokenUsage {
                input_tokens,
                output_tokens: 0,
            },
            response: Some(Self::response_info(&raw)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_transformer_preserves_ai_sdk_response_body() {
        let raw = serde_json::json!({
            "id": "rerank-id",
            "results": [
                { "index": 0, "relevance_score": 0.74 }
            ],
            "meta": {
                "api_version": { "version": "2" },
                "billed_units": { "search_units": 1 }
            }
        });

        let transformer = CohereRerankResponseTransformer {
            provider_id: "cohere".to_string(),
        };

        let response = transformer.transform(raw).expect("transform response");
        let response_info = response.response.expect("response metadata");

        assert!(response_info.model_id.is_none());
        assert!(response_info.headers.is_empty());
        assert_eq!(
            response_info.body.as_ref().expect("raw body")["id"],
            "rerank-id"
        );
        assert_eq!(
            response_info.body.as_ref().expect("raw body")["meta"]["api_version"]["version"],
            "2"
        );
    }
}
