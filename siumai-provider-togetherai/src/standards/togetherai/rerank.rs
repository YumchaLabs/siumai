//! TogetherAI Rerank Standard (Vercel-aligned).
//!
//! Vercel reference: `repo-ref/ai/packages/togetherai/src/reranking/togetherai-reranking-model.ts`
//! API docs: <https://docs.together.ai/reference/rerank-1>

use crate::core::{ProviderContext, ProviderSpec, RerankTransformers};
use crate::error::LlmError;
use crate::execution::transformers::rerank_request::RerankRequestTransformer;
use crate::execution::transformers::rerank_response::RerankResponseTransformer;
use crate::types::{
    RerankDocuments, RerankRequest, RerankResponse, RerankResult, RerankTokenUsage,
};
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct TogetherAiRerankStandard;

impl TogetherAiRerankStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(&self, provider_id: &str) -> RerankTransformers {
        RerankTransformers {
            request: Arc::new(TogetherAiRerankRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(TogetherAiRerankResponseTransformer {
                provider_id: provider_id.to_string(),
            }),
        }
    }

    pub fn create_spec(&self, provider_id: &'static str) -> TogetherAiRerankSpec {
        TogetherAiRerankSpec { provider_id }
    }
}

/// ProviderSpec implementation for TogetherAI reranking.
pub struct TogetherAiRerankSpec {
    provider_id: &'static str,
}

impl ProviderSpec for TogetherAiRerankSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_rerank()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().ok_or_else(|| {
            LlmError::ConfigurationError("TogetherAI API key is required".to_string())
        })?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {api_key}").parse().map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid TogetherAI API key: {e}"))
            })?,
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("static header"),
        );

        // Preserve custom headers (Vercel-aligned: user headers override defaults).
        for (k, v) in &ctx.http_extra_headers {
            if let (Ok(name), Ok(value)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(v),
            ) {
                headers.insert(name, value);
            }
        }

        Ok(headers)
    }

    fn rerank_url(&self, _req: &RerankRequest, ctx: &ProviderContext) -> String {
        crate::utils::url::join_url(&ctx.base_url, "/rerank")
    }

    fn choose_rerank_transformers(
        &self,
        _req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> RerankTransformers {
        TogetherAiRerankStandard::new().create_transformers(&ctx.provider_id)
    }
}

struct TogetherAiRerankRequestTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

impl TogetherAiRerankRequestTransformer {
    fn togetherai_options(
        req: &RerankRequest,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        req.provider_options_map.get_object("togetherai")
    }

    fn get_string_array(
        obj: &serde_json::Map<String, serde_json::Value>,
        camel: &str,
        snake: &str,
    ) -> Option<Vec<String>> {
        let v = obj.get(camel).or_else(|| obj.get(snake))?;
        let arr = v.as_array()?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect();
        if out.is_empty() { None } else { Some(out) }
    }
}

impl RerankRequestTransformer for TogetherAiRerankRequestTransformer {
    fn transform(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        let documents = match &req.documents {
            RerankDocuments::Text(values) => {
                serde_json::Value::Array(values.iter().map(|s| serde_json::json!(s)).collect())
            }
            RerankDocuments::Object(values) => serde_json::Value::Array(values.clone()),
        };

        let mut body = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": documents,
            // Vercel-aligned: reduce response size.
            "return_documents": false,
        });

        if let Some(top_n) = req.top_n {
            body["top_n"] = serde_json::json!(top_n);
        }

        if let Some(opts) = Self::togetherai_options(req)
            && let Some(fields) = Self::get_string_array(opts, "rankFields", "rank_fields")
        {
            body["rank_fields"] = serde_json::json!(fields);
        }

        Ok(body)
    }
}

struct TogetherAiRerankResponseTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

impl RerankResponseTransformer for TogetherAiRerankResponseTransformer {
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

                Ok(RerankResult {
                    document: None,
                    index,
                    relevance_score,
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;

        let input_tokens = raw
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let output_tokens = raw
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        Ok(RerankResponse {
            id,
            results,
            tokens: RerankTokenUsage {
                input_tokens,
                output_tokens,
            },
        })
    }
}
