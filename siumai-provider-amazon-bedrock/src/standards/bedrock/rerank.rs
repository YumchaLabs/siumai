//! Amazon Bedrock Rerank Standard (Vercel-aligned).
//!
//! Vercel reference: `repo-ref/ai/packages/amazon-bedrock/src/reranking/bedrock-reranking-model.ts`

use crate::core::{ProviderContext, ProviderSpec, RerankTransformers};
use crate::error::LlmError;
use crate::execution::transformers::rerank_request::RerankRequestTransformer;
use crate::execution::transformers::rerank_response::RerankResponseTransformer;
use crate::types::{
    RerankDocuments, RerankRequest, RerankResponse, RerankResult, RerankTokenUsage,
};
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct BedrockRerankStandard;

impl BedrockRerankStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(&self, provider_id: &str) -> RerankTransformers {
        RerankTransformers {
            request: Arc::new(BedrockRerankRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(BedrockRerankResponseTransformer {
                provider_id: provider_id.to_string(),
            }),
        }
    }

    pub fn create_spec(&self, provider_id: &'static str) -> BedrockRerankSpec {
        BedrockRerankSpec { provider_id }
    }
}

pub struct BedrockRerankSpec {
    provider_id: &'static str,
}

impl ProviderSpec for BedrockRerankSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_rerank()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("static header"),
        );

        // NOTE: Bedrock normally requires AWS SigV4 signing (or bearer token auth).
        // Users can inject signed headers via `ProviderContext.http_extra_headers`.
        if let Some(api_key) = ctx.api_key.as_deref().filter(|v| !v.trim().is_empty()) {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {api_key}").parse().map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid Bedrock bearer token: {e}"))
                })?,
            );
        }

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
        BedrockRerankStandard::new().create_transformers(&ctx.provider_id)
    }
}

struct BedrockRerankRequestTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

impl BedrockRerankRequestTransformer {
    fn bedrock_options(req: &RerankRequest) -> Option<&serde_json::Map<String, serde_json::Value>> {
        req.provider_options_map.get_object("bedrock")
    }
}

impl RerankRequestTransformer for BedrockRerankRequestTransformer {
    fn transform(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        // Vercel keeps region in provider config; for fixtures we accept providerOptions.bedrock.region.
        let region = req
            .provider_options_map
            .get("bedrock")
            .and_then(|v| v.get("region").and_then(|r| r.as_str()))
            .unwrap_or("us-east-1");

        let docs: Vec<serde_json::Value> = match &req.documents {
            RerankDocuments::Text(values) => values
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "INLINE",
                        "inlineDocumentSource": { "type": "TEXT", "textDocument": { "text": t } },
                    })
                })
                .collect(),
            RerankDocuments::Object(values) => values
                .iter()
                .map(|v| {
                    serde_json::json!({
                        "type": "INLINE",
                        "inlineDocumentSource": { "type": "JSON", "jsonDocument": v },
                    })
                })
                .collect(),
        };

        let mut body = serde_json::json!({
            "queries": [{ "textQuery": { "text": req.query }, "type": "TEXT" }],
            "rerankingConfiguration": {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": req.top_n,
                    "modelConfiguration": {
                        "modelArn": format!("arn:aws:bedrock:{region}::foundation-model/{}", req.model),
                    }
                }
            },
            "sources": docs,
        });

        if let Some(opts) = Self::bedrock_options(req) {
            if let Some(next_token) = opts.get("nextToken").and_then(|v| v.as_str()) {
                body["nextToken"] = serde_json::Value::String(next_token.to_string());
            }
            if let Some(fields) = opts.get("additionalModelRequestFields") {
                body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]
                    ["additionalModelRequestFields"] = fields.clone();
            }
        }

        Ok(body)
    }
}

struct BedrockRerankResponseTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

impl RerankResponseTransformer for BedrockRerankResponseTransformer {
    fn transform(&self, raw: serde_json::Value) -> Result<RerankResponse, LlmError> {
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
                let score = item
                    .get("relevanceScore")
                    .or_else(|| item.get("relevance_score"))
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| LlmError::ParseError("Missing 'relevanceScore' field".into()))?;
                Ok(RerankResult {
                    document: None,
                    index,
                    relevance_score: score,
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;

        Ok(RerankResponse {
            id: "".to_string(),
            results,
            tokens: RerankTokenUsage {
                input_tokens: 0,
                output_tokens: 0,
            },
        })
    }
}
