//! Vertex AI text embedding standard (via `:predict`).
//!
//! Vercel AI SDK parity:
//! - URL: `{baseURL}/models/{model}:predict`
//! - Body: `{ instances: [{ content, task_type, title }...], parameters: { outputDimensionality, autoTruncate } }`
//! - Provider options lookup: `providerOptions["vertex"]` then fallback to `providerOptions["google"]`.

use crate::core::{EmbeddingTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::{EmbeddingRequest, EmbeddingResponse, EmbeddingTaskType, EmbeddingUsage};
use reqwest::header::HeaderMap;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

const VERTEX_USER_AGENT: &str = concat!("siumai/google-vertex/", env!("CARGO_PKG_VERSION"));

fn has_auth_header(headers: &HashMap<String, String>) -> bool {
    headers
        .keys()
        .any(|k| k.eq_ignore_ascii_case("authorization"))
}

fn append_api_key_query(url: String, api_key: &str) -> String {
    let key = urlencoding::encode(api_key);
    if url.contains('?') {
        format!("{url}&key={key}")
    } else {
        format!("{url}?key={key}")
    }
}

fn build_vertex_headers(custom_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
    let builder = HttpHeaderBuilder::new()
        .with_json_content_type()
        .with_user_agent(VERTEX_USER_AGENT)?
        .with_custom_headers(custom_headers)?;
    Ok(builder.build())
}

fn normalize_vertex_model_id(model: &str) -> String {
    let trimmed = model.trim().trim_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(pos) = trimmed.rfind("/models/") {
        return trimmed[(pos + "/models/".len())..].to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("models/") {
        return rest.to_string();
    }
    trimmed.to_string()
}

fn task_type_to_vertex_value(task_type: &EmbeddingTaskType) -> Cow<'static, str> {
    match task_type {
        EmbeddingTaskType::RetrievalQuery => Cow::Borrowed("RETRIEVAL_QUERY"),
        EmbeddingTaskType::RetrievalDocument => Cow::Borrowed("RETRIEVAL_DOCUMENT"),
        EmbeddingTaskType::SemanticSimilarity => Cow::Borrowed("SEMANTIC_SIMILARITY"),
        EmbeddingTaskType::Classification => Cow::Borrowed("CLASSIFICATION"),
        EmbeddingTaskType::Clustering => Cow::Borrowed("CLUSTERING"),
        EmbeddingTaskType::QuestionAnswering => Cow::Borrowed("QUESTION_ANSWERING"),
        EmbeddingTaskType::FactVerification => Cow::Borrowed("FACT_VERIFICATION"),
        EmbeddingTaskType::CodeRetrievalQuery => Cow::Borrowed("CODE_RETRIEVAL_QUERY"),
        EmbeddingTaskType::Unspecified => Cow::Borrowed("UNSPECIFIED"),
    }
}

fn extract_embedding_provider_options(
    map: &crate::types::ProviderOptionsMap,
) -> Option<&serde_json::Value> {
    if let Some(v) = map.get("vertex") {
        return Some(v);
    }
    map.get("google")
}

fn opt_u32(value: Option<&serde_json::Value>) -> Option<u32> {
    value
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok())
}

fn opt_bool(value: Option<&serde_json::Value>) -> Option<bool> {
    value.and_then(|v| v.as_bool())
}

fn opt_string(value: Option<&serde_json::Value>) -> Option<String> {
    value.and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn opt_task_type(value: Option<&serde_json::Value>) -> Option<String> {
    value.and_then(|v| v.as_str()).map(|s| s.to_string())
}

const VERTEX_EMBEDDING_MAX_VALUES_PER_CALL: usize = 2048;

#[derive(Clone, Default)]
pub struct VertexEmbeddingStandard;

impl VertexEmbeddingStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_spec(&self, provider_id: &'static str) -> VertexEmbeddingSpec {
        VertexEmbeddingSpec { provider_id }
    }

    pub fn create_transformers_with_model(&self, model: Option<&str>) -> EmbeddingTransformers {
        EmbeddingTransformers {
            request: Arc::new(VertexEmbeddingRequestTransformer {
                provider_id: "vertex".to_string(),
            }),
            response: Arc::new(VertexEmbeddingResponseTransformer {
                provider_id: "vertex".to_string(),
                model: model.unwrap_or_default().to_string(),
            }),
        }
    }
}

pub struct VertexEmbeddingSpec {
    provider_id: &'static str,
}

impl ProviderSpec for VertexEmbeddingSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_embedding()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_vertex_headers(&ctx.http_extra_headers)
    }

    fn embedding_url(&self, req: &EmbeddingRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        let url = format!("{}/models/{}:predict", base, model);

        if let Some(key) = ctx.api_key.as_deref()
            && !key.is_empty()
            && !has_auth_header(&ctx.http_extra_headers)
        {
            append_api_key_query(url, key)
        } else {
            url
        }
    }

    fn choose_embedding_transformers(
        &self,
        req: &EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        VertexEmbeddingStandard::new().create_transformers_with_model(req.model.as_deref())
    }
}

#[derive(Clone)]
struct VertexEmbeddingRequestTransformer {
    provider_id: String,
}

impl RequestTransformer for VertexEmbeddingRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "vertex embedding transformer does not support chat".to_string(),
        ))
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        if req.input.len() > VERTEX_EMBEDDING_MAX_VALUES_PER_CALL {
            return Err(LlmError::InvalidInput(format!(
                "Too many embedding values for a single call: {} (max {})",
                req.input.len(),
                VERTEX_EMBEDDING_MAX_VALUES_PER_CALL
            )));
        }

        let provider_opts = extract_embedding_provider_options(&req.provider_options_map)
            .and_then(|v| v.as_object());

        let output_dimensionality = req
            .dimensions
            .or_else(|| opt_u32(provider_opts.and_then(|o| o.get("outputDimensionality"))))
            .or_else(|| opt_u32(provider_opts.and_then(|o| o.get("output_dimensionality"))));

        let auto_truncate = opt_bool(provider_opts.and_then(|o| o.get("autoTruncate")))
            .or_else(|| opt_bool(provider_opts.and_then(|o| o.get("auto_truncate"))));

        let title = req
            .title
            .clone()
            .or_else(|| opt_string(provider_opts.and_then(|o| o.get("title"))));

        let task_type = req
            .task_type
            .as_ref()
            .map(|t| task_type_to_vertex_value(t).to_string())
            .or_else(|| {
                opt_task_type(provider_opts.and_then(|o| o.get("taskType")))
                    .or_else(|| opt_task_type(provider_opts.and_then(|o| o.get("task_type"))))
            });

        let instances: Vec<serde_json::Value> = req
            .input
            .iter()
            .map(|content| {
                let mut obj = serde_json::Map::new();
                obj.insert("content".to_string(), serde_json::json!(content));
                if let Some(tt) = &task_type {
                    obj.insert("task_type".to_string(), serde_json::json!(tt));
                }
                if let Some(t) = &title {
                    obj.insert("title".to_string(), serde_json::json!(t));
                }
                serde_json::Value::Object(obj)
            })
            .collect();

        let mut params = serde_json::Map::new();
        if let Some(v) = output_dimensionality {
            params.insert("outputDimensionality".to_string(), serde_json::json!(v));
        }
        if let Some(v) = auto_truncate {
            params.insert("autoTruncate".to_string(), serde_json::json!(v));
        }

        let mut body = serde_json::Map::new();
        body.insert("instances".to_string(), serde_json::Value::Array(instances));
        if !params.is_empty() {
            body.insert("parameters".to_string(), serde_json::Value::Object(params));
        } else {
            body.insert(
                "parameters".to_string(),
                serde_json::Value::Object(serde_json::Map::new()),
            );
        }

        Ok(serde_json::Value::Object(body))
    }
}

#[derive(Clone)]
struct VertexEmbeddingResponseTransformer {
    provider_id: String,
    model: String,
}

impl ResponseTransformer for VertexEmbeddingResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        let predictions = raw
            .get("predictions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                LlmError::ParseError("Vertex embedding response missing predictions".to_string())
            })?;

        let mut embeddings = Vec::with_capacity(predictions.len());
        let mut token_count: u32 = 0;

        for pred in predictions {
            let values = pred
                .get("embeddings")
                .and_then(|v| v.get("values"))
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    LlmError::ParseError(
                        "Vertex embedding response missing embeddings.values".to_string(),
                    )
                })?;

            let mut vec = Vec::with_capacity(values.len());
            for n in values {
                let f = n.as_f64().ok_or_else(|| {
                    LlmError::ParseError("Vertex embedding values must be numbers".to_string())
                })? as f32;
                vec.push(f);
            }
            embeddings.push(vec);

            if let Some(tc) = pred
                .get("embeddings")
                .and_then(|v| v.get("statistics"))
                .and_then(|v| v.get("token_count"))
                .and_then(|v| v.as_u64())
                .and_then(|n| u32::try_from(n).ok())
            {
                token_count = token_count.saturating_add(tc);
            }
        }

        let model = if self.model.is_empty() {
            "vertex-embedding".to_string()
        } else {
            self.model.clone()
        };

        let mut out = EmbeddingResponse::new(embeddings, model);
        if token_count > 0 {
            out = out.with_usage(EmbeddingUsage::new(token_count, token_count));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_embedding_rejects_too_many_values() {
        let transformer = VertexEmbeddingRequestTransformer {
            provider_id: "vertex".to_string(),
        };

        let input = vec!["test".to_string(); VERTEX_EMBEDDING_MAX_VALUES_PER_CALL + 1];
        let req = EmbeddingRequest::new(input);
        let err = transformer
            .transform_embedding(&req)
            .expect_err("expected error");
        assert!(
            matches!(err, LlmError::InvalidInput(_)),
            "expected InvalidInput, got: {err:?}"
        );
    }
}
