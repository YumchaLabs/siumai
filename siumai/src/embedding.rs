//! Embedding model family APIs.
//!
//! This is the recommended Rust-first surface for embeddings:
//! - `embed` for a single request
//! - `embed_many` for batch requests

use crate::request_options::{EffectiveRequestOptions, retry_or_call_with_abort};
use crate::retry_api::RetryOptions;
use siumai_core::error::LlmError;
use siumai_core::types::{
    EmbedManyResult, EmbedResult, EmbeddingModelUsage, HttpConfig, HttpResponseInfo, JSONValue,
    ModelCallResponseData, ProviderMetadata, RequestOptions, provider_metadata_from_object,
};
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::embedding::EmbeddingModel;
pub use siumai_core::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
    EmbeddingTaskType,
};

/// Options for `embedding::embed` and `embedding::embed_many`.
#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `EmbeddingRequest.http_config.timeout` (for each request in a batch).
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `EmbeddingRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
}

fn apply_embedding_call_options(
    mut request: EmbeddingRequest,
    timeout: Option<Duration>,
    headers: &HashMap<String, String>,
) -> EmbeddingRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(t) = timeout {
            http.timeout = Some(t);
        }
        if !headers.is_empty() {
            http.headers.extend(headers.clone());
        }
        request.http_config = Some(http);
    }
    request
}

/// Generate embeddings for a single request.
pub async fn embed<M: EmbeddingModel + ?Sized>(
    model: &M,
    request: EmbeddingRequest,
    options: EmbedOptions,
) -> Result<EmbeddingResponse, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let headers = effective.headers();
    let request = apply_embedding_call_options(request, effective.timeout(), &headers);
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let req = request.clone();
        async move { model.embed(req).await }
    })
    .await
}

/// Generate embeddings for a batch of requests.
pub async fn embed_many<M: EmbeddingModel + ?Sized>(
    model: &M,
    requests: BatchEmbeddingRequest,
    options: EmbedOptions,
) -> Result<BatchEmbeddingResponse, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let mut requests = requests;
    if effective.timeout().is_some() || !effective.headers.is_empty() {
        let timeout = effective.timeout();
        let headers = effective.headers();
        requests.requests = requests
            .requests
            .into_iter()
            .map(|r| apply_embedding_call_options(r, timeout, &headers))
            .collect();
    }
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let req = requests.clone();
        async move { model.embed_many(req).await }
    })
    .await
}

fn model_call_response_data(response: &HttpResponseInfo) -> Option<ModelCallResponseData> {
    let mut data = ModelCallResponseData::new();
    let mut has_data = false;

    if !response.headers.is_empty() {
        data = data.with_headers(response.headers.clone());
        has_data = true;
    }

    if let Some(body) = response.body.clone() {
        data = data.with_body(body);
        has_data = true;
    }

    has_data.then_some(data)
}

fn provider_metadata_from_embedding_metadata(
    provider_id: &str,
    metadata: HashMap<String, JSONValue>,
) -> Option<ProviderMetadata> {
    (!metadata.is_empty()).then(|| provider_metadata_from_object(provider_id.to_string(), metadata))
}

fn embedding_usage_or_default(
    usage: Option<siumai_core::types::EmbeddingUsage>,
) -> EmbeddingModelUsage {
    usage
        .as_ref()
        .map(EmbeddingModelUsage::from)
        .unwrap_or_else(|| EmbeddingModelUsage::new(0))
}

fn project_embed_response(
    provider_id: &str,
    value: String,
    response: EmbeddingResponse,
) -> Result<EmbedResult, LlmError> {
    let EmbeddingResponse {
        embeddings,
        usage,
        metadata,
        response,
        ..
    } = response;
    let embedding = embeddings.into_iter().next().ok_or_else(|| {
        LlmError::ParseError(
            "Embedding model returned no embedding for the input value".to_string(),
        )
    })?;

    let mut result = EmbedResult::new(value, embedding, embedding_usage_or_default(usage));

    if let Some(provider_metadata) =
        provider_metadata_from_embedding_metadata(provider_id, metadata)
    {
        result = result.with_provider_metadata(provider_metadata);
    }

    if let Some(response_data) = response.as_ref().and_then(model_call_response_data) {
        result = result.with_response(response_data);
    }

    Ok(result)
}

fn project_embed_many_response(
    provider_id: &str,
    values: Vec<String>,
    response: EmbeddingResponse,
) -> Result<EmbedManyResult, LlmError> {
    let EmbeddingResponse {
        embeddings,
        usage,
        metadata,
        response,
        ..
    } = response;

    if embeddings.len() != values.len() {
        return Err(LlmError::ParseError(format!(
            "Embedding model returned {} embeddings for {} input values",
            embeddings.len(),
            values.len()
        )));
    }

    let response_data = response.as_ref().and_then(model_call_response_data);
    let mut result = EmbedManyResult::new(values, embeddings, embedding_usage_or_default(usage))
        .with_responses(vec![response_data]);

    if let Some(provider_metadata) =
        provider_metadata_from_embedding_metadata(provider_id, metadata)
    {
        result = result.with_provider_metadata(provider_metadata);
    }

    Ok(result)
}

/// Generate an AI SDK-style `EmbedResult` from a single-value embedding request.
///
/// Use `embedding::embed` when you need the raw Rust-first `EmbeddingResponse`.
pub async fn generate_embedding<M: EmbeddingModel + ?Sized>(
    model: &M,
    request: EmbeddingRequest,
    options: EmbedOptions,
) -> Result<EmbedResult, LlmError> {
    let value = match request.input.as_slice() {
        [value] => value.clone(),
        [] => {
            return Err(LlmError::InvalidParameter(
                "generate_embedding requires exactly one input value".to_string(),
            ));
        }
        values => {
            return Err(LlmError::InvalidParameter(format!(
                "generate_embedding requires exactly one input value, but {} were provided",
                values.len()
            )));
        }
    };

    let response = embed(model, request, options).await?;
    project_embed_response(model.provider_id(), value, response)
}

/// Generate an AI SDK-style `EmbedResult` for one text value.
pub async fn embed_value<M, V>(
    model: &M,
    value: V,
    options: EmbedOptions,
) -> Result<EmbedResult, LlmError>
where
    M: EmbeddingModel + ?Sized,
    V: Into<String>,
{
    generate_embedding(model, EmbeddingRequest::single(value), options).await
}

/// Generate an AI SDK-style `EmbedManyResult` from one multi-value embedding request.
///
/// This projects one underlying model call into `responses[0]`, matching the AI SDK result shape
/// when the provider can embed all values in a single call.
pub async fn generate_embeddings<M: EmbeddingModel + ?Sized>(
    model: &M,
    request: EmbeddingRequest,
    options: EmbedOptions,
) -> Result<EmbedManyResult, LlmError> {
    let values = request.input.clone();
    let response = embed(model, request, options).await?;
    project_embed_many_response(model.provider_id(), values, response)
}

/// Generate an AI SDK-style `EmbedManyResult` for several text values.
pub async fn embed_values<M: EmbeddingModel + ?Sized>(
    model: &M,
    values: Vec<String>,
    options: EmbedOptions,
) -> Result<EmbedManyResult, LlmError> {
    generate_embeddings(model, EmbeddingRequest::new(values), options).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::TimeZone;
    use siumai_core::traits::ModelMetadata;
    use siumai_core::types::{EmbeddingUsage, HttpResponseInfo};

    struct FakeEmbeddingModel {
        response: EmbeddingResponse,
    }

    impl ModelMetadata for FakeEmbeddingModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-embedding"
        }
    }

    #[async_trait]
    impl EmbeddingModel for FakeEmbeddingModel {
        async fn embed(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
            Ok(self.response.clone())
        }

        async fn embed_many(
            &self,
            _requests: BatchEmbeddingRequest,
        ) -> Result<BatchEmbeddingResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "batch embedding is not used in this test".to_string(),
            ))
        }
    }

    fn response_envelope() -> HttpResponseInfo {
        HttpResponseInfo {
            timestamp: chrono::Utc
                .with_ymd_and_hms(2026, 1, 2, 3, 4, 5)
                .single()
                .expect("valid timestamp"),
            model_id: Some("transport-model".to_string()),
            headers: HashMap::from([("x-request-id".to_string(), "req_1".to_string())]),
            body: Some(serde_json::json!({ "raw": true })),
        }
    }

    #[tokio::test]
    async fn generate_embedding_projects_ai_sdk_result() {
        let model = FakeEmbeddingModel {
            response: EmbeddingResponse::new(vec![vec![1.0, 2.0]], "model".to_string())
                .with_usage(EmbeddingUsage::new(7, 7))
                .with_metadata("traceId".to_string(), serde_json::json!("trace_1"))
                .with_response(response_envelope()),
        };

        let result = generate_embedding(
            &model,
            EmbeddingRequest::single("hello"),
            EmbedOptions::default(),
        )
        .await
        .expect("project embed result");

        assert_eq!(result.value, "hello");
        assert_eq!(result.embedding, vec![1.0, 2.0]);
        assert_eq!(result.usage.tokens, 7);
        assert_eq!(
            result.provider_metadata.as_ref().expect("metadata")["fake"]["traceId"],
            serde_json::json!("trace_1")
        );
        let response = result.response.expect("response data");
        assert_eq!(response.headers.expect("headers")["x-request-id"], "req_1");
        assert_eq!(
            response.body.expect("raw body"),
            serde_json::json!({ "raw": true })
        );
    }

    #[tokio::test]
    async fn generate_embeddings_projects_single_call_response() {
        let model = FakeEmbeddingModel {
            response: EmbeddingResponse::new(vec![vec![1.0], vec![2.0]], "model".to_string())
                .with_usage(EmbeddingUsage::new(11, 11))
                .with_response(response_envelope()),
        };

        let result = generate_embeddings(
            &model,
            EmbeddingRequest::new(vec!["a".to_string(), "bb".to_string()]),
            EmbedOptions::default(),
        )
        .await
        .expect("project embedMany result");

        assert_eq!(result.values, vec!["a".to_string(), "bb".to_string()]);
        assert_eq!(result.embeddings, vec![vec![1.0], vec![2.0]]);
        assert_eq!(result.usage.tokens, 11);
        let responses = result.responses.expect("responses");
        assert_eq!(responses.len(), 1);
        assert_eq!(
            responses[0]
                .as_ref()
                .expect("response data")
                .body
                .as_ref()
                .expect("raw body"),
            &serde_json::json!({ "raw": true })
        );
    }
}
