//! Rerank model family APIs.
//!
//! This is the recommended Rust-first surface for reranking:
//! - `rerank`

use crate::request_options::{EffectiveRequestOptions, retry_or_call_with_abort};
use crate::retry_api::RetryOptions;
use chrono::Utc;
use siumai_core::error::LlmError;
use siumai_core::types::{
    HttpConfig, JSONValue, RequestOptions, RerankDocuments, RerankRanking, RerankResponseMetadata,
    RerankResult,
};
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::rerank::RerankingModel;
pub use siumai_core::types::{RerankRankingEntry, RerankRequest, RerankResponse};

/// Options for `rerank::rerank`.
#[derive(Debug, Clone, Default)]
pub struct RerankOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `RerankRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `RerankRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
}

fn apply_rerank_call_options(
    mut request: RerankRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> RerankRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(t) = timeout {
            http.timeout = Some(t);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }
    request
}

/// Rerank candidates for a query.
pub async fn rerank<M: RerankingModel + ?Sized>(
    model: &M,
    request: RerankRequest,
    options: RerankOptions,
) -> Result<RerankResponse, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let request = apply_rerank_call_options(request, effective.timeout(), effective.headers());
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let req = request.clone();
        async move { model.rerank(req).await }
    })
    .await
}

fn fallback_rerank_model_id<M: RerankingModel + ?Sized>(
    request: &RerankRequest,
    model: &M,
) -> String {
    if request.model.is_empty() {
        model.model_id().to_string()
    } else {
        request.model.clone()
    }
}

fn response_metadata_from_rerank_response(
    request_model_id: String,
    response: &RerankResponse,
) -> RerankResponseMetadata {
    let mut metadata = response
        .response
        .as_ref()
        .map(RerankResponseMetadata::from)
        .unwrap_or_else(|| RerankResponseMetadata::new(Utc::now(), request_model_id.clone()));

    if metadata.model_id.is_empty() {
        metadata.model_id = request_model_id;
    }

    metadata.with_id(response.id.clone())
}

fn text_document_for_ranking(
    original_documents: &[String],
    entry: &RerankRankingEntry,
) -> Result<String, LlmError> {
    original_documents
        .get(entry.index as usize)
        .cloned()
        .or_else(|| {
            entry
                .document
                .as_ref()
                .map(|document| document.text.clone())
        })
        .ok_or_else(|| {
            LlmError::ParseError(format!(
                "Rerank response referenced missing document index {}",
                entry.index
            ))
        })
}

fn json_document_for_ranking(
    original_documents: &[JSONValue],
    entry: &RerankRankingEntry,
) -> Result<JSONValue, LlmError> {
    original_documents
        .get(entry.index as usize)
        .cloned()
        .or_else(|| {
            entry
                .document
                .as_ref()
                .map(|document| JSONValue::String(document.text.clone()))
        })
        .ok_or_else(|| {
            LlmError::ParseError(format!(
                "Rerank response referenced missing document index {}",
                entry.index
            ))
        })
}

fn request_documents_as_json(request: &RerankRequest) -> Vec<JSONValue> {
    match &request.documents {
        RerankDocuments::Text(values) => values.iter().cloned().map(JSONValue::String).collect(),
        RerankDocuments::Object(values) => values.clone(),
    }
}

fn project_text_rerank_response(
    request: &RerankRequest,
    response: RerankResponse,
    request_model_id: String,
) -> Result<RerankResult<String>, LlmError> {
    let original_documents = request
        .documents
        .as_text()
        .ok_or_else(|| {
            LlmError::InvalidParameter(
                "rerank_text_result requires text rerank documents".to_string(),
            )
        })?
        .to_vec();
    let ranking = response
        .results
        .iter()
        .map(|entry| {
            text_document_for_ranking(&original_documents, entry)
                .map(|document| RerankRanking::new(entry.index, entry.relevance_score, document))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let metadata = response_metadata_from_rerank_response(request_model_id, &response);

    Ok(RerankResult::new(original_documents, ranking, metadata))
}

fn project_json_rerank_response(
    request: &RerankRequest,
    response: RerankResponse,
    request_model_id: String,
) -> Result<RerankResult<JSONValue>, LlmError> {
    let original_documents = request_documents_as_json(request);
    let ranking = response
        .results
        .iter()
        .map(|entry| {
            json_document_for_ranking(&original_documents, entry)
                .map(|document| RerankRanking::new(entry.index, entry.relevance_score, document))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let metadata = response_metadata_from_rerank_response(request_model_id, &response);

    Ok(RerankResult::new(original_documents, ranking, metadata))
}

/// Rerank candidates and project the response into an AI SDK-style `RerankResult`.
///
/// This JSON-value projection supports both text and structured document requests. Use
/// `rerank::rerank` when you need the raw Rust-first `RerankResponse`.
pub async fn rerank_result<M: RerankingModel + ?Sized>(
    model: &M,
    request: RerankRequest,
    options: RerankOptions,
) -> Result<RerankResult<JSONValue>, LlmError> {
    let request_model_id = fallback_rerank_model_id(&request, model);
    if request.documents.is_empty() {
        return Ok(RerankResult::new(
            request_documents_as_json(&request),
            Vec::new(),
            RerankResponseMetadata::new(Utc::now(), request_model_id),
        ));
    }

    let response = rerank(model, request.clone(), options).await?;
    project_json_rerank_response(&request, response, request_model_id)
}

/// Rerank text candidates and project the response into an AI SDK-style `RerankResult<String>`.
pub async fn rerank_text_result<M: RerankingModel + ?Sized>(
    model: &M,
    request: RerankRequest,
    options: RerankOptions,
) -> Result<RerankResult<String>, LlmError> {
    let request_model_id = fallback_rerank_model_id(&request, model);
    if request.documents.is_empty() {
        let original_documents = request
            .documents
            .as_text()
            .ok_or_else(|| {
                LlmError::InvalidParameter(
                    "rerank_text_result requires text rerank documents".to_string(),
                )
            })?
            .to_vec();
        return Ok(RerankResult::new(
            original_documents,
            Vec::new(),
            RerankResponseMetadata::new(Utc::now(), request_model_id),
        ));
    }

    let response = rerank(model, request.clone(), options).await?;
    project_text_rerank_response(&request, response, request_model_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::TimeZone;
    use siumai_core::traits::ModelMetadata;
    use siumai_core::types::{HttpResponseInfo, RerankDocument, RerankTokenUsage};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct FakeRerankModel {
        response: RerankResponse,
        calls: Arc<AtomicUsize>,
    }

    impl ModelMetadata for FakeRerankModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-rerank"
        }
    }

    #[async_trait]
    impl RerankingModel for FakeRerankModel {
        async fn rerank(&self, _request: RerankRequest) -> Result<RerankResponse, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.response.clone())
        }
    }

    fn response_envelope() -> HttpResponseInfo {
        HttpResponseInfo {
            timestamp: chrono::Utc
                .with_ymd_and_hms(2026, 1, 2, 3, 4, 5)
                .single()
                .expect("valid timestamp"),
            model_id: Some("transport-rerank".to_string()),
            headers: HashMap::from([("x-request-id".to_string(), "rr_req_1".to_string())]),
            body: Some(serde_json::json!({ "raw": "rerank" })),
        }
    }

    fn fake_response() -> RerankResponse {
        RerankResponse {
            id: "rr_1".to_string(),
            results: vec![
                RerankRankingEntry {
                    document: Some(RerankDocument {
                        text: "second".to_string(),
                    }),
                    index: 1,
                    relevance_score: 0.95,
                },
                RerankRankingEntry {
                    document: Some(RerankDocument {
                        text: "first".to_string(),
                    }),
                    index: 0,
                    relevance_score: 0.4,
                },
            ],
            tokens: RerankTokenUsage {
                input_tokens: 10,
                output_tokens: 0,
            },
            response: Some(response_envelope()),
        }
    }

    #[tokio::test]
    async fn rerank_result_projects_ai_sdk_response_metadata() {
        let calls = Arc::new(AtomicUsize::new(0));
        let model = FakeRerankModel {
            response: fake_response(),
            calls: calls.clone(),
        };

        let request = RerankRequest::new(
            "request-rerank".to_string(),
            "query".to_string(),
            vec!["first".to_string(), "second".to_string()],
        );
        let result = rerank_result(&model, request, RerankOptions::default())
            .await
            .expect("project rerank result");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            result.original_documents,
            vec![serde_json::json!("first"), serde_json::json!("second")]
        );
        assert_eq!(
            result.reranked_documents,
            vec![serde_json::json!("second"), serde_json::json!("first")]
        );
        assert_eq!(result.ranking[0].original_index, 1);
        assert_eq!(result.ranking[0].score, 0.95);
        assert_eq!(result.response.id.as_deref(), Some("rr_1"));
        assert_eq!(result.response.model_id, "transport-rerank");
        assert_eq!(
            result.response.headers.as_ref().expect("headers")["x-request-id"],
            "rr_req_1"
        );
        assert_eq!(
            result.response.body.as_ref().expect("raw body"),
            &serde_json::json!({ "raw": "rerank" })
        );
    }

    #[tokio::test]
    async fn rerank_text_result_short_circuits_empty_documents() {
        let calls = Arc::new(AtomicUsize::new(0));
        let model = FakeRerankModel {
            response: fake_response(),
            calls: calls.clone(),
        };

        let request = RerankRequest::new(
            "request-rerank".to_string(),
            "query".to_string(),
            Vec::new(),
        );
        let result = rerank_text_result(&model, request, RerankOptions::default())
            .await
            .expect("empty rerank result");

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(result.original_documents.is_empty());
        assert!(result.ranking.is_empty());
        assert_eq!(result.response.model_id, "request-rerank");
    }
}
