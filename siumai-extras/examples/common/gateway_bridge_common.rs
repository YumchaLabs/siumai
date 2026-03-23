//! Shared helpers for explicit request-normalization gateway examples.
//!
//! These helpers intentionally implement the route shape documented in
//! `docs/workstreams/protocol-bridge-gateway/route-recipes.md` Recipe 1:
//!
//! 1. read the downstream body under `GatewayBridgePolicy`
//! 2. normalize provider-native request JSON into `ChatRequest`
//! 3. pin the normalized request to one concrete backend model handle
//! 4. let the route execute the unified request and transcode the result

use axum::{body::Body, extract::Request, response::Response};
use serde::Deserialize;
use serde_json::Value;
use siumai::experimental::bridge::BridgeReport;
use siumai::prelude::unified::{ChatRequest, LlmError};
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        NormalizeRequestOptions, SourceRequestFormat, normalize_request_json_with_options,
        read_request_body_with_policy,
    },
};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GatewayQuery {
    /// Optional fallback prompt when the request body is empty.
    pub prompt: Option<String>,
}

/// Convert a unified execution failure into a simple example response.
pub fn internal_error_response(error: &LlmError) -> Response {
    let body = format!("gateway execution failed: {}", error.user_message());
    Response::builder()
        .status(500)
        .header("content-type", "text/plain; charset=utf-8")
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::from("internal error")))
}

pub fn bad_request_response(message: impl Into<String>) -> Response {
    Response::builder()
        .status(400)
        .header("content-type", "text/plain; charset=utf-8")
        .body(Body::from(message.into()))
        .unwrap_or_else(|_| Response::new(Body::from("bad request")))
}

/// Convert a rejected request-normalization report into a readable example response.
pub fn rejected_normalize_response(source_name: &str, report: &BridgeReport) -> Response {
    let detail = report
        .warnings
        .first()
        .map(|warning| warning.message.as_str())
        .unwrap_or("bridge policy rejected source request normalization");
    bad_request_response(format!(
        "{source_name} request normalization was rejected: {detail}"
    ))
}

/// Read a downstream request body under policy, falling back to a prompt-derived request body
/// when the body is empty.
pub async fn read_source_request_json_or_prompt<F>(
    request: Request,
    policy: &GatewayBridgePolicy,
    prompt: String,
    fallback: F,
) -> Result<Value, Response>
where
    F: FnOnce(String) -> Value,
{
    let bytes = read_request_body_with_policy(request.into_body(), policy)
        .await
        .map_err(|error| error.to_response(policy))?;

    if bytes.iter().all(|byte| byte.is_ascii_whitespace()) {
        return Ok(fallback(prompt));
    }

    serde_json::from_slice(&bytes).map_err(|error| {
        let message = if policy.passthrough_runtime_errors {
            format!("invalid downstream request json: {error}")
        } else {
            "invalid downstream request json".to_string()
        };
        bad_request_response(message)
    })
}

/// Normalize provider-native request JSON into a backend-pinned unified `ChatRequest`.
pub fn normalize_source_request_for_backend(
    body: &Value,
    source: SourceRequestFormat,
    source_name: &str,
    backend_model_id: &str,
    stream: bool,
    policy: &GatewayBridgePolicy,
) -> Result<ChatRequest, Response> {
    let bridged = normalize_request_json_with_options(
        body,
        source,
        &NormalizeRequestOptions::default().with_policy(policy.clone()),
    )
    .map_err(|error| {
        bad_request_response(format!(
            "failed to normalize {source_name} request: {}",
            error.user_message()
        ))
    })?;
    let (request, report) = bridged
        .into_result()
        .map_err(|report| rejected_normalize_response(source_name, &report))?;

    let _ = report;
    Ok(pin_request_to_backend_model(
        request,
        backend_model_id,
        stream,
    ))
}

/// Pin a normalized request to one concrete backend model handle used by the example route.
pub fn pin_request_to_backend_model(
    mut request: ChatRequest,
    backend_model_id: &str,
    stream: bool,
) -> ChatRequest {
    // Request normalizers preserve the downstream model id, but the example
    // executes against one concrete upstream model handle.
    request.common_params.model = backend_model_id.to_string();
    request.stream = stream;
    request
}
