use axum::{body::Body, extract::Request, response::Response};
use serde::Deserialize;
use serde_json::Value;
use siumai::prelude::unified::{ChatRequest, LlmError};
use siumai_extras::server::{GatewayBridgePolicy, axum::read_request_body_with_policy};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GatewayQuery {
    /// Optional fallback prompt when the request body is empty.
    pub prompt: Option<String>,
}

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
