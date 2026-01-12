//! HTTP error handling and normalization utilities
//!
//! Centralizes common error classification and interceptor notifications
//! to avoid duplicating this logic in individual executors.

use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Classify an HTTP failure using the provider hook if present, otherwise fall back
/// to the generic `retry_api::classify_http_error`.
pub fn classify_http_error(
    provider_id: &str,
    provider_spec: Option<&dyn ProviderSpec>,
    status: u16,
    body_text: &str,
    headers: &HeaderMap,
    fallback_message: Option<&str>,
) -> LlmError {
    if let Some(spec) = provider_spec
        && let Some(e) = spec.classify_http_error(status, body_text, headers)
    {
        return e;
    }
    crate::retry_api::classify_http_error(provider_id, status, body_text, headers, fallback_message)
}

/// Read response body text, classify the error consistently, and notify interceptors.
pub async fn classify_error_with_text(
    provider_id: &str,
    provider_spec: Option<&dyn ProviderSpec>,
    resp: reqwest::Response,
    ctx: &HttpRequestContext,
    interceptors: &[Arc<dyn HttpInterceptor>],
) -> LlmError {
    let status = resp.status();
    let headers = resp.headers().clone();
    let text = resp.text().await.unwrap_or_default();
    let error = classify_http_error(
        provider_id,
        provider_spec,
        status.as_u16(),
        &text,
        &headers,
        status.canonical_reason(),
    );
    for it in interceptors {
        it.on_error(ctx, &error);
    }
    error
}

/// Parse JSON text using unified repair logic and return library error types.
pub fn parse_json_text(text: &str) -> Result<serde_json::Value, LlmError> {
    crate::streaming::parse_json_with_repair(text).map_err(|e| LlmError::ParseError(e.to_string()))
}

/// Read bytes from the response and map to library error types.
pub async fn read_bytes(resp: reqwest::Response) -> Result<Vec<u8>, LlmError> {
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    Ok(bytes.to_vec())
}

/// Parse JSON text; on failure, notify interceptors and return a normalized error.
pub fn parse_json_text_with_ctx(
    provider_id: &str,
    ctx: &HttpRequestContext,
    interceptors: &[Arc<dyn HttpInterceptor>],
    text: &str,
) -> Result<serde_json::Value, LlmError> {
    match parse_json_text(text) {
        Ok(v) => Ok(v),
        Err(e) => {
            for it in interceptors {
                it.on_error(ctx, &e);
            }
            // Wrap as an API error with provider context in the details
            let details = serde_json::json!({
                "provider": provider_id,
                "raw": text,
            });
            Err(LlmError::api_error_with_details(
                200,
                "parse error",
                details,
            ))
        }
    }
}

/// Read bytes; on failure, notify interceptors and return a normalized error.
pub async fn read_bytes_with_ctx(
    provider_id: &str,
    ctx: &HttpRequestContext,
    interceptors: &[Arc<dyn HttpInterceptor>],
    resp: reqwest::Response,
) -> Result<Vec<u8>, LlmError> {
    match resp.bytes().await {
        Ok(b) => Ok(b.to_vec()),
        Err(err) => {
            let e = LlmError::HttpError(format!("{}", err));
            for it in interceptors {
                it.on_error(ctx, &e);
            }
            let details = serde_json::json!({
                "provider": provider_id,
                "context": "read_bytes",
            });
            Err(LlmError::api_error_with_details(
                0,
                "read bytes error",
                details,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::HttpInterceptor;
    use std::sync::{Arc, Mutex};

    struct FlagInterceptor(Arc<Mutex<bool>>);
    impl HttpInterceptor for FlagInterceptor {
        fn on_error(&self, _ctx: &HttpRequestContext, _error: &LlmError) {
            *self.0.lock().unwrap() = true;
        }
    }

    #[tokio::test]
    async fn classify_triggers_interceptor_on_error() {
        // Use mockito 1.7 Server API (async)
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/err")
            .with_status(400)
            .with_body("bad request")
            .create_async()
            .await;

        let url = format!("{}/err", server.url());
        let client = reqwest::Client::new();
        let resp = client.get(url).send().await.expect("send resp");

        let ctx = HttpRequestContext {
            request_id: crate::execution::http::interceptor::generate_request_id(),
            provider_id: "test".into(),
            url: "http://test".into(),
            stream: false,
        };
        let flag = Arc::new(Mutex::new(false));
        let it: Arc<dyn HttpInterceptor> = Arc::new(FlagInterceptor(flag.clone()));
        let err = super::classify_error_with_text("test", None, resp, &ctx, &[it]).await;

        // Should be classified as one of the library's error variants (API/RateLimit/Authentication/etc.)
        match err {
            LlmError::ApiError { .. }
            | LlmError::InvalidInput(_)
            | LlmError::RateLimitError(_)
            | LlmError::AuthenticationError(_)
            | LlmError::ProviderError { .. }
            | LlmError::HttpError(_) => {}
            other => panic!("unexpected error variant: {:?}", other),
        }
        assert!(*flag.lock().unwrap(), "interceptor not triggered");
    }
}
