//! Executor helper utilities
//!
//! Extract small, common utilities so that `common.rs` remains lean:
//! - Safely clone request headers (visible to interceptors)
//! - Apply before-send interceptors
//! - Single 401 retry (JSON and multipart)

use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::execution::http::transport::{HttpTransportStreamBody, HttpTransportStreamResponse};
use futures_util::StreamExt;
use reqwest::ResponseBuilderExt;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::Arc;

// Build-safe header extraction from a `RequestBuilder`.
//
// Purpose
// - Attempts to build the inner `Request` and clone its headers to provide
//   interceptors visibility into the real outgoing headers.
// - If cloning/building fails (e.g., due to non-cloneable body), it falls back
//   to the provided `fallback` header map to ensure a consistent call.
//
// Notes
// - This function never mutates the builder.
// - Caller is responsible for choosing an appropriate `fallback` map
//   (usually the effective/merged headers for this request).
pub(crate) fn headers_from_builder(
    rb: &reqwest::RequestBuilder,
    fallback: &HeaderMap,
) -> HeaderMap {
    rb.try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_else(|| fallback.clone())
}

// Apply `on_before_send` interceptors with a consistent header-clone strategy.
//
// Behavior
// - Clones headers via `headers_from_builder` to give interceptors accurate
//   visibility of outgoing headers (or a safe fallback).
// - Applies interceptors in-order, returning a mutated `RequestBuilder`.
//
// Errors
// - Propagates any error returned by an interceptor.
pub(crate) fn apply_before_send_interceptors(
    interceptors: &[Arc<dyn HttpInterceptor>],
    ctx: &HttpRequestContext,
    rb: reqwest::RequestBuilder,
    body: &serde_json::Value,
    effective_headers: &HeaderMap,
) -> Result<reqwest::RequestBuilder, LlmError> {
    let cloned = headers_from_builder(&rb, effective_headers);
    let mut out = rb;
    for it in interceptors {
        out = it.on_before_send(ctx, out, body, &cloned)?;
    }
    Ok(out)
}

// Materialize a reqwest multipart form into raw bytes for custom transports.
//
// Returns
// - The fully encoded multipart body bytes.
// - The matching `Content-Type` header value with boundary.
//
// Errors
// - Multipart stream errors are surfaced as `LlmError::HttpError`.
pub(crate) async fn build_multipart_body(
    form: reqwest::multipart::Form,
) -> Result<(Vec<u8>, String), LlmError> {
    let boundary = form.boundary().to_string();
    let mut body = Vec::new();
    let mut stream = form.into_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| LlmError::HttpError(e.to_string()))?;
        body.extend_from_slice(&chunk);
    }

    Ok((body, format!("multipart/form-data; boundary={boundary}")))
}

// Inject the final multipart content headers into an outgoing header map.
//
// Notes
// - Existing values are replaced because multipart content metadata must match
//   the encoded body bytes that will be sent through the transport.
pub(crate) fn with_multipart_content_headers(
    mut headers: HeaderMap,
    content_type: &str,
    content_length: usize,
) -> Result<HeaderMap, LlmError> {
    let content_type =
        HeaderValue::from_str(content_type).map_err(|e| LlmError::HttpError(e.to_string()))?;
    let content_length = HeaderValue::from_str(&content_length.to_string())
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    headers.insert(CONTENT_TYPE, content_type);
    headers.insert(CONTENT_LENGTH, content_length);
    Ok(headers)
}

// Convert a custom transport streaming response into a `reqwest::Response`.
//
// Notes
// - This is intentionally used only as an adapter for higher-level helpers
//   that already expect a raw `reqwest::Response` surface (`headers()`, `text()`,
//   `bytes_stream()`, etc.).
// - Interceptor `on_response` hooks remain skipped on the custom-transport path.
pub(crate) fn response_from_stream_transport(
    url: &str,
    response: HttpTransportStreamResponse,
) -> Result<reqwest::Response, LlmError> {
    let parsed_url = reqwest::Url::parse(url).map_err(|e| {
        LlmError::InvalidParameter(format!("Invalid URL '{url}' for transport response: {e}"))
    })?;

    let mut builder = http::Response::builder()
        .status(response.status)
        .url(parsed_url);
    for (name, value) in &response.headers {
        builder = builder.header(name, value);
    }

    let body = match response.body {
        HttpTransportStreamBody::Full(bytes) => reqwest::Body::from(bytes),
        HttpTransportStreamBody::Stream(stream) => reqwest::Body::wrap_stream(stream),
    };

    let response = builder
        .body(body)
        .map_err(|e| LlmError::HttpError(format!("Failed to build transport response: {e}")))?;

    Ok(reqwest::Response::from(response))
}

// Retry once with rebuilt headers using a caller-provided builder closure.
//
// Behavior
// - Builds a new `RequestBuilder` using `build_with_headers` with the provided
//   `effective_headers`.
// - Re-applies `on_before_send` interceptors (headers are cloned for visibility).
// - Sends the request and returns the response.
//
// Expectations
// - The retry is intended for idempotent requests (most LLM HTTP calls are).
// - The original request is expected to have failed with 401 and caller has
//   already decided to retry once.
//
// Errors
// - Network errors or interceptor errors are propagated as `LlmError`.
pub(crate) async fn rebuild_headers_and_retry_once<F>(
    build_with_headers: F,
    interceptors: &[Arc<dyn HttpInterceptor>],
    ctx: &HttpRequestContext,
    body_for_interceptors: &serde_json::Value,
    effective_headers: HeaderMap,
) -> Result<reqwest::Response, LlmError>
where
    F: FnOnce(HeaderMap) -> reqwest::RequestBuilder,
{
    let mut rb_retry = build_with_headers(effective_headers.clone());
    #[cfg(test)]
    {
        rb_retry = rb_retry.header("x-retry-attempt", "1");
    }
    rb_retry = apply_before_send_interceptors(
        interceptors,
        ctx,
        rb_retry,
        body_for_interceptors,
        &effective_headers,
    )?;
    let resp = rb_retry
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    Ok(resp)
}

// Retry once for multipart requests by rebuilding the form and applying interceptors.
//
// Behavior
// - Rebuilds the multipart form by invoking `build_form` again (multipart bodies
//   are not cloneable), attaches `effective_headers`, re-applies interceptors and sends.
//
// Expectations
// - Caller has already decided to retry (e.g., after a 401) and prepared the
//   correct `effective_headers` (including any per-request header merging).
// - The form builder must be side-effect free beyond constructing a new form.
//
// Errors
// - Form builder errors or network/interceptor errors are propagated as `LlmError`.
pub(crate) async fn rebuild_headers_and_retry_once_multipart<F>(
    http_client: &reqwest::Client,
    url: &str,
    interceptors: &[Arc<dyn HttpInterceptor>],
    ctx: &HttpRequestContext,
    effective_headers: HeaderMap,
    build_form: F,
    per_request_timeout: Option<std::time::Duration>,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    let form = build_form()?;
    let mut rb_retry = http_client
        .post(url)
        .headers(effective_headers.clone())
        .multipart(form);
    if let Some(timeout) = per_request_timeout {
        rb_retry = rb_retry.timeout(timeout);
    }
    #[cfg(test)]
    {
        rb_retry = rb_retry.header("x-retry-attempt", "1");
    }
    let empty_json = serde_json::json!({});
    rb_retry = apply_before_send_interceptors(
        interceptors,
        ctx,
        rb_retry,
        &empty_json,
        &effective_headers,
    )?;
    let resp = rb_retry
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    Ok(resp)
}
