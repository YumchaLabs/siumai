//! Common HTTP Execution Layer
//!
//! This module provides unified HTTP request/response handling for all executors,
//! eliminating code duplication across chat/embedding/image/files executors.
//!
//! Key features:
//! - Unified HTTP request sending with interceptors
//! - Automatic 401 retry with header rebuild
//! - Unified error classification
//! - JSON parsing with automatic repair
//! - Per-request header merging
//! - Tracing headers injection
//! - Telemetry integration
//!
//! Retry Helpers
//! - `rebuild_headers_and_retry_once` re-creates a RequestBuilder with rebuilt/effective headers
//!   and re-applies `on_before_send` interceptors before a single retry attempt.
//! - `rebuild_headers_and_retry_once_multipart` does the same for multipart forms by rebuilding
//!   the form (multipart bodies are not cloneable), then re-applies interceptors and retries once.
//! - The helpers assume requests are idempotent (typical for LLM HTTP calls). Callers decide when
//!   to retry (e.g. only on 401) and prepare the correct effective headers.

use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::retry_api::RetryOptions;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Build-safe header extraction from a `RequestBuilder`.
///
/// Purpose
/// - Attempts to build the inner `Request` and clone its headers to provide
///   interceptors visibility into the real outgoing headers.
/// - If cloning/building fails (e.g., due to non-cloneable body), it falls back
///   to the provided `fallback` header map to ensure a consistent call.
///
/// Notes
/// - This function never mutates the builder.
/// - Caller is responsible for choosing an appropriate `fallback` map
///   (usually the effective/merged headers for this request).
fn headers_from_builder(rb: &reqwest::RequestBuilder, fallback: &HeaderMap) -> HeaderMap {
    rb.try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_else(|| fallback.clone())
}

/// Apply `on_before_send` interceptors with a consistent header-clone strategy.
///
/// Behavior
/// - Clones headers via `headers_from_builder` to give interceptors accurate
///   visibility of outgoing headers (or a safe fallback).
/// - Applies interceptors in-order, returning a mutated `RequestBuilder`.
///
/// Errors
/// - Propagates any error returned by an interceptor.
fn apply_before_send_interceptors(
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

/// Retry once with rebuilt headers using a caller-provided builder closure.
///
/// Behavior
/// - Builds a new `RequestBuilder` using `build_with_headers` with the provided
///   `effective_headers`.
/// - Re-applies `on_before_send` interceptors (headers are cloned for visibility).
/// - Sends the request and returns the response.
///
/// Expectations
/// - The retry is intended for idempotent requests (most LLM HTTP calls are).
/// - The original request is expected to have failed with 401 and caller has
///   already decided to retry once.
///
/// Errors
/// - Network errors or interceptor errors are propagated as `LlmError`.
async fn rebuild_headers_and_retry_once<F>(
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

/// Retry once for multipart requests by rebuilding the form and applying interceptors.
///
/// Behavior
/// - Rebuilds the multipart form by invoking `build_form` again (multipart bodies
///   are not cloneable), attaches `effective_headers`, re-applies interceptors and sends.
///
/// Expectations
/// - Caller has already decided to retry (e.g., after a 401) and prepared the
///   correct `effective_headers` (including any per-request header merging).
/// - The form builder must be side-effect free beyond constructing a new form.
///
/// Errors
/// - Form builder errors or network/interceptor errors are propagated as `LlmError`.
async fn rebuild_headers_and_retry_once_multipart<F>(
    http_client: &reqwest::Client,
    url: &str,
    interceptors: &[Arc<dyn HttpInterceptor>],
    ctx: &HttpRequestContext,
    effective_headers: HeaderMap,
    build_form: F,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    let form = build_form()?;
    let mut rb_retry = http_client
        .post(url)
        .headers(effective_headers.clone())
        .multipart(form);
    #[cfg(test)]
    {
        rb_retry = rb_retry.header("x-retry-attempt", "1");
    }
    // Use empty JSON body for interceptor visibility (multipart has no JSON body)
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

/// HTTP request body type
#[derive(Debug)]
pub enum HttpBody {
    /// JSON body
    Json(serde_json::Value),
    /// Multipart form body
    Multipart(reqwest::multipart::Form),
}

/// Configuration for HTTP request execution
#[derive(Clone)]
pub struct HttpExecutionConfig {
    /// Provider ID for logging and telemetry
    pub provider_id: String,
    /// HTTP client
    pub http_client: reqwest::Client,
    /// Provider spec for header building
    pub provider_spec: Arc<dyn ProviderSpec>,
    /// Provider context
    pub provider_context: ProviderContext,
    /// HTTP interceptors (order preserved)
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Retry options
    pub retry_options: Option<RetryOptions>,
}

/// Result of HTTP request execution
pub struct HttpExecutionResult {
    /// Response body as JSON
    pub json: serde_json::Value,
    /// Response status code
    pub status: u16,
    /// Response headers
    pub headers: HeaderMap,
}

/// Result for byte-response requests (e.g., TTS audio bytes)
pub struct HttpBytesResult {
    /// Raw response bytes
    pub bytes: Vec<u8>,
    /// Response status
    pub status: u16,
    /// Response headers
    pub headers: HeaderMap,
}

/// Execute an SSE streaming request with explicit base headers (no ProviderSpec).
/// Returns a ChatStream that converts SSE events via the provided converter.
#[allow(clippy::too_many_arguments)]
pub async fn execute_sse_stream_request_with_headers<C>(
    http_client: &reqwest::Client,
    provider_id: &str,
    url: &str,
    headers_base: HeaderMap,
    body: serde_json::Value,
    interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: Option<RetryOptions>,
    per_request_headers: Option<std::collections::HashMap<String, String>>,
    converter: C,
    disable_compression: bool,
) -> Result<crate::streaming::ChatStream, LlmError>
where
    C: crate::streaming::SseEventConverter + Clone + Send + Sync + 'static,
{
    // Build closure to construct the request (used by retry wrapper as well)
    let build_request = {
        let http = http_client.clone();
        let base = headers_base.clone();
        let url_owned = url.to_string();
        let body_owned = body.clone();
        let interceptors = interceptors.to_vec();
        move || -> Result<reqwest::RequestBuilder, LlmError> {
            let effective_headers = if let Some(req_headers) = per_request_headers.clone() {
                crate::execution::http::headers::merge_headers(base.clone(), &req_headers)
            } else {
                base.clone()
            };
            let mut rb = http
                .post(url_owned.clone())
                .headers(effective_headers.clone())
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .header(reqwest::header::CACHE_CONTROL, "no-cache")
                .header(reqwest::header::CONNECTION, "keep-alive")
                .json(&body_owned);
            if disable_compression {
                rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
            }
            // Apply interceptors (before send)
            let ctx = crate::execution::http::interceptor::HttpRequestContext {
                provider_id: provider_id.to_string(),
                url: url_owned.clone(),
                stream: true,
            };
            let cloned_headers = headers_from_builder(&rb, &effective_headers);
            let mut out_rb = rb;
            for it in &interceptors {
                out_rb = it.on_before_send(&ctx, out_rb, &body_owned, &cloned_headers)?;
            }
            Ok(out_rb)
        }
    };

    let should_retry_401 = retry_options
        .as_ref()
        .map(|opts| opts.retry_401)
        .unwrap_or(true);

    crate::streaming::StreamFactory::create_eventsource_stream_with_retry(
        provider_id,
        url,
        should_retry_401,
        build_request,
        converter,
        interceptors,
    )
    .await
}

/// Execute a JSON streaming request with explicit base headers (no ProviderSpec).
/// Returns a ChatStream that converts line-delimited JSON via the provided converter.
#[allow(clippy::too_many_arguments)]
pub async fn execute_json_stream_request_with_headers<C>(
    http_client: &reqwest::Client,
    provider_id: &str,
    url: &str,
    headers_base: HeaderMap,
    body: serde_json::Value,
    interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: Option<RetryOptions>,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    json_converter: C,
    disable_compression: bool,
) -> Result<crate::streaming::ChatStream, LlmError>
where
    C: crate::streaming::JsonEventConverter + Clone + 'static,
{
    // Merge per-request headers
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers_base.clone(), req_headers)
    } else {
        headers_base.clone()
    };

    // Build request
    let mut rb = http_client
        .post(url)
        .headers(effective_headers.clone())
        .json(&body);
    if disable_compression {
        rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
    }

    // Interceptors (before send)
    let ctx = HttpRequestContext {
        provider_id: provider_id.to_string(),
        url: url.to_string(),
        stream: true,
    };
    rb = apply_before_send_interceptors(interceptors, &ctx, rb, &body, &effective_headers)?;

    // Send
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 401 retry once
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);
        if status.as_u16() == 401 && should_retry_401 {
            for interceptor in interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }
            // Retry with rebuilt headers (no ProviderSpec here; reuse effective headers)
            let builder = |headers: HeaderMap| {
                let mut b = http_client.post(url).headers(headers).json(&body);
                if disable_compression {
                    b = b.header(reqwest::header::ACCEPT_ENCODING, "identity");
                }
                b
            };
            resp = rebuild_headers_and_retry_once(
                builder,
                interceptors,
                &ctx,
                &body,
                effective_headers.clone(),
            )
            .await?;
        }
    }

    // Error classification
    if !resp.status().is_success() {
        let status = resp.status();
        let response_headers = resp.headers().clone();
        let error_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        let error = crate::retry_api::classify_http_error(
            provider_id,
            status.as_u16(),
            &error_text,
            &response_headers,
            None,
        );
        for interceptor in interceptors {
            interceptor.on_error(&ctx, &error);
        }
        return Err(error);
    }

    for interceptor in interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }

    // Convert response into ChatStream via JSON converter
    crate::streaming::StreamFactory::create_json_stream(resp, json_converter).await
}

/// Execute a request that returns bytes using ProviderSpec (JSON only).
/// For multipart bytes request, prefer a specialized path with a form builder.
pub async fn execute_bytes_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpBytesResult, LlmError> {
    // 1. Build base headers
    let base_headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(base_headers.clone(), req_headers)
    } else {
        base_headers.clone()
    };

    // 3. Build request (JSON only)
    let mut rb = config
        .http_client
        .post(url)
        .headers(effective_headers.clone());
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }
    let json_body = match &body {
        HttpBody::Json(json) => {
            rb = rb.json(json);
            json.clone()
        }
        HttpBody::Multipart(_) => {
            return Err(LlmError::InvalidParameter(
                "execute_bytes_request does not support multipart bodies".into(),
            ));
        }
    };

    // 4. Interceptors
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };
    rb = apply_before_send_interceptors(
        &config.interceptors,
        &ctx,
        rb,
        &json_body,
        &effective_headers,
    )?;

    // 5. Send
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 6. 401 retry once
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);
        if status.as_u16() == 401 && should_retry_401 {
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;
            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };
            let builder = |headers: HeaderMap| {
                config
                    .http_client
                    .post(url)
                    .headers(headers)
                    .json(&json_body)
            };
            resp = rebuild_headers_and_retry_once(
                builder,
                &config.interceptors,
                &ctx,
                &json_body,
                retry_effective_headers.clone(),
            )
            .await?;
        }
    }

    // 7. Error classification
    if !resp.status().is_success() {
        let status = resp.status();
        let response_headers = resp.headers().clone();
        let error_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        let error = crate::retry_api::classify_http_error(
            &config.provider_id,
            status.as_u16(),
            &error_text,
            &response_headers,
            None,
        );
        for interceptor in &config.interceptors {
            interceptor.on_error(&ctx, &error);
        }
        return Err(error);
    }

    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    Ok(HttpBytesResult {
        bytes: bytes.to_vec(),
        status: status_code,
        headers: response_headers,
    })
}

/// Execute a multipart HTTP request that returns binary content (bytes)
///
/// Mirrors `execute_multipart_request` but returns raw bytes instead of JSON.
/// Multipart forms cannot be cloned; the `build_form` function will be called
/// for the initial request and again for a retry (if applicable).
pub async fn execute_multipart_bytes_request<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpBytesResult, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    // 1. Build base headers from provider spec
    let base_headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(base_headers.clone(), req_headers)
    } else {
        base_headers.clone()
    };

    // 3. Build form and request
    let form = build_form()?;
    let mut rb = config
        .http_client
        .post(url)
        .headers(effective_headers.clone())
        .multipart(form);
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // Apply interceptors (use empty JSON body for visibility)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };
    let empty_json = serde_json::json!({});
    let cloned_headers = headers_from_builder(&rb, &effective_headers);
    for interceptor in &config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &empty_json, &cloned_headers)?;
    }

    // 4. Send request
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 5. Handle 401 retry once
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);
        if status.as_u16() == 401 && should_retry_401 {
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }
            // Rebuild headers and form
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;
            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };
            let retry_form = build_form()?;
            let mut rb_retry = config
                .http_client
                .post(url)
                .headers(retry_effective_headers.clone())
                .multipart(retry_form);
            #[cfg(test)]
            {
                rb_retry = rb_retry.header("x-retry-attempt", "1");
            }
            let cloned_headers_retry = headers_from_builder(&rb_retry, &retry_effective_headers);
            for interceptor in &config.interceptors {
                rb_retry = interceptor.on_before_send(
                    &ctx,
                    rb_retry,
                    &empty_json,
                    &cloned_headers_retry,
                )?;
            }
            resp = rb_retry
                .send()
                .await
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
        }
    }

    // 6. Classify error if still not successful
    if !resp.status().is_success() {
        let status = resp.status();
        let response_headers = resp.headers().clone();
        let error_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        let error = crate::retry_api::classify_http_error(
            &config.provider_id,
            status.as_u16(),
            &error_text,
            &response_headers,
            None,
        );
        for interceptor in &config.interceptors {
            interceptor.on_error(&ctx, &error);
        }
        return Err(error);
    }

    // 7. Notify interceptors and return bytes
    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    Ok(HttpBytesResult {
        bytes: bytes.to_vec(),
        status: status_code,
        headers: response_headers,
    })
}

/// Execute a JSON HTTP request using explicit base headers (no ProviderSpec).
///
/// This helper is useful for code paths that already have a fully constructed
/// header map and do not rely on ProviderSpec routing or header building.
#[allow(clippy::too_many_arguments)]
pub async fn execute_json_request_with_headers(
    http_client: &reqwest::Client,
    provider_id: &str,
    url: &str,
    headers_base: HeaderMap,
    body: serde_json::Value,
    interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: Option<RetryOptions>,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    stream: bool,
) -> Result<HttpExecutionResult, LlmError> {
    // Merge per-request headers
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers_base.clone(), req_headers)
    } else {
        headers_base.clone()
    };

    // Build request
    let mut rb = http_client
        .post(url)
        .headers(effective_headers.clone())
        .json(&body);
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    let ctx = HttpRequestContext {
        provider_id: provider_id.to_string(),
        url: url.to_string(),
        stream,
    };

    // Interceptors (before send)
    rb = apply_before_send_interceptors(interceptors, &ctx, rb, &body, &effective_headers)?;

    // Send
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 401 retry (once)
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);
        if status.as_u16() == 401 && should_retry_401 {
            // Notify interceptors of retry
            for interceptor in interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }
            // Rebuild headers
            let retry_headers = headers_base.clone();
            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };
            let builder = |headers: HeaderMap| http_client.post(url).headers(headers).json(&body);
            resp = rebuild_headers_and_retry_once(
                builder,
                interceptors,
                &ctx,
                &body,
                retry_effective_headers.clone(),
            )
            .await?;
        }
    }

    // Error classification
    if !resp.status().is_success() {
        let status = resp.status();
        let response_headers = resp.headers().clone();
        let error_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        let error = crate::retry_api::classify_http_error(
            provider_id,
            status.as_u16(),
            &error_text,
            &response_headers,
            None,
        );
        for interceptor in interceptors {
            interceptor.on_error(&ctx, &error);
        }
        return Err(error);
    }

    // Success path
    for interceptor in interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }

    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();
    let text = resp
        .text()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
        .map_err(|e| LlmError::ParseError(e.to_string()))?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// Execute a JSON HTTP request with unified retry, interceptors, and error handling
///
/// This function provides the complete HTTP execution pipeline:
/// 1. Build base headers from provider spec
/// 2. Inject tracing headers
/// 3. Merge per-request headers (if provided)
/// 4. Apply HTTP interceptors
/// 5. Send request
/// 6. Handle 401 retry with header rebuild
/// 7. Classify errors
/// 8. Parse JSON response with automatic repair
///
/// # Arguments
///
/// * `config` - HTTP execution configuration
/// * `url` - Request URL
/// * `body` - Request body (JSON or Multipart)
/// * `per_request_headers` - Optional per-request headers to merge
/// * `stream` - Whether this is a streaming request (for context)
///
/// # Returns
///
/// Returns `HttpExecutionResult` containing parsed JSON, status, and headers
pub async fn execute_json_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    stream: bool,
) -> Result<HttpExecutionResult, LlmError> {
    execute_request(config, url, body, per_request_headers, stream).await
}

// unit tests migrated to integration tests in tests/http_common_retry_401.rs

/// Execute an HTTP request (JSON or Multipart) with unified retry, interceptors, and error handling
pub async fn execute_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    stream: bool,
) -> Result<HttpExecutionResult, LlmError> {
    // 1. Build base headers from provider spec
    let base_headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(base_headers.clone(), req_headers)
    } else {
        base_headers.clone()
    };

    // 4. Build request and apply interceptors
    let mut rb = config
        .http_client
        .post(url)
        .headers(effective_headers.clone());

    // Apply body to request builder
    let json_body = match &body {
        HttpBody::Json(json) => {
            rb = rb.json(json);
            json.clone()
        }
        HttpBody::Multipart(_) => {
            return Err(LlmError::InvalidParameter(
                "Use execute_multipart_request for multipart bodies".into(),
            ));
        }
    };

    // Apply interceptors
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream,
    };

    // Apply before-send interceptors
    rb = apply_before_send_interceptors(
        &config.interceptors,
        &ctx,
        rb,
        &json_body,
        &effective_headers,
    )?;

    // 5. Send request

    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 6. Handle 401 retry with header rebuild
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        if status.as_u16() == 401 && should_retry_401 {
            // Notify interceptors of retry
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            // Rebuild headers and retry once (unified helper)
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;
            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };
            let builder = |headers: HeaderMap| {
                let json = match &body {
                    HttpBody::Json(j) => j,
                    HttpBody::Multipart(_) => unreachable!("multipart not used in JSON retry"),
                };
                config.http_client.post(url).headers(headers).json(json)
            };
            resp = rebuild_headers_and_retry_once(
                builder,
                &config.interceptors,
                &ctx,
                &json_body,
                retry_effective_headers.clone(),
            )
            .await?;
        }
    }

    // 7. Classify errors if still not successful
    if !resp.status().is_success() {
        let status = resp.status();
        let response_headers = resp.headers().clone();
        let error_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());

        // Notify interceptors of error
        let error = crate::retry_api::classify_http_error(
            &config.provider_id,
            status.as_u16(),
            &error_text,
            &response_headers,
            None,
        );
        for interceptor in &config.interceptors {
            interceptor.on_error(&ctx, &error);
        }

        return Err(error);
    }

    // Notify interceptors of successful response
    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }

    // Store status and headers before consuming response
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();

    // 8. Parse JSON response with automatic repair
    let text = resp
        .text()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
        .map_err(|e| LlmError::ParseError(e.to_string()))?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// Execute a multipart HTTP request with unified retry and error handling
///
/// Similar to `execute_json_request` but handles multipart form data.
/// Note: Multipart forms cannot be cloned, so retry requires rebuilding the form.
///
/// # Arguments
///
/// * `config` - HTTP execution configuration
/// * `url` - Request URL
/// * `build_form` - Function to build the multipart form (called for initial request and retry)
/// * `per_request_headers` - Optional per-request headers to merge
///
/// # Returns
///
/// Returns `HttpExecutionResult` containing parsed JSON, status, and headers
pub async fn execute_multipart_request<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    // 1. Build base headers from provider spec
    let base_headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(base_headers.clone(), req_headers)
    } else {
        base_headers.clone()
    };

    // 4. Build form and request
    let form = build_form()?;
    let mut rb = config
        .http_client
        .post(url)
        .headers(effective_headers.clone())
        .multipart(form);

    // Apply interceptors (with empty JSON body for multipart)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };

    let empty_json = serde_json::json!({});
    let cloned_headers = rb
        .try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_else(|| effective_headers.clone());

    for interceptor in &config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &empty_json, &cloned_headers)?;
    }

    // 5. Send request

    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 6. Handle 401 retry
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        if status.as_u16() == 401 && should_retry_401 {
            // Notify interceptors
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            // Rebuild everything for retry
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;

            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };

            resp = rebuild_headers_and_retry_once_multipart(
                &config.http_client,
                url,
                &config.interceptors,
                &ctx,
                retry_effective_headers,
                build_form,
            )
            .await?;
        }
    }

    // 7. Classify errors
    if !resp.status().is_success() {
        let status = resp.status();
        let response_headers = resp.headers().clone();
        let error_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());

        let error = crate::retry_api::classify_http_error(
            &config.provider_id,
            status.as_u16(),
            &error_text,
            &response_headers,
            None,
        );
        for interceptor in &config.interceptors {
            interceptor.on_error(&ctx, &error);
        }

        return Err(error);
    }

    // Notify interceptors
    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }

    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();

    // 8. Parse JSON
    let text = resp
        .text()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
        .map_err(|e| LlmError::ParseError(e.to_string()))?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// Execute a GET request with unified HTTP handling.
///
/// Summary
/// - Builds ProviderSpec headers, merges per-request headers, applies interceptors, sends GET,
///   and retries once on 401 with rebuilt headers.
///
/// Arguments
/// - `config`: HTTP execution configuration (ProviderSpec + context)
/// - `url`: Request URL
/// - `per_request_headers`: Optional per-request header overrides
///
/// Returns
/// - `Ok(HttpExecutionResult)` with parsed JSON, status and headers
/// - `Err(LlmError)` on network/HTTP/interceptor/parse errors
///
/// Example
/// ```ignore
/// use siumai::execution::executors::common::{HttpExecutionConfig, execute_get_request};
/// use siumai::core::{ProviderContext, ProviderSpec};
/// use std::sync::Arc;
///
/// // Minimal ProviderSpec for example (builds static headers and URL routing)
/// #[derive(Clone)]
/// struct ExampleSpec;
/// impl ProviderSpec for ExampleSpec {
///   fn id(&self) -> &'static str { "example" }
///   fn capabilities(&self) -> siumai::traits::ProviderCapabilities { Default::default() }
///   fn build_headers(&self, _ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, siumai::LlmError> {
///     Ok(reqwest::header::HeaderMap::new())
///   }
///   fn chat_url(&self, _stream: bool, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext) -> String { String::new() }
///   fn choose_chat_transformers(&self, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext)
///     -> siumai::core::ChatTransformers { unimplemented!() }
/// }
///
/// # async fn demo() -> Result<(), siumai::LlmError> {
/// let http = reqwest::Client::new();
/// let spec = Arc::new(ExampleSpec);
/// let ctx = ProviderContext::new("example", "https://api.example.com", None, Default::default());
/// let config = HttpExecutionConfig {
///   provider_id: "example".into(),
///   http_client: http,
///   provider_spec: spec,
///   provider_context: ctx,
///   interceptors: vec![],
///   retry_options: Some(siumai::retry_api::RetryOptions::default()),
/// };
/// let res = execute_get_request(&config, "https://api.example.com/ping", None).await?;
/// # Ok(()) }
/// ```
pub async fn execute_get_request(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    // 1. Build base headers from ProviderSpec
    let headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers, req_headers)
    } else {
        headers
    };

    // 4. Create request builder
    let mut rb = config
        .http_client
        .get(url)
        .headers(effective_headers.clone());
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // 5. Apply HTTP interceptors (on_before_send)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };

    let empty_json = serde_json::json!({});
    rb = apply_before_send_interceptors(
        &config.interceptors,
        &ctx,
        rb,
        &empty_json,
        &effective_headers,
    )?;

    // 6. Send request

    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 7. Handle 401 retry
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        if status.as_u16() == 401 && should_retry_401 {
            // Notify interceptors of retry
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            // Rebuild headers for retry
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;

            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };

            let builder = |headers: HeaderMap| config.http_client.get(url).headers(headers);
            resp = rebuild_headers_and_retry_once(
                builder,
                &config.interceptors,
                &ctx,
                &empty_json,
                retry_effective_headers.clone(),
            )
            .await?;
        }

        // If still not successful, classify error
        if !resp.status().is_success() {
            let status = resp.status();
            let headers = resp.headers().clone();
            let text = resp.text().await.unwrap_or_default();
            let error = crate::retry_api::classify_http_error(
                &config.provider_id,
                status.as_u16(),
                &text,
                &headers,
                None,
            );
            // Notify interceptors of error
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }
    }

    // 8. Extract response metadata
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();

    // 9. Parse response body
    let text = resp
        .text()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
        .map_err(|e| LlmError::ParseError(e.to_string()))?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// Execute a DELETE request with unified HTTP handling.
///
/// Summary
/// - Builds ProviderSpec headers, merges per-request headers, applies interceptors, sends DELETE,
///   and retries once on 401 with rebuilt headers. Response may be empty JSON.
///
/// Arguments
/// - `config`: HTTP execution configuration (ProviderSpec + context)
/// - `url`: Request URL
/// - `per_request_headers`: Optional per-request header overrides
///
/// Returns
/// - `Ok(HttpExecutionResult)` with parsed JSON (possibly empty), status and headers
/// - `Err(LlmError)` on network/HTTP/interceptor/parse errors
///
/// Example
/// ```ignore
/// # use siumai::execution::executors::common::{HttpExecutionConfig, execute_delete_request};
/// # use siumai::core::{ProviderContext, ProviderSpec};
/// # use std::sync::Arc;
/// # #[derive(Clone)]
/// # struct ExampleSpec;
/// # impl ProviderSpec for ExampleSpec {
/// #   fn id(&self) -> &'static str { "example" }
/// #   fn capabilities(&self) -> siumai::traits::ProviderCapabilities { Default::default() }
/// #   fn build_headers(&self, _ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, siumai::LlmError> { Ok(reqwest::header::HeaderMap::new()) }
/// #   fn chat_url(&self, _stream: bool, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext) -> String { String::new() }
/// #   fn choose_chat_transformers(&self, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext)
/// #     -> siumai::core::ChatTransformers { unimplemented!() }
/// # }
/// # async fn demo() -> Result<(), siumai::LlmError> {
/// # let http = reqwest::Client::new();
/// # let spec = Arc::new(ExampleSpec);
/// # let ctx = ProviderContext::new("example", "https://api.example.com", None, Default::default());
/// # let config = HttpExecutionConfig { provider_id: "example".into(), http_client: http, provider_spec: spec, provider_context: ctx, interceptors: vec![], retry_options: Some(siumai::retry_api::RetryOptions::default()) };
/// let res = execute_delete_request(&config, "https://api.example.com/resource/1", None).await?;
/// # Ok(()) }
/// ```
pub async fn execute_delete_request(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    // 1. Build base headers from ProviderSpec
    let headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers, req_headers)
    } else {
        headers
    };

    // 4. Create request builder
    let mut rb = config
        .http_client
        .delete(url)
        .headers(effective_headers.clone());
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // 5. Apply HTTP interceptors (on_before_send)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };

    let empty_json = serde_json::json!({});
    rb = apply_before_send_interceptors(
        &config.interceptors,
        &ctx,
        rb,
        &empty_json,
        &effective_headers,
    )?;

    // 6. Send request
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 7. Handle 401 retry
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        if status.as_u16() == 401 && should_retry_401 {
            // Notify interceptors of retry
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            // Rebuild headers for retry
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;

            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };

            let builder = |headers: HeaderMap| config.http_client.delete(url).headers(headers);
            resp = rebuild_headers_and_retry_once(
                builder,
                &config.interceptors,
                &ctx,
                &empty_json,
                retry_effective_headers.clone(),
            )
            .await?;
        }

        // If still not successful, classify error
        if !resp.status().is_success() {
            let status = resp.status();
            let headers = resp.headers().clone();
            let text = resp.text().await.unwrap_or_default();
            let error = crate::retry_api::classify_http_error(
                &config.provider_id,
                status.as_u16(),
                &text,
                &headers,
                None,
            );
            // Notify interceptors of error
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }
    }

    // 8. Extract response metadata
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();

    // 9. Parse response body (may be empty for DELETE)
    let text = resp
        .text()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // For DELETE, response may be empty or minimal JSON
    let json: serde_json::Value = if text.trim().is_empty() {
        serde_json::json!({})
    } else {
        crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?
    };

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// Result of HTTP request execution for binary content
pub struct HttpBinaryResult {
    /// Response body as bytes
    pub bytes: Vec<u8>,
    /// Response status code
    pub status: u16,
    /// Response headers
    pub headers: HeaderMap,
}

/// Execute a GET request for binary content (e.g., file download).
///
/// Summary
/// - Same semantics as `execute_get_request` but returns bytes with status and headers.
///
/// Arguments
/// - `config`: HTTP execution configuration (ProviderSpec + context)
/// - `url`: Request URL
/// - `per_request_headers`: Optional per-request header overrides
///
/// Returns
/// - `Ok(HttpBinaryResult)` on success
/// - `Err(LlmError)` on network/HTTP/interceptor errors
///
/// Example
/// ```ignore
/// # use siumai::execution::executors::common::{HttpExecutionConfig, execute_get_binary};
/// # use siumai::core::{ProviderContext, ProviderSpec};
/// # use std::sync::Arc;
/// # #[derive(Clone)]
/// # struct ExampleSpec;
/// # impl ProviderSpec for ExampleSpec {
/// #   fn id(&self) -> &'static str { "example" }
/// #   fn capabilities(&self) -> siumai::traits::ProviderCapabilities { Default::default() }
/// #   fn build_headers(&self, _ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, siumai::LlmError> { Ok(reqwest::header::HeaderMap::new()) }
/// #   fn chat_url(&self, _stream: bool, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext) -> String { String::new() }
/// #   fn choose_chat_transformers(&self, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext)
/// #     -> siumai::core::ChatTransformers { unimplemented!() }
/// # }
/// # async fn demo() -> Result<(), siumai::LlmError> {
/// # let http = reqwest::Client::new();
/// # let spec = Arc::new(ExampleSpec);
/// # let ctx = ProviderContext::new("example", "https://api.example.com", None, Default::default());
/// # let config = HttpExecutionConfig { provider_id: "example".into(), http_client: http, provider_spec: spec, provider_context: ctx, interceptors: vec![], retry_options: Some(siumai::retry_api::RetryOptions::default()) };
/// let res = execute_get_binary(&config, "https://api.example.com/file.bin", None).await?;
/// # Ok(()) }
/// ```
pub async fn execute_get_binary(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpBinaryResult, LlmError> {
    // 1. Build base headers from ProviderSpec
    let headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers, req_headers)
    } else {
        headers
    };

    // 4. Create request builder
    let mut rb = config
        .http_client
        .get(url)
        .headers(effective_headers.clone());
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // 5. Apply HTTP interceptors (on_before_send)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };

    let empty_json = serde_json::json!({});
    rb = apply_before_send_interceptors(
        &config.interceptors,
        &ctx,
        rb,
        &empty_json,
        &effective_headers,
    )?;

    // 6. Send request
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // 7. Handle 401 retry
    if !resp.status().is_success() {
        let status = resp.status();
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        if status.as_u16() == 401 && should_retry_401 {
            // Notify interceptors of retry
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            // Rebuild headers for retry
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;

            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };

            let builder = |headers: HeaderMap| config.http_client.get(url).headers(headers);
            resp = rebuild_headers_and_retry_once(
                builder,
                &config.interceptors,
                &ctx,
                &empty_json,
                retry_effective_headers.clone(),
            )
            .await?;
        }

        // If still not successful, classify error
        if !resp.status().is_success() {
            let status = resp.status();
            let headers = resp.headers().clone();
            let text = resp.text().await.unwrap_or_default();
            let error = crate::retry_api::classify_http_error(
                &config.provider_id,
                status.as_u16(),
                &text,
                &headers,
                None,
            );
            // Notify interceptors of error
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }
    }

    // 8. Extract response metadata
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();

    // 9. Read binary response body
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| LlmError::HttpError(format!("Failed to read response body: {}", e)))?;

    Ok(HttpBinaryResult {
        bytes: bytes.to_vec(),
        status: status_code,
        headers: response_headers,
    })
}
