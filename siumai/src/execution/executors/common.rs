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

use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::retry_api::RetryOptions;
use reqwest::header::HeaderMap;
use std::sync::Arc;

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
    let cloned_headers = rb
        .try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_default();
    for interceptor in &config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &json_body, &cloned_headers)?;
    }

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
            let mut rb_retry = config
                .http_client
                .post(url)
                .headers(retry_effective_headers)
                .json(&json_body);
            let cloned_headers_retry = rb_retry
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();
            for interceptor in &config.interceptors {
                rb_retry = interceptor.on_before_send(
                    &ctx,
                    rb_retry,
                    &json_body,
                    &cloned_headers_retry,
                )?;
            }
            resp = rb_retry
                .send()
                .await
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
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

/// Execute a JSON HTTP request using explicit base headers (no ProviderSpec).
///
/// This helper is useful for code paths that already have a fully constructed
/// header map and do not rely on ProviderSpec routing or header building.
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

    let ctx = HttpRequestContext {
        provider_id: provider_id.to_string(),
        url: url.to_string(),
        stream,
    };

    // Interceptors (before send)
    let cloned_headers = rb
        .try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_default();
    for interceptor in interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &body, &cloned_headers)?;
    }

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
            let mut rb_retry = http_client
                .post(url)
                .headers(retry_effective_headers)
                .json(&body);
            let cloned_headers_retry = rb_retry
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();
            for interceptor in interceptors {
                rb_retry =
                    interceptor.on_before_send(&ctx, rb_retry, &body, &cloned_headers_retry)?;
            }
            resp = rb_retry
                .send()
                .await
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
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

    // Build effective headers for interceptor visibility
    let cloned_headers = rb
        .try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_default();

    for interceptor in &config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &json_body, &cloned_headers)?;
    }

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

            // Rebuild headers and retry once
            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;

            // Merge per-request headers again
            let retry_effective_headers = if let Some(req_headers) = per_request_headers {
                crate::execution::http::headers::merge_headers(retry_headers, req_headers)
            } else {
                retry_headers
            };

            let mut rb_retry = config
                .http_client
                .post(url)
                .headers(retry_effective_headers);

            // Apply body again
            rb_retry = match &body {
                HttpBody::Json(json) => rb_retry.json(json),
                HttpBody::Multipart(_) => {
                    return Err(LlmError::InvalidParameter(
                        "Use execute_multipart_request for multipart bodies".into(),
                    ));
                }
            };

            // Apply interceptors again
            let cloned_headers_retry = rb_retry
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();

            for interceptor in &config.interceptors {
                rb_retry = interceptor.on_before_send(
                    &ctx,
                    rb_retry,
                    &json_body,
                    &cloned_headers_retry,
                )?;
            }

            resp = rb_retry
                .send()
                .await
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
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
        .unwrap_or_default();

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

            let retry_form = build_form()?;
            let mut rb_retry = config
                .http_client
                .post(url)
                .headers(retry_effective_headers)
                .multipart(retry_form);

            let cloned_headers_retry = rb_retry
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();

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

/// Execute a GET request with unified HTTP handling
///
/// This function provides the same unified HTTP handling as `execute_json_request`,
/// but for GET requests (no request body).
///
/// # Arguments
/// * `config` - HTTP execution configuration
/// * `url` - Request URL
/// * `per_request_headers` - Optional per-request headers to merge
///
/// # Returns
/// * `Ok(HttpExecutionResult)` - Successful response with JSON body
/// * `Err(LlmError)` - HTTP error, parse error, or other failure
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

    // 5. Apply HTTP interceptors (on_before_send)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };

    let empty_json = serde_json::json!({});
    let cloned_headers = rb
        .try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_default();

    for interceptor in &config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &empty_json, &cloned_headers)?;
    }

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

            let mut rb_retry = config.http_client.get(url).headers(retry_effective_headers);

            // Apply interceptors again
            let cloned_headers_retry = rb_retry
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();

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

/// Execute a DELETE request with unified HTTP handling
///
/// This function provides the same unified HTTP handling as `execute_json_request`,
/// but for DELETE requests (no request body, may return empty response).
///
/// # Arguments
/// * `config` - HTTP execution configuration
/// * `url` - Request URL
/// * `per_request_headers` - Optional per-request headers to merge
///
/// # Returns
/// * `Ok(HttpExecutionResult)` - Successful response (may have empty JSON body)
/// * `Err(LlmError)` - HTTP error or other failure
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

    // 5. Apply HTTP interceptors (on_before_send)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };

    let empty_json = serde_json::json!({});
    let cloned_headers = rb
        .try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_default();

    for interceptor in &config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &empty_json, &cloned_headers)?;
    }

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

            let mut rb_retry = config
                .http_client
                .delete(url)
                .headers(retry_effective_headers);

            // Apply interceptors again
            let cloned_headers_retry = rb_retry
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();

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

/// Execute a GET request for binary content (e.g., file download)
///
/// This function is similar to `execute_get_request` but returns binary data
/// instead of parsing JSON.
///
/// # Arguments
/// * `config` - HTTP execution configuration
/// * `url` - Request URL
/// * `per_request_headers` - Optional per-request headers to merge
///
/// # Returns
/// * `Ok(HttpBinaryResult)` - Successful response with binary body
/// * `Err(LlmError)` - HTTP error or other failure
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

    // 5. Apply HTTP interceptors (on_before_send)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };

    let empty_json = serde_json::json!({});
    let cloned_headers = rb
        .try_clone()
        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
        .unwrap_or_default();

    for interceptor in &config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &empty_json, &cloned_headers)?;
    }

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

            let mut rb_retry = config.http_client.get(url).headers(retry_effective_headers);

            // Apply interceptors again
            let cloned_headers_retry = rb_retry
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();

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
