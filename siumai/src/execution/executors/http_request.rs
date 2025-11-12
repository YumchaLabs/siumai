//! Basic HTTP request helpers (non-stream)
//!
//! Provides stable entry points for common GET/DELETE/GET-binary requests
//! as part of gradually splitting functionality out of `common.rs`.

use crate::error::LlmError;

pub use crate::execution::executors::common::{
    HttpBinaryResult, HttpBody, HttpExecutionConfig, HttpExecutionResult,
};

use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::{
    apply_before_send_interceptors, headers_from_builder, rebuild_headers_and_retry_once,
    rebuild_headers_and_retry_once_multipart,
};
use crate::execution::http::interceptor::HttpRequestContext;
use reqwest::header::HeaderMap;
#[cfg(test)]
use std::sync::{Arc, Mutex};

/// Execute a JSON request (non-stream) with explicit headers, for call sites
/// that already have a `HeaderMap`.
#[allow(clippy::too_many_arguments)]
pub async fn execute_json_request_with_headers(
    http_client: &reqwest::Client,
    provider_id: &str,
    url: &str,
    headers_base: HeaderMap,
    body: serde_json::Value,
    interceptors: &[std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>],
    retry_options: Option<crate::retry_api::RetryOptions>,
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
            let builder = |headers: HeaderMap| http_client.post(url).headers(headers).json(&body);
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
        let err =
            exec_errors::classify_error_with_text(provider_id, resp, &ctx, interceptors).await;
        return Err(err);
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
    let json: serde_json::Value =
        exec_errors::parse_json_text_with_ctx(provider_id, &ctx, interceptors, &text)?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// JSON bytes request (non-multipart), building headers via `ProviderSpec`.
pub async fn execute_bytes_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<crate::execution::executors::common::HttpBytesResult, LlmError> {
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
    #[allow(unused_mut)]
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
    let bytes = exec_errors::read_bytes(resp).await?;
    Ok(crate::execution::executors::common::HttpBytesResult {
        bytes,
        status: status_code,
        headers: response_headers,
    })
}

/// Multipart bytes request (built via `ProviderSpec`).
pub async fn execute_multipart_bytes_request<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<crate::execution::executors::common::HttpBytesResult, LlmError>
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
    let rb = {
        #[cfg(test)]
        {
            let mut b = config
                .http_client
                .post(url)
                .headers(effective_headers.clone())
                .multipart(form);
            b = b.header("x-retry-attempt", "0");
            b
        }
        #[cfg(not(test))]
        {
            config
                .http_client
                .post(url)
                .headers(effective_headers.clone())
                .multipart(form)
        }
    };

    // Apply interceptors (use empty JSON body for visibility)
    let ctx = HttpRequestContext {
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };
    let empty_json = serde_json::json!({});
    let cloned_headers = headers_from_builder(&rb, &effective_headers);
    let mut rb2 = rb;
    for interceptor in &config.interceptors {
        rb2 = interceptor.on_before_send(&ctx, rb2, &empty_json, &cloned_headers)?;
    }

    // 4. Send request
    let mut resp = rb2
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
        let err = exec_errors::classify_error_with_text(
            &config.provider_id,
            resp,
            &ctx,
            &config.interceptors,
        )
        .await;
        return Err(err);
    }

    // 7. Notify interceptors and return bytes
    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();
    let bytes = exec_errors::read_bytes(resp).await?;
    Ok(crate::execution::executors::common::HttpBytesResult {
        bytes,
        status: status_code,
        headers: response_headers,
    })
}

/// JSON request (via `ProviderSpec`) as the common entry point.
pub async fn execute_json_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    stream: bool,
) -> Result<HttpExecutionResult, LlmError> {
    execute_request(config, url, body, per_request_headers, stream).await
}

/// JSON request (core implementation). For multipart, use `execute_multipart_request`.
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

    let json: serde_json::Value = exec_errors::parse_json_text_with_ctx(
        &config.provider_id,
        &ctx,
        &config.interceptors,
        &text,
    )?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// Multipart JSON request
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

    // 5. Handle 401 retry with header rebuild
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
            resp = rebuild_headers_and_retry_once_multipart(
                &config.http_client,
                url,
                &config.interceptors,
                &ctx,
                retry_effective_headers.clone(),
                build_form,
            )
            .await?;
        }
    }

    // 6. Classify errors if still not successful
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

    // 7. Notify interceptors of success and parse JSON
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
    let json: serde_json::Value = exec_errors::parse_json_text_with_ctx(
        &config.provider_id,
        &ctx,
        &config.interceptors,
        &text,
    )?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// GET request (JSON response). Temporarily delegates to `common` implementation.
pub async fn execute_get_request(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    // 1. Build headers via ProviderSpec
    let headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers, req_headers)
    } else {
        headers
    };

    // 3. Build request
    #[allow(unused_mut)]
    let mut rb = {
        let mut b = config
            .http_client
            .get(url)
            .headers(effective_headers.clone());
        #[cfg(test)]
        {
            b = b.header("x-retry-attempt", "0");
        }
        b
    };

    // 4. Apply interceptors
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

        // If still not successful, classify
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
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }
    }

    // 7. Parse JSON
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();
    let text = resp
        .text()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    let json: serde_json::Value = exec_errors::parse_json_text_with_ctx(
        &config.provider_id,
        &ctx,
        &config.interceptors,
        &text,
    )?;

    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// DELETE request (JSON response). Temporarily delegates to `common` implementation.
pub async fn execute_delete_request(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    // 1. Build base headers
    let headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers, req_headers)
    } else {
        headers
    };
    // 2. Build request
    #[allow(unused_mut)]
    let mut rb = {
        let mut b = config
            .http_client
            .delete(url)
            .headers(effective_headers.clone());
        #[cfg(test)]
        {
            b = b.header("x-retry-attempt", "0");
        }
        b
    };
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
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
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
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }
    }
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();
    let text = resp
        .text()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    let json: serde_json::Value = if text.trim().is_empty() {
        serde_json::json!({})
    } else {
        exec_errors::parse_json_text_with_ctx(
            &config.provider_id,
            &ctx,
            &config.interceptors,
            &text,
        )?
    };
    Ok(HttpExecutionResult {
        json,
        status: status_code,
        headers: response_headers,
    })
}

/// GET request (binary response). Temporarily delegates to `common` implementation.
pub async fn execute_get_binary(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpBinaryResult, LlmError> {
    // 1. Build base headers
    let headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;
    let effective_headers = if let Some(req_headers) = per_request_headers {
        crate::execution::http::headers::merge_headers(headers, req_headers)
    } else {
        headers
    };
    // 2. Build request
    #[allow(unused_mut)]
    let mut rb = {
        let mut b = config
            .http_client
            .get(url)
            .headers(effective_headers.clone());
        #[cfg(test)]
        {
            b = b.header("x-retry-attempt", "0");
        }
        b
    };
    // 3. Interceptors
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
    // 4. Send
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    // 5. 401 retry
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
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }
    }
    // 6. Return bytes
    let status_code = resp.status().as_u16();
    let response_headers = resp.headers().clone();
    let bytes = exec_errors::read_bytes(resp).await?;
    Ok(HttpBinaryResult {
        bytes,
        status: status_code,
        headers: response_headers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FlagInterceptor(Arc<Mutex<bool>>);
    impl crate::execution::http::interceptor::HttpInterceptor for FlagInterceptor {
        fn on_error(
            &self,
            _ctx: &crate::execution::http::interceptor::HttpRequestContext,
            _error: &crate::error::LlmError,
        ) {
            *self.0.lock().unwrap() = true;
        }
    }

    #[tokio::test]
    async fn json_with_headers_propagates_error_and_interceptor() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/bad")
            .with_status(400)
            .with_body("bad json")
            .create_async()
            .await;

        let url = format!("{}/bad", server.url());
        let client = reqwest::Client::new();
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({"a":1});
        let flag = Arc::new(Mutex::new(false));
        let it: Arc<dyn crate::execution::http::interceptor::HttpInterceptor> =
            Arc::new(FlagInterceptor(flag.clone()));

        let res = execute_json_request_with_headers(
            &client,
            "test",
            &url,
            headers,
            body,
            &[it],
            Some(crate::retry_api::RetryOptions::default()),
            None,
            false,
        )
        .await;

        assert!(res.is_err(), "expected error for 400 response");
        assert!(*flag.lock().unwrap(), "interceptor not triggered");
    }

    #[tokio::test]
    async fn json_with_headers_retries_401_then_200() {
        let mut server = mockito::Server::new_async().await;
        let _m1 = server
            .mock("POST", "/retry")
            .match_header("x-retry-attempt", "0")
            .with_status(401)
            .with_body("unauthorized")
            .expect(1)
            .create_async()
            .await;

        let _m2 = server
            .mock("POST", "/retry")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"ok\":true}")
            .create_async()
            .await;

        let url = format!("{}/retry", server.url());
        let client = reqwest::Client::new();
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({"q":"x"});

        let res = execute_json_request_with_headers(
            &client,
            "test",
            &url,
            headers,
            body,
            &[],
            Some(crate::retry_api::RetryOptions::default()),
            None,
            false,
        )
        .await
        .expect("should succeed after retry");

        assert_eq!(res.status, 200);
        assert_eq!(res.json["ok"], true);
    }

    #[tokio::test]
    async fn json_with_headers_classifies_429_rate_limit() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/rl")
            .with_status(429)
            .with_header("retry-after", "5")
            .with_body("rate limit")
            .create_async()
            .await;

        let url = format!("{}/rl", server.url());
        let client = reqwest::Client::new();
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({});
        let res = execute_json_request_with_headers(
            &client,
            "test",
            &url,
            headers,
            body,
            &[],
            Some(crate::retry_api::RetryOptions::default()),
            None,
            false,
        )
        .await;
        match res {
            Err(crate::error::LlmError::RateLimitError(_)) => {}
            other => panic!("expected RateLimitError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn json_with_headers_classifies_500_api_error() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/e500")
            .with_status(500)
            .with_body("server error")
            .create_async()
            .await;

        let url = format!("{}/e500", server.url());
        let client = reqwest::Client::new();
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({});
        let res = execute_json_request_with_headers(
            &client,
            "test",
            &url,
            headers,
            body,
            &[],
            Some(crate::retry_api::RetryOptions::default()),
            None,
            false,
        )
        .await;
        match res {
            Err(crate::error::LlmError::ApiError { code, .. }) if code == 500 => {}
            other => panic!("expected ApiError(500), got: {:?}", other),
        }
    }
}
