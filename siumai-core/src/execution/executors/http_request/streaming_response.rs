//! HTTP request helpers (streaming_response).

use super::HttpExecutionConfig;
use crate::error::LlmError;
use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::{
    apply_before_send_interceptors, build_multipart_body, headers_from_builder,
    rebuild_headers_and_retry_once, rebuild_headers_and_retry_once_multipart,
    response_from_stream_transport, with_multipart_content_headers,
};
use crate::execution::http::interceptor::HttpRequestContext;
use crate::execution::http::transport::{HttpTransportMultipartRequest, HttpTransportRequest};
use futures_util::TryStreamExt;
use reqwest::header::HeaderMap;

/// JSON request that returns the raw `reqwest::Response` for streaming consumption.
///
/// This is intended for endpoints where the caller needs access to `bytes_stream()`
/// (e.g., progress streams). It keeps the unified behavior for:
/// - ProviderSpec header building
/// - per-request header merging
/// - HTTP interceptors
/// - 401 single retry with rebuilt headers
/// - error classification (reads body text only on non-success)
pub async fn execute_json_request_streaming_response(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_http_config: Option<&crate::types::HttpConfig>,
) -> Result<reqwest::Response, LlmError> {
    let ctx = HttpRequestContext {
        request_id: crate::execution::http::interceptor::generate_request_id(),
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: true,
    };
    execute_json_request_streaming_response_with_ctx(
        config,
        url,
        body,
        per_request_http_config,
        ctx,
    )
    .await
}

/// JSON request that returns the raw `reqwest::Response` for streaming consumption,
/// using a caller-provided `HttpRequestContext` (so interceptors can correlate SSE events).
pub async fn execute_json_request_streaming_response_with_ctx(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_http_config: Option<&crate::types::HttpConfig>,
    ctx: HttpRequestContext,
) -> Result<reqwest::Response, LlmError> {
    // 1. Build base headers from provider spec
    let base_headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let effective_headers = if let Some(req_http) = per_request_http_config {
        config
            .provider_spec
            .merge_request_headers(base_headers.clone(), &req_http.headers)
    } else {
        base_headers.clone()
    };

    // 3. Build request
    let mut rb = config
        .http_client
        .post(url)
        .headers(effective_headers.clone())
        .json(&body);
    if let Some(req_http) = per_request_http_config
        && let Some(timeout) = req_http.timeout
    {
        rb = rb.timeout(timeout);
    }
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // 4. Interceptors (before send)
    rb = apply_before_send_interceptors(&config.interceptors, &ctx, rb, &body, &effective_headers)?;

    if let Some(transport) = &config.transport {
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        let mut response = transport
            .execute_stream(HttpTransportRequest {
                ctx: ctx.clone(),
                url: url.to_string(),
                headers: headers_from_builder(&rb, &effective_headers),
                body: body.clone(),
            })
            .await?;

        if response.status == 401 && should_retry_401 {
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;
            let retry_effective_headers = if let Some(req_http) = per_request_http_config {
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, &req_http.headers)
            } else {
                retry_headers
            };

            let mut rb_retry = config
                .http_client
                .post(url)
                .headers(retry_effective_headers.clone())
                .json(&body);
            if let Some(req_http) = per_request_http_config
                && let Some(timeout) = req_http.timeout
            {
                rb_retry = rb_retry.timeout(timeout);
            }
            #[cfg(test)]
            {
                rb_retry = rb_retry.header("x-retry-attempt", "1");
            }
            rb_retry = apply_before_send_interceptors(
                &config.interceptors,
                &ctx,
                rb_retry,
                &body,
                &retry_effective_headers,
            )?;

            response = transport
                .execute_stream(HttpTransportRequest {
                    ctx: ctx.clone(),
                    url: url.to_string(),
                    headers: headers_from_builder(&rb_retry, &retry_effective_headers),
                    body: body.clone(),
                })
                .await?;
        }

        if !(200..300).contains(&response.status) {
            let bytes = response.body.into_stream().try_concat().await?;
            let text = String::from_utf8_lossy(&bytes);
            let fallback_message = reqwest::StatusCode::from_u16(response.status)
                .ok()
                .and_then(|s| s.canonical_reason());
            let error = exec_errors::classify_http_error(
                &config.provider_id,
                Some(config.provider_spec.as_ref()),
                response.status,
                &text,
                &response.headers,
                fallback_message,
            );
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }

        return response_from_stream_transport(url, response);
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
            let retry_effective_headers = if let Some(req_http) = per_request_http_config {
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, &req_http.headers)
            } else {
                retry_headers
            };
            let body_for_retry = body.clone();
            let builder = move |headers: HeaderMap| {
                let mut rb = config
                    .http_client
                    .post(url)
                    .headers(headers)
                    .json(&body_for_retry);
                if let Some(req_http) = per_request_http_config
                    && let Some(timeout) = req_http.timeout
                {
                    rb = rb.timeout(timeout);
                }
                rb
            };
            resp = rebuild_headers_and_retry_once(
                builder,
                &config.interceptors,
                &ctx,
                &body,
                retry_effective_headers.clone(),
            )
            .await?;
        }
    }

    // 7. Error classification (read text only on non-success)
    if !resp.status().is_success() {
        let err = exec_errors::classify_error_with_text(
            &config.provider_id,
            Some(config.provider_spec.as_ref()),
            resp,
            &ctx,
            &config.interceptors,
        )
        .await;
        return Err(err);
    }

    // 8. Success path: notify interceptors and return raw response for streaming
    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }
    Ok(resp)
}

/// Multipart request that returns the raw `reqwest::Response` for streaming consumption,
/// using a caller-provided `HttpRequestContext` (so interceptors can correlate SSE events).
pub async fn execute_multipart_request_streaming_response_with_ctx<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_http_config: Option<&crate::types::HttpConfig>,
    ctx: HttpRequestContext,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    // 1. Build base headers from provider spec
    let base_headers = config
        .provider_spec
        .build_headers(&config.provider_context)?;

    // 2. Merge per-request headers if provided
    let mut effective_headers = if let Some(req_http) = per_request_http_config {
        config
            .provider_spec
            .merge_request_headers(base_headers.clone(), &req_http.headers)
    } else {
        base_headers.clone()
    };
    // Multipart must own its boundary-based Content-Type; strip JSON Content-Type if present.
    effective_headers.remove(reqwest::header::CONTENT_TYPE);

    let empty_json = serde_json::json!({});
    let per_request_timeout = per_request_http_config.and_then(|hc| hc.timeout);

    if let Some(transport) = &config.transport {
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        let form = build_form()?;
        let (body, content_type) = build_multipart_body(form).await?;
        let transport_headers =
            with_multipart_content_headers(effective_headers.clone(), &content_type, body.len())?;

        let mut rb = config
            .http_client
            .post(url)
            .headers(transport_headers.clone());
        if let Some(timeout) = per_request_timeout {
            rb = rb.timeout(timeout);
        }
        #[cfg(test)]
        {
            rb = rb.header("x-retry-attempt", "0");
        }
        rb = apply_before_send_interceptors(
            &config.interceptors,
            &ctx,
            rb,
            &empty_json,
            &transport_headers,
        )?;

        let mut response = transport
            .execute_multipart_stream(HttpTransportMultipartRequest {
                ctx: ctx.clone(),
                url: url.to_string(),
                headers: headers_from_builder(&rb, &transport_headers),
                body,
            })
            .await?;

        if response.status == 401 && should_retry_401 {
            for interceptor in &config.interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            let retry_headers = config
                .provider_spec
                .build_headers(&config.provider_context)?;
            let mut retry_effective_headers = if let Some(req_http) = per_request_http_config {
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, &req_http.headers)
            } else {
                retry_headers
            };
            retry_effective_headers.remove(reqwest::header::CONTENT_TYPE);

            let retry_form = build_form()?;
            let (retry_body, retry_content_type) = build_multipart_body(retry_form).await?;
            let retry_transport_headers = with_multipart_content_headers(
                retry_effective_headers,
                &retry_content_type,
                retry_body.len(),
            )?;

            let mut rb_retry = config
                .http_client
                .post(url)
                .headers(retry_transport_headers.clone());
            if let Some(timeout) = per_request_timeout {
                rb_retry = rb_retry.timeout(timeout);
            }
            #[cfg(test)]
            {
                rb_retry = rb_retry.header("x-retry-attempt", "1");
            }
            rb_retry = apply_before_send_interceptors(
                &config.interceptors,
                &ctx,
                rb_retry,
                &empty_json,
                &retry_transport_headers,
            )?;

            response = transport
                .execute_multipart_stream(HttpTransportMultipartRequest {
                    ctx: ctx.clone(),
                    url: url.to_string(),
                    headers: headers_from_builder(&rb_retry, &retry_transport_headers),
                    body: retry_body,
                })
                .await?;
        }

        if !(200..300).contains(&response.status) {
            let bytes = response.body.into_stream().try_concat().await?;
            let text = String::from_utf8_lossy(&bytes);
            let fallback_message = reqwest::StatusCode::from_u16(response.status)
                .ok()
                .and_then(|s| s.canonical_reason());
            let error = exec_errors::classify_http_error(
                &config.provider_id,
                Some(config.provider_spec.as_ref()),
                response.status,
                &text,
                &response.headers,
                fallback_message,
            );
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }

        return response_from_stream_transport(url, response);
    }

    // 3. Build form and request
    let form = build_form()?;
    let mut rb = config
        .http_client
        .post(url)
        .headers(effective_headers.clone())
        .multipart(form);
    if let Some(req_http) = per_request_http_config
        && let Some(timeout) = req_http.timeout
    {
        rb = rb.timeout(timeout);
    }
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // 4. Interceptors (before send)
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
            let mut retry_effective_headers = if let Some(req_http) = per_request_http_config {
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, &req_http.headers)
            } else {
                retry_headers
            };
            retry_effective_headers.remove(reqwest::header::CONTENT_TYPE);

            resp = rebuild_headers_and_retry_once_multipart(
                &config.http_client,
                url,
                &config.interceptors,
                &ctx,
                retry_effective_headers.clone(),
                build_form,
                per_request_http_config.and_then(|hc| hc.timeout),
            )
            .await?;
        }
    }

    // 7. Error classification (read text only on non-success)
    if !resp.status().is_success() {
        let status = resp.status();
        let response_headers = resp.headers().clone();
        let error_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        let error = exec_errors::classify_http_error(
            &config.provider_id,
            Some(config.provider_spec.as_ref()),
            status.as_u16(),
            &error_text,
            &response_headers,
            status.canonical_reason(),
        );
        for interceptor in &config.interceptors {
            interceptor.on_error(&ctx, &error);
        }
        return Err(error);
    }

    // 8. Success path: notify interceptors and return raw response for streaming
    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
    }
    Ok(resp)
}

/// Multipart request that returns the raw `reqwest::Response` for streaming consumption.
pub async fn execute_multipart_request_streaming_response<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_http_config: Option<&crate::types::HttpConfig>,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    let ctx = HttpRequestContext {
        request_id: crate::execution::http::interceptor::generate_request_id(),
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: true,
    };
    execute_multipart_request_streaming_response_with_ctx(
        config,
        url,
        build_form,
        per_request_http_config,
        ctx,
    )
    .await
}
