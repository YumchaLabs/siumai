//! HTTP request helpers (multipart).

use super::{HttpExecutionConfig, HttpExecutionResult};
use crate::error::LlmError;
use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::{
    apply_before_send_interceptors, build_multipart_body, headers_from_builder,
    rebuild_headers_and_retry_once_multipart, with_multipart_content_headers,
};
use crate::execution::http::interceptor::HttpRequestContext;
use crate::execution::http::transport::HttpTransportMultipartRequest;

/// Multipart JSON request
pub async fn execute_multipart_request<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_http_config: Option<&crate::types::HttpConfig>,
) -> Result<HttpExecutionResult, LlmError>
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

    let ctx = HttpRequestContext {
        request_id: crate::execution::http::interceptor::generate_request_id(),
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };
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

        let mut result = transport
            .execute_multipart(HttpTransportMultipartRequest {
                ctx: ctx.clone(),
                url: url.to_string(),
                headers: headers_from_builder(&rb, &transport_headers),
                body,
            })
            .await?;

        if result.status == 401 && should_retry_401 {
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

            result = transport
                .execute_multipart(HttpTransportMultipartRequest {
                    ctx: ctx.clone(),
                    url: url.to_string(),
                    headers: headers_from_builder(&rb_retry, &retry_transport_headers),
                    body: retry_body,
                })
                .await?;
        }

        if !(200..300).contains(&result.status) {
            let text = String::from_utf8_lossy(&result.body);
            let fallback_message = reqwest::StatusCode::from_u16(result.status)
                .ok()
                .and_then(|s| s.canonical_reason());
            let error = exec_errors::classify_http_error(
                &config.provider_id,
                Some(config.provider_spec.as_ref()),
                result.status,
                &text,
                &result.headers,
                fallback_message,
            );
            for interceptor in &config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }

        let text = String::from_utf8_lossy(&result.body);
        let json: serde_json::Value = exec_errors::parse_json_text_with_ctx(
            &config.provider_id,
            &ctx,
            &config.interceptors,
            &text,
        )?;

        return Ok(HttpExecutionResult {
            json,
            status: result.status,
            headers: result.headers,
        });
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
    rb = apply_before_send_interceptors(
        &config.interceptors,
        &ctx,
        rb,
        &empty_json,
        &effective_headers,
    )?;

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
            let retry_effective_headers = if let Some(req_http) = per_request_http_config {
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, &req_http.headers)
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
                per_request_timeout,
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
