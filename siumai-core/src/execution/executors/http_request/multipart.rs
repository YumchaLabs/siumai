//! HTTP request helpers (multipart).

use super::{HttpExecutionConfig, HttpExecutionResult};
use crate::error::LlmError;
use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::{
    headers_from_builder, rebuild_headers_and_retry_once_multipart,
};
use crate::execution::http::interceptor::HttpRequestContext;

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
    let mut effective_headers = if let Some(req_headers) = per_request_headers {
        config
            .provider_spec
            .merge_request_headers(base_headers.clone(), req_headers)
    } else {
        base_headers.clone()
    };
    // Multipart must own its boundary-based Content-Type; strip JSON Content-Type if present.
    effective_headers.remove(reqwest::header::CONTENT_TYPE);

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
        request_id: crate::execution::http::interceptor::generate_request_id(),
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
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, req_headers)
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
