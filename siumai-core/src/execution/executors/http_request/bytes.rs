//! HTTP request helpers (bytes).

use super::{HttpBody, HttpExecutionConfig};
use crate::error::LlmError;
use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::headers_from_builder;
use crate::execution::executors::helpers::{
    apply_before_send_interceptors, rebuild_headers_and_retry_once,
};
use crate::execution::http::interceptor::HttpRequestContext;
use reqwest::header::HeaderMap;

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
        config
            .provider_spec
            .merge_request_headers(base_headers.clone(), req_headers)
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
        request_id: crate::execution::http::interceptor::generate_request_id(),
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
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, req_headers)
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
        request_id: crate::execution::http::interceptor::generate_request_id(),
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
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, req_headers)
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
            Some(config.provider_spec.as_ref()),
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
