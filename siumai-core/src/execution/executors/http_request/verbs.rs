//! HTTP request helpers (verbs).

use super::{HttpBinaryResult, HttpExecutionConfig, HttpExecutionResult};
use crate::error::LlmError;
use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::{
    apply_before_send_interceptors, rebuild_headers_and_retry_once,
};
use crate::execution::http::interceptor::HttpRequestContext;
use reqwest::header::HeaderMap;

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
        config
            .provider_spec
            .merge_request_headers(headers, req_headers)
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
        request_id: crate::execution::http::interceptor::generate_request_id(),
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
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, req_headers)
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
            let error = exec_errors::classify_http_error(
                &config.provider_id,
                Some(config.provider_spec.as_ref()),
                status.as_u16(),
                &text,
                &headers,
                status.canonical_reason(),
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
        config
            .provider_spec
            .merge_request_headers(headers, req_headers)
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
        request_id: crate::execution::http::interceptor::generate_request_id(),
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
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, req_headers)
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
            let error = exec_errors::classify_http_error(
                &config.provider_id,
                Some(config.provider_spec.as_ref()),
                status.as_u16(),
                &text,
                &headers,
                status.canonical_reason(),
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

/// DELETE request with a JSON body (via `ProviderSpec`).
///
/// Some provider APIs (e.g. certain local runtimes) accept a JSON body on DELETE.
/// This helper keeps error classification, interceptors and 401 retry behavior aligned
/// with the other common executors.
pub async fn execute_delete_json_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
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

    // 3. Build request
    #[allow(unused_mut)]
    let mut rb = config
        .http_client
        .delete(url)
        .headers(effective_headers.clone())
        .json(&body);
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // 4. Interceptors
    let ctx = HttpRequestContext {
        request_id: crate::execution::http::interceptor::generate_request_id(),
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };
    rb = apply_before_send_interceptors(&config.interceptors, &ctx, rb, &body, &effective_headers)?;

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
            let body_for_retry = body.clone();
            let builder = move |headers: HeaderMap| {
                config
                    .http_client
                    .delete(url)
                    .headers(headers)
                    .json(&body_for_retry)
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

    // 7. Error classification
    if !resp.status().is_success() {
        let status = resp.status();
        let headers = resp.headers().clone();
        let text = resp.text().await.unwrap_or_default();
        let error = exec_errors::classify_http_error(
            &config.provider_id,
            Some(config.provider_spec.as_ref()),
            status.as_u16(),
            &text,
            &headers,
            status.canonical_reason(),
        );
        for interceptor in &config.interceptors {
            interceptor.on_error(&ctx, &error);
        }
        return Err(error);
    }

    // 8. Success path
    for interceptor in &config.interceptors {
        interceptor.on_response(&ctx, &resp)?;
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

/// PATCH request with a JSON body (via `ProviderSpec`).
pub async fn execute_patch_json_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
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

    // 3. Build request
    #[allow(unused_mut)]
    let mut rb = config
        .http_client
        .patch(url)
        .headers(effective_headers.clone())
        .json(&body);
    #[cfg(test)]
    {
        rb = rb.header("x-retry-attempt", "0");
    }

    // 4. Interceptors
    let ctx = HttpRequestContext {
        request_id: crate::execution::http::interceptor::generate_request_id(),
        provider_id: config.provider_id.clone(),
        url: url.to_string(),
        stream: false,
    };
    rb = apply_before_send_interceptors(&config.interceptors, &ctx, rb, &body, &effective_headers)?;

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
            let body_for_retry = body.clone();
            let builder = move |headers: HeaderMap| {
                config
                    .http_client
                    .patch(url)
                    .headers(headers)
                    .json(&body_for_retry)
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

    // 7. Error classification
    if !resp.status().is_success() {
        let status = resp.status();
        let headers = resp.headers().clone();
        let text = resp.text().await.unwrap_or_default();
        let error = exec_errors::classify_http_error(
            &config.provider_id,
            Some(config.provider_spec.as_ref()),
            status.as_u16(),
            &text,
            &headers,
            status.canonical_reason(),
        );
        for interceptor in &config.interceptors {
            interceptor.on_error(&ctx, &error);
        }
        return Err(error);
    }

    // 8. Parse JSON (may be empty)
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
        config
            .provider_spec
            .merge_request_headers(headers, req_headers)
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
        request_id: crate::execution::http::interceptor::generate_request_id(),
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
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, req_headers)
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
            let error = exec_errors::classify_http_error(
                &config.provider_id,
                Some(config.provider_spec.as_ref()),
                status.as_u16(),
                &text,
                &headers,
                status.canonical_reason(),
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
