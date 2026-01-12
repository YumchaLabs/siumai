//! HTTP request helpers (json).

use super::{HttpBody, HttpExecutionConfig, HttpExecutionResult};
use crate::error::LlmError;
use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::{
    apply_before_send_interceptors, headers_from_builder, rebuild_headers_and_retry_once,
};
use crate::execution::http::interceptor::HttpRequestContext;
use crate::execution::http::transport::HttpTransportRequest;
use reqwest::header::HeaderMap;

/// Execute a JSON request (non-stream) with explicit headers, for call sites
/// that already have a `HeaderMap`.
#[allow(clippy::too_many_arguments)]
#[deprecated(
    since = "0.11.0-beta.5",
    note = "Use execute_json_request with HttpExecutionConfig; if you need static headers, use a ProviderSpec whose build_headers() returns that HeaderMap."
)]
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
    use crate::core::ProviderContext;
    use crate::traits::ProviderCapabilities;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[derive(Clone)]
    struct StaticHeadersSpec {
        headers: HeaderMap,
    }

    impl crate::core::ProviderSpec for StaticHeadersSpec {
        fn id(&self) -> &'static str {
            "static_headers"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new()
        }

        fn build_headers(&self, _ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
            Ok(self.headers.clone())
        }

        fn chat_url(
            &self,
            _stream: bool,
            _req: &crate::types::ChatRequest,
            _ctx: &ProviderContext,
        ) -> String {
            unreachable!()
        }

        fn choose_chat_transformers(
            &self,
            _req: &crate::types::ChatRequest,
            _ctx: &ProviderContext,
        ) -> crate::core::ChatTransformers {
            unreachable!()
        }
    }

    let config = HttpExecutionConfig {
        provider_id: provider_id.to_string(),
        http_client: http_client.clone(),
        transport: None,
        provider_spec: Arc::new(StaticHeadersSpec {
            headers: headers_base,
        }),
        provider_context: ProviderContext::new(
            provider_id.to_string(),
            "http://example.invalid".to_string(),
            None,
            HashMap::new(),
        ),
        interceptors: interceptors.to_vec(),
        retry_options,
    };

    execute_json_request(
        &config,
        url,
        HttpBody::Json(body),
        per_request_headers,
        stream,
    )
    .await
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
        config
            .provider_spec
            .merge_request_headers(base_headers.clone(), req_headers)
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
        request_id: crate::execution::http::interceptor::generate_request_id(),
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

    if let Some(transport) = &config.transport {
        let should_retry_401 = config
            .retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        let mut result = transport
            .execute_json(HttpTransportRequest {
                ctx: ctx.clone(),
                url: url.to_string(),
                headers: headers_from_builder(&rb, &effective_headers),
                body: json_body.clone(),
            })
            .await?;

        if result.status == 401 && should_retry_401 {
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

            let mut rb_retry = config
                .http_client
                .post(url)
                .headers(retry_effective_headers.clone())
                .json(&json_body);
            #[cfg(test)]
            {
                rb_retry = rb_retry.header("x-retry-attempt", "1");
            }
            rb_retry = apply_before_send_interceptors(
                &config.interceptors,
                &ctx,
                rb_retry,
                &json_body,
                &retry_effective_headers,
            )?;

            result = transport
                .execute_json(HttpTransportRequest {
                    ctx: ctx.clone(),
                    url: url.to_string(),
                    headers: headers_from_builder(&rb_retry, &retry_effective_headers),
                    body: json_body.clone(),
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
                config
                    .provider_spec
                    .merge_request_headers(retry_headers, req_headers)
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
