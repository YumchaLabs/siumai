//! JSON streaming execution helpers
//!
//! Line-delimited JSON streaming request execution split out from `common.rs`.

use crate::error::LlmError;
use crate::execution::executors::helpers::apply_before_send_interceptors;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::retry_api::RetryOptions;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Execute a JSON-based streaming request (ProviderSpec-agnostic) and
/// return a unified `ChatStream`.
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
    // Merge request headers
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

    // Before-send interceptors
    let ctx = HttpRequestContext {
        provider_id: provider_id.to_string(),
        url: url.to_string(),
        stream: true,
    };
    rb = apply_before_send_interceptors(interceptors, &ctx, rb, &body, &effective_headers)?;

    // Send request
    let mut resp = rb
        .send()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

    // Single retry on 401
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
            let builder = |headers: HeaderMap| {
                let mut b = http_client.post(url).headers(headers).json(&body);
                if disable_compression {
                    b = b.header(reqwest::header::ACCEPT_ENCODING, "identity");
                }
                b
            };
            // Reuse helper for the single retry
            resp = crate::execution::executors::helpers::rebuild_headers_and_retry_once(
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

    crate::streaming::StreamFactory::create_json_stream(resp, json_converter).await
}
