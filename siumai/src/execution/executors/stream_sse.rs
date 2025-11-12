//! SSE streaming execution helpers
//!
//! SSE-based streaming request execution split out from `common.rs`.

use crate::error::LlmError;
use crate::execution::executors::helpers::headers_from_builder;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::retry_api::RetryOptions;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Execute an SSE-based streaming request (ProviderSpec-agnostic) and
/// return a unified `ChatStream`.
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
    // Build the request closure (used by the retry factory)
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
            // Before-send interceptors
            let ctx = HttpRequestContext {
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
