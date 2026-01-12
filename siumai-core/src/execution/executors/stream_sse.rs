//! SSE streaming execution helpers
//!
//! SSE-based streaming request execution split out from `common.rs`.

use crate::error::LlmError;
use crate::execution::executors::helpers::headers_from_builder;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::execution::http::transport::{HttpTransport, HttpTransportRequest};
use crate::retry_api::RetryOptions;
use futures_util::TryStreamExt;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Execute an SSE-based streaming request (ProviderSpec-agnostic) and
/// return a unified `ChatStream`.
#[allow(clippy::too_many_arguments)]
pub async fn execute_sse_stream_request_with_headers<C>(
    http_client: &reqwest::Client,
    provider_id: &str,
    provider_spec: Option<&dyn crate::core::ProviderSpec>,
    url: &str,
    request_id: String,
    headers_base: HeaderMap,
    body: serde_json::Value,
    interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: Option<RetryOptions>,
    per_request_headers: Option<std::collections::HashMap<String, String>>,
    converter: C,
    disable_compression: bool,
    transport: Option<Arc<dyn HttpTransport>>,
) -> Result<crate::streaming::ChatStream, LlmError>
where
    C: crate::streaming::SseEventConverter + Clone + Send + Sync + 'static,
{
    let ctx = HttpRequestContext {
        request_id,
        provider_id: provider_id.to_string(),
        url: url.to_string(),
        stream: true,
    };

    if let Some(transport) = transport {
        fn max_attempts(opts: &RetryOptions) -> u32 {
            match opts.backend {
                crate::retry_api::RetryBackend::Policy => {
                    opts.policy.as_ref().map(|p| p.max_attempts).unwrap_or(1)
                }
                crate::retry_api::RetryBackend::Backoff => 1,
            }
        }

        let retry_cfg = retry_options.as_ref();
        let allow_transport_retry = retry_cfg.map(|o| o.idempotent).unwrap_or(false);
        let attempts: usize = retry_cfg.map(max_attempts).unwrap_or(1).max(1) as usize;

        let should_retry_401 = retry_options
            .as_ref()
            .map(|opts| opts.retry_401)
            .unwrap_or(true);

        let build_transport_request = {
            let http = http_client.clone();
            let base = headers_base.clone();
            let url_owned = url.to_string();
            let body_owned = body.clone();
            let interceptors = interceptors.to_vec();
            let ctx = ctx.clone();
            move || -> Result<HttpTransportRequest, LlmError> {
                let effective_headers = if let Some(req_headers) = per_request_headers.clone() {
                    if let Some(spec) = provider_spec {
                        spec.merge_request_headers(base.clone(), &req_headers)
                    } else {
                        crate::execution::http::headers::merge_headers(base.clone(), &req_headers)
                    }
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

                let request_headers = headers_from_builder(&rb, &effective_headers);
                let mut out_rb = rb;
                for it in &interceptors {
                    out_rb = it.on_before_send(&ctx, out_rb, &body_owned, &request_headers)?;
                }
                let request_headers = headers_from_builder(&out_rb, &effective_headers);

                Ok(HttpTransportRequest {
                    ctx: ctx.clone(),
                    url: url_owned.clone(),
                    headers: request_headers,
                    body: body_owned.clone(),
                })
            }
        };

        for attempt in 1..=attempts {
            let request = build_transport_request()?;
            let response = match transport.execute_stream(request.clone()).await {
                Ok(r) => r,
                Err(err) => {
                    let should_retry = allow_transport_retry
                        && attempt < attempts
                        && matches!(
                            err,
                            LlmError::TimeoutError(_)
                                | LlmError::ConnectionError(_)
                                | LlmError::HttpError(_)
                        );

                    if should_retry {
                        for interceptor in interceptors {
                            interceptor.on_retry(&ctx, &err, attempt);
                        }
                        continue;
                    }

                    for interceptor in interceptors {
                        interceptor.on_error(&ctx, &err);
                    }
                    return Err(err);
                }
            };

            let mut response = response;

            if response.status == 401 && should_retry_401 {
                let retry_error = LlmError::HttpError("401 Unauthorized".to_string());
                for interceptor in interceptors {
                    interceptor.on_retry(&ctx, &retry_error, attempt);
                }
                response = transport.execute_stream(build_transport_request()?).await?;
            }

            if !(200..300).contains(&response.status) {
                let bytes = response.body.into_stream().try_concat().await?;
                let text = String::from_utf8_lossy(&bytes);
                let fallback_message = reqwest::StatusCode::from_u16(response.status)
                    .ok()
                    .and_then(|s| s.canonical_reason());
                let error = crate::execution::executors::errors::classify_http_error(
                    provider_id,
                    provider_spec,
                    response.status,
                    &text,
                    &response.headers,
                    fallback_message,
                );
                for interceptor in interceptors {
                    interceptor.on_error(&ctx, &error);
                }
                return Err(error);
            }

            return crate::streaming::StreamFactory::stream_from_byte_stream_with_sse_fallback(
                response.headers,
                response.body.into_stream(),
                converter,
            )
            .await;
        }

        return Err(LlmError::InternalError(
            "Streaming handshake retry exhausted without error".to_string(),
        ));
    }

    // Build the request closure (used by the retry factory)
    let build_request = {
        let http = http_client.clone();
        let base = headers_base.clone();
        let url_owned = url.to_string();
        let body_owned = body.clone();
        let interceptors = interceptors.to_vec();
        let ctx = ctx.clone();
        move || -> Result<reqwest::RequestBuilder, LlmError> {
            let effective_headers = if let Some(req_headers) = per_request_headers.clone() {
                if let Some(spec) = provider_spec {
                    spec.merge_request_headers(base.clone(), &req_headers)
                } else {
                    crate::execution::http::headers::merge_headers(base.clone(), &req_headers)
                }
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

    crate::streaming::StreamFactory::create_eventsource_stream_with_retry_options(
        provider_id,
        provider_spec,
        url,
        should_retry_401,
        build_request,
        converter,
        interceptors,
        ctx,
        retry_options,
    )
    .await
}
