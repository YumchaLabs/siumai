//! JSON streaming execution helpers
//!
//! Line-delimited JSON streaming request execution split out from `common.rs`.

use crate::error::LlmError;
use crate::execution::executors::errors as exec_errors;
use crate::execution::executors::helpers::{apply_before_send_interceptors, headers_from_builder};
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::execution::http::transport::{HttpTransport, HttpTransportRequest};
use crate::retry_api::RetryOptions;
use futures_util::{StreamExt, TryStreamExt};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Execute a JSON-based streaming request (ProviderSpec-agnostic) and
/// return a unified `ChatStream`.
#[allow(clippy::too_many_arguments)]
pub async fn execute_json_stream_request_with_headers<C>(
    http_client: &reqwest::Client,
    provider_id: &str,
    provider_spec: Option<&dyn crate::core::ProviderSpec>,
    url: &str,
    headers_base: HeaderMap,
    body: serde_json::Value,
    interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: Option<RetryOptions>,
    per_request_http_config: Option<&crate::types::HttpConfig>,
    json_converter: C,
    disable_compression: bool,
    transport: Option<Arc<dyn HttpTransport>>,
) -> Result<crate::streaming::ChatStream, LlmError>
where
    C: crate::streaming::JsonEventConverter + Clone + Send + Sync + 'static,
{
    // Merge request headers
    let effective_headers = if let Some(req_http) = per_request_http_config {
        if let Some(spec) = provider_spec {
            spec.merge_request_headers(headers_base.clone(), &req_http.headers)
        } else {
            crate::execution::http::headers::merge_headers(headers_base.clone(), &req_http.headers)
        }
    } else {
        headers_base.clone()
    };

    let ctx = HttpRequestContext {
        request_id: crate::execution::http::interceptor::generate_request_id(),
        provider_id: provider_id.to_string(),
        url: url.to_string(),
        stream: true,
    };

    if let Some(transport) = transport {
        let mut rb = http_client
            .post(url)
            .headers(effective_headers.clone())
            .json(&body);
        if let Some(req_http) = per_request_http_config
            && let Some(timeout) = req_http.timeout
        {
            rb = rb.timeout(timeout);
        }
        if disable_compression {
            rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
        }

        let request_headers = headers_from_builder(&rb, &effective_headers);
        let rb = apply_before_send_interceptors(interceptors, &ctx, rb, &body, &request_headers)?;
        let request_headers = headers_from_builder(&rb, &effective_headers);

        let request = HttpTransportRequest {
            ctx: ctx.clone(),
            url: url.to_string(),
            headers: request_headers,
            body: body.clone(),
        };

        let mut response = transport.execute_stream(request).await?;

        if response.status == 401
            && retry_options
                .as_ref()
                .map(|opts| opts.retry_401)
                .unwrap_or(true)
        {
            for interceptor in interceptors {
                interceptor.on_retry(&ctx, &LlmError::HttpError("401 Unauthorized".into()), 1);
            }

            let mut retry_rb = http_client
                .post(url)
                .headers(effective_headers.clone())
                .json(&body);
            if let Some(req_http) = per_request_http_config
                && let Some(timeout) = req_http.timeout
            {
                retry_rb = retry_rb.timeout(timeout);
            }
            if disable_compression {
                retry_rb = retry_rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
            }

            let retry_headers = headers_from_builder(&retry_rb, &effective_headers);
            let retry_rb = apply_before_send_interceptors(
                interceptors,
                &ctx,
                retry_rb,
                &body,
                &retry_headers,
            )?;
            let retry_headers = headers_from_builder(&retry_rb, &effective_headers);

            response = transport
                .execute_stream(HttpTransportRequest {
                    ctx: ctx.clone(),
                    url: url.to_string(),
                    headers: retry_headers,
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

        return create_json_stream_from_transport_body(response.body.into_stream(), json_converter)
            .await;
    }

    // Build request
    let mut rb = http_client
        .post(url)
        .headers(effective_headers.clone())
        .json(&body);
    if let Some(req_http) = per_request_http_config
        && let Some(timeout) = req_http.timeout
    {
        rb = rb.timeout(timeout);
    }
    if disable_compression {
        rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
    }

    // Before-send interceptors
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
                if let Some(req_http) = per_request_http_config
                    && let Some(timeout) = req_http.timeout
                {
                    b = b.timeout(timeout);
                }
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
        let error = exec_errors::classify_http_error(
            provider_id,
            provider_spec,
            status.as_u16(),
            &error_text,
            &response_headers,
            status.canonical_reason(),
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

async fn create_json_stream_from_transport_body<C, S>(
    body_stream: S,
    converter: C,
) -> Result<crate::streaming::ChatStream, LlmError>
where
    C: crate::streaming::JsonEventConverter + Clone + Send + Sync + 'static,
    S: futures_util::Stream<Item = Result<Vec<u8>, LlmError>> + Send + Sync + Unpin + 'static,
{
    use tokio_util::codec::{FramedRead, LinesCodec};
    use tokio_util::io::StreamReader;

    let reader = StreamReader::new(body_stream.map(|res| {
        res.map(bytes::Bytes::from)
            .map_err(|e| std::io::Error::other(format!("Stream error: {e}")))
    }));
    let lines = FramedRead::new(reader, LinesCodec::new());
    let end_converter = converter.clone();

    let chat_stream = lines
        .map(|res| match res {
            Ok(line) => Ok(line),
            Err(e) => Err(LlmError::ParseError(format!("JSON line error: {e}"))),
        })
        .then(move |line_res: Result<String, LlmError>| {
            let converter = converter.clone();
            async move {
                match line_res {
                    Ok(line) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            return vec![];
                        }
                        converter.convert_json(trimmed).await
                    }
                    Err(e) => vec![Err(e)],
                }
            }
        })
        .flat_map(futures::stream::iter)
        .chain(futures::stream::iter(
            end_converter.handle_stream_end_events(),
        ));

    Ok(Box::pin(chat_stream))
}
#[cfg(test)]
mod tests {
    use super::create_json_stream_from_transport_body;
    use crate::error::LlmError;
    use crate::streaming::{ChatStreamEvent, JsonEventConverter};
    use crate::types::{ChatResponse, FinishReason};
    use futures_util::StreamExt;
    use serde_json::json;
    use std::future::Future;
    use std::pin::Pin;

    #[derive(Clone)]
    struct MultiEndJsonConverter;

    impl JsonEventConverter for MultiEndJsonConverter {
        fn convert_json<'a>(
            &'a self,
            json_data: &'a str,
        ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>
        {
            Box::pin(async move {
                let value: serde_json::Value =
                    serde_json::from_str(json_data).expect("valid JSON line");
                vec![Ok(ChatStreamEvent::ContentDelta {
                    delta: value["delta"].as_str().expect("delta string").to_string(),
                    index: None,
                })]
            })
        }

        fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
            vec![
                Ok(ChatStreamEvent::Custom {
                    event_type: "test:end".to_string(),
                    data: json!({"phase": "done"}),
                }),
                Ok(ChatStreamEvent::StreamEnd {
                    response: ChatResponse::empty_with_finish_reason(FinishReason::Stop),
                }),
            ]
        }
    }

    #[tokio::test]
    async fn transport_json_stream_emits_multiple_end_events() {
        let body_stream = futures_util::stream::iter(vec![Ok(b"{\"delta\":\"hello\"}\n".to_vec())]);

        let stream = create_json_stream_from_transport_body(body_stream, MultiEndJsonConverter)
            .await
            .expect("stream should be created");

        let events = stream.collect::<Vec<_>>().await;
        assert_eq!(events.len(), 3);

        match &events[0] {
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => assert_eq!(delta, "hello"),
            other => panic!("unexpected first event: {other:?}"),
        }

        match &events[1] {
            Ok(ChatStreamEvent::Custom { event_type, data }) => {
                assert_eq!(event_type, "test:end");
                assert_eq!(data["phase"], "done");
            }
            other => panic!("unexpected second event: {other:?}"),
        }

        match &events[2] {
            Ok(ChatStreamEvent::StreamEnd { response }) => {
                assert_eq!(response.finish_reason, Some(FinishReason::Stop));
            }
            other => panic!("unexpected third event: {other:?}"),
        }
    }
}
