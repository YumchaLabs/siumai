//! Chat executor traits
//!
//! Defines the abstraction that will drive chat operations using transformers
//! and HTTP. For now this is an interface stub for the refactor.

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{ChatRequest, ChatResponse};
use crate::utils::http_interceptor::{HttpInterceptor, HttpRequestContext};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use reqwest::header::HeaderMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[async_trait::async_trait]
pub trait ChatExecutor: Send + Sync {
    async fn execute(&self, req: ChatRequest) -> Result<ChatResponse, LlmError>;
    async fn execute_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError>;
}

/// Generic HTTP-based ChatExecutor that wires transformers and HTTP
pub struct HttpChatExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub request_transformer: Arc<dyn RequestTransformer>,
    pub response_transformer: Arc<dyn ResponseTransformer>,
    pub stream_transformer: Option<Arc<dyn StreamChunkTransformer>>,
    /// Whether to disable compression for streaming requests
    pub stream_disable_compression: bool,
    /// Optional list of HTTP interceptors (order preserved)
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    // Strategy hooks
    pub build_url: Box<dyn Fn(bool) -> String + Send + Sync>,
    pub build_headers: Box<dyn Fn() -> Result<HeaderMap, LlmError> + Send + Sync>,
    /// Optional external parameter transformer (plugin-like), applied to JSON body
    pub before_send: Option<crate::executors::BeforeSendHook>,
}

#[async_trait::async_trait]
impl ChatExecutor for HttpChatExecutor {
    async fn execute(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        // Build request body
        let body = self.request_transformer.transform_chat(&req)?;
        let url = (self.build_url)(false);
        let headers = (self.build_headers)()?;
        let headers_for_interceptors = headers.clone();

        let json_body = if let Some(cb) = &self.before_send {
            cb(&body)?
        } else {
            body
        };

        // Build request and run interceptors
        let mut rb = self.http_client.post(url.clone()).headers(headers);
        let ctx = HttpRequestContext {
            provider_id: self.provider_id.clone(),
            url: url.clone(),
            stream: false,
        };
        for it in &self.interceptors {
            rb = it.on_before_send(&ctx, rb, &json_body, &headers_for_interceptors)?;
        }
        let resp = rb
            .json(&json_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                // Rebuild headers and retry once (helps with refreshed Bearer tokens)
                let url = (self.build_url)(false);
                let headers = (self.build_headers)()?;
                let headers_for_interceptors = headers.clone();
                let json_body = self.request_transformer.transform_chat(&req)?;
                let mut rb = self.http_client.post(url.clone()).headers(headers);
                let ctx = HttpRequestContext {
                    provider_id: self.provider_id.clone(),
                    url: url.clone(),
                    stream: false,
                };
                for it in &self.interceptors {
                    rb = it.on_before_send(&ctx, rb, &json_body, &headers_for_interceptors)?;
                }
                rb.json(&json_body)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?
            } else {
                let headers = resp.headers().clone();
                let text = resp.text().await.unwrap_or_default();
                for it in &self.interceptors {
                    it.on_error(
                        &ctx,
                        &crate::retry_api::classify_http_error(
                            &self.provider_id,
                            status.as_u16(),
                            &text,
                            &headers,
                            None,
                        ),
                    );
                }
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        } else {
            for it in &self.interceptors {
                it.on_response(&ctx, &resp)?;
            }
            resp
        };

        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response JSON: {e}")))?;
        self.response_transformer.transform_chat_response(&json)
    }

    async fn execute_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError> {
        let Some(stream_tx) = &self.stream_transformer else {
            return Err(LlmError::UnsupportedOperation(
                "Streaming not supported by this executor".into(),
            ));
        };
        let body = self.request_transformer.transform_chat(&req)?;
        let url = (self.build_url)(true);

        let transformed = if let Some(cb) = &self.before_send {
            cb(&body)?
        } else {
            body
        };
        // Build request closure for 401 one-shot retry with header rebuild
        let http = self.http_client.clone();
        let header_builder = &self.build_headers;
        let url_for_retry = url.clone();
        let transformed_for_retry = transformed.clone();
        let disable_compression = self.stream_disable_compression;
        let interceptors = self.interceptors.clone();
        let provider_id = self.provider_id.clone();
        let build_request = move || {
            let headers = (header_builder)()?;
            let url_clone = url_for_retry.clone();
            let mut rb = http
                .post(url_clone.clone())
                .headers(headers)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .header(reqwest::header::CACHE_CONTROL, "no-cache")
                .header(reqwest::header::CONNECTION, "keep-alive");
            // Ensure OpenAI bodies include stream flags for compatibility
            // with tests and vercel/ai behavior.
            let mut body_for_send = transformed_for_retry.clone();
            if provider_id.starts_with("openai") {
                body_for_send["stream"] = serde_json::Value::Bool(true);
                if body_for_send.get("stream_options").is_none() {
                    body_for_send["stream_options"] = serde_json::json!({"include_usage": true});
                } else if let Some(obj) = body_for_send["stream_options"].as_object_mut() {
                    obj.entry("include_usage")
                        .or_insert(serde_json::Value::Bool(true));
                }
            }
            rb = rb.json(&body_for_send);
            if disable_compression {
                rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
            }
            // Interceptors on streaming request
            let ctx = HttpRequestContext {
                provider_id: provider_id.clone(),
                url: url_clone,
                stream: true,
            };
            // Build headers again for interceptor visibility
            // Note: headers were already applied above on the builder
            let cloned_headers = rb
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_default();
            for it in &interceptors {
                rb = it.on_before_send(&ctx, rb, &body_for_send, &cloned_headers)?;
            }
            Ok(rb)
        };

        // Use underlying converter via adapter/transformer wrapper
        // The StreamChunkTransformer must implement SseEventConverter via a known inner
        // We expose a small adapter implementing SseEventConverter by delegating to the transformer.
        #[derive(Clone)]
        struct TransformerConverter(Arc<dyn StreamChunkTransformer>);
        impl SseEventConverter for TransformerConverter {
            fn convert_event(
                &self,
                event: Event,
            ) -> Pin<
                Box<
                    dyn Future<Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>>
                        + Send
                        + Sync
                        + '_,
                >,
            > {
                self.0.convert_event(event)
            }
            fn handle_stream_end(
                &self,
            ) -> Option<Result<crate::stream::ChatStreamEvent, LlmError>> {
                self.0.handle_stream_end()
            }
        }

        let converter = TransformerConverter(stream_tx.clone());
        // Wrap converter to notify interceptors of SSE events
        #[derive(Clone)]
        struct InterceptingConverter<C> {
            interceptors: Vec<Arc<dyn HttpInterceptor>>,
            ctx: HttpRequestContext,
            convert: C,
        }
        impl<C> SseEventConverter for InterceptingConverter<C>
        where
            C: SseEventConverter + Clone + Send + Sync + 'static,
        {
            fn convert_event(
                &self,
                event: eventsource_stream::Event,
            ) -> Pin<
                Box<
                    dyn Future<Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>>
                        + Send
                        + Sync
                        + '_,
                >,
            > {
                let interceptors = self.interceptors.clone();
                let ctx = self.ctx.clone();
                let inner = self.convert.clone();
                Box::pin(async move {
                    for it in &interceptors {
                        if let Err(e) = it.on_sse_event(&ctx, &event) {
                            return vec![Err(e)];
                        }
                    }
                    inner.convert_event(event).await
                })
            }
            fn handle_stream_end(
                &self,
            ) -> Option<Result<crate::stream::ChatStreamEvent, LlmError>> {
                self.convert.handle_stream_end()
            }
        }

        let intercepting = InterceptingConverter {
            interceptors: self.interceptors.clone(),
            ctx: HttpRequestContext {
                provider_id: self.provider_id.clone(),
                url: (self.build_url)(true),
                stream: true,
            },
            convert: converter,
        };
        StreamFactory::create_eventsource_stream_with_retry(
            &self.provider_id,
            build_request,
            intercepting,
        )
        .await
    }
}
