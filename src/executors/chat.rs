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
    // Strategy hooks
    pub build_url: Box<dyn Fn(bool) -> String + Send + Sync>,
    pub build_headers: Box<dyn Fn() -> Result<HeaderMap, LlmError> + Send + Sync>,
    /// Optional external parameter transformer (plugin-like), applied to JSON body
    pub before_send: Option<
        Arc<dyn Fn(&serde_json::Value) -> Result<serde_json::Value, LlmError> + Send + Sync>,
    >,
}

#[async_trait::async_trait]
impl ChatExecutor for HttpChatExecutor {
    async fn execute(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        // Build request body
        let body = self.request_transformer.transform_chat(&req)?;
        let url = (self.build_url)(false);
        let headers = (self.build_headers)()?;

        let resp = self
            .http_client
            .post(url)
            .headers(headers)
            .json(
                &(if let Some(cb) = &self.before_send {
                    cb(&body)?
                } else {
                    body
                }),
            )
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }

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
        let headers = (self.build_headers)()?;

        let transformed = if let Some(cb) = &self.before_send {
            cb(&body)?
        } else {
            body
        };
        let request_builder = self
            .http_client
            .post(url)
            .headers(headers)
            .json(&transformed);

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
        StreamFactory::create_eventsource_stream(request_builder, converter).await
    }
}
