//! HTTP transport abstraction (experimental).
//!
//! This module exists to align with the Vercel AI SDK concept of providing a
//! "custom fetch" implementation per provider/model. In Rust, we expose this
//! as an injectable transport that can observe the final URL/headers/body and
//! return a synthetic response without going through `reqwest`.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpRequestContext;
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use std::pin::Pin;

/// Transport-level request data for JSON POST requests.
#[derive(Debug, Clone)]
pub struct HttpTransportRequest {
    pub ctx: HttpRequestContext,
    pub url: String,
    pub headers: HeaderMap,
    pub body: serde_json::Value,
}

/// Transport-level response data.
#[derive(Debug, Clone)]
pub struct HttpTransportResponse {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

/// Streaming transport response body.
///
/// Notes:
/// - `Full` is useful for tests and for providers that buffer the full response.
/// - `Stream` enables true streaming (e.g. SSE), where each item is a raw byte chunk.
pub enum HttpTransportStreamBody {
    Full(Vec<u8>),
    Stream(
        Pin<
            Box<dyn futures_util::Stream<Item = Result<Vec<u8>, LlmError>> + Send + Sync + 'static>,
        >,
    ),
}

impl HttpTransportStreamBody {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::Full(bytes)
    }

    pub fn from_stream<S>(stream: S) -> Self
    where
        S: futures_util::Stream<Item = Result<Vec<u8>, LlmError>> + Send + Sync + 'static,
    {
        Self::Stream(Box::pin(stream))
    }

    pub fn into_stream(
        self,
    ) -> Pin<Box<dyn futures_util::Stream<Item = Result<Vec<u8>, LlmError>> + Send + Sync + 'static>>
    {
        match self {
            Self::Full(bytes) => Box::pin(futures_util::stream::once(async move { Ok(bytes) })),
            Self::Stream(stream) => stream,
        }
    }
}

/// Transport-level response data for streaming requests.
pub struct HttpTransportStreamResponse {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: HttpTransportStreamBody,
}

/// Custom HTTP transport for JSON requests.
///
/// Notes:
/// - This abstraction is currently scoped to non-streaming JSON POST requests.
/// - Interceptor `on_response` hooks are skipped for custom transports because
///   we don't have a `reqwest::Response` instance.
#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError>;

    /// Execute a streaming JSON POST request.
    ///
    /// Providers typically use this for SSE streaming endpoints. The default
    /// implementation returns UnsupportedOperation, so existing transports only
    /// implementing `execute_json` keep working.
    async fn execute_stream(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Custom transport does not support streaming requests".to_string(),
        ))
    }
}
