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

/// Transport-level request data for multipart POST requests.
#[derive(Debug, Clone)]
pub struct HttpTransportMultipartRequest {
    pub ctx: HttpRequestContext,
    pub url: String,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
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

/// Custom HTTP transport for JSON and multipart requests.
///
/// Notes:
/// - This abstraction is currently scoped to non-streaming JSON/multipart POST
///   requests plus streaming JSON/multipart POST requests.
/// - Interceptor `on_response` hooks are skipped for custom transports to keep
///   transport execution semantics stable and independent from synthesized
///   response wrappers.
#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError>;

    /// Execute a non-streaming multipart POST request.
    ///
    /// The request body is provided as fully materialized multipart bytes with
    /// the final `Content-Type`/`Content-Length` already populated in headers.
    /// The default implementation returns UnsupportedOperation, so existing
    /// transports only implementing `execute_json` keep working.
    async fn execute_multipart(
        &self,
        _request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Custom transport does not support multipart requests".to_string(),
        ))
    }

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

    /// Execute a streaming multipart POST request.
    ///
    /// Providers use this for multipart SSE or progress endpoints such as
    /// OpenAI transcription streaming. The request body is provided as fully
    /// materialized multipart bytes with final content headers populated.
    async fn execute_multipart_stream(
        &self,
        _request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Custom transport does not support streaming multipart requests".to_string(),
        ))
    }
}
