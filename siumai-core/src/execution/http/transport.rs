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
}
