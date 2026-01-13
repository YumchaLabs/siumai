//! Common HTTP Execution Layer
//!
//! This module provides unified HTTP request/response handling for all executors,
//! eliminating code duplication across chat/embedding/image/files executors.
//!
//! Key features:
//! - Unified HTTP request sending with interceptors
//! - Automatic 401 retry with header rebuild
//! - Unified error classification
//! - JSON parsing with automatic repair
//! - Per-request header merging
//! - Tracing headers injection
//! - Telemetry integration
//!
//! Retry Helpers
//! - `rebuild_headers_and_retry_once` re-creates a RequestBuilder with rebuilt/effective headers
//!   and re-applies `on_before_send` interceptors before a single retry attempt.
//! - `rebuild_headers_and_retry_once_multipart` does the same for multipart forms by rebuilding
//!   the form (multipart bodies are not cloneable), then re-applies interceptors and retries once.
//! - The helpers assume requests are idempotent (typical for LLM HTTP calls). Callers decide when
//!   to retry (e.g. only on 401) and prepare the correct effective headers.

use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::retry_api::RetryOptions;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// HTTP request body type
#[derive(Debug)]
pub enum HttpBody {
    /// JSON body
    Json(serde_json::Value),
    /// Multipart form body
    Multipart(reqwest::multipart::Form),
}

/// Configuration for HTTP request execution
#[derive(Clone)]
pub struct HttpExecutionConfig {
    /// Provider ID for logging and telemetry
    pub provider_id: String,
    /// HTTP client
    pub http_client: reqwest::Client,
    /// Optional custom transport (Vercel-style "custom fetch" parity).
    pub transport: Option<Arc<dyn HttpTransport>>,
    /// Provider spec for header building
    pub provider_spec: Arc<dyn ProviderSpec>,
    /// Provider context
    pub provider_context: ProviderContext,
    /// HTTP interceptors (order preserved)
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Retry options
    pub retry_options: Option<RetryOptions>,
}

/// Result of HTTP request execution
#[derive(Debug)]
pub struct HttpExecutionResult {
    /// Response body as JSON
    pub json: serde_json::Value,
    /// Response status code
    pub status: u16,
    /// Response headers
    pub headers: HeaderMap,
}

/// Result for byte-response requests (e.g., TTS audio bytes)
#[derive(Debug)]
pub struct HttpBytesResult {
    /// Raw response bytes
    pub bytes: Vec<u8>,
    /// Response status
    pub status: u16,
    /// Response headers
    pub headers: HeaderMap,
}

// moved to stream_sse.rs

// moved to stream_json.rs

/// Execute a request that returns bytes using ProviderSpec (JSON only).
/// For multipart bytes request, prefer a specialized path with a form builder.
pub async fn execute_bytes_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpBytesResult, LlmError> {
    crate::execution::executors::http_request::execute_bytes_request(
        config,
        url,
        body,
        per_request_headers,
    )
    .await
}

/// Execute a multipart HTTP request that returns binary content (bytes)
///
/// Mirrors `execute_multipart_request` but returns raw bytes instead of JSON.
/// Multipart forms cannot be cloned; the `build_form` function will be called
/// for the initial request and again for a retry (if applicable).
pub async fn execute_multipart_bytes_request<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpBytesResult, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    crate::execution::executors::http_request::execute_multipart_bytes_request(
        config,
        url,
        build_form,
        per_request_headers,
    )
    .await
}

/// Execute a JSON HTTP request using explicit base headers (no ProviderSpec).
///
/// This helper is useful for code paths that already have a fully constructed
/// header map and do not rely on ProviderSpec routing or header building.
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
    interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: Option<RetryOptions>,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    stream: bool,
) -> Result<HttpExecutionResult, LlmError> {
    #[allow(deprecated)]
    crate::execution::executors::http_request::execute_json_request_with_headers(
        http_client,
        provider_id,
        url,
        headers_base,
        body,
        interceptors,
        retry_options,
        per_request_headers,
        stream,
    )
    .await
}

/// Execute a JSON HTTP request with unified retry, interceptors, and error handling
///
/// This function provides the complete HTTP execution pipeline:
/// 1. Build base headers from provider spec
/// 2. Inject tracing headers
/// 3. Merge per-request headers (if provided)
/// 4. Apply HTTP interceptors
/// 5. Send request
/// 6. Handle 401 retry with header rebuild
/// 7. Classify errors
/// 8. Parse JSON response with automatic repair
///
/// # Arguments
///
/// * `config` - HTTP execution configuration
/// * `url` - Request URL
/// * `body` - Request body (JSON or Multipart)
/// * `per_request_headers` - Optional per-request headers to merge
/// * `stream` - Whether this is a streaming request (for context)
///
/// # Returns
///
/// Returns `HttpExecutionResult` containing parsed JSON, status, and headers
pub async fn execute_json_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    stream: bool,
) -> Result<HttpExecutionResult, LlmError> {
    crate::execution::executors::http_request::execute_json_request(
        config,
        url,
        body,
        per_request_headers,
        stream,
    )
    .await
}

/// Execute a JSON request and return the raw `reqwest::Response` for streaming consumption.
pub async fn execute_json_request_streaming_response(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<reqwest::Response, LlmError> {
    crate::execution::executors::http_request::execute_json_request_streaming_response(
        config,
        url,
        body,
        per_request_headers,
    )
    .await
}

/// Execute a JSON request and return the raw `reqwest::Response` for streaming consumption,
/// using a caller-provided `HttpRequestContext`.
pub async fn execute_json_request_streaming_response_with_ctx(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    ctx: crate::execution::http::interceptor::HttpRequestContext,
) -> Result<reqwest::Response, LlmError> {
    crate::execution::executors::http_request::execute_json_request_streaming_response_with_ctx(
        config,
        url,
        body,
        per_request_headers,
        ctx,
    )
    .await
}

/// Execute a multipart request and return the raw `reqwest::Response` for streaming consumption.
pub async fn execute_multipart_request_streaming_response<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    crate::execution::executors::http_request::execute_multipart_request_streaming_response(
        config,
        url,
        build_form,
        per_request_headers,
    )
    .await
}

/// Execute a multipart request and return the raw `reqwest::Response` for streaming consumption,
/// using a caller-provided `HttpRequestContext`.
pub async fn execute_multipart_request_streaming_response_with_ctx<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    ctx: crate::execution::http::interceptor::HttpRequestContext,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    crate::execution::executors::http_request::execute_multipart_request_streaming_response_with_ctx(
        config,
        url,
        build_form,
        per_request_headers,
        ctx,
    )
    .await
}

// unit tests migrated to integration tests in tests/http_common_retry_401.rs

/// Execute an HTTP request (JSON or Multipart) with unified retry, interceptors, and error handling
pub async fn execute_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: HttpBody,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
    stream: bool,
) -> Result<HttpExecutionResult, LlmError> {
    crate::execution::executors::http_request::execute_request(
        config,
        url,
        body,
        per_request_headers,
        stream,
    )
    .await
}

/// Execute a multipart HTTP request with unified retry and error handling
///
/// Similar to `execute_json_request` but handles multipart form data.
/// Note: Multipart forms cannot be cloned, so retry requires rebuilding the form.
///
/// # Arguments
///
/// * `config` - HTTP execution configuration
/// * `url` - Request URL
/// * `build_form` - Function to build the multipart form (called for initial request and retry)
/// * `per_request_headers` - Optional per-request headers to merge
///
/// # Returns
///
/// Returns `HttpExecutionResult` containing parsed JSON, status, and headers
pub async fn execute_multipart_request<F>(
    config: &HttpExecutionConfig,
    url: &str,
    build_form: F,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError>
where
    F: Fn() -> Result<reqwest::multipart::Form, LlmError>,
{
    crate::execution::executors::http_request::execute_multipart_request(
        config,
        url,
        build_form,
        per_request_headers,
    )
    .await
}

/// Execute a GET request with unified HTTP handling.
///
/// Summary
/// - Builds ProviderSpec headers, merges per-request headers, applies interceptors, sends GET,
///   and retries once on 401 with rebuilt headers.
///
/// Arguments
/// - `config`: HTTP execution configuration (ProviderSpec + context)
/// - `url`: Request URL
/// - `per_request_headers`: Optional per-request header overrides
///
/// Returns
/// - `Ok(HttpExecutionResult)` with parsed JSON, status and headers
/// - `Err(LlmError)` on network/HTTP/interceptor/parse errors
///
/// Example
/// ```ignore
/// use siumai::experimental::execution::executors::common::{HttpExecutionConfig, execute_get_request};
/// use siumai_core::core::{ProviderContext, ProviderSpec};
/// use std::sync::Arc;
///
/// // Minimal ProviderSpec for example (builds static headers and URL routing)
/// #[derive(Clone)]
/// struct ExampleSpec;
/// impl ProviderSpec for ExampleSpec {
///   fn id(&self) -> &'static str { "example" }
///   fn capabilities(&self) -> siumai::traits::ProviderCapabilities { Default::default() }
///   fn build_headers(&self, _ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, siumai::LlmError> {
///     Ok(reqwest::header::HeaderMap::new())
///   }
///   fn chat_url(&self, _stream: bool, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext) -> String { String::new() }
///   fn choose_chat_transformers(&self, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext)
///     -> siumai_core::core::ChatTransformers { unimplemented!() }
/// }
///
/// # async fn demo() -> Result<(), siumai::LlmError> {
/// let http = reqwest::Client::new();
/// let spec = Arc::new(ExampleSpec);
/// let ctx = ProviderContext::new("example", "https://api.example.com", None, Default::default());
/// let config = HttpExecutionConfig {
///   provider_id: "example".into(),
///   http_client: http,
///   provider_spec: spec,
///   provider_context: ctx,
///   interceptors: vec![],
///   retry_options: Some(siumai::retry_api::RetryOptions::default()),
/// };
/// let res = execute_get_request(&config, "https://api.example.com/ping", None).await?;
/// # Ok(()) }
/// ```
pub async fn execute_get_request(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    crate::execution::executors::http_request::execute_get_request(config, url, per_request_headers)
        .await
}

/// Execute a DELETE request with unified HTTP handling.
///
/// Summary
/// - Builds ProviderSpec headers, merges per-request headers, applies interceptors, sends DELETE,
///   and retries once on 401 with rebuilt headers. Response may be empty JSON.
///
/// Arguments
/// - `config`: HTTP execution configuration (ProviderSpec + context)
/// - `url`: Request URL
/// - `per_request_headers`: Optional per-request header overrides
///
/// Returns
/// - `Ok(HttpExecutionResult)` with parsed JSON (possibly empty), status and headers
/// - `Err(LlmError)` on network/HTTP/interceptor/parse errors
///
/// Example
/// ```ignore
/// # use siumai::experimental::execution::executors::common::{HttpExecutionConfig, execute_delete_request};
/// # use siumai_core::core::{ProviderContext, ProviderSpec};
/// # use std::sync::Arc;
/// # #[derive(Clone)]
/// # struct ExampleSpec;
/// # impl ProviderSpec for ExampleSpec {
/// #   fn id(&self) -> &'static str { "example" }
/// #   fn capabilities(&self) -> siumai::traits::ProviderCapabilities { Default::default() }
/// #   fn build_headers(&self, _ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, siumai::LlmError> { Ok(reqwest::header::HeaderMap::new()) }
/// #   fn chat_url(&self, _stream: bool, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext) -> String { String::new() }
/// #   fn choose_chat_transformers(&self, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext)
/// #     -> siumai_core::core::ChatTransformers { unimplemented!() }
/// # }
/// # async fn demo() -> Result<(), siumai::LlmError> {
/// # let http = reqwest::Client::new();
/// # let spec = Arc::new(ExampleSpec);
/// # let ctx = ProviderContext::new("example", "https://api.example.com", None, Default::default());
/// # let config = HttpExecutionConfig { provider_id: "example".into(), http_client: http, provider_spec: spec, provider_context: ctx, interceptors: vec![], retry_options: Some(siumai::retry_api::RetryOptions::default()) };
/// let res = execute_delete_request(&config, "https://api.example.com/resource/1", None).await?;
/// # Ok(()) }
/// ```
pub async fn execute_delete_request(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    crate::execution::executors::http_request::execute_delete_request(
        config,
        url,
        per_request_headers,
    )
    .await
}

/// Execute a DELETE request with a JSON body using ProviderSpec (rare but supported by some APIs).
pub async fn execute_delete_json_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    crate::execution::executors::http_request::execute_delete_json_request(
        config,
        url,
        body,
        per_request_headers,
    )
    .await
}

/// Execute a PATCH request with a JSON body using ProviderSpec.
pub async fn execute_patch_json_request(
    config: &HttpExecutionConfig,
    url: &str,
    body: serde_json::Value,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpExecutionResult, LlmError> {
    crate::execution::executors::http_request::execute_patch_json_request(
        config,
        url,
        body,
        per_request_headers,
    )
    .await
}

/// Result of HTTP request execution for binary content
pub struct HttpBinaryResult {
    /// Response body as bytes
    pub bytes: Vec<u8>,
    /// Response status code
    pub status: u16,
    /// Response headers
    pub headers: HeaderMap,
}

/// Execute a GET request for binary content (e.g., file download).
///
/// Summary
/// - Same semantics as `execute_get_request` but returns bytes with status and headers.
///
/// Arguments
/// - `config`: HTTP execution configuration (ProviderSpec + context)
/// - `url`: Request URL
/// - `per_request_headers`: Optional per-request header overrides
///
/// Returns
/// - `Ok(HttpBinaryResult)` on success
/// - `Err(LlmError)` on network/HTTP/interceptor errors
///
/// Example
/// ```ignore
/// # use siumai::experimental::execution::executors::common::{HttpExecutionConfig, execute_get_binary};
/// # use siumai_core::core::{ProviderContext, ProviderSpec};
/// # use std::sync::Arc;
/// # #[derive(Clone)]
/// # struct ExampleSpec;
/// # impl ProviderSpec for ExampleSpec {
/// #   fn id(&self) -> &'static str { "example" }
/// #   fn capabilities(&self) -> siumai::traits::ProviderCapabilities { Default::default() }
/// #   fn build_headers(&self, _ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, siumai::LlmError> { Ok(reqwest::header::HeaderMap::new()) }
/// #   fn chat_url(&self, _stream: bool, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext) -> String { String::new() }
/// #   fn choose_chat_transformers(&self, _req: &siumai::types::ChatRequest, _ctx: &ProviderContext)
/// #     -> siumai_core::core::ChatTransformers { unimplemented!() }
/// # }
/// # async fn demo() -> Result<(), siumai::LlmError> {
/// # let http = reqwest::Client::new();
/// # let spec = Arc::new(ExampleSpec);
/// # let ctx = ProviderContext::new("example", "https://api.example.com", None, Default::default());
/// # let config = HttpExecutionConfig { provider_id: "example".into(), http_client: http, provider_spec: spec, provider_context: ctx, interceptors: vec![], retry_options: Some(siumai::retry_api::RetryOptions::default()) };
/// let res = execute_get_binary(&config, "https://api.example.com/file.bin", None).await?;
/// # Ok(()) }
/// ```
pub async fn execute_get_binary(
    config: &HttpExecutionConfig,
    url: &str,
    per_request_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<HttpBinaryResult, LlmError> {
    crate::execution::executors::http_request::execute_get_binary(config, url, per_request_headers)
        .await
}
