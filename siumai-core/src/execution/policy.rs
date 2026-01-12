use std::sync::Arc;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::retry_api::RetryOptions;

/// Unified execution policy for HTTP-based executors.
///
/// Centralizes interceptors, retry options, and JSON before_send hooks
/// so all capabilities (chat/embedding/image/audio/files/rerank) behave
/// consistently.
#[derive(Clone, Default)]
pub struct ExecutionPolicy {
    /// HTTP interceptors (order preserved)
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Retry options (including 401 retry)
    pub retry_options: Option<RetryOptions>,
    /// Optional JSON body transformer (not applied to multipart/bytes)
    pub before_send: Option<crate::execution::executors::BeforeSendHook>,
    /// Streaming-specific: whether to disable compression on streaming requests
    pub stream_disable_compression: bool,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub transport: Option<Arc<dyn HttpTransport>>,
}

impl ExecutionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.interceptors = interceptors;
        self
    }

    pub fn with_retry_options(mut self, retry_options: Option<RetryOptions>) -> Self {
        self.retry_options = retry_options;
        self
    }

    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.before_send = Some(hook);
        self
    }

    pub fn with_stream_disable_compression(mut self, disable: bool) -> Self {
        self.stream_disable_compression = disable;
        self
    }

    pub fn with_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.transport = Some(transport);
        self
    }
}
