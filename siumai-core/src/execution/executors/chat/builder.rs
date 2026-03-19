use std::sync::Arc;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};

use super::HttpChatExecutor;

/// Builder for creating HttpChatExecutor instances
///
/// This builder simplifies the creation of HttpChatExecutor by providing
/// a fluent API and reducing code duplication across providers.
///
/// # Example
/// ```rust,ignore
/// let executor = ChatExecutorBuilder::new("openai", http_client)
///     .with_spec(spec)
///     .with_context(ctx)
///     .with_transformers(req_tx, resp_tx, Some(stream_tx))
///     .with_interceptors(interceptors)
///     .with_middlewares(middlewares)
///     .build();
/// ```
pub struct ChatExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    spec: Option<Arc<dyn crate::core::ProviderSpec>>,
    context: Option<crate::core::ProviderContext>,
    defer_transformer_selection: bool,
    request_transformer: Option<Arc<dyn RequestTransformer>>,
    response_transformer: Option<Arc<dyn ResponseTransformer>>,
    stream_transformer: Option<Arc<dyn StreamChunkTransformer>>,
    json_stream_converter: Option<Arc<dyn crate::streaming::JsonEventConverter>>,
    policy: crate::execution::ExecutionPolicy,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl ChatExecutorBuilder {
    /// Create a new builder with required fields
    pub fn new(provider_id: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            spec: None,
            context: None,
            defer_transformer_selection: false,
            request_transformer: None,
            response_transformer: None,
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new(),
            middlewares: Vec::new(),
        }
    }

    /// Set the provider spec
    pub fn with_spec(mut self, spec: Arc<dyn crate::core::ProviderSpec>) -> Self {
        self.spec = Some(spec);
        self
    }

    /// Set the provider context
    pub fn with_context(mut self, context: crate::core::ProviderContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Set the transformers
    pub fn with_transformers(
        mut self,
        request: Arc<dyn RequestTransformer>,
        response: Arc<dyn ResponseTransformer>,
        stream: Option<Arc<dyn StreamChunkTransformer>>,
    ) -> Self {
        self.request_transformer = Some(request);
        self.response_transformer = Some(response);
        self.stream_transformer = stream;
        self
    }

    /// Set transformers from a ChatTransformers bundle
    pub fn with_transformer_bundle(mut self, bundle: crate::core::ChatTransformers) -> Self {
        self.request_transformer = Some(bundle.request);
        self.response_transformer = Some(bundle.response);
        self.stream_transformer = bundle.stream;
        self.json_stream_converter = bundle.json;
        self
    }

    /// Defer transformer selection until after request middlewares run.
    ///
    /// Some providers choose transformer bundles based on provider options that can be
    /// injected or rewritten by model middlewares. Enabling this mode ensures the
    /// provider spec sees the transformed request before picking request/response/stream
    /// transformers.
    pub fn with_runtime_transformer_selection(mut self) -> Self {
        self.defer_transformer_selection = true;
        self
    }

    /// Set the JSON stream converter
    pub fn with_json_converter(
        mut self,
        converter: Arc<dyn crate::streaming::JsonEventConverter>,
    ) -> Self {
        self.json_stream_converter = Some(converter);
        self
    }

    /// Set whether to disable compression for streaming
    pub fn with_stream_disable_compression(mut self, disable: bool) -> Self {
        self.policy.stream_disable_compression = disable;
        self
    }

    /// Set HTTP interceptors
    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.policy.interceptors = interceptors;
        self
    }

    /// Set model-level middlewares
    pub fn with_middlewares(mut self, middlewares: Vec<Arc<dyn LanguageModelMiddleware>>) -> Self {
        self.middlewares = middlewares;
        self
    }

    /// Set the before_send hook
    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.policy.before_send = Some(hook);
        self
    }

    /// Set retry options
    pub fn with_retry_options(mut self, retry_options: crate::retry_api::RetryOptions) -> Self {
        self.policy.retry_options = Some(retry_options);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.policy.transport = Some(transport);
        self
    }

    /// Build the HttpChatExecutor
    ///
    /// # Panics
    /// Panics if required fields (spec, context, transformers) are not set
    pub fn build(self) -> Arc<HttpChatExecutor> {
        Arc::new(HttpChatExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            request_transformer: self.request_transformer,
            response_transformer: self.response_transformer,
            stream_transformer: self.stream_transformer,
            json_stream_converter: self.json_stream_converter,
            defer_transformer_selection: self.defer_transformer_selection,
            policy: self.policy,
            middlewares: self.middlewares,
            provider_spec: self.spec.expect("provider_spec is required"),
            provider_context: self.context.expect("provider_context is required"),
        })
    }
}
