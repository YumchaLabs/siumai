//! `OpenAI` Client Implementation
//!
//! Main client structure that aggregates all `OpenAI` capabilities.

use secrecy::ExposeSecret;
use std::sync::Arc;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::params::OpenAiParams;
use crate::traits::*;
use crate::types::*;

use super::models::OpenAiModels;
use super::rerank::OpenAiRerank;
use super::types::OpenAiSpecificParams;
use super::utils::get_default_models;
// use crate::execution::executors::chat::ChatExecutor; // not used directly
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;

// Split capability implementations into focused submodules (no API change)
mod audio;
mod chat;
mod embedding;
mod files;
mod image;
mod models;
mod moderation;
mod rerank;
mod responses_admin;
mod speech_streaming;
pub(crate) mod transcription_streaming;

/// `OpenAI` Client
pub struct OpenAiClient {
    /// API key and endpoint configuration
    api_key: secrecy::SecretString,
    base_url: String,
    organization: Option<String>,
    project: Option<String>,
    http_config: HttpConfig,
    /// Models capability implementation
    models_capability: OpenAiModels,
    /// Rerank capability implementation
    rerank_capability: OpenAiRerank,
    /// Common parameters
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
    /// OpenAI-specific configuration
    specific_params: OpenAiSpecificParams,
    /// Optional forced Responses API configuration (client-level default).
    forced_responses_api: Option<crate::types::provider_options::openai::ResponsesApiConfig>,
    /// HTTP client for making requests
    http_client: reqwest::Client,
    /// Tracing configuration
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (not cloned)
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors:
        Vec<std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping
    model_middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
}

impl Clone for OpenAiClient {
    fn clone(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            http_config: self.http_config.clone(),
            models_capability: self.models_capability.clone(),
            rerank_capability: self.rerank_capability.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            specific_params: self.specific_params.clone(),
            forced_responses_api: self.forced_responses_api.clone(),
            http_client: self.http_client.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
            model_middlewares: Vec::new(),
        }
    }
}

impl std::fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("OpenAiClient");

        debug_struct
            .field("provider_id", &"openai")
            .field("model", &self.common_params.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field("has_tracing", &self.tracing_config.is_some());

        // Only show organization/project if they exist (but don't show the actual values)
        if self.specific_params.organization.is_some() {
            debug_struct.field("has_organization", &true);
        }
        if self.specific_params.project.is_some() {
            debug_struct.field("has_project", &true);
        }

        debug_struct.finish()
    }
}

impl OpenAiClient {
    /// Get the base URL used by this client.
    ///
    /// This is primarily intended for debugging and tests. The value
    /// reflects the final base URL after any builder / registry defaults
    /// and overrides have been applied.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Convenience: configure structured outputs using a JSON object schema.
    /// Only applied to Chat Completions requests (`POST /chat/completions`).
    /// For the Responses API, use `OpenAiOptions.responses_api.response_format` (maps to `text.format`).
    pub fn with_json_object_schema(mut self, schema: serde_json::Value, strict: bool) -> Self {
        let fmt = serde_json::json!({
            "type": "json_object",
            "json_schema": {
                "schema": schema,
                "strict": strict
            }
        });
        self.specific_params.response_format = Some(fmt);
        self
    }

    /// Convenience: configure structured outputs using a named JSON schema.
    /// Only applied to Chat Completions requests (`POST /chat/completions`).
    /// For the Responses API, use `OpenAiOptions.responses_api.response_format` (maps to `text.format`).
    pub fn with_json_named_schema<S: Into<String>>(
        mut self,
        name: S,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        let fmt = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": name.into(),
                "schema": schema,
                "strict": strict
            }
        });
        self.specific_params.response_format = Some(fmt);
        self
    }

    /// Creates a new `OpenAI` client with configuration and HTTP client
    pub fn new(config: super::OpenAiConfig, http_client: reqwest::Client) -> Self {
        let mut specific_params = OpenAiSpecificParams {
            organization: config.organization.clone(),
            project: config.project.clone(),
            ..Default::default()
        };

        // Backward-compat: map legacy `OpenAiParams` into the per-client defaults
        // so builder methods like `.response_format()` keep working.
        if let Some(fmt) = config.openai_params.response_format.clone()
            && let Ok(v) = serde_json::to_value(fmt)
        {
            specific_params.response_format = Some(v);
        }
        if let Some(bias) = config.openai_params.logit_bias.clone()
            && let Ok(v) = serde_json::to_value(bias)
        {
            specific_params.logit_bias = Some(v);
        }
        specific_params.logprobs = config.openai_params.logprobs;
        specific_params.top_logprobs = config.openai_params.top_logprobs;
        specific_params.presence_penalty = config.openai_params.presence_penalty;
        specific_params.frequency_penalty = config.openai_params.frequency_penalty;
        specific_params.user = config.openai_params.user.clone();

        let models_capability = OpenAiModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );

        let rerank_capability = OpenAiRerank::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );

        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            organization: config.organization,
            project: config.project,
            http_config: config.http_config,
            models_capability,
            rerank_capability,
            common_params: config.common_params,
            openai_params: config.openai_params,
            specific_params,
            forced_responses_api: None,
            http_client,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(
        &mut self,
        config: Option<crate::observability::tracing::TracingConfig>,
    ) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Force routing chat requests through the OpenAI Responses API (`POST /responses`).
    pub fn set_forced_responses_api(
        &mut self,
        cfg: Option<crate::types::provider_options::openai::ResponsesApiConfig>,
    ) {
        self.forced_responses_api = cfg;
    }

    /// Creates a new `OpenAI` client with configuration (for OpenAI-compatible providers)
    pub fn new_with_config(config: super::OpenAiConfig) -> Self {
        let http_client = reqwest::Client::new();
        Self::new(config, http_client)
    }

    /// Helper: Build ProviderContext with OpenAI-specific extras
    fn build_context(&self) -> crate::core::ProviderContext {
        let mut extras = std::collections::HashMap::new();
        if let Some(fmt) = &self.specific_params.response_format {
            extras.insert("openai.response_format".to_string(), fmt.clone());
        }

        crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone())
        .with_extras(extras)
    }

    /// Helper: Build ChatExecutor with common configuration
    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context();
        let mut spec = crate::providers::openai::spec::OpenAiSpec::new();
        if let Some(cfg) = self.forced_responses_api.clone() {
            spec = spec.with_forced_responses_api(cfg);
        }
        let spec = Arc::new(spec);
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        // Provider-client defaults.
        //
        // - Chat Completions structured output uses `response_format` (Chat schema).
        // - Responses structured output uses `text.format` and is injected by `OpenAiSpec`
        //   via `OpenAiOptions.responses_api.response_format`.
        let use_responses_api = spec
            .chat_url(false, request, &ctx)
            .trim_end()
            .ends_with("/responses");
        let defaults = self.specific_params.clone();
        let openai_params = self.openai_params.clone();
        let client_defaults_hook: Option<crate::execution::executors::BeforeSendHook> =
            Some(Arc::new(move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            let Some(obj) = out.as_object_mut() else {
                return Ok(out);
            };

            // Cross-endpoint defaults (supported by both /chat/completions and /responses).
            if obj.get("tool_choice").is_none()
                && let Some(tc) = openai_params.tool_choice.clone()
                && let Ok(v) = serde_json::to_value(tc)
            {
                obj.insert("tool_choice".to_string(), v);
            }
            if obj.get("parallel_tool_calls").is_none()
                && let Some(v) = openai_params.parallel_tool_calls
            {
                obj.insert("parallel_tool_calls".to_string(), serde_json::Value::Bool(v));
            }
            if obj.get("store").is_none()
                && let Some(v) = openai_params.store
            {
                obj.insert("store".to_string(), serde_json::Value::Bool(v));
            }
            if obj.get("metadata").is_none()
                && let Some(meta) = openai_params.metadata.clone()
                && let Ok(v) = serde_json::to_value(meta)
            {
                obj.insert("metadata".to_string(), v);
            }
            if obj.get("prompt_cache_key").is_none()
                && let Some(v) = openai_params.prompt_cache_key.clone()
            {
                obj.insert("prompt_cache_key".to_string(), serde_json::Value::String(v));
            }
            if obj.get("service_tier").is_none()
                && let Some(ref tier) = openai_params.service_tier
                && let Ok(v) = serde_json::to_value(tier)
            {
                obj.insert("service_tier".to_string(), v);
            }
            if obj.get("top_logprobs").is_none()
                && let Some(v) = openai_params.top_logprobs
            {
                obj.insert("top_logprobs".to_string(), serde_json::json!(v));
            }

            if !use_responses_api {
                // Chat Completions-only defaults.
                if obj.get("n").is_none()
                    && let Some(v) = openai_params.n
                {
                    obj.insert("n".to_string(), serde_json::json!(v));
                }

                if obj.get("response_format").is_none()
                    && let Some(v) = defaults.response_format.clone()
                {
                    obj.insert("response_format".to_string(), v);
                }
                if obj.get("logit_bias").is_none()
                    && let Some(v) = defaults.logit_bias.clone()
                {
                    obj.insert("logit_bias".to_string(), v);
                }
                if obj.get("logprobs").is_none()
                    && let Some(v) = defaults.logprobs
                {
                    obj.insert("logprobs".to_string(), serde_json::Value::Bool(v));
                }
                if obj.get("top_logprobs").is_none()
                    && let Some(v) = defaults.top_logprobs
                {
                    obj.insert("top_logprobs".to_string(), serde_json::json!(v));
                }
                if obj.get("presence_penalty").is_none()
                    && let Some(v) = defaults.presence_penalty
                {
                    obj.insert("presence_penalty".to_string(), serde_json::json!(v));
                }
                if obj.get("frequency_penalty").is_none()
                    && let Some(v) = defaults.frequency_penalty
                {
                    obj.insert("frequency_penalty".to_string(), serde_json::json!(v));
                }
                if obj.get("user").is_none()
                    && let Some(v) = defaults.user.clone()
                {
                    obj.insert("user".to_string(), serde_json::Value::String(v));
                }
            }

            Ok(out)
        }));

        let before_send_hook = match (client_defaults_hook, before_send_hook) {
            (None, None) => None,
            (Some(h), None) | (None, Some(h)) => Some(h),
            (Some(defaults), Some(spec_hook)) => {
                let hook: crate::execution::executors::BeforeSendHook = Arc::new(
                    move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                        let out = defaults(body)?;
                        spec_hook(&out)
                    },
                );
                Some(hook)
            }
        };

        let mut builder = ChatExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    /// Helper: Build EmbeddingExecutor with common configuration
    fn build_embedding_executor(
        &self,
        request: &EmbeddingRequest,
    ) -> Arc<crate::execution::executors::embedding::HttpEmbeddingExecutor> {
        use crate::execution::executors::embedding::EmbeddingExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

        let mut builder = EmbeddingExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build_for_request(request)
    }

    /// Helper: Build ImageExecutor with common configuration
    fn build_image_executor(
        &self,
        request: &ImageGenerationRequest,
    ) -> Arc<crate::execution::executors::image::HttpImageExecutor> {
        use crate::execution::executors::image::ImageExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

        let mut builder = ImageExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build_for_request(request)
    }

    /// Helper: Build AudioExecutor with common configuration
    fn build_audio_executor(&self) -> Arc<crate::execution::executors::audio::HttpAudioExecutor> {
        use crate::execution::executors::audio::AudioExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

        let mut builder = AudioExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    /// Get OpenAI-specific parameters
    pub const fn specific_params(&self) -> &OpenAiSpecificParams {
        &self.specific_params
    }

    /// Get common parameters (primarily for testing and debugging)
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Install HTTP interceptors for all chat requests
    pub fn with_http_interceptors(
        mut self,
        interceptors: Vec<std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    ) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    // chat_capability removed after executors migration

    /// Update OpenAI-specific parameters
    pub fn with_specific_params(mut self, params: OpenAiSpecificParams) -> Self {
        self.specific_params = params;
        self
    }

    /// Set organization
    pub fn with_organization(mut self, organization: String) -> Self {
        self.specific_params.organization = Some(organization);
        self
    }

    /// Set project
    pub fn with_project(mut self, project: String) -> Self {
        self.specific_params.project = Some(project);
        self
    }

    /// Set response format for structured output
    pub fn with_response_format(mut self, format: serde_json::Value) -> Self {
        self.specific_params.response_format = Some(format);
        self
    }

    /// Set logit bias
    pub fn with_logit_bias(mut self, bias: serde_json::Value) -> Self {
        self.specific_params.logit_bias = Some(bias);
        self
    }

    /// Enable logprobs
    pub const fn with_logprobs(mut self, enabled: bool, top_logprobs: Option<u32>) -> Self {
        self.specific_params.logprobs = Some(enabled);
        self.specific_params.top_logprobs = top_logprobs;
        self
    }

    /// Set presence penalty
    pub const fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.specific_params.presence_penalty = Some(penalty);
        self
    }

    /// Set frequency penalty
    pub const fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.specific_params.frequency_penalty = Some(penalty);
        self
    }

    /// Set user identifier
    pub fn with_user(mut self, user: String) -> Self {
        self.specific_params.user = Some(user);
        self
    }
}

impl LlmProvider for OpenAiClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("openai")
    }

    fn supported_models(&self) -> Vec<String> {
        get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_audio()
            .with_embedding()
            .with_image_generation()
            .with_file_management()
            .with_custom_feature("moderation", true)
            .with_custom_feature("model_listing", true)
            .with_custom_feature("structured_output", true)
            .with_custom_feature("batch_processing", true)
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

impl LlmClient for OpenAiClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        LlmProvider::provider_id(self)
    }

    fn supported_models(&self) -> Vec<String> {
        LlmProvider::supported_models(self)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        LlmProvider::capabilities(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        // Provide audio via executor-backed implementation
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        Some(self)
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        Some(self)
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        // Return the image generation capability
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn crate::traits::ImageExtras> {
        Some(self)
    }

    fn as_file_management_capability(
        &self,
    ) -> Option<&dyn crate::traits::FileManagementCapability> {
        Some(self)
    }

    fn as_moderation_capability(&self) -> Option<&dyn crate::traits::ModerationCapability> {
        Some(self)
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
    use crate::execution::transformers::request::RequestTransformer;
    use crate::providers::openai::OpenAiConfig;
    use crate::providers::openai::transformers;
    use std::sync::{Arc, Mutex};

    // Local helpers to construct provider-defined tools for tests without depending
    // on the `siumai::hosted_tools` helper module.
    fn web_search_tool() -> crate::types::Tool {
        crate::types::Tool::provider_defined("openai.web_search", "web_search")
    }

    fn file_search_tool_with(
        max_num_results: Option<u32>,
        ranking: Option<(&str, f64)>,
    ) -> crate::types::Tool {
        use serde_json::json;

        let mut args = json!({
            "vector_store_ids": ["vs1"],
        });

        if let Some(max) = max_num_results {
            args["max_num_results"] = json!(max);
        }
        if let Some((ranker, threshold)) = ranking {
            args["ranking_options"] = json!({
                "ranker": ranker,
                "score_threshold": threshold,
            });
        }

        crate::types::Tool::provider_defined("openai.file_search", "file_search").with_args(args)
    }

    #[test]
    fn test_openai_client_creation() {
        let config = OpenAiConfig::new("test-key");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        assert_eq!(
            LlmProvider::provider_id(&client),
            std::borrow::Cow::Borrowed("openai")
        );
        assert!(!LlmProvider::supported_models(&client).is_empty());
    }

    #[test]
    fn test_openai_client_with_specific_params() {
        let config = OpenAiConfig::new("test-key")
            .with_organization("org-123")
            .with_project("proj-456");
        let client = OpenAiClient::new(config, reqwest::Client::new())
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.3);

        assert_eq!(
            client.specific_params().organization,
            Some("org-123".to_string())
        );
        assert_eq!(
            client.specific_params().project,
            Some("proj-456".to_string())
        );
        assert_eq!(client.specific_params().presence_penalty, Some(0.5));
        assert_eq!(client.specific_params().frequency_penalty, Some(0.3));
    }

    #[test]
    fn test_openai_client_uses_builder_model() {
        let config = OpenAiConfig::new("test-key").with_model("gpt-4");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        // Verify that the client stores the model from the builder
        assert_eq!(client.common_params.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_openai_chat_request_uses_client_model() {
        use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};

        let config = OpenAiConfig::new("test-key").with_model("gpt-4-test");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        // Create a test message
        let message = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: MessageMetadata::default(),
        };

        // Create a ChatRequest to test the legacy chat method
        let request = ChatRequest::builder()
            .messages(vec![message])
            .common_params(client.common_params.clone())
            .build();

        // Test that the request body includes the correct model (via transformers)
        let tx = transformers::OpenAiRequestTransformer;
        let body = tx.transform_chat(&request).unwrap();
        assert_eq!(body["model"], "gpt-4-test");
    }

    #[test]
    fn responses_builtins_and_previous_id_injected_non_stream() {
        // Build config with basic settings
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        // Interceptor to capture transformed JSON body and abort before HTTP send
        struct Capture(Arc<Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = Arc::new(Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![Arc::new(cap)]);

        // Create request with provider_options for Responses API
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_provider_tool(web_search_tool())
                        .with_responses_api(
                            ResponsesApiConfig::new()
                                .with_previous_response("resp_123".to_string()),
                        ),
                );

        // Invoke non-stream chat, which should hit interceptor and abort
        let err = futures::executor::block_on(client.chat_request(request)).unwrap_err();
        match err {
            LlmError::InvalidParameter(s) => assert_eq!(s, "stop"),
            other => panic!("unexpected: {:?}", other),
        }

        // Assert captured body contains previous_response_id and built-in tool
        let body = captured.lock().unwrap().clone().expect("captured body");
        assert_eq!(
            body.get("previous_response_id")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "resp_123"
        );
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        assert!(
            tools
                .iter()
                .any(|t| t.get("type").and_then(|s| s.as_str()) == Some("web_search"))
        );
    }

    #[test]
    fn responses_builtins_dedup_non_stream() {
        // Duplicate built-ins should be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with duplicate built-in tools
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_provider_tool(web_search_tool())
                        .with_provider_tool(web_search_tool())
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let web_count = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("web_search"))
            .count();
        assert_eq!(web_count, 1, "duplicate built-ins must be deduplicated");
    }

    #[test]
    fn responses_file_search_key_includes_max_num_results() {
        // Two file_search entries with same ids but different max_num_results should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with two file_search tools with different max_num_results
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_provider_tool(file_search_tool_with(Some(10), None))
                        .with_provider_tool(file_search_tool_with(Some(20), None))
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(
            files.len(),
            2,
            "file_search entries with different max_num_results must both remain"
        );
        let mut maxes: Vec<u64> = files
            .iter()
            .filter_map(|t| t.get("max_num_results").and_then(|v| v.as_u64()))
            .collect();
        maxes.sort();
        assert_eq!(maxes, vec![10, 20]);
    }

    #[test]
    fn responses_file_search_dedup_respects_ranking_options() {
        // Two file_search entries with same ids and max_num_results but different ranking_options should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with two file_search tools with different ranking_options
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_provider_tool(file_search_tool_with(
                            Some(10),
                            Some(("semantic", 0.6)),
                        ))
                        .with_provider_tool(file_search_tool_with(Some(10), Some(("bm25", 0.2))))
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(
            files.len(),
            2,
            "file_search with different ranking_options must both remain"
        );
    }

    #[test]
    fn responses_file_search_dedup_respects_max_num_results() {
        // Two file_search entries with same ids but different max_num_results should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with two file_search tools with different max_num_results
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_provider_tool(file_search_tool_with(Some(10), None))
                        .with_provider_tool(file_search_tool_with(Some(5), None))
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(
            files.len(),
            2,
            "file_search with different max_num_results must both remain"
        );
    }

    #[test]
    fn responses_response_format_injected_non_stream() {
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with response_format in provider_options
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let response_format = serde_json::json!({
            "type": "json_object"
        });
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(OpenAiOptions::new().with_responses_api(
                    ResponsesApiConfig::new().with_response_format(response_format),
                ));

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body
            .get("text")
            .and_then(|t| t.get("format"))
            .cloned()
            .expect("has text.format");
        assert_eq!(
            rf.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "json_object"
        );
    }

    #[test]
    fn responses_response_format_injected_stream() {
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with named schema response_format
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let response_format = serde_json::json!({
            "type": "json_schema",
            "name": "User",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"]
            }
        });
        let mut request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(OpenAiOptions::new().with_responses_api(
                    ResponsesApiConfig::new().with_response_format(response_format),
                ));
        request.stream = true;

        // trigger stream path
        let _ = futures::executor::block_on(client.chat_stream_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body
            .get("text")
            .and_then(|t| t.get("format"))
            .cloned()
            .expect("has text.format");
        assert_eq!(
            rf.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "json_schema"
        );
        let name = rf
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(name, "User");
    }

    #[test]
    fn test_responses_api_extended_params() {
        // Test that all new ResponsesApiConfig parameters are correctly injected
        let config = OpenAiConfig::new("test-key")
            .with_base_url("https://api.openai.com/v1")
            .with_model("gpt-4o");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        struct Capture(Arc<Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _builder: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = Arc::new(Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![Arc::new(cap)]);

        // Create request with all extended ResponsesApiConfig parameters
        use crate::types::{OpenAiOptions, ResponsesApiConfig, TextVerbosity, Truncation};
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("user_id".to_string(), "test_123".to_string());

        let request = crate::types::ChatRequest::new(vec![
            crate::types::ChatMessage::user("Test message").build(),
        ])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_background(true)
                    .with_include(vec!["file_search_call.results".to_string()])
                    .with_instructions("You are a helpful assistant".to_string())
                    .with_max_tool_calls(10)
                    .with_store(false)
                    .with_truncation(Truncation::Auto)
                    .with_text_verbosity(TextVerbosity::Medium)
                    .with_metadata(metadata.clone())
                    .with_parallel_tool_calls(true),
            ),
        );

        // Trigger chat path
        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");

        // Verify all parameters are injected
        assert_eq!(body.get("background").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            body.get("include")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str()),
            Some("file_search_call.results")
        );
        assert_eq!(
            body.get("instructions").and_then(|v| v.as_str()),
            Some("You are a helpful assistant")
        );
        assert_eq!(
            body.get("max_tool_calls").and_then(|v| v.as_u64()),
            Some(10)
        );
        assert_eq!(body.get("store").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(
            body.get("truncation").and_then(|v| v.as_str()),
            Some("auto")
        );

        // text_verbosity should be nested under "text.verbosity"
        assert_eq!(
            body.get("text")
                .and_then(|t| t.get("verbosity"))
                .and_then(|v| v.as_str()),
            Some("medium")
        );

        assert_eq!(
            body.get("parallel_tool_calls").and_then(|v| v.as_bool()),
            Some(true)
        );

        // Verify metadata
        let meta = body.get("metadata").expect("has metadata");
        assert_eq!(
            meta.get("user_id").and_then(|v| v.as_str()),
            Some("test_123")
        );
    }

    #[tokio::test]
    async fn responses_non_stream_parses_function_call_output_item() {
        use crate::traits::ChatCapability;
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_1",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": "{\"location\":\"tokyo\"}"
                    },
                    {
                        "type": "message",
                        "content": [{"type":"output_text","text":"Done."}]
                    }
                ],
                "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 },
                "finish_reason": "function_call"
            })))
            .mount(&server)
            .await;

        let cfg = OpenAiConfig::new("test-key")
            .with_base_url(format!("{}/v1", server.uri()))
            .with_model("gpt-4o-mini");
        let client = OpenAiClient::new(cfg, reqwest::Client::new());

        let request = crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

        let resp = client.chat_request(request).await.unwrap();
        assert_eq!(resp.content_text(), Some("Done."));

        let parts = resp.content.as_multimodal().expect("expected multimodal");
        let tool = parts
            .iter()
            .find_map(|p| {
                if let crate::types::ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    provider_executed,
                } = p
                {
                    Some((tool_call_id, tool_name, arguments, provider_executed))
                } else {
                    None
                }
            })
            .expect("tool call present");

        assert_eq!(tool.0, "call_1");
        assert_eq!(tool.1, "get_weather");
        assert!(tool.3.is_none()); // user-defined function tool call
        assert_eq!(tool.2["location"], serde_json::json!("tokyo"));
    }

    #[tokio::test]
    async fn responses_stream_function_call_delta_uses_call_id() {
        use crate::traits::ChatCapability;
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        use futures_util::StreamExt;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        let sse = concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"item_abc\",\"output_index\":0,\"delta\":\"{\\\"location\\\":\\\"tokyo\\\"}\",\"sequence_number\":1}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o-mini\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"tokyo\\\"}\"},{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Done.\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3},\"finish_reason\":\"function_call\"}}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(sse, "text/event-stream"),
            )
            .mount(&server)
            .await;

        let cfg = OpenAiConfig::new("test-key")
            .with_base_url(format!("{}/v1", server.uri()))
            .with_model("gpt-4o-mini");
        let client = OpenAiClient::new(cfg, reqwest::Client::new());

        let mut request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));
        request.stream = true;

        let mut stream = client.chat_stream_request(request).await.unwrap();

        let mut saw_added = false;
        let mut saw_args_delta = false;

        while let Some(item) = stream.next().await {
            let ev = item.unwrap();
            match ev {
                crate::streaming::ChatStreamEvent::ToolCallDelta {
                    id,
                    function_name,
                    arguments_delta,
                    ..
                } => {
                    // Both the "item added" and the "arguments delta" should use call_id (call_1).
                    assert_eq!(id, "call_1");
                    if function_name.as_deref() == Some("get_weather") && arguments_delta.is_none() {
                        saw_added = true;
                    }
                    if function_name.is_none()
                        && arguments_delta.as_deref() == Some("{\"location\":\"tokyo\"}")
                    {
                        saw_args_delta = true;
                    }
                }
                crate::streaming::ChatStreamEvent::StreamEnd { response } => {
                    assert_eq!(response.content_text(), Some("Done."));
                    break;
                }
                _ => {}
            }
        }

        assert!(saw_added);
        assert!(saw_args_delta);
    }
}
