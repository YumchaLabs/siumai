//! OpenAI-compatible configuration (protocol layer)
//!
//! This module provides configuration types for OpenAI-compatible providers.

use super::adapter::{
    MetadataExtractingAdapter, ProviderAdapter, RequestBodyTransformer, ResponseMetadataExtractor,
};
use crate::auth::TokenProvider;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::types::{CommonParams, HttpConfig};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Configuration for OpenAI-compatible providers
#[derive(Clone)]
pub struct OpenAiCompatibleConfig {
    /// Provider identifier
    pub provider_id: String,
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the provider
    pub base_url: String,
    /// Model to use
    pub model: String,
    /// Common parameters shared across providers
    pub common_params: CommonParams,
    /// HTTP configuration (timeout, proxy, etc.)
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional Bearer token provider for providers that authenticate with
    /// `Authorization: Bearer <token>` rather than a static API key.
    pub token_provider: Option<Arc<dyn TokenProvider>>,
    /// Custom headers for requests
    pub custom_headers: reqwest::header::HeaderMap,
    /// Provider adapter for handling specifics
    pub adapter: Arc<dyn ProviderAdapter>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional provider-level URL query parameters appended to all compat request routes.
    pub query_params: BTreeMap<String, String>,
    /// Whether streaming chat requests should ask the provider to include usage chunks.
    ///
    /// AI SDK's OpenAI-compatible provider defaults this to `undefined`, which means the request
    /// body omits `stream_options.include_usage` unless callers opt in explicitly.
    pub include_usage: Option<bool>,
    /// Whether compat chat should keep JSON Schema structured outputs instead of downgrading to
    /// `response_format = { "type": "json_object" }`.
    ///
    /// When unset, Siumai keeps the existing permissive behavior for backward compatibility.
    pub supports_structured_outputs: Option<bool>,
    /// Provider-defined tool ids that should not emit the generic compat unsupported warning.
    ///
    /// This is used by providers that tunnel special server-side tools through the shared
    /// OpenAI-compatible runtime but handle warning semantics in provider-owned middleware.
    pub provider_defined_tool_warning_allowlist: BTreeSet<String>,
    /// Optional public request-body transformer, mirroring AI SDK's `transformRequestBody`.
    pub request_body_transformer: Option<Arc<dyn RequestBodyTransformer>>,
}

impl std::fmt::Debug for OpenAiCompatibleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("OpenAiCompatibleConfig");
        ds.field("provider_id", &self.provider_id)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config)
            .field("query_params_len", &self.query_params.len())
            .field("include_usage", &self.include_usage)
            .field(
                "provider_defined_tool_warning_allowlist_len",
                &self.provider_defined_tool_warning_allowlist.len(),
            );
        ds.field(
            "supports_structured_outputs",
            &self.supports_structured_outputs,
        );

        if !self.api_key.is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }
        if self.token_provider.is_some() {
            ds.field("has_token_provider", &true);
        }
        if self.request_body_transformer.is_some() {
            ds.field("has_request_body_transformer", &true);
        }

        ds.finish()
    }
}

impl OpenAiCompatibleConfig {
    fn default_supports_structured_outputs(provider_id: &str) -> Option<bool> {
        match provider_id {
            // These built-in compat presets are expected to preserve JSON Schema outputs on the
            // public path by default rather than falling back to generic `json_object`.
            "openrouter" | "perplexity" | "mistral" => Some(true),
            _ => None,
        }
    }

    /// Create a new configuration
    pub fn new(
        provider_id: &str,
        api_key: &str,
        base_url: &str,
        adapter: Arc<dyn ProviderAdapter>,
    ) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            api_key: api_key.to_string(),
            base_url: super::base_url::normalize_text_base_url(provider_id, base_url),
            model: String::new(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            http_transport: None,
            token_provider: None,
            custom_headers: reqwest::header::HeaderMap::new(),
            adapter,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            query_params: BTreeMap::new(),
            include_usage: None,
            supports_structured_outputs: Self::default_supports_structured_outputs(provider_id),
            provider_defined_tool_warning_allowlist: BTreeSet::new(),
            request_body_transformer: None,
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self.common_params.model = model.to_string();
        self
    }

    /// Set common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Set HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout.
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Control whether to disable compression for streaming requests.
    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    /// Set user agent.
    pub fn with_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Set proxy URL.
    pub fn with_proxy<S: Into<String>>(mut self, proxy: S) -> Self {
        self.http_config.proxy = Some(proxy.into());
        self
    }

    /// Replace HTTP headers stored on the config-level HTTP template.
    pub fn with_http_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.http_config.headers = headers;
        self
    }

    /// Add a single HTTP header to the config-level HTTP template.
    pub fn with_http_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Set an async Bearer token provider.
    pub fn with_token_provider(mut self, token_provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(token_provider);
        self
    }

    /// Install a single HTTP interceptor for requests created by clients built from this config.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Install HTTP interceptors for requests created by clients built from this config.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests created by clients built from this config.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Add a custom header
    pub fn with_header(mut self, key: &str, value: &str) -> Result<Self, LlmError> {
        let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header name: {}", e)))?;
        let header_value = reqwest::header::HeaderValue::from_str(value)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header value: {}", e)))?;

        self.custom_headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top-p.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Merge provider-specific request defaults through the adapter layer.
    pub fn with_provider_specific_config(
        mut self,
        params: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        if params.is_empty() {
            return self;
        }
        let adapter = self.adapter.clone_adapter();
        self.adapter = Arc::from(
            Box::new(super::adapter::ParamMergingAdapter::new(adapter, params))
                as Box<dyn ProviderAdapter>,
        );
        self
    }

    /// Add a single provider-specific request default.
    pub fn with_provider_specific_param(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        let mut params = std::collections::HashMap::new();
        params.insert(key.into(), value);
        self = self.with_provider_specific_config(params);
        self
    }

    /// Install a public response-metadata extractor on top of the current provider adapter.
    ///
    /// This mirrors AI SDK's OpenAI-compatible `metadataExtractor` hook without requiring callers
    /// to reimplement the full provider adapter.
    pub fn with_metadata_extractor(
        mut self,
        extractor: Arc<dyn ResponseMetadataExtractor>,
    ) -> Self {
        let adapter = self.adapter.clone_adapter();
        self.adapter = Arc::from(Box::new(MetadataExtractingAdapter::new(adapter, extractor))
            as Box<dyn ProviderAdapter>);
        self
    }

    /// Control whether streaming chat requests should include `stream_options.include_usage`.
    ///
    /// This mirrors AI SDK's `includeUsage` provider setting for `openai-compatible`.
    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = Some(include_usage);
        self
    }

    /// Alias for `with_include_usage(...)`.
    pub fn include_usage(self, include_usage: bool) -> Self {
        self.with_include_usage(include_usage)
    }

    /// Replace the provider-level URL query parameter map.
    ///
    /// This mirrors AI SDK's `queryParams` provider setting for `openai-compatible`.
    pub fn with_query_params<K, V, I>(mut self, query_params: I) -> Self
    where
        K: Into<String>,
        V: Into<String>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.query_params = query_params
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }

    /// Alias for `with_query_params(...)`.
    pub fn query_params<K, V, I>(self, query_params: I) -> Self
    where
        K: Into<String>,
        V: Into<String>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.with_query_params(query_params)
    }

    /// Insert or replace a single provider-level URL query parameter.
    pub fn with_query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.insert(key.into(), value.into());
        self
    }

    /// Alias for `with_query_param(...)`.
    pub fn query_param(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.with_query_param(key, value)
    }

    /// Control whether compat chat keeps JSON Schema structured outputs.
    ///
    /// `false` mirrors AI SDK's conservative `supportsStructuredOutputs = false` behavior and
    /// downgrades schema requests to `json_object` on the wire.
    pub fn with_supports_structured_outputs(mut self, supports: bool) -> Self {
        self.supports_structured_outputs = Some(supports);
        self
    }

    /// Alias for `with_supports_structured_outputs(...)`.
    pub fn supports_structured_outputs(self, supports: bool) -> Self {
        self.with_supports_structured_outputs(supports)
    }

    /// Install a public request-body transformer for chat requests.
    ///
    /// This mirrors AI SDK's `transformRequestBody` hook and runs after built-in/provider
    /// normalization so callers can customize the final request payload.
    pub fn with_request_body_transformer(
        mut self,
        transformer: Arc<dyn RequestBodyTransformer>,
    ) -> Self {
        self.request_body_transformer = Some(transformer);
        self
    }

    /// Alias for `with_request_body_transformer(...)`.
    pub fn request_body_transformer(self, transformer: Arc<dyn RequestBodyTransformer>) -> Self {
        self.with_request_body_transformer(transformer)
    }

    /// Enable provider-native thinking mode when supported.
    pub fn with_thinking(mut self, enable: bool) -> Self {
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self.with_provider_specific_param(
                    "enable_thinking",
                    serde_json::Value::Bool(enable),
                );
            }
            _ => {
                self = self.with_provider_specific_param(
                    "enable_reasoning",
                    serde_json::Value::Bool(enable),
                );
            }
        }
        self
    }

    /// Set thinking budget for providers that expose it.
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        let clamped_budget = budget.clamp(128, 32768);
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self
                    .with_provider_specific_param(
                        "thinking_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param("enable_thinking", serde_json::Value::Bool(true));
            }
            _ => {
                self = self
                    .with_provider_specific_param(
                        "reasoning_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param(
                        "enable_reasoning",
                        serde_json::Value::Bool(true),
                    );
            }
        }
        self
    }

    /// Unified reasoning toggle.
    pub fn with_reasoning(mut self, enable: bool) -> Self {
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self.with_provider_specific_param(
                    "enable_thinking",
                    serde_json::Value::Bool(enable),
                );
            }
            _ => {
                self = self.with_provider_specific_param(
                    "enable_reasoning",
                    serde_json::Value::Bool(enable),
                );
            }
        }
        self
    }

    /// Unified reasoning budget.
    pub fn with_reasoning_budget(mut self, budget: i32) -> Self {
        let clamped_budget = budget.clamp(128, 32768) as u32;
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self
                    .with_provider_specific_param(
                        "thinking_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param("enable_thinking", serde_json::Value::Bool(true));
            }
            _ => {
                self = self
                    .with_provider_specific_param(
                        "reasoning_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param(
                        "enable_reasoning",
                        serde_json::Value::Bool(true),
                    );
            }
        }
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), LlmError> {
        fn header_map_has_authorization(headers: &reqwest::header::HeaderMap) -> bool {
            headers.contains_key(reqwest::header::AUTHORIZATION)
        }

        fn string_map_has_authorization(
            headers: &std::collections::HashMap<String, String>,
        ) -> bool {
            headers
                .keys()
                .any(|key| key.eq_ignore_ascii_case("authorization"))
        }

        if self.provider_id.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Provider ID cannot be empty".to_string(),
            ));
        }

        if self.api_key.is_empty()
            && self.token_provider.is_none()
            && !header_map_has_authorization(&self.custom_headers)
            && !string_map_has_authorization(&self.http_config.headers)
        {
            return Err(LlmError::ConfigurationError(
                "API key cannot be empty when no Authorization header or token provider is configured"
                    .to_string(),
            ));
        }

        if self.base_url.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Base URL cannot be empty".to_string(),
            ));
        }

        // Validate URL format
        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err(LlmError::ConfigurationError(
                "Base URL must start with http:// or https://".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ProviderCapabilities;
    use std::collections::HashMap;

    #[test]
    fn test_config_creation() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }

            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }

            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }

            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }

            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }

            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }

            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );

        assert_eq!(config.provider_id, "test");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.test.com/v1");
    }

    #[test]
    fn test_deepinfra_config_normalizes_root_base_url_to_openai_text_prefix() {
        let provider = super::super::provider_registry::ProviderConfig {
            id: "deepinfra".to_string(),
            name: "DeepInfra".to_string(),
            base_url: "https://api.deepinfra.com/v1/openai".to_string(),
            field_mappings: super::super::provider_registry::ProviderFieldMappings::default(),
            capabilities: vec![
                "completion".to_string(),
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
            ],
            default_model: Some("meta-llama/Llama-3.3-70B-Instruct".to_string()),
            supports_reasoning: false,
            api_key_env: Some("DEEPINFRA_API_KEY".to_string()),
            api_key_env_aliases: Vec::new(),
        };

        let adapter = Arc::new(super::super::provider_registry::ConfigurableAdapter::new(
            provider,
        ));

        let root = OpenAiCompatibleConfig::new(
            "deepinfra",
            "test-key",
            "https://example.com/deepinfra/v1",
            adapter.clone(),
        );
        let openai = OpenAiCompatibleConfig::new(
            "deepinfra",
            "test-key",
            "https://example.com/deepinfra/v1/openai",
            adapter.clone(),
        );
        let inference = OpenAiCompatibleConfig::new(
            "deepinfra",
            "test-key",
            "https://example.com/deepinfra/v1/inference",
            adapter,
        );

        assert_eq!(root.base_url, "https://example.com/deepinfra/v1/openai");
        assert_eq!(openai.base_url, "https://example.com/deepinfra/v1/openai");
        assert_eq!(
            inference.base_url,
            "https://example.com/deepinfra/v1/openai"
        );
    }

    #[test]
    fn test_config_with_model() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_model("test-model");

        assert_eq!(config.model, "test-model");
    }

    #[test]
    fn test_config_validation() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        // Valid config
        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );
        assert!(config.validate().is_ok());

        // Empty provider ID
        let config = OpenAiCompatibleConfig::new(
            "",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );
        assert!(config.validate().is_err());

        // Empty API key
        let config = OpenAiCompatibleConfig::new(
            "test",
            "",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );
        assert!(config.validate().is_err());

        // Invalid URL
        let config =
            OpenAiCompatibleConfig::new("test", "test-key", "invalid-url", Arc::new(DummyAdapter));
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_with_header() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_header("X-Custom", "test-value")
        .unwrap();

        assert!(config.custom_headers.contains_key("X-Custom"));
    }

    #[test]
    fn test_config_http_convenience_helpers() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_timeout(std::time::Duration::from_secs(15))
        .with_connect_timeout(std::time::Duration::from_secs(3))
        .with_http_stream_disable_compression(true)
        .with_user_agent("siumai-test/1.0")
        .with_proxy("http://127.0.0.1:8080")
        .with_http_headers(std::collections::HashMap::from([(
            "x-one".to_string(),
            "1".to_string(),
        )]))
        .with_http_header("x-two", "2")
        .with_http_interceptor(Arc::new(
            crate::execution::http::interceptor::LoggingInterceptor,
        ));

        assert_eq!(
            config.http_config.timeout,
            Some(std::time::Duration::from_secs(15))
        );
        assert_eq!(
            config.http_config.connect_timeout,
            Some(std::time::Duration::from_secs(3))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(
            config.http_config.user_agent.as_deref(),
            Some("siumai-test/1.0")
        );
        assert_eq!(
            config.http_config.proxy.as_deref(),
            Some("http://127.0.0.1:8080")
        );
        assert_eq!(
            config.http_config.headers.get("x-one"),
            Some(&"1".to_string())
        );
        assert_eq!(
            config.http_config.headers.get("x-two"),
            Some(&"2".to_string())
        );
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn test_config_fluent_common_params_and_reasoning_defaults() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com",
            Arc::new(DummyAdapter),
        )
        .with_model("deepseek-chat")
        .with_temperature(0.4)
        .with_max_tokens(256)
        .with_top_p(0.9)
        .with_stop_sequences(vec!["END".to_string()])
        .with_seed(7)
        .with_reasoning(true)
        .with_reasoning_budget(2048);

        assert_eq!(config.model, "deepseek-chat");
        assert_eq!(config.common_params.model, "deepseek-chat");
        assert_eq!(config.common_params.temperature, Some(0.4));
        assert_eq!(config.common_params.max_tokens, Some(256));
        assert_eq!(config.common_params.top_p, Some(0.9));
        assert_eq!(
            config.common_params.stop_sequences,
            Some(vec!["END".to_string()])
        );
        assert_eq!(config.common_params.seed, Some(7));

        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                super::super::types::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(params["enable_reasoning"], serde_json::json!(true));
        assert_eq!(params["reasoning_budget"], serde_json::json!(2048));
    }

    #[test]
    fn test_config_thinking_budget_maps_for_siliconflow() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            Arc::new(DummyAdapter),
        )
        .with_model("deepseek-ai/DeepSeek-V3.1")
        .with_thinking_budget(8192);

        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                super::super::types::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(params["enable_thinking"], serde_json::json!(true));
        assert_eq!(params["thinking_budget"], serde_json::json!(8192));
    }

    #[test]
    fn test_config_reasoning_maps_for_openrouter() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://openrouter.ai/api/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            Arc::new(DummyAdapter),
        )
        .with_model("openai/gpt-4o")
        .with_reasoning(true)
        .with_reasoning_budget(2048);

        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                super::super::types::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(params["enable_reasoning"], serde_json::json!(true));
        assert_eq!(params["reasoning_budget"], serde_json::json!(2048));
    }

    #[test]
    fn test_config_with_metadata_extractor_wraps_adapter() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let extractor: Arc<dyn ResponseMetadataExtractor> = Arc::new(|raw: &serde_json::Value| {
            raw.get("test_field").map(|value| {
                HashMap::from([(
                    "test-provider".to_string(),
                    serde_json::Value::Object(serde_json::Map::from_iter([(
                        "value".to_string(),
                        value.clone(),
                    )])),
                )])
            })
        });

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_metadata_extractor(extractor);

        let metadata = config
            .adapter
            .extract_response_provider_metadata(&serde_json::json!({
                "test_field": "test-value"
            }))
            .expect("metadata");
        let provider = metadata.get("test-provider").expect("provider metadata");
        assert_eq!(
            provider.get("value"),
            Some(&serde_json::json!("test-value"))
        );
    }

    #[test]
    fn test_config_with_include_usage_records_explicit_setting() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_include_usage(true);

        assert_eq!(config.include_usage, Some(true));
    }

    #[test]
    fn test_config_with_query_params_records_explicit_settings() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_query_params([("api-version", "2025-04-01"), ("tenant", "acme")]);

        assert_eq!(
            config.query_params.get("api-version").map(String::as_str),
            Some("2025-04-01")
        );
        assert_eq!(
            config.query_params.get("tenant").map(String::as_str),
            Some("acme")
        );
    }

    #[test]
    fn test_config_with_supports_structured_outputs_records_policy() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_supports_structured_outputs(false);

        assert_eq!(config.supports_structured_outputs, Some(false));
    }

    #[test]
    fn test_builtin_provider_structured_outputs_defaults() {
        #[derive(Debug, Clone)]
        struct DummyAdapter(&'static str);
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed(self.0)
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let openrouter = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            Arc::new(DummyAdapter("openrouter")),
        );
        let perplexity = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            Arc::new(DummyAdapter("perplexity")),
        );
        let mistral = OpenAiCompatibleConfig::new(
            "mistral",
            "test-key",
            "https://api.mistral.ai/v1",
            Arc::new(DummyAdapter("mistral")),
        );
        let generic = OpenAiCompatibleConfig::new(
            "generic",
            "test-key",
            "https://api.example.com/v1",
            Arc::new(DummyAdapter("generic")),
        );

        assert_eq!(openrouter.supports_structured_outputs, Some(true));
        assert_eq!(perplexity.supports_structured_outputs, Some(true));
        assert_eq!(mistral.supports_structured_outputs, Some(true));
        assert_eq!(generic.supports_structured_outputs, None);
    }

    #[test]
    fn test_config_with_request_body_transformer_stores_hook() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let transformer: Arc<dyn RequestBodyTransformer> =
            Arc::new(|body: &mut serde_json::Value, _model: &str, request_type| {
                assert!(matches!(
                    request_type,
                    super::super::types::RequestType::Chat
                ));
                body["custom"] = serde_json::json!(true);
                Ok(())
            });

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_request_body_transformer(transformer);

        let hook = config
            .request_body_transformer
            .as_ref()
            .expect("request body transformer");
        let mut body = serde_json::json!({});
        hook.transform_request_body(
            &mut body,
            "test-model",
            super::super::types::RequestType::Chat,
        )
        .expect("transform body");
        assert_eq!(body.get("custom"), Some(&serde_json::json!(true)));
    }
}
