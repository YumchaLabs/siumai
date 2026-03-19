//! Azure OpenAI client configuration.
//!
//! This config is intentionally minimal and is expected to be built by higher-level
//! builders (e.g., `siumai-registry` factories) in most cases.

use super::{AzureChatMode, AzureUrlConfig};
use crate::types::{CommonParams, HttpConfig, ProviderOptionsMap};
use std::sync::Arc;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

/// Configuration for Azure OpenAI (OpenAI-compatible endpoints).
#[derive(Clone)]
pub struct AzureOpenAiConfig {
    pub api_key: String,
    /// Base URL prefix, typically `https://{resource}.openai.azure.com/openai`.
    pub base_url: String,
    /// Default model id (Azure deployment id).
    pub common_params: CommonParams,
    /// Default provider options merged before request-local overrides.
    pub provider_options_map: ProviderOptionsMap,
    pub http_config: HttpConfig,
    pub url_config: AzureUrlConfig,
    pub chat_mode: AzureChatMode,
    /// Provider metadata key used by OpenAI Responses mapping.
    pub provider_metadata_key: &'static str,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for AzureOpenAiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAiConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.common_params.model)
            .field("provider_options_map", &self.provider_options_map)
            .field("api_version", &self.url_config.api_version)
            .field(
                "use_deployment_based_urls",
                &self.url_config.use_deployment_based_urls,
            )
            .field("chat_mode", &self.chat_mode)
            .field("provider_metadata_key", &self.provider_metadata_key)
            .field("has_http_transport", &self.http_transport.is_some())
            .finish()
    }
}

impl AzureOpenAiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: String::new(),
            common_params: CommonParams::default(),
            provider_options_map: ProviderOptionsMap::default(),
            http_config: HttpConfig::default(),
            url_config: AzureUrlConfig::default(),
            chat_mode: AzureChatMode::default(),
            provider_metadata_key: "azure",
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub fn with_language_model(self, model: impl Into<String>) -> Self {
        self.with_model(model)
    }

    pub fn with_embedding_model(self, model: impl Into<String>) -> Self {
        self.with_model(model)
    }

    pub fn with_image_model(self, model: impl Into<String>) -> Self {
        self.with_model(model)
    }

    pub fn with_speech_model(self, model: impl Into<String>) -> Self {
        self.with_model(model)
    }

    pub fn with_transcription_model(self, model: impl Into<String>) -> Self {
        self.with_model(model)
    }

    /// Merge default provider options under the Azure provider id.
    pub fn with_provider_options(mut self, options: serde_json::Value) -> Self {
        let mut overrides = ProviderOptionsMap::new();
        overrides.insert("azure", options);
        self.provider_options_map.merge_overrides(overrides);
        self
    }

    /// Merge typed default Azure provider options.
    pub fn with_azure_options(
        self,
        options: crate::provider_options::azure::AzureOpenAiOptions,
    ) -> Self {
        self.with_provider_options(
            serde_json::to_value(options).expect("Azure options should serialize"),
        )
    }

    /// Merge the full default provider options map.
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map.merge_overrides(map);
        self
    }

    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    /// Set request timeout on the canonical config-first HTTP surface.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout on the canonical config-first HTTP surface.
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.url_config.api_version = api_version.into();
        self
    }

    pub fn with_deployment_based_urls(mut self, enabled: bool) -> Self {
        self.url_config.use_deployment_based_urls = enabled;
        self
    }

    pub fn with_url_config(mut self, url_config: AzureUrlConfig) -> Self {
        self.url_config = url_config;
        self
    }

    pub fn with_chat_mode(mut self, mode: AzureChatMode) -> Self {
        self.chat_mode = mode;
        self
    }

    pub fn with_responses_api(self, enabled: bool) -> Self {
        if enabled {
            self.with_chat_mode(AzureChatMode::Responses)
        } else {
            self.with_chat_mode(AzureChatMode::ChatCompletions)
        }
    }

    pub fn with_responses(self) -> Self {
        self.with_chat_mode(AzureChatMode::Responses)
    }

    pub fn with_chat_completions(self) -> Self {
        self.with_chat_mode(AzureChatMode::ChatCompletions)
    }

    pub fn with_provider_metadata_key(mut self, key: &'static str) -> Self {
        self.provider_metadata_key = key;
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Install HTTP interceptors for requests created by clients built from this config.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Append a single HTTP interceptor on the canonical config-first HTTP surface.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
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

    pub fn validate(&self) -> Result<(), crate::error::LlmError> {
        if self.api_key.trim().is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Azure OpenAI api_key cannot be empty".to_string(),
            ));
        }
        if self.base_url.trim().is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Azure OpenAI base_url cannot be empty".to_string(),
            ));
        }
        if self.common_params.model.trim().is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Azure OpenAI model (deployment id) cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::azure::{AzureOpenAiOptions, AzureReasoningEffort};
    use std::{sync::Arc, time::Duration};

    #[test]
    fn azure_openai_config_http_convenience_helpers() {
        let config = AzureOpenAiConfig::new("test-key")
            .with_timeout(Duration::from_secs(9))
            .with_connect_timeout(Duration::from_secs(2))
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(2))
        );
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn azure_config_merges_typed_provider_options() {
        let config = AzureOpenAiConfig::new("test-key")
            .with_azure_options(
                AzureOpenAiOptions::new().with_reasoning_effort(AzureReasoningEffort::Low),
            )
            .with_provider_options(serde_json::json!({
                "responses_api": {
                    "reasoning_summary": "auto"
                }
            }));

        let options = config
            .provider_options_map
            .get("azure")
            .expect("azure options present");
        assert_eq!(options["reasoning_effort"], serde_json::json!("low"));
        assert_eq!(
            options["responses_api"]["reasoning_summary"],
            serde_json::json!("auto")
        );
    }
}
