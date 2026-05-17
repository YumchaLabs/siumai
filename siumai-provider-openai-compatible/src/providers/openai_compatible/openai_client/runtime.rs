use super::super::openai_config::OpenAiCompatibleConfig;
use super::OpenAiCompatibleClient;
use crate::client::LlmClient;
use crate::core::ProviderContext;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::providers::openai_compatible::middleware::{
    OpenAiCompatibleAlibabaCacheControlWarningMiddleware,
    OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware,
    OpenAiCompatibleStructuredOutputsWarningMiddleware, OpenAiCompatibleToolWarningsMiddleware,
};
use crate::retry_api::RetryOptions;
use crate::standards::openai::compat::adapter::OpenAiCompatibleRequestSettings;
use crate::standards::openai::compat::provider_registry::ConfigurableAdapter;
use crate::types::{CommonParams, HttpConfig};
use std::sync::Arc;

pub(crate) fn model_slot_is_missing(model: Option<&str>) -> bool {
    match model {
        Some(value) => value.trim().is_empty(),
        None => true,
    }
}

pub(crate) const DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING: &str =
    "The 'openai-compatible' key in providerOptions is deprecated. Use 'openaiCompatible' instead.";

impl std::fmt::Debug for OpenAiCompatibleClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleClient")
            .field("provider_id", &self.config.provider_id)
            .field("model", &self.config.model)
            .field("base_url", &self.config.base_url)
            .field("has_api_key", &(!self.config.api_key.is_empty()))
            .field("has_retry", &self.retry_options.is_some())
            .field("interceptors", &self.http_interceptors.len())
            .field("middlewares", &self.model_middlewares.len())
            .finish()
    }
}

fn compat_model_middlewares(
    config: &OpenAiCompatibleConfig,
) -> Vec<Arc<dyn LanguageModelMiddleware>> {
    let mut middlewares = config.model_middlewares.clone();

    middlewares.push(Arc::new(
        OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware::new(),
    ));

    if config.supports_structured_outputs != Some(true) {
        middlewares.push(Arc::new(
            OpenAiCompatibleStructuredOutputsWarningMiddleware::new(),
        ));
    }

    if crate::standards::openai::compat::alibaba_cache_control::supports_alibaba_cache_control(
        &config.provider_id,
    ) {
        middlewares.push(Arc::new(
            OpenAiCompatibleAlibabaCacheControlWarningMiddleware::new(config.provider_id.clone()),
        ));
    }

    middlewares.push(Arc::new(
        OpenAiCompatibleToolWarningsMiddleware::for_provider(config.provider_id.clone())
            .with_allowlist(
                config
                    .provider_defined_tool_warning_allowlist
                    .iter()
                    .cloned(),
            ),
    ));

    middlewares
}

impl OpenAiCompatibleClient {
    fn primary_default_model(&self) -> Option<&'static str> {
        super::super::default_models::get_default_chat_model(&self.config.provider_id)
    }

    pub(super) fn resolve_family_model_or_config(
        &self,
        family_default: Option<&'static str>,
    ) -> Option<String> {
        let configured_model = self.config.model.trim();
        if configured_model.is_empty() {
            return family_default.map(str::to_string);
        }

        if self
            .primary_default_model()
            .is_some_and(|default_model| default_model == configured_model)
        {
            return family_default
                .map(str::to_string)
                .or_else(|| Some(self.config.model.clone()));
        }

        Some(self.config.model.clone())
    }

    fn build_base_context(&self) -> ProviderContext {
        // Merge custom headers from HttpConfig + config.custom_headers + adapter.custom_headers
        let mut extra_headers: std::collections::HashMap<String, String> =
            self.config.http_config.headers.clone();
        let cfg_map =
            crate::execution::http::headers::headermap_to_hashmap(&self.config.custom_headers);
        extra_headers.extend(cfg_map);
        let adapter_map = crate::execution::http::headers::headermap_to_hashmap(
            &self.config.adapter.custom_headers(),
        );
        extra_headers.extend(adapter_map);

        ProviderContext::new(
            self.config.provider_id.clone(),
            self.config.base_url.clone(),
            if self.config.api_key.is_empty() {
                None
            } else {
                Some(self.config.api_key.clone())
            },
            extra_headers,
        )
    }

    pub(super) async fn build_context(&self) -> Result<ProviderContext, LlmError> {
        let mut ctx = self.build_base_context();

        let has_auth_header = ctx
            .http_extra_headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("authorization"));

        if !has_auth_header && let Some(token_provider) = &self.config.token_provider {
            let token = token_provider.token().await?;
            ctx.http_extra_headers
                .insert("Authorization".to_string(), format!("Bearer {token}"));
        }

        Ok(ctx)
    }

    pub(super) fn ensure_completion_surface(&self, stream: bool) -> Result<(), LlmError> {
        let caps = self.capabilities();
        if !caps.supports("completion") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support completions",
                self.config.provider_id
            )));
        }
        if stream && !caps.supports("streaming") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support completion streaming",
                self.config.provider_id
            )));
        }

        Ok(())
    }

    pub(super) fn http_wiring(
        &self,
        ctx: ProviderContext,
    ) -> crate::execution::wiring::HttpExecutionWiring {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            self.config.provider_id.clone(),
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring
    }

    /// Build the provider execution context (headers/base_url/api key + extra headers).
    ///
    /// This is primarily used by hybrid providers (e.g. Groq) that reuse the OpenAI-compatible
    /// client for chat/embedding/image but need to invoke non-chat executors with the same
    /// HTTP wiring (client, interceptors, retry).
    pub fn provider_context(&self) -> ProviderContext {
        self.build_base_context()
    }

    /// Clone the underlying `reqwest::Client`.
    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    /// Clone the installed unified retry options.
    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.retry_options.clone()
    }

    /// Clone the installed HTTP interceptors.
    pub fn http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        self.http_interceptors.clone()
    }

    /// Clone the installed custom HTTP transport, if present.
    ///
    /// This is primarily used by hybrid providers (e.g. Groq) that reuse the OpenAI-compatible
    /// client for most capabilities but still need to invoke spec-driven executors with the
    /// same transport wiring.
    pub fn http_transport(
        &self,
    ) -> Option<Arc<dyn crate::execution::http::transport::HttpTransport>> {
        self.config.http_transport.clone()
    }

    /// Clone the config-level common params template.
    pub fn common_params(&self) -> CommonParams {
        self.config.common_params.clone()
    }

    /// Clone the config-level HTTP config template.
    pub fn http_config(&self) -> HttpConfig {
        self.config.http_config.clone()
    }

    /// Clone the provider adapter.
    pub fn adapter(&self) -> Arc<dyn crate::providers::openai_compatible::ProviderAdapter> {
        self.config.adapter.clone()
    }

    pub(super) fn request_settings(&self) -> OpenAiCompatibleRequestSettings {
        OpenAiCompatibleRequestSettings {
            query_params: self.config.query_params.clone(),
            include_usage: self.config.include_usage,
            supports_structured_outputs: self.config.supports_structured_outputs,
            request_body_transformer: self.config.request_body_transformer.clone(),
        }
    }

    pub(super) fn compat_spec(
        &self,
    ) -> crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter {
        crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::with_settings(
            self.config.adapter.clone(),
            self.request_settings(),
        )
    }

    /// Create a new OpenAI compatible client
    pub async fn new(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = compat_model_middlewares(&config);

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        // Create HTTP client with configuration
        let http_client = Self::build_http_client(&config)?;

        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors,
            model_middlewares,
        })
    }

    /// Construct an `OpenAiCompatibleClient` from an `OpenAiCompatibleConfig` (config-first construction).
    ///
    /// This is a convenience alias for `OpenAiCompatibleClient::new(...)` to align naming with
    /// other provider clients (`*_Client::from_config`).
    pub async fn from_config(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        Self::new(config).await
    }

    /// Construct an `OpenAiCompatibleClient` from a built-in provider id + API key.
    ///
    /// This uses the bundled provider registry (base_url + field mappings) and a
    /// configuration-driven adapter (`ConfigurableAdapter`).
    ///
    /// If `model` is None, we fall back to the provider's `default_model` when available.
    pub async fn from_builtin(
        provider_id: &str,
        api_key: &str,
        model: Option<&str>,
    ) -> Result<Self, LlmError> {
        let provider = super::super::config::get_provider_config(provider_id).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Unknown OpenAI-compatible provider id: {provider_id}"
            ))
        })?;

        let model = model.or(provider.default_model.as_deref()).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Missing model for OpenAI-compatible provider: {provider_id}"
            ))
        })?;

        let adapter = std::sync::Arc::new(ConfigurableAdapter::new(provider.clone()));
        let cfg = OpenAiCompatibleConfig::new(&provider.id, api_key, &provider.base_url, adapter)
            .with_model(model);

        Self::from_config(cfg).await
    }

    /// Construct an `OpenAiCompatibleClient` from a built-in provider id, reading the API key from env.
    ///
    /// Env lookup precedence:
    /// 1) `ProviderConfig.api_key_env` (when provided)
    /// 2) `ProviderConfig.api_key_env_aliases` (fallbacks)
    /// 3) `${PROVIDER_ID}_API_KEY` (uppercased, `-` replaced with `_`)
    pub async fn from_builtin_env(
        provider_id: &str,
        model: Option<&str>,
    ) -> Result<Self, LlmError> {
        let provider = super::super::config::get_provider_config(provider_id).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Unknown OpenAI-compatible provider id: {provider_id}"
            ))
        })?;

        fn default_env_var(id: &str) -> String {
            format!("{}_API_KEY", id.to_ascii_uppercase().replace('-', "_"))
        }

        let mut candidates: Vec<String> = Vec::new();
        if let Some(name) = &provider.api_key_env {
            candidates.push(name.clone());
        }
        candidates.extend(provider.api_key_env_aliases.clone());
        candidates.push(default_env_var(&provider.id));

        let api_key = candidates
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .ok_or_else(|| {
                LlmError::MissingApiKey(format!(
                    "API key not found for provider '{provider_id}'. Tried: {}",
                    candidates.join(", ")
                ))
            })?;

        Self::from_builtin(provider_id, &api_key, model).await
    }

    /// Create a new OpenAI compatible client with custom HTTP client
    pub async fn with_http_client(
        config: OpenAiCompatibleConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = compat_model_middlewares(&config);

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors,
            model_middlewares,
        })
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Build HTTP client with configuration
    fn build_http_client(config: &OpenAiCompatibleConfig) -> Result<reqwest::Client, LlmError> {
        // Use unified HTTP client builder
        crate::execution::http::client::build_http_client_from_config(&config.http_config)
    }

    /// Get the provider ID
    pub fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    /// Get the current model
    pub fn model(&self) -> &str {
        &self.config.model
    }
}
