//! OpenAI Compatible Client
//!
//! This module provides a client implementation for OpenAI-compatible providers.

use super::openai_config::OpenAiCompatibleConfig;
use crate::client::LlmClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::providers::openai_compatible::middleware::OpenAiCompatibleAlibabaCacheControlWarningMiddleware;
use crate::providers::openai_compatible::middleware::OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware;
use crate::providers::openai_compatible::middleware::OpenAiCompatibleStructuredOutputsWarningMiddleware;
use crate::providers::openai_compatible::middleware::OpenAiCompatibleToolWarningsMiddleware;
use crate::standards::openai::compat::adapter::OpenAiCompatibleRequestSettings;
use crate::standards::openai::compat::provider_registry::ConfigurableAdapter;
// use crate::providers::openai_compatible::RequestType; // no longer needed here
use crate::retry_api::RetryOptions;
use crate::traits::{
    AudioCapability, ChatCapability, CompletionCapability, EmbeddingCapability,
    ImageGenerationCapability, ModelListingCapability, RerankCapability, SpeechCapability,
    SpeechExtras, TranscriptionCapability, TranscriptionExtras,
};
// use crate::execution::transformers::request::RequestTransformer; // unused
use crate::types::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use siumai_core::traits::ModelMetadata;
use std::sync::Arc;
// removed: HashMap import not needed after legacy removal
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

mod audio;
mod chat;
mod completion;
mod embedding;
mod image;
mod rerank;

/// OpenAI Compatible Chat Response with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiCompatibleChoice>,
    pub usage: Option<OpenAiCompatibleUsage>,
}

/// OpenAI Compatible Choice with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChoice {
    pub index: u32,
    pub message: OpenAiCompatibleMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI Compatible Message with provider-specific fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<OpenAiCompatibleToolCall>>,
    pub tool_call_id: Option<String>,

    // Provider-specific thinking/reasoning fields
    pub thinking: Option<String>,          // Standard thinking field
    pub reasoning_content: Option<String>, // DeepSeek reasoning field
    pub reasoning: Option<String>,         // Alternative reasoning field
}

/// OpenAI Compatible Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiCompatibleFunction>,
}

/// OpenAI Compatible Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleFunction {
    pub name: String,
    pub arguments: String,
}

/// OpenAI Compatible Usage
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// OpenAI compatible client
///
/// This is a separate client implementation that uses the adapter system
/// to handle provider-specific differences without modifying the core OpenAI client.
#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    config: OpenAiCompatibleConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

fn model_slot_is_missing(model: Option<&str>) -> bool {
    match model {
        Some(value) => value.trim().is_empty(),
        None => true,
    }
}

const DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING: &str =
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
        super::default_models::get_default_chat_model(&self.config.provider_id)
    }

    fn resolve_family_model_or_config(
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

    async fn build_context(&self) -> Result<ProviderContext, LlmError> {
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

    fn ensure_completion_surface(&self, stream: bool) -> Result<(), LlmError> {
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

    fn http_wiring(&self, ctx: ProviderContext) -> crate::execution::wiring::HttpExecutionWiring {
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

    fn request_settings(&self) -> OpenAiCompatibleRequestSettings {
        OpenAiCompatibleRequestSettings {
            query_params: self.config.query_params.clone(),
            include_usage: self.config.include_usage,
            supports_structured_outputs: self.config.supports_structured_outputs,
            request_body_transformer: self.config.request_body_transformer.clone(),
        }
    }

    fn compat_spec(
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
        let provider = super::config::get_provider_config(provider_id).ok_or_else(|| {
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
        let provider = super::config::get_provider_config(provider_id).ok_or_else(|| {
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

impl ModelMetadata for OpenAiCompatibleClient {
    fn provider_id(&self) -> &str {
        self.config.provider_id.as_str()
    }

    fn model_id(&self) -> &str {
        self.config.model.as_str()
    }
}

impl OpenAiCompatibleClient {
    /// List available models from the provider
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let spec = std::sync::Arc::new(self.compat_spec());
        let ctx = self.build_context().await?;
        let url = spec.try_models_url(&ctx)?;
        let config = self.http_wiring(ctx).config(spec);

        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;
        let models_response: serde_json::Value = result.json;

        // Parse OpenAI-compatible models response
        let models = models_response
            .get("data")
            .and_then(|data| data.as_array())
            .ok_or_else(|| LlmError::ParseError("Invalid models response format".to_string()))?;

        let mut model_infos = Vec::new();
        for model in models {
            if let Some(model_id) = model.get("id").and_then(|id| id.as_str()) {
                let model_info = ModelInfo {
                    id: model_id.to_string(),
                    name: Some(model_id.to_string()),
                    description: model
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                    owned_by: model
                        .get("owned_by")
                        .and_then(|o| o.as_str())
                        .unwrap_or(&self.config.provider_id)
                        .to_string(),
                    created: model.get("created").and_then(|c| c.as_u64()),
                    capabilities: self.determine_model_capabilities(model_id),
                    context_window: None, // Not typically provided by OpenAI-compatible APIs
                    max_output_tokens: None, // Not typically provided by OpenAI-compatible APIs
                    input_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                    output_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                };
                model_infos.push(model_info);
            }
        }

        Ok(model_infos)
    }

    /// Determine model capabilities based on model ID
    fn determine_model_capabilities(&self, model_id: &str) -> Vec<String> {
        let mut capabilities = vec!["chat".to_string()];

        // Add capabilities based on model name patterns
        if model_id.contains("embed") || model_id.contains("embedding") {
            capabilities.push("embedding".to_string());
        }

        if model_id.contains("rerank") || model_id.contains("bge-reranker") {
            capabilities.push("rerank".to_string());
        }

        if model_id.contains("flux")
            || model_id.contains("stable-diffusion")
            || model_id.contains("kolors")
        {
            capabilities.push("image_generation".to_string());
        }

        // Add thinking capability for supported models
        if self
            .config
            .adapter
            .get_model_config(model_id)
            .supports_thinking
        {
            capabilities.push("thinking".to_string());
        }

        capabilities
    }

    /// Get detailed information about a specific model
    async fn get_model_internal(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // Best-effort: prefer the dedicated retrieve endpoint when the provider supports it,
        // then fallback to the list endpoint, and finally a synthetic ModelInfo.
        let spec = std::sync::Arc::new(self.compat_spec());
        let ctx = self.build_context().await?;
        let url = spec.try_model_url(&model_id, &ctx)?;
        let config = self.http_wiring(ctx).config(spec);

        let direct =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await;

        match direct {
            Ok(result) => {
                let json = result.json;

                // OpenAI model retrieve response is usually a single object.
                if let Some(model_id) = json.get("id").and_then(|id| id.as_str()) {
                    return Ok(ModelInfo {
                        id: model_id.to_string(),
                        name: Some(model_id.to_string()),
                        description: json
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|s| s.to_string()),
                        owned_by: json
                            .get("owned_by")
                            .and_then(|o| o.as_str())
                            .unwrap_or(&self.config.provider_id)
                            .to_string(),
                        created: json.get("created").and_then(|c| c.as_u64()),
                        capabilities: self.determine_model_capabilities(model_id),
                        context_window: None,
                        max_output_tokens: None,
                        input_cost_per_token: None,
                        output_cost_per_token: None,
                    });
                }

                // Some vendors might wrap it as `data: [...]` (rare). Best-effort parse.
                if let Some(model) = json
                    .get("data")
                    .and_then(|d| d.as_array())
                    .and_then(|arr| arr.first())
                    && let Some(model_id) = model.get("id").and_then(|id| id.as_str())
                {
                    return Ok(ModelInfo {
                        id: model_id.to_string(),
                        name: Some(model_id.to_string()),
                        description: model
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|s| s.to_string()),
                        owned_by: model
                            .get("owned_by")
                            .and_then(|o| o.as_str())
                            .unwrap_or(&self.config.provider_id)
                            .to_string(),
                        created: model.get("created").and_then(|c| c.as_u64()),
                        capabilities: self.determine_model_capabilities(model_id),
                        context_window: None,
                        max_output_tokens: None,
                        input_cost_per_token: None,
                        output_cost_per_token: None,
                    });
                }
            }
            Err(LlmError::ApiError { code: 404, .. }) => {
                // Fall through to list+basic.
            }
            Err(e) => {
                // If the provider advertises ModelListingCapability but doesn't support
                // the retrieve endpoint, it may still support listing.
                // For other errors (auth/rate limit/etc.), don't mask the failure.
                return Err(e);
            }
        }

        // Fallback: try to find from list.
        let models = self.list_models_internal().await?;
        if let Some(model) = models.into_iter().find(|m| m.id == model_id) {
            return Ok(model);
        }

        // Final fallback: create a basic model info.
        Ok(ModelInfo {
            id: model_id.clone(),
            name: Some(model_id.clone()),
            description: Some(format!("{} model: {}", self.config.provider_id, model_id)),
            owned_by: self.config.provider_id.clone(),
            created: None,
            capabilities: self.determine_model_capabilities(&model_id),
            context_window: None,
            max_output_tokens: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiCompatibleClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.list_models_internal().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.get_model_internal(model_id).await
    }
}

impl LlmClient for OpenAiCompatibleClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        self.config.adapter.provider_id()
    }

    fn supported_models(&self) -> Vec<String> {
        // Return a basic list - could be enhanced with adapter-specific models
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        let adapter_caps = self.config.adapter.capabilities();
        let has_full_audio =
            adapter_caps.audio || (adapter_caps.speech && adapter_caps.transcription);

        // Convert adapter capabilities to library capabilities
        let mut caps = crate::traits::ProviderCapabilities::new();

        if adapter_caps.chat {
            caps = caps.with_chat();
        }
        if adapter_caps.completion {
            caps = caps.with_completion();
        }
        if adapter_caps.streaming {
            caps = caps.with_streaming();
        }
        if has_full_audio {
            caps = caps.with_audio();
        } else {
            if adapter_caps.speech {
                caps = caps.with_speech();
            }
            if adapter_caps.transcription {
                caps = caps.with_transcription();
            }
        }
        if adapter_caps.embedding {
            caps = caps.with_embedding();
        }
        if adapter_caps.supports("rerank") {
            caps = caps.with_rerank();
        }
        if adapter_caps.tools {
            caps = caps.with_tools();
        }
        if adapter_caps.vision {
            caps = caps.with_vision();
        }
        if self.config.adapter.supports_image_generation() {
            caps = caps.with_image_generation();
        }
        for (name, enabled) in &adapter_caps.custom_features {
            caps = caps.with_custom_feature(name, *enabled);
        }

        caps
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new((*self).clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        if self.config.adapter.capabilities().chat {
            Some(self)
        } else {
            None
        }
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        if self.config.adapter.capabilities().embedding {
            Some(self)
        } else {
            None
        }
    }

    fn as_completion_capability(&self) -> Option<&dyn CompletionCapability> {
        if self.capabilities().supports("completion") {
            Some(self)
        } else {
            None
        }
    }

    fn as_embedding_extensions(&self) -> Option<&dyn crate::traits::EmbeddingExtensions> {
        if self.config.adapter.capabilities().embedding {
            Some(self)
        } else {
            None
        }
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        if self.capabilities().supports("audio") {
            Some(self)
        } else {
            None
        }
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        if self.capabilities().supports("speech") {
            Some(self)
        } else {
            None
        }
    }

    fn as_speech_extras(&self) -> Option<&dyn SpeechExtras> {
        if self.capabilities().supports("speech") {
            Some(self)
        } else {
            None
        }
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        if self.capabilities().supports("transcription") {
            Some(self)
        } else {
            None
        }
    }

    fn as_transcription_extras(&self) -> Option<&dyn TranscriptionExtras> {
        if self.capabilities().supports("transcription") {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_extras(&self) -> Option<&dyn crate::traits::ImageExtras> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        // Keep capability gating consistent with executor-level guards:
        // rerank must be explicitly declared by the adapter/spec.
        if self.config.adapter.capabilities().supports("rerank") {
            Some(self)
        } else {
            None
        }
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::provider_options::{
        AlibabaChatOptions, FireworksChatOptions, FireworksReasoningHistory,
        FireworksThinkingConfig, FireworksThinkingType, MoonshotAIChatOptions,
        MoonshotAIReasoningHistory, MoonshotAIThinkingConfig, MoonshotAIThinkingType,
        OpenRouterOptions, OpenRouterTransform, PerplexityOptions, PerplexitySearchContextSize,
        PerplexitySearchMode, PerplexitySearchRecencyFilter, PerplexityUserLocation,
    };
    use crate::providers::openai_compatible::ext::{
        AlibabaChatRequestExt, FireworksChatRequestExt, MoonshotAIChatRequestExt,
        OpenRouterChatRequestExt, PerplexityChatRequestExt, PerplexityChatResponseExt,
    };
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    fn make_text_streaming_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "compat-chat".to_string(),
            name: "Compat Chat".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: Some("compat-default-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 401,
                headers,
                body: br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 401,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                        .to_vec(),
                ),
            })
        }
    }

    #[derive(Clone)]
    struct JsonResponseTransport {
        response_body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl JsonResponseTransport {
        fn new(response: serde_json::Value) -> Self {
            Self {
                response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for JsonResponseTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 501,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"message":"stream unsupported in test","type":"test_error","code":"unsupported"}}"#
                        .to_vec(),
                ),
            })
        }
    }

    #[derive(Clone)]
    struct SseResponseTransport {
        response_body: Arc<Vec<u8>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl SseResponseTransport {
        fn new(body: impl Into<Vec<u8>>) -> Self {
            Self {
                response_body: Arc::new(body.into()),
                last_stream: Arc::new(Mutex::new(None)),
            }
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for SseResponseTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 501,
                headers,
                body: br#"{"error":{"message":"json unsupported in test","type":"test_error","code":"unsupported"}}"#
                    .to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(self.response_body.as_ref().clone()),
            })
        }
    }

    #[tokio::test]
    async fn chat_request_preserves_request_level_provider_options() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("test-model")
                .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(CommonParams {
                model: "test-model".to_string(),
                ..Default::default()
            })
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");
        assert_eq!(captured.body["my_custom"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn chat_request_runtime_mistral_maps_model_length_finish_reason() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "mistral".to_string(),
            name: "Mistral".to_string(),
            base_url: "https://api.mistral.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "tools".to_string()],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl-mistral-finish",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-large-latest",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "done" },
                "finish_reason": "model_length"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "mistral",
            "test-key",
            "https://api.mistral.ai/v1",
            adapter,
        )
        .with_model("mistral-large-latest")
        .with_http_transport(Arc::new(transport));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let response = client
            .chat_request(
                ChatRequest::builder()
                    .model("mistral-large-latest")
                    .messages(vec![ChatMessage::user("hi").build()])
                    .build(),
            )
            .await
            .expect("response ok");

        assert_eq!(response.finish_reason, Some(FinishReason::Length));
        assert_eq!(response.raw_finish_reason.as_deref(), Some("model_length"));
    }

    #[tokio::test]
    async fn chat_request_runtime_qwen_maps_alibaba_cache_write_usage() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "qwen".to_string(),
            name: "Qwen".to_string(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "tools".to_string()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl-qwen-usage",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen-plus",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "done" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 75,
                "total_tokens": 275,
                "prompt_tokens_details": {
                    "cached_tokens": 120,
                    "cache_creation_input_tokens": 50
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 25
                }
            }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "qwen",
            "test-key",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            adapter,
        )
        .with_model("qwen-plus")
        .with_http_transport(Arc::new(transport));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let response = client
            .chat_request(
                ChatRequest::builder()
                    .model("qwen-plus")
                    .messages(vec![ChatMessage::user("hi").build()])
                    .build(),
            )
            .await
            .expect("response ok");
        let usage = response.usage.expect("usage");

        assert_eq!(usage.normalized_input_tokens().no_cache, Some(30));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(120));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(50));
        assert_eq!(usage.normalized_output_tokens().text, Some(50));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(25));
    }

    #[tokio::test]
    async fn chat_request_runtime_qwen_accepts_alibaba_provider_options() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "qwen".to_string(),
            name: "Qwen".to_string(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "tools".to_string()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "qwen",
            "test-key",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            adapter,
        )
        .with_model("qwen-plus")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("qwen-plus")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "lookup",
                "Lookup value",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .build()
            .with_alibaba_options(
                AlibabaChatOptions::new()
                    .with_enable_thinking(true)
                    .with_thinking_budget(2048)
                    .with_parallel_tool_calls(false),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["enable_thinking"], serde_json::json!(true));
        assert_eq!(captured.body["thinking_budget"], serde_json::json!(2048));
        assert_eq!(
            captured.body["parallel_tool_calls"],
            serde_json::json!(false)
        );
        assert!(captured.body.get("enableThinking").is_none());
        assert!(captured.body.get("thinkingBudget").is_none());
        assert!(captured.body.get("parallelToolCalls").is_none());
    }

    #[tokio::test]
    async fn chat_request_runtime_qwen_applies_alibaba_prompt_cache_control_and_warning() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "qwen".to_string(),
            name: "Qwen".to_string(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "tools".to_string()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl-qwen-cache",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen-plus",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "done" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "qwen",
            "test-key",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            adapter,
        )
        .with_model("qwen-plus")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("qwen-plus")
            .messages(vec![
                ChatMessage::system("system prompt")
                    .with_provider_option(
                        "alibaba",
                        serde_json::json!({ "cacheControl": { "type": "system" } }),
                    )
                    .build(),
                ChatMessage::user("")
                    .with_content_parts(vec![
                        ContentPart::text("inspect").with_provider_option(
                            "qwen",
                            serde_json::json!({ "cache_control": { "type": "user-part" } }),
                        ),
                        ContentPart::image_url("https://example.com/image.png"),
                    ])
                    .with_provider_option(
                        "alibaba",
                        serde_json::json!({ "cacheControl": { "type": "user-message" } }),
                    )
                    .build(),
                ChatMessage::assistant("previous answer")
                    .with_provider_option(
                        "qwen",
                        serde_json::json!({ "cacheControl": { "type": "assistant" } }),
                    )
                    .build(),
                ChatMessage::user("final question")
                    .with_provider_option(
                        "alibaba",
                        serde_json::json!({ "cacheControl": { "type": "final-user" } }),
                    )
                    .build(),
            ])
            .build()
            .with_provider_option(
                "alibaba",
                serde_json::json!({
                    "enableThinking": true,
                    "cacheControl": { "type": "request-level-is-not-forwarded" }
                }),
            );

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["messages"][0]["content"][0]["cache_control"],
            serde_json::json!({ "type": "system" })
        );
        assert_eq!(
            captured.body["messages"][1]["content"][0]["cache_control"],
            serde_json::json!({ "type": "user-part" })
        );
        assert_eq!(
            captured.body["messages"][1]["content"][1]["cache_control"],
            serde_json::json!({ "type": "user-message" })
        );
        assert_eq!(
            captured.body["messages"][2]["content"][0]["cache_control"],
            serde_json::json!({ "type": "assistant" })
        );
        assert_eq!(
            captured.body["messages"][3]["content"][0]["cache_control"],
            serde_json::json!({ "type": "final-user" })
        );
        assert_eq!(captured.body["enable_thinking"], serde_json::json!(true));
        assert!(captured.body.get("cacheControl").is_none());

        assert_eq!(
            response.warnings,
            Some(vec![crate::types::Warning::other(
                crate::standards::openai::compat::alibaba_cache_control::CACHE_BREAKPOINT_LIMIT_WARNING,
            )])
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_xai_preserves_stable_fields_at_transport_boundary() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "xai".to_string(),
            name: "xAI".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new("xai", "test-key", "https://api.x.ai/v1", adapter)
            .with_model("grok-4")
            .with_supports_structured_outputs(true)
            .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model_params(CommonParams {
                model: "grok-4".to_string(),
                frequency_penalty: Some(0.2),
                presence_penalty: Some(0.4),
                ..CommonParams::default()
            })
            .messages(vec![ChatMessage::user("hi").build()])
            .stop_sequences(vec!["END".to_string()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto",
                    "reasoningEffort": "high",
                    "enableReasoning": true,
                    "reasoningBudget": 2048
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert!(captured.body.get("stop").is_none());
        assert!(captured.body.get("frequency_penalty").is_none());
        assert!(captured.body.get("presence_penalty").is_none());
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert!(captured.body.get("enableReasoning").is_none());
        assert!(captured.body.get("enable_reasoning").is_none());
        assert!(captured.body.get("reasoningBudget").is_none());
        assert!(captured.body.get("reasoning_budget").is_none());
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_deepseek_normalizes_thinking_options() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .seed(1234)
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "enableReasoning": true,
                    "reasoningBudget": 2048,
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["thinking"],
            serde_json::json!({
                "type": "enabled"
            })
        );
        assert!(captured.body.get("enableReasoning").is_none());
        assert!(captured.body.get("enable_reasoning").is_none());
        assert!(captured.body.get("reasoningBudget").is_none());
        assert!(captured.body.get("reasoning_budget").is_none());
        assert!(captured.body.get("seed").is_none());
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({ "type": "json_object" })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_provider_specific_known_compat_options_map_to_ai_sdk_fields() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            adapter,
        )
        .with_model("openai/gpt-4o")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "openrouter",
                serde_json::json!({
                    "user": "compat-user-9",
                    "reasoningEffort": "high",
                    "textVerbosity": "medium",
                    "strictJsonSchema": false
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["user"], serde_json::json!("compat-user-9"));
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(captured.body["verbosity"], serde_json::json!("medium"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": false
                }
            })
        );
        assert!(captured.body.get("strictJsonSchema").is_none());
        assert!(captured.body.get("textVerbosity").is_none());
    }

    #[tokio::test]
    async fn chat_request_runtime_fireworks_provider_options_normalize_to_wire_shape() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "fireworks".to_string(),
            name: "Fireworks".to_string(),
            base_url: "https://api.fireworks.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            adapter,
        )
        .with_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("accounts/fireworks/models/llama-v3p1-8b-instruct")
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_provider_option(
                "fireworks",
                serde_json::json!({ "reasoningEffort": "minimal" }),
            )
            .with_fireworks_options(
                FireworksChatOptions::new()
                    .with_thinking(
                        FireworksThinkingConfig::new()
                            .with_type(FireworksThinkingType::Enabled)
                            .with_budget_tokens(2048),
                    )
                    .with_reasoning_history(FireworksReasoningHistory::Interleaved),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("low"));
        assert_eq!(
            captured.body["reasoning_history"],
            serde_json::json!("interleaved")
        );
        assert_eq!(
            captured.body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 2048
            })
        );
        assert!(captured.body.get("reasoningHistory").is_none());
        assert!(captured.body["thinking"].get("budgetTokens").is_none());
    }

    #[tokio::test]
    async fn chat_request_runtime_moonshotai_provider_options_normalize_to_wire_shape() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "moonshotai".to_string(),
            name: "Moonshot AI".to_string(),
            base_url: "https://api.moonshot.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "reasoning".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: Some("MOONSHOT_API_KEY".to_string()),
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "moonshotai",
            "test-key",
            "https://api.moonshot.ai/v1",
            adapter,
        )
        .with_model("kimi-k2-thinking")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("kimi-k2-thinking")
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_provider_option("moonshotai", serde_json::json!({ "user": "compat-user-7" }))
            .with_moonshotai_options(
                MoonshotAIChatOptions::new()
                    .with_thinking(
                        MoonshotAIThinkingConfig::new()
                            .with_type(MoonshotAIThinkingType::Enabled)
                            .with_budget_tokens(2048),
                    )
                    .with_reasoning_history(MoonshotAIReasoningHistory::Interleaved),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["user"], serde_json::json!("compat-user-7"));
        assert_eq!(
            captured.body["reasoning_history"],
            serde_json::json!("interleaved")
        );
        assert_eq!(
            captured.body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 2048
            })
        );
        assert!(captured.body.get("reasoningHistory").is_none());
        assert!(captured.body["thinking"].get("budgetTokens").is_none());
    }

    #[tokio::test]
    async fn chat_request_runtime_provider_defined_tools_emit_ai_sdk_warning() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl_tool_warning",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok"
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::provider_defined(
                "openai.web_search",
                "web_search",
            )])
            .build();

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert!(captured.body.get("tools").is_none());
        assert_eq!(
            response.warnings,
            Some(vec![crate::types::Warning::unsupported(
                "provider-defined tool openai.web_search",
                None::<String>,
            )])
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_deprecated_openai_compatible_key_emits_warning() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl_deprecated_key",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok"
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_provider_option(
                "openai-compatible",
                serde_json::json!({ "user": "compat-user-legacy" }),
            );

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["user"],
            serde_json::json!("compat-user-legacy")
        );
        assert_eq!(
            response.warnings,
            Some(vec![crate::types::Warning::other(
                "The 'openai-compatible' key in providerOptions is deprecated. Use 'openaiCompatible' instead.",
            )])
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_openrouter_preserves_stable_fields_and_vendor_params_at_transport_boundary()
     {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "reasoning".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            adapter,
        )
        .with_model("openai/gpt-4o")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "openrouter",
                serde_json::json!({
                    "transforms": ["middle-out"],
                    "someVendorParam": true,
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_openrouter_typed_options_preserve_final_request_shape() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "reasoning".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            adapter,
        )
        .with_model("openai/gpt-4o")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_openrouter_options(
                OpenRouterOptions::new()
                    .with_transform(OpenRouterTransform::MiddleOut)
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_openrouter_preserves_stable_fields_reasoning_defaults_and_vendor_params_at_transport_boundary()
     {
        let transport = CaptureTransport::default();
        let client = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            crate::builder::BuilderBase::default(),
            "openrouter",
        )
        .api_key("test")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()))
        .build()
        .await
        .expect("builder should succeed");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_openrouter_options(
                OpenRouterOptions::new()
                    .with_transform(OpenRouterTransform::MiddleOut)
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured
                .headers
                .get(ACCEPT)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(captured.body["stream"], serde_json::json!(true));
        assert_eq!(captured.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(2048));
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_perplexity_preserves_stable_fields_and_vendor_params_at_transport_boundary()
     {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "perplexity",
                serde_json::json!({
                    "search_mode": "academic",
                    "someVendorParam": true,
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_perplexity_preserves_stable_fields_and_typed_vendor_params_at_transport_boundary()
     {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "perplexity",
                serde_json::json!({
                    "search_mode": "web",
                    "return_images": false,
                    "web_search_options": {
                        "search_context_size": "low"
                    }
                }),
            )
            .with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_search_context_size(PerplexitySearchContextSize::High)
                    .with_return_images(true)
                    .with_user_location(PerplexityUserLocation::new().with_country("US"))
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured
                .headers
                .get(ACCEPT)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(captured.body["stream"], serde_json::json!(true));
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(captured.body["return_images"], serde_json::json!(true));
        assert_eq!(
            captured.body["web_search_options"]["search_context_size"],
            serde_json::json!("high")
        );
        assert_eq!(
            captured.body["web_search_options"]["user_location"]["country"],
            serde_json::json!("US")
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert!(captured.body.get("searchMode").is_none());
        assert!(captured.body.get("returnImages").is_none());
        assert!(captured.body.get("webSearchOptions").is_none());
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_perplexity_typed_options_preserve_final_request_shape() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
                    .with_return_images(true)
                    .with_search_context_size(PerplexitySearchContextSize::High)
                    .with_user_location(PerplexityUserLocation::new().with_country("US"))
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(
            captured.body["search_recency_filter"],
            serde_json::json!("month")
        );
        assert_eq!(captured.body["return_images"], serde_json::json!(true));
        assert_eq!(
            captured.body["web_search_options"]["search_context_size"],
            serde_json::json!("high")
        );
        assert_eq!(
            captured.body["web_search_options"]["user_location"]["country"],
            serde_json::json!("US")
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_perplexity_exposes_typed_response_metadata() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl-perplexity-test",
            "object": "chat.completion",
            "created": 1_741_392_000,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Rust async tooling kept improving across the ecosystem."
                    },
                    "finish_reason": "stop"
                }
            ],
            "citations": ["https://example.com/rust"],
            "images": [
                {
                    "image_url": "https://images.example.com/rust.png",
                    "origin_url": "https://example.com/rust",
                    "height": 900,
                    "width": 1600
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 17,
                "total_tokens": 28,
                "citation_tokens": 7,
                "num_search_queries": 2,
                "reasoning_tokens": 3,
                "cost": {
                    "input_tokens_cost": 0.12,
                    "output_tokens_cost": 0.34,
                    "request_cost": 0.01,
                    "total_cost": 0.47
                }
            }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["return_images"], serde_json::json!(true));
        assert_eq!(
            response.content_text(),
            Some("Rust async tooling kept improving across the ecosystem.")
        );

        let meta = response.perplexity_metadata().expect("perplexity metadata");
        assert_eq!(
            meta.citations.as_ref(),
            Some(&vec!["https://example.com/rust".to_string()])
        );
        assert_eq!(meta.images.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.images
                .as_ref()
                .and_then(|images| images.first())
                .map(|image| image.image_url.as_str()),
            Some("https://images.example.com/rust.png")
        );
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.citation_tokens),
            Some(7)
        );
        assert_eq!(
            meta.usage
                .as_ref()
                .and_then(|usage| usage.num_search_queries),
            Some(2)
        );
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.reasoning_tokens),
            Some(3)
        );
        assert_eq!(
            meta.cost.as_ref().and_then(|cost| cost.request_cost),
            Some(0.01)
        );
        assert_eq!(
            meta.cost.as_ref().and_then(|cost| cost.total_cost),
            Some(0.47)
        );
        assert_eq!(meta.extra.get("citations"), None);
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_perplexity_exposes_typed_response_metadata() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = SseResponseTransport::new(
            br#"data: {"id":"1","model":"sonar","created":1718345013,"citations":["https://example.com/rust"],"choices":[{"index":0,"delta":{"content":"Rust","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"sonar","created":1718345013,"choices":[{"index":0,"delta":{"content":" ecosystem","role":null},"finish_reason":"stop"}],"images":[{"image_url":"https://images.example.com/rust.png","origin_url":"https://example.com/rust","height":900,"width":1600}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28,"citation_tokens":7,"num_search_queries":2,"reasoning_tokens":3,"cost":{"input_tokens_cost":0.12,"output_tokens_cost":0.34,"request_cost":0.01,"total_cost":0.47}}}

data: [DONE]

"#.to_vec(),
        );

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

        let stream = crate::traits::ChatCapability::chat_stream_request(&client, request)
            .await
            .expect("stream ok");
        let events = stream.collect::<Vec<_>>().await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(captured.body["return_images"], serde_json::json!(true));

        let end = events
            .iter()
            .find_map(|event| match event {
                Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");
        assert_eq!(
            end.usage.as_ref().and_then(|usage| usage.total_tokens()),
            Some(28)
        );

        let meta = end.perplexity_metadata().expect("perplexity metadata");
        assert_eq!(
            meta.citations.as_ref(),
            Some(&vec!["https://example.com/rust".to_string()])
        );
        assert_eq!(meta.images.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.citation_tokens),
            Some(7)
        );
        assert_eq!(
            meta.usage
                .as_ref()
                .and_then(|usage| usage.num_search_queries),
            Some(2)
        );
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.reasoning_tokens),
            Some(3)
        );
        assert_eq!(
            meta.cost.as_ref().and_then(|cost| cost.request_cost),
            Some(0.01)
        );
        assert_eq!(
            meta.cost.as_ref().and_then(|cost| cost.total_cost),
            Some(0.47)
        );
        assert_eq!(meta.extra.get("citations"), None);
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_preserves_lossless_text_and_reasoning_deltas() {
        let adapter = make_text_streaming_adapter();
        let transport = SseResponseTransport::new(
            br#"data: {"id":"1","model":"compat-model","choices":[{"index":0,"delta":{"content":"alpha","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"compat-model","choices":[{"index":0,"delta":{"content":"\n"},"finish_reason":null}]}

data: {"id":"1","model":"compat-model","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}

data: {"id":"1","model":"compat-model","choices":[{"index":0,"delta":{"reasoning_content":"\n\n"},"finish_reason":null}]}

data: {"id":"1","model":"compat-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

"#.to_vec(),
        );

        let cfg = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            adapter,
        )
        .with_model("compat-model")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("compat-model")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let stream = crate::traits::ChatCapability::chat_stream_request(&client, request)
            .await
            .expect("stream ok");
        let events = stream.collect::<Vec<_>>().await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(captured.body["stream"], serde_json::json!(true));

        let mut text_deltas = Vec::new();
        let mut reasoning_deltas = Vec::new();

        for event in events {
            let event = event.expect("stream event");
            if let Some(delta) = event.text_delta() {
                text_deltas.push(delta.to_string());
            }
            if let Some(delta) = event.reasoning_delta() {
                reasoning_deltas.push(delta.to_string());
            }
        }

        assert_eq!(text_deltas, vec!["alpha", "\n", ""]);
        assert_eq!(reasoning_deltas, vec!["\n\n"]);
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_xai_strips_stream_only_fields_at_transport_boundary() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "xai".to_string(),
            name: "xAI".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new("xai", "test-key", "https://api.x.ai/v1", adapter)
            .with_model("grok-4")
            .with_supports_structured_outputs(true)
            .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("grok-4")
            .messages(vec![ChatMessage::user("hi").build()])
            .stop_sequences(vec!["END".to_string()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto",
                    "reasoningEffort": "high"
                }),
            );

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert!(captured.body.get("stop").is_none());
        assert!(captured.body.get("stream_options").is_none());
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_omits_stream_options_by_default_for_compat_provider() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert!(captured.body.get("stream_options").is_none());
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_include_usage_restores_stream_options() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_include_usage(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured.body.get("stream_options"),
            Some(&serde_json::json!({ "include_usage": true }))
        );
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_applies_request_body_transformer() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let transformer = Arc::new(
            |body: &mut serde_json::Value, _model: &str, _request_type| {
                body["custom"] = serde_json::json!(true);
                body.as_object_mut()
                    .expect("object body")
                    .remove("stream_options");
                Ok(())
            },
        );

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_include_usage(true)
        .with_request_body_transformer(transformer)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert!(captured.body.get("stream_options").is_none());
        assert_eq!(captured.body.get("custom"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_appends_query_params_to_transport_url() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_query_params([("api-version", "2025-04-01"), ("tenant", "acme")])
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured.url,
            "https://api.deepseek.com/v1/chat/completions?api-version=2025-04-01&tenant=acme"
        );
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_structured_outputs_policy_defaults_to_json_object() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({ "type": "object", "properties": {} }),
            ))
            .build();

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({ "type": "json_object" })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_structured_outputs_default_emits_warning_and_json_object() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "{\"answer\":\"ok\"}"
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } }
                }),
            ))
            .build();

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({ "type": "json_object" })
        );
        assert_eq!(
            response.warnings,
            Some(vec![crate::types::Warning::unsupported(
                "responseFormat",
                Some("JSON response format schema is only supported with structuredOutputs"),
            )])
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_structured_outputs_enabled_preserves_schema_without_warning() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl_2",
            "object": "chat.completion",
            "created": 1,
            "model": "openai/gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "{\"answer\":\"ok\"}"
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            adapter,
        )
        .with_model("openai/gpt-4o")
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build();

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
        assert!(response.warnings.as_ref().is_none_or(Vec::is_empty));
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_structured_outputs_default_emits_warning_and_json_object()
    {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = SseResponseTransport::new(
            br#"data: {"id":"1","model":"deepseek-chat","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"{\"answer\":\"ok\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}

data: [DONE]

"#
            .to_vec(),
        );

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } }
                }),
            ))
            .build();

        let stream = crate::traits::ChatCapability::chat_stream_request(&client, request)
            .await
            .expect("stream ok");
        let events = stream.collect::<Vec<_>>().await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({ "type": "json_object" })
        );

        let end = events
            .iter()
            .find_map(|event| match event {
                Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");

        assert_eq!(
            end.warnings,
            Some(vec![crate::types::Warning::unsupported(
                "responseFormat",
                Some("JSON response format schema is only supported with structuredOutputs"),
            )])
        );
    }

    #[tokio::test]
    async fn builder_installs_provider_specific_params_adapter() {
        let client = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            crate::builder::BuilderBase::default(),
            "deepseek",
        )
        .api_key("test")
        .model("deepseek-reasoner")
        .reasoning(true)
        .build()
        .await
        .expect("builder should succeed");

        let mut chat_body = serde_json::json!({});
        client
            .config
            .adapter
            .transform_request_params(
                &mut chat_body,
                "deepseek-reasoner",
                crate::providers::openai_compatible::types::RequestType::Chat,
            )
            .unwrap();
        assert_eq!(
            chat_body.get("thinking"),
            Some(&serde_json::json!({
                "type": "enabled"
            }))
        );
        assert!(chat_body.get("enable_reasoning").is_none());

        let mut emb_body = serde_json::json!({});
        client
            .config
            .adapter
            .transform_request_params(
                &mut emb_body,
                "text-embedding-3-small",
                crate::providers::openai_compatible::types::RequestType::Embedding,
            )
            .unwrap();
        assert!(emb_body.get("enable_reasoning").is_none());
        assert!(emb_body.get("thinking").is_none());
    }

    #[tokio::test]
    async fn builder_runtime_openrouter_reasoning_helpers_preserve_final_request_shape() {
        let transport = CaptureTransport::default();
        let client = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            crate::builder::BuilderBase::default(),
            "openrouter",
        )
        .api_key("test")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .with_supports_structured_outputs(true)
        .with_http_transport(Arc::new(transport.clone()))
        .build()
        .await
        .expect("builder should succeed");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build();

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(2048));
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn test_client_creation() {
        let provider_config = crate::standards::openai::compat::provider_registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::standards::openai::compat::provider_registry::ProviderFieldMappings::default(
                ),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        };

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        )
        .with_model("test-model");

        let client = OpenAiCompatibleClient::new(config).await.unwrap();
        assert_eq!(client.provider_id(), "test");
        assert_eq!(client.model(), "test-model");
    }

    #[tokio::test]
    async fn test_client_validation() {
        let provider_config = crate::standards::openai::compat::provider_registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::standards::openai::compat::provider_registry::ProviderFieldMappings::default(
                ),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        };

        // Invalid config should fail
        let config = OpenAiCompatibleConfig::new(
            "",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        );

        assert!(OpenAiCompatibleClient::new(config).await.is_err());
    }
}
