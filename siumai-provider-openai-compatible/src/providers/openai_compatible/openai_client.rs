//! OpenAI Compatible Client
//!
//! This module provides a client implementation for OpenAI-compatible providers.

use super::openai_config::OpenAiCompatibleConfig;
use crate::client::LlmClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
use crate::execution::executors::image::{HttpImageExecutor, ImageExecutor};
// use crate::providers::openai_compatible::RequestType; // no longer needed here
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    ChatCapability, EmbeddingCapability, ImageExtras, ImageGenerationCapability,
    ModelListingCapability, RerankCapability,
};
// use crate::execution::transformers::request::RequestTransformer; // unused
use crate::types::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
// removed: HashMap import not needed after legacy removal
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

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

impl OpenAiCompatibleClient {
    fn build_context(&self) -> ProviderContext {
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
            Some(self.config.api_key.clone()),
            extra_headers,
        )
    }

    fn http_wiring(&self, ctx: ProviderContext) -> crate::execution::wiring::HttpExecutionWiring {
        crate::execution::wiring::HttpExecutionWiring::new(
            self.config.provider_id.clone(),
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone())
    }

    /// Build the provider execution context (headers/base_url/api key + extra headers).
    ///
    /// This is primarily used by hybrid providers (e.g. Groq) that reuse the OpenAI-compatible
    /// client for chat/embedding/image but need to invoke non-chat executors with the same
    /// HTTP wiring (client, interceptors, retry).
    pub fn provider_context(&self) -> ProviderContext {
        self.build_context()
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

    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        let ctx = self.build_context();
        let spec = Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder =
            ChatExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_transformer_bundle(bundle)
                .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
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

    fn build_embedding_executor(&self, request: &EmbeddingRequest) -> Arc<HttpEmbeddingExecutor> {
        use crate::execution::executors::embedding::EmbeddingExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let mut builder = EmbeddingExecutorBuilder::new(
            self.config.provider_id.clone(),
            self.http_client.clone(),
        )
        .with_spec(spec)
        .with_context(ctx)
        .with_interceptors(self.http_interceptors.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build_for_request(request)
    }

    fn build_image_executor(&self, request: &ImageGenerationRequest) -> Arc<HttpImageExecutor> {
        use crate::execution::executors::image::ImageExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let mut builder =
            ImageExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_interceptors(self.http_interceptors.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build_for_request(request)
    }

    /// Execute a non-stream chat via ProviderSpec
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute a stream chat via ProviderSpec
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
    /// Create a new OpenAI compatible client
    pub async fn new(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

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
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        })
    }

    /// Create a new OpenAI compatible client with custom HTTP client
    pub async fn with_http_client(
        config: OpenAiCompatibleConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        })
    }

    // Removed legacy build_headers; headers are created in executor closures

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

    // Removed legacy build_chat_request; executors use transformers directly

    // Removed legacy parse_chat_response; response transformer handles mapping

    // removed legacy send_request; executors handle requests
}

// Removed legacy chat_with_tools_inner; ChatCapability now uses HttpChatExecutor

#[async_trait]
impl ChatCapability for OpenAiCompatibleClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Build unified ChatRequest
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.config.http_config.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let req = builder.build();

        // Execute via ProviderSpec
        self.chat_request_via_spec(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Unified ChatRequest
        let mut builder = crate::types::ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.config.http_config.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        // Execute via ProviderSpec
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiCompatibleClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let req = crate::types::EmbeddingRequest::new(texts).with_model(self.config.model.clone());
        let exec = self.build_embedding_executor(&req);
        EmbeddingExecutor::execute(&*exec, req).await
    }

    fn embedding_dimension(&self) -> usize {
        // Default dimension, could be made configurable per model
        1536
    }
}

#[async_trait]
impl crate::traits::EmbeddingExtensions for OpenAiCompatibleClient {
    async fn embed_with_config(
        &self,
        mut request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        if request.model.as_deref().unwrap_or("").is_empty() {
            request.model = Some(self.config.model.clone());
        }

        let exec = self.build_embedding_executor(&request);
        EmbeddingExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl RerankCapability for OpenAiCompatibleClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};

        let ctx = self.build_context();
        let spec = std::sync::Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );

        let mut builder =
            RerankExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_interceptors(self.http_interceptors.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        let exec = builder.build_for_request(&request);
        RerankExecutor::execute(&*exec, request).await
    }
}

impl OpenAiCompatibleClient {
    /// List available models from the provider
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let spec = std::sync::Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let ctx = self.build_context();
        let url = spec.models_url(&ctx);
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
        let spec = std::sync::Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let ctx = self.build_context();
        let url = spec.model_url(&model_id, &ctx);
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

#[async_trait]
impl ImageGenerationCapability for OpenAiCompatibleClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if !self.config.adapter.supports_image_generation() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image generation",
                self.config.provider_id
            )));
        }
        let exec = self.build_image_executor(&request);
        ImageExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl ImageExtras for OpenAiCompatibleClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if !self.config.adapter.supports_image_editing() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image editing",
                self.config.provider_id
            )));
        }

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
        ImageExecutor::execute_edit(&*exec, request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if !self.config.adapter.supports_image_variations() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image variations",
                self.config.provider_id
            )));
        }

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
        ImageExecutor::execute_variation(&*exec, request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_sizes()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_formats()
    }

    fn supports_image_editing(&self) -> bool {
        self.config.adapter.supports_image_editing()
    }

    fn supports_image_variations(&self) -> bool {
        self.config.adapter.supports_image_variations()
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

        // Convert adapter capabilities to library capabilities
        let mut caps = crate::traits::ProviderCapabilities::new();

        if adapter_caps.chat {
            caps = caps.with_chat();
        }
        if adapter_caps.streaming {
            caps = caps.with_streaming();
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

        caps
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new((*self).clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        if self.config.adapter.capabilities().embedding {
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
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[tokio::test]
    async fn build_chat_executor_wires_before_send_for_custom_options_with_runtime_provider_id() {
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

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("test-model");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(CommonParams {
                model: "test-model".to_string(),
                ..Default::default()
            })
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let exec = client.build_chat_executor(&req);
        assert!(exec.policy.before_send.is_some());
    }

    #[tokio::test]
    async fn build_embedding_executor_wires_before_send_and_interceptors() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "embedding".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("text-embedding-3-small");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap()
            .with_http_interceptors(vec![Arc::new(NoopInterceptor)]);

        let req = EmbeddingRequest::single("hi")
            .with_model("text-embedding-3-small")
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let exec = client.build_embedding_executor(&req);
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.before_send.is_none());
        assert!(
            exec.provider_spec
                .embedding_before_send(&req, &exec.provider_context)
                .is_some()
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
            chat_body.get("enable_reasoning"),
            Some(&serde_json::Value::Bool(true))
        );

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

// Removed legacy rerank parsing; rerank now routed through OpenAI Rerank Standard
