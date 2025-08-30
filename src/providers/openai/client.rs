//! `OpenAI` Client Implementation
//!
//! Main client structure that aggregates all `OpenAI` capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::params::OpenAiParams;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;

use super::chat::OpenAiChatCapability;
use super::images::OpenAiImages;
use super::models::OpenAiModels;
use super::rerank::OpenAiRerank;
use super::responses::OpenAiResponses;
use super::types::OpenAiSpecificParams;
use super::utils::get_default_models;

/// `OpenAI` Client
#[allow(dead_code)]
pub struct OpenAiClient {
    /// Chat capability implementation
    chat_capability: OpenAiChatCapability,
    /// Models capability implementation
    models_capability: OpenAiModels,
    /// Rerank capability implementation
    rerank_capability: OpenAiRerank,
    /// Image generation capability implementation
    images_capability: OpenAiImages,
    /// Common parameters
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
    /// OpenAI-specific configuration
    specific_params: OpenAiSpecificParams,
    /// HTTP client for making requests
    http_client: reqwest::Client,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (not cloned)
    #[allow(dead_code)]
    _tracing_guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
    /// Responses API toggle
    use_responses_api: bool,
    /// Previous response id for chaining
    previous_response_id: Option<String>,
    /// Built-in tools for Responses API
    built_in_tools: Vec<crate::types::OpenAiBuiltInTool>,
    /// Web search config
    web_search_config: crate::types::WebSearchConfig,
}

impl Clone for OpenAiClient {
    fn clone(&self) -> Self {
        Self {
            chat_capability: self.chat_capability.clone(),
            models_capability: self.models_capability.clone(),
            rerank_capability: self.rerank_capability.clone(),
            images_capability: self.images_capability.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            specific_params: self.specific_params.clone(),
            http_client: self.http_client.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
            web_search_config: self.web_search_config.clone(),
        }
    }
}

impl std::fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("OpenAiClient");

        debug_struct
            .field("provider_name", &"openai")
            .field("model", &self.common_params.model)
            .field("base_url", &self.chat_capability.base_url)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field("use_responses_api", &self.use_responses_api)
            .field("has_tracing", &self.tracing_config.is_some())
            .field("built_in_tools_count", &self.built_in_tools.len());

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
    /// Creates a new `OpenAI` client with configuration and HTTP client
    pub fn new(config: super::OpenAiConfig, http_client: reqwest::Client) -> Self {
        let specific_params = OpenAiSpecificParams {
            organization: config.organization.clone(),
            project: config.project.clone(),
            ..Default::default()
        };

        let chat_capability = OpenAiChatCapability::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
            config.common_params.clone(),
        );

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
        );

        let images_capability = OpenAiImages::new(config.clone(), http_client.clone());

        Self {
            chat_capability,
            models_capability,
            rerank_capability,
            images_capability,
            common_params: config.common_params,
            openai_params: config.openai_params,
            specific_params,
            http_client,
            tracing_config: None,
            _tracing_guard: None,
            use_responses_api: config.use_responses_api,
            previous_response_id: config.previous_response_id,
            built_in_tools: config.built_in_tools,
            web_search_config: config.web_search_config,
        }
    }

    /// Set the tracing guard to keep tracing system active
    pub(crate) fn set_tracing_guard(
        &mut self,
        guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
    ) {
        self._tracing_guard = guard;
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(&mut self, config: Option<crate::tracing::TracingConfig>) {
        self.tracing_config = config;
    }

    /// Creates a new `OpenAI` client with configuration (for OpenAI-compatible providers)
    pub fn new_with_config(config: super::OpenAiConfig) -> Self {
        let http_client = reqwest::Client::new();
        Self::new(config, http_client)
    }
    /// Decide whether to use Responses API for current client config (auto routes gpt-5*)
    pub(crate) fn should_use_responses(&self) -> bool {
        let cfg = super::config::OpenAiConfig {
            api_key: self.chat_capability.api_key.clone(),
            base_url: self.chat_capability.base_url.clone(),
            organization: self.chat_capability.organization.clone(),
            project: self.chat_capability.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.chat_capability.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
        };
        super::utils::should_route_responses(&cfg)
    }

    /// Creates a new `OpenAI` client (legacy constructor for backward compatibility)
    #[allow(clippy::too_many_arguments)]
    pub fn new_legacy(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        common_params: CommonParams,
        openai_params: OpenAiParams,
        http_config: HttpConfig,
        organization: Option<String>,
        project: Option<String>,
    ) -> Self {
        let config = super::OpenAiConfig {
            api_key: secrecy::SecretString::from(api_key),
            base_url,
            organization,
            project,
            common_params,
            openai_params,
            http_config,
            web_search_config: crate::types::WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
        };

        Self::new(config, http_client)
    }

    /// Get OpenAI-specific parameters
    pub const fn specific_params(&self) -> &OpenAiSpecificParams {
        &self.specific_params
    }

    /// Get common parameters (for testing and debugging)
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Get chat capability (for testing and debugging)
    pub const fn chat_capability(&self) -> &OpenAiChatCapability {
        &self.chat_capability
    }

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

#[async_trait]
impl ChatCapability for OpenAiClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        if self.should_use_responses() {
            let config = super::config::OpenAiConfig {
                api_key: self.chat_capability.api_key.clone(),
                base_url: self.chat_capability.base_url.clone(),
                organization: self.chat_capability.organization.clone(),
                project: self.chat_capability.project.clone(),
                common_params: self.common_params.clone(),
                openai_params: self.openai_params.clone(),
                http_config: self.chat_capability.http_config.clone(),
                web_search_config: self.web_search_config.clone(),
                use_responses_api: true,
                previous_response_id: self.previous_response_id.clone(),
                built_in_tools: self.built_in_tools.clone(),
            };
            let responses = OpenAiResponses::new(self.http_client.clone(), config);
            responses.chat_with_tools(messages, tools).await
        } else {
            // Create a ChatRequest from messages and tools, using client's configuration
            let request = ChatRequest {
                messages,
                tools,
                common_params: self.common_params.clone(),
                provider_params: Some(ProviderParams::from_openai(self.openai_params.clone())),
                http_config: None,
                web_search: None,
                stream: false,
            };
            self.chat_capability.chat(request).await
        }
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        if self.should_use_responses() {
            let config = super::config::OpenAiConfig {
                api_key: self.chat_capability.api_key.clone(),
                base_url: self.chat_capability.base_url.clone(),
                organization: self.chat_capability.organization.clone(),
                project: self.chat_capability.project.clone(),
                common_params: self.common_params.clone(),
                openai_params: self.openai_params.clone(),
                http_config: self.chat_capability.http_config.clone(),
                web_search_config: self.web_search_config.clone(),
                use_responses_api: true,
                previous_response_id: self.previous_response_id.clone(),
                built_in_tools: self.built_in_tools.clone(),
            };
            let responses = OpenAiResponses::new(self.http_client.clone(), config);
            responses.chat_stream(messages, tools).await
        } else {
            // Now that OpenAiChatCapability has the correct common_params, we can use the trait method directly
            self.chat_capability.chat_stream(messages, tools).await
        }
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        // Create an OpenAiEmbeddings instance using the client's configuration
        let config = super::config::OpenAiConfig {
            api_key: self.chat_capability.api_key.clone(),
            base_url: self.chat_capability.base_url.clone(),
            organization: self.chat_capability.organization.clone(),
            project: self.chat_capability.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.chat_capability.http_config.clone(),
            web_search_config: crate::types::WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
        };

        let embeddings = super::embeddings::OpenAiEmbeddings::new(config, self.http_client.clone());
        embeddings.embed(texts).await
    }

    fn embedding_dimension(&self) -> usize {
        // Return dimension based on model
        let model = if !self.common_params.model.is_empty() {
            &self.common_params.model
        } else {
            "text-embedding-3-small"
        };

        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // Default fallback
        }
    }

    fn max_tokens_per_embedding(&self) -> usize {
        8192 // OpenAI's current limit
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ]
    }
}

impl LlmProvider for OpenAiClient {
    fn provider_name(&self) -> &'static str {
        "openai"
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
            .with_custom_feature("structured_output", true)
            .with_custom_feature("batch_processing", true)
            .with_custom_feature("rerank", true)
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

impl LlmClient for OpenAiClient {
    fn provider_name(&self) -> &'static str {
        LlmProvider::provider_name(self)
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
        // OpenAI client doesn't directly implement AudioCapability
        // Audio is handled through separate OpenAiAudio struct
        None
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        // Return the image generation capability
        Some(self)
    }
}

#[async_trait]
impl RerankCapability for OpenAiClient {
    /// Rerank documents based on their relevance to a query
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        self.rerank_capability.rerank(request).await
    }

    /// Get the maximum number of documents that can be reranked
    fn max_documents(&self) -> Option<u32> {
        self.rerank_capability.max_documents()
    }

    /// Get supported rerank models for this provider
    fn supported_models(&self) -> Vec<String> {
        self.rerank_capability.supported_models()
    }
}

#[async_trait]
impl ImageGenerationCapability for OpenAiClient {
    /// Generate images from text prompts.
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.images_capability.generate_images(request).await
    }

    /// Edit an existing image with a text prompt.
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.images_capability.edit_image(request).await
    }

    /// Create variations of an existing image.
    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.images_capability.create_variation(request).await
    }

    /// Get supported image sizes for this provider.
    fn get_supported_sizes(&self) -> Vec<String> {
        self.images_capability.get_supported_sizes()
    }

    /// Get supported response formats for this provider.
    fn get_supported_formats(&self) -> Vec<String> {
        self.images_capability.get_supported_formats()
    }

    /// Check if the provider supports image editing.
    fn supports_image_editing(&self) -> bool {
        self.images_capability.supports_image_editing()
    }

    /// Check if the provider supports image variations.
    fn supports_image_variations(&self) -> bool {
        self.images_capability.supports_image_variations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;

    #[test]
    fn test_openai_client_creation() {
        let config = OpenAiConfig::new("test-key");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        assert_eq!(LlmProvider::provider_name(&client), "openai");
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
    fn test_openai_client_legacy_constructor() {
        let client = OpenAiClient::new_legacy(
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            OpenAiParams::default(),
            HttpConfig::default(),
            None,
            None,
        );

        assert_eq!(LlmProvider::provider_name(&client), "openai");
        assert!(!LlmProvider::supported_models(&client).is_empty());
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
            tool_calls: None,
            tool_call_id: None,
        };

        // Create a ChatRequest to test the legacy chat method
        let request = ChatRequest {
            messages: vec![message],
            tools: None,
            common_params: client.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };

        // Test that the request body includes the correct model
        let body = client
            .chat_capability
            .build_chat_request_body(&request)
            .unwrap();
        assert_eq!(body["model"], "gpt-4-test");
    }
}
