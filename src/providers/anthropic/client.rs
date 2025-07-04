//! Anthropic Client Implementation
//!
//! Main client structure that aggregates all Anthropic capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::params::AnthropicParams;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;

use super::chat::AnthropicChatCapability;
use super::models::AnthropicModels;
use super::types::AnthropicSpecificParams;
use super::utils::get_default_models;

/// Anthropic Client
#[allow(dead_code)]
pub struct AnthropicClient {
    /// Chat capability implementation
    chat_capability: AnthropicChatCapability,
    /// Models capability implementation
    models_capability: AnthropicModels,
    /// Common parameters
    common_params: CommonParams,
    /// Anthropic-specific parameters
    anthropic_params: AnthropicParams,
    /// Anthropic-specific configuration
    specific_params: AnthropicSpecificParams,
}

impl AnthropicClient {
    /// Creates a new Anthropic client
    pub fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        common_params: CommonParams,
        anthropic_params: AnthropicParams,
        http_config: HttpConfig,
    ) -> Self {
        let specific_params = AnthropicSpecificParams::default();

        let chat_capability = AnthropicChatCapability::new(
            api_key.clone(),
            base_url.clone(),
            http_client.clone(),
            http_config.clone(),
            specific_params.clone(),
        );

        let models_capability = AnthropicModels::new(api_key, base_url, http_client, http_config);

        Self {
            chat_capability,
            models_capability,
            common_params,
            anthropic_params,
            specific_params,
        }
    }

    /// Get Anthropic-specific parameters
    pub const fn specific_params(&self) -> &AnthropicSpecificParams {
        &self.specific_params
    }

    /// Get common parameters (for testing and debugging)
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Get chat capability (for testing and debugging)
    pub const fn chat_capability(&self) -> &AnthropicChatCapability {
        &self.chat_capability
    }

    /// Update Anthropic-specific parameters
    pub fn with_specific_params(mut self, params: AnthropicSpecificParams) -> Self {
        self.specific_params = params;
        self
    }

    /// Enable beta features
    pub fn with_beta_features(mut self, features: Vec<String>) -> Self {
        self.specific_params.beta_features = features;
        self
    }

    /// Enable prompt caching
    pub fn with_cache_control(mut self, cache_control: super::cache::CacheControl) -> Self {
        self.specific_params.cache_control = Some(cache_control);
        self
    }

    /// Enable thinking mode with specified budget tokens
    pub fn with_thinking_mode(mut self, budget_tokens: Option<u32>) -> Self {
        let config = budget_tokens.map(super::thinking::ThinkingConfig::enabled);
        self.specific_params.thinking_config = config;
        self
    }

    /// Enable thinking mode with default budget (10k tokens)
    pub fn with_thinking_enabled(mut self) -> Self {
        self.specific_params.thinking_config =
            Some(super::thinking::ThinkingConfig::enabled(10000));
        self
    }

    /// Set custom metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.specific_params.metadata = Some(metadata);
        self
    }

    /// Add a beta feature
    pub fn add_beta_feature(mut self, feature: String) -> Self {
        self.specific_params.beta_features.push(feature);
        self
    }

    /// Enable prompt caching with ephemeral type
    pub fn with_ephemeral_cache(self) -> Self {
        self.with_cache_control(super::cache::CacheControl::ephemeral())
    }
}

#[async_trait]
impl ChatCapability for AnthropicClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Create a ChatRequest with client's configuration
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };

        let headers = super::utils::build_headers(
            &self.chat_capability.api_key,
            &self.chat_capability.http_config.headers,
        )?;
        let body = self
            .chat_capability
            .build_chat_request_body(&request, Some(&self.specific_params))?;
        let url = format!("{}/v1/messages", self.chat_capability.base_url);

        let response = self
            .chat_capability
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Anthropic API error: {error_text}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let anthropic_response: super::types::AnthropicChatResponse = response.json().await?;
        self.chat_capability.parse_chat_response(anthropic_response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a new chat capability with current configuration for streaming
        let chat_capability = super::chat::AnthropicChatCapability::new(
            self.chat_capability.api_key.clone(),
            self.chat_capability.base_url.clone(),
            self.chat_capability.http_client.clone(),
            self.chat_capability.http_config.clone(),
            self.specific_params.clone(),
        );

        // Create a ChatRequest with client's configuration for streaming
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };

        let headers = super::utils::build_headers(
            &chat_capability.api_key,
            &chat_capability.http_config.headers,
        )?;
        let request_body =
            chat_capability.build_chat_request_body(&request, Some(&self.specific_params))?;

        let response = chat_capability
            .http_client
            .post(format!("{}/v1/messages", chat_capability.base_url))
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Anthropic API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create stream from response with UTF-8 decoder
        use crate::utils::Utf8StreamDecoder;
        use futures_util::StreamExt;
        use std::sync::{Arc, Mutex};

        let decoder = Arc::new(Mutex::new(Utf8StreamDecoder::new()));
        let decoder_for_flush = decoder.clone();

        let stream = response.bytes_stream();
        let decoded_stream = stream.filter_map(move |chunk_result| {
            let decoder = decoder.clone();
            async move {
                match chunk_result {
                    Ok(chunk) => {
                        // Use UTF-8 decoder to handle incomplete sequences
                        let decoded_chunk = {
                            let mut decoder = decoder.lock().unwrap();
                            decoder.decode(&chunk)
                        };

                        if !decoded_chunk.is_empty() {
                            if let Some(event) =
                                super::chat::AnthropicChatCapability::parse_sse_event(
                                    &decoded_chunk,
                                )
                            {
                                return Some(event);
                            }
                        }
                        None
                    }
                    Err(e) => Some(Err(LlmError::StreamError(format!("Stream error: {e}")))),
                }
            }
        });

        // Add flush operation
        let flush_stream = futures_util::stream::once(async move {
            let remaining = {
                let mut decoder = decoder_for_flush.lock().unwrap();
                decoder.flush()
            };

            if !remaining.is_empty() {
                super::chat::AnthropicChatCapability::parse_sse_event(&remaining)
            } else {
                None
            }
        })
        .filter_map(|result| async move { result });

        let final_stream = decoded_stream.chain(flush_stream);
        Ok(Box::pin(final_stream))
    }
}

#[async_trait]
impl ModelListingCapability for AnthropicClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

impl LlmClient for AnthropicClient {
    fn provider_name(&self) -> &'static str {
        "anthropic"
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
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_client_creation() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        );

        assert_eq!(client.provider_name(), "anthropic");
        assert!(!client.supported_models().is_empty());
    }

    #[test]
    fn test_anthropic_client_with_specific_params() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        )
        .with_beta_features(vec!["feature1".to_string(), "feature2".to_string()])
        .with_thinking_enabled()
        .with_ephemeral_cache();

        assert_eq!(client.specific_params().beta_features.len(), 2);
        assert!(client.specific_params().thinking_config.is_some());
        assert!(
            client
                .specific_params()
                .thinking_config
                .as_ref()
                .unwrap()
                .is_enabled()
        );
        assert!(client.specific_params().cache_control.is_some());
    }

    #[test]
    fn test_anthropic_client_beta_features() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        )
        .add_beta_feature("computer-use-2024-10-22".to_string())
        .add_beta_feature("prompt-caching-2024-07-31".to_string());

        assert_eq!(client.specific_params().beta_features.len(), 2);
        assert!(
            client
                .specific_params()
                .beta_features
                .contains(&"computer-use-2024-10-22".to_string())
        );
        assert!(
            client
                .specific_params()
                .beta_features
                .contains(&"prompt-caching-2024-07-31".to_string())
        );
    }
}
