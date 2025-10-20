//! Custom Provider Implementation Guide
//!
//! This module provides comprehensive documentation and examples for implementing
//! custom AI providers using the siumai library framework.

use crate::custom_provider::*;
use crate::error::LlmError;
use crate::stream::ChatStream;
// Note: types are used in the examples and implementations below
#[allow(unused_imports)]
use crate::types::*;
use async_trait::async_trait;
use serde_json;

/// # Custom Provider Implementation Guide
///
/// This guide shows you how to implement a custom AI provider for the siumai library.
///
/// ## Step 1: Implement the `CustomProvider` trait
///
/// ```rust,no_run
/// use siumai::prelude::*;
/// use async_trait::async_trait;
///
/// pub struct MyCustomProvider {
///     name: String,
///     base_url: String,
///     api_key: String,
/// }
///
/// #[async_trait]
/// impl CustomProvider for MyCustomProvider {
///     fn name(&self) -> &str {
///         &self.name
///     }
///
///     fn supported_models(&self) -> Vec<String> {
///         vec!["my-model-v1".to_string(), "my-model-v2".to_string()]
///     }
///
///     fn capabilities(&self) -> ProviderCapabilities {
///         ProviderCapabilities::new()
///             .with_chat()
///             .with_streaming()
///             .with_tools()
///     }
///
///     async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError> {
///         // Implement your API call here
///         todo!()
///     }
///
///     async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError> {
///         // Implement streaming API call here
///         todo!()
///     }
/// }
/// ```
///
/// ## Step 2: Create a configuration and client
///
/// ```rust,ignore
/// # use siumai::prelude::*;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = CustomProviderConfig::new(
///     "my-provider",
///     "https://api.myprovider.com/v1",
///     "your-api-key"
/// )
/// .with_model("my-model-v1")
/// .with_header("User-Agent", "my-app/1.0")
/// .with_timeout(30)
/// .with_param("temperature", 0.7);
///
/// let provider = Box::new(MyCustomProvider::new(config.clone()));
/// let client = CustomProviderClient::new(provider, config)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Step 3: Use the client
///
/// ```rust,no_run
/// # use siumai::prelude::*;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = quick_openai().await?;
/// let messages = vec![user!("Hello, how are you?")];
/// let response = client.chat_with_tools(messages, None).await?;
/// println!("Response: {}", response.content.text().unwrap_or(""));
/// # Ok(())
/// # }
/// ```
/// Example: Hugging Face Provider
///
/// This example shows how to implement a provider for Hugging Face's Inference API
pub struct HuggingFaceProvider {
    http_client: reqwest::Client,
    config: CustomProviderConfig,
}

impl HuggingFaceProvider {
    pub fn new(config: CustomProviderConfig) -> Self {
        let http_client = reqwest::Client::new();
        Self {
            http_client,
            config,
        }
    }

    /// Convert messages to Hugging Face format
    fn convert_messages(&self, messages: &[ChatMessage]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": match msg.role {
                        MessageRole::System => "system",
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::Developer => "system", // Developer messages are treated as system messages
                        MessageRole::Tool => "tool",
                    },
                    "content": match &msg.content {
                        MessageContent::Text(text) => text,
                        MessageContent::MultiModal(_) => "[multimodal content not supported]",
                    }
                })
            })
            .collect()
    }

    /// Build request payload
    fn build_request_payload(&self, request: &CustomChatRequest) -> serde_json::Value {
        let mut payload = serde_json::json!({
            "model": request.model,
            "messages": self.convert_messages(&request.messages),
            "stream": request.stream,
        });

        // Add custom parameters
        for (key, value) in &request.params {
            payload[key] = value.clone();
        }

        payload
    }

    /// Parse response from Hugging Face API
    fn parse_response(
        &self,
        response_data: serde_json::Value,
    ) -> Result<CustomChatResponse, LlmError> {
        let content = response_data
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|content| content.as_str())
            .unwrap_or("")
            .to_string();

        let finish_reason = response_data
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(|reason| reason.as_str())
            .map(std::string::ToString::to_string);

        let usage = response_data.get("usage").map(|usage_data| Usage {
            prompt_tokens: usage_data
                .get("prompt_tokens")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0),
            completion_tokens: usage_data
                .get("completion_tokens")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0),
            total_tokens: usage_data
                .get("total_tokens")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0),
            reasoning_tokens: None,
            cached_tokens: None,
        });

        let mut response = CustomChatResponse::new(content);

        if let Some(reason) = finish_reason {
            response = response.with_finish_reason(reason);
        }

        if let Some(usage) = usage {
            response = response.with_usage(usage);
        }

        Ok(response)
    }
}

#[async_trait]
impl CustomProvider for HuggingFaceProvider {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "microsoft/DialoGPT-medium".to_string(),
            "microsoft/DialoGPT-large".to_string(),
            "facebook/blenderbot-400M-distill".to_string(),
            "facebook/blenderbot-1B-distill".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat().with_streaming()
    }

    async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let payload = self.build_request_payload(&request);

        let mut req_builder = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        // Add custom headers
        for (key, value) in &self.config.headers {
            req_builder = req_builder.header(key, value);
        }

        let response = req_builder
            .json(&payload)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("Hugging Face API error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response(response_data)
    }

    async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError> {
        // For this example, we'll implement a simple streaming simulation
        // In practice, you'd handle Server-Sent Events (SSE) from the API

        use crate::stream::ChatStreamEvent;
        use futures::stream;

        let response = self.chat(request).await?;

        // Simulate streaming by splitting the response into chunks
        let content = response.content;
        let words: Vec<&str> = content.split_whitespace().collect();

        let events: Vec<Result<ChatStreamEvent, LlmError>> = words
            .into_iter()
            .enumerate()
            .map(|(i, word)| {
                let delta = if i == 0 {
                    word.to_string()
                } else {
                    format!(" {word}")
                };
                Ok(ChatStreamEvent::ContentDelta { delta, index: None })
            })
            .collect();

        let stream = stream::iter(events);
        Ok(Box::pin(stream))
    }

    fn validate_config(&self, config: &CustomProviderConfig) -> Result<(), LlmError> {
        // Call the default validation first
        if config.name.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Provider name cannot be empty".to_string(),
            ));
        }
        if config.base_url.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Base URL cannot be empty".to_string(),
            ));
        }
        if config.api_key.is_empty() {
            return Err(LlmError::InvalidParameter(
                "API key cannot be empty".to_string(),
            ));
        }

        // Add Hugging Face-specific validation
        if !config.base_url.contains("huggingface") && !config.base_url.contains("hf.co") {
            return Err(LlmError::InvalidParameter(
                "Base URL should be a Hugging Face endpoint".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for Hugging Face provider
pub struct HuggingFaceProviderBuilder {
    config: Option<CustomProviderConfig>,
}

impl Default for HuggingFaceProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HuggingFaceProviderBuilder {
    pub const fn new() -> Self {
        Self { config: None }
    }

    pub fn with_config(mut self, config: CustomProviderConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_api_key<S: Into<String>>(self, api_key: S) -> Self {
        let config = CustomProviderConfig::new(
            "huggingface",
            "https://api-inference.huggingface.co/models",
            &api_key.into(),
        );
        self.with_config(config)
    }
}

impl CustomProviderBuilder for HuggingFaceProviderBuilder {
    fn build(self) -> Result<Box<dyn CustomProvider>, LlmError> {
        let config = self
            .config
            .ok_or_else(|| LlmError::ConfigurationError("Configuration is required".to_string()))?;

        let provider = HuggingFaceProvider::new(config);
        Ok(Box::new(provider))
    }
}

/// Utility functions for custom provider development
pub mod utils {
    use super::*;

    /// Convert standard `ChatMessage` to a generic JSON format
    pub fn message_to_json(message: &ChatMessage) -> serde_json::Value {
        serde_json::json!({
            "role": match message.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Developer => "system", // Developer messages are treated as system messages
                MessageRole::Tool => "tool",
            },
            "content": match &message.content {
                MessageContent::Text(text) => serde_json::Value::String(text.clone()),
                MessageContent::MultiModal(parts) => {
                    let content_parts: Vec<serde_json::Value> = parts.iter().map(|part| {
                        match part {
                            ContentPart::Text { text } => serde_json::json!({
                                "type": "text",
                                "text": text
                            }),
                            ContentPart::Image { image_url, detail } => serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": detail.as_deref().unwrap_or("auto")
                                }
                            }),
                            ContentPart::Audio { audio_url, format } => serde_json::json!({
                                "type": "audio",
                                "audio_url": audio_url,
                                "format": format
                            }),
                        }
                    }).collect();
                    serde_json::Value::Array(content_parts)
                }
            }
        })
    }

    /// Create a simple error response
    pub fn create_error_response(error_message: &str) -> CustomChatResponse {
        CustomChatResponse::new(format!("Error: {error_message}"))
            .with_finish_reason("error")
            .with_metadata("error", true)
    }

    /// Validate model name against supported models
    pub fn validate_model(model: &str, supported_models: &[String]) -> Result<(), LlmError> {
        if !supported_models.contains(&model.to_string()) {
            return Err(LlmError::InvalidParameter(format!(
                "Model '{}' is not supported. Supported models: {}",
                model,
                supported_models.join(", ")
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_provider_creation() {
        let config = CustomProviderConfig::new(
            "huggingface",
            "https://api-inference.huggingface.co/models",
            "test-key",
        );

        let provider = HuggingFaceProvider::new(config);
        assert_eq!(provider.name(), "huggingface");
        assert!(!provider.supported_models().is_empty());
    }

    #[test]
    fn test_message_to_json_conversion() {
        let message = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };

        let json = utils::message_to_json(&message);
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello");
    }
}
