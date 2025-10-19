//! Custom Provider Example
//!
//! This example demonstrates how to create a custom AI provider implementation
//! that can integrate with any AI API. We'll create a mock provider that simulates
//! a real AI service with custom features.

use async_trait::async_trait;
use futures::{StreamExt, stream};
use serde_json::json;
use siumai::custom_provider::*;
use siumai::error::LlmError;
use siumai::prelude::*;
use siumai::stream::{ChatStream, ChatStreamEvent};

use std::time::Duration;

/// Example custom provider that simulates a fictional AI service
#[derive(Debug, Clone)]
pub struct ExampleCustomProvider {
    config: CustomProviderConfig,
    _http_client: reqwest::Client,
}

impl ExampleCustomProvider {
    /// Create a new instance of the custom provider
    pub fn new(config: CustomProviderConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout.unwrap_or(30)))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            _http_client: http_client,
        }
    }

    /// Convert messages to the custom API format
    fn convert_messages(&self, messages: &[ChatMessage]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|msg| {
                json!({
                    "role": match msg.role {
                        MessageRole::System => "system",
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::Developer => "system",
                        MessageRole::Tool => "tool",
                    },
                    "content": match &msg.content {
                        MessageContent::Text(text) => json!(text),
                        MessageContent::MultiModal(parts) => {
                            let content_parts: Vec<serde_json::Value> = parts.iter().map(|part| {
                                match part {
                                    ContentPart::Text { text } => json!({
                                        "type": "text",
                                        "text": text
                                    }),
                                    ContentPart::Image { image_url, detail } => json!({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url,
                                            "detail": detail.as_deref().unwrap_or("auto")
                                        }
                                    }),
                                    ContentPart::Audio { audio_url, format } => json!({
                                        "type": "audio",
                                        "audio_url": audio_url,
                                        "format": format
                                    }),
                                }
                            }).collect();
                            json!(content_parts)
                        }
                        #[cfg(feature = "structured-messages")]
                        MessageContent::Json(v) => json!(v),
                    }
                })
            })
            .collect()
    }

    /// Build the request payload for the custom API
    fn build_request_payload(&self, request: &CustomChatRequest) -> serde_json::Value {
        let mut payload = json!({
            "model": request.model,
            "messages": self.convert_messages(&request.messages),
            "stream": request.stream,
        });

        // Add custom parameters from the request
        for (key, value) in &request.params {
            payload[key] = value.clone();
        }

        // Add provider-specific parameters
        for (key, value) in &self.config.custom_params {
            payload[key] = value.clone();
        }

        payload
    }

    /// Simulate an API response (in real implementation, this would make HTTP requests)
    async fn simulate_api_call(
        &self,
        _payload: serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Return a mock response that looks like a real AI API response
        Ok(json!({
            "id": "custom-response-123",
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": "custom-model-v1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a response from the custom AI provider! 🚀\n\nI can handle various types of requests and provide intelligent responses. This example demonstrates how to integrate any AI API with the siumai library."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 45,
                "total_tokens": 70
            }
        }))
    }

    /// Parse the API response into our standard format
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
            .unwrap_or("No response content")
            .to_string();

        let finish_reason = response_data
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(|reason| reason.as_str())
            .map(|s| s.to_string());

        let usage = response_data.get("usage").map(|usage_data| Usage {
            prompt_tokens: usage_data
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0),
            completion_tokens: usage_data
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0),
            total_tokens: usage_data
                .get("total_tokens")
                .and_then(|v| v.as_u64())
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

        // Add custom metadata
        response = response.with_metadata("provider", "example-custom");
        response = response.with_metadata("api_version", "v1.0");

        Ok(response)
    }
}

#[async_trait]
impl CustomProvider for ExampleCustomProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "custom-model-v1".to_string(),
            "custom-model-v2".to_string(),
            "custom-model-fast".to_string(),
            "custom-model-accurate".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("custom_reasoning", true)
            .with_custom_feature("fast_inference", true)
    }

    async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError> {
        // Validate the model
        if !self.supported_models().contains(&request.model) {
            return Err(LlmError::InvalidParameter(format!(
                "Model '{}' is not supported by this provider",
                request.model
            )));
        }

        let payload = self.build_request_payload(&request);

        // In a real implementation, you would make an HTTP request here
        // let response = self.http_client
        //     .post(&format!("{}/chat/completions", self.config.base_url))
        //     .header("Authorization", format!("Bearer {}", self.config.api_key))
        //     .json(&payload)
        //     .send()
        //     .await?;

        // For this example, we simulate the API call
        let response_data = self.simulate_api_call(payload).await?;
        self.parse_response(response_data)
    }

    async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError> {
        // Get the full response first
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
            .chain(std::iter::once(Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("stream-end".to_string()),
                    content: MessageContent::Text("".to_string()),
                    model: Some("custom-model-v1".to_string()),
                    usage: None,
                    finish_reason: Some(FinishReason::Stop),
                    tool_calls: None,
                    thinking: None,
                    metadata: std::collections::HashMap::new(),
                },
            })))
            .collect();

        // Create a stream with simulated delays
        let stream = stream::iter(events).then(|event| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            event
        });

        Ok(Box::pin(stream))
    }

    fn validate_config(&self, config: &CustomProviderConfig) -> Result<(), LlmError> {
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

        // Add custom validation logic here
        if !config.base_url.starts_with("https://") {
            return Err(LlmError::InvalidParameter(
                "Base URL must use HTTPS".to_string(),
            ));
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Custom Provider Example");
    println!("==========================\n");

    // Step 1: Create provider configuration
    println!("1. Creating custom provider configuration...");
    let mut config = CustomProviderConfig::new(
        "example-ai",
        "https://api.example-ai.com/v1",
        "your-api-key-here",
    );

    // Set the default model
    config = config.with_model("custom-model-v1");

    // Add custom headers
    config = config.with_header("User-Agent", "siumai-example/1.0");
    config = config.with_header("X-Custom-Header", "example-value");

    // Add custom parameters
    config = config.with_param("temperature", 0.7);
    config = config.with_param("max_tokens", 1000);
    config = config.with_param("custom_feature", true);

    // Set timeout
    config = config.with_timeout(60);

    println!("✅ Configuration created for provider: {}", config.name);

    // Step 2: Create the custom provider
    println!("\n2. Creating custom provider instance...");
    let provider = Box::new(ExampleCustomProvider::new(config.clone()));
    println!(
        "✅ Provider created with capabilities: {:?}",
        provider.capabilities()
    );

    // Step 3: Create the client
    println!("\n3. Creating custom provider client...");
    let client = CustomProviderClient::new(provider, config)?;
    println!("✅ Client created successfully");

    // Step 4: Test basic chat functionality
    println!("\n4. Testing basic chat functionality...");
    let messages = vec![
        system!("You are a helpful AI assistant created by Example AI."),
        user!("Hello! Can you tell me about custom AI providers?"),
    ];

    let response = client.chat_with_tools(messages.clone(), None).await?;
    println!(
        "🤖 Response: {}",
        response.content.text().unwrap_or("No content")
    );

    if let Some(usage) = &response.usage {
        println!(
            "📊 Usage: {} prompt + {} completion = {} total tokens",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    // Step 5: Test streaming functionality
    println!("\n5. Testing streaming functionality...");
    let stream_messages = vec![user!(
        "Can you explain the benefits of custom providers in a streaming response?"
    )];

    let mut stream = client.chat_stream(stream_messages, None).await?;
    print!("🌊 Streaming response: ");

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                print!("{delta}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ChatStreamEvent::StreamEnd { .. } => {
                println!("\n✅ Streaming completed");
                break;
            }
            _ => {} // Handle other event types as needed
        }
    }

    // Step 6: Demonstrate error handling
    println!("\n6. Testing error handling...");
    let invalid_messages = vec![user!("Test with unsupported model")];

    // Try to use an unsupported model
    let mut invalid_request =
        CustomChatRequest::new(invalid_messages, "unsupported-model".to_string());
    invalid_request.stream = false;

    match client.provider().chat(invalid_request).await {
        Ok(_) => println!("❌ Expected error but got success"),
        Err(e) => println!("✅ Correctly handled error: {e}"),
    }

    // Step 7: Demonstrate different model configurations
    println!("\n7. Testing different model configurations...");

    // Create a configuration with a different model
    let fast_config = CustomProviderConfig::new(
        "example-ai-fast",
        "https://api.example-ai.com/v1",
        "your-api-key-here",
    )
    .with_model("custom-model-fast")
    .with_param("speed_mode", true);

    let fast_provider = Box::new(ExampleCustomProvider::new(fast_config.clone()));
    let fast_client = CustomProviderClient::new(fast_provider, fast_config)?;

    let fast_messages = vec![user!("Quick question: What's 2+2?")];

    let fast_response = fast_client.chat_with_tools(fast_messages, None).await?;
    println!(
        "🚀 Fast model response: {}",
        fast_response.content.text().unwrap_or("No content")
    );

    println!("\n🎉 Custom provider example completed successfully!");
    println!("\n💡 Key takeaways:");
    println!("   • Custom providers allow integration with any AI API");
    println!("   • Implement the CustomProvider trait for your specific API");
    println!("   • Handle both regular and streaming responses");
    println!("   • Add proper error handling and validation");
    println!("   • Support custom parameters and headers");
    println!("   • Configure different models for different use cases");

    Ok(())
}
