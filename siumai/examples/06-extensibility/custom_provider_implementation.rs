//! Custom Complete Provider
//!
//! This example demonstrates how to implement a completely custom AI provider.
//! Suitable for private APIs, non-OpenAI-compatible APIs, and more.
//!
//! # Use Cases
//!
//! - Private/internal AI services
//! - Non-OpenAI-compatible APIs
//! - Custom-protocol AI services
//! - Local model services
//!
//! # Run
//!
//! ```bash
//! cargo run --example custom_provider_implementation
//! ```

use async_trait::async_trait;
use futures::stream;
use siumai::custom_provider::{CustomChatRequest, CustomChatResponse, CustomProvider};
use siumai::prelude::*;

/// Custom provider example — simulates a simple AI service
#[derive(Clone)]
pub struct MyCustomProvider {
    /// API key
    api_key: String,
    /// Base URL
    base_url: String,
}

impl MyCustomProvider {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self { api_key, base_url }
    }
}

#[async_trait]
impl CustomProvider for MyCustomProvider {
    fn name(&self) -> &str {
        "my-custom-provider"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "custom-model-v1".to_string(),
            "custom-model-v2".to_string(),
            "custom-model-pro".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError> {
        println!("📤 Sending request to custom provider");
        println!("   API Key: {}...", &self.api_key[..8]);
        println!("   Base URL: {}", self.base_url);
        println!("   Model: {}", request.model);
        println!("   Messages: {}", request.messages.len());

        // In a real implementation, call the actual API here

        // Simulate API response
        let response_text = format!(
            "This is a response from {}. I received your message.",
            self.name()
        );

        use std::collections::HashMap;
        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), serde_json::json!(self.name()));
        metadata.insert("model".to_string(), serde_json::json!(request.model));

        Ok(CustomChatResponse {
            content: response_text,
            finish_reason: Some("stop".to_string()),
            usage: Some(Usage::new(10, 20)),
            tool_calls: None,
            metadata,
        })
    }

    async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError> {
        println!("📤 Sending streaming request to custom provider");
        println!("   Model: {}", request.model);

        // Simulate streaming response
        let chunks = vec![
            "This ",
            "is ",
            "a ",
            "streaming ",
            "response ",
            "from ",
            "custom ",
            "provider.",
        ];

        let events: Vec<Result<ChatStreamEvent, LlmError>> = chunks
            .into_iter()
            .map(|chunk| {
                Ok(ChatStreamEvent::ContentDelta {
                    delta: chunk.to_string(),
                    index: None,
                })
            })
            .collect();

        Ok(Box::pin(stream::iter(events)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Custom Complete Provider Example\n");

    // ============================================================
    // Example 1: Create custom provider
    // ============================================================
    println!("📝 Example 1: Create custom provider");
    println!("======================================\n");

    let provider = MyCustomProvider::new(
        "sk-custom-api-key-12345".to_string(),
        "https://api.my-custom-ai.com".to_string(),
    );

    println!("✅ Created custom provider:");
    println!("   Name: {}", provider.name());
    println!("   Supported models: {:?}", provider.supported_models());
    println!("   Capabilities: {:?}", provider.capabilities());
    println!();

    // ============================================================
    // Example 2: Use custom provider for chat
    // ============================================================
    println!("📝 Example 2: Use custom provider for chat");
    println!("============================================\n");

    let request = CustomChatRequest::new(
        vec![ChatMessage::user("Hello! Please introduce yourself.").build()],
        "custom-model-v1".to_string(),
    );

    let response = provider.chat(request).await?;

    println!("\n✅ Received response:");
    println!("   {}", response.content);
    println!("   Token usage: {:?}", response.usage);
    println!();

    // ============================================================
    // Example 3: Use streaming response
    // ============================================================
    println!("📝 Example 3: Use streaming response");
    println!("======================================\n");

    let request = CustomChatRequest::new(
        vec![ChatMessage::user("Tell me a story").build()],
        "custom-model-pro".to_string(),
    );

    let mut stream = provider.chat_stream(request).await?;

    print!("📥 Streaming response: ");
    use futures::StreamExt;
    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::ContentDelta { delta, .. } = event? {
            print!("{}", delta);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
    println!("\n   ✅ Done");
    println!();

    // ============================================================
    // Implementation Points
    // ============================================================
    println!("💡 Implementation Points");
    println!("====================================\n");

    println!("1. Implement CustomProvider trait");
    println!("   - name(): Provider name");
    println!("   - supported_models(): Supported models");
    println!("   - capabilities(): Provider capabilities");
    println!("   - chat(): Synchronous chat");
    println!("   - chat_stream(): Streaming chat");
    println!();

    println!("2. Handle requests and responses");
    println!("   - Parse ChatRequest");
    println!("   - Call your API");
    println!("   - Convert to ChatResponse");
    println!();

    println!("3. Implement streaming (optional)");
    println!("   - Return Stream<Item = ChatStreamEvent>");
    println!("   - Send content chunk by chunk");
    println!();

    println!("🎉 Example complete!");
    println!();
    println!("📚 Key Takeaways:");
    println!("   1. CustomProvider allows completely custom AI providers");
    println!("   2. Works with any API protocol");
    println!("   3. Supports both sync and streaming responses");
    println!("   4. Full control over request and response format");

    Ok(())
}
