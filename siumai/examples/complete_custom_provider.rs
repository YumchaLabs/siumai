//! Complete Custom Provider Example
//!
//! This example shows a complete implementation of a custom provider,
//! including:
//! - ProviderSpec implementation
//! - Custom transformers
//! - Builder pattern for easy configuration
//! - Integration with the executor system
//!
//! This demonstrates the full pattern you would follow to add support
//! for a new LLM provider to Siumai.
//!
//! Run with: cargo run --example complete_custom_provider --features openai

use siumai::core::{ChatTransformers, ProviderContext, ProviderSpec};
use siumai::error::LlmError;
use siumai::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
use siumai::execution::transformers::request::RequestTransformer;
use siumai::execution::transformers::response::ResponseTransformer;
use siumai::traits::ProviderCapabilities;
use siumai::types::{ChatMessage, ChatRequest, ChatResponse};
use std::sync::Arc;

// ============================================================================
// Step 1: Define Provider Configuration
// ============================================================================

/// Configuration for MyCustomProvider
#[derive(Clone)]
pub struct MyCustomProviderConfig {
    pub api_key: String,
    pub base_url: String,
    pub custom_setting: Option<String>,
}

impl MyCustomProviderConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com/v1".to_string(), // Using OpenAI as example
            custom_setting: None,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_custom_setting(mut self, setting: String) -> Self {
        self.custom_setting = Some(setting);
        self
    }
}

// ============================================================================
// Step 2: Implement ProviderSpec
// ============================================================================

/// ProviderSpec for MyCustomProvider
pub struct MyCustomProviderSpec {
    config: MyCustomProviderConfig,
}

impl MyCustomProviderSpec {
    pub fn new(config: MyCustomProviderConfig) -> Self {
        Self { config }
    }
}

impl ProviderSpec for MyCustomProviderSpec {
    fn id(&self) -> &'static str {
        "my-custom-provider"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

        let mut headers = HeaderMap::new();

        // Add API key
        if let Some(ref api_key) = ctx.api_key {
            headers.insert(
                HeaderName::from_static("authorization"),
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| LlmError::InvalidParameter(e.to_string()))?,
            );
        }

        // Add custom header if configured
        if let Some(ref setting) = self.config.custom_setting {
            headers.insert(
                HeaderName::from_static("x-custom-setting"),
                HeaderValue::from_str(setting)
                    .map_err(|e| LlmError::InvalidParameter(e.to_string()))?,
            );
        }

        // Add any extra headers from context
        for (key, value) in &ctx.http_extra_headers {
            headers.insert(
                HeaderName::from_bytes(key.as_bytes())
                    .map_err(|e| LlmError::InvalidParameter(e.to_string()))?,
                HeaderValue::from_str(value)
                    .map_err(|e| LlmError::InvalidParameter(e.to_string()))?,
            );
        }

        Ok(headers)
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        ChatTransformers {
            request: Arc::new(MyCustomRequestTransformer),
            response: Arc::new(MyCustomResponseTransformer),
            stream: None, // Could implement custom stream transformer
            json: None,
        }
    }
}

// ============================================================================
// Step 3: Implement Request Transformer
// ============================================================================

struct MyCustomRequestTransformer;

impl RequestTransformer for MyCustomRequestTransformer {
    fn provider_id(&self) -> &str {
        "my-custom-provider"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // For this example, we'll use OpenAI format as base
        // In a real implementation, you would transform to your provider's format
        let openai_tx = siumai::providers::openai::transformers::request::OpenAiRequestTransformer;
        let body = openai_tx.transform_chat(req)?;

        // You could add custom transformations here
        // For example, adding provider-specific fields

        Ok(body)
    }
}

// ============================================================================
// Step 4: Implement Response Transformer
// ============================================================================

struct MyCustomResponseTransformer;

impl ResponseTransformer for MyCustomResponseTransformer {
    fn provider_id(&self) -> &str {
        "my-custom-provider"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        // For this example, we'll use OpenAI format as base
        // In a real implementation, you would parse your provider's response format
        let openai_tx =
            siumai::providers::openai::transformers::response::OpenAiResponseTransformer;
        openai_tx.transform_chat_response(raw)
    }
}

// ============================================================================
// Step 5: Create Client with Builder Pattern
// ============================================================================

/// Client for MyCustomProvider
pub struct MyCustomProviderClient {
    config: MyCustomProviderConfig,
    http_client: reqwest::Client,
}

impl MyCustomProviderClient {
    pub fn new(config: MyCustomProviderConfig) -> Self {
        Self {
            config,
            http_client: reqwest::Client::new(),
        }
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }

    /// Execute a chat request
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let spec = Arc::new(MyCustomProviderSpec::new(self.config.clone()));
        let ctx = ProviderContext::new(
            "my-custom-provider",
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            Default::default(),
        );

        // Get transformers from spec
        let transformers = spec.choose_chat_transformers(&request, &ctx);

        let executor = HttpChatExecutor {
            provider_id: "my-custom-provider".to_string(),
            http_client: self.http_client.clone(),
            request_transformer: transformers.request,
            response_transformer: transformers.response,
            stream_transformer: transformers.stream,
            json_stream_converter: transformers.json,
            policy: siumai::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(false),
            middlewares: vec![],
            provider_spec: spec,
            provider_context: ctx,
        };

        executor.execute(request).await
    }
}

// ============================================================================
// Example Usage
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Complete Custom Provider Example\n");

    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Step 1: Create configuration
    let config =
        MyCustomProviderConfig::new(api_key).with_custom_setting("my-custom-value".to_string());

    // Step 2: Create client
    let client = MyCustomProviderClient::new(config);

    // Step 3: Create request
    let request = ChatRequest::new(vec![
        ChatMessage::user("What is the capital of France?").build(),
    ])
    .with_model_params(
        siumai::types::CommonParamsBuilder::new()
            .model("gpt-4o-mini")
            .build()
            .unwrap(),
    );

    println!("📤 Sending request...\n");

    // Step 4: Execute request
    let response = client.chat(request).await?;

    println!("📥 Response:");
    if let siumai::types::MessageContent::Text(text) = &response.content {
        println!("{}\n", text);
    } else {
        println!("{:?}\n", response.content);
    }

    println!("✅ Complete custom provider example finished!\n");
    println!("📚 Implementation Steps:");
    println!("   1. ✅ Defined provider configuration");
    println!("   2. ✅ Implemented ProviderSpec trait");
    println!("   3. ✅ Created request transformer");
    println!("   4. ✅ Created response transformer");
    println!("   5. ✅ Built client with builder pattern");
    println!("\n💡 To add a real provider:");
    println!("   - Implement transformers for provider's API format");
    println!("   - Add provider-specific parameters");
    println!("   - Implement streaming if supported");
    println!("   - Add comprehensive error handling");
    println!("   - Write tests using mock HTTP servers");

    Ok(())
}
