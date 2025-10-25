//! Custom ProviderSpec Implementation Example
//!
//! This example demonstrates how to create a custom provider by implementing
//! the ProviderSpec trait. This is useful when:
//! - Adding support for a new LLM provider
//! - Creating a mock provider for testing
//! - Wrapping an existing provider with custom behavior
//!
//! Run with: cargo run --example custom_provider_spec --features openai

use siumai::core::{ChatTransformers, ProviderContext, ProviderSpec};
use siumai::error::LlmError;
use siumai::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
use siumai::traits::ProviderCapabilities;
use siumai::execution::transformers::request::RequestTransformer;
use siumai::execution::transformers::response::ResponseTransformer;
use siumai::types::{ChatMessage, ChatRequest, ChatResponse};
use std::sync::Arc;

/// Custom provider spec that wraps OpenAI but adds custom behavior
struct CustomProviderSpec {
    /// Custom prefix to add to all prompts
    prompt_prefix: String,
}

impl CustomProviderSpec {
    fn new(prompt_prefix: String) -> Self {
        Self { prompt_prefix }
    }
}

impl ProviderSpec for CustomProviderSpec {
    fn id(&self) -> &'static str {
        "custom-provider"
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

        // Add API key if present
        if let Some(ref api_key) = ctx.api_key {
            headers.insert(
                HeaderName::from_static("authorization"),
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| LlmError::InvalidParameter(e.to_string()))?,
            );
        }

        // Add custom headers
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
        // Use OpenAI-compatible endpoint
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        ChatTransformers {
            request: Arc::new(CustomRequestTransformer {
                prompt_prefix: self.prompt_prefix.clone(),
            }),
            response: Arc::new(CustomResponseTransformer),
            stream: None, // Use default stream transformer
            json: None,
        }
    }
}

/// Custom request transformer that adds a prefix to prompts
struct CustomRequestTransformer {
    prompt_prefix: String,
}

impl RequestTransformer for CustomRequestTransformer {
    fn provider_id(&self) -> &str {
        "custom-provider"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Reuse OpenAI transformer as base
        let openai_tx = siumai::providers::openai::transformers::request::OpenAiRequestTransformer;
        let mut body = openai_tx.transform_chat(req)?;

        // Modify the first user message to add our prefix
        if let Some(messages) = body.get_mut("messages").and_then(|v| v.as_array_mut()) {
            for msg in messages.iter_mut() {
                if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                    if let Some(content) = msg.get_mut("content").and_then(|c| c.as_str()) {
                        let new_content = format!("{}\n\n{}", self.prompt_prefix, content);
                        msg["content"] = serde_json::json!(new_content);
                        break; // Only modify the first user message
                    }
                }
            }
        }

        Ok(body)
    }
}

/// Custom response transformer (just uses OpenAI format)
struct CustomResponseTransformer;

impl ResponseTransformer for CustomResponseTransformer {
    fn provider_id(&self) -> &str {
        "custom-provider"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Reuse OpenAI transformer
        let openai_tx =
            siumai::providers::openai::transformers::response::OpenAiResponseTransformer;
        openai_tx.transform_chat_response(raw)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Custom ProviderSpec Example\n");

    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Create custom spec with a prompt prefix
    let spec = Arc::new(CustomProviderSpec::new(
        "You are a helpful assistant that always responds in a friendly tone.".to_string(),
    ));

    // Create provider context
    let ctx = ProviderContext::new(
        "custom-provider",
        "https://api.openai.com/v1".to_string(),
        Some(api_key),
        Default::default(),
    );

    // Get transformers from spec
    let transformers = spec.choose_chat_transformers(&ChatRequest::new(vec![]), &ctx);

    // Create executor with our custom spec
    let executor = HttpChatExecutor {
        provider_id: "custom-provider".to_string(),
        http_client: reqwest::Client::new(),
        request_transformer: transformers.request,
        response_transformer: transformers.response,
        stream_transformer: transformers.stream,
        json_stream_converter: transformers.json,
        stream_disable_compression: false,
        interceptors: vec![],
        middlewares: vec![],
        provider_spec: spec,
        provider_context: ctx,
        before_send: None,
        retry_options: None,
    };

    // Create a chat request
    let request = ChatRequest::new(vec![ChatMessage::user("Hello! How are you?").build()])
        .with_model_params(
            siumai::types::CommonParamsBuilder::new()
                .model("gpt-4o-mini")
                .build()
                .unwrap(),
        );

    println!("📤 Sending request with custom prompt prefix...\n");

    // Execute the request
    let response = executor.execute(request).await?;

    println!("📥 Response:");
    if let siumai::types::MessageContent::Text(text) = &response.content {
        println!("{}\n", text);
    } else {
        println!("{:?}\n", response.content);
    }

    println!("✅ Custom ProviderSpec example completed!");
    println!("\n💡 Key Points:");
    println!("   - Implemented ProviderSpec trait for custom behavior");
    println!("   - Created custom request transformer to modify prompts");
    println!("   - Reused OpenAI transformers as base implementation");
    println!("   - Used HttpChatExecutor with custom spec");

    Ok(())
}
