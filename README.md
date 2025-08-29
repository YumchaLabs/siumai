# Siumai - Unified LLM Interface Library for Rust

[![Crates.io](https://img.shields.io/crates/v/siumai.svg)](https://crates.io/crates/siumai)
[![Documentation](https://docs.rs/siumai/badge.svg)](https://docs.rs/siumai)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Siumai (烧卖) is a unified LLM interface library for Rust that provides a consistent API across multiple AI providers. It features capability-based trait separation, type-safe parameter handling, and comprehensive streaming support.

## 🎯 Two Ways to Use Siumai

Siumai offers two distinct approaches to fit your needs:

1. **`Provider`** - For provider-specific clients with access to all features
2. **`Siumai::builder()`** - For unified interface with provider-agnostic code

Choose `Provider` when you need provider-specific features, or `Siumai::builder()` when you want maximum portability.

## 🌟 Features

- **🔌 Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini, Ollama, and custom providers
- **🎯 Capability-Based Design**: Separate traits for chat, audio, vision, tools, and embeddings
- **🔧 Builder Pattern**: Fluent API with method chaining for easy configuration
- **🌊 Streaming Support**: Full streaming capabilities with event processing
- **🛡️ Type Safety**: Leverages Rust's type system for compile-time safety
- **🔄 Parameter Mapping**: Automatic translation between common and provider-specific parameters
- **📦 HTTP Customization**: Support for custom reqwest clients and HTTP configurations
- **🎨 Multimodal**: Support for text, images, and audio content
- **⚡ Async/Await**: Built on tokio for high-performance async operations
- **🔁 Retry Mechanisms**: Intelligent retry with exponential backoff and jitter
- **🛡️ Error Handling**: Advanced error classification with recovery suggestions
- **✅ Parameter Validation**: Cross-provider parameter validation and optimization

## 🚀 Quick Start

Add Siumai to your `Cargo.toml`:

```toml
[dependencies]
# By default, all providers are included
siumai = "0.9.1"
tokio = { version = "1.0", features = ["full"] }
```

### 🎛️ Feature Selection

Siumai allows you to include only the providers you need, reducing compilation time and binary size:

```toml
[dependencies]
# Only OpenAI
siumai = { version = "0.9.1", features = ["openai"] }

# Multiple specific providers
siumai = { version = "0.9.1", features = ["openai", "anthropic", "google"] }

# All providers (same as default)
siumai = { version = "0.9.1", features = ["all-providers"] }

# Only local AI (Ollama)
siumai = { version = "0.9.1", features = ["ollama"] }
```

#### Available Features

| Feature | Providers | Description |
|---------|-----------|-------------|
| `openai` | OpenAI + compatible | OpenAI, DeepSeek, OpenRouter, SiliconFlow |
| `anthropic` | Anthropic | Claude models with thinking mode |
| `google` | Google | Gemini models with multimodal capabilities |
| `ollama` | Ollama | Local AI models |
| `xai` | xAI | Grok models with reasoning |
| `groq` | Groq | Ultra-fast inference |
| `all-providers` | All | Complete provider support (default) |

### Provider-Specific Clients

Use `Provider` when you need access to provider-specific features:

```rust
// Cargo.toml: siumai = { version = "0.9.1", features = ["openai"] }
use siumai::models;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get a client specifically for OpenAI
    let openai_client = Provider::openai()
        .api_key("your-openai-key")
        .model(models::openai::GPT_4)
        .temperature(0.7)
        .build()
        .await?;

    // You can now call both standard and OpenAI-specific methods
    let response = openai_client.chat(vec![user!("Hello!")]).await?;
    // let assistant = openai_client.create_assistant(...).await?; // Example of specific feature

    println!("OpenAI says: {}", response.text().unwrap_or_default());
    Ok(())
}
```

### Unified Interface

Use `Siumai::builder()` when you want provider-agnostic code:

```rust
// Cargo.toml: siumai = { version = "0.9.1", features = ["anthropic"] }
use siumai::models;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a unified client, backed by Anthropic
    let client = Siumai::builder()
        .anthropic()
        .api_key("your-anthropic-key")
        .model(models::anthropic::CLAUDE_SONNET_3_5)
        .build()
        .await?;

    // Your code uses the standard Siumai interface
    let request = vec![user!("What is the capital of France?")];
    let response = client.chat(request).await?;

    // If you decide to switch to OpenAI, you only change the builder and feature.
    // The `.chat(request)` call remains identical.
    println!("The unified client says: {}", response.text().unwrap_or_default());
    Ok(())
}
```

> **💡 Feature Tip**: When using specific providers, make sure to enable the corresponding feature in your `Cargo.toml`. If you try to use a provider without its feature enabled, you'll get a compile-time error with a helpful message.

```rust
use siumai::models;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a unified client, backed by Anthropic
    let client = Siumai::builder()
        .anthropic()
        .api_key("your-anthropic-key")
        .model(models::anthropic::CLAUDE_SONNET_3_5)
        .build()
        .await?;

    // Your code uses the standard Siumai interface
    let request = vec![user!("What is the capital of France?")];
    let response = client.chat(request).await?;

    // If you decide to switch to OpenAI, you only change the builder.
    // The `.chat(request)` call remains identical.
    println!("The unified client says: {}", response.text().unwrap_or_default());
    Ok(())
}
```

### Multimodal Messages

```rust
use siumai::prelude::*;

// Create a message with text and image - use builder for complex messages
let message = ChatMessage::user("What do you see in this image?")
    .with_image("https://example.com/image.jpg".to_string(), Some("high".to_string()))
    .build();

let request = ChatRequest::builder()
    .messages(vec![message])
    .build();
```

### Streaming

```rust
use siumai::prelude::*;
use futures::StreamExt;

// Create a streaming request
let stream = client.chat_stream(request).await?;

// Process stream events
let response = collect_stream_response(stream).await?;
println!("Final response: {}", response.text().unwrap_or(""));
```

## 🏗️ Architecture

Siumai uses a capability-based architecture that separates different AI functionalities:

### Core Traits

- **`ChatCapability`**: Basic chat functionality
- **`AudioCapability`**: Text-to-speech and speech-to-text
- **`ImageGenerationCapability`**: Image generation, editing, and variations
- **`VisionCapability`**: Image analysis and understanding
- **`ToolCapability`**: Function calling and tool usage
- **`EmbeddingCapability`**: Text embeddings
- **`RerankCapability`**: Document reranking and relevance scoring

### Provider-Specific Traits

- **`OpenAiCapability`**: OpenAI-specific features (structured output, batch processing)
- **`AnthropicCapability`**: Anthropic-specific features (prompt caching, thinking mode)
- **`GeminiCapability`**: Google Gemini-specific features (search integration, code execution)

## 📚 Examples

### Different Providers

#### Provider-Specific Clients

```rust
use siumai::models;

// OpenAI - with provider-specific features
let openai_client = Provider::openai()
    .api_key("sk-...")
    .model(models::openai::GPT_4)
    .temperature(0.7)
    .build()
    .await?;

// Anthropic - with provider-specific features
let anthropic_client = Provider::anthropic()
    .api_key("sk-ant-...")
    .model(models::anthropic::CLAUDE_SONNET_3_5)
    .temperature(0.8)
    .build()
    .await?;

// Ollama - with provider-specific features
let ollama_client = Provider::ollama()
    .base_url("http://localhost:11434")
    .model(models::ollama::LLAMA_3_2)
    .temperature(0.7)
    .build()
    .await?;
```

#### Unified Interface

```rust
use siumai::models;

// OpenAI through unified interface
let openai_unified = Siumai::builder()
    .openai()
    .api_key("sk-...")
    .model(models::openai::GPT_4)
    .temperature(0.7)
    .build()
    .await?;

// Anthropic through unified interface
let anthropic_unified = Siumai::builder()
    .anthropic()
    .api_key("sk-ant-...")
    .model(models::anthropic::CLAUDE_SONNET_3_5)
    .temperature(0.8)
    .build()
    .await?;

// Ollama through unified interface
let ollama_unified = Siumai::builder()
    .ollama()
    .base_url("http://localhost:11434")
    .model(models::ollama::LLAMA_3_2)
    .temperature(0.7)
    .build()
    .await?;
```

### Custom HTTP Client

```rust
use siumai::models;
use std::time::Duration;

let custom_client = reqwest::Client::builder()
    .timeout(Duration::from_secs(60))
    .user_agent("my-app/1.0")
    .build()?;

// With provider-specific client
let client = Provider::openai()
    .api_key("your-key")
    .model(models::openai::GPT_4)
    .http_client(custom_client.clone())
    .build()
    .await?;

// With unified interface
let unified_client = Siumai::builder()
    .openai()
    .api_key("your-key")
    .model(models::openai::GPT_4)
    .http_client(custom_client)
    .build()
    .await?;
```

### Concurrent Usage with Clone

All clients support `Clone` for concurrent usage scenarios:

```rust
use siumai::prelude::*;
use std::sync::Arc;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client
    let client = Provider::openai()
        .api_key("your-key")
        .model(models::openai::GPT_4)
        .build()
        .await?;

    // Clone for concurrent usage
    let client_arc = Arc::new(client);
    let mut handles = vec![];

    for i in 0..5 {
        let client_clone = Arc::clone(&client_arc);
        let handle = task::spawn(async move {
            let messages = vec![user!(format!("Task {}: What is AI?", i))];
            client_clone.chat(messages).await
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let response = handle.await??;
        println!("Response: {}", response.text().unwrap_or_default());
    }

    Ok(())
}
```

#### Direct Clone Usage

```rust
// Clone clients directly (lightweight operation)
let client1 = Provider::openai()
    .api_key("your-key")
    .model(models::openai::GPT_4)
    .build()
    .await?;

let client2 = client1.clone(); // Shares HTTP client and configuration

// Both clients can be used independently
let response1 = client1.chat(vec![user!("Hello from client 1")]).await?;
let response2 = client2.chat(vec![user!("Hello from client 2")]).await?;
```

### Provider-Specific Features

```rust
use siumai::models;

// OpenAI with structured output (provider-specific client)
let openai_client = Provider::openai()
    .api_key("your-key")
    .model(models::openai::GPT_4)
    .response_format(ResponseFormat::JsonObject)
    .frequency_penalty(0.1)
    .build()
    .await?;

// Anthropic with caching (provider-specific client)
let anthropic_client = Provider::anthropic()
    .api_key("your-key")
    .model(models::anthropic::CLAUDE_SONNET_3_5)
    .cache_control(CacheControl::Ephemeral)
    .thinking_budget(1000)
    .build()
    .await?;

// Ollama with local model management (provider-specific client)
let ollama_client = Provider::ollama()
    .base_url("http://localhost:11434")
    .model(models::ollama::LLAMA_3_2)
    .keep_alive("10m")
    .num_ctx(4096)
    .num_gpu(1)
    .build()
    .await?;

// Unified interface with reasoning (works across all providers)
let unified_client = Siumai::builder()
    .anthropic()  // or .openai(), .ollama(), etc.
    .api_key("your-key")
    .model(models::anthropic::CLAUDE_SONNET_3_5)
    .temperature(0.7)
    .max_tokens(1000)
    .reasoning(true)        // ✅ Unified reasoning interface
    .reasoning_budget(5000) // ✅ Works across all providers
    .build()
    .await?;
```

### 🔄 Clone Support & Concurrent Usage

All siumai clients implement `Clone` for easy concurrent usage. The clone operation is lightweight as it shares the underlying HTTP client and configuration:

#### Basic Clone Usage

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Provider::openai()
        .api_key("your-key")
        .model("gpt-4")
        .build()
        .await?;

    // Clone is lightweight - shares HTTP client and config
    let client1 = client.clone();
    let client2 = client.clone();

    // All clients work independently
    let response1 = client1.chat(vec![user!("Hello from client 1")]).await?;
    let response2 = client2.chat(vec![user!("Hello from client 2")]).await?;

    Ok(())
}
```

#### Concurrent Processing with Arc

```rust
use siumai::prelude::*;
use std::sync::Arc;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Provider::anthropic()
        .api_key("your-key")
        .model("claude-3-sonnet-20240229")
        .build()
        .await?;

    // Use Arc for shared ownership across tasks
    let client_arc = Arc::new(client);
    let mut handles = vec![];

    // Process multiple requests concurrently
    for i in 0..5 {
        let client_clone = Arc::clone(&client_arc);
        let handle = task::spawn(async move {
            let messages = vec![user!(format!("Question {}: What is AI?", i))];
            client_clone.chat(messages).await
        });
        handles.push(handle);
    }

    // Collect all responses
    for handle in handles {
        let response = handle.await??;
        println!("Response: {}", response.text().unwrap_or_default());
    }

    Ok(())
}
```

#### Multi-Provider Concurrent Usage

```rust
use siumai::prelude::*;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create clients for different providers
    let openai_client = Provider::openai()
        .api_key("openai-key")
        .model("gpt-4")
        .build()
        .await?;

    let anthropic_client = Provider::anthropic()
        .api_key("anthropic-key")
        .model("claude-3-sonnet-20240229")
        .build()
        .await?;

    // Query multiple providers concurrently
    let openai_handle = task::spawn({
        let client = openai_client.clone();
        async move {
            client.chat(vec![user!("What is your name?")]).await
        }
    });

    let anthropic_handle = task::spawn({
        let client = anthropic_client.clone();
        async move {
            client.chat(vec![user!("What is your name?")]).await
        }
    });

    // Get responses from both providers
    let (openai_response, anthropic_response) =
        tokio::try_join!(openai_handle, anthropic_handle)?;

    println!("OpenAI: {}", openai_response?.text().unwrap_or_default());
    println!("Anthropic: {}", anthropic_response?.text().unwrap_or_default());

    Ok(())
}
```

> **Performance Note**: Clone operations are lightweight because:
> - HTTP clients use internal connection pooling (Arc-based)
> - Configuration parameters are small and cheap to clone
> - No duplicate network connections are created

### Advanced Features

#### Parameter Validation and Optimization

```rust
use siumai::models;
use siumai::params::EnhancedParameterValidator;

let params = CommonParams {
    model: models::openai::GPT_4.to_string(),
    temperature: Some(0.7),
    max_tokens: Some(1000),
    // ... other parameters
};

// Validate parameters for a specific provider
let validation_result = EnhancedParameterValidator::validate_for_provider(
    &params,
    &ProviderType::OpenAi,
)?;

// Optimize parameters for better performance
let mut optimized_params = params.clone();
let optimization_report = EnhancedParameterValidator::optimize_for_provider(
    &mut optimized_params,
    &ProviderType::OpenAi,
);
```

#### Retry Mechanisms

```rust
use siumai::retry::{RetryPolicy, RetryExecutor};

let policy = RetryPolicy::new()
    .with_max_attempts(3)
    .with_initial_delay(Duration::from_millis(1000))
    .with_backoff_multiplier(2.0);

let executor = RetryExecutor::new(policy);

let result = executor.execute(|| async {
    client.chat_with_tools(messages.clone(), None).await
}).await?;
```

#### Error Handling and Classification

```rust
use siumai::error_handling::{ErrorClassifier, ErrorContext};

match client.chat_with_tools(messages, None).await {
    Ok(response) => println!("Success: {}", response.text().unwrap_or("")),
    Err(error) => {
        let context = ErrorContext::default();
        let classified = ErrorClassifier::classify(&error, context);

        println!("Error category: {:?}", classified.category);
        println!("Severity: {:?}", classified.severity);
        println!("Recovery suggestions: {:?}", classified.recovery_suggestions);
    }
}
```

## 🔧 Configuration

### Common Parameters

All providers support these common parameters:

- `model`: Model name
- `temperature`: Randomness (0.0-2.0)
- `max_tokens`: Maximum output tokens
- `top_p`: Nucleus sampling parameter
- `stop_sequences`: Stop generation sequences
- `seed`: Random seed for reproducibility

### Provider-Specific Parameters

Each provider can have additional parameters:

**OpenAI:**
- `response_format`: Output format control
- `tool_choice`: Tool selection strategy
- `frequency_penalty`: Frequency penalty
- `presence_penalty`: Presence penalty

**Anthropic:**

- `cache_control`: Prompt caching settings
- `thinking_budget`: Thinking process budget
- `system`: System message handling

**Ollama:**

- `keep_alive`: Model memory duration
- `raw`: Bypass templating
- `format`: Output format (json, etc.)
- `numa`: NUMA support
- `num_ctx`: Context window size
- `num_gpu`: GPU layers to use

### Ollama Local AI Examples

#### Basic Chat with Local Model

```rust
use siumai::prelude::*;

// Connect to local Ollama instance
let client = Provider::ollama()
    .base_url("http://localhost:11434")
    .model(models::ollama::LLAMA_3_2)
    .temperature(0.7)
    .build()
    .await?;

let messages = vec![user!("Explain quantum computing in simple terms")];
let response = client.chat_with_tools(messages, None).await?;
println!("Ollama says: {}", response.content);
```

#### Advanced Ollama Configuration

```rust
use siumai::providers::ollama::{OllamaClient, OllamaConfig};

let config = OllamaConfig::builder()
    .base_url("http://localhost:11434")
    .model(models::ollama::LLAMA_3_2)
    .keep_alive("10m")           // Keep model in memory
    .num_ctx(4096)              // Context window
    .num_gpu(1)                 // Use GPU acceleration
    .numa(true)                 // Enable NUMA
    .think(true)                // Enable thinking mode for thinking models
    .option("temperature", serde_json::Value::Number(
        serde_json::Number::from_f64(0.8).unwrap()
    ))
    .build()?;

let client = OllamaClient::new_with_config(config);

// Generate text with streaming
let mut stream = client.generate_stream("Write a haiku about AI".to_string()).await?;
while let Some(event) = stream.next().await {
    // Process streaming response
}
```

#### Thinking Models with Ollama

```rust
use siumai::prelude::*;

// Use thinking models like DeepSeek-R1
let client = LlmBuilder::new()
    .ollama()
    .base_url("http://localhost:11434")
    .model(models::ollama::DEEPSEEK_R1)
    .reasoning(true)            // Enable reasoning mode
    .temperature(0.7)
    .build()
    .await?;

let messages = vec![
    user!("Solve this step by step: What is 15% of 240?")
];

let response = client.chat(messages).await?;

// Access the model's thinking process
if let Some(thinking) = &response.thinking {
    println!("🧠 Model's reasoning: {}", thinking);
}

// Get the final answer
if let Some(answer) = response.content_text() {
    println!("📝 Final answer: {}", answer);
}
```

### OpenAI API Feature Examples

#### Responses API (OpenAI-Specific)

OpenAI's Responses API provides stateful conversations, background processing, and built-in tools:

```rust
use siumai::models;
use siumai::providers::openai::responses::{OpenAiResponses, ResponsesApiCapability};
use siumai::providers::openai::config::OpenAiConfig;
use siumai::types::OpenAiBuiltInTool;
use siumai::prelude::*;

// Create Responses API client with built-in tools
let config = OpenAiConfig::new("your-api-key")
    .with_model(models::openai::GPT_4O)
    .with_responses_api(true)
    .with_built_in_tool(OpenAiBuiltInTool::WebSearch);

let client = OpenAiResponses::new(reqwest::Client::new(), config);

// Basic chat with built-in tools
let messages = vec![user!("What's the latest news about AI?")];
let response = client.chat_with_tools(messages, None).await?;
println!("Response: {}", response.content.all_text());

// Background processing for complex tasks
let complex_messages = vec![user!("Research quantum computing and write a summary")];
let background_response = client
    .create_response_background(
        complex_messages,
        None,
        Some(vec![OpenAiBuiltInTool::WebSearch]),
        None,
    )
    .await?;

// Check if background task is ready
let is_ready = client.is_response_ready(&background_response.id).await?;
if is_ready {
    let final_response = client.get_response(&background_response.id).await?;
    println!("Background result: {}", final_response.content.all_text());
}
```

#### Text Embedding

```rust
use siumai::models;
use siumai::prelude::*;

// Unified interface - works with any provider that supports embeddings
let client = Siumai::builder()
    .openai()
    .api_key("your-api-key")
    .model(models::openai::TEXT_EMBEDDING_3_SMALL)
    .build()
    .await?;

let texts = vec!["Hello, world!".to_string()];
let response = client.embed(texts).await?;
println!("Got {} embeddings with {} dimensions",
         response.embeddings.len(),
         response.embeddings[0].len());

// Provider-specific interface for advanced features
let embeddings_client = Provider::openai()
    .api_key("your-api-key")
    .build()
    .await?;

let response = embeddings_client.embed(texts).await?;
```

#### Text-to-Speech

```rust
use siumai::models;
use siumai::providers::openai::{OpenAiConfig, OpenAiAudio};
use siumai::traits::AudioCapability;
use siumai::types::TtsRequest;

let config = OpenAiConfig::new("your-api-key");
let client = OpenAiAudio::new(config, reqwest::Client::new());

let request = TtsRequest {
    text: "Hello, world!".to_string(),
    voice: Some("alloy".to_string()),
    format: Some("mp3".to_string()),
    speed: Some(1.0),
    model: Some(models::openai::TTS_1.to_string()),
    extra_params: std::collections::HashMap::new(),
};

let response = client.text_to_speech(request).await?;
std::fs::write("output.mp3", response.audio_data)?;
```

#### Image Generation

Generate images using OpenAI DALL-E or SiliconFlow models:

```rust
use siumai::prelude::*;
use siumai::traits::ImageGenerationCapability;
use siumai::types::ImageGenerationRequest;

// OpenAI DALL-E
let client = LlmBuilder::new()
    .openai()
    .api_key("your-openai-api-key")
    .build()
    .await?;

let request = ImageGenerationRequest {
    prompt: "A futuristic city with flying cars at sunset".to_string(),
    size: Some("1024x1024".to_string()),
    count: 1,
    model: Some("dall-e-3".to_string()),
    quality: Some("hd".to_string()),
    style: Some("vivid".to_string()),
    ..Default::default()
};

let response = client.generate_images(request).await?;
for image in response.images {
    if let Some(url) = image.url {
        println!("Generated image: {}", url);
    }
}

// SiliconFlow with advanced parameters
use siumai::providers::openai_compatible::siliconflow;

let siliconflow_client = LlmBuilder::new()
    .siliconflow()
    .api_key("your-siliconflow-api-key")
    .build()
    .await?;

let sf_request = ImageGenerationRequest {
    prompt: "A beautiful landscape with mountains".to_string(),
    negative_prompt: Some("blurry, low quality".to_string()),
    size: Some("1024x1024".to_string()),
    count: 1,
    model: Some(siliconflow::KOLORS.to_string()),
    steps: Some(20),
    guidance_scale: Some(7.5),
    seed: Some(42),
    ..Default::default()
};

let sf_response = siliconflow_client.generate_images(sf_request).await?;
```

## 🧪 Testing

### Unit and Mock Tests

Run the standard test suite (no API keys required):

```bash
cargo test
```

### Integration Tests

Run mock integration tests:

```bash
cargo test --test integration_tests
```

### Real LLM Integration Tests

**⚠️ These tests use real API keys and make actual API calls!**

Siumai includes comprehensive integration tests that verify functionality against real LLM providers. These tests are ignored by default to prevent accidental API usage.

#### Quick Setup

1. **Set API keys** (you only need keys for providers you want to test):
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   export GEMINI_API_KEY="your-key"
   # ... other providers
   ```

2. **Run tests**:
   ```bash
   # Test all available providers
   cargo test test_all_available_providers -- --ignored --nocapture

   # Test specific provider
   cargo test test_openai_integration -- --ignored --nocapture
   ```

#### Using Helper Scripts

For easier setup, use the provided scripts that automatically load `.env` files:

```bash
# Create .env file from template (optional)
cp .env.example .env
# Edit .env with your API keys

# Run the script
# Linux/macOS
./scripts/run_integration_tests.sh

# Windows
scripts\run_integration_tests.bat
```

#### Test Coverage

Each provider test includes:
- ✅ **Non-streaming chat**: Basic request/response
- 🌊 **Streaming chat**: Real-time response streaming
- 🔢 **Embeddings**: Text embedding generation (if supported)
- 🧠 **Reasoning**: Advanced reasoning/thinking capabilities (if supported)

#### Supported Providers

| Provider     | Chat | Streaming | Embeddings | Reasoning | Rerank | Images |
|--------------|------|-----------|------------|-----------|--------|--------|
| OpenAI       | ✅   | ✅        | ✅         | ✅ (o1)   | ❌     | ✅     |
| Anthropic    | ✅   | ✅        | ❌         | ✅ (thinking) | ❌     | ❌     |
| Gemini       | ✅   | ✅        | ✅         | ✅ (thinking) | ❌     | ✅     |
| DeepSeek     | ✅   | ✅        | ❌         | ✅ (reasoner) | ❌     | ❌     |
| OpenRouter   | ✅   | ✅        | ❌         | ✅ (o1 models) | ❌     | ❌     |
| SiliconFlow  | ✅   | ✅        | ✅         | ✅ (reasoner) | ✅     | ✅     |
| Groq         | ✅   | ✅        | ❌         | ❌        | ❌     | ❌     |
| xAI          | ✅   | ✅        | ❌         | ✅ (Grok) | ❌     | ❌     |

See [tests/README.md](tests/README.md) for detailed instructions.

### Examples

Run examples:

```bash
cargo run --example basic_usage
```

## 📖 Documentation

- [API Documentation](https://docs.rs/siumai)
- [Examples](examples/)
- [Integration Tests](tests/)

### 🛠️ Developer Documentation

- [Adding OpenAI-Compatible Providers](docs/adding-openai-compatible-providers.md) - Step-by-step guide for contributors
- [OpenAI-Compatible Architecture](docs/openai-compatible-architecture.md) - Architecture design and principles

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

- Inspired by the need for a unified LLM interface in Rust
- Built with love for the Rust community
- Special thanks to all contributors

---

Made with ❤️ by the YumchaLabs team
