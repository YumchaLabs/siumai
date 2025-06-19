# Siumai - Unified LLM Interface Library for Rust

[![Crates.io](https://img.shields.io/crates/v/siumai.svg)](https://crates.io/crates/siumai)
[![Documentation](https://docs.rs/siumai/badge.svg)](https://docs.rs/siumai)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Siumai (烧卖) is a unified LLM interface library for Rust that provides a consistent API across multiple AI providers. It features capability-based trait separation, type-safe parameter handling, and comprehensive streaming support.

## 🌟 Features

- **🔌 Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini, and custom providers
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
siumai = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

### Basic Usage

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an OpenAI client
    let client = llm()
        .openai()
        .api_key("your-api-key")
        .model("gpt-4")
        .temperature(0.7)
        .build()
        .await?;

    // Create a chat request - simple messages use macros directly
    let request = ChatRequest::builder()
        .message(system!("You are a helpful assistant"))
        .message(user!("What is the capital of France?"))
        .build();

    // Send the request
    let response = client.chat(request).await?;
    println!("Response: {}", response.text().unwrap_or(""));

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
    .message(message)
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
- **`VisionCapability`**: Image analysis and generation
- **`ToolCapability`**: Function calling and tool usage
- **`EmbeddingCapability`**: Text embeddings

### Provider-Specific Traits

- **`OpenAiCapability`**: OpenAI-specific features (structured output, batch processing)
- **`AnthropicCapability`**: Anthropic-specific features (prompt caching, thinking mode)
- **`GeminiCapability`**: Google Gemini-specific features (search integration, code execution)

## 📚 Examples

### Different Providers

```rust
// OpenAI
let openai_client = llm()
    .openai()
    .api_key("sk-...")
    .model("gpt-4")
    .temperature(0.7)
    .build()
    .await?;

// Anthropic
let anthropic_client = llm()
    .anthropic()
    .api_key("sk-ant-...")
    .model("claude-3-5-sonnet-20241022")
    .temperature(0.8)
    .cache_control(CacheControl::Ephemeral)
    .build()
    .await?;
```

### Custom HTTP Client

```rust
use std::time::Duration;

let custom_client = reqwest::Client::builder()
    .timeout(Duration::from_secs(60))
    .user_agent("my-app/1.0")
    .build()?;

let client = llm()
    .with_http_client(custom_client)
    .openai()
    .api_key("your-key")
    .model("gpt-4")
    .build()
    .await?;
```

### Provider-Specific Parameters

```rust
// OpenAI with structured output
let openai_client = llm()
    .openai()
    .api_key("your-key")
    .model("gpt-4")
    .response_format(ResponseFormat::JsonObject)
    .frequency_penalty(0.1)
    .build()
    .await?;

// Anthropic with caching
let anthropic_client = llm()
    .anthropic()
    .api_key("your-key")
    .model("claude-3-5-sonnet-20241022")
    .cache_control(CacheControl::Ephemeral)
    .thinking_budget(1000)
    .build()
    .await?;
```

### Advanced Features

#### Parameter Validation and Optimization

```rust
use siumai::params::EnhancedParameterValidator;

let params = CommonParams {
    model: "gpt-4".to_string(),
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

### OpenAI API Feature Examples

#### Text Embedding

```rust
use siumai::providers::openai::{OpenAiConfig, OpenAiEmbeddings};
use siumai::traits::EmbeddingCapability;

let config = OpenAiConfig::new("your-api-key");
let client = OpenAiEmbeddings::new(config, reqwest::Client::new());

let texts = vec!["Hello, world!".to_string()];
let response = client.embed(texts).await?;
println!("Embedding dimension: {}", response.embeddings[0].len());
```

#### Text-to-Speech

```rust
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
    model: Some("tts-1".to_string()),
    extra_params: std::collections::HashMap::new(),
};

let response = client.text_to_speech(request).await?;
std::fs::write("output.mp3", response.audio_data)?;
```

#### Image Generation

```rust
use siumai::providers::openai::{OpenAiConfig, OpenAiImages};
use siumai::traits::ImageGenerationCapability;
use siumai::types::ImageGenerationRequest;

let config = OpenAiConfig::new("your-api-key");
let client = OpenAiImages::new(config, reqwest::Client::new());

let request = ImageGenerationRequest {
    prompt: "A beautiful sunset".to_string(),
    model: Some("dall-e-3".to_string()),
    size: Some("1024x1024".to_string()),
    count: 1,
    ..Default::default()
};

let response = client.generate_images(request).await?;
for image in response.images {
    if let Some(url) = image.url {
        println!("Image URL: {}", url);
    }
}
```

## 🧪 Testing

Run the test suite:

```bash
cargo test
```

Run integration tests:

```bash
cargo test --test integration_tests
```

Run examples:

```bash
cargo run --example basic_usage
```

## 📖 Documentation

- [API Documentation](https://docs.rs/siumai)
- [Examples](examples/)
- [Integration Tests](tests/)

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

Made with ❤️ by the Siumai team
