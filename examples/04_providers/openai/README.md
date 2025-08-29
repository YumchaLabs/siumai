# OpenAI Provider Examples

Examples demonstrating OpenAI-specific features and capabilities with Siumai.

## Examples

### [basic_chat.rs](basic_chat.rs)
**💬 Basic OpenAI chat functionality**

Essential OpenAI chat features and model selection.

```bash
cargo run --example openai_basic_chat
```

**What you'll learn:**
- OpenAI model selection (GPT-4, GPT-4o, GPT-3.5)
- Temperature and parameter tuning
- Token usage optimization
- Response format options

### [openai_responses_api.rs](openai_responses_api.rs)
**🧩 Responses API (stateful conversations, background jobs, built-in tools)**

Demonstrates OpenAI Responses API usage including background processing and conversation chaining.

```bash
cargo run --example openai_responses_api
```

Notes:
- Requires OPENAI_API_KEY
- The “List Responses” section is commented out by default because OpenAI requires a browser session key (session-based auth) for GET /v1/responses.
- If you have a valid session key flow, set OPENAI_SESSION_KEY and the example will attempt to list responses using it; otherwise it prints a message and continues.


### [enhanced_features.rs](enhanced_features.rs)
**🚀 Advanced OpenAI capabilities**

Advanced features like JSON mode, function calling, and structured outputs.

```bash
cargo run --example openai_enhanced_features
```

**What you'll learn:**
- JSON mode for structured responses
- Function calling and tool usage
- System fingerprints
- Response format control

### [vision_processing.rs](vision_processing.rs)
**👁️ GPT-4 Vision capabilities**

Image analysis and multimodal processing with GPT-4 Vision.

```bash
cargo run --example openai_vision_processing
```

**What you'll learn:**
- Image upload and analysis
- Vision prompt optimization
- Detail level control
- Cost optimization for vision

### [audio_processing.rs](audio_processing.rs)
**🎵 Whisper and TTS integration**

Audio transcription with Whisper and text-to-speech generation.

```bash
# Note: This example is not yet implemented
# cargo run --example openai_audio_processing
```

**What you'll learn:**
- Audio transcription with Whisper
- Text-to-speech with TTS models
- Audio format handling
- Streaming audio processing

### [image_generation_showcase.rs](image_generation_showcase.rs)
**🎨 DALL-E image generation showcase**

Comprehensive demonstration of OpenAI's DALL-E image generation capabilities.

```bash
cargo run --example image_generation_showcase
```

**What you'll learn:**
- Image generation with DALL-E 2 and DALL-E 3
- Different image sizes and quality settings
- Prompt optimization for better results
- Image format handling and saving
- Cost optimization strategies
- Error handling and retry logic

### [files_api.rs](files_api.rs)
**📁 Files API and Assistants**

File management and processing for Assistants API.

```bash
# Note: This example is not yet implemented
# cargo run --example openai_files_api
```

**What you'll learn:**
- File upload and management
- Document processing
- Assistants API integration
- File-based conversations

## Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Model Selection Guide

### GPT Models

| Model | Use Case | Cost | Speed |
|-------|----------|------|-------|
| gpt-4o | Best overall performance | High | Medium |
| gpt-4o-mini | Balanced cost/performance | Low | Fast |
| gpt-4-turbo | Complex reasoning | High | Medium |
| gpt-3.5-turbo | Simple tasks | Very Low | Very Fast |

### Specialized Models

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| whisper-1 | Audio transcription | Audio | Text |
| tts-1 | Text-to-speech | Text | Audio |
| dall-e-3 | Image generation | Text | Image |
| gpt-4-vision | Image analysis | Text + Image | Text |

## Key Features

### JSON Mode
```rust
let client = LlmBuilder::new()
    .openai()
    .api_key(&api_key)
    .model("gpt-4o-mini")
    .response_format("json_object")
    .build()
    .await?;
```

### Function Calling
```rust
let tools = vec![
    Tool::function("get_weather", "Get weather information")
        .parameter("location", "string", "City name")
        .required(&["location"])
];

let response = client.chat_with_tools(messages, tools).await?;
```

### Vision Processing
```rust
let message = ChatMessage::user("What's in this image?")
    .with_image("path/to/image.jpg".to_string(), Some("high".to_string()))
    .build();
```

### Audio Processing
```rust
// Transcription
let transcription = client.transcribe_audio("audio.mp3").await?;

// Text-to-speech
let audio = client.text_to_speech("Hello world", "alloy").await?;
```

## Cost Optimization

### Model Selection
- Use `gpt-4o-mini` for most tasks
- Reserve `gpt-4o` for complex reasoning
- Use `gpt-3.5-turbo` for simple tasks

### Token Management
```rust
let client = LlmBuilder::new()
    .openai()
    .model("gpt-4o-mini")
    .max_tokens(500)  // Limit response length
    .temperature(0.1) // Lower for consistency
    .build()
    .await?;
```

### Vision Optimization
```rust
// Use "low" detail for simple images
let message = ChatMessage::user("Describe this image")
    .with_image("image.jpg", Some("low"))  // Costs less
    .build();
```

## Best Practices

### Authentication
```rust
// Use environment variables
let api_key = std::env::var("OPENAI_API_KEY")?;

// Or use a secure configuration system
let client = LlmBuilder::new()
    .openai()
    .api_key(&api_key)
    .build()
    .await?;
```

### Error Handling
```rust
match client.chat(messages).await {
    Ok(response) => {
        // Handle success
    }
    Err(LlmError::RateLimitError(_)) => {
        // Implement backoff
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
    Err(LlmError::AuthenticationError(_)) => {
        // Check API key
    }
    Err(e) => {
        // Handle other errors
    }
}
```

### Rate Limiting
```rust
use tokio::time::{sleep, Duration};

// Simple rate limiting
for request in requests {
    let response = client.chat(request).await?;
    sleep(Duration::from_millis(100)).await; // 10 RPS
}
```

## Common Patterns

### Structured Output
```rust
let messages = vec![
    system!("Respond in valid JSON format with 'answer' and 'confidence' fields"),
    user!("What is the capital of France?")
];

let client = LlmBuilder::new()
    .openai()
    .response_format("json_object")
    .build()
    .await?;
```

### Conversation Management
```rust
let mut conversation = vec![
    system!("You are a helpful assistant")
];

// Add user message
conversation.push(user!("Hello!"));

// Get response and add to conversation
let response = client.chat(conversation.clone()).await?;
if let Some(text) = response.content_text() {
    conversation.push(assistant!(text));
}
```

### Streaming with Progress
```rust
use futures_util::StreamExt;

let mut stream = client.chat_stream(messages, None).await?;
let mut response_text = String::new();

while let Some(event) = stream.next().await {
    match event? {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            response_text.push_str(&delta);
            print!("{}", delta); // Real-time display
        }
        ChatStreamEvent::StreamEnd { .. } => break,
        _ => {}
    }
}
```

## Troubleshooting

### Authentication Issues
- Verify API key is correct and active
- Check if you have sufficient credits
- Ensure API key has required permissions

### Rate Limit Errors
- Implement exponential backoff
- Reduce request frequency
- Consider upgrading your plan

### Model Access Issues
- Some models require special access
- Check model availability in your region
- Verify model name spelling

### Cost Management
- Monitor token usage regularly
- Set up billing alerts
- Use cheaper models for development

## Next Steps

- **[Core Features](../../02_core_features/)**: Learn provider-agnostic patterns
- **[Advanced Features](../../03_advanced_features/)**: Explore sophisticated capabilities
- **[Use Cases](../../05_use_cases/)**: See complete applications

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Model Pricing](https://openai.com/pricing)
- [Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
