# MiniMaxi Provider

MiniMaxi is a multi-modal AI platform providing text generation, speech synthesis, image generation, and more.

## Features

- **Text Generation**: M2 model with Anthropic-compatible API (supports thinking blocks, tool use)
- **Speech Synthesis**: Speech 2.6 HD/Turbo models
- **Image Generation**: Image-01 and Image-01-Live models
- **Video Generation**: Coming soon (Hailuo 2.3 models)
- **Music Generation**: Coming soon (Music 2.0 model)

## API Standards

MiniMaxi uses different API standards for different capabilities:

### Chat API (Anthropic Standard)

- **Endpoint**: `https://api.minimaxi.com/anthropic/v1/messages`
- **Authentication**: `x-api-key` header
- **Supported Models**: MiniMax-M2, MiniMax-M2-Stable
- **Features**: Thinking blocks, tool use, streaming

### Audio/Image APIs (OpenAI Standard)

- **Endpoint**: `https://api.minimaxi.com/v1/*`
- **Authentication**: `Authorization: Bearer <token>` header
- **Image Models**: image-01, image-01-live
- **Audio Models**: speech-2.6-hd, speech-2.6-turbo

## Setup

### API Key

Set your MiniMaxi API key as an environment variable:

```bash
export MINIMAXI_API_KEY=your-api-key-here
```

Or provide it directly when building the client:

```rust
use siumai::prelude::*;

let client = LlmBuilder::new()
    .minimaxi()
    .api_key("your-api-key-here")
    .build()
    .await?;
```

### Cargo Features

Enable the `minimaxi` feature in your `Cargo.toml`:

```toml
[dependencies]
siumai = { version = "0.11", features = ["minimaxi"] }
```

## Usage

### Basic Chat

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .minimaxi()
        .model("MiniMax-M2")
        .build()
        .await?;

    let messages = vec![
        user!("你好！请介绍一下你自己。"),
    ];

    let response = client.chat(messages).await?;
    println!("Response: {}", response.content);

    Ok(())
}
```

### Streaming Chat

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .minimaxi()
        .model("MiniMax-M2")
        .build()
        .await?;

    let messages = vec![
        user!("写一首关于人工智能的诗。"),
    ];

    let mut stream = client.chat_stream(messages, None).await?;
    
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => {
                use siumai::streaming::ChatStreamEvent;
                match event {
                    ChatStreamEvent::ContentDelta { delta, .. } => {
                        print!("{}", delta);
                    }
                    ChatStreamEvent::Done { .. } => {
                        println!("\n");
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    Ok(())
}
```

### Function Calling (Tools)

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .minimaxi()
        .model("MiniMax-M2")
        .build()
        .await?;

    let messages = vec![
        user!("What's the weather like in Beijing?"),
    ];

    let tools = vec![
        Tool::new(
            "get_weather",
            "Get the current weather for a location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }),
        ),
    ];

    let response = client.chat_with_tools(messages, Some(tools)).await?;
    
    if let Some(tool_calls) = response.tool_calls {
        for call in tool_calls {
            println!("Function: {}", call.function.name);
            println!("Arguments: {}", call.function.arguments);
        }
    }

    Ok(())
}
```

## Available Models

You can use model constants from `siumai::providers::minimaxi::model_constants`:

```rust
use siumai::providers::minimaxi::model_constants;

// Text models
model_constants::text::MINIMAX_M2           // "MiniMax-M2"
model_constants::text::MINIMAX_M2_STABLE    // "MiniMax-M2-Stable"

// Audio models
model_constants::audio::SPEECH_2_6_HD       // "speech-2.6-hd"
model_constants::audio::SPEECH_2_6_TURBO    // "speech-2.6-turbo"

// Voice IDs
model_constants::voice::MALE_QN_QINGSE      // "male-qn-qingse"
model_constants::voice::FEMALE_SHAONV       // "female-shaonv"

// Image models
model_constants::images::IMAGE_01           // "image-01"
model_constants::images::IMAGE_01_LIVE      // "image-01-live"
```

### Text Generation

- `MiniMax-M2` - Latest model with thinking capabilities
- `MiniMax-M2-Stable` - Stable version for production use

### Speech Synthesis

- `speech-2.6-hd` - High-definition speech synthesis
- `speech-2.6-turbo` - Fast speech synthesis

### Image Generation

- `image-01` - Standard image generation
- `image-01-live` - Real-time image generation with style control

## Configuration

### Custom Base URL

```rust
let client = LlmBuilder::new()
    .minimaxi()
    .base_url("https://custom-api.minimaxi.com/v1")
    .build()
    .await?;
```

### Timeout Configuration

```rust
use std::time::Duration;

let client = LlmBuilder::new()
    .minimaxi()
    .timeout(Duration::from_secs(60))
    .connect_timeout(Duration::from_secs(10))
    .build()
    .await?;
```

### Retry Configuration

```rust
use siumai::retry_api::RetryOptions;

let client = LlmBuilder::new()
    .minimaxi()
    .with_retry(RetryOptions::backoff())
    .build()
    .await?;
```

### Tracing/Observability

```rust
let client = LlmBuilder::new()
    .minimaxi()
    .debug_tracing()  // Enable debug tracing
    .pretty_json(true)  // Pretty-print JSON in logs
    .build()
    .await?;
```

## Architecture

The MiniMaxi provider leverages the Anthropic standard layer for request/response transformation, as MiniMaxi's text generation API is Anthropic-compatible. This provides:

- Consistent API interface with other providers
- Automatic request/response transformation
- Built-in streaming support with thinking blocks
- Tool calling (function calling) support
- Extended thinking capabilities

## Examples

See the [examples directory](../../../../examples/) for more examples:

- `minimaxi_basic.rs` - Basic usage examples

## API Documentation

For more information about MiniMaxi's API, visit:

- [MiniMaxi Platform Documentation](https://platform.minimaxi.com/docs/guides/platform-intro)
- [Text API Reference](https://platform.minimaxi.com/docs/api-reference/text-intro)
- [Anthropic-Compatible API](https://platform.minimaxi.com/docs/api-reference/text-anthropic-api)
- [OpenAI-Compatible API](https://platform.minimaxi.com/docs/api-reference/text-openai-api)
- [Function Calling Reference](https://platform.minimaxi.com/docs/api-reference/text-m2-function-call-refer)

## License

This provider implementation is part of the siumai library and follows the same license.
