# MiniMaxi Provider

MiniMaxi is a multi-modal AI platform providing text generation, speech synthesis, video generation, image generation, and music generation capabilities.

## Features

- **Text Generation**: M2 model with OpenAI-compatible API
- **Speech Synthesis**: Speech 2.6 HD/Turbo models
- **Video Generation**: Hailuo 2.3 & 2.3 Fast models
- **Image Generation**: Image generation capabilities
- **Music Generation**: Music 2.0 model

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

### Text Generation

- `MiniMax-M2` - Main text generation model (default)

### Speech Synthesis

- `speech-2.6-hd` - High-definition speech synthesis
- `speech-2.6-turbo` - Fast speech synthesis

### Video Generation

- `hailuo-2.3` - Video generation model
- `hailuo-2.3-fast` - Fast video generation

### Music Generation

- `music-2.0` - Music generation model

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

The MiniMaxi provider leverages the OpenAI standard layer for request/response transformation, as MiniMaxi's text generation API is OpenAI-compatible. This provides:

- Consistent API interface with other providers
- Automatic request/response transformation
- Built-in streaming support
- Tool calling (function calling) support

## Examples

See the [examples directory](../../../../examples/) for more examples:

- `minimaxi_basic.rs` - Basic usage examples

## API Documentation

For more information about MiniMaxi's API, visit:

- [MiniMaxi Platform Documentation](https://platform.minimaxi.com/docs/guides/platform-intro)
- [Text API Reference](https://platform.minimaxi.com/docs/api-reference/text-intro)
- [OpenAI-Compatible API](https://platform.minimaxi.com/docs/api-reference/text-openai-api)
- [Function Calling Reference](https://platform.minimaxi.com/docs/api-reference/text-m2-function-call-refer)

## License

This provider implementation is part of the siumai library and follows the same license.

