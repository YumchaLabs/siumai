# MiniMaxi Provider Examples

MiniMaxi is a Chinese AI provider offering multi-modal capabilities including chat, text-to-speech, image generation, video generation, and music generation.

## 🌐 API Endpoints

MiniMaxi provides two API endpoints:

- **China**: `https://api.minimaxi.com` (default)
- **Global**: `https://api.minimax.io`

## 🔑 Setup

Get your API key from [MiniMaxi Console](https://platform.minimaxi.com/):

```bash
export MINIMAXI_API_KEY="your-api-key-here"
```

## 📚 Examples

### 1. Basic Chat (`basic.rs`)

Basic chat completion with MiniMaxi's MiniMax-M2 model.

**Run:**
```bash
cargo run --example basic --features minimaxi
```

**Features:**
- Simple chat completion
- Streaming responses
- Multi-modal chat (text + images)
- Audio generation (TTS)
- Image generation
- Video generation
- Music generation

## 🎵 Music Generation

MiniMaxi supports AI music generation with the `music-2.0` model.

**See:** [`music-generation.md`](./music-generation.md)

**Features:**
- Generate music from text prompts
- Optional lyrics support
- Customizable audio settings (sample rate, bitrate, format)
- Instrumental or vocal music

**Example:**
```rust
use siumai::prelude::*;
use siumai::types::music::MusicGenerationRequest;

let client = MinimaxiClient::from_env()?;

let request = MusicGenerationRequest::new(
    "music-2.0",
    "Indie folk, melancholic, introspective"
)
.with_lyrics("[verse]\nWalking down the empty street\n[chorus]\nMemories fade away");

let response = client.generate_music(request).await?;
std::fs::write("output.mp3", &response.audio_data)?;
```

## 🎬 Video Generation

MiniMaxi supports AI video generation with the `hailuo-2.3` model.

**See:** [`video-generation.md`](./video-generation.md)

**Features:**
- Text-to-video generation
- Multiple resolutions (720P, 1080P)
- Customizable duration (6s, 10s)
- Asynchronous task-based generation

**Example:**
```rust
use siumai::prelude::*;
use siumai::types::video::VideoGenerationRequest;

let client = MinimaxiClient::from_env()?;

let request = VideoGenerationRequest::new(
    "hailuo-2.3",
    "A beautiful sunset over the ocean with waves crashing"
)
.with_duration(6)
.with_resolution("1080P");

// Create video generation task
let response = client.create_video_task(request).await?;
println!("Task ID: {}", response.task_id);

// Poll for completion
loop {
    let status = client.query_video_task(&response.task_id).await?;
    match status.status {
        VideoTaskStatus::Success => {
            println!("Video ready! File ID: {:?}", status.file_id);
            break;
        }
        VideoTaskStatus::Processing => {
            println!("Still processing...");
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        VideoTaskStatus::Fail => {
            println!("Task failed");
            break;
        }
        _ => {}
    }
}
```

## 🎨 Image Generation

Generate images using MiniMaxi's image generation models.

**Example:**
```rust
use siumai::prelude::*;
use siumai::types::image::ImageGenerationRequest;

let client = MinimaxiClient::from_env()?;

let request = ImageGenerationRequest::new(
    "A serene mountain landscape at dawn"
)
.with_model("image-model-v1")
.with_size("1024x1024");

let response = client.generate_image(request).await?;
for (i, image) in response.images.iter().enumerate() {
    std::fs::write(format!("image_{}.png", i), &image.data)?;
}
```

## 🔊 Text-to-Speech

Convert text to speech using MiniMaxi's TTS models.

**Example:**
```rust
use siumai::prelude::*;
use siumai::types::audio::TtsRequest;

let client = MinimaxiClient::from_env()?;

let request = TtsRequest::new("Hello, this is a test of MiniMaxi TTS".to_string())
    .with_model("speech-2.6-hd")
    .with_voice("female-tianmei");

let response = client.text_to_speech(request).await?;
std::fs::write("speech.mp3", &response.audio_data)?;
```

## 🌟 Supported Models

### Chat Models
- `MiniMax-M2` - Latest chat model

### Audio Models (TTS)
- `speech-2.6-hd` - High-quality TTS
- `speech-2.6-turbo` - Fast TTS

### Video Models
- `hailuo-2.3` - Standard video generation
- `hailuo-2.3-fast` - Fast video generation

### Music Models
- `music-2.0` - Music generation with lyrics support

### Image Models
- Various image generation models

## 📖 Additional Resources

- [MiniMaxi Official Documentation](https://platform.minimaxi.com/document)
- [API Reference](https://platform.minimaxi.com/document/api)
- [Pricing](https://platform.minimaxi.com/document/price)

## 💡 Tips

1. **API Endpoint**: Use the global endpoint (`https://api.minimax.io`) if you're outside China for better latency
2. **Video Generation**: Video generation is asynchronous and may take 30-60 seconds
3. **Music Generation**: Music generation returns hex-encoded audio data that's automatically decoded
4. **Rate Limits**: Be aware of rate limits on the free tier
5. **Audio Format**: TTS and Music APIs return audio in the format specified in the request (default: MP3)

## 🔧 Configuration

You can customize the MiniMaxi client:

```rust
use siumai::providers::minimaxi::{MinimaxiConfig, MinimaxiClient};

let config = MinimaxiConfig::new("your-api-key")
    .with_base_url("https://api.minimax.io")  // Use global endpoint
    .with_model("MiniMax-M2");

let client = MinimaxiClient::new(config, reqwest::Client::new())
    .with_retry(RetryOptions::backoff().with_max_attempts(3))
    .with_interceptors(vec![Arc::new(LoggingInterceptor::new())]);
```

## 🚀 Next Steps

1. Try the basic example: `cargo run --example basic --features minimaxi`
2. Explore music generation: Read [`music-generation.md`](./music-generation.md)
3. Explore video generation: Read [`video-generation.md`](./video-generation.md)
4. Check out the [main examples README](../../README.md) for more examples

