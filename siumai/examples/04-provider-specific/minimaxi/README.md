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

### 1. Basic Chat (`minimaxi_basic.rs`)

Basic chat completion with MiniMaxi's MiniMax-M2 model.

**Run:**
```bash
cargo run -p siumai --example minimaxi_basic --features minimaxi
```

**Features:**
- Simple chat completion
- Streaming responses
- Multi-modal chat (text + images)
- Image generation
- Speech (TTS) via unified `SpeechCapability`
- MiniMaxi video/music via extension APIs (see below)

### 2. Provider Extensions (`*_ext.rs`)

MiniMaxi exposes vendor-specific knobs via `siumai::provider_ext::minimaxi::*` helpers.

**Run:**
```bash
cargo run -p siumai --example minimaxi_tts-ext --features minimaxi
cargo run -p siumai --example minimaxi_video-ext --features minimaxi
cargo run -p siumai --example minimaxi_music-ext --features minimaxi
```

## 🎵 Music Generation

MiniMaxi supports AI music generation with the `music-2.0` model.

**See:** [`music-generation.md`](./music-generation.md) and [`music-ext.rs`](./music-ext.rs)

**Features:**
- Generate music from text prompts
- Optional lyrics support
- Customizable audio settings (sample rate, bitrate, format)
- Instrumental or vocal music

**Example:**
```rust
use siumai::prelude::extensions::*;
use siumai::prelude::unified::*;
use siumai::provider_ext::minimaxi::music::MinimaxiMusicRequestBuilder;

let client = Siumai::builder()
    .minimaxi()
    .api_key(&std::env::var("MINIMAXI_API_KEY")?)
    .build()
    .await?;

let request = MinimaxiMusicRequestBuilder::new("Indie folk, melancholic, introspective")
    .lyrics_template()
    .format("mp3")
    .build();

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
use siumai::prelude::extensions::*;
use siumai::prelude::unified::*;
use siumai::provider_ext::minimaxi::video::MinimaxiVideoRequestBuilder;

let client = Siumai::builder()
    .minimaxi()
    .api_key(&std::env::var("MINIMAXI_API_KEY")?)
    .build()
    .await?;

let request = MinimaxiVideoRequestBuilder::new(
    "hailuo-2.3",
    "A beautiful sunset over the ocean with waves crashing"
)
.duration(6)
.resolution("1080P")
.build();

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
use siumai::prelude::unified::*;
use siumai::types::ImageGenerationRequest;

let client = Siumai::builder()
    .minimaxi()
    .api_key(&std::env::var("MINIMAXI_API_KEY")?)
    .build()
    .await?;

let request = ImageGenerationRequest {
    prompt: "A serene mountain landscape at dawn".to_string(),
    model: Some("image-01".to_string()),
    size: Some("1024x1024".to_string()),
    count: 1,
    ..Default::default()
};

let response = client.generate_images(request).await?;
for (i, image) in response.images.iter().enumerate() {
    if let Some(url) = &image.url {
        println!("image[{i}] url = {url}");
    }
}
```

## 🔊 Text-to-Speech

Convert text to speech using MiniMaxi's TTS models.

**Example:**
```rust
use siumai::prelude::unified::*;
use siumai::provider_ext::minimaxi::tts::MinimaxiTtsRequestBuilder;

let client = Siumai::builder()
    .minimaxi()
    .api_key(&std::env::var("MINIMAXI_API_KEY")?)
    .build()
    .await?;

let request = MinimaxiTtsRequestBuilder::new("Hello, this is a test of MiniMaxi TTS")
    .model("speech-2.6-hd")
    .voice_id("female-tianmei")
    .format("mp3")
    .build();

let response = client.tts(request).await?;
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
use siumai::prelude::*;
use siumai::retry_api::RetryOptions;

let client = Siumai::builder()
    .minimaxi()
    .api_key("your-api-key")
    .base_url("https://api.minimax.io") // Use global endpoint
    .model("MiniMax-M2")
    .with_retry(RetryOptions::backoff().with_max_attempts(3))
    .http_debug(true)
    .build()
    .await?;
```

## 🚀 Next Steps

1. Try the basic example: `cargo run -p siumai --example minimaxi_basic --features minimaxi`
2. Explore music generation: Read [`music-generation.md`](./music-generation.md)
3. Explore video generation: Read [`video-generation.md`](./video-generation.md)
4. Check out the [main examples README](../../README.md) for more examples
