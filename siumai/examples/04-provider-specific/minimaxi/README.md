# MiniMaxi Provider Examples

MiniMaxi is a multi-modal provider with chat, speech, image, music, and video capabilities.

## Package tier

- provider-owned package
- config-first is the preferred provider-specific path
- music and video remain extension-oriented capabilities

## API Endpoints

MiniMaxi currently exposes two commonly used API endpoints:

- China: `https://api.minimaxi.com` (default)
- Global: `https://api.minimax.io`

## Setup

Get your API key from the MiniMaxi console and set:

```bash
export MINIMAXI_API_KEY="your-api-key-here"
```

## Examples

### 1. Basic Chat (`minimaxi_basic.rs`)

Run:

```bash
cargo run -p siumai --example minimaxi_basic --features minimaxi
```

Highlights:

- chat completion
- streaming
- multimodal chat
- image generation
- speech via the Stable `speech::synthesize` family API
- MiniMaxi-specific video/music flows via extension APIs

### 2. Provider Extensions

MiniMaxi exposes vendor-specific knobs through `siumai::provider_ext::minimaxi::*`.

Run:

```bash
cargo run -p siumai --example minimaxi_tts-ext --features minimaxi
cargo run -p siumai --example minimaxi_video-ext --features minimaxi
cargo run -p siumai --example minimaxi_music-ext --features minimaxi
```

## Music Generation

See:

- `music-generation.md`
- `music-ext.rs`

Typical capabilities:

- generate music from text prompts
- optional lyrics support
- configurable sample rate, bitrate, and output format
- instrumental or vocal output

## Video Generation

See:

- `video-generation.md`
- `video-ext.rs`

Typical capabilities:

- text-to-video generation
- multiple resolutions such as `720P` and `1080P`
- configurable duration
- asynchronous task-based workflow

## Image Generation

MiniMaxi image generation remains available through the Stable image family where possible,
with provider-specific extras kept under the provider-owned package.

## Recommended Reading Order

1. `minimaxi_basic.rs`
2. `tts-ext.rs`
3. `video-ext.rs`
4. `music-ext.rs`
5. `music-generation.md` / `video-generation.md`

## Notes

MiniMaxi is a real provider-owned package in Siumai. Keep examples here config-first and extension-oriented rather than treating them as generic compat examples.
