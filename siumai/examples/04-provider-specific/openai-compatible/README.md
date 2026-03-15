# OpenAI-Compatible Providers Examples

This directory contains examples for vendors that currently live on the shared OpenAI-compatible runtime.

## Package tier

- compat vendor view or compat preset
- preferred construction here is config-first or built-in compat client construction
- builder examples in this directory are convenience / compatibility demos, not the preferred architectural default

## Recommended Reading Order

1. typed compat vendor views first:
   - `openrouter-embedding.rs`
   - `openrouter-transforms.rs`
   - `perplexity-search.rs`
2. compat preset stories next:
   - `moonshot-basic.rs`
   - `moonshot-long-context.rs`
   - `moonshot-tools.rs`
   - `siliconflow-rerank.rs`
   - `jina-rerank.rs`
   - `voyageai-rerank.rs`
   - `siliconflow-image.rs`
   - `siliconflow-speech.rs`
   - `siliconflow-transcription.rs`
   - `together-speech.rs`
   - `together-transcription.rs`
   - `fireworks-transcription.rs`
3. builder convenience examples only when you specifically want a comparison with compatibility-style setup

The file `moonshot-siumai-builder.rs` is intentionally kept as a compatibility demo, not the default path for new code.

## Surface Note

- examples in this directory are primarily compat vendor views or preset stories built on `OpenAiCompatibleClient`
- `openrouter` and `perplexity` are the preferred examples for typed vendor extensions on the compat runtime
- compat audio is split by vendor capability rather than forced symmetry:
  - `together` and `siliconflow` expose both TTS and transcription on the shared compat runtime
  - `fireworks` currently stays transcription-only on its dedicated audio host
- compat image generation is also provider-specific:
  - `siliconflow` and `together` expose image generation on the shared compat runtime
  - `fireworks` does not currently participate in the compat image story
- preset vendors such as `siliconflow`, `together`, and `fireworks` should stay in this compatibility story until they earn promotion through stronger provider-owned public semantics
- this directory does **not** imply that every OpenAI-compatible vendor should become a dedicated top-level provider package

## Included Example Themes

### Typed compat vendor views

- `openrouter-embedding.rs` - Stable embedding request options on `OpenAiCompatibleClient`
- `openrouter-transforms.rs` - typed `OpenRouterOptions` plus typed `OpenRouterMetadata` on `OpenAiCompatibleClient`
- `perplexity-search.rs` - typed `PerplexityOptions` plus typed response metadata on `OpenAiCompatibleClient`

These are the preferred examples when you want to understand what a
**compat vendor view** looks like in Siumai: typed vendor helpers layered on the
shared compat runtime without inventing a provider-owned client package.

### Compat presets

#### Moonshot (Kimi)

Moonshot is documented here as an OpenAI-compatible preset story.

Examples:

- `moonshot-basic.rs` - config-first basic chat
- `moonshot-tools.rs` - function/tool calling
- `moonshot-long-context.rs` - long-context usage
- `moonshot-siumai-builder.rs` - compatibility/builder comparison demo

### Compat preset image generation

- `siliconflow-image.rs` - config-first image generation on `OpenAiCompatibleClient`
- `together-image.rs` - config-first Together image generation on `OpenAiCompatibleClient`

### Compat preset rerank

- `siliconflow-rerank.rs` - config-first SiliconFlow rerank on `OpenAiCompatibleClient`
- `jina-rerank.rs` - config-first Jina rerank on `OpenAiCompatibleClient`
- `voyageai-rerank.rs` - config-first VoyageAI rerank on `OpenAiCompatibleClient`

### Compat preset audio split

- `siliconflow-speech.rs` - config-first SiliconFlow TTS on `OpenAiCompatibleClient`
- `siliconflow-transcription.rs` - config-first SiliconFlow STT on `OpenAiCompatibleClient`
- `together-speech.rs` - config-first Together TTS on `OpenAiCompatibleClient`
- `together-transcription.rs` - config-first Together STT on `OpenAiCompatibleClient`
- `fireworks-transcription.rs` - config-first Fireworks STT on `OpenAiCompatibleClient`

## Quick Start

```bash
export OPENROUTER_API_KEY="your-api-key-here"
cargo run --example openrouter-transforms --features openai

export OPENROUTER_API_KEY="your-api-key-here"
cargo run --example openrouter-embedding --features openai

export PERPLEXITY_API_KEY="your-api-key-here"
cargo run --example perplexity-search --features openai

export SILICONFLOW_API_KEY="your-api-key-here"
cargo run --example siliconflow-rerank --features openai

export SILICONFLOW_API_KEY="your-api-key-here"
cargo run --example siliconflow-image --features openai

export SILICONFLOW_API_KEY="your-api-key-here"
cargo run --example siliconflow-speech --features openai

export SILICONFLOW_API_KEY="your-api-key-here"
export SILICONFLOW_AUDIO_FILE="/path/to/audio.mp3"
cargo run --example siliconflow-transcription --features openai

export TOGETHER_API_KEY="your-api-key-here"
cargo run --example together-image --features openai

export JINA_API_KEY="your-api-key-here"
cargo run --example jina-rerank --features openai

export VOYAGEAI_API_KEY="your-api-key-here"
cargo run --example voyageai-rerank --features openai

export TOGETHER_API_KEY="your-api-key-here"
cargo run --example together-speech --features openai

export TOGETHER_API_KEY="your-api-key-here"
export TOGETHER_AUDIO_FILE="/path/to/audio.mp3"
cargo run --example together-transcription --features openai

export FIREWORKS_API_KEY="your-api-key-here"
export FIREWORKS_AUDIO_FILE="/path/to/audio.mp3"
cargo run --example fireworks-transcription --features openai

export MOONSHOT_API_KEY="your-api-key-here"
cargo run --example moonshot-basic --features openai
```

## Moonshot Notes

Moonshot is strong in long-context and Chinese/English bilingual workflows.

Typical examples in this directory cover:

- long context
- tool calling
- bilingual prompts
- config-first compat construction

## Model Selection Notes

Typical Moonshot model examples include:

| Model | Context window | Best for |
|---|---:|---|
| `kimi-k2-0905-preview` | 256K | latest features, agentic coding, long context |
| `moonshot-v1-128k` | 128K | long documents and research |
| `moonshot-v1-32k` | 32K | long articles and conversations |
| `moonshot-v1-8k` | 8K | short chats and quick tasks |

## Notes

Use this directory when you want compat runtime examples with vendor-specific typed helpers or preset wiring.
If you need a real provider-owned package story, prefer directories under `04-provider-specific/<provider>` that map to provider-owned `Config` / `Client` surfaces.
