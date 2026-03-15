# Provider-Specific Examples

This directory contains provider-focused examples for Siumai V4.

Read these examples with the following package tiers in mind:

- provider-owned package: full provider boundary with provider-owned `Config` / `Client`
- focused provider package: only the provider-specific capabilities we intentionally expose
- compat vendor view / preset: examples layered on the shared OpenAI-compatible runtime

## Recommended reading order

1. prefer registry-first examples elsewhere in `examples/` for application architecture
2. use this directory when you need provider-specific setup or typed extension APIs
3. treat builder-based examples here as convenience demos unless the file says otherwise

## Directory map

### Provider-owned packages

- `openai/`
- `anthropic/`
- `google/`
- `bedrock/`
- `deepseek/`
- `groq/`
- `xai/`
- `ollama/`
- `minimaxi/`

These directories should prefer config-first examples and provider-owned extension APIs.

### Focused provider packages

- `cohere/`
- `togetherai/`

These directories should stay narrow and only document the capabilities they truly own.

### Compat vendor view / preset examples

- `openai-compatible/`

This directory is for typed vendor views and preset stories on top of `OpenAiCompatibleClient`.
It is not a signal that every OpenAI-compatible vendor should become a separate provider package.

Current notable examples in that story include:

- `openrouter-embedding.rs`
- `openrouter-transforms.rs`
- `perplexity-search.rs`
- `siliconflow-rerank.rs`
- `jina-rerank.rs`
- `voyageai-rerank.rs`
- `siliconflow-image.rs`
- `siliconflow-speech.rs`
- `siliconflow-transcription.rs`
- `together-image.rs`
- `together-speech.rs`
- `together-transcription.rs`
- `fireworks-transcription.rs`
- `moonshot-basic.rs`
- `moonshot-tools.rs`
- `moonshot-siumai-builder.rs` (compatibility/convenience demo)

## Secondary provider example guide

When you want provider-specific examples beyond the major package surfaces, prefer them in this order:

1. provider-owned config-first examples
2. compat vendor-view examples on `OpenAiCompatibleClient`
3. builder demos only when comparing migration/convenience flows

Recommended starting points:

- provider-owned wrappers:
  - `deepseek/reasoning.rs`
  - `groq/structured-output.rs`
  - `xai/reasoning.rs`
  - `xai/structured-output.rs`
  - `xai/tts.rs`
  - `xai/web-search.rs`
  - `ollama/structured-output.rs`
  - `ollama/metadata.rs`
- compat vendor views:
  - `openai-compatible/openrouter-embedding.rs`
  - `openai-compatible/openrouter-transforms.rs`
  - `openai-compatible/perplexity-search.rs`
  - `openai-compatible/siliconflow-rerank.rs`
  - `openai-compatible/jina-rerank.rs`
  - `openai-compatible/voyageai-rerank.rs`
  - `openai-compatible/siliconflow-image.rs`
  - `openai-compatible/siliconflow-speech.rs`
  - `openai-compatible/siliconflow-transcription.rs`
  - `openai-compatible/together-image.rs`
  - `openai-compatible/together-speech.rs`
  - `openai-compatible/together-transcription.rs`
  - `openai-compatible/fireworks-transcription.rs`
- compatibility demo only:
  - `openai-compatible/moonshot-siumai-builder.rs`

## Policy

- prefer config-first inside real provider package boundaries
- keep focused packages focused
- keep compat preset examples under the compat narrative until promotion criteria are met
- do not add new provider directories just to mirror AI SDK package names
