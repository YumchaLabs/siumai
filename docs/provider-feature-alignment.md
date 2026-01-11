# Provider Feature Alignment (Vercel AI SDK)

This document is a **high-level feature matrix** for Siumai providers during the Alpha.5 fearless refactor.
It is intentionally coarse-grained: the goal is to align on **modules/capabilities first**, then iterate on
field-level and fixture-level parity.

## Scope

We align on:

1. **Stable model families** (Vercel AI SDK concept):
   - `LanguageModel` (chat + streaming)
   - `EmbeddingModel`
   - `ImageModel` (image generation)
   - `RerankModel`
   - `SpeechModel` (TTS)
   - `TranscriptionModel` (STT)
2. **Cross-cutting capabilities**:
   - Tools (function tools + provider-hosted tools)
   - Vision
   - File management
   - Thinking/reasoning (provider-specific)

## Source of truth

For native providers (non OpenAI-compatible adapters), the declared capability set lives in:

- `siumai-registry/src/native_provider_metadata.rs`

This doc should stay consistent with that file.

OpenAI-compatible adapters (e.g. DeepSeek/OpenRouter/SiliconFlow) are configured via:

- `siumai-provider-openai-compatible` (provider configs + capabilities list)

## OpenAI-compatible providers (presets)

For OpenAI-compatible providers, Siumai treats the provider id as a **preset** that selects:

- base URL
- response field mappings (content/thinking/tool_calls)
- a best-effort capabilities list

The canonical implementation lives in:

- `siumai-provider-openai-compatible` (preset registry)
- `siumai-protocol-openai` (protocol mapping for Chat Completions + Responses)

### Baseline expectations

All presets are treated as:

- Language: Y (chat)
- Streaming: Y (SSE)

Everything else (tools/embedding/rerank/image/vision/reasoning) is **provider-specific** and should be read from the preset config.

### Example presets (non-exhaustive)

| Preset id | Tools | Vision | Embedding | Image | Rerank | Reasoning |
| --- | --- | --- | --- | --- | --- | --- |
| `deepseek` | Y | Y | N | N | N | Y |
| `openrouter` | Y | Y | Y | N | N | Y |
| `siliconflow` | Y | Y | Y | Y | Y | Y |
| `together` | Y | Y | Y | Y | N | N |

See: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs`.

## Legend

- Y Supported
- P Partially supported / varies by model
- N Not supported / not applicable

## Matrix: Model families + core capabilities

| Provider id | Vercel package | Primary crate(s) | Language | Streaming | Tools | Vision | Embedding | Image | Rerank | Speech (TTS) | Transcription (STT) | Files | Thinking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai` | `@ai-sdk/openai` | `siumai-provider-openai` + `siumai-protocol-openai` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | P |
| `azure` | `@ai-sdk/azure` | `siumai-provider-azure` + `siumai-protocol-openai` | Y | Y | Y | N | Y | Y | N | Y | Y | Y | N |
| `anthropic` | `@ai-sdk/anthropic` | `siumai-provider-anthropic` + `siumai-protocol-anthropic` | Y | Y | Y | Y | N | N | N | N | N | N | Y |
| `gemini` | `@ai-sdk/google` | `siumai-provider-gemini` + `siumai-protocol-gemini` | Y | Y | Y | Y | Y | Y | N | N | N | Y | Y |
| `vertex` | `@ai-sdk/google-vertex` | `siumai-provider-google-vertex` + `siumai-protocol-gemini` | Y | Y | Y | Y | Y | Y (Imagen) | N | N | N | N | P |
| `anthropic-vertex` | `@ai-sdk/google-vertex` (Anthropic via Vertex) | `siumai-provider-google-vertex` + `siumai-protocol-anthropic` | Y | Y | Y | N | N | N | N | N | N | N | N |
| `groq` | `@ai-sdk/groq` | `siumai-provider-groq` | Y | Y | Y | N | N | N | N | Y | Y | N | N |
| `xai` | `@ai-sdk/xai` | `siumai-provider-xai` | Y | Y | Y | Y | N | N | N | N | N | N | N |
| `ollama` | N | `siumai-provider-ollama` | Y | Y | Y | N | Y | N | N | N | N | N | N |
| `minimaxi` | N | `siumai-provider-minimaxi` | Y | Y | Y | N | N | Y | N | Y | N | Y | N |
| `cohere` | `@ai-sdk/cohere` | `siumai-provider-cohere` | N | N | N | N | N | N | Y | N | N | N | N |
| `togetherai` | `@ai-sdk/togetherai` | `siumai-provider-togetherai` | N | N | N | N | N | N | Y | N | N | N | N |
| `bedrock` | `@ai-sdk/amazon-bedrock` | `siumai-provider-amazon-bedrock` | Y | Y | Y | N | N | N | Y | N | N | N | N |

Notes:

- The matrix describes **declared provider wiring + protocol support**, not whether a specific model id supports a feature.
- If a provider supports a capability in practice but does not declare it in `native_provider_metadata`, treat it as a documentation bug and fix the metadata (so the registry and docs stay consistent).

## Matrix: Tooling surfaces (high-level)

| Provider id | Function tools | Provider-defined tools (`Tool::ProviderDefined`) | Tool approval | Streaming tool-result bridging |
| --- | --- | --- | --- | --- |
| `openai` | Y | Y | Y | Y |
| `azure` | Y | Y | P | Y |
| `anthropic` | Y | Y | Y | Y |
| `gemini` | Y | Y (`google.*`) | P | P (tool-loop recommended) |
| `vertex` | Y | Y | P | P (tool-loop recommended) |

See also:

- `docs/provider-defined-tools-alignment.md`
- `docs/streaming-bridge-alignment.md`
- `docs/provider-implementation-alignment.md`
- `siumai-extras/src/server/axum.rs` (`to_transcoded_sse_response`)
- `siumai-extras/src/server/tool_loop.rs` (`tool_loop_chat_stream`)

## How to use this matrix (workflow)

1. **Pick a provider row** and decide the target capability set (Y / P / N).
2. For each Y cell:
   - ensure a **request fixture** and **response fixture** exist (or an integration test if a fixture is impractical),
   - ensure streaming behavior is validated (when applicable).
3. For each P cell:
   - document the supported subset (models, feature flags, protocol variants),
   - add tests that assert the expected warnings / lossy fallbacks.
