# Siumai Architecture Overview

This document explains the core architecture after the 0.11 refactor.

## Layers

- Transformers (provider-agnostic)
  - `src/transformers/request.rs` — map unified requests to provider payloads
  - `src/transformers/response.rs` — map provider responses to unified types
  - `src/transformers/stream.rs` — convert provider streaming chunks to `ChatStreamEvent` (multi-event)
  - `src/transformers/audio.rs`, `src/transformers/files.rs` — TTS/STT and file APIs

- Executors (HTTP orchestration)
  - `src/executors/chat.rs` — non-streaming and streaming chat via `HttpChatExecutor`
  - `src/executors/embedding.rs` — embeddings via `HttpEmbeddingExecutor`
  - `src/executors/image.rs` — image gen/edit/variation via `HttpImageExecutor`
  - `src/executors/audio.rs` — TTS/STT via `HttpAudioExecutor`
  - `src/executors/files.rs` — file upload/list/retrieve/delete/content via `HttpFilesExecutor`

- Streaming Utilities
  - `src/utils/streaming.rs` + `src/utils/sse_stream.rs` — SSE parsing with `eventsource-stream`,
    UTF-8 safety, multi-event emission helpers (`StreamFactory`, `SseEventConverter`)

- Registry + Factory
  - `src/registry/mod.rs` — central provider registry (ids, base_url, capabilities, adapters)
  - `src/registry/factory.rs` — provider construction helpers used by builders

## Design Goals

- Single source of truth for provider configuration (Registry)
- No duplication across streaming/non-streaming paths (Transformers + Executors)
- Multi-event streaming for accurate, lossless event reconstruction
- Backward-compatible public API; refactor localized internally

## Typical Flow

1. Builder resolves provider (via Registry) and calls Factory
2. Factory constructs the client with appropriate Transformers
3. Executors perform HTTP calls and use Transformers to map requests/responses
4. Streaming paths use `StreamFactory` + provider-specific `SseEventConverter`

## Notes

- Capability checks are advisory and never block operations
- Use `retry_api` facade for retrying operations
- Prefer Registry/Factory when adding or changing providers

## Provider Client Structure

- OpenAI
  - `siumai/src/providers/openai/client.rs` – client type and shared helpers (build executors/context)
  - `siumai/src/providers/openai/client/chat.rs` – Chat capability
  - `siumai/src/providers/openai/client/embedding.rs` – Embedding capability
  - `siumai/src/providers/openai/client/image.rs` – Image capability
  - `siumai/src/providers/openai/client/audio.rs` – Audio capability
  - `siumai/src/providers/openai/client/rerank.rs` – Rerank capability
  - `siumai/src/providers/openai/client/files.rs` – Files capability
  - `siumai/src/providers/openai/client/models.rs` – ModelListing capability

- Anthropic
  - `siumai/src/providers/anthropic/client.rs` – client type and shared helpers
  - `siumai/src/providers/anthropic/client/chat.rs` – Chat capability

- Gemini
  - `siumai/src/providers/gemini/client.rs` – client type and shared helpers
  - `siumai/src/providers/gemini/client/embedding.rs` – Embedding capability
  - `siumai/src/providers/gemini/client/image.rs` – Image capability
  - `siumai/src/providers/gemini/client/models.rs` – ModelListing capability

- Groq
  - `siumai/src/providers/groq/client.rs` – client type and shared helpers
  - `siumai/src/providers/groq/client/audio.rs` – Audio capability

- Ollama
  - `siumai/src/providers/ollama/client.rs` – client type and shared helpers (delegates to chat/embeddings)
  - Chat and Embedding in `siumai/src/providers/ollama/chat.rs`, `siumai/src/providers/ollama/embeddings.rs`

- xAI
  - `siumai/src/providers/xai/client.rs` – client type and shared helpers (delegates to chat/models)
  - Chat/Streaming in `siumai/src/providers/xai/chat.rs`, `siumai/src/providers/xai/streaming.rs`

Note: This refactor only changes file organization; public API remains stable. All capabilities share Executors + Transformers.

## Headers & Tracing

- Use `execution::http::headers::ProviderHeaders::*` to build provider headers consistently:
  - `openai(api_key, org, project, custom_headers)`
  - `anthropic(api_key, custom_headers)` (supports `anthropic-beta` via custom headers)
  - `groq`, `xai`, `gemini`, `ollama`
- Tracing headers: prefer OpenTelemetry via `siumai-extras` middleware (W3C traceparent), or use HTTP interceptors to add custom headers when needed.
- Merge order for custom headers (where applicable): base ProviderHeaders → `http_config.headers` → provider‑specific extras (e.g., adapter/custom maps).
- Multipart endpoints (files/audio/image uploads) must not force `application/json`; rely on Executors to set `multipart/form-data`.
