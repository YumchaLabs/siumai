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

## Headers & Tracing

- Use `utils::http_headers::ProviderHeaders::*` to build provider headers consistently:
  - `openai(api_key, org, project, custom_headers)`
  - `anthropic(api_key, custom_headers)` (supports `anthropic-beta` via custom headers)
  - `groq`, `xai`, `gemini`, `ollama`
- Always inject tracing headers with `inject_tracing_headers(&mut headers)` to include `X-Trace-Id`, `X-Span-Id`, and optionally `traceparent` (W3C).
- Merge order for custom headers (where applicable): base ProviderHeaders → `http_config.headers` → provider‑specific extras (e.g., adapter/custom maps).
- Multipart endpoints (files/audio/image uploads) must not force `application/json`; rely on Executors to set `multipart/form-data`.
