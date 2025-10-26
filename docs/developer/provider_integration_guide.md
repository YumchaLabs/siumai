# Provider Integration Guide

This guide explains how to integrate a new provider with the 0.11+ architecture.

Two main paths:

1) OpenAI-Compatible providers (preferred if the provider exposes an OpenAI-like API)
2) Native providers (full custom mapping)

Builder Flow (since 0.11)
- `SiumaiBuilder::build` delegates to `src/provider/build.rs`.
- Keep `src/provider.rs` slim; do not add provider-specific build logic there.
- For OpenAI-compatible providers, prefer using Registry+Factory helpers to avoid duplication.

Headers & Tracing (all providers)
- Build headers via `execution::http::headers::ProviderHeaders::*` (OpenAI/Anthropic/Groq/xAI/Gemini/Ollama).
- Inject tracing headers (`inject_tracing_headers`) for `X-Trace-Id`, `X-Span-Id`, optional `traceparent`.
- Merge `http_config.headers` for custom entries (auth overrides, org/project, feature flags like `anthropic-beta`).
- For multipart (files/audio/image uploads), do not set `application/json`; the executor sets `multipart/form-data`.

## 1) OpenAI-Compatible Providers

Implement an adapter and register the provider.

Steps:
- Add a `ProviderConfig` entry (id, name, base_url, capabilities, field mappings)
  - See: `src/providers/openai_compatible/config.rs`
- Implement a `ProviderAdapter` (custom headers, model validation, field access)
  - See: `src/providers/openai_compatible/adapter.rs`
- Register adapter in the compat registry (if not already)
  - See: `src/providers/openai_compatible/registry.rs`
- Builder path: no changes required — `Siumai::builder().provider_id("your_id")`
  automatically resolves via `ProviderRegistryV2` and constructs a client through
  `build_openai_compatible_client`.

Minimum mapping responsibilities:
- Request params shaping for Chat/Embedding/Image (if needed) via adapter hooks
- Field mappings for content/thinking/tool_calls for both non-streaming and streaming

## 2) Native Providers

Implement provider-specific Transformers and wire them with Executors.

Steps:
- Implement Transformers
  - `RequestTransformer` for Chat/Embedding/Image requests
  - `ResponseTransformer` for Chat/Embedding/Image responses
  - `StreamChunkTransformer` (if streaming)
- Use Executors in your client
  - `HttpChatExecutor` (non-streaming + streaming)
  - `HttpEmbeddingExecutor`, `HttpImageExecutor`, etc.
- Register the provider in `ProviderRegistryV2`
  - Add a `register_native(...)` record with base_url and capabilities
- Add a Factory helper to construct the client (if needed), and let builders
  call the Factory

## Minimal Example: Streaming Converter

```rust
use eventsource_stream::Event;
use siumai::error::LlmError;
use siumai::stream::ChatStreamEvent;
use siumai::transformers::stream::StreamChunkTransformer;
use std::future::Future;
use std::pin::Pin;

#[derive(Clone)]
pub struct MyStreamTransformer;

impl StreamChunkTransformer for MyStreamTransformer {
    fn provider_id(&self) -> &str { "my_provider" }

    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>> {
        Box::pin(async move {
            // Parse event.data as JSON and emit one or more ChatStreamEvent items
            // Example: emit a StreamStart once, then ContentDelta for each chunk
            // Return empty vec for keep-alives or unknown types
            let mut out = Vec::new();
            // ... parse and push ChatStreamEvent::ContentDelta { .. } etc.
            out
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Optionally emit final StreamEnd if your provider doesn't send one
        None
    }
}
```

## Registry/Factory

- `src/registry/mod.rs` — central registry for provider records
- `src/registry/factory.rs` — helpers to build clients using Transformers/Executors

Registering a native provider:

```rust
let registry = siumai::registry::global_registry();
let mut guard = registry.lock().unwrap();
guard.register_native(
    "my_provider",
    "My Provider",
    Some("https://api.myprovider.com/v1".to_string()),
    siumai::traits::ProviderCapabilities::new().with_chat().with_streaming(),
);
```

## Tips

- Keep mapping logic in Transformers, not in clients
- Use `StreamFactory::create_eventsource_stream` for SSE providers
- Prefer Registry over hardcoding base URLs in builders/clients
- Add tests covering edge shapes: multiple choices, nested tool_calls,
  empty fields, and usage accumulation

## Streaming Test Best Practices

- For unit-level validation of converters, construct `eventsource_stream::Event` values and
  call `convert_event(event)` directly; verify multi-event emission.
- For end-to-end SSE parsing, build a byte stream of `data: ...\n\n` chunks and call
  `into_sse_stream()` (see `tests/streaming/openai_compatible_end_to_end_sse_test.rs`).
- Include a `[DONE]` SSE line and call `handle_stream_end()` to finalize the sequence.
- Validate the presence/order of key events: `StreamStart`, one or more `ContentDelta`,
  optional `ThinkingDelta`/`ToolCallDelta`/`UsageUpdate`, and `StreamEnd`.
