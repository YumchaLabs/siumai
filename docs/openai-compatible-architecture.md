# OpenAI‑Compatible Architecture

This note explains how Siumai supports providers that expose an OpenAI‑like REST API while preserving the library’s unified programming model.

## Layers

- Unified Interface: `Siumai::builder()` exposes provider‑agnostic chat/streaming/embedding APIs
- Client Configuration: `LlmBuilder` builds HTTP clients, timeouts, proxies, headers
- Parameter Layer: Transformers validate and map common params to provider formats
- Provider Implementations:
  - Native: `providers/openai`, `providers/anthropic`, `providers/gemini`, etc.
  - OpenAI‑Compatible: `providers/openai_compatible/*` (adapters + registry)

## How Compatibility Works

1) Registry declares a provider id, base URL, and field/header mappings.
2) Requests use OpenAI shapes by default; adapters apply overrides as needed.
3) Streaming events are normalized to `ChatStreamEvent` (content/tool/thinking deltas).
4) Errors are classified and surfaced via `LlmError` with recovery hints.

## When To Use

- Rapidly onboard a provider that claims OpenAI API compatibility.
- Keep portability: switch providers by changing `provider_name`/`base_url`/`api_key`.
- Prefer native providers when you need deep, non‑OpenAI features.

## Gotchas

- Model naming may differ; expose sensible defaults in `default_models.rs`.
- Some providers require extra headers or beta flags; capture in the adapter.
- Streaming tool‑call indices/ids can differ; rely on the stream processor’s id‑based merge logic.

## References

- Source: `src/providers/openai_compatible/`
- Parameter layer: unified in `src/transformers/*`
- Stream normalization: `src/stream.rs`
