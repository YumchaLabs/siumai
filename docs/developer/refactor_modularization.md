# Refactor Modularization Plan (0.11+)

This document tracks the ongoing modularization refactor aimed at improving
maintainability, testability, and clarity of responsibilities.

## Goals

- Reduce oversized files into cohesive modules with single responsibility
- Remove code duplication across providers (adapters, headers, transformers)
- Keep public API stable
- Leverage Registry/Factory + Transformers/Executors consistently

## Status (current)

- Executors + Transformers adopted across providers
- Registry/Factory wired for OpenAI-compatible and native providers (OpenAI/Anthropic/Gemini)
- Streaming unified via eventsource-stream + multi-event converters
- Tests added for multi-event streaming sequences (unit + end-to-end SSE)
- Shared OpenAI standard adapter extracted to `src/providers/openai/adapter.rs`
- Traits modularization completed for: chat, embedding, image, vision, files, moderation,
  model_listing, completion, timeout, rerank, capabilities, provider_specific, audio (all re-exported via `traits.rs`)
- OpenAI transformers split into cohesive submodules with a `mod.rs` facade
- Gemini types split into cohesive submodules with a `mod.rs` facade
- Provider facade split: `SiumaiBuilder::build` delegated to `src/provider/build.rs`
- Headers unification (ProviderHeaders) progress:
  - OpenAI: chat/stream/embedding/audio/files/moderation unified with `ProviderHeaders::openai` (+ tracing, + custom headers)
  - Anthropic: unified via `ProviderHeaders::anthropic` (beta header supported via custom headers)
  - Groq: unified via `ProviderHeaders::groq`
  - xAI: unified via `ProviderHeaders::xai`
  - Gemini: chat/stream/embedding/image/files unified via `ProviderHeaders::gemini` (+ tracing, + custom headers)
  - OpenAI Rerank: unified via `ProviderHeaders::openai` (+ tracing, merges `http_config.headers`)
- Request headers tests:
  - Unit coverage in `tests/providers/provider_headers_test.rs`
  - End-to-end header flow tests in `tests/providers/request_headers_flow_test.rs` (OpenAI Files, Anthropic chat with beta, Gemini Files)
- Config improvement:
  - `GeminiConfig` now includes optional `http_config` to pass custom headers/proxy/user-agent

## Next Milestones

1) Traits module split
- [Done] Split `src/traits.rs` into capability-specific modules (chat, embedding, image, files,
  rerank, moderation, models, completion, timeout, vision, provider-specific, capabilities, audio)
- [Done] Keep `traits.rs` as a re-export facade (no public API change)
- [Done] Remove temporary `__legacy_*` placeholders

2) OpenAI transformers split
- [Done] Split `src/providers/openai/transformers.rs` into `transformers/{request.rs,response.rs,files.rs,stream.rs}`
- [Done] Kept `mod.rs` re-exports to preserve paths (`super::transformers::*`)
- Concentrate OpenAI-specific mapping in this folder; client calls remain via Executors

3) Gemini types split
- [Done] Split `src/providers/gemini/types.rs` into `types/{config.rs,generation.rs,content.rs}` (streaming models remain in `streaming.rs`)
- [Done] Added `types/mod.rs` as a facade re-export to avoid import breakage
- Keep streaming converters in `src/providers/gemini/streaming.rs` (local types untouched)

4) Provider facade split
- [Done] Extracted `Siumai` construction logic from `src/provider.rs` into `src/provider/build.rs`
- [Done] Kept `provider.rs` slim (facade + proxies only); `build_legacy` temporarily retained for reference

5) Headers unification
- [In Progress] Replace repeated header-building closures with `ProviderHeaders::*` utilities
  - [Done] OpenAI/Anthropic/Groq/xAI/Gemini unified (multipart paths carefully avoid JSON content-type)
  - [Done] OpenAI-Compatible: unified around `ProviderHeaders::openai` and merged `adapter.custom_headers` + `http_config.headers` + `config.custom_headers`, with tracing injection preserved
- Keep explicit tracing injection via `inject_tracing_headers`

## Migration Notes (developer-facing)

- `GeminiConfig` gained a new field: `http_config: Option<HttpConfig>`.
  - If constructing via struct literal, add `http_config: Some(HttpConfig::default())`.
  - Prefer using builders: `.with_http_config(HttpConfig::default())`.
- OpenAI header customization should go through `http_config.headers` (merged into ProviderHeaders paths).
- No public API surface changed for transformers/executors; module splits are re-exported via facades.
- OpenAI Rerank: Direct use of `OpenAiRerank::new` now requires an extra `HttpConfig` argument (for custom headers/proxy/user-agent). Usage via `OpenAiClient` is unchanged.

## Next Milestones

6) OpenAI-Compatible headers unification
- [Done] Unified header building in `openai_compatible` client using `ProviderHeaders::openai` as base, then merged adapter/custom headers and `http_config.headers`.
- [Done] Added E2E header test for OpenAI-compatible provider (auth + adapter custom headers present).

7) Provider facade cleanup
- [Done] Removed `build_legacy` after stabilizing `build.rs` path.
- [Done] Provider facade remains slim; `SiumaiBuilder::build` delegates to `src/provider/build.rs`.

8) Header flow test expansion
- [Done] Add flow tests for Groq/xAI chat paths to validate user-agent/auth/custom headers.
- [Done] Add negative tests ensuring no spurious `content-type: application/json` on multipart requests (audio/image/forms) for Groq STT and OpenAI Files upload.

9) Rerank header unification (optional)
- [Done] Unified `openai/rerank.rs` headers via `ProviderHeaders::openai` and wired `http_config` for custom headers.

## Guidelines

- Prefer small, verifiable PRs; run full test suite after each step
- Avoid changing external APIs; use facades to re-export
- Keep documentation and examples in sync with moves

## Appendix

- Architecture: `docs/developer/architecture.md`
- Integration Guide: `docs/developer/provider_integration_guide.md`
