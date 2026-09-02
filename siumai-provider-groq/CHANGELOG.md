# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.11](https://github.com/YumchaLabs/siumai/compare/siumai-provider-groq-v0.11.0-beta.10...siumai-provider-groq-v0.11.0-beta.11) - 2026-09-02

### Added

- *(facade)* expose direct transport configuration
- *(providers)* add Gemini multimodal embedding and Cohere transcription

### Fixed

- *(review)* harden facade convergence contracts

### Other

- *(facade)* teach the typed Siumai hierarchy
- *(facade)* document unified family calls (U6)
- close trusted CONNECT milestone
- close direct transport infrastructure milestone
- *(providers)* finish HTTP transport settings migration

## [0.11.0-beta.10](https://github.com/YumchaLabs/siumai/compare/siumai-provider-groq-v0.11.0-beta.9...siumai-provider-groq-v0.11.0-beta.10) - 2026-08-13

### Added

- *(siumai)* expose OpenAI Responses WebSocket surface
- *(provider)* [**breaking**] deepen xAI and Groq media surfaces
- *(provider)* [**breaking**] expand Deepgram and ElevenLabs audio families
- *(provider)* [**breaking**] deepen MiniMax portable and voice surfaces
- *(provider)* [**breaking**] deepen Kimi and ARK surfaces
- *(provider)* [**breaking**] expand flagship and Chinese surfaces
- *(openai)* [**breaking**] add portable families and native resources
- *(gemini)* [**breaking**] add provider-owned product surfaces
- *(gemini)* [**breaking**] add faithful Interactions language
- *(core)* [**breaking**] enforce typed stream settlement
- *(core)* [**breaking**] enforce canonical language replay semantics
- *(minimax)* [**breaking**] canonicalize provider identity and protocols
- *(openai)* add provider-faithful Responses and Realtime runtime

### Fixed

- *(openai)* canonicalize partial responses terminals

### Other

- narrow Anthropic Skills support claims
- publish flagship provider journeys
- close validation ownership migration
- *(language)* [**breaking**] unify terminal outcomes and settlement
- *(options)* [**breaking**] replace origin layers with exact patches
- *(core)* [**breaking**] remove runtime model policy
- *(core)* [**breaking**] bind provider options to configured instances
- *(openai)* make Responses settlement dialect-owned
- *(openai)* move prompt cache intent to content annotations
- align beta.9 migration and downstream handoff
- *(gemini)* [**breaking**] establish product-level provider
- *(providers)* [**breaking**] extract Moonshot and Volcengine owners
- *(openai)* [**breaking**] rename Responses protocol module
- [**breaking**] rebuild Siumai around provider-faithful family contracts
- establish Siumai Next architecture baseline

### Added

- Added portable buffered speech, URL-audio transcription, and URL-audio translation handles.
- Added typed Remote MCP options and output projections for the provider-owned Responses surface.

## [0.11.0-beta.9](https://github.com/YumchaLabs/siumai/compare/siumai-provider-groq-v0.11.0-beta.8...siumai-provider-groq-v0.11.0-beta.9) - 2026-05-27

### Other

- harden clean architecture boundaries
- Merge branch 'main' of https://github.com/YumchaLabs/siumai
- deepen provider and bridge module boundaries

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-provider-groq-v0.11.0-beta.7...siumai-provider-groq-v0.11.0-beta.8) - 2026-05-18

### Other

- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- harden crate boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs
﻿
### Added

- Added AI SDK-style `GroqProviderSettings` and `VERSION` exports for the audited package-level
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset.

- The provider-owned/public Groq surface now also exposes the AI SDK-style
  `GroqTranscriptionModelOptions` alias in addition to the earlier
  `GroqLanguageModelOptions` / deprecated `GroqProviderOptions` pair.
- Groq now also exposes AI SDK-style browser-search provider tools on the stable/public Rust path
  via `groq.browser_search`, including facade access through `provider_ext::groq::{tools,
  provider_tools}`.
- Groq typed response metadata now also preserves the upstream stable response-metadata fields
  `id`, `modelId`, and `timestamp`, and `GroqChatResponseExt` can read them consistently on
  non-stream plus stream-end responses across provider-owned/config/runtime paths.

### Fixed

- Groq transcription request coverage now follows the canonical shared `audio` input surface after
  the AI SDK-style STT request-shape alignment.
- Groq transcription typed options now serialize AI SDK-style `responseFormat` /
  `timestampGranularities`, multipart request shaping forwards `language` and
  `timestamp_granularities[]`, and STT responses retain `language` / `duration` plus raw
  `segments` / `x_groq` metadata instead of discarding them.
- The provider-owned Groq wrapper now preserves JSON Schema structured outputs on the
  config-first stream/chat path instead of silently downgrading them to
  `response_format = { "type": "json_object" }`.
- Groq chat runtime now recognizes `groq.browser_search` on supported GPT-OSS models, injects the
  native `{ "type": "browser_search" }` tool while preserving mixed function tools and
  `tool_choice`, and emits the same unsupported-model warning shape as AI SDK when browser search
  is requested on unsupported models.
- Groq typed language-model options now match the current AI SDK enum surface for
  `service_tier` and `reasoning_effort`, and the built-in `KIMI_K2_INSTRUCT` constant now points
  to the refreshed `moonshotai/kimi-k2-instruct-0905` model id instead of the decommissioned
  predecessor.
- Groq typed language-model options now also serialize AI SDK-style camelCase provider option keys
  (`serviceTier`, `reasoningEffort`, `reasoningFormat`, `topLogprobs`, `parallelToolCalls`,
  `user`, `structuredOutputs`, `strictJsonSchema`), while the provider-owned runtime still lowers
  them to Groq's wire snake_case fields and keeps `structuredOutputs: false` aligned with the
  expected JSON-object downgrade warning.
- Groq's built-in model catalog now matches the audited `@ai-sdk/groq` chat/transcription surface
  more closely: missing current chat ids such as `gemma2-9b-it`, `llama-guard-3-8b`,
  `llama3-{8b,70b}-8192`, `qwen-qwq-32b`, `qwen-2.5-32b`, and
  `deepseek-r1-distill-qwen-32b` are restored, while obsolete `compound-beta`, old vision/tool-use
  previews, and `gemma-7b-it` are removed from the public Groq catalog.
- Groq provider construction now also tracks the audited `groq-provider.ts` settings contract more
  closely: the compat `groq` preset resolves `GROQ_API_KEY`, `GroqConfig` now supports
  `from_env()` / `with_api_key(...)`, and `GroqBuilder` exposes the AI SDK-style `headers(...)`
  alias on top of the existing Rust-native HTTP configuration surface.

### Changed

- The provider-owned typed option surface now also exposes AI SDK-style
  `GroqLanguageModelOptions` plus deprecated `GroqProviderOptions`, reducing public naming drift
  against `@ai-sdk/groq`.
- Groq's provider-owned PlayAI speech models are now grouped separately from the AI SDK-aligned
  chat/transcription catalog, so the public model constants more clearly distinguish upstream
  package parity from Rust-only provider extensions.
- The provider-owned Groq wrapper no longer re-exports `GroqSttOptions` / `GroqTtsOptions` from
  the main `providers::groq::*` and facade `provider_ext::groq::{options::*, *}` lanes; those
  Rust-only audio escape hatches remain available under `providers::groq::ext::audio_options::*`
  and `provider_ext::groq::ext::audio_options::*`.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Groq provider extracted into its own crate as part of the workspace split.

### Changed

- Fixture parity and transport wiring aligned with the split architecture.
