# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
