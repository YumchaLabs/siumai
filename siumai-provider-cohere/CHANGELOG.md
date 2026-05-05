# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-provider-cohere-v0.11.0-beta.6...siumai-provider-cohere-v0.11.0-beta.7) - 2026-05-05

### Added

- add AI SDK rerank result views
- align provider settings package surfaces
- *(types)* align shared ai sdk type surface
- refactor

### Fixed

- *(text)* retain provider request metadata
- *(completion)* preserve response bodies
- *(chat)* preserve provider response bodies
- *(cohere)* preserve embedding response body
- *(cohere)* preserve rerank response body
- *(rerank)* preserve raw response envelopes
- *(cohere)* align citation provider metadata
- map Cohere JSON object response format

### Other

- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events
- remove legacy event builder delta helpers

### Added

- Native Cohere `/v2` support now covers chat, embeddings, and rerank from the same provider
  crate, with public typed exports for `CohereClient`, chat/embed/rerank options, request
  extensions, and Cohere thinking/embedding enums.
- Curated Cohere `chat` / `embedding` / `rerank` model constants are now exported from the provider
  package, alongside AI SDK-style option aliases
  (`CohereLanguageModelOptions`, `CohereEmbeddingModelOptions`,
  `CohereRerankingModelOptions`) and deprecated migration aliases for side-by-side compile checks.
- Native Cohere provider now also exposes package-level `CohereProviderSettings` plus `VERSION` on
  the provider-owned/public Rust surface. The new settings carrier keeps provider construction
  model-agnostic (`into_builder()`, `into_builder_for_model(...)`, `into_config_for_model(...)`),
  and the underlying builder/config surfaces now expose honest header helpers instead of requiring
  indirect HTTP-config mutation for audited package-level header parity.

### Changed

- The canonical native Cohere path now requires an explicit model id instead of inheriting the old
  rerank-biased provider-wide default model.

### Fixed

- The provider crate now matches the audited AI SDK `@ai-sdk/cohere` architecture more closely:
  the native package is no longer rerank-only, and the OpenAI-compatible Cohere preset is no
  longer the canonical embedding story.
- Native Cohere chat now mirrors the audited AI SDK provider-defined-tool warning behavior more
  closely: provider-defined tools are filtered from `/v2/chat` requests but still surface as
  stable `unsupported { feature: "provider-defined tool <id>" }` warnings on both non-stream
  responses and streamed terminal `ChatResponse` values.
- Native Cohere chat streaming now also emits a stable `StreamStart` part plus runtime `raw`
  chunks when `includeRawChunks` is enabled, bringing the stream lane closer to the audited AI SDK
  `stream-start -> raw -> response-metadata -> ...` order.
- Native Cohere chat streaming now also preserves first-chunk parse-failure lifecycle ordering:
  invalid SSE payloads emit `stream-start` before optional runtime `raw` and the parse error, and
  a later real `message-start` chunk no longer duplicates the stream-start event after that
  fallback path.
- Native Cohere chat responses and streamed terminal `ChatResponse` values now preserve the raw
  Cohere `finish_reason` on stable `raw_finish_reason` instead of dropping it after normalization.
- Cohere embedding request shaping now also enforces the audited AI SDK `outputDimension` contract
  (`256`, `512`, `1024`, or `1536`) instead of silently accepting arbitrary values.
- Cohere embedding execution now also enforces the audited AI SDK batch-size guard: one native
  `/v2/embed` call may carry at most `96` inputs.
- Cohere rerank response transformation now preserves an AI SDK-style response envelope with the
  raw response body even when callers use the provider transformer directly outside the HTTP
  executor path.
- Native Cohere chat response transformation now builds against the current shared
  response-metadata field.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Cohere fixture parity updates (including rerank coverage).
