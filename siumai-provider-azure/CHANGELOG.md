# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-provider-azure-v0.11.0-beta.6...siumai-provider-azure-v0.11.0-beta.7) - 2026-05-05

### Added

- align provider settings package surfaces
- *(ai-sdk)* align shared structural surfaces and builder helpers
- *(audio)* preserve request metadata on audio results
- *(speech)* align shared tts request options
- *(audio)* align speech and transcription helper results
- *(media)* align helper empty-result semantics
- refactor
- *(streaming)* align gemini stable parts and extras consumers

### Fixed

- *(ci)* satisfy clippy feature matrix
- *(azure)* remove openai provider dependency
- *(completion)* preserve response bodies
- *(completion)* preserve raw finish reasons
- align file upload provider option defaults
- *(openai)* warn on unsupported speech options

### Other

- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events
- stop emitting legacy stream deltas
- *(release)* prepare v0.11.0-beta.7

### Added

- Native Azure OpenAI provider now also exposes package-level `AzureOpenAIProviderSettings` plus
  `VERSION` on the provider-owned/public Rust surface. The new settings carrier keeps provider
  construction model-agnostic (`into_builder()`, `into_builder_for_model(...)`,
  `into_config_for_model(...)`), Azure builder/config surfaces now also expose honest
  `resourceName` and header helpers, and audited package-level inputs such as `apiVersion` and
  `useDeploymentBasedUrls` now have a direct Rust-side settings carrier instead of only indirect
  builder/config wiring.

### Fixed

- Native Azure OpenAI completion now follows the AI SDK completion-family execution path on the
  real Azure `/completions` deployment route: non-stream and streamed calls preserve
  `api-version`, structured prompts follow the audited completion materialization rules, completion
  provider options merge audited OpenAI/Azure namespaces, and completion responses preserve raw
  `choices[0].logprobs` under provider-owned Azure metadata.
- Native Azure completion streaming now also honors runtime-only `includeRawChunks` on the audited
  `/completions` deployment route: stable `stream-start`, `raw`, `response-metadata`, `text-*`,
  and terminal `finish` parts are emitted while preserving legacy `ContentDelta` / `StreamEnd`.
- Native Azure completion streaming terminal responses now preserve raw provider finish reasons on
  stable `ChatResponse.raw_finish_reason` instead of dropping them at stream end.
- Azure content-part metadata helpers now preserve metadata on stable `reasoning-file` and
  `custom` parts in addition to the older content-part variants.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Azure OpenAI provider crate and request/URL fixtures.

### Changed

- Provider wired into the registry and builder surface.
