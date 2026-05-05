# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-provider-togetherai-v0.11.0-beta.6...siumai-provider-togetherai-v0.11.0-beta.7) - 2026-05-05

### Added

- add AI SDK rerank result views
- align provider settings package surfaces
- *(togetherai)* align package alias names with ai sdk
- *(provider)* align package surfaces across providers
- refactor

### Fixed

- *(togetherai)* preserve rerank response metadata
- *(rerank)* preserve raw response envelopes

### Other

- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events

### Added

- Add public `TogetherAiImageOptions` plus `TogetherAiImageRequestExt` for
  `ImageGenerationRequest` and `ImageEditRequest`, matching the audited AI SDK
  `TogetherAIImageModelOptions` lane under `providerOptions.togetherai` with camelCase input
  aliases and merge-safe request helpers.
- Add curated TogetherAI `chat/completion/embedding/image/rerank` model constants plus AI SDK-style
  `TogetherAiImageModelOptions` / `TogetherAiRerankingModelOptions` aliases, keeping deprecated
  compatibility aliases available for side-by-side package export checks.
- Native TogetherAI provider now also exposes package-level `TogetherAIProviderSettings` plus
  `VERSION` on the provider-owned/public Rust surface. The new settings carrier keeps provider
  construction model-agnostic and maps the audited `apiKey` / `baseURL` / `headers` / `fetch`
  subset onto the real provider-owned builder/config path.

### Fixed

- Preserve AI SDK-style rerank response metadata (`modelId` and raw response body) in the
  provider-owned response transformer, including direct fixture-transformer usage outside the
  HTTP executor path.

## [0.11.0-beta.5] - 2026-01-15

### Added

- TogetherAI fixture parity updates (including rerank coverage).
