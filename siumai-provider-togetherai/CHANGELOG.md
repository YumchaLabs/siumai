# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.9](https://github.com/YumchaLabs/siumai/compare/siumai-provider-togetherai-v0.11.0-beta.8...siumai-provider-togetherai-v0.11.0-beta.9) - 2026-05-27

### Other

- update changelogs for architecture boundary refactor
- harden clean architecture boundaries
- Merge branch 'main' of https://github.com/YumchaLabs/siumai
- deepen provider and bridge module boundaries

### Added

- Added provider-owned TogetherAI image runtime support, including generation/edit body mapping,
  provider-option merging, image edit validation, response parsing, and HTTP execution.
- Added shared TogetherAI JSON header construction for provider-owned image and rerank paths.

### Changed

- The registry now composes TogetherAI provider-owned image and rerank clients plus the shared
  OpenAI-compatible text/audio runtime; TogetherAI image execution no longer lives in the registry
  factory.

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-provider-togetherai-v0.11.0-beta.7...siumai-provider-togetherai-v0.11.0-beta.8) - 2026-05-18

### Other

- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- harden crate boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs

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
