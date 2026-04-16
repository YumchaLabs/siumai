# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add public `TogetherAiImageOptions` plus `TogetherAiImageRequestExt` for
  `ImageGenerationRequest` and `ImageEditRequest`, matching the audited AI SDK
  `TogetherAIImageModelOptions` lane under `providerOptions.togetherai` with camelCase input
  aliases and merge-safe request helpers.
- Add curated TogetherAI `chat/completion/embedding/image/rerank` model constants plus AI SDK-style
  `TogetherAiImageModelOptions` / `TogetherAiRerankingModelOptions` aliases, keeping deprecated
  compatibility aliases available for side-by-side package export checks.

## [0.11.0-beta.5] - 2026-01-15

### Added

- TogetherAI fixture parity updates (including rerank coverage).
