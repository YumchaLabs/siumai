# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-provider-deepseek-v0.11.0-beta.7...siumai-provider-deepseek-v0.11.0-beta.8) - 2026-05-18

### Other

- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- harden crate boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs

### Added

- The provider-owned typed surface now exposes AI SDK-style `DeepSeekLanguageModelOptions` with
  deprecated `DeepSeekChatOptions` migration coverage.
- The provider-owned public model surface now exposes curated `chat` constants plus
  `models::ALL_CHAT` / `model_sets` for the stable `deepseek-chat` and `deepseek-reasoner`
  subset.
- Native DeepSeek provider now also exposes package-level `DeepSeekProviderSettings` plus
  `VERSION` on the provider-owned/public Rust surface. The new settings carrier keeps provider
  construction model-agnostic and maps the audited `apiKey` / `baseURL` / `headers` / `fetch`
  subset onto the real OpenAI-compatible-backed builder/config path.

### Fixed

- DeepSeek request/response metadata now follows the audited AI SDK custom provider-root contract
  more closely: request shaping reads provider-owned options from the runtime namespace instead of
  hardcoded `deepseek`, response metadata stays under that resolved root, and typed helpers now
  expose keyed metadata accessors for explicit custom-root reads.

## [0.11.0-beta.5] - 2026-01-15

### Added

- DeepSeek provider crate and initial fixture alignment.
