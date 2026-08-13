# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.10](https://github.com/YumchaLabs/siumai/compare/siumai-provider-deepseek-v0.11.0-beta.9...siumai-provider-deepseek-v0.11.0-beta.10) - 2026-08-13

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
- *(deepseek)* [**breaking**] adopt verified V4 model routes
- *(openai)* add provider-faithful Responses and Realtime runtime

### Other

- narrow Anthropic Skills support claims
- publish flagship provider journeys
- close validation ownership migration
- *(language)* [**breaking**] unify terminal outcomes and settlement
- *(options)* [**breaking**] replace origin layers with exact patches
- *(core)* [**breaking**] bind provider options to configured instances
- *(openai)* make Responses settlement dialect-owned
- *(openai)* move prompt cache intent to content annotations
- *(gemini)* [**breaking**] establish product-level provider
- *(providers)* [**breaking**] extract Moonshot and Volcengine owners
- *(openai)* [**breaking**] rename Responses protocol module
- [**breaking**] rebuild Siumai around provider-faithful family contracts
- establish Siumai Next architecture baseline

### Changed

- Add a provider-owned Anthropic-compatible Messages mode with an independent official endpoint,
  `x-api-key` authentication, replay-domain isolation, deterministic direct/stream contracts, and
  fail-closed validation for unsupported Anthropic controls.
- Replace the legacy universal-client wrapper with a provider-owned, model-independent
  `DeepSeekProvider` backed by the configured OpenAI-compatible language runtime.
- Expose Chat Completions and Responses as explicit API modes with provider-owned typed options,
  open model identifiers, and dated official support evidence.
- Preserve DeepSeek reasoning replay, JSON-object structured-output fallback, cache hit/miss
  accounting, reasoning usage, and canonical stream termination semantics; stable Chat now rejects
  the provider's beta-only strict-tools option before transport submission.
- Add an explicit beta Chat runtime for strict function tools and assistant-prefix completion. The
  beta endpoint, replay domain, recursive schema validation, and message annotation are separate
  from the stable Chat registration.

### Removed

- Remove the legacy builder, configuration, middleware, capability/specification bags, compatibility
  aliases, and duplicated standards modules.

## [0.11.0-beta.9](https://github.com/YumchaLabs/siumai/compare/siumai-provider-deepseek-v0.11.0-beta.8...siumai-provider-deepseek-v0.11.0-beta.9) - 2026-05-27

### Other

- harden clean architecture boundaries
- Merge branch 'main' of https://github.com/YumchaLabs/siumai
- deepen provider and bridge module boundaries

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
