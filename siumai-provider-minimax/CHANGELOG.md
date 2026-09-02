# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.11](https://github.com/YumchaLabs/siumai/compare/siumai-provider-minimax-v0.11.0-beta.10...siumai-provider-minimax-v0.11.0-beta.11) - 2026-09-02

### Added

- *(facade)* expose direct transport configuration
- *(providers)* add Gemini multimodal embedding and Cohere transcription

### Fixed

- *(review)* harden facade convergence contracts
- *(providers)* refresh audited model contracts

### Other

- *(facade)* teach the typed Siumai hierarchy
- *(facade)* document unified family calls (U6)
- close trusted CONNECT milestone
- close direct transport infrastructure milestone
- *(providers)* finish HTTP transport settings migration

## [0.11.0-beta.10](https://github.com/YumchaLabs/siumai/compare/siumai-provider-minimax-v0.11.0-beta.9...siumai-provider-minimax-v0.11.0-beta.10) - 2026-08-13

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
- *(anthropic)* [**breaking**] align current Messages contracts
- *(core)* [**breaking**] enforce typed stream settlement
- *(core)* [**breaking**] enforce canonical language replay semantics
- *(minimax)* [**breaking**] canonicalize provider identity and protocols
- *(openai)* add provider-faithful Responses and Realtime runtime
- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming
- clean codes, do some refactor
- improve middleware and support mcp integration in siumai-extras
- improve orchestrator and rework examples
- use workspace, extract siumai-extras crate, fix bugs
- unified methods on ChatCapability. improve structured out parity
- support telemetry
- add registry. middleware and orchestrator, support json schema
- support http interceptor, add fixture tests, fix streaming and response api, add examples
- support vertex and improve auth
- support parameters mapping and add before send, add tests
- new retry api, clean codes. version v0.10.3
- unified HTTP client across providers
- completely refactored OpenAI-compatible provider system with centralized configuration through unified registry system. Update to v0.10.0
- added type-safe embedding configuration options for each provider (GeminiEmbeddingOptions with task types, OpenAiEmbeddingOptions with custom dimensions, OllamaEmbeddingOptions with model parameters) through extension traits, enabling optimized embeddings while maintaining unified interface
- add basic openai image generation support. add siliconflow rerank support #3 , fix #4
- All client types, builders, and configuration structs now implement `Clone` for seamless concurrent usage and multi-threading scenarios
- add provider feature flags
- add send sync support to request builder
- more model constants and cleanup
- ollama support thinking and streaming
- add real llm integration test
- support embedding
- support embed
- support openai response api
- ollama support thinking
- re-work examples, improve interface, support ollama, add ci
- support ollama
- implement anthropic and openai providers
- init project

### Fixed

- fix clippy and modify changelog
- modify readme
- fix clippy and tests
- fixe SiumaiBuilder to allow Ollama provider creation without API key, as Ollama doesn't require authentication, update to v0.9.0

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
- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs
- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events
- *(refactor)* finalize fearless refactor policies and release gate
- clarify builder as compat
- *(examples)* migrate to family APIs
- *(examples)* config-first openai-compatible vendors
- align migration and websocket examples
- beta.6 recommended usage
- add beta.6 migration guide
- align docs with family APIs
- *(examples)* switch to family APIs
- *(openai)* document responses default and ws constraints
- Merge branch 'refactor'
- reorganize docs structure
- *(examples)* migrate provider_ext paths after scoping
- [**breaking**] split provider crates and extract openai-compatible base
- *(alpha.5)* provider-owned standards and extensions
- split library into modules
- clean codes, improve
- clean codes, improve
- clean codes, improve orchestrator
- refactor http config
- optimize code file structure
- remove old http mod
- remove provider model, restructure codes
- clean codes, remove provider specific streaming struct since we have executors
- continue to split codes and add tests
- refactor using transformer
- go on refactor adapter architecture
- fix doc
- simplify model constants and cleanup architecture
- prepare for v0.8.0
- prepare for v0.7.0
- separate type file
- release v0.4.0
- change interface

### Added

- Added portable ImageModel and SpeechModel adapters alongside the existing typed native image,
  speech, music, and video APIs.
- Added mode-specific endpoint/replay ownership for Messages and OpenAI-compatible language paths,
  including fail-closed custom-domain handling for caller-controlled endpoints.

## [0.11.0-beta.9](https://github.com/YumchaLabs/siumai/compare/siumai-provider-minimax-v0.11.0-beta.8...siumai-provider-minimax-v0.11.0-beta.9) - 2026-05-27

### Other

- retire minimax ollama model constants
- harden clean architecture boundaries
- Merge branch 'main' of https://github.com/YumchaLabs/siumai
- deepen provider and bridge module boundaries

### Removed

- Removed the legacy provider-local `providers::minimax::model_constants` module. Use the curated
  `providers::minimax::models` surface for chat, speech, image, video, and music model ids.

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-provider-minimax-v0.11.0-beta.7...siumai-provider-minimax-v0.11.0-beta.8) - 2026-05-18

### Other

- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- harden crate boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs

### Added

- MiniMax now exposes a provider-owned curated model surface for the public families (`chat`,
  `speech`, `video`, `music`, `image`) so facade/catalog/default-model consumers can reuse one
  source instead of handwritten arrays.

### Fixed

- Stream metadata normalization now also reaches typed finish parts, so MiniMax streaming no
  longer leaks `providerMetadata["anthropic"]` on the stable finish-part lane.

## [0.11.0-beta.5] - 2026-01-15

### Added

- MiniMax provider extracted into its own crate as part of the workspace split.

### Changed

- Dependency graph decoupled to reduce cross-provider coupling.
