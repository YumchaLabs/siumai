# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added portable final-result/batch transcription over the provider-owned Speech-to-Text API while
  retaining buffered speech synthesis as the portable family surface.

## [0.11.0-beta.9](https://github.com/YumchaLabs/siumai/compare/siumai-provider-elevenlabs-v0.11.0-beta.8...siumai-provider-elevenlabs-v0.11.0-beta.9) - 2026-05-27

### Added

- *(elevenlabs)* add voice edit resources
- *(elevenlabs)* add PVC verification resources
- *(elevenlabs)* add PVC sample resources
- *(elevenlabs)* add PVC voice metadata resources
- *(elevenlabs)* add IVC voice creation
- *(elevenlabs)* add voice delete resources
- *(elevenlabs)* add voice settings resources
- *(elevenlabs)* add pronunciation dictionary download
- *(elevenlabs)* add pronunciation dictionary rule mutations
- *(elevenlabs)* add pronunciation dictionary metadata update
- *(elevenlabs)* add pronunciation dictionary file creation
- *(elevenlabs)* add pronunciation dictionary rule creation
- *(elevenlabs)* add pronunciation dictionary resources
- *(elevenlabs)* add voice resources
- *(elevenlabs)* add audio provider crate
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

- clean ci all-features failures
- fix clippy and modify changelog
- modify readme
- fix clippy and tests
- fixe SiumaiBuilder to allow Ollama provider creation without API key, as Ollama doesn't require authentication, update to v0.9.0

### Other

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
