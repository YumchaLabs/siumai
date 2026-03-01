# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-spec-v0.11.0-beta.5...siumai-spec-v0.11.0-beta.6) - 2026-03-01

### Added

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

- *(spec)* remove reqwest conversion feature
- *(spec)* make reqwest optional
- *(spec)* introduce siumai-spec crate
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
