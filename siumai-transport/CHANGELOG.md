# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.11](https://github.com/YumchaLabs/siumai/compare/siumai-transport-v0.11.0-beta.10...siumai-transport-v0.11.0-beta.11) - 2026-09-02

### Added

- *(mcp)* support trusted HTTP routing
- *(transport)* add trusted CONNECT routing
- *(facade)* expose direct transport configuration
- add one-shot call deadlines and retry caps
- *(providers)* add Gemini multimodal embedding and Cohere transcription

### Fixed

- *(review)* harden facade convergence contracts
- *(transport)* close route authority review gaps
- *(mcp)* [**breaking**] bound transport and prevent hidden replay

### Other

- *(facade)* teach the typed Siumai hierarchy
- *(facade)* document unified family calls (U6)
- close trusted CONNECT milestone
- close direct transport infrastructure milestone
- *(transport)* centralize provider HTTP settings
- *(options)* [**breaking**] narrow raw provider authority

## [0.11.0-beta.10](https://github.com/YumchaLabs/siumai/compare/siumai-transport-v0.11.0-beta.9...siumai-transport-v0.11.0-beta.10) - 2026-08-13

### Added

- *(siumai)* expose OpenAI Responses WebSocket surface
- *(transport)* add exact RFC 6598 endpoint grant
- *(provider)* [**breaking**] deepen xAI and Groq media surfaces
- *(provider)* [**breaking**] expand Deepgram and ElevenLabs audio families
- *(provider)* [**breaking**] deepen MiniMax portable and voice surfaces
- *(provider)* [**breaking**] deepen Kimi and ARK surfaces
- *(provider)* [**breaking**] expand flagship and Chinese surfaces
- *(openai)* [**breaking**] add portable families and native resources
- *(gemini)* [**breaking**] add provider-owned product surfaces
- *(gemini)* [**breaking**] add faithful Interactions language
- *(core)* [**breaking**] enforce typed stream settlement
- *(minimax)* [**breaking**] canonicalize provider identity and protocols
- *(openai)* add provider-faithful Responses and Realtime runtime
- [**breaking**] establish provider-faithful family surface
- *(transport)* add bounded replay-safe provider transport
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

- *(transport)* require scoped local network grants
- *(transport)* version auth refresh and retry classification
- *(transport)* bound resource lifecycles and diagnostics
- *(transport)* share persistent SSE event IDs
- fix clippy and modify changelog
- modify readme
- fix clippy and tests
- fixe SiumaiBuilder to allow Ollama provider creation without API key, as Ollama doesn't require authentication, update to v0.9.0

### Other

- narrow Anthropic Skills support claims
- publish flagship provider journeys
- *(openai)* [**breaking**] remove invented request validation
- close validation ownership migration
- *(gemini)* [**breaking**] establish product-level provider
- *(providers)* [**breaking**] extract Moonshot and Volcengine owners
- [**breaking**] rebuild Siumai around provider-faithful family contracts
- *(transport)* make WebSocket sessions concurrently terminal
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
