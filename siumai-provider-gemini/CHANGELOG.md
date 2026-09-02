# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.11](https://github.com/YumchaLabs/siumai/compare/siumai-provider-gemini-v0.11.0-beta.10...siumai-provider-gemini-v0.11.0-beta.11) - 2026-09-02

### Added

- *(facade)* expose direct transport configuration
- *(providers)* add Gemini multimodal embedding and Cohere transcription

### Fixed

- *(review)* harden facade convergence contracts

### Other

- *(facade)* teach the typed Siumai hierarchy
- *(facade)* document unified family calls (U6)
- simplify transport settings ownership
- close trusted CONNECT milestone
- close direct transport infrastructure milestone
- *(providers)* adopt shared HTTP transport settings
- *(providers)* [**breaking**] remove volatile model eligibility gates

## [0.11.0-beta.10](https://github.com/YumchaLabs/siumai/compare/siumai-provider-gemini-v0.11.0-beta.9...siumai-provider-gemini-v0.11.0-beta.10) - 2026-08-13

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
- *(minimax)* [**breaking**] canonicalize provider identity and protocols
- *(gemini)* [**breaking**] refresh current model policies
- *(openai)* add provider-faithful Responses and Realtime runtime
- [**breaking**] establish provider-faithful family surface

### Fixed

- *(gemini)* avoid guessed tool capability warnings

### Other

- narrow Anthropic Skills support claims
- publish flagship provider journeys
- close validation ownership migration
- *(language)* [**breaking**] unify terminal outcomes and settlement
- *(options)* [**breaking**] replace origin layers with exact patches
- *(core)* [**breaking**] remove runtime model policy
- *(core)* [**breaking**] bind provider options to configured instances
- align beta.9 migration and downstream handoff
- *(gemini)* [**breaking**] establish product-level provider
- *(providers)* [**breaking**] extract Moonshot and Volcengine owners
- [**breaking**] rebuild Siumai around provider-faithful family contracts
- establish Siumai Next architecture baseline

### Changed

- Replaced the image-shaped `GoogleImage*` public API with product-level `Gemini*` types without
  compatibility aliases.
- Moved stable-v1 Interactions image wire mapping into `siumai-protocol-gemini` and made endpoint
  provenance independent from caller-supplied transport policy labels.
- Made `GeminiProvider` the product-level owner for Interactions language, portable text embedding,
  buffered speech, Files, and typed Veo jobs, while keeping unsupported product areas explicitly
  provider-owned or deferred.

### Added

- Added stable-v1 Interactions language and typed portable family registrations for the implemented
  embedding, image, and speech slices.
- Added bounded provider-owned Files and Veo submit/status clients with open model identifiers and
  no SDK-maintained region or availability catalog.

## [0.11.0-beta.9](https://github.com/YumchaLabs/siumai/compare/siumai-provider-gemini-v0.11.0-beta.8...siumai-provider-gemini-v0.11.0-beta.9) - 2026-05-27

### Fixed

- clean ci all-features failures

### Other

- refresh provider model catalogs
- move gemini provider metadata into protocol crate
- harden clean architecture boundaries
- Merge branch 'main' of https://github.com/YumchaLabs/siumai
- isolate provider-owned response adapters
- deepen provider and bridge module boundaries

### Changed

- Re-export Gemini typed provider metadata from `siumai-protocol-gemini` so the provider crate no
  longer owns GenerateContent response metadata shapes.

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-provider-gemini-v0.11.0-beta.7...siumai-provider-gemini-v0.11.0-beta.8) - 2026-05-18

### Added

- reconnect google interactions streams
- stream google interactions events
- execute google interactions non-stream requests
- parse google interactions responses
- add google interactions agent request conversion
- add google interactions request conversion
- expose google interactions package boundary

### Other

- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- harden crate boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs

### Fixed

- Gemini content-part metadata helpers now include stable `reasoning-file` and `custom` parts so
  the V4-capable content model does not regress typed metadata extraction.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Gemini provider extracted into its own crate as part of the workspace split.
- Expanded tool and streaming fixtures aligned with Vercel AI SDK.

### Fixed

- Tool result encoding alignment and Imagen default behavior parity.
