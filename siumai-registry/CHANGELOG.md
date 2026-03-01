# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-registry-v0.11.0-beta.5...siumai-registry-v0.11.0-beta.6) - 2026-03-01

### Added

- *(registry)* set default models for rerank providers
- *(registry)* add cohere and togetherai rerank factories
- *(openai)* default unified builder to responses api
- *(openai-websocket)* expose unified builder session APIs
- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- *(registry)* prefer native metadata in provider catalog
- *(clippy)* satisfy -D warnings

### Other

- *(features)* include cohere and togetherai in all-providers
- *(features)* narrow all-providers aggregation
- *(registry)* reserve metadata-only provider ids
- *(registry)* expand ProviderFactory contract tests coverage
- *(registry)* use centralized provider ids in factories
- *(registry)* remove redundant OpenRouter factory
- *(registry)* centralize provider ids and variant parsing
- *(registry)* expand ProviderFactory contract tests
- *(registry)* add ProviderFactory contract tests
- *(resolver)* provider_id-first inference helpers
- *(builder)* route by provider_id only
- *(registry)* factory-first unified build routing
- *(registry)* propagate provider_id to BuildContext
- *(registry)* delegate base_url defaults
- *(registry)* let factories resolve api keys
- *(registry)* default provider_id for provider()
- *(registry)* normalize provider ids
- *(registry)* dedupe BuildContext
- *(registry)* reuse provider_id mapping
- *(registry)* centralize provider resolution
- *(openai-compatible)* move vendor macro
- *(openai)* document responses default and ws constraints
- *(streaming)* delegate cancellation to provider handles
- *(streaming)* improve cancellable streams
- *(changelog)* release 0.11.0-beta.5

## [0.11.0-beta.5] - 2026-01-15

### Added

- Registry extracted as a standalone crate (factories + handles), aligning with the workspace split.
- Wiring for provider crates to be constructed via factories instead of hard-coded built-ins.

### Changed

- Modularized provider/factory modules to reduce cross-layer coupling.
