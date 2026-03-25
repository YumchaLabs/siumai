# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-provider-openai-compatible-v0.11.0-beta.5...siumai-provider-openai-compatible-v0.11.0-beta.6) - 2026-03-25

### Added

- *(v4)* align provider parity and typed metadata surfaces
- *(providers)* align openai azure and compat packages
- *(providers)* wire interceptors and middlewares from config
- *(openai-compatible)* add from_builtin constructors
- *(provider)* add from_config for azure and openai-compatible
- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- resolve workspace clippy warnings
- *(provider)* unify chat request normalization and base url defaults
- *(embedding)* preserve request config on family model paths

### Other

- *(refactor)* finalize fearless refactor policies and release gate
- clarify builder as compat
- *(examples)* migrate to family APIs
- *(examples)* config-first openai-compatible vendors
- align migration and websocket examples
- beta.6 recommended usage
- add beta.6 migration guide
- *(release)* bump to 0.11.0-beta.6
- align docs with family APIs
- *(examples)* switch to family APIs
- *(client)* decouple LlmClient from ChatCapability
- *(openai-compatible)* move vendor macro
- *(openai)* document responses default and ws constraints
- *(changelog)* release 0.11.0-beta.5

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI-compatible vendor presets and adapter registry extracted into a dedicated crate.

### Changed

- Version and dependency alignment with the split workspace layout.
