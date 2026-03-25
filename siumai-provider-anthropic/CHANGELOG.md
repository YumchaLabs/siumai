# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-provider-anthropic-v0.11.0-beta.5...siumai-provider-anthropic-v0.11.0-beta.6) - 2026-03-25

### Added

- *(v4)* align provider parity and typed metadata surfaces
- *(anthropic)* align builder helpers with config-first params
- *(providers)* align anthropic gemini and vertex packages
- *(providers)* wire interceptors and middlewares from config
- *(provider)* config-first constructors
- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- resolve workspace clippy warnings
- *(provider)* unify chat request normalization and base url defaults
- *(providers)* classify invalid mime as InvalidParameter

### Other

- add bridge customization example and coverage
- unify anthropic request overlays and bridge fixtures
- *(refactor)* finalize fearless refactor policies and release gate
- clarify builder as compat
- *(providers)* assert llmclient capability wiring
- *(examples)* migrate to family APIs
- *(examples)* config-first openai-compatible vendors
- align migration and websocket examples
- beta.6 recommended usage
- add beta.6 migration guide
- *(release)* bump to 0.11.0-beta.6
- align docs with family APIs
- *(examples)* switch to family APIs
- *(client)* decouple LlmClient from ChatCapability
- *(openai)* document responses default and ws constraints
- *(changelog)* release 0.11.0-beta.5

## [0.11.0-beta.5] - 2026-01-15

### Added

- Native Anthropic provider extracted into its own crate as part of the workspace split.
- Files API helper and beta wiring, plus container metadata support.
- Message batches and `count_tokens` helpers.
- Computer-use provider tools (and related fixture alignment).

### Fixed

- Vercel-aligned JSON tool streaming behavior.
