# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-protocol-gemini-v0.11.0-beta.5...siumai-protocol-gemini-v0.11.0-beta.6) - 2026-03-25

### Added

- unify gemini bridge support across vertex builds
- preserve gemini response bridge metadata
- add gemini request ingress bridge
- *(v4)* align provider parity and typed metadata surfaces
- *(core)* converge family-model runtime and traits
- *(providers)* wire interceptors and middlewares from config
- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- *(ci)* align provider fixtures and registry behavior
- *(protocol)* classify invalid mime as InvalidParameter

### Other

- add gemini streaming feature surface coverage
- add gemini family stream bridge fixtures
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
- *(changelog)* release 0.11.0-beta.5

## [0.11.0-beta.5] - 2026-01-15

### Added

- Gemini / GenerateContent protocol mapping split out into a dedicated crate.
- GenerateContent SSE stream serialization helpers.

### Fixed

- Vercel-aligned tool call parsing/serialization and official endpoint shaping.
