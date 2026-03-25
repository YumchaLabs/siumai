# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-protocol-anthropic-v0.11.0-beta.5...siumai-protocol-anthropic-v0.11.0-beta.6) - 2026-03-25

### Added

- *(stream)* enforce anthropic block ordering
- *(bridge)* preserve native response reasoning metadata
- *(v4)* align provider parity and typed metadata surfaces
- *(core)* converge family-model runtime and traits
- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- *(build)* gate structured JSON response branches by feature
- *(streaming)* avoid duplicate anthropic terminal frames in openai responses transcoding
- *(ci)* align provider fixtures and registry behavior
- dedupe anthropic transcoding for provider tools
- preserve anthropic provider tool result fidelity
- preserve anthropic mcp server name metadata
- preserve anthropic tool search roundtrip fidelity
- preserve anthropic citation fidelity in response bridge
- *(streaming)* align anthropic sse and openai finish chunks

### Other

- add anthropic streaming feature surface coverage
- unify anthropic request overlays and bridge fixtures
- normalize anthropic web fetch response shape
- unify protocol stream serializer helpers
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

- Anthropic protocol mapping split out into a dedicated crate.
- Shared streaming/transcoding helpers (used by gateway/proxy layers).

### Fixed

- Vercel-aligned JSON tool streaming and `responseFormat` behavior.
