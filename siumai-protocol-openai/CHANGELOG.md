# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-protocol-openai-v0.11.0-beta.5...siumai-protocol-openai-v0.11.0-beta.6) - 2026-03-25

### Added

- preserve openai response approval items
- *(stream)* preserve finish reasons in responses sse
- *(bridge)* roundtrip openai response sources
- *(bridge)* roundtrip provider-executed response tools
- *(bridge)* preserve native response reasoning metadata
- *(v4)* align provider parity and typed metadata surfaces
- *(core)* converge family-model runtime and traits
- *(spec)* support per-request http config
- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- *(build)* gate structured JSON response branches by feature
- *(ci)* align provider fixtures and registry behavior
- resolve workspace clippy warnings
- preserve openai responses error stream replay
- tighten openai responses stream roundtrip fidelity
- map structured output to openai responses text format
- preserve openai responses provider tool source identity
- preserve openai responses web search sources
- *(openai)* omit chat-completions stream_options on responses stream
- *(protocol)* classify invalid mime as InvalidParameter
- *(protocol-openai)* map mime_str errors to LlmError

### Other

- centralize openai responses tool stream event builders
- extend openai response roundtrip fixtures
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

- OpenAI(-like) protocol mapping split out into a dedicated crate.
- OpenAI Responses SSE stream serialization helpers (gateway/proxy use-cases).
- Protocol-level JSON response encoders for transcoding.

### Fixed

- Vercel-aligned parsing/serialization for Responses API stream parts and fixtures.
