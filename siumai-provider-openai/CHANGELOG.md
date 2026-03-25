# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-provider-openai-v0.11.0-beta.5...siumai-provider-openai-v0.11.0-beta.6) - 2026-03-25

### Added

- *(v4)* align provider parity and typed metadata surfaces
- *(openai)* align builder aliases with config-first naming
- *(providers)* align openai azure and compat packages
- *(providers)* wire interceptors and middlewares from config
- *(providers)* expand config-first construction
- *(provider)* config-first constructors
- *(openai)* add WS done marker toggle
- *(openai-websocket)* expose unified builder session APIs
- *(openai-websocket)* add incremental session helpers
- *(openai)* ws connection aging
- *(openai)* remote cancel on stream cancellation
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- *(provider)* unify chat request normalization and base url defaults
- correct openai audio model fallback defaults
- *(openai)* omit chat-completions stream_options on responses stream
- *(embedding)* preserve request config on family model paths
- *(openai)* backfill direct chat request defaults
- *(openai)* keep responses defaults out of non-chat requests
- *(clippy)* satisfy -D warnings
- *(openai)* retry websocket on connection limit

### Other

- *(refactor)* finalize fearless refactor policies and release gate
- clarify builder as compat
- *(openai)* make new_with_config honor config http and wiring
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
- *(streaming)* improve cancellable streams
- *(changelog)* release 0.11.0-beta.5

## [0.11.0-beta.5] - 2026-01-15

### Added

- Native OpenAI provider extracted into its own crate as part of the workspace split.
- Vercel-aligned Responses API request/response shaping and fixtures.

### Fixed

- Preserve transcription text on SSE EOF.
- Default moderation model selection when no model is provided.
- Parsing/validation improvements for Files API and Responses stream events.
