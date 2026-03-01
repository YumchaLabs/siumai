# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-core-v0.11.0-beta.5...siumai-core-v0.11.0-beta.6) - 2026-03-01

### Added

- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- *(core)* relax stream Sync bounds
- *(streaming)* propagate inner cancel on drop

### Other

- *(spec)* remove reqwest conversion feature
- *(features)* include cohere and togetherai in all-providers
- *(features)* narrow all-providers aggregation
- *(spec)* make reqwest optional
- *(openai-compatible)* move vendor macro
- *(spec)* introduce siumai-spec crate
- *(openai)* document responses default and ws constraints
- *(streaming)* delegate cancellation to provider handles
- *(streaming)* improve cancellable streams
- *(changelog)* release 0.11.0-beta.5

## [0.11.0-beta.5] - 2026-01-15

### Added

- Provider-agnostic core extracted from the facade crate as part of the workspace split.
- Injectable HTTP transport (custom `fetch` parity), including streaming use-cases.
- Typed V3 stream parts and cross-protocol transcoding foundations used by gateway/proxy layers.

### Fixed

- Stricter SSE JSON parsing to reduce silent drift.
