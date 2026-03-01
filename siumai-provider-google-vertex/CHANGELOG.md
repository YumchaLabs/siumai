# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.6](https://github.com/YumchaLabs/siumai/compare/siumai-provider-google-vertex-v0.11.0-beta.5...siumai-provider-google-vertex-v0.11.0-beta.6) - 2026-03-01

### Added

- *(openai)* ws connection aging
- *(openai)* remote cancel for websocket session
- *(openai)* add WebSocket mode for responses streaming

### Fixed

- *(google-vertex)* wire http_transport into image executor

### Other

- *(openai)* document responses default and ws constraints
- *(changelog)* release 0.11.0-beta.5

## [0.11.0-beta.5] - 2026-01-15

### Added

- Dedicated Google Vertex AI provider crate (split from the previous Gemini/Google layer).
- Builder aliases and vercel-aligned tool exposure for Vertex setups.
- ADC auth auto-enable behavior (Vercel parity) and embedding batch size guards.

### Fixed

- Imagen API-key mode now appends `?key=...` to endpoint URLs.
- Anthropic-on-Vertex request shaping aligned with Vercel behavior.
