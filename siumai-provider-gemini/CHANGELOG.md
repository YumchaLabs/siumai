# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
