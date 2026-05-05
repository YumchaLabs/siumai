# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-provider-gemini-v0.11.0-beta.6...siumai-provider-gemini-v0.11.0-beta.7) - 2026-05-05

### Added

- *(google)* align package settings and streaming surfaces
- *(video)* align result materialization with ai sdk
- *(files)* align upload helper contract with ai sdk
- *(provider)* align package surfaces across providers
- *(video)* align task-based video model surface
- refactor
- *(alignment)* align image and media input contracts
- *(streaming)* align gemini stable parts and extras consumers
- *(media)* align provider-owned image and video surfaces with ai sdk

### Fixed

- *(ci)* align response fixtures and clippy checks
- *(gemini)* align video provider metadata
- *(video)* honor provider polling options

### Other

- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events
- stop emitting legacy stream deltas
- *(release)* prepare v0.11.0-beta.7

### Fixed

- Gemini content-part metadata helpers now include stable `reasoning-file` and `custom` parts so
  the V4-capable content model does not regress typed metadata extraction.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Gemini provider extracted into its own crate as part of the workspace split.
- Expanded tool and streaming fixtures aligned with Vercel AI SDK.

### Fixed

- Tool result encoding alignment and Imagen default behavior parity.
