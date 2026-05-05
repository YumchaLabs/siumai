# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-provider-minimaxi-v0.11.0-beta.6...siumai-provider-minimaxi-v0.11.0-beta.7) - 2026-05-05

### Added

- *(ai-sdk)* align shared structural surfaces and builder helpers
- *(files)* align upload helper contract with ai sdk
- *(audio)* preserve request metadata on audio results
- *(audio)* align speech and transcription helper results
- *(media)* align helper empty-result semantics
- *(video)* align task-based video model surface
- refactor
- *(streaming)* align gemini stable parts and extras consumers
- *(media)* align provider-owned image and video surfaces with ai sdk

### Fixed

- *(ci)* align response fixtures and clippy checks
- *(minimaxi)* handle json object response format

### Other

- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events
- *(release)* prepare v0.11.0-beta.7
- *(minimaxi)* move video vendor fields behind provider options

### Added

- MiniMaxi now exposes a provider-owned curated model surface for the public families (`chat`,
  `speech`, `video`, `music`, `image`) so facade/catalog/default-model consumers can reuse one
  source instead of handwritten arrays.

### Fixed

- Stream metadata normalization now also reaches typed finish parts, so MiniMaxi streaming no
  longer leaks `providerMetadata["anthropic"]` on the stable finish-part lane.

## [0.11.0-beta.5] - 2026-01-15

### Added

- MiniMax provider extracted into its own crate as part of the workspace split.

### Changed

- Dependency graph decoupled to reduce cross-provider coupling.
