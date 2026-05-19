# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-provider-minimaxi-v0.11.0-beta.7...siumai-provider-minimaxi-v0.11.0-beta.8) - 2026-05-18

### Other

- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- harden crate boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs

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
