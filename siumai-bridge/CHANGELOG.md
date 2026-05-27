# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.9](https://github.com/YumchaLabs/siumai/compare/siumai-bridge-v0.11.0-beta.8...siumai-bridge-v0.11.0-beta.9) - 2026-05-27

### Fixed

- harden feature matrix clippy
- clean ci all-features failures

### Other

- polish release changelog
- split bridge request codecs
- update changelogs for architecture boundary refactor
- harden clean architecture boundaries
- Merge branch 'main' of https://github.com/YumchaLabs/siumai
- lock bridge response ownership boundary
- deepen provider and bridge module boundaries

### Changed

- Bridge request normalization keeps the existing public helper functions but now preserves OpenAI Responses, OpenAI Chat Completions, Anthropic Messages, and Gemini GenerateContent request behavior through dedicated wire-format handlers.
- Gemini GenerateContent request normalization delegates to the protocol-owned Gemini adapter while bridge reporting, loss policy, hooks, lifecycle, customization, and target dispatch remain in `siumai-bridge`.

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/releases/tag/siumai-bridge-v0.11.0-beta.8) - 2026-05-18

### Other

- lock bridge stable stream parts
- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- converge fearless architecture boundaries
