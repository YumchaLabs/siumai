# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-provider-xai-v0.11.0-beta.6...siumai-provider-xai-v0.11.0-beta.7) - 2026-05-05

### Added

- align provider settings package surfaces
- *(video)* align result materialization with ai sdk
- *(files)* align upload helper contract with ai sdk
- *(audio)* preserve request metadata on audio results
- *(audio)* align speech and transcription helper results
- *(media)* align helper empty-result semantics
- *(video)* align task-based video model surface
- refactor
- *(transcription)* require media type for audio inputs
- *(alignment)* align image and media input contracts
- *(xai)* align responses response and stream semantics
- *(media)* align provider-owned image and video surfaces with ai sdk

### Fixed

- align provider streaming bridges
- *(ci)* align response fixtures and clippy checks
- *(ci)* satisfy clippy feature matrix
- *(rerank)* preserve raw response envelopes
- *(xai)* align image and video metadata
- *(files)* align provider metadata
- *(xai)* surface responses cost metadata
- *(xai)* request streaming usage by default
- *(xai)* align reasoning defaults with AI SDK
- *(video)* honor provider polling options
- align file upload provider option defaults
- *(xai)* align responses request filtering with ai sdk

### Other

- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events
- *(release)* prepare v0.11.0-beta.7
﻿
### Added

- Added AI SDK-style `XaiProviderSettings` and `VERSION` exports for the audited package-level
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset.

### Changed

- Aligned xAI Responses response and SSE semantics with the audited AI SDK boundary:
  reasoning metadata now stays under `providerMetadata.xai`, parsed responses no longer emit
  top-level provider metadata for plain text paths, and `file_search` tool response/stream outputs
  now normalize to the stable snake_case tool name plus camelCase result fields.
- xAI's compat-backed chat config now enables structured outputs by default, preserving stable
  JSON Schema response formats on the wire instead of falling back to `json_object`.
- xAI native image generation/edit now consume the canonical shared top-level `aspectRatio` field
  before provider-owned options/legacy extras, and unsupported image warnings now more closely
  match the audited AI SDK shape for shared `size` / `seed`.
- The provider-owned typed option surface now also exposes AI SDK-style alias names:
  `XaiLanguageModelChatOptions`, `XaiLanguageModelResponsesOptions`,
  `XaiImageModelOptions`, and `XaiVideoModelOptions`, plus the audited deprecated provider alias
  names kept by upstream for migration parity.

## [0.11.0-beta.5] - 2026-01-15

### Added

- xAI provider extracted into its own crate as part of the workspace split.

### Changed

- Fixture parity and Responses stream mapping aligned with Vercel AI SDK.


