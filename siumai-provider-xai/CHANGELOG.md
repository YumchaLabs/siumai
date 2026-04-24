# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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


