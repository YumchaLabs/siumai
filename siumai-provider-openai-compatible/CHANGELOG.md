# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- OpenAI-compatible content-part metadata helpers now cover stable `reasoning-file` and `custom`
  parts instead of assuming the older content-part subset only.
- OpenAI-compatible public provider surfaces now expose an AI SDK-style response metadata
  extractor hook: `ResponseMetadataExtractor`, `OpenAiCompatibleConfig::with_metadata_extractor`,
  and `OpenAiCompatibleBuilder::with_metadata_extractor` can extend built-in provider metadata
  extraction without requiring a custom `ProviderAdapter`.
- OpenAI-compatible public config/builder surfaces now also expose AI SDK-style request settings:
  `with_include_usage(...)` controls whether compat chat streams send
  `stream_options.include_usage`, default compat requests now omit that field until explicitly
  enabled, and `RequestBodyTransformer` / `with_request_body_transformer(...)` mirror AI SDK
  `transformRequestBody` on the final normalized chat payload.

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI-compatible vendor presets and adapter registry extracted into a dedicated crate.

### Changed

- Version and dependency alignment with the split workspace layout.
