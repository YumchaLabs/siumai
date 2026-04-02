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
- OpenAI-compatible public config/builder/runtime surfaces now also expose AI SDK-style
  `queryParams` and provider-level `supportsStructuredOutputs` concepts: compat route generation
  now appends deterministic provider query params across chat / embeddings / image / audio /
  rerank / model-listing paths, compat chat now defaults to downgrading JSON Schema outputs to
  `response_format = { "type": "json_object" }` while emitting a stable
  `unsupported { feature: "responseFormat" }` warning middleware on the chat response path, and
  callers can opt back into wire-level `json_schema` by setting
  `supports_structured_outputs(true)`.
- OpenAI-compatible chat runtime shaping now also honors AI SDK-style known compat request options
  from canonical `providerOptions.openaiCompatible` and provider-owned keys: `user`,
  `reasoningEffort`, `textVerbosity`, and `strictJsonSchema` now map to wire `user`,
  `reasoning_effort`, `verbosity`, and `response_format.json_schema.strict` instead of leaking as
  raw camelCase request fields.
- OpenAI-compatible chat runtime now also installs AI SDK-style provider-defined tool warnings on
  the default response path: provider-defined tools remain filtered out of Chat Completions
  requests, and successful chat responses now emit `unsupported { feature: "provider-defined tool
  <id>" }` warnings without extra user-installed middleware.
- OpenAI-compatible chat runtime now also installs the AI SDK deprecation warning for legacy
  `providerOptions['openai-compatible']`: the deprecated key still works for audited compat chat
  options, and successful chat responses now emit `other { message: "The 'openai-compatible' key
  in providerOptions is deprecated. Use 'openaiCompatible' instead." }` on the default response
  path.

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI-compatible vendor presets and adapter registry extracted into a dedicated crate.

### Changed

- Version and dependency alignment with the split workspace layout.
