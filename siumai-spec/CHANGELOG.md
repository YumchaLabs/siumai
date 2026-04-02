# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add the Vercel-aligned `Warning::Compatibility` variant and helper constructor.
- Add the AI SDK-style `Warning::Unsupported { feature, details }` variant and normalize helper constructors to emit that shared shape while keeping legacy unsupported warning variants for compatibility.
- `Usage` now makes AI SDK-style `inputTokens` / `outputTokens` / `raw` the canonical stable layer, while legacy `prompt/completion/total` counts remain available through compatibility accessors/serde plus normalized helper APIs for migration-safe provider/protocol code.
- Extend unified `source` parts with optional `mediaType`, `filename`, and `providerMetadata` fields for document-style sources.
- Refactor unified `source` parts into a stricter AI SDK-style URL/document union via `SourcePart`, while preserving `sourceType`-based wire serialization and compatibility decoding.
- Extend unified `tool-approval-request` / `tool-approval-response` parts with request `providerMetadata` and response `reason`.
- Add first-class `providerOptions` to `ChatMessage`, request-capable `ContentPart` variants, and tool-result output/content shapes, including builder/helper APIs for stable mutation.
- Add first-class V4 `custom` / `reasoning-file` content parts plus explicit tool-result content variants (`file-data`, `file-url`, `file-id`, `image-data`, `image-url`, `image-file-id`) and a stable provider-keyed `ToolResultFileId`.
- Add first-class runtime `ChatStreamPart` semantics and `ChatStreamEvent::Part`, so the stable streaming surface can carry AI SDK V4 stream-part concepts such as `source`, `response-metadata`, `stream-start warnings`, `finish`, `custom`, `file`, and `reasoning-file`.
- Add runtime-only `ChatStreamReplay` plus `ChatStreamEvent::PartWithReplay`, so protocol serializers can carry same-protocol replay hints such as OpenAI Responses `rawItem` / `outputIndex` without widening `ChatStreamPart` or overloading generic `providerMetadata`.

### Fixed

- `Usage` now tracks whether legacy totals are actually known, so builders, merges, serde, and normalized helpers stop materializing unknown compatibility totals as `0` when only partial AI SDK-style usage data is available.
- `Usage` no longer exposes legacy `prompt/completion/total` counts as public storage fields, so new code must use builders/constructors and compatibility accessors instead of struct literals or direct field reads.
- `ProviderOptionsMap` serde now normalizes provider ids during JSON decode and re-emits the canonical `openaiCompatible` wire key during encode, so JSON request fixtures and builder-authored requests share the same lookup behavior.
- `ResponseMetadata` now serializes the stable AI SDK field names `modelId` / `timestamp` while continuing to accept legacy `model` / `created` aliases during decode.

## [0.11.0-beta.6] - 2026-03-02

### Added

- Support runtime-only per-request HTTP overrides (headers + timeout) used by the facade family call options.

