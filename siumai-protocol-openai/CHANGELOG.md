# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- OpenAI Responses and OpenAI-compatible usage parsing/serialization now converge on the shared AI SDK-style `inputTokens` / `outputTokens` / `raw` model, preserving provider-native `raw` usage plus `input_tokens_details.cached_tokens` and `output_tokens_details.reasoning_tokens` during JSON and SSE replay.
- OpenAI Responses request conversion now forwards `tool-approval-response.reason` on MCP approval items instead of dropping it.
- OpenAI Responses request conversion now uses stable message/part/tool-result `providerOptions` as the canonical request-time lane for item ids, reasoning payloads, and image detail; assistant tool-call ids no longer read request-side `providerMetadata.openai`.
- OpenAI Responses request conversion now matches the stricter canonical provider boundary more closely: reasoning and compaction items no longer read request-side `provider_metadata`, encrypted reasoning without `itemId` is forwarded as a first-class reasoning item, tool-result approval-id skipping reads only output `providerOptions.openai`, image detail reads only part/tool-result `providerOptions`, and assistant tool-call ids now also stay on canonical `providerOptions`.
- OpenAI Responses request normalization now writes request-side `itemId`, `reasoningEncryptedContent`, and `imageDetail` back into canonical `providerOptions.openai` slots instead of response-style `provider_metadata.openai`, and `input_image` normalization now restores AI-SDK-shaped user image file parts rather than collapsing them into the older image-only shape.
- OpenAI Responses tool-result conversion now preserves explicit `file-data` / `file-url` / `file-id` / `image-data` / `image-url` / `image-file-id` shapes instead of collapsing them through a coarse image/file union.
- OpenAI Responses request fixtures now lock native `file_id` roundtrips for tool-result `image-file-id` / `file-id`, and provider-keyed `ToolResultFileId` inputs now have regression coverage proving OpenAI-native ids win when projecting to Responses input items.
- OpenAI Responses fixture baselines now match the stable canonical model instead of older compatibility shapes: tool-result attachments no longer use the removed generic `file` variant, unsupported settings are asserted via `unsupported { feature }`, and exact response roundtrips now pin `Usage.inputTokens` / `Usage.outputTokens` / `Usage.raw`.
- OpenAI Responses SSE serialization now accepts the new runtime `ChatStreamEvent::Part(ChatStreamPart)` semantic channel directly by routing it through the typed stream-part compatibility bridge instead of requiring provider-scoped custom events first, and it now normalizes those parts before locking serializer state so direct part replay no longer self-deadlocks.
- OpenAI Responses SSE parsing now emits first-class runtime stream parts for `stream-start`, `response-metadata`, non-tool `text-*`, `reasoning-*`, `source`, successful `finish`, and provider-hosted tool / MCP / approval semantics, and document sources can now reserialize back to Responses annotations via `providerMetadata.openai.fileId` even when the stable source shape no longer carries a top-level document URL.
- OpenAI Responses same-protocol replay of provider-hosted tool / MCP / approval items now uses a dedicated runtime replay carrier for `rawItem` / `outputIndex`, so parser output, bridge output, and SSE serialization no longer depend on loose provider-scoped custom JSON extras for those hints.
- OpenAI Responses failed/unknown finish replay now preserves `null` usage totals end-to-end instead of materializing zero counts inside buffered terminal events.

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI(-like) protocol mapping split out into a dedicated crate.
- OpenAI Responses SSE stream serialization helpers (gateway/proxy use-cases).
- Protocol-level JSON response encoders for transcoding.

### Fixed

- Vercel-aligned parsing/serialization for Responses API stream parts and fixtures.
