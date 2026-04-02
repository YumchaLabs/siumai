# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Anthropic streaming now preserves extended usage fields such as `cache_creation_input_tokens`, `cache_read_input_tokens`, `server_tool_use`, and `service_tier` in terminal responses and SSE re-serialization.
- Typed Anthropic metadata extraction now derives nested `usage` fields and aliases consistently, so provider metadata round-trips keep raw usage fidelity.
- Anthropic Messages JSON replay now derives `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, and `output_tokens` from the normalized AI SDK-style usage model when raw provider usage is missing or partial.
- Anthropic Messages response parsing now keeps `Usage.raw` to the stable AI SDK-style raw subset while preserving the full provider `usage` payload under `provider_metadata.anthropic.usage`, and absent optional raw fields are no longer serialized as `null`.
- Anthropic request conversion now degrades document-style unified `source` parts into text placeholders without assuming URL-only fields, preserving available title / filename / media type context.
- Anthropic Messages request conversion now prefers stable message/part `providerOptions` for cache control and document citation metadata, with legacy `metadata.custom` / `provider_metadata` paths retained only as compatibility fallbacks.
- Anthropic document citations/title/context and per-part cache control are now sourced only from canonical request-side `providerOptions.anthropic`; legacy `message.metadata.custom["anthropic_document_*"]`, `message.metadata.custom["anthropic_content_cache_*"]`, and request-side file `provider_metadata` no longer affect request conversion.
- Anthropic non-stream response parsing and same-protocol replay now keep `thinking` / `redacted_thinking` on stable reasoning parts instead of message-level shims: responses surface part-level `providerMetadata.anthropic.signature` / `redactedData`, prompt replay reads part-level Anthropic replay metadata, and Anthropic JSON replay no longer depends on `metadata.custom["anthropic_*"]` thinking keys.
- Anthropic tool-result conversion and JSON replay now understand explicit Vercel-style `image-data`, `image-url`, `file-data` (PDF), and `file-url` content parts, while degrading unsupported file-id variants explicitly.
- Anthropic SSE serialization now accepts the new runtime `ChatStreamEvent::Part(ChatStreamPart)` semantic channel by bridging it through the typed stream-part compatibility path instead of assuming only legacy deltas or custom payloads, and it now normalizes those parts before locking serializer state so runtime-part replay cannot self-deadlock.
- Anthropic SSE serialization now also honors `V3UnsupportedPartBehavior::AsText` on direct `ChatStreamEvent::Part/PartWithReplay` inputs, so unsupported stable parts such as `tool-approval-request` no longer disappear when the source stream is already on the typed runtime-part lane.
- Anthropic Messages SSE parsing now emits runtime AI SDK-style parts directly for `stream-start`, `response-metadata`, `text-*`, standard local `tool-input-*` / `tool-call`, `reasoning-*`, `source`, and successful `finish` semantics, and Anthropic SSE re-serialization now replays `tool-input-*` runtime parts without dropping block closures.
- Anthropic Messages SSE parsing now maps provider-hosted server tools and MCP tool use/results onto stable `tool-input-*` / `tool-call` / `tool-result` parts, and SSE replay now rebuilds provider-hosted Anthropic content blocks from that stable lane instead of depending on provider-scoped custom events.
- Anthropic Messages SSE `signature_delta` now maps to stable `reasoning-delta` parts with `providerMetadata.anthropic.signature`, `redacted_thinking` replay now follows `reasoning-start.providerMetadata.anthropic.redactedData`, and same-protocol replay no longer depends on the legacy `anthropic:thinking-signature-delta` custom event.
- Anthropic SSE message-delta replay now only injects `input_tokens` / `output_tokens` when the stable usage totals are actually known, so unknown finish usage no longer collapses to zero during stream reserialization.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Anthropic protocol mapping split out into a dedicated crate.
- Shared streaming/transcoding helpers (used by gateway/proxy layers).

### Fixed

- Vercel-aligned JSON tool streaming and `responseFormat` behavior.
