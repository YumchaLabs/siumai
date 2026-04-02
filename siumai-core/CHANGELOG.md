# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- OpenAI-compatible chat/stream response decoding now normalizes usage through the shared AI SDK-style `inputTokens` / `outputTokens` / `raw` model instead of only preserving legacy totals and partial detail fields.
- The historical `LanguageModelV3StreamPart` typed overlay is now a V4-capable superset with first-class `custom` and `reasoning-file` parts, and OpenAI-compatible `AsText` fallback can now degrade those parts into explicit text instead of silently dropping them.
- The upgraded typed stream-part overlay now also exposes public `LanguageModelV4*` aliases so new code can use AI SDK-aligned names without losing compatibility with the historical `LanguageModelV3*` surface.
- `LanguageModelV3StreamPart` can now convert to and from the new spec-level `ChatStreamEvent::Part(ChatStreamPart)` runtime semantic channel, and `EventBuilder` exposes a first-class `add_part(...)` helper.
- `EventBuilder` now also exposes `add_part_with_replay(...)`, and the shared stream processor / encoder helpers treat runtime replay-bearing part events the same as ordinary semantic part events.
- Anthropic `reasoning-*` typed custom-event conversion now preserves stable `id` and `providerMetadata`, so protocol serializers can replay AI SDK-style reasoning signatures and redacted-thinking metadata from the semantic part surface instead of dropping to ad hoc custom payloads.
- `StreamProcessor` now preserves terminal response envelope fields from `StreamEnd`, including `id`, `model`, `audio`, `system_fingerprint`, `service_tier`, and `warnings`.
- Final stream aggregation now falls back to `StreamStart` metadata when no terminal response is available and retains terminal multimodal parts that were not rebuilt from deltas, such as sources and provider-decorated tool calls.
- `StreamProcessor` now consumes structured runtime stream parts directly, preserving streamed warnings, finish reasons, provider metadata, sources, custom content, generated files, reasoning files, tool approval requests, and completed tool result parts instead of dropping them at the transport boundary.
- OpenAI Responses stream bridging now rebuilds provider-hosted tool / MCP replay as stable part events with a runtime replay carrier instead of synthesizing provider-scoped custom payloads with embedded `rawItem` JSON.
- OpenAI-compatible stream decoding now keeps observed terminal chunk fields such as `system_fingerprint` and `service_tier` on `StreamEnd`, including the EOF fallback path when no explicit finish chunk is emitted.
- OpenAI-compatible streaming now matches AI SDK model-router metadata timing more closely: placeholder Azure `prompt_filter_results` preludes with empty `id` / `model` and `created = 0` no longer synthesize early `response-metadata` or `1970-01-01` timestamps before the first real metadata chunk arrives.
- OpenAI-compatible SSE serialization now reuses the shared OpenAI chat-usage writer, preserving usage detail fields and provider-unknown totals instead of flattening replayed `usage` chunks to synthetic zero-valued legacy integers.
- OpenAI-compatible replay of typed V3 `finish` parts now preserves unknown usage totals as `null` when no stable prompt/completion totals are available.
- OpenAI-compatible chat/non-stream response decoding now maps `message.annotations` / `delta.annotations` URL citations into stable `source` parts, and URL `source` stream parts now serialize back into chat-completions `annotations`.
- `systemMessageMode=remove` warnings now use the explicit `compatibility` warning shape instead of a generic warning string.
- OpenAI-compatible and OpenAI Chat request conversion now read canonical message/part `providerOptions` for request-only behavior such as extra params, assistant reasoning replay hints, and image detail; request-side `provider_metadata` / `message.metadata.custom` are no longer treated as input on those main paths.

## [0.11.0-beta.6] - 2026-03-02

### Added

- Apply per-request `HttpConfig` overrides (headers + timeout) at the HTTP executor layer, including streaming requests.
- Add convenience helpers on `dyn LlmClient` for full chat requests (`chat_request`, `chat_stream_request`).

## [0.11.0-beta.5] - 2026-01-15

### Added

- Provider-agnostic core extracted from the facade crate as part of the workspace split.
- Injectable HTTP transport (custom `fetch` parity), including streaming use-cases.
- Typed V3 stream parts and cross-protocol transcoding foundations used by gateway/proxy layers.

### Fixed

- Stricter SSE JSON parsing to reduce silent drift.
