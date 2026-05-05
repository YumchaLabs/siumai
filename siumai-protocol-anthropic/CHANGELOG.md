# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-protocol-anthropic-v0.11.0-beta.6...siumai-protocol-anthropic-v0.11.0-beta.7) - 2026-05-05

### Added

- align provider settings package surfaces
- *(anthropic)* align metadata and request option surfaces
- *(ai-sdk)* continue package surface parity alignment
- *(types)* align shared ai sdk type surface
- refactor
- *(anthropic)* align messages replay and streaming surfaces

### Fixed

- align provider streaming bridges
- *(ci)* align response fixtures and clippy checks
- *(ci)* satisfy clippy feature matrix
- *(text)* retain provider request metadata
- *(completion)* preserve response bodies
- *(chat)* preserve provider response bodies
- *(text)* preserve raw response envelopes
- *(anthropic)* prefer raw usage replay
- *(anthropic)* sum usage iterations
- *(anthropic)* replay raw stop reasons in bridges
- *(anthropic)* preserve raw finish reasons
- *(anthropic)* preserve extended usage roundtrips
- *(streaming)* preserve anthropic provider tool replay

### Other

- add beta 7 migration guidance
- prepare beta release notes
- rename typed stream overlay
- update stream examples for typed events
- remove legacy chat stream events
- stop emitting legacy stream deltas
- make anthropic stream serializer typed-only
- emit typed anthropic stream parts

### Fixed

- Anthropic header construction now supports the audited alternate-auth path: when callers provide
  an `Authorization` header and no API key, the protocol header builder no longer emits an empty
  `x-api-key`.
- Anthropic native structured output now follows the current AI SDK request contract more closely:
  native JSON Schema output lowers to `output_config.format` instead of deprecated
  `output_format`, request-option overlays merge `output_config.format`, `output_config.effort`,
  and `output_config.task_budget` onto one shared object instead of overwriting sibling fields,
  request normalization/bridge restoration now prefer `output_config.format` while remaining
  backward-compatible with legacy `output_format`, and stream-side structured-output mode
  selection now stays consistent with the final request body even when tools are present.
- Anthropic request-option normalization/body finalization now also preserve the remaining audited
  provider option fields more faithfully: `inferenceGeo` lowers to native `inference_geo`, and
  adaptive thinking `display` survives the normalize -> overlay -> finalize path instead of being
  dropped when Anthropic thinking is rebuilt.
- Anthropic container-skill request shaping now follows the current AI SDK public contract more
  closely: request overlays lower custom `providerReference.anthropic` values onto native
  `container.skills[].skill_id`, while same-protocol request normalization restores custom skills
  back onto `providerReference` instead of flattening every skill to `skillId`.
- Anthropic request and cache conversion now accept prompt-side provider-owned user image/document
  references, mapping them onto native `source: { type: "file", file_id }` blocks and resolving
  canonical `ProviderReference` entries only from the Anthropic provider namespace.
- Anthropic request finalization now auto-injects the `files-api-2025-04-14` beta token whenever
  prompt-side provider-owned file references require the Files API request path.
- Anthropic request-option normalization/body overlays now cover the newer AI SDK Anthropic
  request keys more completely (`thinking`, `sendReasoning`, `disableParallelToolUse`,
  `cacheControl`, `metadata.userId`, `mcpServers`, `contextManagement`, `speed`,
  `anthropicBeta`), and plain `anthropic-standard` builds no longer reference
  `MessageContent::Json` unless `structured-messages` is enabled.
- Anthropic request-body finalization now applies enabled-thinking token semantics consistently:
  final request bodies add `budget_tokens` onto `max_tokens` before known-model capping even when
  thinking came from legacy provider-specific params rather than only from provider options.
- Anthropic streaming now preserves extended usage fields such as `cache_creation_input_tokens`, `cache_read_input_tokens`, `server_tool_use`, and `service_tier` in terminal responses and SSE re-serialization, and terminal `Usage.raw` now keeps the full provider-native Anthropic usage object instead of a trimmed subset.
- Typed Anthropic metadata now distinguishes the narrow AI SDK-style `AnthropicMessageMetadata`
  shape from the wider Rust `AnthropicMetadata` helper, and `AnthropicChatResponseExt` exposes
  both `anthropic_message_metadata*()` and `anthropic_metadata*()` accessors accordingly.
- The narrow `AnthropicMessageMetadata.container` surface now also uses dedicated required-field
  message container/skill structs, while the wider `AnthropicMetadata` helper keeps tolerating
  partial container data for compatibility and intermediate replay paths.
- Typed Anthropic metadata extraction now derives nested `usage` fields and aliases consistently, so provider metadata round-trips keep raw usage fidelity.
- Anthropic Messages JSON replay now derives `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, and `output_tokens` from the normalized AI SDK-style usage model when raw provider usage is missing or partial.
- Anthropic Messages response parsing now keeps the full provider-native `usage` payload on both `Usage.raw` and `provider_metadata.anthropic.usage`, so nested/forward-compatible Anthropic usage fields survive AI SDK-style raw snapshots instead of being trimmed to a local subset.
- Anthropic request conversion now degrades document-style unified `source` parts into text placeholders without assuming URL-only fields, preserving available title / filename / media type context.
- Anthropic Messages request conversion now prefers stable message/part `providerOptions` for cache control and document citation metadata, with legacy `metadata.custom` / `provider_metadata` paths retained only as compatibility fallbacks.
- Anthropic document citations/title/context and per-part cache control are now sourced only from canonical request-side `providerOptions.anthropic`; legacy `message.metadata.custom["anthropic_document_*"]`, `message.metadata.custom["anthropic_content_cache_*"]`, and request-side file `provider_metadata` no longer affect request conversion.
- Anthropic non-stream response parsing and same-protocol replay now keep `thinking` / `redacted_thinking` on stable reasoning parts instead of message-level shims: responses surface part-level `providerMetadata.anthropic.signature` / `redactedData`, prompt replay reads part-level Anthropic replay metadata, and Anthropic JSON replay no longer depends on `metadata.custom["anthropic_*"]` thinking keys.
- Anthropic custom provider ids now match the audited AI SDK contract more closely: request
  shaping merges canonical `providerOptions.anthropic` with provider-owned custom keys, custom
  keys override canonical fields when both are present, typed Anthropic metadata accessors now
  support custom roots, and top-level non-stream / finish / stream-end `providerMetadata`
  duplicates onto the custom provider root only when that custom request key was actually used.
- Anthropic response conversion now also matches the audited AI SDK source/message-metadata shape
  more closely: non-stream text citations and `web_search_tool_result` blocks emit stable
  `source` parts, request-scoped citation documents now feed the non-stream response transformer
  for document citation `title` / `filename` / `mediaType` resolution, source ids now use stable
  `id-*` generation across stream/non-stream citation and web-search sources, and absent
  `container` / `contextManagement` metadata now remains visible as explicit `null` on
  non-stream and final stream-end Anthropic provider metadata.
- Anthropic tool-result conversion and JSON replay now understand explicit Vercel-style `image-data`, `image-url`, `file-data` (PDF), and `file-url` content parts, while degrading unsupported file-id variants explicitly.
- Anthropic SSE serialization now accepts the new runtime `ChatStreamEvent::Part(ChatStreamPart)` semantic channel by bridging it through the typed stream-part compatibility path instead of assuming only legacy deltas or custom payloads, and it now normalizes those parts before locking serializer state so runtime-part replay cannot self-deadlock.
- Anthropic SSE serialization now also honors `UnsupportedStreamPartBehavior::AsText` on direct `ChatStreamEvent::Part/PartWithReplay` inputs, so unsupported stable parts such as `tool-approval-request` no longer disappear when the source stream is already on the typed runtime-part lane.
- Anthropic Messages SSE parsing now emits runtime AI SDK-style parts directly for `stream-start`, `response-metadata`, `text-*`, standard local `tool-input-*` / `tool-call`, `reasoning-*`, `source`, and successful `finish` semantics, and Anthropic SSE re-serialization now replays `tool-input-*` runtime parts without dropping block closures.
- Anthropic Messages SSE parsing now maps provider-hosted server tools and MCP tool use/results onto stable `tool-input-*` / `tool-call` / `tool-result` parts, and SSE replay now rebuilds provider-hosted Anthropic content blocks from that stable lane instead of depending on provider-scoped custom events.
- Anthropic Messages SSE `signature_delta` now maps to stable `reasoning-delta` parts with `providerMetadata.anthropic.signature`, `redacted_thinking` replay now follows `reasoning-start.providerMetadata.anthropic.redactedData`, and same-protocol replay no longer depends on the legacy `anthropic:thinking-signature-delta` custom event.
- Anthropic SSE message-delta replay now only injects `input_tokens` / `output_tokens` when the stable usage totals are actually known, so unknown finish usage no longer collapses to zero during stream reserialization.
- Anthropic Messages SSE now also preserves AI SDK-style first-chunk parse-failure lifecycle
  ordering: when the first SSE payload cannot be parsed, the stream still emits `stream-start`
  first, then optional runtime `raw`, and only then surfaces the parse error or API error.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Anthropic protocol mapping split out into a dedicated crate.
- Shared streaming/transcoding helpers (used by gateway/proxy layers).

### Fixed

- Vercel-aligned JSON tool streaming and `responseFormat` behavior.
