# AI SDK Structural Alignment - Audit

Last updated: 2026-03-30

This note records the current structural parity status against the AI SDK provider contracts.

Status legend:

- `Green`: aligned enough for the intended stable semantics
- `Amber`: partially aligned; important slots still missing or still provider-scoped
- `Red`: materially misaligned; should drive refactor work

## Current summary

The current branch has now closed four concrete structural gaps:

- Anthropic streaming preserves extended usage and provider metadata across decode/encode
  round-trips.
- Shared warnings now expose AI SDK-style `unsupported` / `compatibility` categories through a
  compatibility-superset model.
- Request-side `providerOptions` now exist on messages, request-capable content parts, and
  tool-result output/content shapes, and the main request converters prefer them over historical
  metadata shims.
- Stable content now includes first-class V4 `custom` and `reasoning-file` parts.
- Stable `source` now uses a strict URL/document union while keeping compatibility wire
  serialization.
- Stable tool-result content now models the explicit AI SDK V4 subtypes:
  - `text`
  - `file-data`
  - `file-url`
  - `file-id`
  - `image-data`
  - `image-url`
  - `image-file-id`
  - `custom`
- The typed stream-part overlay now includes V4 `custom` / `reasoning-file` and explicit lossy
  fallback coverage for unsupported protocol reserialization.
- The runtime stream transport now carries a first-class `ChatStreamEvent::Part(ChatStreamPart)`
  semantic channel that can represent AI SDK V4 stream-part semantics without routing them through
  `Custom`.
- `StreamProcessor` and the main OpenAI/OpenAI-compatible/Anthropic/Gemini stream serializers now
  bridge that new runtime part channel instead of assuming every richer stream semantic must first
  become a provider-scoped custom event.
- OpenAI Responses and Anthropic SSE serializers now normalize runtime `Part` events before taking
  serializer state locks, so stable-part replay no longer hangs on recursive lock re-entry.
- OpenAI Responses parsing now emits the runtime part channel directly for `stream-start`,
  `response-metadata`, non-tool `text-*`, `reasoning-*`, `source`, successful terminal `finish`,
  and provider-hosted tool / MCP / approval semantics; protocol-only `rawItem` / `outputIndex`
  fidelity now rides a separate runtime replay carrier instead of staying inside loose custom
  payloads.
- Anthropic parsing now emits the runtime part channel directly for `stream-start`,
  `response-metadata`, `text-*`, standard local `tool-input-*` / `tool-call`, provider-hosted
  server tool / MCP `tool-*`, `reasoning-*`, `source`, and successful terminal `finish`
  semantics. Anthropic reasoning `signature_delta` and `redacted_thinking` now also stay on that
  stable lane through `reasoning-*` part `providerMetadata`, matching AI SDK behavior instead of
  relying on provider-scoped custom events.
- Gemini parsing now emits the runtime part channel directly for `reasoning-*`, `source`,
  provider-executed `tool-call` / `tool-result`, and `emit_v3_tool_call_parts=true` function-call
  paths instead of tunneling those semantics through `gemini:*` custom events.
- The current OpenAI/OpenAI-compatible/Anthropic/Gemini usage serializers and parsers no longer
  synthesize zero-valued legacy totals when provider usage totals are unknown or explicitly `null`;
  Gemini usage replay also now keeps `cachedContentTokenCount` / `trafficType` during SSE
  round-trips and counts thinking tokens inside total output usage.
- OpenAI/OpenAI-compatible tool-message conversion has now been reviewed against the explicit V4
  tool-result content variants; because the wire contract is still string-only for tool messages,
  those variants intentionally degrade to JSON strings while preserving the inner typed content
  array shape.

That moves the main remaining risk away from the stable prompt/content surface and toward protocol
coverage, stream modeling, usage modeling, and final shape cleanup.

## Structural gap table

| Area | AI SDK reference | Current Siumai anchor | Status | Notes |
| --- | --- | --- | --- | --- |
| Shared warning semantics | `shared/v4/shared-v4-warning.ts` | `siumai-spec/src/types/common.rs` | Green | AI SDK-style `unsupported` / `compatibility` / `other` semantics now exist on the stable surface; legacy unsupported variants remain as compatibility inputs. |
| Shared source shape | `language-model-v4-source.ts` | `siumai-spec/src/types/chat/content/part.rs` | Green | Stable `SourcePart` now models a strict URL/document union while preserving `sourceType` wire compatibility. |
| Tool approval shape | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/part.rs` | Green | `tool-approval-response.reason`, `tool-approval-request.providerMetadata`, and prompt-side `providerOptions` are present. |
| Message-level provider options | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/message.rs` | Green | `ChatMessage` exposes first-class `providerOptions`, serde aliases, and helper APIs. |
| Part-level provider options | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/part.rs` | Green | Request-capable stable parts carry `providerOptions`, with helper accessors/builders. |
| Tool-result output/content provider options | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/tool_result.rs` | Green | `ToolResultOutput` and `ToolResultContentPart` now model AI SDK-style `providerOptions`. |
| Tool-result content granularity | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/tool_result.rs` | Green | Stable tool-result content now models the explicit V4 file/image/id variants plus `custom`, including `ToolResultFileId` for provider-keyed ids. |
| Request/response provider boundary | `shared-v4-provider-options.ts` + `shared-v4-provider-metadata.ts` | `siumai-core/src/standards/openai/utils.rs`, `siumai-protocol-anthropic/src/standards/anthropic/utils/content.rs` | Amber | OpenAI-compatible, OpenAI Responses, and Anthropic request paths now prefer `providerOptions`, but bounded compatibility fallbacks still read legacy metadata. |
| V4 custom content | `language-model-v4-custom-content.ts` | stable content + protocol converters | Amber | Stable `ContentPart::Custom` exists, but true provider support is intentionally scoped: OpenAI Responses `openai.compaction`, Anthropic tool-result `tool_reference`, Gemini skips unsupported custom parts. |
| V4 reasoning-file content | `language-model-v4-reasoning-file.ts` | stable content + protocol converters | Amber | Stable `ContentPart::ReasoningFile` exists and Gemini has true wire support, but OpenAI/Anthropic still degrade where no native request equivalent exists. |
| Stable usage shape | `language-model-v4-usage.ts` | `siumai-spec/src/types/usage.rs` | Green | `Usage` now carries AI SDK-style `inputTokens` / `outputTokens` / `raw` alongside compatibility totals, normalized helpers bridge legacy callers, and the main OpenAI/OpenAI-compatible/Anthropic/Gemini replay paths preserve unknown/null totals instead of forcing zeroes. |
| Low-level runtime stream event model | `language-model-v4-stream-part.ts` | `siumai-spec/src/types/streaming.rs` | Amber | `ChatStreamEvent` now exposes a first-class `Part(ChatStreamPart)` semantic channel plus a separate runtime replay carrier for protocol-only hints. OpenAI Responses, Anthropic, and Gemini parser coverage improved materially; true protocol-only replay now stays isolated in the carrier, while AI SDK-stable provider metadata such as Anthropic reasoning `signature` / `redactedData` remains on the semantic part lane. |
| Runtime part serializer safety | AI SDK runtime expectation for direct part replay | OpenAI Responses + Anthropic streaming serializers | Green | OpenAI Responses and Anthropic now normalize `Part -> Custom` before taking serializer locks, eliminating self-deadlock on direct stable-part replay. |
| Typed stream-part overlay | `language-model-v4-stream-part.ts` | `siumai-core/src/streaming/stream_part.rs` | Amber | The historical V3-named overlay is now a V4-capable superset and includes `custom` / `reasoning-file`, but the runtime event layer is still thinner and protocol-native handling remains partial. |
| Stream terminal response fidelity | AI SDK runtime expectation | `siumai-core/src/streaming/processor.rs` | Green | Terminal envelope fields and extra terminal content are preserved. |
| OpenAI-compatible stream terminal fidelity | OpenAI-compatible runtime parity | `siumai-core/src/standards/openai/compat/streaming.rs` | Green | Terminal chunk metadata now survives through `StreamEnd`, including EOF fallback finalization. |
| Anthropic usage/metadata roundtrip | Anthropic runtime parity | `siumai-protocol-anthropic/src/standards/anthropic/streaming/*` | Green | Extended usage and typed metadata extraction are preserved more reliably. |

## Concrete misalignment hotspots

## 1. Protocol coverage for the new stable parts is intentionally scoped

What landed:

- OpenAI Responses:
  - true request-side support for `openai.compaction`
  - explicit tool-result content mapping for `file-data`, `file-url`, `file-id`, `image-data`,
    `image-url`, and `image-file-id`
- Gemini:
  - true request/response support for `reasoning-file`
  - explicit `image-data` tool-result handling
  - unsupported tool-result variants still degrade to JSON text fallbacks
- OpenAI/OpenAI-compatible chat/tool messages:
  - no native structured tool-result content array exists on the wire
  - explicit V4 tool-result content variants are therefore preserved as JSON-string tool message
    payloads instead of being silently flattened or dropped
- Anthropic:
  - explicit tool-result mapping for images, PDF documents, URL-backed documents, and
    `tool_reference`
  - top-level `custom` / `reasoning-file` still degrade because Anthropic does not expose a stable
    request-time equivalent

Why this is still `Amber`:

- arbitrary `custom` is not a cross-provider story yet
- `reasoning-file` is not a true round-trip on every provider
- explicit tool-result file/image/id parts are stable, but not every protocol can carry them
  natively

## 2. Stable stream semantics are still split between two incomplete layers

Today:

- `ChatStreamEvent` remains the runtime transport enum
- it now includes a first-class `Part(ChatStreamPart)` semantic channel
- `LanguageModelV3StreamPart` remains the historical typed overlay used by gateway/protocol work

Recent improvement:

- `LanguageModelV3StreamPart` already carried the missing V4 `custom` and `reasoning-file` slots.
- `ChatStreamEvent::Part(ChatStreamPart)` now mirrors the important V4 stream-part semantics at the
  stable runtime layer.
- `StreamProcessor` and the main OpenAI/OpenAI-compatible/Anthropic/Gemini serializers can bridge
  that new runtime part channel instead of requiring a provider-scoped `Custom` payload first.
- Unsupported protocol reserialization still degrades V4-only parts explicitly through lossy text
  fallback instead of silently dropping them.
- OpenAI Responses parser now uses that channel for the non-tool AI SDK-stable semantics that map
  cleanly to `language-model-v3-stream-part.ts`.
- Gemini parser now uses that channel for reasoning/source/provider-executed tool semantics, which
  removes the old provider-scoped `gemini:*` wrapper from the main stable path.

Remaining problem:

- Anthropic parser/serializer coverage is now much closer, and provider-hosted server tool / MCP
  replay plus reasoning signature/redacted fidelity are now on the stable part lane through AI
  SDK-style `providerMetadata`
- OpenAI Responses tool/MCP helper events now use the stable part lane plus a separate runtime
  replay carrier for protocol-only `rawItem` / `outputIndex`
- direct serializer re-entry is now fixed for OpenAI Responses and Anthropic, so the remaining
  stream-model work is about stable shape completeness rather than transport deadlocks
- the overlay keeps the historical V3 name, so the compatibility story is clear internally but not
  yet fully normalized externally

Required direction:

- keep migrating providers/parsers toward emitting `ChatStreamEvent::Part` directly where useful,
  with the remaining work now focused on parity cleanup and any future truly protocol-only replay
  hints
- keep adapters to/from the runtime transport layer and the historical V3-named overlay
- stop treating major cross-provider stream semantics as provider-scoped `Custom` by default

## 3. Usage parity now uses a compatibility-superset stable surface

Current shape:

- legacy totals remain available:
  - `prompt_tokens`
  - `completion_tokens`
  - `total_tokens`
- AI SDK target shape is now present on the stable surface:
  - `inputTokens`
  - `outputTokens`
  - `raw`

What changed:

- stream and non-stream aggregation paths now populate the richer usage view first
- protocol serializers/parsers preserve provider-native raw usage where available
- compatibility helpers keep totals-based callers viable while protocol code targets the richer shape

## 4. Source has a final strict union, but provider-native coverage still varies

Already fixed in this branch:

- `mediaType`
- `filename`
- `providerMetadata`
- a strict `SourcePart::{Url, Document}` stable shape

Still open:

- some protocol/provider paths still inspect the wire-level `sourceType` string because their
  native payload structs are intentionally protocol-shaped
- provider-native request handling for `source` is still uneven and often degrades explicitly

## 5. Request-time controls still have bounded legacy fallback paths

Current state:

- the stable request surface now exposes the right `providerOptions` slots
- the main request converters read those slots first
- compatibility fallbacks still read historical response-style metadata in bounded paths
- the unified Rust content surface still carries both `providerOptions` and `providerMetadata`,
  which is wider than the AI SDK prompt contract but currently remains the pragmatic shared stable
  shape

Required direction:

- finish auditing the remaining converter/helper paths
- keep metadata-as-input only as an explicit migration bridge
- remove or deprecate those bridges once tests are broad enough
- keep treating `providerOptions` as the only canonical request-time input channel even while the
  shared content superset remains in place

## Recommended priority order

### P0 - Finish the remaining boundary cleanup

- audit the last request paths that still read response-style metadata
- remove bounded metadata-as-input bridges once coverage is sufficient

### P1 - Stable stream parity

- keep auditing whether any provider besides OpenAI Responses truly needs the replay-carrier
  pattern; Anthropic reasoning signature/redacted fields now intentionally stay on stable
  `providerMetadata` because AI SDK models them there
- keep tightening the mapping between runtime events, typed stream parts, and protocol-native
  serializers

### P1 - Stable usage parity

- AI-SDK-shaped usage model
- protocol and stream aggregation migrated to it

### P1 - Provider coverage cleanup

- decide where true provider support is expected for `custom`
- decide whether any additional providers should get true `reasoning-file` or tool-result file-id
  support

### P2 - Shape cleanup

- warning normalization cleanup
- compatibility adapters and deprecations

## Already-landed branch notes

The current branch should now be treated as the new baseline for this workstream:

- shared compatibility warning support
- widened `source` support
- strict `source` URL/document union
- widened `tool-approval-*` support
- Anthropic/OpenAI-compatible stream-end fidelity fixes
- OpenAI Responses approval-reason forwarding
- first-class message-level and part-level `providerOptions`
- first-class V4 `custom` and `reasoning-file` stable content parts
- V4-capable typed stream-part overlay for `custom` / `reasoning-file`
- explicit V4 tool-result content modeling (`file-data` / `file-url` / `file-id` / `image-data` /
  `image-url` / `image-file-id` / `custom`)
- Gemini reasoning-file wire support
- OpenAI Responses custom compaction support
- Anthropic tool-result `tool_reference` support
- OpenAI Responses parser-side stable part migration for stream-start/response-metadata/text /
  reasoning / source / successful finish semantics
- Anthropic parser-side stable part migration for stream-start/response-metadata/text / local
  tool-input / tool-call / reasoning / source / successful finish semantics
- Gemini parser-side stable part migration for reasoning / source / provider-executed tool parts
