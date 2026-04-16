# AI SDK Structural Alignment - Audit

Last updated: 2026-04-14

This note records the current structural parity status against the AI SDK provider contracts.

Status legend:

- `Green`: aligned enough for the intended stable semantics
- `Amber`: partially aligned; important slots still missing or still provider-scoped
- `Red`: materially misaligned; should drive refactor work

## Current summary

The current branch has now closed four concrete structural gaps:

- Anthropic streaming preserves extended usage and provider metadata across decode/encode
  round-trips.
- The public `siumai::protocol::anthropic::streaming::AnthropicEventConverter` surface now also
  has an explicit top-level regression for GitHub issue `#17`, so
  `cache_read_input_tokens`, `service_tier`, and `server_tool_use` are pinned on the facade
  roundtrip instead of only inside protocol-crate tests.
- Shared warnings now expose AI SDK-style `unsupported` / `compatibility` categories through a
  compatibility-superset model.
- Request-side `providerOptions` now exist on messages, request-capable content parts, and
  tool-result output/content shapes, and the main audited request converters now use them as the
  canonical request-time lane instead of historical metadata shims.
- Shared response-side `providerMetadata` is now explicitly unified around one AI SDK-style
  provider-rooted map: `ProviderMetadataMap` backs chat/completion/content/stream/file/skill
  results, helper accessors centralize the `provider_id -> object` contract, and the remaining UI
  exception is intentional because upstream `convertToModelMessages()` forwards UI
  `providerMetadata` as request-side `providerOptions`.
- The wider provider/helper surface has now been swept onto that same provider-rooted metadata
  contract too: Gemini/Vertex, Azure completion, Bedrock, Ollama, DeepSeek, Groq, xAI, and
  MiniMaxi typed metadata accessors no longer assume nested `HashMap<String, HashMap<...>>`
  layouts, Anthropic skills/streaming metadata now use the same shared root, and the audited
  `cargo check -p siumai --all-features` lane is green again after the response-side metadata
  cleanup.
- The follow-up fixture/example sweep is now also green on that same contract:
  `siumai-extras::StepResult`, OpenAI/OpenAI-compatible/Anthropic bridge fixtures, OpenAI
  Responses protocol round-trip tests, Anthropic thinking helpers, and the gateway loss-policy
  example no longer smuggle response metadata through nested `HashMap<String, HashMap<...>>`
  shims.
- The higher-level `uploadFile` helper now also carries canonical `providerOptions` through the
  shared `FileUploadRequest` / `UploadFileOptions` lane, so provider-owned upload controls no
  longer need ad hoc side channels: OpenAI/Azure honor `purpose` / `expiresAfter`, Gemini honors
  `displayName` plus polling controls, and Anthropic explicitly warns when extra upload options
  are ignored on its current beta files path.
- The follow-up shared-structure audit also closed two adjacent drift points exposed by the same
  pass: Ollama and Cohere now follow the newer `FilePartSource` split instead of assuming all
  user files/images are bare `MediaSource`, and Anthropic response/json conversion now handles
  `MessageContent::Json` explicitly instead of leaving that newer stable content variant outside
  the audited provider path.
- Anthropic custom provider-key semantics now match the audited AI SDK contract more closely:
  request shaping merges canonical `providerOptions.anthropic` with provider-owned custom keys
  such as `my-custom-anthropic`, custom keys override canonical fields when both are present, and
  top-level non-stream / finish / stream-end `providerMetadata` now duplicates onto the custom
  root only when that custom request key was actually used.
- Anthropic provider-defined tool/version drift against `repo-ref/ai` is now closed on the
  audited surface too: the shared tool catalog plus Anthropic request/header/parse/streaming
  layers cover `web_search_20260209`, `web_fetch_20260209`, `code_execution_20260120`, and
  `computer_20251124`; beta-header injection matches the upstream web/computer rules; and the
  2026 web-tool path now marks provider-executed `code_execution` calls as dynamic when no
  explicit code-execution tool was requested, matching AI SDK semantics.
- DeepSeek now follows the same AI SDK-style custom provider-root rule on its audited chat path:
  request shaping reads provider-owned options from the runtime provider namespace instead of
  hardcoded `deepseek`, response metadata is emitted under that resolved root, and the typed
  helper surface now exposes keyed accessors so custom DeepSeek-compatible ids can read the same
  metadata contract explicitly.
- DeepSeek provider-owned streaming now also matches the audited `@ai-sdk/deepseek` request
  contract across the public Rust entrypoints: the native DeepSeek package remains chat-only, and
  provider-owned stream requests now always emit `stream_options.include_usage = true` instead of
  diverging between the unified and config/provider builder paths.
- Gemini request conversion now also matches the audited `google|vertex` namespace precedence more
  closely: request-time provider options plus per-part `thoughtSignature` replay are read from the
  runtime namespace first (`vertex` for Vertex, `google` otherwise) with the canonical sibling key
  as fallback, mirroring the upstream AI SDK fix for mixed Google/Vertex request payloads.
- OpenAI-compatible response coverage now also mirrors that audited split more closely:
  non-stream response fixtures pin finalized tool-call
  `extra_content.google.thought_signature -> providerMetadata.{provider}.thoughtSignature`, the
  no-network OpenRouter public path now locks the same response metadata through unified/provider/
  config/registry entrypoints, and the requested camelCase metadata-key variant stays covered by
  the lower-level compat transformer/streaming tests.
- OpenAI typed response metadata helpers now also expose keyed accessors (`*_metadata_with_key`)
  for the shared OpenAI provider root, and the built-in Perplexity compat preset now exposes the
  AI SDK-shaped `providerMetadata.perplexity.{images,usage,cost}` structure instead of raw
  snake_case `usage/images` payloads.
- Perplexity's provider-owned typed option surface is now also cleaner on the audited package
  boundary: `PerplexityOptions` / `PerplexityWebSearchOptions` serialize with AI SDK-style
  camelCase on the public Rust surface, legacy snake_case input remains accepted as an alias, and
  the shared compat request boundary now owns the explicit lowering onto Perplexity's snake_case
  wire fields.
- The shared Fireworks compat preset no longer drifts on the completion-family boundary:
  built-in compat metadata now advertises `completion` on the config/registry path, public
  Fireworks completion parity tests pin the real `/completions` route across
  siumai/provider/config/registry entrypoints, and the typed Fireworks facade now includes the
  upstream empty embedding-option object plus deprecated alias names for package-surface
  comparison against `repo-ref/ai/packages/fireworks/src/index.ts`.
- The dedicated MoonshotAI compat wrapper no longer drifts on the audited public package boundary:
  canonical public/runtime identity is now `moonshotai`, the historical `moonshot` id is hidden
  behind a migration alias only, typed `thinking` / `reasoningHistory` options normalize onto the
  Moonshot wire keys (`thinking.budget_tokens`, `reasoning_history`), curated Kimi model/default
  constants now match the audited package subset more closely, and public-path guards explicitly
  keep completion/image/embedding unsupported to mirror the upstream chat-only package contract.
- The dedicated Mistral wrapper boundary is now documented explicitly too: after re-checking
  `repo-ref/ai/packages/mistral`, the audited package split remains `chat + embedding`, public
  typed options stay AI SDK-style camelCase, compat normalization owns the final
  `safe_prompt` / document-limit wire lowering, and completion/image stay intentionally
  unsupported on that wrapper boundary.
- xAI, Groq, and Amazon Bedrock now also expose AI SDK-style provider-option alias names on the
  provider-owned/public facade boundary, so public package comparison against
  `repo-ref/ai/packages/{xai,groq,amazon-bedrock}/src/index.ts` no longer has to translate those
  option exports manually.
- The Bedrock stream audit baseline is now tighter too: after re-checking
  `repo-ref/ai/packages/amazon-bedrock/src/bedrock-chat-language-model.ts`, the current upstream
  `doStream()` contract itself does not expose a native `source` or arbitrary `custom`
  stream-part lane, so those are no longer treated as known structural gaps on the audited
  Bedrock stream path.
- The shared JSON transport baseline is tighter too: stream-end synthesis no longer runs eagerly
  before EOF, so stateful converters keep terminal response text on clean shutdown. Bedrock
  reserved-JSON structured-output extraction is now green again on the executor, provider-owned,
  and public-path regression lanes.
- Native OpenAI now also exposes the main AI SDK-style typed option names on the provider-owned
  and public facade boundary: `OpenAILanguageModel{Chat,Responses,Completion}Options`,
  `OpenAIEmbeddingModelOptions`, `OpenAISpeechModelOptions`,
  `OpenAITranscriptionModelOptions`, and `OpenAIFilesOptions` now exist alongside the older
  Rust-first types, and the newly exposed speech/transcription option surfaces are backed by real
  request-shaping behavior instead of name-only aliases.
- Google Vertex now also exposes the safe AI SDK-style typed option alias subset on the
  provider-owned/public facade boundary: `GoogleVertexEmbeddingModelOptions`,
  `GoogleVertexImageModelOptions`, and deprecated `GoogleVertexImageProviderOptions` now map onto
  the existing native embedding/Imagen option types. Vertex video aliases remain intentionally
  deferred because the native provider crate does not yet own a real video runtime surface.
- Anthropic-on-Vertex public streaming is back on the expected compatibility contract too:
  stable `text-delta` / `reasoning-delta` parts now replay legacy `ContentDelta` /
  `ThinkingDelta` shadows through the shared stream factories, and metadata-only redacted thinking
  placeholders no longer leak as empty strings through `response.reasoning()` / `message.reasoning()`.
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
  `response-metadata`, non-tool `text-*`, `reasoning-*`, direct
  `response.custom_tool_call_input.* -> tool-input-*`, `source` including web-search citations,
  terminal `finish` on both successful and failed responses, and provider-hosted tool / MCP /
  approval semantics; protocol-only `rawItem` / `outputIndex` fidelity now rides a separate
  runtime replay carrier instead of staying inside loose custom payloads.
- `LanguageModelV3StreamPart::{from_runtime_part,to_runtime_part}` are now both public and
  test-backed, so bridge/gateway code can move directly between `ChatStreamPart` and the
  V4-capable typed overlay without detouring through provider-prefixed custom payloads.
- Experimental bridge primitive remappers now operate on direct `Part/PartWithReplay` tool
  semantics instead of only legacy `ToolCallDelta`, and they explicitly invalidate stale OpenAI
  Responses `rawItem` replay payloads after semantic rewrites so stable parts and replay metadata
  do not diverge.
- Provider-facing OpenAI Responses, Anthropic, and Gemini streaming extension helpers now also
  consume direct `Part` / `PartWithReplay` semantics first and only fall back to legacy
  custom-event shadows, so the public helper surface no longer depends on parser-specific
  `Custom` emission for source or provider-hosted tool inspection.
- The typed stream-part overlay now also names its protocol serializer escape hatch explicitly:
  `to_protocol_custom_event(...)` is the canonical lowering API for provider-native custom-event
  reserialization, while the older `to_custom_event(...)` remains only as a thin compatibility
  alias instead of implying that `Custom` is the preferred runtime lane.
- The shared OpenAI Responses bridge is now narrower too: it upgrades legacy custom payloads only
  when they already parse as stable V3/part shapes, and it no longer keeps bespoke event-type
  special cases for parser-era Gemini/Anthropic reasoning/tool shadow events that the audited
  mainline parsers no longer emit.
- Anthropic parsing now emits the runtime part channel directly for `stream-start`,
  `response-metadata`, `text-*`, standard local `tool-input-*` / `tool-call`, provider-hosted
  server tool / MCP `tool-*`, `reasoning-*`, `source`, and successful terminal `finish`
  semantics. Anthropic reasoning `signature_delta` and `redacted_thinking` now also stay on that
  stable lane through `reasoning-*` part `providerMetadata`, matching AI SDK behavior instead of
  relying on provider-scoped custom events.
- Gemini parsing now emits the runtime part channel directly for `reasoning-*`, `source`,
  provider-executed `tool-call` / `tool-result`, and `emit_v3_tool_call_parts=true` function-call
  paths instead of tunneling those semantics through `gemini:*` custom events.
- Gemini reasoning streaming no longer emits parser-side `gemini:reasoning` shadow duplicates on
  top of the stable runtime part lane; only the stable `reasoning-*` parts plus legacy
  `ThinkingDelta` compatibility remain on the main stream path.
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
| Tool approval shape | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/part.rs` | Green | `tool-approval-response.reason`, `tool-approval-response.providerExecuted`, `tool-approval-request.providerMetadata`, and prompt-side `providerOptions` are present. |
| Message-level provider options | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/message.rs` | Green | `ChatMessage` exposes first-class `providerOptions`, serde aliases, and helper APIs. |
| Part-level provider options | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/part.rs` | Green | Request-capable stable parts carry `providerOptions`, with helper accessors/builders. |
| Tool-result output/content provider options | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/tool_result.rs` | Green | `ToolResultOutput` and `ToolResultContentPart` now model AI SDK-style `providerOptions`. |
| Tool-result content granularity | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/tool_result.rs` | Green | Stable tool-result content now models the explicit V4 file/image/id variants plus `custom`, including `ToolResultFileId` for provider-keyed ids. |
| Shared provider metadata root | `packages/ai/src/types/provider-metadata.ts` | `siumai-spec/src/types/provider_metadata/mod.rs`, `siumai-core/src/streaming/processor.rs`, `siumai-core/src/standards/openai/compat/streaming.rs`, `siumai/src/files.rs`, `siumai/src/skills.rs` | Green | Shared `ProviderMetadataMap` now matches AI SDK `ProviderMetadata` semantics closely enough to be the stable baseline: provider-owned metadata lives under one explicit provider root object, helper accessors/mergers enforce that contract centrally, stream/chat/completion/upload results all reuse the same shape, and typed OpenAI/OpenAI-compatible/Anthropic helpers no longer assume lane-specific nested `HashMap` layouts. UI `providerMetadata` remains outside this row on purpose because AI SDK maps it to request-side `providerOptions`. |
| Request/response provider boundary | `shared-v4-provider-options.ts` + `shared-v4-provider-metadata.ts` | `siumai-core/src/standards/openai/utils.rs`, `siumai-protocol-anthropic/src/standards/anthropic/utils/content.rs`, `siumai-protocol-anthropic/src/standards/anthropic/{chat,transformers,streaming}` | Green | OpenAI-compatible, OpenAI Responses, and Anthropic request paths now use canonical `providerOptions` on the audited request boundary; request-side `providerMetadata` and `message.metadata.custom` no longer participate in those main request-only behaviors, and Anthropic custom provider ids now mirror AI SDK top-level metadata duplication semantics (`anthropic` plus the used custom root). |
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
  cleanly to `language-model-v3-stream-part.ts`, plus direct `custom_tool_call_input ->
  tool-input-*`, citation `source`, and buffered failed-finish emission.
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

## 5. Request-time controls now have a canonical audited boundary

Current state:

- the stable request surface now exposes the right `providerOptions` slots
- the main audited request converters now read those slots as the only canonical request-time input
- Anthropic Messages request fixtures now also pin the transport boundary for message/part request
  controls:
  - message-level `providerOptions.anthropic.cacheControl` is covered on the request-serialization
    path
  - part-level document `providerOptions.anthropic.{citations,title,context}` is fixture-backed
    on both request serialization and source-normalization replay
- the unified Rust content surface still carries both `providerOptions` and `providerMetadata`,
  which is wider than the AI SDK prompt contract but currently remains the pragmatic shared stable
  shape

Required direction:

- keep extending the same audit standard to any remaining unaudited converter/helper paths
- keep `providerMetadata` response-time only in intent, even though the shared stable superset
  still carries both lanes
- keep treating `providerOptions` as the only canonical request-time input channel even while the
  shared content superset remains in place

## Recommended priority order

### P0 - Stable stream parity

- keep auditing whether any provider besides OpenAI Responses truly needs the replay-carrier
  pattern; Anthropic reasoning signature/redacted fields now intentionally stay on stable
  `providerMetadata` because AI SDK models them there
- keep tightening the mapping between runtime events, typed stream parts, and protocol-native
  serializers

### P1 - Provider-boundary coverage expansion

- keep checking the remaining lesser-used provider/protocol helpers for accidental request-side
  metadata reads
- preserve `providerOptions` as the only canonical request-time lane on newly aligned paths

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
- DeepSeek provider-owned stream default now matches the audited AI SDK `include_usage` contract
- shared JSON transport EOF synthesis now preserves stateful `StreamEnd.response` content
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
  reasoning / tool-input / source / successful + failed finish semantics
- Anthropic parser-side stable part migration for stream-start/response-metadata/text / local
  tool-input / tool-call / reasoning / source / successful finish semantics
- Gemini parser-side stable part migration for reasoning / source / provider-executed tool parts
