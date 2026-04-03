# AI SDK Structural Alignment - Data Structure Matrix

Last updated: 2026-04-02

This note compares the most important stable/runtime data structures against the current AI SDK
provider contracts in `repo-ref/ai`.

References:

- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-prompt.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-part.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-source.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-usage.ts`
- `repo-ref/ai/packages/provider/src/language-model/v3/language-model-v3-stream-part.ts`

Status legend:

- `Green`: structurally aligned enough to be the stable baseline
- `Amber`: structurally close, but important fidelity or coverage gaps remain
- `Red`: missing enough structure that further refactor should be treated as mandatory

## Executive summary

The stable prompt/content, usage, and request-boundary layers are now close to AI SDK V4 shape
parity.

The remaining structural risk is concentrated in streaming:

- the stable stream-part shape exists
- the runtime transport has a first-class `ChatStreamEvent::Part(ChatStreamPart)` channel
- but parser/provider coverage is still incomplete
- and some protocol-native replay fields still live outside the stable model

The biggest remaining gaps are:

1. parser/provider stream-part emission is still uneven outside the already-audited
   OpenAI Responses / Anthropic / Gemini main paths
2. the runtime stream story is still split across `ChatStreamEvent`, `ChatStreamPart`, and the
   compatibility overlay even though the public naming gap is now reduced by the
   `LanguageModelV4*` aliases
3. shared non-chat request abstractions are improved but not fully AI SDK-complete:
   `ImageEditRequest` now models typed multi-input `images[]` + `mask` semantics, and
   `ImageVariationRequest` now also carries a typed file/url image input instead of a raw byte bag.
   Those typed image inputs now also expose per-input `providerOptions`, bringing the shared image
   input lane closer to AI SDK `ImageModelV4File`. The shared image request family now exposes
   top-level `aspectRatio` plus shared `seed` across
   generation/edit/variation, compat/OpenAI image warning semantics now match the AI SDK-style
   `unsupported { feature }` lane for top-level `aspectRatio` / `seed`, and xAI / Google / Vertex
   supported image paths now consume those canonical shared fields directly. The remaining gaps are
   now more runtime/provider-specific: URL-backed edit inputs are still provider-conditional on
   multipart/inline paths, some variation-specific transformers remain incomplete, and the shared
   video request shape still carries a few provider-centric behaviors that should likely stay
   behind provider-owned typed layers
4. the OpenAI-compatible provider-settings surface is now materially closer to AI SDK:
   `metadataExtractor`, `includeUsage`, `transformRequestBody`, `queryParams`, and an explicit
   `supportsStructuredOutputs` provider-level policy all exist on the public config/builder/runtime
   path now, and the default `supportsStructuredOutputs` behavior now also matches AI SDK's
   conservative default `false`

## Structure matrix

| Surface | AI SDK reference | Current Siumai anchor | Status | Notes |
| --- | --- | --- | --- | --- |
| Prompt message envelope | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/message.rs` | Green | Message-level `providerOptions` exists, request converters now prefer it over legacy metadata shims, the bridge-side message compaction path now preserves single-text parts when they carry canonical provider options, and the public macro/example surface has been swept to compile against that field. |
| Prompt content parts | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/part.rs` | Green | Stable content now includes `custom`, `reasoning-file`, strict `source`, tool approval fields, and request-side `providerOptions`. OpenAI Responses request normalization now also restores wire `input_image` items back into AI-SDK-shaped user file parts with canonical image-detail `providerOptions`. |
| Tool-result output | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/tool_result.rs` | Green | `text`, `json`, `execution-denied`, `error-text`, `error-json`, and `content` are all modeled with `providerOptions`. OpenAI/OpenAI-compatible tool-message conversion has also been reviewed: the wire contract is string-only, so structured content degrades to a JSON string rather than a lossy flat string. |
| Tool-result content variants | `language-model-v4-prompt.ts` | `siumai-spec/src/types/chat/content/tool_result.rs` | Green | Explicit `file-data`, `file-url`, `file-id`, `image-data`, `image-url`, `image-file-id`, and `custom` variants are present, including provider-keyed ids. OpenAI/OpenAI-compatible tool messages preserve these typed inner variants inside the JSON string payload, OpenAI Responses request fixtures now lock native `file_id` roundtrip coverage for `image-file-id` / `file-id`, and the remaining user-side `type:file` payloads in fixtures are intentional `ContentPart::File` cases rather than legacy tool-result shapes. |
| Shared image request surface | `packages/provider/src/image-model/v4/image-model-v4-call-options.ts` | `siumai-spec/src/types/image.rs`, `siumai-protocol-openai/src/standards/openai/image.rs`, `siumai-protocol-openai/src/standards/openai/compat/spec.rs`, `siumai-protocol-gemini/src/standards/gemini/transformers/request.rs`, `siumai-provider-google-vertex/src/standards/vertex_imagen.rs`, `siumai-provider-xai/src/providers/xai/image.rs` | Amber | The shared Rust image request family now exposes top-level `aspectRatio` across generation/edit/variation, shared `seed` across the same family, typed per-input `providerOptions` on file/url edit inputs, and a typed variation image input instead of a raw byte-only field, which brings the core call-option shape much closer to AI SDK `ImageModelV4CallOptions` plus `ImageModelV4File`. OpenAI/OpenAI-compatible generation/edit/variation warnings now surface AI SDK-style unsupported `aspectRatio` / `seed` semantics, and the audited xAI / Google / Vertex image paths now consume canonical top-level `aspectRatio` / `seed` on their supported routes instead of depending on provider-only options. The remaining structural risk is no longer the top-level request shape itself; it is concentrated in provider/runtime coverage, especially URL-backed edit materialization on multipart/inline providers and incomplete variation-specific transformer coverage such as the Vertex Imagen variation path. |
| Provider-owned image/video requests | `packages/xai/src/xai-image-model.ts`, `packages/xai/src/xai-video-model.ts`, `packages/xai/src/xai-image-options.ts`, `packages/xai/src/xai-video-options.ts`, `packages/provider/src/video-model/v4/video-model-v4-call-options.ts`, `packages/provider/src/video-model/v4/video-model-v4-file.ts` | `siumai-spec/src/types/image.rs`, `siumai-spec/src/types/video.rs`, `siumai-provider-xai/src/providers/xai/image.rs`, `siumai-provider-xai/src/providers/xai/video.rs`, `siumai-provider-gemini/src/providers/gemini/video.rs`, `siumai-provider-minimaxi/src/providers/minimaxi/video.rs` | Green | The provider-owned image/video routing is now much closer to the audited AI SDK shape on both the shared and provider boundaries: typed `XaiImageOptions` / `XaiVideoOptions` plus request ext traits are public, image generation/edit map to `/images/generations` and `/images/edits`, xAI video create/query map to `/videos/generations|edits` and `GET /videos/{request_id}`, shared `ImageEditRequest` models typed multi-input `images[]` + `mask`, and shared `VideoGenerationRequest` now carries typed `VideoGenerationInput` file/url inputs plus canonical `count` (`n`), `fps`, and `seed`. Those typed video inputs now also expose per-input `providerOptions`, matching the audited `VideoModelV4File` direction more closely. xAI now warns-and-filters unsupported `n` / `fps` / `seed`, Gemini/Veo consumes canonical `count` / `seed` / typed image input and surfaces warnings for unsupported URL/FPS cases, and MiniMaxi video-only knobs now also live on provider-owned `MinimaxiVideoOptions` instead of shared top-level fields. The remaining major gap in this row is the shared async materialization story for URL-backed image edit inputs on multipart/inline provider paths. |
| Speech/transcription request surface | `packages/provider/src/speech-model/v4/speech-model-v4-call-options.ts`, `packages/provider/src/transcription-model/v4/transcription-model-v4-call-options.ts` | `siumai-spec/src/types/audio.rs`, `siumai-core/src/execution/executors/audio.rs`, `siumai-protocol-openai/src/standards/openai/audio.rs`, `siumai-provider-openai/src/providers/openai/client/audio.rs` | Amber | TTS is structurally close enough at the stable request layer: text/voice/format/speed/providerOptions are already present. STT/translation has now been moved onto the same canonical direction as AI SDK `TranscriptionModelV4CallOptions`: the shared request surface uses `audio + mediaType + providerOptions`, stable request structs no longer carry the old `audio_data | file_path` split, helper-level `transcribe_file(...)` / `translate_file(...)` materialize local paths before the request reaches executors/providers, and OpenAI multipart shaping now consumes that canonical audio input directly. The main remaining gap in this row is that `mediaType` is still optional on the stable Rust request types, so the contract is closer to AI SDK but not yet fully tightened to the upstream required-input semantics. |
| Source union | `language-model-v4-source.ts` | `siumai-spec/src/types/chat/content/part.rs` + `siumai-spec/src/types/streaming.rs` | Green | Stable source now uses a strict `Url | Document` union and preserves `providerMetadata`. OpenAI Responses response/stream roundtrip fixtures already exercise document-style sources through file-search and code-interpreter cases, and the typed provider metadata now preserves document citation `type` / `index` across non-stream JSON and same-protocol SSE roundtrips, so this is no longer just a static type-level alignment claim. |
| Usage | `language-model-v4-usage.ts` | `siumai-spec/src/types/usage.rs` | Green | AI SDK `inputTokens` / `outputTokens` / `raw` are now the canonical stable storage. Legacy `prompt/completion/total` counts were reduced to private compatibility seeds plus accessors/serde (`prompt_tokens()`, `completion_tokens()`, `total_tokens()`), so public callers no longer depend on compatibility fields as the storage root. The audited OpenAI/OpenAI-compatible/Anthropic/Gemini replay paths preserve unknown/null totals correctly, OpenAI Responses exact response fixtures assert the nested/raw usage view explicitly, and Anthropic keeps `Usage.raw` to the stable provider-raw subset while preserving the full provider-native usage payload under `provider_metadata.anthropic.usage`. |
| Text completion family | `packages/openai-compatible/src/completion/*` | no stable Rust capability / request-response family yet | Red | The Rust public surface still has no dedicated completion capability analogous to AI SDK `completionModel()`. This is now a documented architecture gap rather than a hidden protocol bug: closing it would require a new stable completion request/response family and public trait surface, not just another compat hook on the existing chat path. |
| Typed stream-part overlay | `language-model-v4-stream-part.ts` | `siumai-core/src/streaming/stream_part.rs` | Green | The overlay is already a V4-capable superset, and it now also exposes public `LanguageModelV4*` aliases so new code can use AI SDK-aligned naming while the historical `LanguageModelV3*` names remain as compatibility shims. |
| Runtime semantic stream carrier | `language-model-v4-stream-part.ts` | `siumai-spec/src/types/streaming.rs` | Amber | `ChatStreamPart` mirrors the major AI SDK stream semantics, Siumai intentionally keeps stream `tool-result` free of a synthetic `providerExecuted` flag because AI SDK stream parts do not model it, true protocol-only replay hints now live in a separate runtime replay carrier instead of leaking through `providerMetadata` or loose custom JSON, and the stable runtime stream types are now re-exported through the normal streaming/prelude surface so downstream code no longer needs `__private::types`. Provider-owned stable metadata, such as Anthropic reasoning `signature` / `redactedData`, stays on the semantic part itself because AI SDK exposes it there. |
| High-level stream consumers | `packages/ai/src/generate-object/stream-object.ts`, `packages/ai/src/agent/tool-loop-agent.ts` | `siumai-extras/src/highlevel/object.rs`, `siumai-extras/src/server/tool_loop.rs`, `siumai-extras/src/server/axum/sse.rs` | Green | `stream_object` now accumulates stable `ToolInputDelta` / `ToolCall` parts before falling back to legacy deltas, the extras tool loop deduplicates stable-part vs legacy tool accumulation by source so mixed streams do not double-apply arguments, and the Axum SSE adapter now exposes runtime `Part` / `PartWithReplay` as explicit `event: part` frames instead of silently dropping the upgraded semantic lane. |
| Direct part serialization safety | AI SDK runtime expectation | OpenAI Responses + Anthropic serializers | Green | OpenAI Responses and Anthropic now normalize `Part -> Custom` before taking serializer state locks, so runtime-part replay no longer deadlocks. |
| Parser-side stable-part emission | AI SDK runtime expectation | provider protocol parsers | Amber | OpenAI Responses and Gemini now emit `Part` for major stable semantics; Anthropic now emits `Part` for `stream-start`, `response-metadata`, `text-*`, provider-hosted server tool / MCP `tool-*`, standard local `tool-input-*` / `tool-call`, `reasoning-*`, `source`, and successful `finish`; OpenAI-compatible chat chunks now emit stable lifecycle parts for `stream-start`, `response-metadata`, `text-*`, `reasoning-*`, `finish`, URL `source`, and the tool lifecycle (`tool-input-start` / `tool-input-delta` / `tool-input-end` / `tool-call`) while keeping legacy shadow deltas for compatibility, and serializer/processor consumers now deduplicate mixed stable+legacy tool streams by first source. Public response/roundtrip fixtures now pin the main citation-facing paths (`message.annotations` and `delta.annotations`) plus same-protocol `response-metadata`, streamed terminal `logprobs`, AI SDK-style `acceptedPredictionTokens` / `rejectedPredictionTokens` mirrored from terminal `usage.completion_tokens_details`, and terminal response-envelope `system_fingerprint` / `service_tier` fidelity, so the remaining work is mainly rarer transport/protocol-hint edges. |
| Request/response provider boundary | `shared-v4-provider-options.ts` + `shared-v4-provider-metadata.ts` | shared stable types + provider converters | Green | The stable slots now behave canonically on the audited request paths: Anthropic same-protocol reasoning replay no longer depends on message-level custom keys, Anthropic document citations/title/context plus per-part cache control no longer read request-time values from response-side file `provider_metadata` or legacy `message.metadata.custom` shims, Anthropic Messages request normalization now canonicalizes document/cache-control request semantics directly onto part `providerOptions.anthropic`, the experimental direct-pair reasoning bridge plus Anthropic reasoning inspection no longer treat request-side `providerMetadata.anthropic|openai` as replay input, OpenAI Responses request conversion/warnings/normalization now use canonical `providerOptions` for reasoning/image-detail/approval-id request behavior and assistant tool-call ids, OpenAI Responses response parsing/serialization now also preserves AI SDK-style typed response metadata on the stable boundary (`responseId` / `serviceTier` at response level, `itemId` / `phase` / raw `annotations` on text parts, and document citation `type` / `index` on sources) without duplicating assistant message ids on response-level provider metadata for the main OpenAI/Azure parsed path, xAI Responses now follows its own audited AI SDK split on both sides as well (`providerMetadata.xai.itemId` on reasoning only, metadata-free text/source parts, assistant xAI message/tool-call ids staying on direct request items, top-level `reasoning.{effort,summary}` / `topLogprobs` / `previousResponseId` / `store+include` request knobs mapped on the Responses path, Responses-only knobs stripped back out of the xAI `/chat/completions` path, and the xAI provider-tool surface now matches the audited `web_search` / `x_search` / `code_execution` / `view_image` / `view_x_video` / `file_search` / `mcp` set with snake_case arg serialization plus dropped invalid server-tool `tool_choice` forcing), the public Rust xAI tool layer now also mirrors the AI SDK factory surface with typed arg structs/builders (`WebSearchArgs`, `XSearchArgs`, `FileSearchArgs`, `McpArgs`, `*_with(...)`) instead of raw JSON bags, the provider-owned xAI typed surface is now split the same way (`XaiChatOptions` vs `XaiResponsesOptions`) with enum-backed reasoning/include slots instead of raw `String` / `Vec<String>` bags, xAI chat typed options now also cover `parallel_function_calling`, normalize deprecated `xHandles` into wire `included_x_handles`, align `with_default_search()` with the upstream `maxSearchResults=20` default, and model search sources as a discriminated union (`web` / `news` / `x` / `rss`) instead of a single wide struct, plain or empty assistant `output_text` no longer collapses back to bare `MessageContent::Text`, OpenAI Chat and OpenAI-compatible request conversion no longer treat request-side `providerMetadata` / `message.metadata.custom` as input for image-detail or extra-param behavior, OpenAI/OpenAI-compatible chat responses now also mirror `completion_tokens_details.accepted_prediction_tokens` / `rejected_prediction_tokens` into AI SDK-style typed provider metadata (`acceptedPredictionTokens` / `rejectedPredictionTokens`), the compat response-metadata ownership is now provider-owned rather than shared-whitelist-driven, and the public compat builder/config surface now exposes the same concept as an explicit `ResponseMetadataExtractor` hook instead of forcing callers to replace the whole adapter. That public compat request-settings lane now also covers AI SDK-style `includeUsage`, `transformRequestBody`, `queryParams`, and an explicit `supportsStructuredOutputs` provider-level policy: default compat chat streams omit `stream_options.include_usage` until explicitly enabled, callers can customize the final normalized compat chat payload through a public `RequestBodyTransformer` hook, compat route generation now appends deterministic provider query params across chat / embeddings / image generation-edit-variation / audio / rerank / model-listing paths, compat chat now defaults to downgrading JSON Schema outputs to wire `json_object` while surfacing a stable `unsupported { feature: "responseFormat" }` warning middleware unless callers explicitly opt back into wire `json_schema` with `supportsStructuredOutputs = true`, the audited known compat chat options now follow AI SDK parse/mapping semantics from deprecated `openai-compatible`, canonical `openaiCompatible`, and provider-owned keys (`user`, `reasoningEffort`, `textVerbosity`, `strictJsonSchema` -> `user`, `reasoning_effort`, `verbosity`, `response_format.json_schema.strict`), provider-defined tools now emit AI SDK-style `unsupported { feature: "provider-defined tool <id>" }` warnings on the default compat runtime response path while still being filtered out of Chat Completions requests, and the deprecated `providerOptions['openai-compatible']` request key now also emits the AI SDK-style `other` deprecation warning while preserving that audited compatibility lane. The shared `ProviderOptionsMap` serde now normalizes JSON provider ids so wire `openaiCompatible` keys hit the same canonical lookup path as builder-authored options, and builder/provider-owned/config-first/registry public-path parity now pins those canonical semantics for OpenAI Chat `providerOptions.openai.imageDetail` plus OpenAI-compatible message/part/tool-result `providerOptions.openaiCompatible`. |

## Provider-by-provider stream status

| Provider/protocol | Stable part parsing | Stable part serialization | Main remaining gap |
| --- | --- | --- | --- |
| OpenAI Responses | Green for `stream-start`, `response-metadata`, non-tool `text-*`, `reasoning-*`, `source`, `tool-input-*`, provider-hosted tool / MCP / approval events, successful `finish`, and `error` | Green for direct runtime `Part` replay after the deadlock fix, including raw-item / output-index replay via the runtime carrier | Remaining work is no longer structural on the OpenAI Responses stream lane; follow-up is mostly cleanup and parity review |
| xAI Responses | Green for `stream-start`, `response-metadata`, xAI-style `text-*` / `reasoning-*` / `source` / `finish`, plus `web_search` / `x_search` / `file_search` tool-input mapping and AI SDK-style finalized `custom_tool_call` emission for `x_search` / `view_x_video` | Green on the shared OpenAI Responses SSE serializer path after restoring xAI reasoning `providerMetadata.xai.itemId`, missing-start backfill, and audited `custom_tool_call` buffering/finalization semantics | Remaining work is mainly broader fixture coverage beyond the audited text/tool lanes and larger provider-native image/video families |
| Anthropic Messages SSE | Amber | Green for direct runtime `Part` replay after the deadlock fix | Main stable semantics, including provider-hosted server tool / MCP tool-call/tool-result replay and reasoning signature/redacted replay, now use the stable part lane. Unsupported parts on that same lane also honor `AsText` fallback now, so lossy transcoding no longer drops approval semantics; the remaining work is mostly cleanup around rarer provider-specific custom/raw hints |
| Gemini GenerateContent SSE | Green for reasoning/source/provider-executed tool parts and optional tool-call part mode | Amber/Green depending on part kind; the current serializer already routes direct runtime parts without the OpenAI/Anthropic lock bug | Usage replay now preserves `cachedContentTokenCount` / `trafficType` and counts reasoning inside total output usage; the remaining work is mostly parity cleanup rather than transport shape |
| OpenAI-compatible chat chunks | Amber for stable parts with legacy shadow compatibility | Green with explicit lossy fallback for unsupported V4-only parts | Parser-side lifecycle semantics now ride the direct `Part` lane for `stream-start`, `response-metadata`, `text-*`, `reasoning-*`, `finish`, the tool lifecycle, and URL `source` citations from `delta.annotations`, with stable-first emission plus first-source-wins deduplication preventing duplicate tool argument accumulation. Stable URL `source` parts now also reserialize back into chat-completions `delta.annotations`, non-stream chat responses map `message.annotations` onto stable `source` content parts, and fixture/public-path coverage now pins both citation paths, upstream `text -> tool-call -> source` ordering, same-protocol `response-metadata` / streamed terminal `logprobs` / `acceptedPredictionTokens` / `rejectedPredictionTokens`, and terminal response-envelope `system_fingerprint` / `service_tier` fidelity. Azure model-router `prompt_filter_results` preludes with empty `id` / `model` plus `created = 0` no longer trigger premature response-metadata emission, and compat response metadata is now adapter-owned rather than inferred by a shared whitelist, so the remaining work is mostly raw-hint coverage rather than core semantic parity. |

## What is still not structurally complete

### 1. Protocol-only replay hints now have an explicit runtime carrier where they are truly protocol-only

OpenAI Responses `rawItem` / `outputIndex` replay is now modeled through a separate runtime replay
carrier attached alongside stable `ChatStreamPart` semantics.

That means:

- `ChatStreamPart` stays semantic
- protocol replay state is not stuffed into generic `providerMetadata`
- same-protocol replay no longer depends on loose provider-scoped custom JSON payloads

Current conclusion after auditing Anthropic against
`repo-ref/ai/packages/anthropic/src/anthropic-messages-language-model.ts`:

- Anthropic reasoning `signature` and `redactedData` are not protocol-only replay state
- AI SDK exposes them on stable `reasoning-*` part `providerMetadata`
- Siumai now follows that same rule

So the dedicated runtime replay carrier remains the right tool for true wire-only fidelity such as
OpenAI Responses `rawItem` / `outputIndex`, but it is no longer the preferred home for Anthropic
reasoning signatures.

### 2. The runtime stream story is structurally correct but still split

Today Siumai has three relevant layers:

- `ChatStreamEvent`
- `ChatStreamPart`
- `LanguageModelV3StreamPart` / `LanguageModelV4StreamPart`

This is workable, but still more complicated than the AI SDK union story.

Recommendation:

- keep `ChatStreamEvent` as the transport/runtime envelope
- keep `ChatStreamPart` as the primary semantic runtime part
- keep the new `LanguageModelV4*` aliases as the recommended public naming for the upgraded
  overlay
- keep `LanguageModelV3StreamPart` as a compatibility name, not as the long-term semantic center

### 3. Parser migration is still uneven

The runtime part channel is only fully useful once parsers emit it directly when they already know
the stable meaning.

Current priority order:

1. finish parser/provider migration onto `ChatStreamEvent::Part` for the remaining OpenAI-compatible raw-hint and rarer provider-specific edges
2. continue tightening public-path fixture coverage for the remaining OpenAI-compatible raw-hint edges beyond the now-covered metadata/logprobs/prediction-token paths (the Azure model-router placeholder-metadata prelude is now covered)
3. finish the shared non-chat follow-ups, especially URL-backed image edit materialization on multipart/inline providers and any leftover provider-owned video knobs
4. decide whether the extras SSE `event: part` transport should become the documented public semantic export lane or remain an internal adapter contract
## Recommended next fearless-refactor cuts

### Cut 1 - Split semantic part data from protocol replay hints

Target result:

- stable AI SDK semantics remain in `ChatStreamPart`
- protocol replay hints move into a separate runtime carrier
- serializers stop depending on provider-scoped custom payloads for same-protocol replay

### Cut 2 - Finish parser-side migration to `ChatStreamEvent::Part`

Target result:

- OpenAI-compatible no longer relies on legacy-only transport deltas for tool-call lifecycle semantics
- the remaining parser/protocol bridges become easier to audit because stable semantics share one runtime lane

### Cut 3 - Finish the shared non-chat cleanup pass

Target result:

- URL-backed image edit inputs have a deliberate shared materialization strategy instead of
  provider-conditional rejection on multipart/inline paths
- the remaining shared video surface stays provider-neutral, with provider-owned knobs continuing
  to move behind typed provider options where needed
