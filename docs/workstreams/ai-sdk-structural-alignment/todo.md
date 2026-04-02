# AI SDK Structural Alignment - TODO

Last updated: 2026-04-02

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## 0) Lock the audit and target references

- [x] Create a dedicated workstream folder for the structural parity pass.
- [x] Record the main AI SDK V4 provider reference files.
- [x] Record the current Siumai anchor files for prompt/content/usage/stream semantics.
- [x] Write down the current red/amber/green parity table.

## Track A - Shared semantic fixes

- [x] Add a Vercel-aligned `compatibility` warning shape.
- [x] Move the `systemMessageMode=remove` warning to that compatibility category.
- [x] Extend unified `source` parts with:
  - `mediaType`
  - `filename`
  - `providerMetadata`
- [x] Extend unified `tool-approval-response` with `reason`.
- [x] Extend unified `tool-approval-request` with `providerMetadata`.
- [x] Normalize unsupported warnings behind an AI-SDK-shaped `unsupported { feature }`
  compatibility layer while keeping legacy unsupported variants for compatibility.
- [x] Decide the final Rust shape for `source`:
  - strict `Url | Document` enum carried as `SourcePart`
  - compatibility-preserving `sourceType` wire serialization

## Track B - Fix the request/response provider boundary

- [x] Add message-level `providerOptions`.
- [x] Add content-part-level `providerOptions` where AI SDK prompt/content parts support them.
- [x] Add helper APIs/builders for message/part provider options so callers do not need to mutate
  raw maps by hand.
- [x] Audit every request conversion path that still reads `provider_metadata` for request-only
  behavior.
- [x] Narrow the OpenAI Responses compatibility shim to the upstream-kept subset:
  - request conversion / warnings / normalization now use canonical `providerOptions` for
    reasoning `itemId` / `reasoningEncryptedContent`, image `imageDetail`, compaction metadata,
    and MCP approval ids
  - assistant tool-call `itemId` is now also pinned to canonical `providerOptions`; legacy
    `provider_metadata` is ignored on the audited request path
- [x] Remove Anthropic request-side document-setting fallback reads from response-style channels:
  - document citations/title/context no longer read file `provider_metadata` or
    `message.metadata.custom["anthropic_document_*"]`
  - per-part cache control no longer reads
    `message.metadata.custom["anthropic_content_cache_*"]`
- [x] Remove the Anthropic message-level thinking replay shim from the main request path:
  - Anthropic prompt replay now reads reasoning-part Anthropic replay metadata instead of
    `message.metadata.custom["anthropic_*"]`
  - Anthropic assistant-message replay helper now writes next-turn `providerOptions.anthropic`
    on reasoning parts instead of message-level custom keys
- [x] Align the experimental request bridge with the same provider boundary:
  - Anthropic direct-pair and normalization paths now treat reasoning replay metadata as
    request-side `providerOptions` and response-side `providerMetadata`
  - OpenAI encrypted reasoning -> Anthropic redacted thinking bridge annotations now land on
    reasoning-part `providerOptions.anthropic.redactedData`
  - Anthropic Messages request normalization now writes document citations/title/context and
    content-block cache control directly onto canonical part `providerOptions.anthropic` instead
    of bouncing those request semantics through `message.metadata.custom`
  - bridge-side Anthropic cache-limit inspection now reads canonical part
    `providerOptions.anthropic.cacheControl`
  - direct-pair reasoning replay helpers and bridge-side Anthropic reasoning inspection no longer
    treat request-side `providerMetadata.anthropic|openai` as input
- [x] Migrate OpenAI/OpenAI-compatible request conversion away from metadata-as-input for the main
  user-visible request paths.
  - OpenAI Chat `imageDetail` now reads only canonical part `providerOptions.openai|azure`.
  - OpenAI-compatible extra request params now read only canonical message/part/tool-result
      `providerOptions.openaiCompatible`; request-side `providerMetadata.openaiCompatible` and
      `message.metadata.custom.openaiCompatible` are no longer treated as input.
  - `ProviderOptionsMap` serde now normalizes JSON provider ids, so wire `openaiCompatible` keys
    resolve the same way as builder-inserted provider options.
- [x] Move OpenAI-compatible response-metadata extraction behind provider-owned adapter policy.
  - shared compat decoding no longer decides metadata namespaces through a hardcoded whitelist
  - `OpenAiStandardAdapter` / `ConfigurableAdapter` now opt known providers into `sources`,
    `logprobs`, and prediction-token metadata explicitly
  - generic OpenAI-compatible providers now match AI SDK `openai-compatible` defaults more closely
    by not inferring those metadata fields unless the provider adapter opts in
- [x] Add a public OpenAI-compatible response metadata extractor hook on the config/builder surface.
  - `ResponseMetadataExtractor` now models the AI SDK-style extension point directly
  - `OpenAiCompatibleConfig::with_metadata_extractor(...)` wraps the current adapter instead of
    forcing users to replace it
  - `OpenAiCompatibleBuilder::with_metadata_extractor(...)` and the public `siumai` facade
    re-export now lock that hook on the user-visible path too
- [x] Align OpenAI-compatible provider-level `includeUsage` behavior with AI SDK.
  - compat chat streams now omit `stream_options.include_usage` by default
  - `OpenAiCompatibleConfig::with_include_usage(...)` and
    `OpenAiCompatibleBuilder::with_include_usage(...)` now opt the field back in explicitly
  - xAI/Groq keep their provider-specific stripping behavior because they do not accept
    `stream_options` on the audited chat-completions path
- [x] Add a public OpenAI-compatible request-body transformer hook on the config/builder surface.
  - `RequestBodyTransformer` now mirrors AI SDK `transformRequestBody`
  - compat runtime request settings apply the hook after built-in/provider normalization on the
    final chat payload
  - public facade imports now expose the hook and request-settings type
- [x] Finish the remaining OpenAI-compatible provider-settings audit against
  `repo-ref/ai/packages/openai-compatible/src/openai-compatible-provider.ts`.
  - provider-level `queryParams` now flows through config/builder/runtime/spec URL generation for
    compat chat / embeddings / image generation-edit-variation / audio / rerank / models
  - provider-level `supportsStructuredOutputs` now has an explicit public/runtime policy surface:
    compat chat now defaults to downgrading JSON Schema chat outputs to wire `json_object`,
    emits a stable `unsupported { feature: "responseFormat" }` warning middleware, and preserves
    wire `json_schema` only when callers explicitly set `supportsStructuredOutputs = true`
- [x] Migrate Anthropic request conversion away from metadata-as-input for the main user-visible
  request paths.
- [x] Remove the remaining temporary request-side metadata fallbacks on the audited paths.
  - OpenAI Responses assistant tool-call `itemId` now also ignores legacy metadata input and
    requires canonical `providerOptions`.
  - OpenAI Chat / OpenAI-compatible main request conversion paths no longer keep metadata-input
    fallbacks.
  - experimental request-bridge reasoning paths no longer keep `providerMetadata` fallbacks.
- [x] Migrate canonical OpenAI Responses request fixtures away from request-side
  `providerMetadata`.
- [x] Stop Anthropic document and per-part cache-control builder helpers from writing legacy
  `message.metadata.custom` request shims when canonical part `providerOptions` are available.
- [x] Add regression tests that prove request conversion prefers `providerOptions` over historical
  metadata shims.
- [x] Finish the xAI chat typed-option parity sweep against `repo-ref/ai/packages/xai/src/xai-chat-options.ts`.
  - `parallel_function_calling` is now part of the typed xAI surface across options/builder/config/shared facade
  - deprecated `xHandles` input is normalized to wire `included_x_handles`
  - `with_default_search()` now matches the upstream `maxSearchResults=20` default
  - typed `SearchSource` now follows the upstream discriminated-union split instead of one permissive field bag
  - the provider-owned typed surface is now explicitly split into `XaiChatOptions` and
    `XaiResponsesOptions`
  - reasoning/include slots are now enum-backed typed wrappers instead of raw
    `String` / `Vec<String>` bags
- [x] Finish the xAI Responses provider-tool parity sweep against
  `repo-ref/ai/packages/xai/src/tool/*` and `responses/xai-responses-prepare-tools.ts`.
  - public tool factories now cover `web_search`, `x_search`, `code_execution`, `view_image`,
    `view_x_video`, `file_search`, and `mcp`
  - the public Rust xAI tool surface now also has typed arg structs plus factory-style helpers
    (`WebSearchArgs`, `XSearchArgs`, `FileSearchArgs`, `McpArgs`, `*_with(...)`,
    `mcp_server_with(...)`) instead of relying on raw `.with_args(json)` for the main audited
    provider-tool path
  - xAI Responses request conversion now maps those tool args to the audited snake_case wire
    shape (`allowed_domains`, `allowed_x_handles`, `vector_store_ids`, `server_url`, ...)
  - xAI server-side provider tools now drop invalid specific `tool_choice` forcing instead of
    serializing OpenAI-only builtin/function payloads
  - unknown xAI provider-defined tools are skipped instead of being forwarded blindly as raw
    `type`
  - xAI `custom_tool_call` SSE handling now follows the audited AI SDK semantics more closely:
    finalized `tool-input-*` plus `tool-call` emission for `x_search` / `view_x_video` is
    deferred to `response.output_item.done`, while `response.custom_tool_call_input.*` only
    buffers finalized input
- [x] Finish the xAI provider-owned non-chat parity sweep against
  `repo-ref/ai/packages/xai/src/xai-image-model.ts`,
  `repo-ref/ai/packages/xai/src/xai-video-model.ts`,
  `repo-ref/ai/packages/xai/src/xai-image-options.ts`, and
  `repo-ref/ai/packages/xai/src/xai-video-options.ts`.
  - typed `XaiImageOptions` / `XaiVideoOptions` plus request ext traits are now public on the
    provider-owned and facade surfaces
  - xAI image generation/edit now route through `/images/generations` and `/images/edits`
  - xAI video create/query now route through `/videos/generations|edits` and
    `GET /videos/{request_id}`
  - registry/native metadata/public-path parity now treat xAI image generation and video task
    APIs as first-class provider-owned features instead of intentional unsupported paths
  - shared `VideoGenerationRequest` / `VideoGenerationResponse` /
    `VideoTaskStatusResponse` now preserve AI SDK-style `providerOptions`, per-request
    `HttpConfig`, `aspectRatio`, `videoUrl`, metadata, warnings, and response envelopes
- [~] Close the remaining shared non-chat structure gaps after the xAI provider-owned parity pass.
  - [x] Refactor `ImageEditRequest` into a typed multi-input image surface closer to AI SDK
    `files[]` + `mask` semantics.
    - shared `ImageEditInput` / `ImageEditFileData` now model file-backed and URL-backed edit
      inputs
    - xAI image edit now maps 1 input -> `image` and many inputs -> `images`
    - OpenAI/OpenAI-compatible multipart and Vertex inline edit paths now accept multiple
      file-backed source images
  - [~] Decide how URL-backed image edit inputs should be materialized on multipart/inline
    provider paths.
    - xAI already accepts URL-backed edit inputs directly
    - OpenAI/OpenAI-compatible/Vertex currently reject URL-backed edit inputs until an async
      prefetch/materialization layer exists
  - [x] Stabilize the shared AI SDK-style video knob/input surface instead of leaving it as an
    open design question.
    - `VideoGenerationRequest` now carries canonical `count` (`n`), `fps`, `seed`, and typed
      `VideoGenerationInput` image/video inputs
    - the older raw `seed_image` / `seed_video` byte fields were removed during the refactor
    - xAI now warns-and-filters unsupported `n` / `fps` / `seed` knobs on the provider-owned path
    - Gemini/Veo now consumes canonical `count` / `seed` / typed image input and surfaces
      warnings for unsupported URL/FPS cases
    - MiniMaxi now filters unsupported generic/shared video fields instead of serializing the
      entire shared request object blindly
  - [x] Move the remaining MiniMaxi-centric top-level video request fields behind provider-owned
    typed options/builders.
    - shared `VideoGenerationRequest` no longer carries `prompt_optimizer`,
      `fast_pretreatment`, `callback_url`, or `aigc_watermark`
    - provider-owned `MinimaxiVideoOptions` now carries those knobs through
      `providerOptions["minimaxi"]`
    - MiniMaxi video request/body shaping and public provider-owned builder helpers were updated to
      use that provider-owned lane instead of shared top-level fields

## Track C - Finish V4-capable prompt/content modeling

- [x] Add V4 `custom` content parts.
- [x] Add V4 `reasoning-file` content parts.
- [x] Add explicit tool-result content variants:
  - `file-data`
  - `file-url`
  - `file-id`
  - `image-data`
  - `image-url`
  - `image-file-id`
  - `custom`
- [x] Add a stable provider-keyed tool-result file-id helper.
- [x] Add `providerOptions` support to tool-result output/content shapes where AI SDK prompt types
  allow them.
- [x] Review whether text/reasoning/tool-call/tool-result/source/tool-approval parts should expose
  both input-side `providerOptions` and output-side `providerMetadata`, or whether a narrower split
  is safer in Rust.
  - current recommendation: keep the shared stable content superset, but treat
    `providerOptions` as the canonical request-time channel and `providerMetadata` as
    response-time observation only
- [x] Audit protocol coverage after the new stable parts exist:
  - [x] OpenAI Responses request/gateway mapping for explicit tool-result file/image/id content
  - [x] Anthropic request/gateway mapping for explicit tool-result image/PDF/url content plus
    `tool_reference`
  - [x] Gemini request mapping for explicit `image-data` tool-result content and JSON fallback for
    unsupported variants
  - [x] OpenAI-compatible chat/tool-result parity review for the explicit variants
    - confirmed string-only tool-message degradation is required by the OpenAI-compatible wire
      contract, and explicit V4 content variants are preserved inside the JSON string payload
  - [x] Decide whether any additional providers should get true `reasoning-file` or `custom`
    support rather than explicit degradation
    - current audit conclusion: no additional audited provider should get native support here yet
    - Gemini remains the only true `reasoning-file` wire path
    - OpenAI Responses keeps provider-specific `custom` support for `openai.compaction`
    - Anthropic keeps `tool_reference`-style custom tool-result mapping
    - xAI and OpenAI-compatible continue explicit degradation/skip behavior for unsupported
      assistant-side `reasoning-file` / `custom`
- [x] Sweep public macros/tests/examples and provider helper matches after the stable
  `providerOptions` rollout so all-features builds keep compiling against the expanded stable
  content model.

## Track D - Strengthen the stable stream model

- [x] Decide the stream-part direction:
  - keep the historical `LanguageModelV3StreamPart` name for compatibility
  - upgrade its shape into a V4-capable superset instead of introducing a second primary type first
- [x] Document the intended relationship between:
  - `ChatStreamEvent`
  - typed stream parts
  - protocol-owned event/state machines
- [~] Add first-class stable stream support for semantics that were commonly pushed through
  `Custom`:
  - [x] source
  - [x] tool approval request
  - [x] file
  - [x] reasoning-file
  - [x] stream-start warnings
  - [x] response metadata
  - [x] custom content
  - [x] runtime `ChatStreamEvent` promotion beyond the thin transport layer via
    `Part(ChatStreamPart)`
- [~] Add adapters between the runtime stream event layer and the chosen V4-capable stream-part
  layer.
- [x] Expose AI SDK-aligned `LanguageModelV4*` public aliases for the upgraded typed stream-part
  overlay so new code no longer has to use the historical `LanguageModelV3*` names.
- [~] Ensure bridge/gateway serializers use that stronger stream-part contract where appropriate.
  - [x] extras Axum SSE adapters now surface direct runtime `Part` / `PartWithReplay` as explicit
    `event: part` frames instead of dropping the upgraded semantic lane
  - [x] extras high-level consumers (`stream_object`, tool-loop gateway) now consume stable tool
    lifecycle parts before falling back to legacy deltas, with source-aware deduplication for
    mixed streams
- [x] Normalize serializer re-entry so `Part -> Custom` compatibility bridging does not deadlock
  when protocol serializers hold internal state locks.
- [~] Migrate provider parsers to emit `ChatStreamEvent::Part` directly where the richer semantic
  contract is available at parse time.
  - [x] OpenAI Responses major stable semantics
  - [x] Anthropic `stream-start` / `response-metadata` / `text-*` / local `tool-input-*` /
    `tool-call` / `reasoning-*` / `source` / successful `finish`
  - [x] Gemini reasoning/source/provider-executed tool semantics
  - [x] OpenAI-compatible `stream-start` / `response-metadata` / `text-*` / `reasoning-*` /
    `finish` lifecycle semantics on the direct runtime `Part` lane
  - [x] Anthropic provider-hosted server tool / MCP stable-part strategy
  - [x] Anthropic reasoning signature/redacted cleanup on the stable part lane
    - `signature_delta` now maps to `reasoning-delta.providerMetadata.anthropic.signature`
    - `redacted_thinking` replay now uses `reasoning-start.providerMetadata.anthropic.redactedData`
    - no dedicated replay carrier was added because AI SDK already models these as stable metadata
  - [x] OpenAI Responses raw-item / output-index replay carrier
  - [x] Make OpenAI-compatible tool-call deltas emit stable `tool-input-*` / `tool-call` parts
    without duplicating legacy delta accumulation in `StreamProcessor`
    - parser now emits stable tool lifecycle parts before legacy shadow deltas
    - OpenAI-compatible chat serialization uses first-source-wins deduplication for mixed
      stable/legacy tool streams
    - `StreamProcessor`, tool-loop gateway, and `stream_object` all keep the stable tool call
      intact at final response/object assembly time
  - [x] Make OpenAI-compatible URL citations follow the stable `source` contract
    - streaming parser now maps chat-completions `delta.annotations[*].url_citation` to
      `ChatStreamPart::Source`
    - OpenAI-compatible SSE reserialization now maps stable URL `source` parts back to
      chat-completions `delta.annotations`
    - non-stream chat responses now map `message.annotations[*].url_citation` to
      `ContentPart::Source`

## Track E - Converge usage semantics

- [x] Introduce an AI-SDK-shaped stable usage view:
  - `inputTokens`
  - `outputTokens`
  - `raw`
- [x] Decide whether the current `Usage` becomes:
  - a compatibility wrapper
  - a compatibility alias layer
  - or a nested legacy view carried alongside the new shape
- [x] Migrate stream aggregation to fill the richer usage model.
- [x] Migrate protocol serializers/parsers to preserve raw usage under the richer stable model.
- [x] Add fixture/no-network coverage for usage detail preservation on:
  - OpenAI Responses
  - Anthropic Messages
  - Gemini GenerateContent
  - OpenAI-compatible streaming
- [x] Audit the currently migrated OpenAI/OpenAI-compatible/Anthropic/Gemini replay paths for
  unknown/null totals so they stop forcing provider-unknown usage into synthetic zero-valued legacy
  counts.
- [x] Align Gemini usage replay with AI SDK output accounting by treating
  `candidatesTokenCount + thoughtsTokenCount` as total output usage and preserving
  `cachedContentTokenCount` / `trafficType` during SSE round-trips.
- [x] Lock Anthropic `Usage.raw` to the stable provider-raw subset and keep full provider-native
  `usage` fidelity under `provider_metadata.anthropic.usage`, omitting absent optional raw fields
  instead of serializing `null` placeholders.
- [x] Make AI SDK-shaped usage canonical at the stable layer:
  - legacy `prompt/completion/total` counts are no longer public storage fields on `Usage`
  - compatibility callers use accessors/serde (`prompt_tokens()`, `completion_tokens()`,
    `total_tokens()`) or explicit constructors/builders

## Track F - Validation

- [x] Add regression coverage for terminal response-envelope preservation on streaming paths.
- [x] Add regression coverage for Anthropic extended usage preservation.
- [x] Add regression coverage for `Warning::Compatibility`.
- [x] Add regression coverage for widened `source` and `tool-approval-*` fields.
- [x] Add regression coverage for OpenAI Responses approval `reason` forwarding.
- [x] Add regression coverage for V4 `custom` / `reasoning-file` stable serialization.
- [x] Add regression coverage for explicit tool-result content serialization and protocol
  conversions.
- [x] Add regression coverage proving Anthropic `AsText` fallback still works when unsupported
  parts arrive on the direct runtime `Part/PartWithReplay` lane.
- [x] Refresh fixture-backed Anthropic/OpenAI transcoding assertions to the current stable typed
  semantics for hosted OpenAI tools (`mcp_tool_use` / `mcp_tool_result`) instead of older custom
  event expectations.
- [~] Add fixture-backed request tests for message-level and part-level `providerOptions`.
  - [x] OpenAI Chat fixture coverage now pins canonical part `providerOptions.openai.imageDetail`
    request input.
  - [x] OpenAI-compatible fixture coverage now pins canonical message/part/tool-result
    `providerOptions.openaiCompatible` request input.
- [x] Add file-id/image-file-id provider-roundtrip coverage where a provider has a native
  equivalent.
- [x] Refresh OpenAI Responses input/response fixtures to the canonical stable tool-result /
  warning / usage shapes so exact tests no longer depend on removed generic `file` tool-result
  parts or legacy `unsupported-setting` fixture baselines.
- [x] Lock OpenAI/Azure Responses non-stream `message.content[*].output_text` to AI SDK-style
  structured text parts on the stable response boundary.
  - plain and empty assistant message text now remain `ContentPart::Text` parts instead of
    collapsing to bare `MessageContent::Text`
  - typed `providerMetadata.{openai|azure}.itemId` is now preserved even without
    `phase` / `annotations`
  - the main OpenAI/Azure parsed response path no longer duplicates message `itemId` on
    top-level response `provider_metadata`
- [x] Preserve richer OpenAI Responses typed response metadata on exact non-stream and
  same-protocol replay paths.
  - response-level `responseId` / `serviceTier`
  - text-part `itemId` / `phase` / raw `annotations`
  - document-source citation `type` / `index`
- [x] Align xAI Responses response/stream metadata boundaries with the audited AI SDK xAI
  provider behavior.
  - non-stream text/source parts intentionally omit provider metadata
  - non-stream reasoning parts use `providerMetadata.xai.itemId`
  - xAI responses no longer emit top-level response `provider_metadata`
  - xAI streaming reasoning parts now carry `providerMetadata.xai.itemId`, while `text-*` /
    `finish` stay metadata-free and `output_item.done` backfills a missing `reasoning-start`
- [x] Align xAI request-side provider options and ids with the audited AI SDK split.
  - xAI Responses top-level options now cover `reasoningEffort` / `reasoningSummary`,
    `topLogprobs -> logprobs=true`, `previousResponseId`, and
    `store=false -> include += reasoning.encrypted_content`
  - assistant xAI message `providerOptions.xai.itemId` stays on the direct request item instead of
    collapsing into OpenAI `item_reference`
  - assistant xAI tool calls now emit stable `id`, `call_id`, and `status: "completed"`
  - xAI `/chat/completions` request normalization now keeps supported chat fields while stripping
    Responses-only knobs (`reasoningSummary`, `previousResponseId`, `include`, `store`)
  - typed request-side coverage now also composes `XaiChatOptions` + `XaiResponsesOptions` into
    one canonical `providerOptions.xai` object before transport normalization when both audited
    surfaces need to appear on the same request
- [x] Add same-protocol response/stream roundtrip tests for document-style `source` parts.
- [x] Add fixture-backed/public-path coverage for OpenAI-compatible URL citation `source` parts.
  - non-stream `message.annotations[*].url_citation -> ContentPart::Source` is now pinned by
    `openai-compatible-annotations-source.1`
  - same-protocol chat-completions bridge coverage now pins
    `delta.annotations[*].url_citation -> source(url) -> delta.annotations`
- [x] Add public-path parity tests covering the final request boundary:
  - builder
  - provider-owned client
  - config-first client
  - registry
  - OpenAI Chat now has builder/provider/config/registry parity coverage for canonical part
    `providerOptions.openai.imageDetail`, including a regression proving legacy request-side
    `providerMetadata.openai.imageDetail` is ignored.
  - OpenAI-compatible now has builder/provider/config/registry parity coverage for canonical
    message/part/tool-result `providerOptions.openaiCompatible`, including regressions proving
    request-side `providerMetadata.openaiCompatible` and
    `message.metadata.custom.openaiCompatible` do not participate in the main request path.

## Track G - Cleanup and migration notes

- [x] Update `docs/alignment/*` documents once the new stable contracts land.
- [x] Update the structural alignment workstream docs for the new stable content/tool-result state.
- [x] Update the structural alignment workstream docs for the runtime `ChatStreamEvent::Part`
  migration.
- [x] Add changelog notes for the current Unreleased changes.
- [x] Add migration notes if public request/response JSON changes in a user-visible way.
- [x] Expose stable runtime stream types (`ChatStreamPart`, `ChatStreamToolCall`, etc.) through
  the public streaming/prelude surface so downstream code no longer needs `__private::types`.
- [x] Remove or deprecate compatibility fallbacks after the new paths have enough test coverage.
- [~] Narrow the remaining OpenAI-compatible parity audit to explicit fixture/public-path holes.
  - `message.annotations` / `delta.annotations` URL citation parity is now fixture-backed
  - same-protocol chat-completions roundtrip fixtures now also pin `response-metadata` and
    streamed `logprobs` fidelity on the public path
  - the same public-path fixtures now also pin AI SDK-style
    `acceptedPredictionTokens` / `rejectedPredictionTokens` mirrored from
    `usage.completion_tokens_details`
  - the same public-path fixture now also pins terminal response-envelope fields such as
    `system_fingerprint` / `service_tier`
  - Azure model-router `prompt_filter_results` preludes with empty `id` / `model` and
    `created = 0` now defer `response-metadata` until a real metadata chunk arrives
  - the remaining OpenAI-compatible parity audit is now mostly rarer raw-hint cleanup rather than
    missing core source/metadata/logprobs/prediction-token fixtures

## Current branch notes

This branch now has a clearer baseline than the first draft of the workstream:

- shared compatibility warning support
- widened `source` support
- strict `source` URL/document union
- widened `tool-approval-*` support
- Anthropic/OpenAI-compatible stream-end fidelity fixes
- OpenAI Responses approval-reason forwarding
- first-class request-side `providerOptions`
- first-class V4 `custom` and `reasoning-file`
- explicit V4 tool-result content modeling
- a V4-capable typed stream-part overlay with `custom` / `reasoning-file`
- a first-class runtime `ChatStreamEvent::Part(ChatStreamPart)` semantic channel
- partial but explicit provider coverage for those new stable parts
- OpenAI-compatible stable tool lifecycle emission plus first-source-wins mixed-stream deduplication
- fixture-backed OpenAI-compatible URL citation/source coverage on both non-stream and
  same-protocol stream roundtrip paths
- public streaming/prelude exports for stable runtime stream types, removing the need for
  `__private::types` in the current high-level/gateway surfaces
- AI SDK-shaped `Usage` as the canonical storage layer, with legacy totals reduced to
  compatibility accessors/serde
