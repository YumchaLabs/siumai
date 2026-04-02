# AI SDK Structural Alignment - Milestones

Last updated: 2026-04-01

This workstream is tracked with explicit acceptance criteria.

## ASA-M0 - Scope and audit locked

Acceptance criteria:

- The workstream scope is explicitly documented.
- The main AI SDK references are listed.
- The current misalignment table is recorded with concrete repository anchors.

Status: completed

## ASA-M1 - Shared semantic gaps closed

Acceptance criteria:

- Shared warning semantics expose the required AI SDK categories.
- Shared `source` and `tool-approval-*` shapes can preserve the important missing fields.
- Recent stream-end fidelity fixes remain covered by tests.

Current state:

- `Warning::Compatibility` is present.
- `source.mediaType` / `filename` / `providerMetadata` are present.
- `tool-approval-response.reason`, `tool-approval-request.providerMetadata`, and part-level
  `providerOptions` are present.
- The shape is still not fully AI SDK V4-complete because some request-boundary cleanup remains.

Status: in progress

## ASA-M2 - Request/response provider boundary corrected

Acceptance criteria:

- Message and content-part request controls have first-class `providerOptions`.
- Request converters no longer need response-style `provider_metadata` for request-only behavior.
- Temporary metadata-based fallback paths are documented and bounded.

Current state:

- `ChatMessage`, request-capable content parts, and tool-result output/content parts expose
  first-class `providerOptions`.
- OpenAI-compatible, OpenAI Responses, and Anthropic request conversion prefer `providerOptions`
  and only fall back to legacy metadata shims for compatibility.
- OpenAI Chat and OpenAI-compatible request conversion now also agree on the main request-only
  boundary:
  - OpenAI Chat image detail reads only canonical part `providerOptions.openai|azure`
  - OpenAI-compatible extra request params read only canonical message/part/tool-result
    `providerOptions.openaiCompatible`
  - the shared `ProviderOptionsMap` serde now normalizes JSON provider ids, so external
    `openaiCompatible` request keys resolve the same way as builder-inserted provider options
- OpenAI Responses request conversion, warning snapshots, and request normalization now also agree
  on that split: reasoning/image-detail/approval-id request inputs use canonical
  `providerOptions`, and the only remaining OpenAI Responses request-side metadata fallback is the
  upstream-compatible assistant tool-call `itemId` shim.
- xAI provider-owned non-chat request/response boundaries now also match the audited AI SDK split
  much more closely:
  - typed `XaiImageOptions` / `XaiVideoOptions` now live on the public provider-owned/facade
    surface
  - native image generation/edit route through `/images/generations` and `/images/edits`
  - native video create/query route through `/videos/generations|edits` and
    `GET /videos/{request_id}`
  - registry/native metadata/public-path parity now expose xAI image generation and video task
    support as first-class provider-owned capabilities instead of intentional unsupported paths
- MiniMaxi video request shaping now follows that same provider-owned split too:
  - shared `VideoGenerationRequest` no longer carries MiniMaxi-only top-level knobs
  - typed `MinimaxiVideoOptions` now owns `prompt_optimizer`, `fast_pretreatment`,
    `callback_url`, and `aigc_watermark` under `providerOptions["minimaxi"]`
  - the provider-owned MiniMaxi video builder keeps matching fluent helpers while routing them
    through that namespaced option lane instead of the shared request type
- The xAI chat typed surface was re-audited against `repo-ref/ai/packages/xai/src/xai-chat-options.ts`:
  `parallel_function_calling` is now exposed end-to-end, deprecated `xHandles` now normalizes to
  wire `included_x_handles`, and `with_default_search()` now matches the upstream
  `maxSearchResults=20` default instead of keeping a stale local value.
- The xAI search-source data structure now also matches the upstream shape more closely:
  `SearchSource` was refactored from a single permissive struct into a discriminated union over
  `web` / `news` / `x` / `rss`, which removes invalid cross-source field combinations from the
  typed provider surface while preserving deprecated `xHandles -> included_x_handles` input
  normalization.
- The xAI typed provider surface is now split the same way as the audited AI SDK reference:
  `XaiChatOptions` carries chat-only knobs, `XaiResponsesOptions` carries Responses-only knobs,
  and the main reasoning/include slots now use enum-backed typed wrappers instead of raw
  `String` / `Vec<String>` bags.
- Anthropic document citations/title/context and per-part cache control no longer read
  request-time values from response-style file `provider_metadata` or legacy
  `message.metadata.custom["anthropic_*"]` shims; the message builder helpers now write the
  canonical part `providerOptions.anthropic` path only.
- Anthropic same-protocol reasoning replay now follows the same split as AI SDK:
  response parsing stores `signature` / `redactedData` on reasoning-part
  `providerMetadata.anthropic`, the provider helper translates that back onto next-turn
  reasoning-part `providerOptions.anthropic`, and the Anthropic prompt converter no longer needs
  message-level `metadata.custom["anthropic_*"]` thinking keys.
- The experimental request bridge now follows that same split for Anthropic/OpenAI reasoning
  replay, including direct-pair request serialization and normalization paths.
- Regression tests cover message-level and part-level precedence at the request boundary.
- Experimental bridge-side OpenAI reasoning eligibility checks now follow the same canonical split
  instead of treating reasoning-part `provider_metadata.openai` as a request-time replay signal.
- Anthropic Messages request normalization now also follows that split for document/cache-control
  request semantics: content-block cache control plus document citations/title/context are
  canonicalized directly onto part `providerOptions.anthropic`, and bridge-side cache-limit
  inspection reads those canonical part options instead of legacy `message.metadata.custom` maps.
- Experimental direct-pair reasoning replay now matches that same boundary too: Anthropic ->
  OpenAI and OpenAI -> Anthropic helpers plus bridge-side Anthropic reasoning inspection no longer
  accept request-side `providerMetadata` as replay input.
- Public-path parity now also pins that canonical boundary on the actual entrypoints:
  - OpenAI Chat builder/provider/config/registry paths agree on canonical part
    `providerOptions.openai.imageDetail`, and legacy request-side
    `providerMetadata.openai.imageDetail` no longer affects the emitted request body.
  - OpenAI-compatible builder/provider/config/registry paths agree on canonical
    message/part/tool-result `providerOptions.openaiCompatible`, and request-side
    `providerMetadata.openaiCompatible` / `message.metadata.custom.openaiCompatible` no longer
    affect the emitted request body.
- The prompt-boundary review is now documented: the shared stable content surface stays as a
  pragmatic superset for now, but `providerOptions` is the canonical request-time channel and
  `providerMetadata` remains response-time only in intent.
- A final audit/removal pass is still needed before this can be marked complete.

Status: in progress

## ASA-M3 - Stable prompt/content model reaches V4-capable parity

Acceptance criteria:

- Message-level `providerOptions` exist.
- Part-level `providerOptions` exist where AI SDK prompt/content parts support them.
- V4 `custom` and `reasoning-file` content parts exist as first-class stable types.
- Tool-result content models the explicit AI SDK file/image/id variants.
- `Source` has a final structural shape decision (strict union or equivalent-safe Rust design).

Current state:

- Message-level `providerOptions` are present on `ChatMessage`.
- Part-level `providerOptions` are present on request-capable stable content parts.
- Tool-result output/content provider options are modeled.
- V4 `custom` and V4 `reasoning-file` stable content parts are present.
- Explicit tool-result content variants are modeled with `ToolResultFileId`.
- Stable `SourcePart` now models a strict URL/document union.
- Shared non-chat request/response structs now also expose more of the AI SDK provider-owned shape
  needed by xAI image/video:
  - `ImageEditRequest` now exposes AI SDK-style typed `images[]` + `mask` semantics through
    shared `ImageEditInput` / `ImageEditFileData`
  - `VideoGenerationRequest` now includes canonical `providerOptions`, per-request `HttpConfig`,
    and `aspectRatio`
  - `VideoGenerationResponse` / `VideoTaskStatusResponse` now carry metadata, warnings, response
    envelopes, and direct `videoUrl` / `duration` fields where providers expose them
- Provider coverage for that new image-edit structure is now explicit:
  - xAI accepts file-backed and URL-backed edit inputs and maps single-vs-multi inputs to the
    audited `image` / `images` wire split
  - OpenAI/OpenAI-compatible multipart edit and Vertex inline edit now accept multiple
    file-backed inputs
  - URL-backed edit inputs are still intentionally rejected on those multipart/inline paths until
    an async materialization layer exists
- Protocol coverage is now explicit but still scoped by provider:
  - OpenAI Responses has true compaction and explicit tool-result file/image/id support.
  - OpenAI Responses fixture baselines now also distinguish canonical user-side `ContentPart::File`
    from tool-result-only `image-*` / `file-*` variants, so the removed generic `file`
    tool-result shape is no longer hiding inside request fixtures.
  - OpenAI/OpenAI-compatible chat/tool-message conversion has now been reviewed for the explicit
    V4 tool-result variants; because the wire contract is string-only there, those variants are
    preserved as JSON-string payloads rather than widened into a fake native shape.
  - OpenAI-compatible non-stream chat responses now also map `message.annotations.url_citation`
    into stable URL `source` parts, matching the upstream AI SDK chat response behavior.
  - Gemini has true `reasoning-file` support and explicit `image-data` tool-result handling.
  - Anthropic has explicit tool-result image/PDF/url handling plus `tool_reference`.

Status: in progress

## ASA-M4 - Stable stream model reaches V4-capable parity

Acceptance criteria:

- A V4-capable stable stream-part contract exists.
- Major AI SDK stream semantics are not forced through `Custom` by default.
- Adapters between runtime stream events and stable stream parts are covered by tests.
- Bridge/gateway serializers use the stable stream-part contract consistently.

Current state:

- The historical `LanguageModelV3StreamPart` overlay is now being treated as the compatibility
  carrier for a V4-capable typed stream-part contract.
- The upgraded overlay now also exposes public `LanguageModelV4*` aliases, so new code can use
  AI SDK-aligned names without losing compatibility with the historical V3 surface.
- That overlay now includes first-class V4 `custom` and `reasoning-file` parts in addition to the
  older text/reasoning/tool/source/finish shapes.
- `ChatStreamEvent` now exposes a first-class `Part(ChatStreamPart)` semantic channel so the
  runtime layer can represent AI SDK stream-part semantics without tunneling them through
  provider-scoped `Custom` events.
- `StreamProcessor` plus the OpenAI/OpenAI-compatible/Anthropic/Gemini serializers now bridge that
  runtime part channel instead of assuming only legacy transport events or custom payloads.
- OpenAI-compatible serializer coverage now proves those unsupported V4-only parts degrade
  explicitly via `AsText` instead of disappearing.
- OpenAI Responses and Anthropic serializers now normalize runtime `Part` events before locking
  protocol serialization state, so direct stable-part replay is no longer vulnerable to recursive
  lock re-entry hangs.
- Anthropic serializer-side fallback now also covers unsupported parts that already arrive on the
  direct `Part/PartWithReplay` lane, so lossy transcoding does not silently drop approval
  semantics just because the source parser emitted typed parts instead of legacy custom events.
- OpenAI Responses parser now emits the runtime part channel for stream-start/response-metadata,
  non-tool text/reasoning/source, and successful finish semantics.
- OpenAI Responses provider-hosted tool / MCP / approval replay now uses an explicit runtime replay
  carrier instead of leaving `rawItem` / `outputIndex` inside loose provider-scoped custom event
  payloads.
- Anthropic parser now emits the runtime part channel for stream-start/response-metadata,
  text/reasoning/source, provider-hosted server tool / MCP tool-call/tool-result semantics,
  standard local tool-input/tool-call, and successful finish semantics.
- Anthropic reasoning `signature_delta` / `redacted_thinking` now also align with AI SDK-style
  `reasoning-*` part `providerMetadata`, so same-protocol replay no longer depends on a bespoke
  `anthropic:thinking-signature-delta` custom event.
- Fixture-backed OpenAI -> Anthropic transcoding expectations are now aligned with the current
  stable replay semantics: hosted OpenAI web search and MCP tool executions surface as Anthropic
  `mcp_tool_use` / `mcp_tool_result`, while unsupported approval requests only degrade to text in
  lossy mode.
- Gemini parser now emits the runtime part channel for reasoning/source/provider-executed tool
  semantics and optional function-call tool-part mode.
- OpenAI-compatible chat chunk parsing now also emits runtime parts for `stream-start`,
  `response-metadata`, `text-*`, `reasoning-*`, URL `source`, and successful `finish` lifecycle semantics
  while intentionally keeping legacy content/thinking/tool-call deltas in parallel for
  compatibility.
- OpenAI-compatible tool-call parsing now also emits stable `tool-input-start` /
  `tool-input-delta` / `tool-input-end` / `tool-call` parts before the legacy shadow deltas, and
  OpenAI-compatible chat reserialization now applies first-source-wins deduplication so mixed
  stable/legacy tool streams do not duplicate argument accumulation.
- OpenAI-compatible chat reserialization now also maps stable URL `source` parts back into
  chat-completions `delta.annotations`, closing the same-protocol `source` roundtrip gap.
- OpenAI-compatible fixture/public-path coverage now also locks the main citation-facing paths:
  non-stream `message.annotations` response fixtures pin the AI SDK-aligned `text -> tool-call ->
  source(url)` ordering, and chat-completions same-protocol roundtrip tests pin
  `delta.annotations -> source(url) -> delta.annotations`.
- OpenAI-compatible same-protocol chat-completions roundtrip fixtures now also pin
  `response-metadata` and terminal streamed `logprobs` fidelity on the public path.
- Those same public-path roundtrip fixtures now also pin terminal response-envelope fields such as
  `system_fingerprint` / `service_tier`, and the bridge now prefers `StreamEnd` over earlier
  `finish` parts when the target protocol needs the richer envelope.
- OpenAI-compatible and OpenAI Chat response metadata extraction now also mirror
  `usage.completion_tokens_details.accepted_prediction_tokens` /
  `rejected_prediction_tokens` into AI SDK-style typed provider metadata
  (`acceptedPredictionTokens` / `rejectedPredictionTokens`) on both non-stream and stream-end
  public paths.
- `StreamProcessor` final response assembly now preserves stable streamed `tool-call` parts even
  when the same id also used the accumulated tool-input lane, closing the last direct consumer bug
  that blocked OpenAI-compatible tool lifecycle migration.
- The public facade now re-exports stable runtime stream types such as `ChatStreamPart` and
  `ChatStreamToolCall` through the normal streaming/prelude surface, so downstream code no longer
  has to reach into `__private::types` for the main stable stream contract.
- The remaining stream-model gap is now mostly cleanup/parity review and any future truly
  protocol-only replay details beyond the current OpenAI Responses raw-item carrier, plus the
  remaining OpenAI-compatible cleanup around rarer raw-hint edges rather than missing
  source/metadata/logprobs/prediction-token public-path fixtures.

Status: in progress

## ASA-M5 - Stable usage model converges with the AI SDK

Acceptance criteria:

- An AI-SDK-shaped usage view exists at the stable surface.
- Stream and non-stream response paths populate that richer usage view.
- Legacy totals-based usage callers still have a documented compatibility path.

Current state:

- `Usage` now exposes AI SDK-style `inputTokens` / `outputTokens` / `raw` and internally tracks
  whether legacy totals are actually known.
- Legacy `prompt/completion/total` counts are no longer public storage fields on `Usage`; stable
  callers now use compatibility accessors/serde or explicit builders/constructors.
- OpenAI Responses, OpenAI-compatible, Anthropic, and Gemini replay paths now preserve
  provider-unknown / `null` totals instead of forcing zero-valued legacy counts.
- OpenAI Responses exact response roundtrip fixtures now assert the canonical nested usage view
  plus provider-native `raw` usage instead of only relying on derived legacy totals in fixture
  files.
- Anthropic response parsing and fixtures now distinguish the stable `Usage.raw` subset from the
  full provider-native `provider_metadata.anthropic.usage` payload, and absent optional raw
  fields are omitted instead of serialized as `null`.
- Gemini usage replay now aligns output-total accounting with AI SDK expectations by treating
  `candidatesTokenCount + thoughtsTokenCount` as total completion usage and by preserving
  `cachedContentTokenCount` / `trafficType` across SSE round-trips.

Status: completed

## ASA-M6 - Provider and protocol migration completed

Acceptance criteria:

- OpenAI/OpenAI-compatible/Anthropic/Gemini request paths use the corrected request boundary.
- Typed provider helpers are aligned with the final stable slots.
- Fixture, public-path, and no-network tests cover the migrated behavior.
- Changelog and migration notes are prepared where public behavior changes materially.

Current state:

- The main request-boundary migrations and the stable content refactor are landed for
  OpenAI-compatible, OpenAI Responses, Anthropic, and Gemini.
- Provider coverage for the new stable parts is partially complete and documented.
- OpenAI Responses request fixtures now lock exact roundtrip behavior for tool-result
  `image-file-id` / `file-id`, and provider-keyed `ToolResultFileId` inputs are covered through
  request normalization so OpenAI-native `file_id` selection cannot regress silently.
- OpenAI Responses input and response fixture suites now pin the stable canonical request/result
  shapes directly, including explicit tool-result attachment variants, `unsupported { feature }`
  warnings, and nested/raw usage fields.
- OpenAI/Azure Responses non-stream response parsing now keeps every assistant `output_text` as a
  structured stable text part, including plain and empty message text, so AI SDK-style
  `providerMetadata.{openai|azure}.itemId` no longer disappears behind the single-text fast path,
  and the same parsed response path no longer duplicates that message item id at the top-level
  response provider metadata layer.
- OpenAI Responses typed response metadata now also preserves `responseId` / `serviceTier` at the
  response level, `itemId` / `phase` / raw `annotations` on text parts, and document citation
  `type` / `index` on sources across non-stream JSON and same-protocol SSE roundtrips.
- xAI Responses response/stream metadata now also follows the audited AI SDK provider split:
  non-stream text/source parts stay metadata-free, reasoning parts use
  `providerMetadata.xai.itemId`, the parsed response no longer emits top-level
  `provider_metadata`, and xAI SSE now keeps `providerMetadata.xai.itemId` only on
  `reasoning-*` while leaving `text-*` / `finish` metadata-free.
- xAI request-side parity now also covers the audited AI SDK request knobs and ids:
  assistant xAI messages/tool calls stay as direct request items instead of OpenAI
  `item_reference`s, tool calls now emit stable `id + call_id + status`, top-level
  `reasoning.{effort,summary}` / `top_logprobs` / `previous_response_id` /
  `store=false -> include += reasoning.encrypted_content` are mapped on the shared
  Responses request transformer, and the xAI `/chat/completions` path now strips those
  Responses-only knobs back out after normalizing the supported chat fields.
- The xAI Responses provider-tool surface was also re-audited against
  `repo-ref/ai/packages/xai/src/tool/*` and `responses/xai-responses-prepare-tools.ts`:
  public Rust tool factories now cover `web_search`, `x_search`, `code_execution`,
  `view_image`, `view_x_video`, `file_search`, and `mcp`; request serialization now maps
  their args to the audited snake_case wire shape; unknown xAI provider-defined tools are
  skipped instead of being forwarded as raw `type`; specific `tool_choice` forcing is now
  dropped for xAI server-side provider tools instead of emitting invalid Responses payloads;
  the public Rust surface now also exposes typed xAI tool args/builders
  (`WebSearchArgs`, `XSearchArgs`, `FileSearchArgs`, `McpArgs`, `*_with(...)`);
  and xAI `custom_tool_call` SSE emission now mirrors the audited AI SDK flow by buffering
  `response.custom_tool_call_input.*` and emitting finalized `tool-input-*` plus `tool-call`
  at `response.output_item.done`.
- Same-protocol document-source roundtrip coverage is now present through the OpenAI Responses
  response and stream fixture suites (`file-search-tool.*`, `code-interpreter-tool.1`), so the
  strict `Url | Document` source union is exercised beyond static type-only tests.
- OpenAI-compatible exact fixtures now also lock URL citation/source parity on the public response
  path (`message.annotations -> ContentPart::Source`) and on the same-protocol chat-completions
  bridge path (`delta.annotations -> source(url) -> delta.annotations`).
- OpenAI-compatible same-protocol chat-completions roundtrip fixtures now also lock public-path
  `response-metadata`, streamed terminal `logprobs`, and terminal response-envelope
  `system_fingerprint` / `service_tier` fidelity.
- OpenAI typed provider helpers now also expose AI SDK-style
  `acceptedPredictionTokens` / `rejectedPredictionTokens` mirrored from chat
  `completion_tokens_details`, with non-stream, stream-end, and same-protocol roundtrip
  regression coverage.
- Public macros/examples/tests and provider helper surfaces have been swept so the message-level
  `providerOptions` rollout and new stable `reasoning-file` / `custom` parts no longer break
  all-features compilation.
- OpenAI Responses, Anthropic, Gemini, and OpenAI-compatible parser-side stable stream-part
  migration is now landed for the major AI SDK-stable semantics, with the remaining stream work
  mostly concentrated in OpenAI-compatible tool-call edges, provider-hosted replay edges, and
  protocol-only raw carriers.
- The xAI provider-owned image/video surface is now aligned on the public/runtime boundary, so
  the remaining non-chat work is shared-abstraction cleanup rather than provider routing:
  - `ImageEditRequest` now models richer AI SDK-style multi-input arrays, but URL-backed edit
    inputs are still provider-conditional because multipart/inline paths do not yet have a shared
    async materialization layer
  - the shared video request type now has a stabilized AI SDK-style shape instead of leaving
    `fps` / `seed` / `n` undecided:
    - raw `seed_image` / `seed_video` bytes were replaced with typed `VideoGenerationInput`
      file/url inputs
    - canonical `count` (`n`), `fps`, and `seed` knobs now live on the shared request type
    - xAI/Gemini/MiniMaxi provider-owned paths were updated so those shared fields are either
      mapped or intentionally warned-and-filtered instead of leaking through ad hoc serialization
  - the remaining video cleanup question is narrower now: whether the leftover MiniMaxi-specific
    top-level fields should be moved fully behind provider-owned typed surfaces
- Changelog/workstream notes plus a focused migration note for the `Usage` canonicalization are
  now present.
- Final public-path parity and cleanup work remain open.

Status: in progress
