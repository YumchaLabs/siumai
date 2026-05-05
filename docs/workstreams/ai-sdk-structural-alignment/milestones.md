# AI SDK Structural Alignment - Milestones

Last updated: 2026-04-22

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
- The shared prompt/content boundary now also carries first-class provider-owned file/image
  references through `ProviderReference`, so the last major V4 prompt-shape gap in this milestone
  is closed.
- The audited OpenAI-compatible stream lane now also keeps AI SDK-style semantic error terminals:
  explicit top-level error payloads and invalid JSON chunks emit stable `error` plus error
  `finish` / `StreamEnd`, and the shared regression suite also locks finish-time prediction-token
  provider metadata plus metadata-extractor merging on that streaming path.

Status: completed

## ASA-M2 - Request/response provider boundary corrected

Acceptance criteria:

- Message and content-part request controls have first-class `providerOptions`.
- Request converters no longer need response-style `provider_metadata` for request-only behavior.
- Temporary metadata-based fallback paths are removed from the audited request paths or left
  outside the canonical boundary as explicit compatibility surfaces.

Current state:

- `ChatMessage`, request-capable content parts, and tool-result output/content parts expose
  first-class `providerOptions`.
- OpenAI-compatible, OpenAI Responses, and Anthropic request conversion now use canonical
  `providerOptions` on the audited request paths instead of response-style metadata shims.
- OpenAI Chat and OpenAI-compatible request conversion now also agree on the main request-only
  boundary:
  - OpenAI Chat image detail reads only canonical part `providerOptions.openai|azure`
  - OpenAI-compatible extra request params read only canonical message/part/tool-result
    `providerOptions.openaiCompatible`
  - the shared `ProviderOptionsMap` serde now normalizes JSON provider ids, so external
    `openaiCompatible` request keys resolve the same way as builder-inserted provider options
- OpenAI Responses request conversion, warning snapshots, and request normalization now also agree
  on that split: reasoning/image-detail/approval-id request inputs and assistant tool-call ids all
  use canonical `providerOptions`, so the main OpenAI Responses request path no longer reads
  request-side `provider_metadata`.
- OpenAI Responses exact request/response parity now also locks the audited stable tool shape more
  tightly:
  - legacy compat `function_call` cases preserve provider-native `raw_finish_reason`
  - provider-executed Responses tool calls/results preserve stable `dynamic` plus tool-result
    `input`
  - hosted dynamic `local_shell` / `shell` / `apply_patch` items now serialize back to native
    Responses item types on the response bridge path instead of generic function-call fallbacks
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
- Anthropic custom provider ids now also match the audited AI SDK provider boundary more closely:
  request shaping merges canonical `providerOptions.anthropic` with provider-owned custom keys,
  custom keys override canonical fields when both are present, and top-level non-stream / finish /
  stream-end `providerMetadata` duplicates onto the custom provider root only when that custom key
  was actually used by the request.
- DeepSeek custom provider ids now match that same audited provider-boundary rule on the main
  chat path: request shaping reads provider-owned options from the runtime namespace instead of
  hardcoded `deepseek`, response metadata stays under that resolved root, and typed helpers now
  expose keyed accessors for explicit custom-root reads.
- Gemini request shaping now also mirrors the upstream `google|vertex` precedence fix: request
  provider options and `thoughtSignature` replay use the runtime namespace first and only then
  fall back to the canonical sibling key.
- OpenAI-compatible response parity now also has explicit fixture/public-path anchors for Gemini
  thought signatures: finalized compat tool calls preserve
  `extra_content.google.thought_signature` as `providerMetadata.{provider}.thoughtSignature` on
  both the direct fixture harness and the no-network OpenRouter public path, while lower-level
  compat tests still lock the requested camelCase metadata-key variant separately.
- The experimental request bridge now follows that same split for Anthropic/OpenAI reasoning
  replay, including direct-pair request serialization and normalization paths.
- Regression tests cover message-level and part-level precedence at the request boundary.
- Experimental bridge-side OpenAI reasoning eligibility checks now follow the same canonical split
  instead of treating reasoning-part `provider_metadata.openai` as a request-time replay signal.
- Anthropic Messages request normalization now also follows that split for document/cache-control
  request semantics: content-block cache control plus document citations/title/context are
  canonicalized directly onto part `providerOptions.anthropic`, and bridge-side cache-limit
  inspection reads those canonical part options instead of legacy `message.metadata.custom` maps.
- Anthropic Messages reverse normalization now also restores the audited AI SDK request-side
  shapes instead of leaving raw wire keys behind: request-level `thinking.budgetTokens`,
  `cacheControl`, `metadata.userId`, `mcpServers`, `container`, `contextManagement`, and `speed`
  now round-trip back onto canonical `providerOptions.anthropic`, and provider-defined tool args
  such as `maxUses` / `userLocation` no longer stay in wire snake_case after normalization.
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
- The remaining OpenAI-family request-edge drift around audio/compat/Responses conversion is now
  also closed on the audited paths:
  - shared `TtsRequest.language` now lowers into the OpenAI-family JSON speech body when the
    provider defaults explicitly opt into it, instead of remaining a warning-only dead field
  - built-in compat `groq` now defaults `supportsStructuredOutputs = true`, matching the audited
    AI SDK package policy for Groq JSON Schema requests
  - OpenAI Responses tool-role messages that only contain intentionally skipped parts are omitted
    from request conversion instead of failing with `Tool message missing tool result`
  - OpenAI typed control options now also expose the audited AI SDK fields more completely:
    chat options now include `systemMessageMode`, Responses options now include
    `systemMessageMode`, `forceReasoning`, and `contextManagement`, those control fields stay off
    the wire unless they affect prompt conversion, and
    `contextManagement[].compactThreshold` now lowers correctly onto the native
    `context_management[].compact_threshold` request shape
- The prompt-boundary review is now documented: the shared stable content surface stays as a
  pragmatic superset for now, but `providerOptions` is the canonical request-time channel and
  `providerMetadata` remains response-time only in intent.
- The shared response-side metadata root is now also explicit:
  - `siumai-spec::types::ProviderMetadataMap` provides the AI SDK-style `provider_id -> object`
    contract shared by `ChatResponse`, `CompletionResponse`, content parts, stream parts, and the
    high-level file/skill upload results
  - OpenAI/OpenAI-compatible/Anthropic typed accessors now read provider-owned objects from that
    shared root instead of each lane inventing its own nested map shape
  - UI `providerMetadata` intentionally remains on `ProviderOptionsMap` because upstream
    `convertToModelMessages()` forwards it as request-time `providerOptions`
- DeepInfra lower-contract URL handling now also matches the audited AI SDK single-provider
  surface more closely on the public/config/provider-owned text lanes:
  - shared compat config normalizes root, `/openai`, and `/inference` custom base URLs onto the
    canonical `/openai` text-family prefix
  - provider-owned builder/config and top-level builder/provider/registry stream paths now emit
    equivalent `/openai/chat/completions` requests for the same root base URL input
  - the public-path stream audit now also pins that `includeRawChunks` stays runtime-only and
    that finish-time `metadataExtractor` merging survives on those DeepInfra lanes
- DeepInfra's typed package surface is now also slightly more complete at the model-id layer:
  provider-owned and top-level public facades expose
  `DeepInfraChatModelId` / `DeepInfraCompletionModelId` /
  `DeepInfraEmbeddingModelId` / `DeepInfraImageModelId` as explicit aliases instead of forcing
  DeepInfra callers back onto the generic compat model-id names only.
- DeepInfra's package-level entry surface is now also clearer:
  `provider_ext::deepinfra` exposes `deepinfra()` and `create_deepinfra()` as unified-provider
  builder helpers, and the module docs now call out that `DeepInfraClient` / `DeepInfraConfig`
  are lower-level compat text-family aliases rather than the complete hybrid provider entrypoint.
- The audited AI SDK package facades now follow a consistent entry-helper rule:
  compat-promoted wrappers `provider_ext::{mistral,perplexity,fireworks,moonshotai}` and the
  audited provider-owned facades (`openai`, `anthropic`, `azure`, `google`, `bedrock`, `cohere`,
  `togetherai`, `google_vertex::vertex`, `groq`, `xai`, `deepseek`) all expose package-level
  `provider()` plus `create_provider()` builder helpers instead of forcing callers back to root
  namespaces, while DeepInfra already follows the same rule on its hybrid wrapper surface. The one
  intentional non-match is the generic `openai_compatible` package: upstream exports a settings-
  driven factory (`createOpenAICompatible(settings)`), so Rust keeps that slice on the lower-level
  `OpenAiCompatibleBuilder` / `OpenAiCompatibleConfig` / `OpenAiCompatibleClient::from_config(...)`
  boundary instead of inventing a misleading zero-arg facade helper.
- Provider feature-gated audit now also passes across the audited built-in/provider-compatible
  crates:
  - xAI file/video tests were refreshed to the current shared upload/video request structs
  - MiniMaxi tests now preserve the intentional `video prompt = Option<String>` /
    `music prompt = String` split
  - Azure native completion metadata now fills the shared `ResponseMetadata.headers` field instead
    of instantiating the older struct shape
- Protocol/top-level `all-features` audit now also passes on the audited compilation surfaces:
  - `siumai-core`, `siumai-protocol-openai`, `siumai-protocol-anthropic`,
    `siumai-protocol-gemini`, and top-level `siumai` compile on their real feature combinations
  - stale Anthropic protocol streaming fixtures plus top-level experimental-bridge/transcoding
    tests now instantiate the newer shared `ResponseMetadata.headers` field
  - top-level MiniMaxi/public retry file-upload tests now also follow the shared
    `filename: Option<String>` contract
- Native OpenAI / Azure / Bedrock package-level provider settings are now also explicitly aligned
  on the honest Rust boundary:
  - provider-owned/public facades expose
    `OpenAIProviderSettings`, `AzureOpenAIProviderSettings`, and
    `AmazonBedrockProviderSettings`
  - the same package boundaries now also expose `VERSION`
  - unsupported upstream fields such as OpenAI `name` and Bedrock credential-provider inputs are
    tracked explicitly under `docs/workstreams/provider-settings-surface-alignment/` instead of
    being faked on the public Rust surface
- Native Cohere package-level provider settings now follow the same rule:
  - provider-owned/public facades expose `CohereProviderSettings` plus `VERSION`
  - supported `baseURL` / `apiKey` / `headers` / `fetch` inputs are backed by real builder/config
    behavior
  - upstream `generateId` is tracked as deferred until the runtime owns a comparable hook
- Native DeepSeek and TogetherAI package-level provider settings now follow the same rule:
  - provider-owned/public facades expose `DeepSeekProviderSettings` and
    `TogetherAIProviderSettings` plus `VERSION`
  - supported `apiKey` / `baseURL` / `headers` / `fetch` inputs are backed by real builder/config
    behavior
Status: completed

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
- Stable user `image` / `file` parts now also model provider-owned file references directly
  through `FilePartSource::ProviderReference` and shared `ProviderReference` helpers/builders.
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
- Protocol coverage is now explicit and the main provider-reference request lanes are audited:
  - OpenAI Chat maps provider-owned user `image` / `file` prompt parts onto native `file_id`
    request payloads and OpenAI-compatible explicitly rejects those unsupported references.
  - OpenAI Responses maps provider-owned user `image` / `file` prompt parts onto native
    `input_image.file_id` / `input_file.file_id`, while normalization converts wire `file_id`
    input items back into canonical provider references.
  - Anthropic Messages and cache conversion map provider-owned user image/document parts onto
    `source: { type: "file", file_id }` and automatically inject the required `files-api` beta.
  - Bedrock explicitly rejects provider-owned file references on the request path instead of
    silently degrading them.
- Broader protocol coverage remains scoped by provider:
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
- The image architecture rule is now explicit too: the audited AI SDK surface does expose a shared
  `ProviderV4.imageModel(...)` + `ImageModelV4` + `generateImage(...)` contract, but that is only
  a stable call/result boundary. Hybrid providers are still expected to keep provider-owned image
  runtimes where upstream packages do so, instead of forcing those routes back into one generic
  OpenAI-compatible image executor.

Status: completed

## ASA-M4 - Stable stream model reaches V4-capable parity

Acceptance criteria:

- A V4-capable stable stream-part contract exists.
- Major AI SDK stream semantics are not forced through `Custom` by default.
- Adapters between runtime stream events and stable stream parts are covered by tests.
- Bridge/gateway serializers use the stable stream-part contract consistently.

Current state:

- The historical `TypedStreamPart` overlay is now being treated as the compatibility
  carrier for a V4-capable typed stream-part contract.
- The upgraded overlay now also exposes public `LanguageModelV4*` aliases, so new code can use
  AI SDK-aligned names without losing compatibility with the historical V3 surface.
- That overlay now includes first-class V4 `custom` and `reasoning-file` parts in addition to the
  older text/reasoning/tool/source/finish shapes.
- `ChatStreamEvent` now exposes a first-class `Part(ChatStreamPart)` semantic channel so the
  runtime layer can represent AI SDK stream-part semantics without tunneling them through
  provider-scoped `Custom` events.
- `TypedStreamPart::{from_runtime_part,to_runtime_part}` are now both public and
  covered by roundtrip tests, so bridge code can switch between the runtime semantic carrier and
  the typed V4-capable overlay without re-entering through custom-event JSON.
- `StreamProcessor` plus the OpenAI/OpenAI-compatible/Anthropic/Gemini serializers now bridge that
  runtime part channel instead of assuming only legacy transport events or custom payloads.
- OpenAI-compatible serializer coverage now proves those unsupported V4-only parts degrade
  explicitly via `AsText` instead of disappearing.
- OpenAI Responses and Anthropic serializers now normalize runtime `Part` events before locking
  protocol serialization state, so direct stable-part replay is no longer vulnerable to recursive
  lock re-entry hangs.
- Experimental bridge primitive remappers now also rewrite tool ids/names on direct
  `Part/PartWithReplay` events, and stale OpenAI Responses `rawItem` replay payloads are dropped
  after semantic remaps so gateway serialization cannot observe mismatched stable-vs-replay tool
  identity.
- Anthropic serializer-side fallback now also covers unsupported parts that already arrive on the
  direct `Part/PartWithReplay` lane, so lossy transcoding does not silently drop approval
  semantics just because the source parser emitted typed parts instead of legacy custom events.
- OpenAI Responses parser now emits the runtime part channel for stream-start/response-metadata,
  non-tool text/reasoning, direct `response.custom_tool_call_input.* -> tool-input-*`,
  `source` including web-search citations, and both successful and failed finish semantics.
- OpenAI Responses fixture parity was re-audited after the shared upload/metadata refactors:
  provider-executed MCP approval responses now require explicit `providerExecuted: true` in the
  local request fixtures to match upstream AI SDK request conversion, and the full
  `openai_responses_*` nextest suite is green again on that contract.
- Shared `FinishReason` serde now mirrors AI SDK string-union behavior even for unknown/custom
  reasons: `FinishReason::Other("other")` serializes as plain `"other"` (while legacy object
  input still deserializes), so failed OpenAI Responses streams again emit the stable
  `finishReason.unified = "other"` shape instead of a Rust enum object.
- Provider-facing OpenAI Responses, Anthropic, and Gemini stream helpers now consume stable
  `Part` / `PartWithReplay` events first and keep legacy custom-event parsing only as a backward
  compatibility fallback.
- The shared OpenAI Responses bridge now upgrades more of that legacy/custom typed payload family directly
  onto the runtime semantic lane too: `raw`, `custom`, `file`, and `reasoning-file` custom
  payloads no longer have to stay provider-prefixed `Custom` events once their stable meaning is
  already known.
- The typed stream-part overlay now makes the serializer-only downgrade boundary explicit too:
  `to_protocol_custom_event(...)` is the canonical provider-wire lowering API and the older
  `to_custom_event(...)` alias has been removed.
- Gemini parser-side text/reasoning streaming now stays on the runtime semantic lane: stable
  `stream-start`, `text-*`, `reasoning-*`, `file` / `reasoning-file`, and successful `finish`
  parts are emitted directly, shared shadow expansion has been removed, and the older
  `gemini:reasoning` custom shadow is no longer emitted from the audited parser.
- The shared OpenAI Responses bridge no longer carries bespoke `gemini:*` / `anthropic:*`
  event-type upgrade branches once those parser-era shadows disappeared from the audited mainline
  protocol paths; stable-shape custom payloads still upgrade through the generic typed parser.
- OpenAI Responses provider-hosted tool / MCP / approval replay now uses an explicit runtime replay
  carrier instead of leaving `rawItem` / `outputIndex` inside loose provider-scoped custom event
  payloads.
- Anthropic parser now emits the runtime part channel for stream-start/response-metadata,
  text/reasoning/source, provider-hosted server tool / MCP tool-call/tool-result semantics,
  standard local tool-input/tool-call, and successful finish semantics.
- Anthropic reasoning `signature_delta` / `redacted_thinking` now also align with AI SDK-style
  `reasoning-*` part `providerMetadata`, so same-protocol replay no longer depends on a bespoke
  `anthropic:thinking-signature-delta` custom event.
- Anthropic latest provider-defined tool parity is now covered on that same audited stream/runtime
  path: `web_search_20260209`, `web_fetch_20260209`, `code_execution_20260120`, and
  `computer_20251124` exist on the shared/provider surfaces, request headers inject the audited
  beta tokens, and the 2026 web-tool path now marks implicit provider-executed
  `code_execution` calls as `dynamic` when AI SDK would inject them automatically.
- Fixture-backed OpenAI -> Anthropic transcoding expectations are now aligned with the current
  stable replay semantics: hosted OpenAI web search and MCP tool executions surface as Anthropic
  `mcp_tool_use` / `mcp_tool_result`, while unsupported approval requests only degrade to text in
  lossy mode.
- Gemini parser now emits the runtime part channel for `stream-start`, runtime-opt-in `raw`,
  `text-*`, `reasoning-*`, `file` / `reasoning-file`, `source`, provider-executed tool semantics,
  top-level `error`, successful `finish`, and optional function-call tool-part mode.
- Gemini request-aware stream transformers now also forward
  `ChatRequest.stream_options.include_raw_chunks`, top-level `{"error": ...}` and invalid JSON
  chunks no longer skip stable error semantics, and EOF fallback now closes active text/reasoning
  lanes before emitting `finish(unknown)` plus `StreamEnd`.
- OpenAI-compatible chat chunk parsing now also emits runtime parts for `stream-start`,
  `response-metadata`, `text-*`, `reasoning-*`, URL `source`, and successful `finish` lifecycle semantics
  while intentionally keeping legacy content/thinking/tool-call deltas in parallel for
  compatibility.
- OpenAI-compatible tool-call parsing now also emits stable `tool-input-start` /
  `tool-input-delta` / `tool-input-end` / `tool-call` parts before the legacy shadow deltas, and
  OpenAI-compatible chat reserialization now applies first-source-wins deduplication so mixed
  stable/legacy tool streams do not duplicate argument accumulation.
- OpenAI-compatible streaming now also closes two audited AI SDK parity gaps on the shared parser:
  same-chunk reasoning opens/emits before text on the stable part lane, and explicit
  `finish_reason = "tool_calls"` chunks finalize pending stable `tool-input-end` / `tool-call`
  parts including empty-input tool calls without duplicating already-completed tool calls on later
  empty chunks.
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
- JSON-stream executor end-event synthesis now happens at real EOF instead of stream construction
  time, so stateful converters keep accumulated terminal response content on clean shutdown.
  Bedrock clean-EOF reserved-JSON extraction is now covered at the core executor layer, the
  provider-owned Bedrock converter layer, and the public-path structured-output lane.
- DeepSeek provider-owned streaming now matches the audited `@ai-sdk/deepseek` contract on the
  public Rust surface too: builder/config/registry/unified entrypoints all keep the provider
  package chat-only boundary, and native DeepSeek chat streams now always send
  `stream_options.include_usage = true`.
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
- Anthropic response parsing and fixtures now keep the full provider-native usage object on both
  `Usage.raw` and `provider_metadata.anthropic.usage`, while still deriving the stable token
  breakdown from known Anthropic usage fields.
- Gemini usage replay now aligns output-total accounting with AI SDK expectations by treating
  `candidatesTokenCount + thoughtsTokenCount` as total completion usage and by preserving
  `cachedContentTokenCount` / `trafficType` across SSE round-trips.
- High-level extras usage/completion surfaces now also align more closely with AI SDK:
  `Usage.merge()` and orchestrator `total_usage` aggregation drop provider-native `raw` on
  aggregation, `on_finish` now receives an explicit finish event, `StepResult` now carries
  stable `call_id`, stable `model { provider, model_id }`, `step_number`, unified `content`,
  step-scoped `request` / `response`, telemetry `function_id` / `metadata`, and stable
  `raw_finish_reason`, and `StreamOrchestration` resolves aggregated `total_usage` instead of
  leaving basic stream callers with empty steps only.
- The shared/provider response layer now also exposes the raw finish-reason chain needed by that
  extras surface: `ChatResponse.raw_finish_reason` exists on the stable type, shared
  OpenAI-compatible chat/stream plus OpenAI Responses decoding propagate raw finish causes where
  available, native Bedrock/Cohere keep their raw stop reasons, and audited
  OpenAI/OpenAI-compatible/Azure completion streams preserve raw finish reasons in their terminal
  `ChatResponse`.
- The extras orchestration family now requires `LanguageModel` instead of bare `ChatCapability`,
  which lets the public step/result surface expose stable model identity directly from the bound
  model contract instead of attempting provider/model inference from response metadata.
- Extras orchestration now also has one canonical AI SDK-style runtime `context` lane:
  `OrchestratorOptions` / `OrchestratorStreamOptions` accept initial context,
  `PrepareStepContext.context` can inspect it, `PrepareStepResult.context` can replace it,
  `ToolResolver::{call_tool_with_context, call_tool_stream_with_context}` expose
  backward-compatible context-aware tool hooks, `StepResult.context` plus
  `OrchestratorFinishEvent.context` surface the resolved step/final state, and the stream path
  now applies `prepare_step` / `tool_choice` / `active_tools` / `context` on the first streamed
  step as well.
- Extras prepare-step control now also matches the audited AI SDK shape more closely:
  `PrepareStepContext.model` exposes the base model, `PrepareStepResult::with_model(...)`
  can swap the `LanguageModel` for an individual step on both non-stream and stream paths, and
  later steps fall back to the base model again unless they are explicitly overridden.
- `StepResult` now also exposes more AI-SDK-like normalized derived views:
  `text()` concatenates all top-level text parts from unified step content, and standardized
  `tool_call_views()` / `tool_result_views()` plus static/dynamic splits now exist with resolved
  tool inputs for results.
- Streamed extras orchestration now also evaluates custom stop conditions after each step, matching
  the non-stream loop more closely instead of silently dropping builder/agent stop rules on the
  spawned streaming path.

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
- The higher-level AI SDK `uploadFile` helper gap is now much narrower on the public Rust surface:
  `siumai::files::upload(...)` exists with canonical `UploadFileOptions` / `UploadFileResult`,
  direct shared `DataContent` plus byte/string inputs, auto request-side media-type detection, URL
  rejection, stable `providerReference` results, built-in adapters for the current file-capable
  unified/provider clients, and canonical `providerOptions` on both the high-level helper and
  shared `FileUploadRequest`. Shared file upload/result filenames are now optional, missing
  filenames are no longer normalized to `blob`, helper `filename` / `mediaType` are no longer
  backfilled from request-time fallbacks, and helper `providerMetadata` now stays provider-owned
  instead of injecting generic file bookkeeping. OpenAI/Azure honor provider-scoped `purpose` /
  `expiresAfter`, Gemini honors `displayName` plus polling controls through provider options, and
  Anthropic explicitly warns when extra upload options are ignored on the current beta files path.
  The provider-owned Anthropic `files()` resource now also reuses the shared
  `FileManagementCapability` contract for upload/list/retrieve/delete/content directly, so the
  high-level helper no longer needs an Anthropic-only upload adapter and the redundant provider-
  local file wrapper layer is gone.
- The higher-level AI SDK `uploadSkill` helper gap is now much narrower on the public Rust
  surface: `siumai::skills::upload(...)` exists with canonical `UploadSkillFile` /
  `UploadSkillOptions` / `UploadSkillResult`, provider-owned OpenAI/Anthropic `skills()`
  resources, a shared `SkillsCapability` interface, AI SDK-style `providerReference` /
  `providerMetadata` results, Anthropic latest-version metadata fetch alignment, public
  regression coverage across provider, registry, and facade entrypoints, and the provider-owned
  OpenAI/Anthropic resources now also reuse the shared request/result structs directly instead of
  exposing parallel provider-local wrapper file/result types.
- The higher-level AI SDK UI-message helper gap is now also closed at the structural layer on the
  public Rust surface:
  - shared `UiMessage` / `UiMessagePart` / `UiToolPart` types now live on the stable type layer
  - `siumai-core::ui` now exposes `validate_ui_messages`, `convert_to_model_messages`, and
    `convert_to_chat_request`
  - `siumai::ui` now re-exports those helpers, and the historical `siumai::types::*` import path
    is restored for the shared data structures
  - conversion now preserves the main AI SDK semantics audited in this pass, including
    system-text compaction, user file/provider-reference mapping, assistant `step-start` block
    splitting, data-part callback conversion, incomplete-tool filtering, and stable
    `tool-approval-response.providerExecuted`
- Shared provider metadata is now also structurally aligned with the audited AI SDK references:
  `ProviderMetadataMap` is the single response-side root map for chat/completion/content/stream/
  file/skill results, and the remaining intentional UI exception is documented explicitly because
  upstream `convertToModelMessages()` treats UI `providerMetadata` as request-side
  `providerOptions`.
- The broader provider/helper surface has now been swept to that same root as well:
  Gemini/Vertex, Azure completion, Bedrock, Ollama, Anthropic skills/stream metadata, DeepSeek,
  Groq, xAI, and MiniMaxi typed metadata helpers all read/write provider-rooted object payloads,
  and the audited `cargo check -p siumai --all-features` verification lane is green again.
- The latest stream-alignment regression fixes are now documented and covered too:
  stateful JSON stream-end synthesis no longer runs eagerly before EOF, which restores clean-EOF
  Bedrock structured-output extraction, and DeepSeek provider-owned streams now converge with
  AI SDK's hardcoded `stream_options.include_usage = true` behavior across all public entrypoints.
- TogetherAI's package boundary now also matches the audited AI SDK split more closely:
  canonical `togetherai` keeps chat/completion/embedding/speech/transcription on the shared
  OpenAI-compatible runtime, image generation/edit now use provider-owned
  `/images/generations` semantics on the unified path, native rerank remains provider-owned, and
  public typed `TogetherAiImageOptions` now mirrors the AI SDK image option lane instead of
  relying on raw open JSON maps.
- MoonshotAI's package boundary now also matches the audited `@ai-sdk/moonshotai` split more
  closely:
  canonical public/runtime identity is `moonshotai`, the historical `moonshot` id now survives
  only as a hidden migration alias, `Provider::moonshotai()` / `Siumai::builder().moonshotai()`
  plus `provider_ext::moonshotai::*` expose the public wrapper surface, typed
  `thinking` / `reasoningHistory` options normalize onto the audited wire keys, and completion /
  embedding / image stay intentionally unsupported because the upstream package itself is
  language-model-only.
- Mistral's package boundary is now also documented explicitly instead of only implicitly through
  tests and catalog wiring: the audited `@ai-sdk/mistral` split remains `chat + embedding`, public
  typed options stay camelCase while compat normalization owns the final wire-key lowering, and a
  dedicated `docs/workstreams/mistral-package-surface-alignment/` record now captures the
  intentional lack of completion/image support.
- Helper-surface follow-up alignment also closed two typed metadata drift points on the audited
  OpenAI-family helpers: OpenAI now exposes keyed metadata accessors alongside the canonical
  `openai` root, and the built-in Perplexity compat preset now reshapes `providerMetadata` into
  the AI SDK-style `images/imageUrl|originUrl`, `usage.citationTokens|numSearchQueries`, and
  `cost.*` layout instead of forwarding raw snake_case response fragments.
- The same Perplexity wrapper audit also now separates public typed options from the wire contract
  more cleanly: `PerplexityOptions` / `PerplexityWebSearchOptions` serialize with camelCase on the
  public Rust surface, compat request normalization explicitly lowers those fields onto
  Perplexity's snake_case wire keys, and the public typed form now wins over legacy raw aliases
  when both are present.
- The same compatibility sweep also closed adjacent stable-structure drift on the public provider
  path: Ollama/Cohere now follow the shared `FilePartSource` split, and Anthropic JSON response
  conversion now handles `MessageContent::Json`.
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
  - provider-reference-only final video assets are now also materially narrower as a gap:
    - shared `MaterializedVideoAsset` plus `materialize_video_reference(...)` now exist on the
      task-oriented video-family contract
    - audited Gemini and MiniMaxi paths now reuse their provider-owned file runtimes to eagerly
      materialize provider references during `siumai::video::generate(...)`
    - remaining deferred work is mainly limited to providers that still need a different
      authenticated download runtime, such as current Vertex `gs://...` video outputs
- Changelog/workstream notes plus a focused migration note for the `Usage` canonicalization are
  now present.
- The remaining high-level AI SDK helper gaps are now clearer and smaller:
  the Rust crate now has `UIMessage` / `convertToModelMessages` structural support, and the
  remaining work is the intentionally absent `useChat`-style stateful frontend hook layer plus
  the final provider-owned file-upload option design beyond explicit `purpose`.
- Gemini protocol adapters now also compile against the newer `FilePartSource` split, restoring
  the broader multi-feature verification lane used for the upload-file audit.
- Anthropic Messages request fixture coverage now also locks one of the remaining provider-option
  edge shapes on the request boundary:
  - message-level `providerOptions.anthropic.cacheControl` lowering onto the final text block
  - part-level document `providerOptions.anthropic.{citations,title,context}` preservation
  - normalization fixtures now explicitly pin the canonical restored part-level Anthropic replay
    shape after that transport lowering
- Final public-path parity and cleanup work remain open.

Status: in progress
