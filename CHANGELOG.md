# Changelog

This file lists noteworthy changes. Sections are grouped by version to make upgrades clearer.

## [Unreleased]

### Added

- Stable prompt/content modeling now includes Vercel-aligned `custom`, `reasoning-file`, and explicit tool-result content variants (`file-data`, `file-url`, `file-id`, `image-data`, `image-url`, `image-file-id`) with provider-keyed file-id support.
- Shared image edit typing now exposes AI SDK-style multi-input `images[]` + `mask` semantics through
  public `ImageEditInput` and `ImageEditFileData` types on the extensions/facade surface.
- Shared video generation typing now exposes AI SDK-style typed file/url inputs through public
  `VideoGenerationInput`, plus canonical `count` (`n`), `fps`, and `seed` request knobs in place
  of the older raw `seed_image` / `seed_video` byte fields.

### Fixed

- Stable `Usage` now exposes AI SDK-style `inputTokens` / `outputTokens` / `raw`, and OpenAI-compatible, OpenAI Responses, Anthropic, and Gemini protocol paths now round-trip richer usage breakdowns instead of rebuilding provider-specific partial views.
- `Usage` now treats AI SDK-style usage as the canonical stable storage layer. Legacy `prompt/completion/total` counts remain available only through compatibility accessors/serde, and the public/examples/tests surface has been migrated off direct field access.
- The typed stable stream-part layer in `siumai-core` is now a V4-capable superset that includes first-class `custom` and `reasoning-file` parts, and OpenAI-compatible reserialization now degrades those unsupported V4-only parts into explicit text in `AsText` mode instead of silently dropping them.
- The upgraded typed stream-part overlay now also exposes public `LanguageModelV4*` aliases, so new code and docs can use AI SDK-aligned naming without depending on the historical `LanguageModelV3*` compatibility names.
- The runtime streaming contract now includes a first-class `ChatStreamEvent::Part(ChatStreamPart)` semantic channel plus a separate runtime replay carrier for protocol-only hints, and the main stream processor plus OpenAI/OpenAI-compatible/Anthropic/Gemini serializers now bridge that richer part model instead of forcing major V4 stream semantics through provider-scoped `Custom` payloads.
- OpenAI Responses and Anthropic SSE serializers now normalize runtime `ChatStreamEvent::Part(ChatStreamPart)` values before taking protocol serialization state locks, which fixes direct stable-part replay hangs caused by recursive lock re-entry.
- OpenAI Responses, Anthropic, Gemini, and OpenAI-compatible parser paths now consume that stable part channel directly for their main AI SDK-aligned stream semantics. OpenAI Responses provider-hosted tool / MCP / approval replay now rides the runtime replay carrier instead of loose `rawItem` / `outputIndex` custom payloads, Anthropic now emits runtime parts for `stream-start`, `response-metadata`, `text-*`, provider-hosted `server_tool_use` / MCP `tool-*`, standard local `tool-input-*` / `tool-call`, `reasoning-*`, `source`, and successful `finish` semantics, and OpenAI-compatible chat chunks now emit lifecycle parts for `stream-start`, `response-metadata`, `text-*`, `reasoning-*`, and `finish` while keeping legacy deltas in parallel for compatibility. Anthropic `signature_delta` and `redacted_thinking` now also stay on that stable lane through reasoning-part `providerMetadata`, matching AI SDK stream behavior instead of relying on provider-scoped custom events.
- OpenAI-compatible tool streaming now emits stable `tool-input-start` / `tool-input-delta` / `tool-input-end` / `tool-call` parts before the legacy shadow deltas, chat-completions reserialization now deduplicates mixed stable+legacy tool streams with first-source-wins semantics, and `StreamProcessor` now preserves final stable `tool-call` parts instead of dropping them during final response assembly.
- OpenAI-compatible chat responses now map URL citations from non-stream `message.annotations` and streaming `delta.annotations` into stable `source` parts, and stable URL `source` parts now round-trip back into chat-completions `annotations` during SSE reserialization.
- OpenAI-compatible exact alignment coverage now also pins those citation semantics on the public paths: non-stream chat-response fixtures lock `text -> tool-call -> source(url)` ordering for `message.annotations`, same-protocol chat-completions roundtrip tests lock `delta.annotations -> source(url) -> delta.annotations`, and the compat chat-response fixture suite now asserts the canonical AI SDK-style `Usage.inputTokens/outputTokens/raw` shape instead of only legacy totals.
- OpenAI-compatible same-protocol chat-completions roundtrip coverage now also preserves public-path `response-metadata`, terminal streamed `logprobs`, AI SDK-style `acceptedPredictionTokens` / `rejectedPredictionTokens` mirrored from `completion_tokens_details`, and terminal response-envelope `system_fingerprint` / `service_tier` fidelity. The bridge now prefers the richer `StreamEnd` envelope over earlier finish parts for chat-completions terminal chunks, and the terminal serializer maps stable finish-part/provider metadata logprobs back into the canonical chat-completions `choices[].logprobs.content` shape.
- Stable runtime stream types such as `ChatStreamPart`, `ChatStreamToolCall`, and related replay metadata are now re-exported through the normal streaming/prelude surface, and the current high-level/gateway code paths no longer need `__private::types` for the main stable stream contract.
- Anthropic streaming now preserves extended usage fields such as `cache_creation_input_tokens`, `cache_read_input_tokens`, `server_tool_use`, and `service_tier` across decode/encode round-trips.
- Anthropic Messages response parsing now keeps `Usage.raw` as the stable AI SDK-aligned provider-raw subset while preserving the full provider `usage` object under `provider_metadata.anthropic.usage`; absent optional raw fields are omitted instead of emitted as `null`.
- Anthropic request-side document citations/title/context and per-part cache control now read only canonical `providerOptions.anthropic` inputs; legacy `message.metadata.custom["anthropic_document_*"]`, `message.metadata.custom["anthropic_content_cache_*"]`, and request-side file `provider_metadata` no longer participate in Anthropic request conversion.
- Anthropic same-protocol SSE replay now preserves reasoning signature and redacted-thinking fidelity from stable `reasoning-*` parts, so direct runtime-part replay no longer depends on the legacy `anthropic:thinking-signature-delta` custom event.
- Anthropic SSE transcoding now applies `AsText` fallback even when unsupported AI SDK parts arrive on the direct `ChatStreamEvent::Part/PartWithReplay` lane, so lossy OpenAI Responses -> Anthropic replay no longer silently drops `tool-approval-request`, and the fixture-backed OpenAI/Anthropic transcoding suite now matches the stable `mcp_tool_use` / `mcp_tool_result` semantics emitted for hosted OpenAI tools.
- Anthropic non-stream request/response replay now also keeps reasoning metadata on the AI SDK-aligned part boundary: Anthropic JSON responses parse `thinking` / `redacted_thinking` into `ContentPart::Reasoning.providerMetadata.anthropic`, the Anthropic assistant-message helper translates that response metadata into next-turn `providerOptions.anthropic`, and Anthropic prompt/JSON replay no longer depends on message-level `metadata.custom["anthropic_*"]` thinking shims.
- The experimental request bridge now follows that same request-vs-response split for Anthropic/OpenAI reasoning replay, so Anthropic thinking/redacted-thinking and OpenAI encrypted reasoning annotations live on part-level `providerOptions` / `providerMetadata` instead of legacy message-level custom shims.
- Stable `Usage` now preserves known-vs-unknown legacy totals internally, so OpenAI-compatible, OpenAI Responses, Anthropic, and Gemini replay paths no longer collapse provider-unknown or `null` usage totals into synthetic zero counts during JSON/SSE serialization.
- Gemini usage replay now counts `candidatesTokenCount + thoughtsTokenCount` as total output usage, preserves `cachedContentTokenCount` / `trafficType` during SSE round-trips, and keeps raw Gemini usage fields when the stable layer does not know a replacement value.
- `StreamProcessor` now preserves terminal response envelope fields from `StreamEnd`, including `id`, `model`, `audio`, `system_fingerprint`, `service_tier`, and `warnings`, while also keeping terminal multimodal parts that were not rebuilt from deltas.
- OpenAI-compatible streaming now carries top-level terminal chunk fields such as `system_fingerprint` and `service_tier` into `StreamEnd`, including EOF fallback finalization when the server omits an explicit finish chunk.
- OpenAI-compatible streaming now also matches AI SDK Azure model-router metadata timing more closely: empty `prompt_filter_results` preludes with `created = 0` and blank `id` / `model` no longer emit placeholder response metadata before the first real metadata chunk arrives.
- OpenAI-compatible response metadata extraction now follows an AI SDK-style provider-owned policy instead of a shared compat-layer whitelist: `OpenAiStandardAdapter` / `ConfigurableAdapter` opt specific providers into `sources` / `logprobs` / prediction-token metadata, Perplexity keeps its hosted-search extras as a provider-specific special case, and generic OpenAI-compatible providers no longer infer those metadata fields by default.
- OpenAI-compatible public config/builder surfaces now also expose an AI SDK-style response `metadataExtractor` hook through `ResponseMetadataExtractor`, `OpenAiCompatibleConfig::with_metadata_extractor(...)`, and `OpenAiCompatibleBuilder::with_metadata_extractor(...)`, so callers can extend provider metadata without reimplementing the whole compat adapter.
- OpenAI-compatible provider-level request settings now align more closely with AI SDK `openai-compatible`: chat streaming omits `stream_options.include_usage` by default unless callers opt in through `OpenAiCompatibleConfig::with_include_usage(true)` / `OpenAiCompatibleBuilder::with_include_usage(true)`, public `queryParams`-style URL settings now flow through compat chat / embeddings / image generation-edit-variation / audio / rerank / model-listing routes, and the final compat chat body can now be customized through a public `RequestBodyTransformer` hook that mirrors AI SDK `transformRequestBody`.
- OpenAI-compatible public config/builder/runtime surfaces now also expose an explicit provider-level `supportsStructuredOutputs` policy aligned with AI SDK semantics: compat chat now defaults to downgrading JSON Schema outputs to `response_format = { "type": "json_object" }` while emitting a stable `unsupported { feature: "responseFormat" }` warning on chat responses, and callers can opt back into wire-level `json_schema` by setting `supportsStructuredOutputs = true`.
- OpenAI-compatible chat request shaping now also honors AI SDK-style known compat provider options from both canonical `providerOptions.openaiCompatible` and provider-owned keys: `user`, `reasoningEffort`, `textVerbosity`, and `strictJsonSchema` are mapped onto the final wire body (`user`, `reasoning_effort`, `verbosity`, `response_format.json_schema.strict`) instead of leaking through as raw compatibility keys.
- OpenAI-compatible chat responses now also surface AI SDK-style warnings for provider-defined tools on the default runtime path: provider-defined tools are still filtered out of Chat Completions requests, and successful chat responses now include `unsupported { feature: "provider-defined tool <id>" }` warnings without requiring callers to install a custom middleware.
- Unified warnings now expose AI SDK-style `unsupported` / `compatibility` shapes through a compatibility-superset model, and `systemMessageMode=remove` is reported through the `compatibility` warning type instead of a generic message.
- Unified `source` and `tool-approval-*` parts now preserve document/approval fields needed for closer AI SDK parity, including source `mediaType` / `filename` / `providerMetadata`, approval request `providerMetadata`, and approval response `reason`.
- Stable `source` parts now use a stricter URL/document union shape instead of a loose `sourceType + optional fields` bag, while preserving the same wire-level `sourceType` serialization.
- OpenAI Responses request conversion now forwards approval reasons, while Gemini and Anthropic source fallback paths now handle document-style source parts without assuming URL-only payloads.
- OpenAI/Azure Responses non-stream response parsing now matches AI SDK response content shape more closely: assistant `message.content[*].output_text` is always preserved as structured text parts on the stable boundary, including plain and empty text, so typed `providerMetadata.{openai|azure}.itemId` is no longer lost behind the single-text fast path, and the canonical OpenAI Responses JSON encoder now uses that text-part metadata instead of consuming or duplicating a legacy top-level response `itemId`. The same OpenAI Responses response-side alignment sweep also preserves `responseId` / `serviceTier`, text-part `phase` / raw `annotations`, and document citation `type` / `index` across exact JSON and SSE roundtrips.
- xAI Responses now follows the audited `repo-ref/ai` boundary more closely: non-stream text/source parts stay metadata-free, reasoning parts use `providerMetadata.xai.itemId` instead of the shared OpenAI namespace, top-level response `provider_metadata` is omitted, and xAI SSE reasoning parts now also carry `providerMetadata.xai.itemId` while `text-*` / `finish` remain metadata-free. The xAI SSE converter also backfills a missing `reasoning-start` before `reasoning-end` when upstream closes a reasoning item without an earlier start event.
- xAI request-side parity moved closer to the audited AI SDK split as well: the shared Responses request transformer now maps xAI `reasoningEffort` / `reasoningSummary`, `topLogprobs -> logprobs=true`, `previousResponseId`, and `store=false -> include += reasoning.encrypted_content`, assistant xAI message ids no longer collapse into OpenAI-style `item_reference`, assistant xAI tool calls now emit stable ids plus `status: "completed"`, and the OpenAI-compatible xAI chat path now normalizes supported chat fields while stripping Responses-only knobs (`reasoningSummary`, `previousResponseId`, `include`, `store`) before hitting `/chat/completions`.
- xAI chat typed options now also align more closely with `repo-ref/ai`: `parallel_function_calling` is exposed end-to-end on the typed surface, deprecated `xHandles` input now normalizes to wire `included_x_handles` instead of leaking as `x_handles`, and `with_default_search()` now matches the upstream `maxSearchResults=20` default.
- The provider-owned xAI option model is now split to match the audited AI SDK structure more closely: chat-only knobs live on `XaiChatOptions`, Responses-only knobs live on `XaiResponsesOptions`, and the main reasoning/include slots now use enum-backed typed wrappers instead of raw `String` / `Vec<String>` bags while preserving forward-compatible string passthrough for newly introduced upstream values.
- xAI search-source typing now follows the AI SDK structure more closely: `SearchSource` is modeled as a discriminated union over `web` / `news` / `x` / `rss` instead of a single permissive field bag, while deprecated `xHandles` input still normalizes to `included_x_handles`.
- xAI Responses tool preparation now matches the audited AI SDK `packages/xai/src/tool/*` and `responses/xai-responses-prepare-tools.ts` surface more closely: public Rust tool factories now cover `web_search`, `x_search`, `code_execution`, `view_image`, `view_x_video`, `file_search`, and `mcp`; xAI tool args now serialize to the expected snake_case request shape (`allowed_domains`, `allowed_x_handles`, `vector_store_ids`, `server_url`, etc.); unknown xAI provider-defined tools are no longer forwarded blindly; and xAI server-side provider tools are no longer forced through invalid Responses `tool_choice` payloads.
- The xAI provider-defined tool surface now also exposes typed Rust arg models and factory-style helpers for the audited AI SDK tool configs: `WebSearchArgs`, `XSearchArgs`, `FileSearchArgs`, and `McpArgs`, plus `web_search_with(...)`, `x_search_with(...)`, `file_search_with(...)`, and `mcp_server_with(...)`, so callers no longer need raw `.with_args(json)` for the main xAI hosted-tool path.
- xAI Responses SSE custom-tool streaming now matches the audited `repo-ref/ai/packages/xai/src/responses/xai-responses-language-model.ts` flow more closely: xAI `custom_tool_call` items (`x_search`, `view_x_video`) defer the finalized `tool-input-start` / `tool-input-delta` / `tool-input-end` plus `tool-call` emission until `response.output_item.done`, `response.custom_tool_call_input.*` is treated as input buffering instead of a second public event lane, and the fixture-backed xAI stream regression suite now covers `web_search`, `file_search`, and `x_search` on the stable part boundary.
- xAI public model constants were refreshed to match the current AI SDK reference set more closely, including `grok-4-1-fast-*`, `grok-4-fast-*`, `grok-4.20-*`, `grok-code-fast-1`, and `grok-3-mini-latest`.
- xAI provider-owned image/video parity now follows the audited AI SDK split much more closely: typed `XaiImageOptions` / `XaiVideoOptions` plus request ext traits are public, xAI native image generation/edit now use `/images/generations` and `/images/edits`, xAI native video create/query now use `/videos/generations|edits` and `GET /videos/{request_id}`, registry/native metadata/public-path parity now expose xAI image generation and video task support, and the shared video request/response types now carry AI SDK-style `providerOptions`, per-request `HttpConfig`, `aspectRatio`, `videoUrl`, metadata, warnings, and response envelopes.
- Shared image edit/provider boundaries now align more closely with AI SDK image-model semantics:
  `ImageEditRequest` carries typed multi-input `images[]` plus typed `mask`, xAI native edit now
  emits single-input `image` vs multi-input `images`, and OpenAI/OpenAI-compatible multipart edit
  plus Vertex inline edit now accept multiple file-backed source images. URL-backed edit inputs are
  already supported on xAI and are explicitly rejected on multipart/inline provider paths until a
  shared async materialization layer exists.
- Shared/provider video request shaping now follows the audited AI SDK split more closely as well:
  xAI video creation warns-and-filters unsupported `n` / `fps` / `seed` knobs while consuming
  typed `VideoGenerationInput` image/video inputs, Gemini/Veo now maps canonical `count` / `seed` /
  typed image input onto the provider body and surfaces warnings for unsupported URL/FPS cases, and
  MiniMaxi no longer serializes the entire shared request struct blindly on its provider-owned video
  path.
- MiniMaxi video request shaping now fully leaves the shared boundary for vendor-only knobs:
  `prompt_optimizer`, `fast_pretreatment`, `callback_url`, and `aigc_watermark` moved off the
  shared `VideoGenerationRequest` shape into provider-owned typed `MinimaxiVideoOptions` carried via
  `providerOptions["minimaxi"]`, while the MiniMaxi video builder keeps those fluent helpers on the
  provider-owned extension surface.
- Stable prompt/content request boundaries now expose first-class `providerOptions` on messages, request-capable content parts, and tool-result output/content shapes. OpenAI-compatible, OpenAI Chat, OpenAI Responses, and Anthropic request conversion now read those canonical fields for the main user-visible request paths instead of request-side `providerMetadata` / `message.metadata.custom`, and the last audited OpenAI Responses assistant tool-call metadata shim has been removed so those request paths stay on canonical `providerOptions` only.
- `ProviderOptionsMap` serde now normalizes provider ids during JSON decode and restores the canonical `openaiCompatible` wire key during encode, so JSON-fed requests and builder-constructed requests no longer diverge on provider-option lookup semantics.
- Public-path parity coverage now locks that corrected request boundary on the real entrypoints too: OpenAI Chat builder/provider/config/registry paths all agree on canonical part `providerOptions.openai.imageDetail`, and OpenAI-compatible builder/provider/config/registry paths all agree on canonical message/part/tool-result `providerOptions.openaiCompatible` while ignoring the removed request-side metadata input channels.
- Anthropic Messages request normalization and bridge inspection now stay on that canonical boundary too: document citations/title/context plus content-block cache control are restored directly onto part `providerOptions.anthropic`, bridge-side cache-limit/drop reporting reads those canonical part options, and single-text messages no longer lose part-level provider options when normalization compacts message content.
- The experimental request bridge no longer treats request-side reasoning `providerMetadata.anthropic|openai` as replay input on its Anthropic/OpenAI direct-pair paths, so redacted/encrypted reasoning replay now requires canonical part `providerOptions` all the way through inspection and pair-specific preprocessing.
- OpenAI Responses, Anthropic, and Gemini protocol paths now consume the explicit tool-result file/image variants instead of the older coarse image/file union, preserving more AI SDK V4 semantics at the wire boundary.
- OpenAI Responses request bridging/normalization now has fixture-backed coverage for native tool-result `image-file-id` / `file-id` roundtrips, including provider-keyed `ToolResultFileId` selection preferring OpenAI-native ids on the Responses wire shape.
- OpenAI Responses input and response fixture baselines now follow the current stable canonical shapes: tool-result attachments use explicit `image-data` / `image-url` / `file-data` variants instead of the removed generic `file` tool-result shape, unsupported-settings warnings are asserted through `unsupported { feature }`, and exact response roundtrips now lock `Usage.inputTokens` / `Usage.outputTokens` / `Usage.raw`.
- OpenAI Responses request-side boundary handling is now closer to AI SDK canonical behavior: request conversion, warning snapshots, and request normalization all treat `providerOptions` as the primary carrier for reasoning `itemId` / `reasoningEncryptedContent`, image `imageDetail`, and MCP approval ids; canonical request fixtures were migrated away from request-side `providerMetadata`, while the remaining tool-call `itemId` metadata fallback is now an explicit narrow compatibility shim kept only because upstream AI SDK still accepts it.
- Gemini usage metadata now preserves `trafficType` during JSON replay instead of dropping it at the type boundary.
- Public macros, bridge tests, fixture tests, and example surfaces now compile against message/part/tool-result `providerOptions`, and provider metadata helpers across Azure/OpenAI-compatible/Gemini/Vertex were extended to cover the new stable `reasoning-file` / `custom` variants without breaking all-features builds.

### Migration Notes

- `Usage` canonicalization and compatibility-accessor migration: `docs/workstreams/ai-sdk-structural-alignment/migration-notes.md`


## [0.11.0-beta.6] - 2026-03-24

### Highlights

- Fearless Refactor V3 introduces family-first Rust APIs (`siumai::{text,embedding,image,rerank,speech,transcription}`) while moving legacy method-style entry points into an explicit compatibility module (`siumai::compat`).
- The recommended construction path now clearly favors registry/config-first provider clients; builder-style entry points remain available as migration-friendly compatibility conveniences.
- Advanced bridge/gateway support becomes much more usable, with request normalization plus cross-protocol request/response/stream transcoding for proxy and gateway scenarios.
- Provider parity improved across native and OpenAI-compatible integrations, so facade, registry, and direct provider clients behave more consistently.
- Family calls now support richer per-call overrides, and config-first providers can attach interceptors and model middlewares directly from `*_Config`.

### Added

- Model-family V3 traits in `siumai-core` for text, embedding, image, rerank, speech, and transcription.
- New family API modules in the `siumai` facade: `siumai::{text,embedding,image,rerank,speech,transcription}`.
- `siumai::tooling`, a runtime tool surface that binds tool schemas to executable handlers for orchestrator/tool-loop workflows.
- `siumai::compat`, an explicit home for legacy compatibility entry points.
- Per-request `HttpConfig` overrides (headers + timeout), including streaming requests.
- Convenience methods on `dyn LlmClient` for full chat requests: `chat_request` and `chat_stream_request`.
- Experimental bridge APIs for inbound request normalization and outbound response/stream transcoding, with customization hooks for hosted tools and gateway adapters.
- Axum gateway runtime and policy helpers, plus reference examples for OpenAI, Anthropic, Gemini, and cross-protocol proxy flows.
- Broader config-first constructors/helper aliases across native and OpenAI-compatible providers, with more consistent exposure of provider params and metadata on the facade.
- OpenAI Responses WebSocket session helpers, including incremental sessions, remote cancellation, and recovery/fallback controls.
- Public batch embedding helpers for Gemini and Vertex models.

### Changed

- Documentation and examples now consistently prefer registry/config-first construction and family APIs for inference (for example, `text::generate`).
- Focused provider wrapper packages, provider factories, and handle routing were aligned so text/image/rerank/embedding flows behave more consistently across registry, facade, and direct provider construction paths.
- Provider request normalization and default base URL behavior are more consistent across facade, registry, and gateway paths.

### Deprecated

- `Siumai::builder()` remains available, but is deprecated as the primary construction style. Prefer `registry::global().language_model("provider:model")` or `*Client::from_config(...)` for new code.

### Fixed

- Streaming and bridge fidelity across OpenAI Responses/Chat, Anthropic, Gemini, and Vertex now preserves more wire-level details, including finish reasons, citations, reasoning metadata, approval items, provider tool results, and web/file search source identity.
- Structured output mapping and JSON-repair/content-filter handling are more reliable across both provider-native and bridged responses.
- OpenAI defaults no longer leak Responses-only settings into non-chat requests; audio fallback defaults and chat default backfilling were corrected.
- Family-model and registry paths now preserve per-request config, capability forwarding, and parity for embedding/image/rerank flows more reliably.

### Migration guide

- Full guide: `docs/migration/migration-0.11.0-beta.6.md`

## [0.11.0-beta.5] - 2026-01-15

### Highlights

- The public API is explicitly Vercel-aligned and fixed to the 6 stable model families: Language / Embedding / Image / Rerank / Speech (TTS) / Transcription (STT).
- Provider-specific features (web search, file search stores, thinking replay, etc.) are **extensions by design**: provider-hosted tools (`hosted_tools::*`) + `providerOptions` + typed `provider_ext::*`.
- Gemini now supports a clean Vertex AI setup (regional base URL helper, ADC token provider, and resource-style model id normalization).
- Fearless refactor phase: workspace split into `siumai-core` (runtime/types/standards), provider crates (`siumai-provider-*`), and `siumai-registry` (factories/handles); `siumai` remains the recommended facade crate.

### Breaking changes

- Unified web search was removed. Use provider-hosted tools instead:
  - OpenAI: `siumai::hosted_tools::openai::web_search()` (Responses API via `OpenAiOptions::with_responses_api`)
  - Anthropic: `siumai::hosted_tools::anthropic::web_search_20250305()`
  - Gemini: `siumai::hosted_tools::google::*` (e.g. `google_search()`, `file_search()`)
- `siumai::providers::<provider>::*` is now a stable alias for `siumai::provider_ext::<provider>::*` (Vercel-aligned).
  - Use `siumai::prelude::unified::*` for the unified surface.
  - Use `siumai::provider_ext::<provider>::*` for provider-specific APIs.
  - For protocol-layer helpers, use `siumai::experimental::*` (advanced) or depend on the relevant provider crate directly (e.g. `siumai-provider-openai`).
- The facade surface was tightened to reduce accidental cross-layer coupling.
  - Removed stable entry points: `siumai::{types,traits,error,streaming}::*`
  - Prefer: `use siumai::prelude::unified::*;`
  - For non-unified extension capabilities: `use siumai::extensions::*;` + `use siumai::extensions::types::*;`
- `LlmBuilder` is no longer re-exported from `siumai::prelude::unified::*` (breaking).
  - Prefer `registry::global()` for new code, or use builder-style construction via `Siumai::builder()` (unified) / `Provider::<provider>()` / `siumai::provider_ext::<provider>::*` (provider-specific).
- Provider-specific capability traits were removed from the core surface (e.g. `traits::{OpenAiCapability, AnthropicCapability, GeminiCapability, ...}`).
  - Use `siumai::prelude::unified::*` for the stable surface, and `siumai::prelude::extensions::*` / `siumai::provider_ext::<provider>` for opt-in provider-specific features.
- “Audio” is no longer a first-class unified family: prefer `SpeechCapability` (TTS) and `TranscriptionCapability` (STT).
  - For OpenAI SSE audio/transcript streaming, use provider extensions: `siumai::provider_ext::openai::{speech_streaming, transcription_streaming}`.
- OpenAI’s public API does not expose a rerank endpoint.
  - If you call rerank with the default OpenAI base URL (`https://api.openai.com/v1`), Siumai returns `UnsupportedOperation`.
  - For rerank, use a rerank-capable provider (e.g. `cohere`, `togetherai`, `bedrock`) or an OpenAI-compatible vendor that exposes `/rerank` (e.g. `siliconflow`).
- Vertex base URL helper now prefers the regional host (`https://{location}-aiplatform.googleapis.com`); if you hardcoded `https://aiplatform.googleapis.com` you may want to update.

### Added

- New unified prelude modules:
  - `siumai::prelude::unified::*` (6 model families only; recommended for new code)
  - `siumai::prelude::extensions::*` (non-family capabilities; opt-in)
- Registry handles for all six model families:
  - `registry.reranking_model(..)`, `registry.speech_model(..)`, `registry.transcription_model(..)`
- Local test tier scripts for faster iteration during fearless refactors:
  - `./scripts/test-fast.sh`, `./scripts/test-smoke.sh`, `./scripts/test-full.sh`
- M1 “core trio” smoke scripts (fixture audit + transcoding + tool-loop gateway):
  - Windows: `./scripts/test-m1.bat`
  - Unix: `./scripts/test-m1.sh`
- Split-phase architecture docs:
  - `docs/architecture/architecture-refactor-plan.md`
  - `docs/architecture/capability-surface.md`
  - `docs/architecture/provider-extensions.md`
- Vertex (Gemini) example:
  - `siumai/examples/04-provider-specific/google/vertex_chat.rs` (`--features "google gcp"`)
- Vercel-aligned provider-hosted tools (provider-executed tools)
  - OpenAI Responses API: `hosted_tools::openai::{web_search,web_search_preview}`
  - Anthropic Messages: `hosted_tools::anthropic::web_search_20250305`
  - Gemini: `hosted_tools::google::{google_search,file_search,code_execution,url_context,enterprise_web_search}`
- OpenAI provider extensions (non-unified streaming)
  - TTS SSE audio streaming: `siumai::provider_ext::openai::speech_streaming::tts_sse_stream`
  - STT SSE transcript streaming: `siumai::provider_ext::openai::transcription_streaming::stt_sse_stream` (see `examples/04-provider-specific/openai/stt_sse_streaming.rs`)
- Anthropic provider extension example
  - Thinking replay: `examples/04-provider-specific/anthropic/thinking-replay-ext.rs`
- Gateway/proxy streaming utilities (Vercel-aligned `parseStreamPart` / `formatStreamPart` concept):
  - Stream encoders: `siumai::experimental::streaming::{encode_chat_stream_as_sse, encode_chat_stream_as_jsonl}` (serialize `ChatStreamEvent` back into provider-native wire formats)
  - Non-streaming JSON encoders: `siumai::experimental::encoding::{JsonResponseConverter, encode_chat_response_as_json}` (serialize `ChatResponse` back into provider-native JSON responses)
  - Bidirectional SSE support for proxying:
    - OpenAI Responses SSE stream serialization (Vercel-aligned `openai:*` stream parts)
    - OpenAI-compatible Chat Completions SSE stream serialization
    - Gemini GenerateContent SSE stream serialization
  - Cross-provider stream part bridge for gateway output:
    - `siumai_core::streaming::OpenAiResponsesStreamPartsBridge` (maps `gemini:*` / `anthropic:*` custom parts into `openai:*` parts)
  - Alignment notes: `docs/alignment/streaming-bridge-alignment.md`
  - Fixture drift audit script (against `repo-ref/ai`): `./scripts/audit_vercel_fixtures.py`
- Provider correctness and parity audit docs (official APIs + Vercel reference):
  - Global checklist: `docs/alignment/provider-implementation-alignment.md`
  - Official API audits: `docs/alignment/official/*-official-api-alignment.md` (OpenAI, Anthropic, Gemini, Google Vertex, Anthropic on Vertex, Azure OpenAI, Groq, xAI, Amazon Bedrock, Cohere, TogetherAI, Ollama)

### Changed

- OpenAI-compatible builder `provider_specific_config` is now applied to chat requests via the compat adapter layer.
- Model listing and model retrieval endpoints are now spec-driven (`ProviderSpec::{models_url, model_url}`) to support non-OpenAI routes (e.g. Anthropic `/v1/models`, Ollama `/api/tags`) without provider-specific URL plumbing.
- Advanced orchestrator examples are now maintained under `siumai-extras/examples/*` (the `siumai` facade focuses on low-level provider/client APIs).
- HTTP execution now supports an injectable transport (`fetch` / `HttpTransport`) across providers, including streaming use-cases (gateway parity with Vercel's `fetch(customTransport)`).
- OpenAI moderation now defaults to `omni-moderation-latest` when no model is provided.
- Gateway/proxy streaming policies are now explicit:
  - V3 parts that cannot be represented in a target wire format follow `V3UnsupportedPartBehavior` (drop in strict mode, lossy text downgrade in `AsText` mode), including `tool-approval-request`, `raw`, and `file` parts.
- Gateways can also transcode non-streaming results into provider JSON responses:
  - `siumai-extras::server::axum::{to_transcoded_json_response, TargetJsonFormat}`

### Deprecated

- `AudioCapability` (compat trait): prefer `SpeechCapability` + `TranscriptionCapability` on the unified surface.
- `ModelListingCapability`, `ModerationCapability`, `FileManagementCapability` on the top-level: prefer `siumai::prelude::extensions::*`.
  - `VisionCapability` remains available for compatibility, but vision is treated as multimodal Chat (Vercel-aligned) rather than a separate family.
- Low-level HTTP helper `execute_json_request_with_headers` (for custom provider code): prefer `HttpExecutionConfig` + `execute_json_request` and/or a `ProviderSpec` with a stable `build_headers()` implementation.

### Removed

- Legacy unified web search types and helpers:
  - `siumai::types::web_search` (`WebSearchConfig` etc.)
  - `siumai-extras::web_search`
- Provider-specific capability traits from the core surface:
  - `traits::{OpenAiCapability, AnthropicCapability, GeminiCapability, ...}`
  - Use `siumai::provider_ext::<provider>`, provider-hosted tools (`siumai::hosted_tools::<provider>`), and `providerOptions` instead.
- `OpenAiCompatibleSpec` (legacy fallback): use `OpenAiCompatibleSpecWithAdapter` (adapter-injected spec only).

### Fixed

- Gemini base URL defaults are now consistent across protocol and provider metadata (`https://generativelanguage.googleapis.com/v1beta`).
- Gemini Vertex AI URLs now accept resource-style model names (e.g. `models/gemini-2.0-flash`) and normalize them to prevent duplicate `/models/...` segments.
- Gemini tool result encoding is now Vercel-aligned (tool role maps to `function_call_output`/`functionResponse`; assistant URL-based `fileData` is rejected).
- Vertex `base_url_for_vertex(...)` now prefers the regional host (`https://{location}-aiplatform.googleapis.com`) to match official docs (location `global` still uses `https://aiplatform.googleapis.com`).
- Vertex enterprise auth now auto-enables ADC bearer token wiring (and does not overwrite user-provided `Authorization` headers), matching Vercel behavior.
- Vertex Imagen requests in API-key mode now append `?key=...` to the endpoint URL (Vercel parity).
- Anthropic on Vertex now matches Vercel request shaping:
  - Uses `:rawPredict` / `:streamRawPredict` (instead of `?alt=sse`)
  - Injects `anthropic_version: "vertex-2023-10-16"` and omits the `model` field from the request body
- Anthropic orchestrator steps now forward `providerMetadata.anthropic.container.id` into `providerOptions.anthropic.container.id` automatically.
- OpenAI rerank response parsing no longer panics on missing optional fields (runtime unwrap removal).
- OpenAI Responses API now matches Vercel warning semantics:
  - `conversation` and `previousResponseId` may both be sent (warning emitted instead of hard error).
  - Unsupported standardized settings (`seed`, `topK`, `stopSequences`, penalties) are ignored with warnings.
- OpenAI transcription SSE streaming now preserves accumulated deltas when the stream ends via `[DONE]`/EOF (best-effort `Done.text`).
- Gemini/Anthropic system instruction semantics are now Vercel-aligned (system/developer messages must appear at the beginning; provider-specific exceptions handled internally).
- Anthropic streaming metadata is preserved and surfaced via `provider_metadata["anthropic"]` (thinking replay signatures, redacted thinking, and normalized sources/citations).
- SSE decoding is stricter for JSON payloads (invalid frames no longer silently corrupt downstream state).
- CI no longer depends on `zsh` for local test scripts (`./scripts/test-*.sh` now use `bash`).
- CI clippy steps no longer build examples (reduces disk pressure and avoids `No space left on device` in runners).
- `MessageContent::Json` now downgrades to text where the provider does not define an input JSON content part (compile-clean across feature matrices).

### Migration guide (beta.5)

- Full guide: `docs/migration/migration-0.11.0-beta.5.md`
- If you used unified web search, switch to provider-hosted tools:
  - OpenAI: `siumai::hosted_tools::openai::web_search()` + Responses API (`OpenAiOptions::with_responses_api`)
  - Anthropic: `siumai::hosted_tools::anthropic::web_search_20250305()`
  - Gemini: `siumai::hosted_tools::google::google_search()` / `file_search()` / `url_context()` / `enterprise_web_search()`
  - See `docs/architecture/provider-extensions.md` for the supported matrix and examples.
- If you want the smallest stable API surface, prefer `use siumai::prelude::unified::*;` and only opt into extensions when needed.
- If you previously relied on provider-specific capability traits, prefer `siumai::provider_ext::<provider>` or downcast via `Siumai::downcast_client::<T>()` for typed provider APIs while still constructing via the unified builder.

## [0.11.0-beta.4] - 2025-12-03

### Added

- Provider factories for Anthropic on Vertex AI and MiniMaxi
  - New factory types: `AnthropicVertexProviderFactory`, `MiniMaxiProviderFactory`
  - Both fully support the shared `BuildContext` (HTTP client/config, tracing, middlewares, retry)
- Default registry factory wiring
  - `registry::helpers::create_registry_with_defaults()` now pre-registers factories for:
    - Native providers: `openai`, `anthropic`, `anthropic-vertex`, `gemini`, `groq`, `xai`, `ollama`, `minimaxi`
    - All built-in OpenAI-compatible providers (DeepSeek, SiliconFlow, OpenRouter, Together, Fireworks, etc.)
- `siumai-extras` workflow and memory abstractions
  - New `WorkflowBuilder<M>` + `Workflow<M>` on top of `Orchestrator<M>` and `ToolLoopAgent<M>`
  - Semantic worker role helpers and constants: `WORKER_PLANNER`, `WORKER_CODER`, `WORKER_RESEARCHER`
  - Pluggable `WorkflowMemory` trait and in-process `InMemoryWorkflowMemory` implementation
  - Example `workflow_planner_coder` showing planner + coder + in-memory memory using OpenAI
- Unified structured output decoding helpers in `siumai-extras`
  - `structured_output::OutputDecodeConfig` used by high-level object helpers, agents, orchestrator, and workflows
  - Shared JSON repair, shape hints, and optional JSON Schema validation (via `schema` feature)

### Changed

- Base URL override semantics for native providers
  - Custom `base_url` values for OpenAI, Gemini, Anthropic, Ollama, xAI, and MiniMaxi are now treated as full API prefixes
  - When a custom `base_url` is set, Siumai no longer appends provider default paths such as `/v1` or `/v1beta`; callers must include any required path segments explicitly
  - Default base URLs (e.g. `https://api.openai.com/v1`, `https://generativelanguage.googleapis.com/v1beta`) are still used when no override is provided
- Unified construction path for `SiumaiBuilder` and Registry
  - `SiumaiBuilder::build()` no longer calls provider helpers directly
  - Instead, it builds a `BuildContext` and delegates to the corresponding `ProviderFactory::language_model_with_ctx()`
  - Ensures that HTTP config, custom `reqwest::Client`, API keys, base URLs, tracing, interceptors, middlewares, and retry options behave identically across:
    - `Siumai::builder()...build()`
    - `registry::global().language_model("provider:model")`
- Anthropic / Gemini / Groq / xAI / Ollama / MiniMaxi registry factories
  - All providers now construct clients via the shared helper functions in `registry::factory` (or their own config/client types) using `BuildContext`
  - Registry-level HTTP interceptors and model middlewares are consistently installed across all clients
- Retry option propagation (builder + registry)
  - `BuildContext.retry_options` is now applied uniformly to all supported providers (OpenAI, Anthropic, Anthropic Vertex, Gemini, Groq, xAI, Ollama, MiniMaxi, and all OpenAI-compatible adapters)
  - `Siumai::builder().with_retry(...)` and `RegistryOptions.retry_options` configure the underlying provider clients via their unified `set_retry_options` / `with_retry` APIs, rather than adding separate ad hoc layers
- Siumai outer retry wrapper semantics
  - `SiumaiBuilder::build()` no longer automatically wraps the resulting `Siumai` instance in an additional retry layer
  - Recommended usage: configure retry via the builder or registry; `Siumai::with_retry_options(...)` remains available as an explicit, opt-in wrapper for advanced scenarios
- Orchestrator and high-level object helpers moved to `siumai_extras`
  - `siumai::orchestrator::*` and `siumai::highlevel::object::*` are now provided by `siumai_extras::orchestrator` and `siumai_extras::highlevel::object`
  - Core `siumai` focuses on low-level provider/client APIs; application-level workflows (agents, structured objects with schema validation) live in `siumai_extras`
- `siumai-extras` structured output API clean-up
  - Renamed extras-side decode config from `StructuredOutputConfig` to `OutputDecodeConfig` to clarify separation from provider-native structured output configs (e.g., OpenAI)
  - High-level `generate_object` / `stream_object`, `ToolLoopAgent` structured output, `Orchestrator::run_typed`, and `Workflow::run_typed` are all backed by the same decode pipeline
- Orchestrator / workflow ergonomics
  - Added `tool_choice(...)` and `active_tools(...)` builders on `OrchestratorBuilder`, `Orchestrator`, and `WorkflowBuilder`
  - These are thin sugar over `prepare_step`, mirroring Vercel AI SDK's `toolChoice` / `activeTools` for common cases
- Provider registry metadata
  - Added a native `anthropic-vertex` entry (with alias `google-vertex-anthropic` and `claude` model prefix) to align routing between builder and registry

### Removed

- Deprecated top-level helper modules from the core crate
  - Removed `siumai::benchmarks`; benchmarking and diagnostics helpers now live in `siumai-extras` or in user code
  - Removed the `siumai::telemetry` shim; telemetry is now wired via `siumai::experimental::observability::telemetry` in the core crate and `siumai-extras::telemetry` for subscriber setup

## [0.11.0-beta.3] - 2025-11-09

### Added

- Unified model-level middleware on `Siumai::builder()`
  - New APIs: `add_model_middleware(...)`, `with_model_middlewares(...)`
  - Auto middlewares now also apply to the unified builder path
- OpenTelemetry 0.31 compatibility
  - Switch to `SdkTracerProvider`, use `Resource::builder_*`, update `PeriodicReader::builder(...)`
  - Updated example under `siumai-extras/examples/opentelemetry_tracing.rs`

### Changed

- MiniMaxi moved to factory flow for consistency
  - Middlewares and interceptors are installed uniformly across all providers
- Consolidated builder helpers and advanced HTTP options
  - Shared utilities for API key/base URL/model normalization
  - Parity of advanced HTTP options between `Siumai::builder()` and `LlmBuilder`

### Fixed

- Applied gzip/brotli/cookie_store flags when building HTTP client
- Correct model propagation for OpenAI‑compatible in unified builder
- Env var loading for OpenAI‑compatible (`{PROVIDER_ID}_API_KEY`)
- Default/alias model handling across providers

## [0.11.0-beta.2] - 2025-11-08

### Added

- MiniMaxi provider support with multi-modal capabilities (text, speech, image generation).
- **Gemini File Search (RAG) support** - Provider-specific implementation for Gemini's File Search API
  - File Search Store management (create, list, get, delete)
  - Example: `siumai/examples/04-provider-specific/google/file_search.rs`

## [0.11.0-beta.1] - 2025-10-28

This beta delivers a major refactor of module layout, execution/streaming, and provider integration. Design inspired by Cherry Studio’s transformer design and the Vercel AI SDK’s adapter architecture.

### Added
- Provider Registry and model handles (`siumai/src/registry/*`)
  - Unified string-based `provider:model` resolution with LRU caching and optional TTL
  - Customizable registry options (middlewares, interceptors, retry)
- HTTP Interceptors (`execution::http::interceptor`)
  - Request/response hooks and SSE event observation
  - Built-in `LoggingInterceptor`
- Execution layer and middleware system (`execution::{executors,transformers,middleware}`)
  - Auto middlewares based on provider/model (defaults/clamping/reasoning extraction)
- Orchestrator rework (`siumai-extras/src/orchestrator/*`)
  - Multi-step tool calling, agent pattern, tool approval, streaming tool execution
  - See examples under `siumai/examples/03-advanced-features/orchestrator/`
- High-level object APIs (`siumai_extras::highlevel::object`)
  - `generate_object` / `stream_object` for provider-agnostic typed JSON outputs
  - Optional JSON repair and schema validation; partial object streaming
- `siumai-extras` crate
  - Optional features: `schema`, `telemetry`, `opentelemetry`, `server`, `mcp`
- Example rework (`siumai/examples/`)

### Changed
- Workspace split into `siumai` and `siumai-extras`.
- Unified streaming events (start/delta/usage/end); improved UTF‑8-safe chunking and tag extraction.
- Unified retry facade (`retry_api`) with idempotency and 401 token refresh retry.
- OpenAI‑compatible providers consolidated via adapter; consistent transformers/executors paths.
- Clippy cleanups; boxed large enum variants internally (minor internal breaking).

### Removed
- Top-level `examples/` moved to `siumai/examples/`.
- Removed obsolete `docs/openapi.documented.yml`.

### Fixed
- Ensure `before_send_hook` is correctly applied across providers.
- UTF‑8 safety: tag extraction, string slicing, streaming chunk boundaries, and token masking.
- Reliability fixes in streaming, headers, and parameter mapping; expanded fixture-based tests.

### Known Issues
- OpenAI Responses API `web_search` is not implemented; calling returns `UnsupportedOperation`.

### Stability
- This is a beta pre-release; minor API adjustments may follow.

### Roadmap
- Starting with `0.11.0-beta.5`, the workspace will be split into multiple crates (core / providers / extras) to mirror the architectural separation already present in the code. The `0.11.0-beta.4` release focuses on closing the feature loop and stabilizing the unified crate API before this split.

### API Keys and Environment Variables

- OpenAI: `.api_key(..)` or `OPENAI_API_KEY` (env fallback)
- Anthropic: `.api_key(..)` or `ANTHROPIC_API_KEY` (env fallback)
- Groq: `.api_key(..)` or `GROQ_API_KEY` (env fallback)
- Gemini: `.api_key(..)` or `GEMINI_API_KEY` (env fallback)
- xAI: `.api_key(..)` or `XAI_API_KEY` (env fallback)
- Ollama: no API key (local service, default `http://localhost:11434`)
- OpenAI‑compatible via Builder: `.api_key(..)` or `{PROVIDER_ID}_API_KEY`
- OpenAI‑compatible via Registry: reads `{PROVIDER_ID}_API_KEY` (e.g., `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`)

### Migration Guide

#### Tracing Subscriber Initialization

**Before (v0.10.3 and earlier):**
```rust
use siumai::tracing::{init_default_tracing, init_debug_tracing, TracingConfig, OutputFormat};

// Initialize with default configuration
init_default_tracing()?;

// Or with custom configuration
let config = TracingConfig::builder()
    .log_level_str("debug")?
    .output_format(OutputFormat::Json)
    .build();
init_tracing(config)?;
```

**After (v0.11.0):**

Option 1: Use `siumai-extras::telemetry` for advanced configuration:
```rust
use siumai_extras::telemetry;

// Add to Cargo.toml:
// For the beta release:
// siumai-extras = { version = "0.11.0-beta.3", features = ["telemetry"] }

// Initialize with default configuration
telemetry::init_default()?;

// Or with custom configuration
let config = telemetry::SubscriberConfig::builder()
    .log_level_str("debug")?
    .output_format(telemetry::OutputFormat::Json)
    .build();
telemetry::init_subscriber(config)?;
```

Option 2: Use `tracing-subscriber` directly for simple cases:
```rust
// Add to Cargo.toml:
// tracing-subscriber = "0.3"

// Simple console logging
tracing_subscriber::fmt::init();
```

#### JSON Schema Validation

**Before:**
```rust
// Schema validation was not available in core siumai
```

**After:**
```rust
use siumai_extras::schema;

// Add to Cargo.toml:
// For the beta release:
// siumai-extras = { version = "0.11.0-beta.3", features = ["schema"] }

// Validate JSON against schema
schema::validate_json(&instance, &schema)?;

// Or use the validator for multiple validations
let validator = schema::SchemaValidator::new(&schema)?;
validator.validate(&instance)?;
```

#### MCP Integration (NEW)

MCP integration is now available as an optional feature in `siumai-extras`:

```toml
[dependencies]
siumai = { version = "0.11", features = ["openai"] }
siumai-extras = { version = "0.11", features = ["mcp"] }
```

**Quick Start:**
```rust
use siumai::prelude::unified::*;
use siumai_extras::mcp::mcp_tools_from_stdio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Connect to MCP server
    let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;

    // 2. Create model (registry is recommended for new code)
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    // 3. Use with orchestrator
    let (response, _) = siumai_extras::orchestrator::generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        vec![siumai_extras::orchestrator::step_count_is(10)],
        Default::default(),
    ).await?;

    Ok(())
}
```

**Supported Transports:**
- **Stdio**: `mcp_tools_from_stdio("node server.js")` - Local development
- **SSE**: `mcp_tools_from_sse("http://localhost:8080/sse")` - Remote servers
- **HTTP**: `mcp_tools_from_http("http://localhost:3000/mcp")` - Stateless

**Documentation:**
- Integration guide: `siumai/docs/guides/MCP_INTEGRATION.md`
- API reference: `siumai-extras/docs/MCP_FEATURE.md`
- Examples: `siumai/examples/05-integrations/mcp/`

## [0.10.3] - 2025-10-10

### Added
- Unified retry API `retry_api` (`retry`, `retry_for_provider`, `retry_with`).
- Builder-level retry options: `with_retry(...)` for `Siumai` and provider builders (OpenAI, Gemini, Anthropic, Groq, xAI, Ollama, OpenAI-compatible).
- Convenience methods: `chat_with_retry`, `ask_with_retry` on `ChatExtensions`.
- Stream processor: overflow handler now accepts closures.

### Deprecated
- `retry_strategy` (planned removal in 0.11).

### Changed
- SiliconFlow and OpenRouter now use the OpenAI-compatible adapter path.
- Simplified tracing guard type and provider identification; removed an unused `Siumai` field.

### Fixed
- Responses API `web_search` now returns `UnsupportedOperation` when not implemented.

### Migration
- Replace `retry_strategy` usage with the unified `retry_api` facade:
  - Use `retry`, `retry_for_provider`, or `retry_with(RetryOptions::...)`.
  - Prefer builder-level `with_retry(...)` for chat operations (applies to Siumai and provider builders).
- `retry_strategy` is deprecated and will be removed in `0.11`.

## [0.10.2] - 2025-10-04

- Unified HTTP client across providers, exposed fine-grained HTTP options on SiumaiBuilder, added with_http_client for Gemini/Custom, and updated docs/examples.

## [0.10.1] - 2025-09-14

### Fixed

- **OpenAI StreamDelta Thinking Field Support** - Fixed #7: Added unified thinking field priority handling (reasoning_content > thinking > reasoning) to OpenAI StreamDelta, matching OpenAI-compatible adapter behavior for consistent thinking content processing across all providers
- **OpenAiCompatibleBuilder Base URL Configuration** - Fixed #7: Added base_url() method to OpenAiCompatibleBuilder enabling custom base URLs for self-deployed OpenAI-compatible servers, alternative endpoints, and local development scenarios

## [0.10.0] - 2025-08-29

### Added

- **Provider-Specific Embedding Configurations** - Added type-safe embedding configuration options for each provider (GeminiEmbeddingOptions with task types, OpenAiEmbeddingOptions with custom dimensions, OllamaEmbeddingOptions with model parameters) through extension traits, enabling optimized embeddings while maintaining unified interface
- **Enhanced ChatMessage System** - Improved ChatMessage with better serialization/deserialization support
- **OpenAI-Compatible Adapter System** - Completely refactored OpenAI-compatible provider system with centralized configuration through unified registry system, supporting 36 providers: DeepSeek, OpenRouter, Together AI, Fireworks, Perplexity, Mistral, Cohere, Zhipu, Moonshot, Doubao, Qwen, 01.AI, Baichuan, SiliconFlow (with comprehensive chat, embeddings, image generation, and document reranking capabilities), Groq, xAI, GitHub Copilot, GitHub Models, Nvidia, Hyperbolic, Jina AI, VoyageAI, StepFun, MiniMax, Infini AI, ModelScope, Hunyuan, Baidu Cloud, Tencent Cloud TI, Xirang, 302.AI, AiHubMix, PPIO, OcoolAI, Poe, and enhanced OpenAI-compatible builder with comprehensive HTTP configuration support including timeout, proxy, and custom headers.
- **Secure Debug Trait Implementation** - Implemented custom Debug trait for all client types with complete sensitive information hiding (API keys, tokens) using clean `has_*` flags instead of masked values, providing production-safe debugging output.

### Fixed

- **StreamStart Event Generation** - Fixed missing StreamStart events in streaming responses across all providers, now properly emitting metadata (id, model, created, provider, request_id) at stream beginning. Implemented multi-event emission architecture that preserves all content while ensuring StreamStart events.

### Changed

- **Streaming Architecture** - Refactored streaming traits to support multi-event emission (breaking change for internal APIs only, user-facing APIs unchanged)
- **Provider Implementations** - All providers now use optimized multi-event conversion logic for better content preservation and consistency

## [0.9.1] - 2025-08-28

### Added

- **Comprehensive Clone Support** - All client types, builders, and configuration structs now implement `Clone` for seamless concurrent usage and multi-threading scenarios

## [0.9.0] - 2025-08-27

### Added

- **Provider Feature Flags** - Added optional feature flags for selective provider inclusion (`openai`, `anthropic`, `google`, `ollama`, `xai`, `groq`) with build-time validation

### Fixed

- **Ollama API Key Requirement** - Fixed SiumaiBuilder to allow Ollama provider creation without API key, as Ollama doesn't require authentication

## [0.8.1] - 2025-08-25

### Added

- **RequestBuilder Send+Sync Support** - Added Send+Sync constraints to RequestBuilder trait for better multi-threading support

### Fixed

- **Type Downcasting Anti-pattern** - Replaced runtime type downcasting with capability methods in `LlmClient` trait
- **Memory Limits in Stream Processing** - Added configurable buffer limits (10MB content, 5MB thinking, 100 tool calls) with overflow handlers
- **Inconsistent Macro Return Types** - All message macros now consistently return `ChatMessage` instead of mixed types
- **Send+Sync Static Assertions** - Added compile-time verification for error type thread safety

### Added

- **Application-Level Timeout Support** - New `TimeoutCapability` trait provides timeout control for complete operations including retries, complementing existing HTTP-level timeouts

## [0.8.0] - 2025-08-13

### Breaking Changes

- **Security and Reliability Improvements** - Introduced `secrecy` crate for secure API key handling, `backoff` crate for professional retry mechanisms.
- **Streaming Infrastructure Overhaul** - Replaced custom streaming implementations with `eventsource-stream` for professional SSE parsing and UTF-8 handling across all providers.

### Added

- OpenAI Responses API support (sync, streaming, background, tools, chaining)
- **Simplified Model Constants** - Introduced simplified namespace for model constants (`siumai::models`) with direct access to model names. Replaced complex categorization system with intuitive model selection: `models::openai::GPT_4O`, `models::anthropic::CLAUDE_OPUS_4_1`, `models::gemini::GEMINI_2_5_FLASH`. Provides better IDE auto-completion and faster model discovery without abstract groupings.

### Fixed

- **URL Compatibility** - Fixed URL construction across all providers to handle base URLs with and without trailing slashes correctly, preventing double slash issues in API endpoints.
- **Anthropic API Compatibility** - Fixed Anthropic API max_tokens requirement by automatically setting default value (4096) when not provided, resolving "max_tokens: Field required" errors.
- **xAI Grok Streaming** - Implemented complete streaming support for xAI Grok models with reasoning capabilities. Added `XaiEventConverter` and `XaiStreaming` components that handle real-time content streaming, reasoning content processing (`reasoning_content` field), tool calling, and usage statistics including reasoning tokens. The implementation follows the same reliable eventsource-stream architecture used by other providers.
- **Common Parameters Not Applied** - Fixed issue where `common_params` (temperature, max_tokens, top_p, etc.) were not being applied in certain scenarios when using ChatCapability trait methods. The parameter passing mechanism has been corrected to ensure both common parameters and provider-specific parameters are properly merged and sent to API endpoints in both streaming and non-streaming modes across all providers.

### Internal

- **Request Builder Refactoring** - Internally refactored parameter construction and request builder implementation for improved maintainability and consistency across all providers.
- **Enhanced Test Coverage** - Added comprehensive test suites including mock framework, concurrency safety tests, network error handling tests, resource management tests, and configuration validation tests to ensure production-ready reliability.
- **Architecture Cleanup** - Removed redundant `UnifiedLlmClient` struct that was a thin wrapper around `ClientWrapper`, simplifying the architecture and reducing API confusion. Removed unused `ClientFactory` that duplicated functionality already provided by `SiumaiBuilder`. Fixed misleading error messages in embedding capability that incorrectly stated OpenAI and Gemini don't support embeddings. Corrected inconsistent documentation examples and parameter type formats across README and examples. All Clippy warnings have been resolved and code consistency has been improved throughout the codebase.

## [0.7.0] - 2025-08-02

### Fixed

- **Code Quality and Documentation** - Fixed all clippy warnings, documentation URL formatting, memory leaks in string interner and optimized re-exports to reduce namespace pollution
- **Tool Call Streaming** - Fixed incomplete tool call arguments in streaming responses. Previously, only the first SSE event from each HTTP chunk was processed, causing tool call parameters to be truncated. Now all SSE events are properly parsed and queued, ensuring complete tool call arguments in streaming mode. [#1](https://github.com/YumchaLabs/siumai/issues/1)
- **StreamEnd Events** - Fixed missing or incorrect StreamEnd events in streaming responses. StreamEnd events are now properly sent when `finish_reason` is received, with correct finish reason values (Stop, ToolCalls, Length, ContentFilter).
- **Send + Sync Markers** - Added Send + Sync bounds to all capability traits and stream types for proper multi-threading support. [#2](https://github.com/YumchaLabs/siumai/issues/2)

## [0.6.0] - 2025-08-01

### Added

- **Unified Embedding API** - Unified embedding API through `Siumai` client with builder patterns and provider-specific optimizations for OpenAI, Gemini, and Ollama

## [0.5.1] - 2025-07-27

### Added

- **Unified Tracing API** - All provider builders (Anthropic, Gemini, Ollama, Groq, xAI) now support tracing methods (`debug_tracing()`, `json_tracing()`, `minimal_tracing()`, `pretty_json()`, `mask_sensitive_values()`)

## [0.5.0] - 2025-07-27

### Added

- **Enhanced Tracing and Monitoring System** - Complete HTTP request/response tracing with security features
  - **Pretty JSON Formatting** - `.pretty_json(true)` enables human-readable JSON bodies and headers

    ```rust
    .debug_tracing().pretty_json(true)  // Multi-line indented JSON
    ```

  - **Sensitive Value Masking** - `.mask_sensitive_values(true)` automatically masks API keys and tokens (enabled by default)

    ```rust
    // Default: "Bearer sk-1...cdef" (secure)
    .mask_sensitive_values(false)  // Shows full keys (not recommended)
    ```

  - **Comprehensive HTTP Tracing** - Automatic logging of request/response headers, bodies, timing, and status codes
  - **Multiple Tracing Modes** - `.debug_tracing()`, `.json_tracing()`, `.minimal_tracing()` for different use cases
  - **UTF-8 Stream Handling** - Proper handling of multi-byte characters in streaming responses
  - **Security by Default** - API keys automatically masked as `sk-1...cdef` to prevent accidental exposure

## [0.4.0] - 2025-06-22

### Added

- **Groq Provider Support** - Added high-performance Groq provider with ultra-fast inference for Llama, Mixtral, Gemma, and Whisper models
- **xAI Provider Support** - Added dedicated xAI provider with Grok models support, reasoning capabilities, and thinking content processing
- **OpenAI Responses API Support** - Complete implementation of OpenAI's Responses API
  - Stateful conversations with automatic context management
  - Background processing for long-running tasks (`create_response_background`)
  - Built-in tools support (Web Search, File Search, Computer Use)
  - Response lifecycle management (`get_response`, `cancel_response`, `list_responses`)
  - Response chaining with `continue_conversation` method
  - New types: `ResponseStatus`, `ResponseMetadata`, `ListResponsesQuery`
  - New trait: `ResponsesApiCapability` for Responses API specific functionality
- Configuration enhancements for Responses API
  - `with_responses_api()` - Enable Responses API mode
  - `with_built_in_tool()` - Add built-in tools (WebSearch, FileSearch, ComputerUse)
  - `with_previous_response_id()` - Chain responses together
- Comprehensive documentation and examples for Responses API usage

### Changed

- **BREAKING**: Simplified `ChatStreamEvent` enum for better consistency
  - Unified `ThinkingDelta` and `ReasoningDelta` into single `ThinkingDelta` event
  - Removed duplicate `Usage` event (kept `UsageUpdate`)
  - Removed duplicate `Done` event (kept `StreamEnd`)
  - Reduced from 10 to 7 stream event types while maintaining full functionality
- Enhanced `OpenAiConfig` with Responses API specific fields
- Updated examples to demonstrate Responses API capabilities

### Fixed

- Updated all examples to use new `StreamEnd` event instead of deprecated `Done` event
  - Fixed `simple_chatbot.rs`, `streaming_chat.rs`, and `capability_detection.rs` examples
  - Ensured all streaming examples work with the simplified event structure

## [0.3.0] - 2025-06-21

### Added

- `ChatExtensions` trait with convenience methods (ask, translate, explain)
- Capability proxies: `AudioCapabilityProxy`, `EmbeddingCapabilityProxy`, `VisionCapabilityProxy`
- Static string methods (`user_static`, `system_static`) for zero-copy literals
- LRU response cache with configurable capacity
- `as_any()` method for type-safe client casting

### Fixed

- Streaming output JSON parsing errors caused by network packet truncation
- UTF-8 character handling in streaming responses across all providers
- Inconsistent streaming architecture between providers

### Changed

- **BREAKING**: Capability access returns proxies directly (no `Result<Proxy, Error>`)
- **BREAKING**: Capability checks are advisory only, never block operations
- Split `ChatCapability` into core functionality and extensions
- Improved error handling with better retry logic and HTTP status handling
- Optimized parameter validation and string processing performance
- Refactored streaming implementations with dedicated modules for better maintainability
- Added line/JSON buffering mechanisms to handle incomplete data chunks
- Unified streaming architecture across OpenAI, Anthropic, Ollama, and Gemini providers

### Removed

- `register_capability()` and `get_capability()` methods
- `with_capability()`, `with_audio()`, `with_embedding()` deprecated methods
- `FinishReason::FunctionCall` (use `ToolCalls` instead)
- Automatic capability warnings

## [0.2.0] - 2025-06-21

### Added

- Ollama provider support (chat, streaming, embeddings, model management)
- Multimodal support for vision-capable models
- `PartialEq` support for `MessageContent` and `ContentPart`

## [0.1.0] - 2025-06-20

### Added

- Initial release with unified LLM interface
- Providers: OpenAI, Anthropic Claude, Google Gemini, xAI, OpenRouter, DeepSeek
- Capabilities: Chat, Audio, Vision, Tools, Embeddings
- Streaming support and multimodal content
- Retry mechanisms and parameter validation
- Macros: `user!()`, `system!()`, `assistant!()`, `tool!()`
