# AI SDK Structural Alignment - TODO

Last updated: 2026-04-26

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
- [x] Record the bounded AI SDK root-export audit.
  - `generateText` is now covered by the real single-step Rust `generate_text(...)` projection.
  - `streamText` is covered at the passive event/result-carrier layer, while the full multi-lane
    `StreamTextResult` runtime remains deferred until Rust owns the tee/backpressure design.
  - `createTextStreamResponse` is covered on the real Rust HTTP server boundary by
    `siumai-extras::server::axum::to_text_stream_response(...)`; Node `ServerResponse` piping
    remains intentionally outside Rust core.
  - `createDownload` is covered by Rust utility helpers that validate safe download URLs, support
    inline `data:` payloads, preserve media type, and enforce the upstream 2 GiB default size
    limit.
  - `SerialJobExecutor` is covered by a cloneable Rust async executor that serializes submitted
    jobs through a shared FIFO mutex and preserves each job's return value.
  - Pure provider-utils HTTP/string helpers are covered by Rust utility functions for header
    normalization/combining, `user-agent` suffix appending, media-type extension mapping, file
    extension stripping, and single trailing-slash removal.
  - `injectJsonInstruction` / `injectJsonInstructionIntoMessages` are covered by Rust prompt
    helpers over `ModelMessage` with the same generic and schema instruction defaults.
  - `parseJSON` / `safeParseJSON` / `isParsableJson` are covered by Rust JSON helpers with
    secure prototype-property rejection and schema-validator variants.
  - `parseProviderOptions` is covered by a Rust helper that extracts and validates one
    provider-scoped `ProviderOptionsMap` entry through the existing `Schema` runtime validator.
  - `AbstractChat`, `callCompletionApi`, and `convertFileListToFileUIParts` are intentionally
    deferred because they belong to the browser UI transport/state/FileList runtime rather than
    core passive data structures.
  - `gateway` / `createGateway` / `GatewayModelId` are intentionally deferred as a separate
    Vercel Gateway provider-package boundary; Siumai should not add root-level fake exports for
    them without a real provider/gateway client.
- [x] Re-verify GitHub issue `YumchaLabs/siumai#17` against local main.
  - `siumai-protocol-anthropic` already preserves Anthropic extended usage roundtrips locally.
  - `cargo nextest run -p siumai-protocol-anthropic --features anthropic-standard --test anthropic_streaming_feature_surface_test`
    passes on the current branch, so the remaining active gaps are elsewhere.
- [x] Align Anthropic custom provider-key request/response semantics with `repo-ref/ai`.
  - request parsing now merges canonical `providerOptions.anthropic` plus provider-owned custom
    keys such as `my-custom-anthropic`, with the custom key taking precedence
  - top-level non-stream / finish / stream-end `providerMetadata` now duplicates onto the custom
    root only when that custom request key was actually used
- [x] Close the remaining Anthropic provider-defined tool/version drift against `repo-ref/ai`.
  - the shared/provider tool surface now includes `web_search_20260209`,
    `web_fetch_20260209`, `code_execution_20260120`, and `computer_20251124`
  - Anthropic request/header shaping now matches the audited beta-token rules for those versions,
    including `code-execution-web-tools-2026-02-09` and `computer-use-2025-11-24`
  - when `web_*_20260209` is present without an explicit code-execution tool,
    provider-executed `code_execution` now surfaces `dynamic = true` on both non-stream and SSE
    paths, matching the upstream AI SDK `hasWebTool20260209WithoutCodeExecution()` behavior
  - regression verification for these crates must run with the feature gate enabled, e.g.
    `cargo nextest run -p siumai-protocol-anthropic --features anthropic --no-fail-fast` and
    `cargo nextest run -p siumai-provider-anthropic --features anthropic --no-fail-fast`
- [x] Collapse the AI SDK TogetherAI provider split on the main public path.
  - canonical `togetherai` is now the unified provider id for
    chat/completion/embedding/speech/transcription plus provider-owned image and native rerank
  - canonical image generation/edit now follow TogetherAI's provider-owned
    `/images/generations` JSON contract, including `image_url` edits, mask rejection, and typed
    `TogetherAiImageOptions` on the public Rust surface
  - the older `together` id remains as an OpenAI-compatible alias/preset instead of the public
    canonical provider surface
- [x] Promote DeepInfra from a compat-only preset to a first-class AI SDK-style provider surface.
  - canonical `deepinfra` is now a built-in provider id with native metadata and unified
    builder/registry/public-path coverage
  - chat/completion/embedding reuse the shared OpenAI-compatible runtime, while image generation
    and edit now route through DeepInfra-owned `/inference/{model}` and `/openai/images/edits`
  - DeepInfra-specific OpenAI-compatible usage normalization now corrects inconsistent
    reasoning/completion totals before building the stable `Usage` layer
  - the provider-facing Rust package surface now also exposes provider-scoped
    `DeepInfra{Chat,Completion,Embedding,Image}ModelId` aliases so DeepInfra no longer lags the
    other promoted provider packages on the main typed model-id lane
  - `provider_ext::deepinfra` now also exposes package-level unified-provider entry helpers
    `deepinfra()` and `create_deepinfra()`, so the Rust package surface no longer forces callers
    to jump out to unrelated root namespaces just to reach the unified DeepInfra builder
- [x] Promote Vertex MaaS from an implicit compat-only lane to a first-class AI SDK-style provider surface.
  - canonical `vertex-maas` is now a built-in provider id with `google-vertex-maas` and
    `vertex.maas` aliases
  - chat/completion/embedding reuse the shared OpenAI-compatible runtime on Vertex's
    `/endpoints/openapi` base URL derived from `project + location`
  - Google Bearer auth now works through token providers or preexisting `Authorization` headers
    without requiring a fake non-empty API key
- [x] Close the remaining stable provider-typing gap around Google Vertex wrappers.
  - `ProviderType::{Vertex, AnthropicVertex, VertexMaas}` now all exist
  - provider catalog / retry / validator layers no longer downgrade `vertex` or
    `anthropic-vertex` to `Custom(...)`
- [x] Close the next stable provider-typing gap around built-in native AI SDK-style providers.
  - `ProviderType::{Azure, Cohere, TogetherAi, Bedrock}` now all exist
  - provider catalog / retry / validator layers no longer degrade those built-in providers to
    `Custom(...)`
- [x] Promote the next AI SDK-packaged OpenAI-compatible providers to first-class stable identity.
  - `ProviderType::{Mistral, Fireworks, Perplexity}` now all exist
  - provider catalog / retry / validator / unified-interface checks no longer degrade those ids to
    `Custom(...)`
  - the registry/catalog story now keeps those AI SDK-packaged vendors on first-class provider
    identity even while execution still reuses the shared OpenAI-compatible runtime
- [x] Remove provider-wide default-model fallback from the unified Google Vertex wrappers.
  - `Siumai::builder().vertex()` and `Siumai::builder().anthropic_vertex()` now require explicit
    `model` ids, matching AI SDK's family-specific model constructors instead of injecting
    ambiguous provider defaults
- [x] Promote Cohere from the old rerank-only/compat-embedding split into a first-class AI SDK
  unified provider surface.
  - AI SDK `@ai-sdk/cohere` exposes one provider for `languageModel()` + `embeddingModel()` +
    `rerankingModel()`
  - canonical `cohere` now routes chat/embedding/rerank through the native `/v2` provider crate
  - public builder/registry/catalog/metadata/tests now all lock the unified provider story
  - the OpenAI-compatible Cohere preset remains opt-in compatibility only, and the native unified
    path now requires explicit model ids
- [x] Lock the image-architecture rule to the audited AI SDK boundary.
  - AI SDK does expose a shared image interface through `ProviderV4.imageModel(...)`,
    `ImageModelV4`, and `generateImage(...)`
  - that shared layer is intentionally a call-shape/result-shape contract, not a generic
    provider-agnostic execution contract
  - hybrid providers such as `fireworks`, `deepinfra`, and `togetherai` should therefore keep
    image generation/edit on provider-owned runtimes even when their text families still reuse the
    shared OpenAI-compatible stack
  - follow-up audits should treat any regression back to a generic compat image executor as
    architecture drift unless the upstream AI SDK package itself consolidates that runtime
- [x] Audit the remaining package-boundary drift for newly promoted AI SDK compat providers.
  - `mistral` and `perplexity` now keep the audited chat-only package boundary on the public/runtime
    capability surface instead of inheriting generic compat completion support
  - canonical top-level builders now also have no-network public-path anchors on those audited
    package boundaries: `mistral` top-level chat + chat-stream + embedding and `perplexity`
    top-level chat + chat-stream all converge with config-first + registry routing, while both
    providers still reject
    `completion_model(...)`
  - `mistral` now also has an explicit embedding default (`mistral-embed`) on the shared compat
    family-default table
  - audited Mistral language-model option typing is now also promoted on the public surface:
    `provider_ext::mistral::{MistralChatOptions, MistralLanguageModelOptions,
    MistralChatRequestExt}` mirror the AI SDK package-owned chat option lane without raw
    `providerOptions.mistral` JSON
  - audited Mistral compat request shaping now also follows the AI SDK package defaults more
    closely: built-in `mistral` config/runtime defaults preserve JSON Schema structured outputs,
    and provider-owned `safePrompt`, document limits, `structuredOutputs`, `parallelToolCalls`,
    and `reasoningEffort` normalize onto the wire contract on the public/config/registry paths
  - `mistral` now also has a dedicated package-alignment workstream under
    `docs/workstreams/mistral-package-surface-alignment/`, so its audited `chat + embedding`
    boundary is documented separately from the generic compat-family review
  - audited Mistral and Perplexity curated model subsets are now also promoted into provider-owned
    Rust constants and reused by compat defaults plus the public provider catalog instead of
    relying on stale ad hoc strings such as the old Perplexity `llama-3.1-sonar-small-128k-online`
    default
  - audited Perplexity provider-owned typed options now also keep an AI SDK-style camelCase public
    surface (`PerplexityOptions`, `PerplexityWebSearchOptions`), while the shared compat boundary
    explicitly lowers known fields such as `searchMode`, `returnImages`, and
    `webSearchOptions.searchContextSize` onto Perplexity's snake_case wire contract only at
    transport time
  - `fireworks` now also mirrors the audited AI SDK unified provider surface on the public/runtime
    path: chat/completion/embedding/transcription still reuse the shared OpenAI-compatible runtime,
    while image generation/edit now route through provider-owned Fireworks workflow and
    `image_generation` endpoints under the canonical `fireworks` id
  - audited Fireworks language-model option typing is now also promoted on the public surface:
    `provider_ext::fireworks::{FireworksChatOptions, FireworksLanguageModelOptions,
    FireworksChatRequestExt}` mirror AI SDK `thinking` / `reasoningHistory` without raw JSON
  - the audited Fireworks type surface now also includes the package-owned empty embedding option
    object and deprecated alias names:
    `FireworksEmbeddingModelOptions`, `FireworksProviderOptions`,
    `FireworksEmbeddingProviderOptions`
  - audited Fireworks curated chat/completion/embedding/image model subsets are now also promoted
    into provider-owned Rust constants and reused by the public provider catalog instead of being
    maintained as separate ad hoc string lists
  - the shared compat Fireworks preset now also advertises `completion` on the config/registry
    path, so generic compat Fireworks clients no longer drift from the audited package boundary
  - the shared compat TogetherAI/Together and DeepInfra presets now also advertise `completion`
    explicitly in static provider metadata, so completion-capable hybrid wrappers no longer depend
    on capability inference alone for AI SDK package-boundary parity
  - public Rust package facades for compat-wrapped AI SDK packages now also expose provider-scoped
    `*Client/*Config` aliases for `mistral`, `perplexity`, `fireworks`, and `deepinfra`, making
    side-by-side package-boundary review less dependent on the generic compat module names
  - audited AI SDK package facades now also expose package-level provider entry helpers in their
    own namespaces instead of forcing callers back to root-only entrypoints:
    `provider_ext::mistral::{mistral, create_mistral}`,
    `provider_ext::perplexity::{perplexity, create_perplexity}`,
    `provider_ext::fireworks::{fireworks, create_fireworks}`, and
    `provider_ext::moonshotai::{moonshotai, create_moonshotai}` on the compat-promoted lane, plus
    native/provider-owned package helpers such as `provider_ext::openai::{openai, create_openai}`,
    `provider_ext::anthropic::{anthropic, create_anthropic}`,
    `provider_ext::azure::{azure, create_azure}`,
    `provider_ext::google::{google, create_google}`,
    `provider_ext::bedrock::{bedrock, create_amazon_bedrock}`,
    `provider_ext::cohere::{cohere, create_cohere}`,
    `provider_ext::togetherai::{togetherai, create_togetherai}`,
    `provider_ext::google_vertex::{vertex, create_vertex}`,
    `provider_ext::groq::{groq, create_groq}`, `provider_ext::xai::{xai, create_xai}`, and
    `provider_ext::deepseek::{deepseek, create_deepseek}`
  - the provider/runtime Google alias lane now also matches the audited `@ai-sdk/google` entry
    naming more closely: `Provider::google()` and `Siumai::builder().google()` are stable aliases
    of the existing Gemini runtime, and `provider_ext::google::{google, create_google}` mirrors
    the package-level `google` / `createGoogle` export names instead of exposing only `gemini`
  - generic `provider_ext::openai_compatible` intentionally does **not** grow a fake zero-arg
    `create_*` helper: upstream `createOpenAICompatible(settings)` is a generic lower-level
    factory rather than a provider package facade, so the closer Rust equivalent remains the
    existing `OpenAiCompatibleBuilder` / `OpenAiCompatibleConfig` /
    `OpenAiCompatibleClient::from_config(...)` surface
  - `moonshotai` now also mirrors its dedicated AI SDK chat-only wrapper more closely:
    canonical public/runtime id is `moonshotai`, `moonshot` is retained only as a hidden compat
    alias, `Provider::moonshotai()` / `Siumai::builder().moonshotai()` plus
    `provider_ext::moonshotai::{MoonshotAIClient, MoonshotAIConfig, model_sets, recommended}`
    now exist on the public Rust surface, and the shared compat runtime explicitly keeps
    completion/image/embedding unsupported on that wrapper boundary
  - audited MoonshotAI typed request shaping is now also promoted on the public surface:
    `provider_ext::moonshotai::{MoonshotAIChatOptions, MoonshotAILanguageModelOptions,
    MoonshotAIChatRequestExt}` mirror the AI SDK `thinking` / `reasoningHistory` lane without raw
    `providerOptions.moonshotai` JSON
  - audited MoonshotAI compat request shaping now also follows the AI SDK package defaults more
    closely: request normalization maps `thinking.budgetTokens -> thinking.budget_tokens` and
    `reasoningHistory -> reasoning_history`, canonical default/base-url metadata now resolve to
    `moonshotai` with hidden alias fallback, and the curated Kimi K2 + Moonshot V1 model subset
    is reused by public defaults plus the provider catalog instead of older preview-style names
  - TypeScript-only package exports such as `MistralProviderSettings`,
    `PerplexityProviderSettings`, `MoonshotAIProviderSettings`, and per-package `VERSION` remain
    intentionally deferred on those compat facades because Rust already treats `Config` / builder
    inputs as the stable provider-settings contract across aligned packages
  - lower-contract URL alignment is now also pinned for the audited completion-capable compat
    wrappers: `togetherai`, `deepinfra`, `fireworks`, and `vertex-maas`
  - Vertex MaaS lower-contract coverage now uses the real project/location-derived
    `/endpoints/openapi` base URL instead of the placeholder compat config URL
  - remaining follow-up is narrower now: keep auditing provider-owned text/image option shaping
    and curated model coverage for the newly promoted compat vendors, and decide later whether the
    hidden low-level `moonshot` alias should be deleted entirely after downstream migration
- [x] Re-run provider-crate unit-test compilation on the real feature-gated surfaces.
  - the built-in and compat-wrapped provider crates now compile under their audited provider
    features instead of only the default workspace feature mix
  - xAI file/video tests were updated for `FileUploadRequest.filename: Option<String>` and the
    provider-owned `XaiVideoRequestExt` request helper
  - MiniMaxi regression tests now distinguish `VideoGenerationRequest.prompt: Option<String>` from
    `MusicGenerationRequest.prompt: String`
  - Azure native completion metadata now materializes the shared `ResponseMetadata.headers` field,
    matching the newer stable response-metadata surface
- [x] Re-run protocol and top-level facade compilation on real `all-features` combinations.
  - `siumai-core --all-features`, `siumai-protocol-{openai,anthropic,gemini} --all-features`,
    and `siumai --all-features` now all compile on the audited test surfaces
  - remaining stale tests were refreshed to the current shared contracts:
    Anthropic protocol streaming fixtures, top-level experimental bridge/transcoding tests, and
    MiniMaxi/public retry file-upload tests now all accept `ResponseMetadata.headers` and
    optional upload filenames
- [x] Close the remaining package-level provider-settings gap for native OpenAI / Azure / Bedrock.
  - provider-owned and public facades now expose
    `OpenAIProviderSettings`, `AzureOpenAIProviderSettings`, and
    `AmazonBedrockProviderSettings`
  - the same audited package boundaries now also expose `VERSION` on the Rust side
  - supported vs deferred upstream fields are now tracked under
    `docs/workstreams/provider-settings-surface-alignment/`
- [x] Extend the package-level provider-settings pass to native Cohere.
  - provider-owned and public facades now expose `CohereProviderSettings` plus `VERSION`
  - supported `baseURL` / `apiKey` / `headers` / `fetch` fields now have a direct carrier
  - upstream `generateId` remains documented as deferred until Cohere has a real stable-ID hook
- [x] Extend the package-level provider-settings pass to native DeepSeek and TogetherAI.
  - provider-owned and public facades now expose `DeepSeekProviderSettings` and
    `TogetherAIProviderSettings` plus `VERSION`
  - supported `apiKey` / `baseURL` / `headers` / `fetch` fields now have direct carriers

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
- [x] Add a runtime-only `includeRawChunks` request lane for chat/text streaming.
  - stable `StreamRequestOptions` now carries raw-chunk behavior outside provider wire payloads
  - `siumai::text::StreamOptions.include_raw_chunks` now maps onto `ChatRequest.stream_options`
- [x] Fix eager JSON stream-end synthesis on the shared transport executor.
  - JSON transport stream end-events are now synthesized only after the upstream body actually
    drains, so stateful converters keep accumulated terminal response content on clean EOF
  - Bedrock reserved-JSON structured-output extraction now stays aligned with AI SDK expectations
    on clean EOF instead of returning an empty `StreamEnd.response`
- [x] Emit stable `raw` stream parts for the main AI SDK chat parser lanes.
  - OpenAI-compatible chat chunks now emit `stream-start -> raw -> response-metadata -> ...`
  - Anthropic Messages SSE now emits `stream-start -> raw -> response-metadata -> ...`
  - Gemini GenerateContent SSE now emits runtime-opt-in
    `stream-start -> raw -> text|reasoning|file|source|...` on the audited parser lane
  - Native Bedrock Converse JSON streaming now emits
    `stream-start -> response-metadata -> raw -> ...` on the first parsed chunk, matching the
    upstream Bedrock `start()` preamble more closely while also carrying request warnings,
    default-model metadata, provider-error `error` parts, and streamed `stopSequence` /
    `raw_finish_reason` retention on the terminal response
  - first-chunk parse failures now also preserve the upstream lifecycle instead of skipping the
    stable stream start on the audited lanes:
    OpenAI-compatible, Anthropic, native OpenAI completion, and native Cohere keep
    `stream-start -> raw? -> parse-error`, native Bedrock now keeps its first-chunk preamble
    before optional `raw` and the parse error, and Gemini now keeps
    `stream-start -> raw? -> error` on invalid-JSON / top-level-error chunks, with EOF fallback
    later closing active text/reasoning lanes and emitting `finish(unknown)`
- [x] Narrow the typed stream-part protocol downgrade API.
  - `LanguageModelV3StreamPart::to_protocol_custom_event(...)` is now the canonical
    provider-native serializer lowering hook
  - `to_custom_event(...)` remains only as a thin compatibility alias instead of being the
    primary documented API name
  - OpenAI Responses, Anthropic, and Gemini serializers now call the explicit protocol-lowering
    API directly
- [x] Promote legacy V3 non-tool stream-part payloads onto the runtime semantic lane.
  - `OpenAiResponsesStreamPartsBridge` now upgrades stable-shape `raw`, `custom`, `file`, and
    `reasoning-file` payloads via `LanguageModelV3StreamPart::to_part_event()` instead of
    preserving them as loose provider-scoped `Custom` events
  - targeted verification now covers both `siumai-core` bridge unit tests and the top-level
    Gemini/Vertex stream bridge roundtrip fixture test
- [x] Freeze the Axum SSE stable-part transport envelope.
  - `event: part` is now the preferred semantic export lane for Siumai-owned Axum SSE adapters
  - both `Part` and `PartWithReplay` now serialize through one stable `{ part, replay }` JSON
    envelope instead of switching shape by runtime event kind
  - direct tests now pin `replay: null` for plain `Part` and populated replay payloads for
    `PartWithReplay`
- [x] Finish the main native Bedrock stable-part migration on the audited Converse JSON lane.
  - streaming now emits stable `text-*`, `reasoning-*`, `tool-input-*`, `tool-call`, and
    terminal `finish` parts while preserving the older legacy shadow deltas/events for
    compatibility
  - terminal Bedrock finish parts now carry cache-aware usage plus Bedrock provider metadata such
    as `trace`, `performanceConfig`, `serviceTier`, `cacheWriteInputTokens`, `cacheDetails`,
    `stopSequence`, and `isJsonResponseFromTool`
  - non-stream Bedrock chat responses now also retain reasoning provider metadata, default-model
    identity, request warnings, and upstream-style Mistral tool-call-id normalization
- [x] Finish the audited native Bedrock request/options alignment pass for the main Converse lane.
  - `provider_ext::bedrock` now exposes typed `BedrockReasoningConfig`,
    `BedrockReasoningEffort`, `BedrockReasoningType`, and `BedrockServiceTier` alongside
    `BedrockChatOptions`
  - request shaping now preserves unknown top-level `providerOptions.bedrock` passthrough fields
    instead of dropping everything except `additionalModelRequestFields`
  - Anthropic Bedrock requests now derive `additionalModelResponseFieldPaths`,
    `additionalModelRequestFields.thinking`, `anthropic_beta`, top-level `serviceTier`, and the
    upstream `maxReasoningEffort` routing split (`output_config.effort`,
    `reasoning_effort`, nested `reasoningConfig`)
  - Anthropic structured JSON output now also prefers native
    `additionalModelRequestFields.output_config.format` on the same audited Bedrock routes as the
    AI SDK, including the thinking-enabled fallback path on older Anthropic Bedrock models
  - fixture/public-surface coverage now pins the new Anthropic request field path and typed
    provider option serialization
- [x] Finish the audited native Bedrock prompt/message conversion pass for the main Converse lane.
  - message-level `providerOptions.bedrock.cachePoint` now survives on leading
    system/developer blocks plus user/tool/assistant blocks, and both `cachePoint` with
    `cache_point` input aliases are accepted on that request path
  - user `file` parts now map to Bedrock `document` / `image` blocks like the AI SDK Bedrock
    converter, strip filenames from the first dot for stable document names, and honor typed
    Bedrock document citations through `ContentPart::File.providerOptions.bedrock.citations`
  - assistant reasoning replay now reads canonical `providerOptions.bedrock.signature` /
    `redactedData`, preserves signed reasoning bytes without trimming, keeps empty text separators
    when reasoning blocks are present, and trims only the final unsigned Bedrock reasoning block
    where upstream does
  - response-side Bedrock reasoning metadata now also has a typed public/replay path:
    `BedrockContentPartExt::bedrock_reasoning_metadata()` reads typed `signature` /
    `redactedData` from reasoning parts, and
    `provider_ext::bedrock::assistant_message_with_reasoning_metadata(...)` converts those
    response-side fields back into replayable request-side `providerOptions.bedrock`
  - tool-result `content` now supports Bedrock `text` plus `image-data`, and request-side Mistral
    tool ids are normalized on both assistant tool calls and tool-result `toolUseId` values
- [x] Finish completion-family `includeRawChunks` / `raw` parity on the new `/completions`
  stream lane.
  - `CompletionRequest.stream_options` and `siumai::completion::StreamOptions.include_raw_chunks`
    now carry runtime-only raw-chunk intent without leaking it into provider wire payloads
  - OpenAI-compatible, native OpenAI, and native Azure completion SSE now emit
    `stream-start -> raw -> response-metadata -> text-start -> text-delta ... -> text-end ->
    finish` on the stable part lane while preserving legacy `ContentDelta` / `StreamEnd`
- [x] Expose the runtime stream-request structure on the stable public facade.
  - `StreamRequestOptions` is now reachable from `siumai::prelude::unified::*`,
    `siumai::completion::*`, and `siumai::text::*`
  - public-surface compile coverage now locks explicit request-level `includeRawChunks`
    construction instead of relying only on helper-layer `StreamOptions`

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
  - the public compat package surface now also mirrors the upstream export name through
    `MetadataExtractor`, so facade/package audits no longer have to special-case the Rust-only
    trait name
  - the same package surface now also exposes a generic `ProviderErrorStructure<T>` helper for
    AI SDK-style provider error decoding/message extraction instead of leaving that exported data
    structure unrepresented on the Rust side
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
- [x] Align the provider-owned DeepSeek streaming default with AI SDK `@ai-sdk/deepseek`.
  - provider-owned DeepSeek remains a chat-only package surface on builder/config/registry/public
    paths; embedding/image/rerank stay intentionally unsupported
  - native DeepSeek stream requests now always send `stream_options.include_usage = true`, matching
    the upstream package's hardcoded stream request body instead of diverging between public paths
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
  - audited known compat chat options now also follow AI SDK mapping semantics from deprecated
    `openai-compatible`, canonical `openaiCompatible`, and provider-owned keys:
    `user`, `reasoningEffort`, `textVerbosity`, and `strictJsonSchema` now map to wire
    `user`, `reasoning_effort`, `verbosity`, and `response_format.json_schema.strict`
  - provider-defined tools now also emit AI SDK-style
    `unsupported { feature: "provider-defined tool <id>" }` warnings on the default compat
    runtime response path while still being filtered from Chat Completions requests
  - legacy `providerOptions['openai-compatible']` now also emits the AI SDK-style deprecation
    warning while preserving the audited compatibility lane for known compat chat options
- [x] Align DeepInfra custom text-family base-URL semantics with AI SDK `@ai-sdk/deepinfra`.
  - shared compat DeepInfra text config now normalizes root, `/openai`, and `/inference` inputs
    onto the canonical `/openai` text-family prefix instead of treating a root base URL as the
    final compat API prefix
  - provider-owned compat builder/config plus top-level builder/provider/registry paths now emit
    equivalent `/openai/chat/completions` requests when callers pass a root DeepInfra base URL
  - public streaming coverage now also locks that request equivalence while keeping
    `includeRawChunks` runtime-only and preserving finish-time `metadataExtractor` merging on the
    DeepInfra public/provider-owned stream lanes
- [x] Align OpenAI-compatible image provider options and warning semantics with AI SDK.
  - compat image generation/edit/variation now merge provider-owned options from deprecated
    `openai-compatible`, canonical `openaiCompatible`, and provider-owned keys instead of only
    `providerOptions.openai|azure`
  - compat image generation now emits stable `unsupported { feature: "seed" }` warnings instead
    of silently dropping `seed`
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
- [x] Finish the Anthropic request-side typed-option parity sweep against
  `repo-ref/ai/packages/anthropic/src/anthropic-messages-options.ts`.
  - the typed request-side surface now covers AI SDK-shaped `thinking`
    (`adaptive | enabled | disabled`), `sendReasoning`, `disableParallelToolUse`,
    `cacheControl`, `metadata.userId`, `mcpServers`, `contextManagement`, `toolStreaming`,
    `effort`, `speed`, and `anthropicBeta`
  - builder/provider/config/shared-facade defaults now take typed
    `AnthropicContextManagementConfig` instead of raw JSON payloads
  - `container.skills` now uses a typed `AnthropicContainerSkillType` and accepts upstream
    `skillId` camelCase input during provider-option deserialization, so the public typed surface
    is closer to AI SDK than the previous wide-string placeholder
  - Anthropic enabled-thinking request shaping now also matches the upstream
    `maxOutputTokens + thinkingBudget` semantics on the final request body, including legacy
    specific-params paths that previously skipped that adjustment
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
- [x] Close the remaining shared non-chat structure gaps after the xAI provider-owned parity pass.
  - [x] Refactor `ImageEditRequest` into a typed multi-input image surface closer to AI SDK
    `files[]` + `mask` semantics.
    - shared `ImageEditInput` / `ImageEditFileData` now model file-backed and URL-backed edit
      inputs
    - xAI image edit now maps 1 input -> `image` and many inputs -> `images`
    - OpenAI/OpenAI-compatible multipart and Vertex inline edit paths now accept multiple
      file-backed source images
  - [x] Decide how URL-backed image edit inputs should be materialized on multipart/inline
    provider paths.
    - xAI already accepts URL-backed edit inputs directly
    - shared `HttpImageExecutor` now materializes `data:` / `http:` / `https:` URL-backed
      edit/variation inputs before synchronous OpenAI/OpenAI-compatible/Vertex
      multipart/inline transformers run
  - [x] Close the shared image call-option structure gaps against AI SDK V4.
    - shared Rust image requests now expose top-level `aspectRatio` across generation/edit/variation
    - shared Rust image requests now expose `seed` across generation/edit/variation rather than
      only on generation
    - typed `ImageEditInput` file/url inputs now also expose per-input `providerOptions`
    - `ImageVariationRequest` now carries a typed file/url image input instead of a raw byte-only
      field, bringing it closer to AI SDK `ImageModelV4File`
  - [x] Finish the remaining provider/runtime adoption of the shared image call-option surface.
    - OpenAI/OpenAI-compatible now surface AI SDK-style unsupported `aspectRatio` / `seed`
      warnings on generation/edit/variation
    - xAI/Google/Vertex supported image paths now consume canonical top-level `aspectRatio` / `seed`
    - Vertex Imagen variation now also has a native variation-specific request transformer on the
      shared image-variation surface, and builder/config/registry/public-path parity now covers
      data-url-backed requests on that route
  - [x] Stabilize the shared AI SDK-style video knob/input surface instead of leaving it as an
    open design question.
    - `VideoGenerationRequest` now carries canonical `count` (`n`), `fps`, `seed`, and typed
      `VideoGenerationInput` image/video inputs
    - typed `VideoGenerationInput` file/url inputs now also expose per-input `providerOptions`
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
  - [x] Promote video to a formal family-model / registry surface instead of leaving it on
    extension-only handles.
    - `siumai-core` now exposes task-oriented `VideoModelV3` / `VideoModelV4` / `VideoModel`
    - `siumai-registry` now exposes dedicated `video_model_family_with_ctx(...)`,
      `ProviderRegistryHandle::video_model(...)`, and `VideoModelHandle`
    - `siumai::video::{create_task, query_task}` now provides the stable facade helper lane
    - `siumai::video::{wait_for_task, generate}` now also provides a Rust-first polling helper
      lane above the same task-oriented contract
    - the older `LanguageModelHandle` video capability remains as a compatibility bridge, and the
      remaining AI SDK gap is now the smaller result/runtime boundary around provider-owned final
      assets rather than model construction, batching, URL materialization, or basic auto-polling
  - [x] Add provider-owned materialization adapters for the audited provider-reference-only video
    results.
    - shared `MaterializedVideoAsset` now exists on the video type surface
    - `VideoGenerationCapability` / `VideoModelV3` now expose
      `materialize_video_reference(...)`
    - `siumai::video::generate(...)` now best-effort materializes audited provider references
      through the same model-capability dispatch chain
    - Gemini and MiniMaxi now reuse their existing file-management runtimes for that path
    - providers that still need a separate authenticated download runtime (for example current
      Vertex `gs://...` video outputs) remain intentionally deferred on the raw URL-backed path
  - [x] Refactor the shared transcription/audio-input surface toward AI SDK V4.
    - shared STT and audio-translation requests now use canonical
      `audio + mediaType + providerOptions`
    - stable request types no longer expose the old `audio_data | file_path` split
    - local file-path materialization moved behind helper-level `transcribe_file(...)` /
      `translate_file(...)` instead of living on the stable request shape
    - `AudioTranslationRequest` was updated in the same pass so STT/translation stay aligned
  - [x] Tighten the shared transcription `mediaType` field to required for full AI SDK parity.
    - stable `SttRequest` / `AudioTranslationRequest` now require `mediaType`
    - `from_audio(...)` / `from_base64(...)` constructors now take `mediaType`
    - helper/file-based convenience paths now infer `mediaType` eagerly or fail fast

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
- [x] Accept the newer AI SDK provider-reference aliases on tool-result content:
  - `file-reference`
  - `image-file-reference`
  - `providerReference` field aliases on the provider-keyed file reference payload
- [x] Flip canonical tool-result provider-reference emission to AI SDK names.
  - legacy `file-id` / `image-file-id` plus `fileId` remain accepted as compatibility aliases
  - stable Rust serialization now emits canonical
    `type: "file-reference" | "image-file-reference"` with `providerReference`
- [x] Add `providerOptions` support to tool-result output/content shapes where AI SDK prompt types
  allow them.
- [x] Promote first-class `providerReference` on user `file` / `image` prompt parts.
  - stable prompt/content now models provider-owned file/image references directly through
    `FilePartSource::ProviderReference` and shared `ProviderReference`
  - builder/helper coverage now includes `ContentPart::{image,file}_provider_reference(...)` plus
    `ChatMessageBuilder::{with_image,with_file}_provider_reference(...)`
  - OpenAI Chat/Responses, Anthropic Messages, and bridge normalization now all map native
    `file_id` request shapes onto canonical provider references on the stable side
  - OpenAI Responses still keeps the upstream `fileIdPrefixes` compatibility option on the
    provider runtime, but the bridge no longer needs to emit that deprecated hint once the stable
    prompt already carries canonical provider references
- [x] Review whether text/reasoning/tool-call/tool-result/source/tool-approval parts should expose
  both input-side `providerOptions` and output-side `providerMetadata`, or whether a narrower split
  is safer in Rust.
  - current recommendation: keep the shared stable content superset, but treat
    `providerOptions` as the canonical request-time channel and `providerMetadata` as
    response-time observation only
- [x] Align the shared `providerMetadata` root shape with AI SDK `ProviderMetadata`.
  - `siumai-spec::types::ProviderMetadataMap` now serves as the shared provider-rooted metadata
    map for response/content/stream/upload structures
  - `ChatResponse`, `CompletionResponse`, `ContentPart`, `ChatStreamPart`, `uploadFile`, and
    `uploadSkill` now all preserve `provider_id -> object` semantics instead of ad hoc nested
    map shapes
  - OpenAI-compatible/OpenAI/Anthropic typed metadata accessors now read those provider-rooted
    objects directly
  - `UIMessage.providerMetadata` intentionally remains on the request-side
    `ProviderOptionsMap` story because AI SDK `convertToModelMessages()` treats it as
    `providerOptions`
- [x] Tighten `UiToolPart` toward AI SDK's state-discriminated union semantics.
  - kept the current serde/public compatibility story for existing callers
  - `UiToolPart::validate()` plus `validate_ui_messages()` now enforce the state-specific
    required/forbidden field matrix for `approval`, `output`, `errorText`, `rawInput`,
    `resultProviderMetadata`, and `preliminary`
  - Rust now also exposes a public typed `UiToolInvocation` / `UiToolInvocationState` overlay so
    callers can work with a true state-discriminated union shape without giving up the existing
    wide serde-compatible `UiToolPart`
- [x] Add schema-aware UI validation parity for AI SDK `validateUIMessages(...)`.
  - `validate_ui_messages_with_schemas(...)` now keeps the existing structural validation and can
    additionally validate message metadata, `data-*` parts, and static tool input/output payloads
    against caller-supplied schemas
  - `siumai-core` intentionally stays free of a hard `jsonschema` dependency by accepting a
    caller-provided schema validator callback/adapter instead of compiling schemas itself
- [x] Add tool-aware UI conversion parity for AI SDK `convertToModelMessages({ tools })`.
  - current Rust UI conversion now matches the audited default `output-error` split
    (`providerExecuted -> error-json`, local tool result -> `error-text`)
  - `ExecutableTool` / `ExecutableTools` now carry a runtime `to_model_output` mapper, and
    `convert_to_model_messages_with_tooling` / `convert_to_chat_request_with_tooling` apply it
    during UI-to-model conversion
  - remaining gap is API-shape-level: Rust currently exposes a dedicated synchronous tooling-aware
    helper instead of AI SDK's inline async `tools` option
- [x] Tighten the shared function-tool schema toward AI SDK `Tool.inputSchema/outputSchema`.
  - `ToolFunction` still keeps `parameters` as the internal storage field, but stable public
    serialization now emits AI SDK-style `inputSchema` while deserialization remains backward
    compatible with legacy `parameters` and snake_case `input_schema`
  - higher-level Rust helpers also expose explicit `input_schema()` / `with_input_schema(...)`
    accessors so AI SDK naming is no longer just a comment-only concept
  - optional AI SDK-style `outputSchema` metadata now lives on the shared function-tool shape and
    can be attached through `Tool` / `ExecutableTool` helpers without changing current provider
    request shaping
  - OpenAI Responses request normalization now also preserves `inputSchema`, `outputSchema`, and
    `inputExamples` when callers provide AI SDK-style function-tool JSON
  - provider/request shapers intentionally keep emitting provider-native `parameters` on the wire;
    the alignment change is limited to the stable portable Rust JSON surface
- [x] Add the missing AI SDK provider-tool deferred-result metadata to the shared portable shape.
  - `ProviderDefinedTool` now carries optional `supportsDeferredResults`
  - the audited Anthropic deferred-result tool factories now mark the upstream-known versions
    explicitly (`web_search_{20250305,20260209}`, `web_fetch_{20250910,20260209}`,
    `tool_search_*_20251119`, `code_execution_{20250825,20260120}`)
  - the audited high-level Rust runtime now consumes that metadata on the main orchestration paths:
    deferred provider tool calls stay pending across steps, response-native provider `tool-result`
    parts populate `StepResult.tool_results`, and gateway tool loops no longer terminate early just
    because no local client tool executed in the deferred-provider case
- [x] Carry the remaining audited AI SDK local-tool runtime metadata on `ExecutableTool` and consume
  it on the main orchestrator paths.
  - `ExecutableTool` / `ExecutableTools` now expose runtime-only metadata for `dynamic`,
    `contextSchema`, `needsApproval`, and `onInputStart` / `onInputDelta` / `onInputAvailable`
    without polluting the stable wire-facing `Tool` shape
  - `ToolResolver::runtime_tool_metadata(...)` lets extras consume that runtime metadata without
    forcing all existing resolvers to change shape
  - non-stream and stream orchestrator paths now surface local `tool-approval-request` parts when
    runtime approval is required without the legacy Rust callback, invoke `onInputAvailable` on
    legal local tool calls, invoke streamed `onInputStart` / `onInputDelta` / `onInputAvailable`
    callbacks on the first streamed step, and preserve runtime-dynamic tool flags on
    `StepResult.dynamic_tool_calls()` / `dynamic_tool_results()`
  - first-turn approval continuity now also mirrors the audited AI SDK
    `collectToolApprovals()` / `generateText()` / `streamText()` behavior more closely:
    only the last `role=tool` message is scanned for `tool-approval-response`, approvals with an
    already-present tool result are skipped, approved local tools execute before the next model
    call, denied approvals synthesize `execution-denied`, and provider-executed approvals are
    forwarded back into the next prompt while denied provider approvals additionally carry
    `output.providerOptions.openai.approvalId`
  - remaining intentional gap is narrower now: `contextSchema` is still parity/type-surface
    metadata rather than enforced runtime validation
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
- [x] Add first-class stable stream support for semantics that were commonly pushed through
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
- [x] Add adapters between the runtime stream event layer and the chosen V4-capable stream-part
  layer.
  - [x] `LanguageModelV3StreamPart::{from_runtime_part,to_runtime_part}` are now both public so
    bridges/gateways can move explicitly between the typed overlay and stable runtime
    `ChatStreamPart` contract without round-tripping through provider-prefixed custom JSON
  - [x] unit coverage now locks that dual adapter on runtime tool-call roundtrips
- [x] Expose AI SDK-aligned `LanguageModelV4*` public aliases for the upgraded typed stream-part
  overlay so new code no longer has to use the historical `LanguageModelV3*` names.
- [x] Ensure bridge/gateway serializers use that stronger stream-part contract where appropriate.
  - [x] extras Axum SSE adapters now surface direct runtime `Part` / `PartWithReplay` as explicit
    `event: part` frames instead of dropping the upgraded semantic lane
  - [x] extras high-level consumers (`stream_object`, tool-loop gateway) now consume stable tool
    lifecycle parts before falling back to legacy deltas, with source-aware deduplication for
    mixed streams
  - [x] extras text consumers (`stream_object`, tool-loop assistant-history accumulation,
    streamed orchestrator fallback, Axum `to_text_stream()`) now consume stable `TextDelta`
    parts directly instead of depending on legacy `ContentDelta` shadows
  - [x] shared stream wrappers (`StreamFactory`, `SimulateStreamingMiddleware`) now treat stable
    `TextDelta` parts as existing text before synthesizing fallback legacy deltas, avoiding
    duplicate tail text on semantic-only streams
  - [x] public streaming examples, migration snippets, gateway transform examples, and
    bridge/transcode tests now match stable `Part(TextDelta)` / `PartWithReplay(TextDelta)` as
    first-class streamed text instead of teaching legacy-only `ContentDelta` consumers
  - [x] `OpenAiResponsesStreamPartsBridge` now promotes parseable legacy/custom v3 payloads onto
    stable `Part` / `PartWithReplay` events instead of only renaming them into `openai:*`
    custom prefixes, so OpenAI Responses gateway paths default to the stronger semantic contract
  - [x] bridge primitive remappers now also rewrite tool ids/names on direct
    `Part` / `PartWithReplay` events and drop stale OpenAI Responses `rawItem` replay hints when a
    semantic remap would otherwise leave replay metadata inconsistent with the stable part
- [x] Normalize serializer re-entry so `Part -> Custom` compatibility bridging does not deadlock
  when protocol serializers hold internal state locks.
- [x] Migrate provider parsers to emit `ChatStreamEvent::Part` directly where the richer semantic
  contract is available at parse time.
  - [x] OpenAI Responses major stable semantics
  - [x] Anthropic `stream-start` / `response-metadata` / `text-*` / local `tool-input-*` /
    `tool-call` / `reasoning-*` / `source` / successful `finish`
  - [x] Gemini `stream-start` / `text-*` / `reasoning-*` / `file` / `reasoning-file` / `source` /
    provider-executed tool semantics / successful `finish` / top-level `error`
    - request-aware Gemini/Vertex stream transformers now forward
      `ChatRequest.stream_options.include_raw_chunks` into the parser and emit stable `raw` parts
      on that runtime lane
    - Gemini parser-side text now stays stable-first (`TextDelta`) and shared shadow expansion owns
      the compatibility `ContentDelta`, avoiding duplicate text on direct parser consumers
    - mixed stable/legacy text and reasoning streams are now deduplicated by first source in
      `StreamProcessor`, so Gemini compatibility shadows no longer double-count final output
  - [x] OpenAI-compatible `stream-start` / `response-metadata` / `text-*` / `reasoning-*` /
    `finish` lifecycle semantics on the direct runtime `Part` lane
    - EOF / `[DONE]` fallback on the compat lane now also closes active `text-*` /
      `reasoning-*` parts, finalizes unfinished tool-call lifecycles, and emits a stable
      `finish` part instead of only a legacy `StreamEnd`
    - explicit top-level compat `{"error": ...}` chunks and invalid JSON chunks now also
      terminate on stable `error` plus error `finish` / `StreamEnd` semantics instead of
      surfacing only transport parse failures
    - compat stream regression coverage now also pins finish-time
      `acceptedPredictionTokens` / `rejectedPredictionTokens` and public metadata-extractor
      merging on the streaming path
  - [x] Anthropic provider-hosted server tool / MCP stable-part strategy
  - [x] Anthropic reasoning signature/redacted cleanup on the stable part lane
    - `signature_delta` now maps to `reasoning-delta.providerMetadata.anthropic.signature`
    - `redacted_thinking` replay now uses `reasoning-start.providerMetadata.anthropic.redactedData`
    - no dedicated replay carrier was added because AI SDK already models these as stable metadata
  - [x] Anthropic `compaction` stream blocks now map to stable `text-*` parts
    - `content_block_start(type=compaction)` now emits `text-start` with
      `providerMetadata.anthropic.type = "compaction"`
    - `compaction_delta` now emits stable `text-delta`, aggregates into final stream text, and
      same-protocol Anthropic SSE replay preserves `compaction` / `compaction_delta`
  - [x] Anthropic deferred/programmatic `tool_use` handling now follows the audited runtime path
    - `message_start.message.content[*].tool_use` now emits stable
      `tool-input-start -> tool-input-delta -> tool-input-end -> tool-call`
    - Anthropic `caller` metadata now rides
      `tool-call.providerMetadata.anthropic.caller.{type,toolId}` on both preloaded
      `message_start` and normal `content_block_start/stop` tool-use paths
    - same-protocol Anthropic SSE replay now also preserves that `caller` metadata instead of
      degrading those tool calls to caller-less `tool_use` blocks
  - [x] Anthropic stream finish metadata now tracks `message_delta.container` like the audited SDK
    - non-terminal `message_delta.container` updates now survive through `message_stop` into both
      stable `finish.providerMetadata.anthropic.container` and final `StreamEnd`
    - a later `message_delta` without `container` now clears earlier message-start container state,
      matching the audited latest-delta-wins behavior
  - [x] Anthropic stream finish metadata now also tracks `message_delta.stop_sequence` like the
    audited SDK
    - non-terminal `message_delta.stop_sequence` updates now survive through `message_stop` into
      both stable `finish.providerMetadata.anthropic.stopSequence` and final `StreamEnd`
    - a later `message_delta` without `stop_sequence` now clears earlier message-start or
      intermediate stop-sequence state, matching the audited latest-delta-wins behavior
  - [x] Anthropic typed response metadata now exposes the remaining audited message metadata fields
    on the main path
    - `AnthropicMetadata.stop_sequence` now matches AI SDK `AnthropicMessageMetadata.stopSequence`
    - `AnthropicMetadata.iterations` now matches AI SDK `AnthropicMessageMetadata.iterations`
      across both non-stream and stream finish responses
    - `AnthropicMetadata.context_management` is now a typed `appliedEdits` union matching AI SDK
      `AnthropicMessageMetadata.contextManagement`, including the `compact_20260112` branch on the
      response-side provider metadata path
  - [x] Anthropic non-stream response content/provider-metadata now follows the audited source and
    null-shape semantics more closely
    - non-stream text citations and `web_search_tool_result` blocks now emit stable `source` parts
      instead of only raw citation/provider metadata side channels
    - request-scoped citation documents now flow into the non-stream Anthropic response
      transformer, so PDF/text document citations resolve `title` / `filename` / `mediaType`
      the same way as the audited SDK path
    - Anthropic source ids now use stable `id-*` generation across both non-stream and stream
      citation/web-search source paths instead of provider/protocol-derived ids
    - non-stream responses and final stream-end provider metadata now keep
      `container: null` / `contextManagement: null` when absent, matching the audited
      `AnthropicMessageMetadata` null-key behavior
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
- [x] Preserve the full provider-native Anthropic `usage` object on both
  `Usage.raw` and `provider_metadata.anthropic.usage`, while still deriving the stable
  AI SDK-style token breakdown from known fields.
- [x] Make AI SDK-shaped usage canonical at the stable layer:
  - legacy `prompt/completion/total` counts are no longer public storage fields on `Usage`
  - compatibility callers use accessors/serde (`prompt_tokens()`, `completion_tokens()`,
    `total_tokens()`) or explicit constructors/builders
  - stable `Usage.merge()` now follows AI SDK `addLanguageModelUsage()` and drops `raw` on
    aggregation instead of recursively merging provider-native payloads
  - extras orchestrator `StepResult::merge_usage()` / `AgentResult::total_usage()` now also follow
    AI SDK `totalUsage` semantics and drop per-step `raw` usage even in the single-step case
- [x] Align extras finish/result surfaces with AI SDK high-level completion semantics.
  - `OrchestratorFinishEvent` now carries the final response, final step, full `steps`, and
    aggregated `total_usage` for `on_finish`
  - `StepResult` now also carries:
    - stable `call_id`
    - `step_number`
    - stable `model { provider, model_id }`
    - unified `content`
    - step-scoped `request` / `response`
    - telemetry `function_id` / `metadata`
    - stable `raw_finish_reason`
  - `ChatResponse.raw_finish_reason` is now the canonical stable carrier, and audited
    OpenAI-compatible / OpenAI Responses / Bedrock / Cohere plus audited completion-stream paths
    now propagate provider-native raw finish reasons into extras step results.
  - extras orchestrator/agent/workflow now bind `LanguageModel` instead of bare
    `ChatCapability`, so the step result can expose stable model identity without heuristic
    inference
  - `OrchestratorBuilder` / `Orchestrator` / `WorkflowBuilder` finish callbacks now apply to both
    non-stream and stream paths
  - `StreamOrchestration` now resolves `total_usage`, and the basic stream path no longer returns
    an empty `steps` list by default
- [x] Add AI SDK-style extras orchestration runtime `context` flow.
  - `OrchestratorContext` is now the stable open JSON-object carrier for extras orchestration
  - `OrchestratorOptions` and `OrchestratorStreamOptions` both accept initial `context`
  - `PrepareStepContext.context` can read the current state and `PrepareStepResult.context` can
    replace it for the current and subsequent steps
  - `ToolResolver::{call_tool_with_context, call_tool_stream_with_context}` now exist as
    backward-compatible extension points, with default implementations delegating to the old
    methods
  - `StepResult.context` and `OrchestratorFinishEvent.context` now expose the step-local and final
    resolved context
  - the stream path now also honors `prepare_step`, `tool_choice`, `active_tools`, and `context`
    on the first streamed step instead of only on non-stream follow-ups
- [x] Bring the extras stream loop and step payload a bit closer to AI SDK high-level shape.
  - `OrchestratorStreamOptions.stop_conditions` now exists and the stream loop evaluates those
    conditions after each step, matching the non-stream orchestrator behavior more closely
  - `OrchestratorBuilder` and `ToolLoopAgent::stream` now propagate cloned stop conditions into
    the spawned stream loop instead of silently dropping them
  - `StepResult` now stores unified `content` composed from assistant response parts plus tool
    results, instead of forcing callers to reconstruct step content from `response + messages`
  - convenience projections now exist for `content()`, `reasoning_parts()`, `reasoning_text()`,
    `files()`, and `sources()`
  - `StepResult.text()` now concatenates all top-level text parts instead of returning only the
    first one
  - standardized extras projections now expose `tool_call_views()` / `tool_result_views()` plus
    `static_tool_calls()` / `dynamic_tool_calls()` and
    `static_tool_results()` / `dynamic_tool_results()` with resolved tool inputs for results
  - `PrepareStepContext.model` and `PrepareStepResult::with_model(...)` now enable per-step
    `LanguageModel` overrides on both non-stream and stream orchestration paths

## Track F - Validation

- [x] Add regression coverage for terminal response-envelope preservation on streaming paths.
- [x] Add regression coverage for Anthropic extended usage and full raw usage preservation.
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
- [x] Add fixture-backed request tests for message-level and part-level `providerOptions`.
  - Anthropic Messages fixture coverage now pins message-level
    `providerOptions.anthropic.cacheControl` lowering plus part-level document
    `providerOptions.anthropic.{citations,title,context}` request input, and normalization tests
    now explicitly assert the canonical part-level Anthropic replay shape restored from wire JSON.
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
- [x] Align OpenAI Responses exact request/response fixtures with the stable AI SDK tool model.
  - legacy compat `function_call` cases now preserve provider-native `raw_finish_reason`
  - provider-executed Responses tool calls/results now pin stable `dynamic` plus tool-result
    `input`
  - hosted dynamic tools (`local_shell` / `shell` / `apply_patch`) now roundtrip back to native
    Responses item types instead of degrading to generic function calls on the bridge path
- [x] Re-audit the remaining OpenAI Responses fixture and stream binary surface after the shared
  metadata/upload refactors.
  - stale MCP approval request fixtures now explicitly mark provider-executed
    `tool-approval-response` parts with `providerExecuted: true`, matching
    `repo-ref/ai/packages/openai/src/responses/convert-to-openai-responses-input.ts`
  - shared `FinishReason::Other(...)` now serializes as the AI SDK string-union shape instead of
    Rust's legacy externally tagged object while still accepting the old object form on read, so
    failed Responses streams again expose `finishReason.unified = "other"` on the stable surface
  - OpenAI-family audio shaping now also keeps the shared speech boundary honest: `TtsRequest`
    `language` lowers into the JSON body only when provider defaults explicitly opt into it, and
    STT multipart regression coverage now asserts the real serialized form body instead of a
    brittle debug string
  - built-in compat `groq` now defaults `supportsStructuredOutputs = true`, matching the audited
    AI SDK provider policy instead of downgrading JSON Schema requests to `json_object`
  - OpenAI Responses tool-role messages that contain only intentionally skipped parts (for example
    non-provider-executed approval responses) are now omitted instead of failing request
    conversion with `Tool message missing tool result`
  - OpenAI typed control options now also mirror the audited `@ai-sdk/openai` surface more
    closely: `OpenAILanguageModelChatOptions` exposes `systemMessageMode`, while
    `OpenAILanguageModelResponsesOptions` exposes `systemMessageMode`, `forceReasoning`, and
    `contextManagement`; public OpenAI/Azure re-exports and import-compile coverage lock those
    types, and `/responses` request shaping lowers
    `contextManagement[].compactThreshold` to wire `context_management[].compact_threshold`
  - the full top-level `openai_responses_*` nextest sweep and
    `siumai-protocol-openai --lib --all-features` audit are green again after those follow-up
    fixes
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
- [x] Narrow the remaining OpenAI-compatible parity audit to explicit fixture/public-path holes.
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
  - compat chat now also follows the audited raw/camelCase provider-key contract more closely:
    provider-owned passthrough options merge raw + camelCase keys with camelCase taking
    precedence, non-stream + stream-finish provider metadata keep the resolved request-side
    namespace key with an explicit provider root, and `extra_content.google.thought_signature`
    now survives on finalized compat tool calls as `providerMetadata.{provider}.thoughtSignature`
  - non-stream compat response fixtures now also pin the direct runtime-provider-root path for
    tool-call `extra_content.google.thought_signature -> providerMetadata.test-provider`
  - no-network OpenRouter public-path coverage now also pins that same finalized tool-call
    thought-signature metadata across `Siumai` / provider / config / registry entrypoints
  - requested camelCase metadata-key variants remain locked by the lower-level compat
    transformer/streaming tests, so the public audit no longer has a missing fixture/path hole
  - lower-level compat streaming regressions now also lock same-chunk `reasoning -> text`
    ordering plus explicit `finish_reason = "tool_calls"` finalization of pending/empty tool
    calls without duplicate replays on trailing empty chunks
- [x] Add a dedicated Rust text-completion family for AI SDK `completionModel()`.
  - `siumai-spec` now exposes stable `CompletionRequest` / `CompletionResponse`
  - `siumai-core` now exposes `CompletionCapability` plus `CompletionModel{V3}`
  - `siumai-registry` now exposes `completion_model(...)` and `CompletionModelHandle`
  - `siumai-provider-openai-compatible` now executes real `/completions` generate/stream paths
    with AI SDK-style prompt materialization, warnings, and provider-option normalization
  - native `siumai-provider-openai` and `siumai-provider-azure` now also execute real
    `/completions` generate/stream paths on their direct provider routes
  - native OpenAI/Azure registry factories and provider metadata now advertise completion-family
    support on the direct provider path
  - compat/native completion metadata now preserves raw `choices[0].logprobs` instead of reusing
    chat-only logprob extraction
  - completion streaming intentionally reuses the shared `ChatStream` runtime lane instead of
    introducing a second event family
- [x] Add a stable high-level file-upload helper aligned with AI SDK `uploadFile`.
  - `siumai::files::upload(...)` now exists with public `UploadFileOptions`,
    `UploadFileResult`, and `UploadFileProviderMetadata`
  - the helper now auto-detects request media type from bytes, falls back to `text/plain` or
    `application/octet-stream`, rejects URL inputs like AI SDK, and accepts shared `DataContent`
    directly instead of an upload-only wrapper
  - shared `FileUploadRequest` / `FileObject` filenames are now optional, missing filenames are no
    longer normalized to `blob`, and helper `filename` / `mediaType` are no longer backfilled
    when the provider response omitted them
  - stable result shaping now returns canonical `providerReference` plus provider-owned metadata
    extras instead of helper-injected generic file bookkeeping
  - built-in adapters now cover the current file-capable public surfaces:
    unified `Siumai`, registry `LanguageModelHandle`, OpenAI/Azure/Gemini/MiniMaxi
    file-management clients/resources, and Anthropic beta files clients
  - public regression coverage now locks media-type detection, URL rejection, omitted filename
    behavior, provider-owned metadata passthrough, and explicit MiniMaxi purpose requirements
- [x] Audit the remaining high-level AI SDK helper gaps after `uploadFile`.
  - `uploadSkill` now exists as `siumai::skills::upload(...)`, with public
    `UploadSkillFile` / `UploadSkillOptions` / `UploadSkillResult`, shared `SkillsCapability`,
    and provider-owned OpenAI / Anthropic `skills()` resources
  - current audited skill-upload coverage now matches the AI SDK provider scope (`openai` +
    `anthropic`); OpenAI mirrors the AI SDK `displayTitle -> unsupported` warning behavior,
    Anthropic follows the extra version-metadata fetch path, unified/registry callers now bridge
    through the same capability lane, and the provider-owned resources now also reuse the shared
    `SkillUploadRequest` / `SkillUploadResult` contract instead of keeping provider-local wrapper
    types
  - `uploadFile` now also matches the AI SDK file-helper call surface more closely:
    shared `FileUploadRequest` and high-level `UploadFileOptions` carry canonical
    `providerOptions`, OpenAI/Azure honor provider-scoped `purpose` / `expiresAfter`, and Gemini
    now honors `displayName` plus poll interval/timeout provider options on the upload path
  - Anthropic provider-owned `files()` now also converges on that same shared file-management
    contract: `AnthropicFiles` / `AnthropicClient` implement `FileManagementCapability`,
    upload/list/retrieve/delete reuse shared file-management structs directly, and the old
    Anthropic-only helper bridge plus provider-local file wrapper layer are removed
  - shared upload/chat/completion/stream/content metadata now also converges on one
    provider-rooted `ProviderMetadataMap`, matching
    `repo-ref/ai/packages/ai/src/types/provider-metadata.ts`
  - Rust now has a first-class AI SDK-style `UIMessage` structural layer:
    `siumai::types::{UiMessage, UiMessagePart, UiToolPart, UiToolInvocation, ...}` plus
    `siumai::ui::{validate_ui_messages, validate_ui_messages_with_schemas,
    convert_to_model_messages, convert_to_chat_request}`
  - `UIMessage.providerMetadata` intentionally stays on the request-side `providerOptions`
    contract because upstream `convertToModelMessages()` forwards it there
  - the remaining frontend-side gap is narrower now: Rust still intentionally has no
    AI SDK-style `useChat` stateful hook layer

## Current branch notes

This branch now has a clearer baseline than the first draft of the workstream:

- shared compatibility warning support
- widened `source` support
- strict `source` URL/document union
- widened `tool-approval-*` support
- Anthropic/OpenAI-compatible stream-end fidelity fixes
- Shared JSON transport EOF synthesis now preserves stateful terminal response content, which
  closes the clean-EOF Bedrock structured-output regression exposed during this workstream
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
- a dedicated stable completion family for AI SDK `completionModel()`
- a dedicated stable high-level file-upload helper for AI SDK `uploadFile`
- a shared `ProviderMetadataMap` root that now backs chat/completion/content/stream/file/skill
  response metadata with AI SDK-style `provider_id -> object` semantics
- first-class unified provider stories for `deepinfra`, `vertex-maas`, and native `/v2` `cohere`
- first-class provider typing for `Cohere`, `TogetherAi`, and `Bedrock` instead of silently
  degrading those built-in providers to `Custom(...)`
- the Gemini protocol adapters now also compile against the newer `FilePartSource` split, so the
  multi-feature verification lane covering `openai + azure + anthropic + google + minimaxi` is
  green again
- the wider all-features provider-helper sweep is now also green:
  provider-owned metadata helpers across Gemini/Vertex, Azure completion, Bedrock, Ollama,
  Anthropic skills, DeepSeek, Groq, xAI, and MiniMaxi all use the shared `ProviderMetadataMap`
  root object shape, Ollama/Cohere follow `FilePartSource`, and Anthropic JSON response shaping
  now covers `MessageContent::Json`
- the latest provider-boundary/helper follow-up is now landed too:
  Anthropic and DeepSeek custom provider roots follow the audited AI SDK request/response rule,
  Gemini `google|vertex` request precedence matches the upstream namespace fix, OpenAI typed
  helpers expose keyed accessors, and the built-in Perplexity compat preset now exposes the AI
  SDK-style typed `images/usage/cost` metadata shape instead of raw snake_case fragments
