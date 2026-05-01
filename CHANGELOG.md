# Changelog

This file lists noteworthy changes. Sections are grouped by version to make upgrades clearer.

## [Unreleased]

### Added

- OpenAI-compatible package-surface parity now also exposes compat-backed AI SDK-style
  `DeepSeekProviderSettings`, `GroqProviderSettings`, `TogetherAIProviderSettings`, and
  `XaiProviderSettings`, plus provider-scoped client/config/model-id aliases.
- OpenAI-compatible model catalogs now include AI SDK-aligned `groq`, `xai`, and `togetherai`
  namespaces instead of the older partial Groq/xAI lists.
- OpenAI-compatible Alibaba/Qwen parity now supports the AI SDK `alibaba` preset, `qwen` preset
  aliases, typed `AlibabaChatOptions` / `QwenChatOptions`, and request normalization for
  `enableThinking`, `thinkingBudget`, and `parallelToolCalls`.
- AI SDK `text-stream` HTTP response helper parity now has a real Axum server boundary:
  `siumai_extras::server::axum::{to_text_stream_response,
  to_text_stream_response_with_options, TextStreamResponseOptions}` wraps `ChatStream` text output
  as `text/plain; charset=utf-8`, matching the upstream `createTextStreamResponse` behavior on the
  Rust server-adapter path without pretending Node `ServerResponse` exists in core.
- AI SDK util download parity now has real Rust helpers:
  `siumai::{create_download, download_url, validate_download_url,
  read_response_with_size_limit, DEFAULT_MAX_DOWNLOAD_SIZE}` and the same unified-prelude exports
  mirror the upstream `createDownload` / `validateDownloadUrl` / size-limit behavior for
  `http`, `https`, and inline `data:` URLs while returning the existing passive `DownloadError`
  carrier.
- AI SDK util `SerialJobExecutor` parity is now available as `siumai::SerialJobExecutor` and in
  `prelude::unified`, providing a cloneable async executor that serializes concurrently submitted
  jobs while preserving each job's return value or error.
- AI SDK provider-utils HTTP/string helper parity now exposes real Rust utilities from the root
  facade and unified prelude: `normalize_headers`, `normalize_optional_headers`,
  `normalize_header_map`, `combine_headers`, `with_user_agent_suffix`,
  `extract_response_headers`, `media_type_to_extension`, `strip_file_extension`, and
  `without_trailing_slash`.
- AI SDK provider-utils runtime/error helper parity now exposes `VERSION`,
  `get_runtime_environment_user_agent`, `get_error_message`, `delay`, and `is_abort_error`. The
  runtime user-agent helper reports Rust truthfully as `runtime/rust` rather than emulating
  JavaScript host globals, and delay/abort handling uses Tokio plus the existing `CancelHandle`
  instead of modeling browser `AbortSignal` directly.
- AI SDK provider-utils optional-value and setting-loader parity now exposes
  `Arrayable`, `as_array`, `is_non_nullable`, `filter_nullable`, `remove_undefined_entries`,
  `load_api_key`, `load_setting`, `load_optional_setting`, and their option structs. These map JS
  nullish values to Rust `Option` / `Arrayable` and use Rust's typed string environment semantics
  instead of modeling JavaScript-only non-string branches.
- AI SDK provider-utils `uint8-utils` parity now exposes
  `convert_base64_to_uint8_array`, `convert_uint8_array_to_base64`, and `convert_to_base64`,
  including URL-safe base64 input normalization.
- AI SDK provider-utils image-file conversion parity now exposes
  `convert_image_model_file_to_data_uri` over the existing `ImageEditInput` carrier, returning
  URL-backed inputs as-is and converting base64/binary image file inputs into data URIs when a
  media type is present.
- AI SDK provider-utils streaming tool-call tracker parity now exposes
  `StreamingToolCallTracker`, `StreamingToolCallDelta`,
  `StreamingToolCallFunctionDelta`, `StreamingToolCallTrackerOptions`, and
  `StreamingToolCallTypeValidation`. The helper accumulates OpenAI-compatible streaming
  tool-call argument deltas and emits typed V4 `tool-input-*` plus final `tool-call` stream parts.
- AI SDK V4 stream finish-reason parity now keeps typed low-level stream overlays on the upstream
  `finishReason.unified` union values (`stop`, `length`, `content-filter`, `tool-calls`, `error`,
  `other`) while preserving Siumai-specific `StopSequence` / provider-specific reasons in
  `finishReason.raw` and still accepting legacy underscore values on input.
- AI SDK V4 language-model usage parity now uses provider-facing `u64` token carriers for
  `LanguageModelV4Usage`, matching upstream `number | undefined` semantics without leaking the
  stable Rust `Usage` layer's `u32` compatibility limit into provider V4 overlays.
- AI SDK V4 stream usage payloads now also omit unknown token subfields instead of serializing
  `null`, matching upstream `undefined` token-count behavior on streamed `finish.usage` parts.
- AI SDK V4 call-options parity now also keeps `LanguageModelV4CallOptions.max_output_tokens` as a
  provider-facing `u64`, while the stable high-level settings structs retain their existing `u32`
  compatibility surface.
- AI SDK V4 generated file/reasoning-file content now uses a dedicated generated-file data carrier
  for `string | Uint8Array` output payloads instead of reusing prompt data content that can also
  represent URLs.
- AI SDK V4 provider `tool-result.result` payloads now reject `null` on the passive result and
  stream overlays, matching upstream `NonNullable<JSONValue>` semantics.
- AI SDK V4 model-facing tools now keep `provider.args` and function-tool `inputExamples[].input`
  as JSON objects on the provider overlay, matching upstream `Record<string, unknown>` /
  `JSONObject` boundaries while leaving the wider stable tool input layer compatible.
- AI SDK V4 prompt tool-result outputs now use a dedicated provider-facing overlay for canonical
  `content` parts, mapping stable image/file reference variants to upstream `file-*` shapes and
  falling back to valid JSON output for legacy content that V4 cannot express directly.
- AI SDK `CallWarning` now uses the strict shared V4 warning union instead of aliasing the wider
  stable `Warning` compatibility enum; legacy stable warning variants are normalized to
  `unsupported { feature, details }` during AI SDK result projection.
- AI SDK V4 custom prompt/output overlays now enforce the upstream `{provider}.{provider-type}`
  custom kind shape while leaving stable `CustomPart` / `CustomOutput` compatible with older
  arbitrary string values.
- AI SDK V4 prompt `providerOptions` now use a provider-facing object-only overlay matching
  upstream `SharedV4ProviderOptions = Record<string, JSONObject>`; stable prompt/tool-result
  projections filter non-object provider option entries instead of emitting invalid V4 shapes.
- AI SDK V4 generated content/result `providerMetadata` now use a provider-facing object-only
  overlay matching upstream `SharedV4ProviderMetadata = Record<string, JSONObject>`; stable
  output/source projections filter non-object metadata entries while the wider stable
  `ProviderMetadataMap` remains backward-compatible.
- AI SDK V4 stream parts now use a dedicated provider-facing stream overlay instead of aliasing
  the historical V3 stream union, so streamed `providerMetadata` follows the same object-only V4
  boundary while V3 compatibility parsing remains permissive.
- AI SDK provider-utils JSON instruction parity now exposes
  `inject_json_instruction`, `inject_json_instruction_into_messages`,
  `JsonInstructionOptions`, and `JsonInstructionMessageOptions`, matching the upstream prompt and
  first-system-message injection defaults for schema and generic JSON responses.
- AI SDK provider-utils JSON parse parity now exposes `parse_json`, `safe_parse_json`,
  `is_parsable_json`, `parse_json_with_schema`, and `safe_parse_json_with_schema`, including
  upstream-style forbidden prototype-property rejection and explicit safe result carriers.
- AI SDK provider-utils `parseProviderOptions` parity now exposes `parse_provider_options` for
  validating a provider-scoped `ProviderOptionsMap` entry with an existing Rust `Schema`.
- AI SDK provider-utils provider-reference parity now exposes `resolve_provider_reference` and
  `is_provider_reference`, backed by the existing `ProviderReference` / `FilePartSource` carriers.
- AI SDK provider-utils reasoning mapper parity now exposes `ReasoningLevel`,
  `ReasoningBudgetOptions`, `DEFAULT_REASONING_BUDGET_PERCENTAGES`,
  `is_custom_reasoning`, `map_reasoning_to_provider_effort`, and
  `map_reasoning_to_provider_budget`, using the existing shared `Warning` carrier for
  unsupported/compatibility notices.
- AI SDK provider-utils `validateTypes` parity now exposes `validate_types`,
  `safe_validate_types`, and `TypeValidationResult` over the existing `Schema` and
  `TypeValidationError` carriers. Schemas without a runtime validator fail explicitly instead of
  pretending Rust can perform TypeScript-style unchecked generic casts.
- AI SDK provider-utils schema parity now has an honest Rust surface:
  `siumai::types` and `prelude::unified` expose `Schema`, `ValidationResult`, `FlexibleSchema`,
  `LazySchema`, `json_schema`, `json_schema_with_validator`, `lazy_schema`, `as_schema`,
  `as_schema_or_empty`, and `empty_json_schema`. Zod and TypeScript Standard Schema adapters remain
  explicitly deferred until backed by real Rust validation/conversion behavior.
- AI SDK provider-utils ID helper parity is now exposed from the Rust facade:
  `siumai::{IdGenerator, IdGeneratorOptions, create_id_generator, generate_id}` and the same names
  in `prelude::unified` mirror the upstream non-cryptographic `createIdGenerator` / `generateId`
  contract with Rust `Result`-based option validation.
- AI SDK provider-utils URL support parity now exposes `SupportedUrlMap`, `UrlSupportRegex`, and
  `is_url_supported`, matching upstream media-type wildcard/prefix handling plus URL regex checks
  over a typed Rust table.
- AI SDK provider-utils tool helper parity is now easier to import directly:
  `siumai::{tool, dynamic_tool, ToolExecutionOptions, ToolExecuteFunction, ToolSet}` and the same
  names in `prelude::unified` expose the existing runtime tool binding surface without merging
  Rust closures into the passive provider-facing `Tool` schema.
- AI SDK provider-utils tool-name mapping parity is now importable from the stable facade:
  `siumai::{create_tool_name_mapping, ToolNameMapping}` maps provider-defined custom tool names
  to provider-native tool names and back using the existing portable `Tool` carrier.
- AI SDK model-facing tool-choice parity now exposes `LanguageModelV4ToolChoice` and
  `prepare_tool_choice(...)`, preserving the upstream split where high-level `ToolChoice` accepts
  string-like inputs while provider-facing V4 calls receive `{ type: ... }` objects.
- AI SDK model-facing tool-shape parity now exposes `LanguageModelV4FunctionTool` and
  `LanguageModelV4ProviderTool` projections, keeping user-facing `Tool` metadata such as
  `outputSchema`, `title`, `isProviderExecuted`, and deferred-result hints off the narrower
  provider-call tool objects.
- V4 function-tool `inputExamples` now use the explicit model-facing
  `LanguageModelV4FunctionToolInputExample { input }` shape, while stable `ToolFunction` examples
  still accept older raw-object values and project them into the upstream `{ input: ... }` form.
- AI SDK model-facing prompt parity now exposes `LanguageModelV4Prompt`,
  `LanguageModelV4Message`, V4 prompt part overlays, and
  `prepare_language_model_v4_prompt(...)`, converting stable `ModelMessage` values to the
  provider prompt shape where images become `file` parts, provider file references become direct
  `data` maps, assistant approval requests are filtered, and provider-executed approval responses
  are preserved.
- V4 prompt `tool-approval-response` parts now preserve request-side `providerOptions`, matching
  the upstream provider prompt shape instead of rejecting those options during stable
  `ContentPart` / `ModelMessage` conversion.
- AI SDK model-facing call-options parity now exposes `LanguageModelV4CallOptions` and
  `LanguageModelV4Tool`, keeping prompt, V4-projected tools, tool choice, headers, abort handle,
  reasoning, and provider options together in the same provider-call overlay without replacing the
  existing reusable `LanguageModelCallOptions` / `RequestOptions` split.
- AI SDK model-facing generate-result parity now exposes `LanguageModelV4Content`,
  `LanguageModelV4GenerateResult`, `LanguageModelV4StreamResult`, V4 file/reasoning-file/tool
  content parts, V4 finish/usage/response metadata, and `LanguageModelV4DataContent`, preserving
  the narrower upstream provider result shape separately from high-level `GenerateTextContentPart`.
- AI SDK provider-facing language-model interface parity now exposes `LanguageModelV4`,
  `LanguageModelV4Stream`, and `LanguageModelV4DoStreamResult` from the Rust text-family surface.
  The trait mirrors the upstream `supportedUrls`, `doGenerate`, and `doStream` provider contract
  over V4 call/result overlays without replacing the stable high-level `LanguageModel` runtime.
- AI SDK provider-utils provider-tool factory parity now has a real Rust carrier/facade:
  provider tools serialize as `type: "provider"` and preserve `isProviderExecuted`,
  `inputSchema`, `outputSchema`, `args`, and `supportsDeferredResults`, while
  `create_provider_defined_tool_factory`,
  `create_provider_defined_tool_factory_with_output_schema`, and
  `create_provider_executed_tool_factory` are available from the root facade and unified prelude.
- AI SDK provider-utils `ToolCall` / `ToolResult` passive data structures now preserve the
  current output-side metadata fields: `providerMetadata`, `title`, invalid-tool `error`,
  `invalid`, and preliminary tool results. Their serialized output now also preserves the
  upstream `type: "tool-call"` / `type: "tool-result"` discriminators. The upstream
  `Static*` / `Dynamic*` / `Typed*` tool call, result, and error exports are now available as
  Rust aliases over the same carriers, with the `dynamic` flag kept as data.
- AI SDK `generateText` tool approval output parts are now represented directly by
  `ToolApprovalRequestOutput` / `ToolApprovalResponseOutput`, including the nested full `toolCall`
  payload and `isAutomatic` / `providerExecuted` flags.
- AI SDK text-output basic content, file, and reasoning parts now have direct Rust data structures:
  `TextOutput`, `CustomOutput`, `FileOutput`, `GeneratedFile`, `ReasoningOutput`, and
  `ReasoningFileOutput`, preserving the upstream nested generated-file `base64` / `mediaType`
  shape plus provider metadata. `DefaultGeneratedFile` and the backwards-compatible
  `Experimental_GeneratedImage` export are aliases over `GeneratedFile`, and
  `DefaultGeneratedFileWithType` now covers the upstream `type: "file"` generated-file class
  projection. `GenerateTextContentPart` now also provides the passive output-side content union
  without reusing prompt/runtime `ContentPart`.
- AI SDK `generateText` result envelopes now have passive Rust data structures:
  `ResponseMessage`, `GenerateTextResponseMetadata`, `GenerateTextModelInfo`,
  `GenerateTextReasoningPart`, `GenerateTextStepReasoningPart`, `GenerateTextStepResult`,
  `StepResult`, `DefaultStepResult`, and `GenerateTextResult`. Step reasoning intentionally uses
  the provider-utils `data` / `mediaType` / `providerOptions` shape, while final result reasoning
  keeps the output-side `file` / `providerMetadata` shape.
- AI SDK-style `generateText` now has a real Rust helper projection via
  `siumai::generate_text(...)` / `siumai::text::generate_text(...)`. It calls the existing
  text-family model once, then projects `ChatResponse` into a single-step `GenerateTextResult`
  with content, reasoning, source, tool-call, tool-result, usage, finish reason, response metadata,
  and provider metadata fields. Full AI SDK agent/tool-loop execution remains intentionally
  separate from this single-step projection.
- AI SDK `streamText` output events now have a passive `TextStreamPart` union and named
  `TextStream*Part` structures matching `generate-text/stream-text-result.ts`, including the
  higher-level `text` reasoning/text deltas, `start-step` / `finish-step`, `finish.totalUsage`,
  `abort`, `raw.rawValue`, and aliases over the existing tool output parts.
- AI SDK single model-call streaming events now have a passive `LanguageModelStreamPart` union
  matching `generate-text/stream-language-model-call.ts`, including `model-call-start`,
  `model-call-response-metadata`, and `model-call-end`, plus the upstream experimental type alias
  names without claiming the runtime `experimental_streamLanguageModelCall` helper is implemented.
- AI SDK `generateText` / `streamText` callback event payloads now have passive Rust structures:
  `CallbackModelInfo`, `GenerateTextStartEvent`, `GenerateTextStepStartEvent`,
  `GenerateTextStepEndEvent`, `GenerateTextEndEvent`, `StreamTextChunkEvent`,
  `StreamTextLifecycleChunk`, `ToolOutput`, `ToolExecutionStartEvent`, and
  `ToolExecutionEndEvent`, plus deprecated upstream alias names such as `OnStartEvent` and
  `OnToolCallFinishEvent`. These mirror `core-events.ts` / `tool-execution-events.ts` as data
  carriers without claiming runtime callback wiring.
- AI SDK telemetry options now have an importable passive shape:
  `TelemetryOptions` preserves the upstream `isEnabled`, `recordInputs`, `recordOutputs`, and
  `functionId` fields while leaving callback-style telemetry integrations to the real runtime
  dispatcher layer.
- AI SDK `error/index.ts` parity now has passive Rust carriers for the high-value serializable
  error data that was still missing from the import surface: `InvalidArgumentError`,
  `InvalidStreamPartError`, `InvalidToolApprovalError`, `ToolCallNotFoundForApprovalError`,
  `NoImageGeneratedError`, `NoObjectGeneratedError`, `NoOutputGeneratedError`,
  `NoSpeechGeneratedError`, `NoTranscriptGeneratedError`, `NoVideoGeneratedError`,
  `UnsupportedModelVersionError`, `UIMessageStreamError`, `InvalidMessageRoleError`,
  `MessageConversionError`, `RetryError`, and `RetryErrorReason`. The same pass now also covers
  the provider-level errors re-exported from `@ai-sdk/provider` such as `AISDKError`,
  `APICallError`, `EmptyResponseBodyError`, `InvalidPromptError`, `InvalidResponseDataError`,
  `JSONParseError`, `LoadAPIKeyError`, `LoadSettingError`, `NoContentGeneratedError`,
  `NoSuchModelError`, `NoSuchProviderReferenceError`, `TooManyEmbeddingValuesForCallError`,
  `TypeValidationContext`, `TypeValidationError`, and `UnsupportedFunctionalityError`. These are
  data-shape exports and do not replace Siumai's runtime error hierarchy.
- AI SDK provider-utils `DownloadError` is now available as a passive error carrier with `url`,
  `statusCode`, `statusText`, `cause`, and upstream-style default messages for failed downloads.
- AI SDK registry `NoSuchProviderError` is now represented as a passive error carrier with
  `modelId`, `modelType`, `providerId`, and `availableProviders`, matching the public registry
  package's data shape without replacing Siumai's native registry handles.
- AI SDK generate-text step-control payloads now have passive Rust structures and helpers:
  symbolic `StopCondition` plus `is_step_count`, `is_loop_finished`, `has_tool_call`, and
  `is_stop_condition_met`; `filter_active_tools` plus the upstream experimental/deprecated helper
  aliases `experimental_filter_active_tools` and `step_count_is`; `PrepareStepOptions` /
  `PrepareStepResult`; `ToolApprovalStatus` / `ToolApprovalConfiguration` /
  `ToolApprovalDecisionContext`; and `ToolCallRepairContext` /
  `ToolCallRepairFunctionError` / `NoSuchToolError` / `InvalidToolInputError` /
  `ToolCallRepairError` / `ToolCallRepairResult`. The repair-function input error union now stays
  separate from AI SDK's repair-failure wrapper error, matching
  `tool-call-repair-function.ts` and `tool-call-repair-error.ts`. Function-valued upstream
  callbacks remain represented as data carriers or Rust helper functions only.
- AI SDK `pruneMessages` parity is now available as `prune_messages(...)` with
  `PruneMessagesOptions`, `PruneReasoningMode`, `PruneToolCallRule`, `PruneToolCallMode`, and
  `PruneEmptyMessagesMode`, covering reasoning pruning, tool-call/result/approval pruning, and
  empty-message removal over the shared `ModelMessage` carrier.
- AI SDK embed/rerank/image/video/audio helper result envelopes now have passive Rust data
  structures:
  `EmbedResult`, `EmbedManyResult`, embed callback/model-call event payloads,
  `ModelCallResponseData`, `RerankResult`, `RerankRanking`, rerank callback/model-call event
  payloads, `GenerateImageResult`, `Experimental_GenerateImageResult`, `GeneratedAudioFile`,
  `DefaultGeneratedAudioFile`, `DefaultGeneratedAudioFileWithType`, `GenerateVideoResult`,
  `SpeechResult`, `Experimental_SpeechResult`, `TranscriptionResult`,
  `Experimental_TranscriptionResult`, and `TranscriptionSegment`. These mirror the `embed`,
  `embedMany`, `rerank`, `generateImage`, `generateVideo`, `generateSpeech`, and `transcribe`
  result object shapes without changing the existing provider-owned runtime helpers. The previous
  provider-level single rerank item is now named `RerankRankingEntry`, freeing `RerankResult` for
  the AI SDK-style result envelope.
- AI SDK `GenerateImagePrompt` parity is now represented as an importable untagged Rust prompt
  union over text-only prompts or `{ images, text?, mask? }` image prompts. It is exported from
  `siumai::image` and `prelude::unified`, serializes to the upstream shape, and can be converted
  into `GenerateImageRequest` without adding a fake image-generation runtime.
- Deprecated AI SDK experimental helper spellings now have honest Rust-style aliases over the
  existing real helper paths: `experimental_generate_image`, `experimental_generate_speech`,
  `experimental_generate_video`, and `experimental_transcribe`.
- AI SDK text-output tool failure parts now have direct Rust data structures:
  `ToolError`, `ToolOutputDenied`, `StaticToolOutputDenied`, and `TypedToolOutputDenied`.
- AI SDK provider-utils stream parsing parity now has a public Rust wrapper:
  `siumai::parse_json_event_stream` and `prelude::unified::parse_json_event_stream` parse SSE
  `data:` JSON payloads while ignoring `[DONE]`, using Rust stream item errors instead of a
  TypeScript-style `ParseResult` union.
- AI SDK utility parity now exposes pure Rust helpers for stable data operations:
  `cosine_similarity`, `get_text_from_data_url`, and `is_deep_equal_data` are available from the
  root facade and `prelude::unified`, mirroring `cosineSimilarity`, `getTextFromDataUrl`, and
  `isDeepEqualData` with Rust `Result` errors for invalid inputs.
- AI SDK `generateObject` parity now has a non-streaming Rust helper:
  `siumai::structured_output::generate_object` plus root/prelude re-exports for
  `generate_object`, `generate_array`, `generate_enum`, `GenerateObjectOptions`,
  `GenerateObjectSchema`, and `GenerateObjectResult` set JSON Schema response format on the
  language-model call, parse the returned JSON, run typed Rust schema validators when provided, and
  project finish reason, usage, warnings, request/response metadata, and provider metadata onto the
  AI SDK-style result shape. Array and enum helpers use the same wrapped output strategies as
  upstream, `generate_choice` mirrors AI SDK `generateText` `output.choice(...)`, `generate_json`
  mirrors `output.json(...)` with first-class schema-less `ResponseFormat::json_object()`, and
  `GenerateObjectOptions::with_repair_text_fn(...)` mirrors the non-streaming repair callback path
  after parse or validation failure. The passive AI SDK import surface now also exposes the
  structured-output callback/event payloads (`GenerateObjectStartEvent`,
  `GenerateObjectStepStartEvent`, `GenerateObjectStepEndEvent`, `GenerateObjectEndEvent`,
  `GenerateObjectResponseMetadata`, `GenerateObjectOutputStrategy`) plus `ObjectStreamPart` and
  its named variants for `streamObject` event data. The full `StreamObjectResult` runtime helper
  remains explicitly deferred.
- Structured object generation failures now use an AI SDK-style `LlmError::NoObjectGenerated`
  variant that preserves generated text, response metadata, usage, finish reason, and the
  underlying parse/validation cause.
- AI SDK `parsePartialJson` parity is now available as
  `siumai::structured_output::{fix_partial_json, parse_partial_json}` plus
  `PartialJsonParseState` / `PartialJsonParseResult`, providing the scanner-based partial JSON
  repair foundation required for future `streamText` structured-output transforms.
- Structured-output streaming now has a narrow Rust projection:
  `siumai::structured_output::partial_json_value_stream(...)` consumes a `ChatStream` and emits
  changed partial JSON values plus the final parsed JSON value, without claiming the full AI SDK
  multi-lane `StreamTextResult` contract yet.
- Native OpenAI / Azure / Bedrock package-surface parity is now tighter on the Rust facade:
  `provider_ext::{openai,azure,bedrock}` now expose package-level
  `OpenAIProviderSettings`, `AzureOpenAIProviderSettings`, and
  `AmazonBedrockProviderSettings` carriers plus `VERSION`; the provider-owned builder/config
  surfaces now also expose the minimal honest helper set required to support those carriers
  directly (`headers` / `header` across the three providers, plus Azure `resourceName`), and the
  supported vs deferred upstream field matrix is now tracked under
  `docs/workstreams/provider-settings-surface-alignment/`.
- Native Cohere package-surface parity now follows the same provider-settings rule:
  `provider_ext::cohere` now exposes `CohereProviderSettings` plus `VERSION`, with model-agnostic
  `into_builder()` / `into_builder_for_model(...)` / `into_config_for_model(...)` helpers and
  direct header/fetch/base-url support on the provider-owned builder/config surface. Upstream
  `generateId` remains explicitly deferred until the Cohere runtime owns a comparable stable ID
  hook.
- Native Anthropic package-surface parity now follows the provider-settings rule too:
  `provider_ext::anthropic::{AnthropicProviderSettings, VERSION}` exposes the audited
  `apiKey` / `authToken` / `baseURL` / `headers` / `fetch` subset, with `authToken` mapped to
  `Authorization: Bearer ...` without forcing an empty `x-api-key`. Upstream `generateId` and
  `name` remain explicitly deferred.
- DeepSeek and TogetherAI package surfaces now also expose AI SDK-style provider settings carriers:
  `provider_ext::deepseek::{DeepSeekProviderSettings, VERSION}` and
  `provider_ext::togetherai::{TogetherAIProviderSettings, VERSION}` mirror the supported
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset with model-agnostic conversion
  helpers.
- xAI now joins that provider-settings pass: `provider_ext::xai::{XaiProviderSettings, VERSION}`
  exposes the audited `apiKey` / `baseURL` / `headers` / `fetch` package settings subset.
  Upstream xAI uses `generateId` internally but does not expose it on `XaiProviderSettings`, so it is
  not tracked as a deferred xAI settings field.
- Groq now joins that provider-settings pass: `provider_ext::groq::{GroqProviderSettings, VERSION}`
  exposes the audited `apiKey` / `baseURL` / `headers` / `fetch` package settings subset and
  converts it into the provider-owned builder/config paths.
- Mistral now joins the provider-settings pass through the shared OpenAI-compatible runtime:
  `provider_ext::mistral::{MistralProviderSettings, VERSION}` exposes the audited
  `apiKey` / `baseURL` / `headers` / `fetch` subset, while upstream `generateId` remains
  explicitly deferred until the compat runtime owns a stable ID hook.
- Perplexity now joins the provider-settings pass through the shared OpenAI-compatible runtime:
  `provider_ext::perplexity::{PerplexityProviderSettings, VERSION}` exposes the audited
  `apiKey` / `baseURL` / `headers` / `fetch` subset.
- DeepInfra now joins the provider-settings pass through the shared OpenAI-compatible runtime:
  `provider_ext::deepinfra::{DeepInfraProviderSettings, VERSION}` exposes the audited
  `apiKey` / `baseURL` / `headers` / `fetch` subset.
- MoonshotAI now joins the provider-settings pass through the shared OpenAI-compatible runtime:
  `provider_ext::moonshotai::{MoonshotAIProviderSettings, VERSION}` exposes the audited
  `apiKey` / `baseURL` / `headers` / `fetch` subset.
- Fireworks now joins the provider-settings pass through the shared OpenAI-compatible runtime:
  `provider_ext::fireworks::{FireworksProviderSettings, VERSION}` exposes the audited
  `apiKey` / `baseURL` / `headers` / `fetch` subset.
- Generic `@ai-sdk/openai-compatible` package-surface parity now has a dedicated carrier:
  `provider_ext::openai_compatible::{OpenAICompatibleProviderSettings, VERSION}` exposes the
  audited `name` / `baseURL` / `apiKey` / `headers` / `queryParams` / `fetch` / `includeUsage` /
  `supportsStructuredOutputs` / `transformRequestBody` / `metadataExtractor` subset. Generic
  settings intentionally use a plain compat adapter instead of reusing built-in provider presets,
  and can target unauthenticated local/private gateways when no API key or authorization header is
  configured.
- Google Vertex MaaS now joins the provider-settings pass through the shared OpenAI-compatible
  runtime: `provider_ext::vertex_maas::{GoogleVertexMaasProviderSettings, VERSION}` exposes the
  audited `project` / `location` / `baseURL` / `headers` / `fetch` subset, derives the
  `/endpoints/openapi` base URL from project/location with environment fallbacks, and uses a Rust
  token-provider analogue instead of modeling Node's `googleAuthOptions` object directly.
- `@ai-sdk/google-vertex` package-surface parity is now tighter on the Rust facade:
  `provider_ext::google_vertex` / `providers::vertex` now expose `VERSION` plus a dedicated
  `GoogleVertexProviderSettings` input struct with `into_builder()` /
  `into_builder_for_model(...)`, `GoogleVertexBuilder` now mirrors the upstream non-callable
  `image` / `imageModel` / `video` / `videoModel` family helpers, curated grouped Vertex model ids
  now cover the current audited chat/embedding/image/video package contracts (including
  `text-embedding-005`, `gemini-embedding-2-preview`, `imagen-4.0-ultra-generate-001`, and
  `gemini-2.5-flash-image`), `GoogleVertexClient::supported_models()` now reuses that same curated
  source instead of only returning the configured model id, `generateId` now has an honest Rust
  analogue all the way through settings/builder/config into the Vertex chat/stream Gemini
  transformer runtime (so custom stable tool/source IDs actually work), the provider-option surface
  now also exposes typed Rust enums for the audited Vertex image/video enum domains plus fluent
  `VertexEmbeddingOptions` builders, Vertex Imagen result parsing now keeps the audited
  `revisedPrompt` semantics on `GeneratedImage.revised_prompt`, Vertex video result metadata no
  longer duplicates inline/base64 payloads into public `provider_metadata.videos[]` while the
  task-based runtime preserves a hidden raw-video carrier for reconstruction, and the slice is tracked under
  `docs/workstreams/google-vertex-package-surface-alignment/`.
- `@ai-sdk/google-vertex/anthropic` package-surface parity is now materially tighter too:
  `provider_ext::anthropic_vertex` / `providers::anthropic_vertex` now expose a dedicated
  `GoogleVertexAnthropicProviderSettings` carrier, package-level constructor aliases
  `vertex_anthropic()` / `create_vertex_anthropic()`, the audited Vertex-supported Anthropic tool
  subset under `tools` plus `provider_tools` / `hosted_tools`, and the narrower typed Anthropic
  metadata names (`AnthropicMessageMetadata`, `AnthropicMessageContainerMetadata`,
  `AnthropicMessageContainerSkill`, `AnthropicUsageIteration`) alongside the older wide helper.
  The same wrapper path now also exposes `GoogleVertexAnthropicMessagesModelId`, and its curated
  `models::{chat, ALL_CHAT}` subset is updated to the current audited upstream union instead of the
  stale local `*-latest` aliases. Anthropic-on-Vertex construction is also no longer artificially
  `base_url`-only on the audited paths: provider builder, settings wrapper, and
  registry/unified-builder factory paths now all derive the canonical
  `/publishers/anthropic/models` base URL from explicit `project + location` or
  `GOOGLE_VERTEX_PROJECT` + `GOOGLE_VERTEX_LOCATION`, while still honoring explicit `base_url`
  overrides. This slice is tracked under
  `docs/workstreams/anthropic-vertex-package-surface-alignment/`.
- Google Vertex Gemini image runtime now follows the audited `@ai-sdk/google-vertex` split more
  honestly: `gemini-* image` model ids route through `:generateContent` instead of the Imagen
  `:predict` path, generate/edit/variation now serialize as Gemini multi-part image requests with
  `responseModalities = ["IMAGE"]`, `aspectRatio` and `seed` map into Gemini
  `generationConfig`, Gemini-specific open image options now stay scoped to
  `providerOptions.vertex` on the image-model lane, `mask` and `n > 1` now reject on the Gemini
  image path, and the shared image executor now lets providers opt out of forced URL
  materialization so Vertex Gemini can preserve native URL-backed `fileData.fileUri` inputs on
  edit/variation.
- Stable shared request-facing transport types are now exposed on the Rust facade:
  `CancelHandle`, `TimeoutConfiguration`, `TimeoutConfigurationSettings`, and `RequestOptions`
  live on `siumai::types::*` / `siumai::prelude::unified::*`; runtime stream internals now also
  reuse the shared `CancelHandle` type owned by `siumai-spec`, and this slice is tracked under
  `docs/workstreams/request-options-alignment/`.
- AI SDK-style `RequestOptions` are now consumed by the stable facade helper option structs across
  text, completion, embedding, image, video, speech, transcription, and rerank families. The shared
  adapter maps `maxRetries` to retry attempts with the AI SDK default of 2 when `request_options`
  is present, merges materialized headers and `timeout.totalMs` into helper request HTTP config
  where available, honors `abortSignal` for helper calls and stream handles, and documents the
  remaining `stepMs` / `chunkMs` / tool-timeout runtime gaps in
  `docs/workstreams/request-options-alignment/`.
- Stable shared `LanguageModelCallOptions` / `LanguageModelReasoning` are now exposed on the Rust
  facade as AI SDK-style model-facing generation-control projections from `CommonParams`;
  `CommonParamsBuilder` now also supports `max_completion_tokens`, `CommonParams::cache_hash()`
  now includes it, and this slice is tracked under
  `docs/workstreams/language-model-call-options-alignment/`.
- Shared usage helpers now mirror the AI SDK `usage.ts` helper surface with Rust-style names:
  `as_language_model_usage`, `create_null_language_model_usage`, `add_language_model_usage`, and
  `add_image_model_usage` are exported through the stable facade alongside `LanguageModelUsage`
  and `ImageModelUsage`.
- The shared `Embedding` vector alias from the AI SDK `types/embedding-model.ts` surface is now
  available through `siumai::types::*` and `siumai::prelude::unified::*`.
- The existing stable `EmbeddingModel` trait is now directly exported from
  `siumai::prelude::unified::*`, matching the AI SDK `types/index.ts` model-family export shape.
- The existing runtime `LanguageModelMiddleware` trait is now directly exported from
  `siumai::prelude::unified::*`; embedding/image middleware remain intentionally deferred until
  those model families have real middleware execution hooks.
- The stable video family is now visible from `siumai::prelude::unified::*` as `video`,
  `VideoModel`, `VideoModelV3`, and `VideoModelV4`, matching the already-audited
  `types/video-model.ts` surface.
- The registry `ProviderFactory` interface is now directly visible from
  `siumai::prelude::unified::*` as the honest Rust equivalent of the AI SDK provider model-family
  factory contract; the historical `siumai::Provider` builder entry point remains a compat/top-level
  construction helper, not that provider interface.
- The shared type workstream now includes an explicit audit matrix for
  `repo-ref/ai/packages/ai/src/types/*`, including completed Rust surfaces and the deferred
  embedding/image middleware hooks.
- Shared `ToolChoice` serialization now matches the AI SDK `types/language-model.ts` contract:
  forced tool choices serialize as `{ "type": "tool", "toolName": "..." }` while still accepting
  the previous Rust enum object shape on input for compatibility.
- Shared `FinishReason` serialization now uses the AI SDK public values (`tool-calls`,
  `content-filter`, `other`) while still accepting provider/legacy snake_case values on input.
- The shared language-model `Source` citation shape from the AI SDK `types/language-model.ts`
  surface is now available through `siumai::types::*` and `siumai::prelude::unified::*`, with a
  strict fixed `type: "source"` marker and URL/document `sourceType` payload.
- Prompt-side AI SDK compatibility helpers are now exposed on the stable Rust facade:
  deprecated `CallSettings` now exists as the shared projection of `LanguageModelCallOptions`
  plus non-timeout `RequestOptions`, free timeout helper functions mirror the AI SDK
  `get*TimeoutMs()` helpers over `TimeoutConfiguration`, and this slice is tracked under
  `docs/workstreams/prompt-call-settings-alignment/`.
- Stable AI SDK-style prompt/message shared types are now exposed on the Rust facade:
  `ModelMessage`, `Prompt`, `StandardizedPrompt`, the prompt content-part structs, and explicit
  `ModelMessageConversionError` / `PromptValidationError` now live on `siumai::types::*` and
  `siumai::prelude::unified::*`; prompt-owned shared `DataContent` plus
  `convert_data_content_to_base64_string(...)`, `convert_data_content_to_uint8_array(...)`, and
  `convert_uint8_array_to_text(...)` now also mirror the AI SDK helper role, prompt role/type
  discriminators now deserialize strictly instead of accepting mismatched wire values, the shared
  conversion layer intentionally narrows richer `ChatMessage` / `ContentPart` values instead of
  aliasing them directly, and this slice is tracked under
  `docs/workstreams/prompt-model-message-surface-alignment/`.
- Shared AI SDK-style type surface is now exposed on the stable Rust facade: `JSONSchema7`,
  `JSONValue`, `CallWarning`, `ProviderMetadata`, `ImageModelProviderMetadata`, `LanguageModelUsage`,
  `EmbeddingModelUsage`, `ImageModelUsage`, and the shared request/response metadata structs now
  live on `siumai::types::*` and `siumai::prelude::unified::*`; the shared facade now also
  exposes provider-utils-style `ProviderOptions`, `Context`, `ToolCall`, and `ToolResult`,
  stable `ResponseMetadata` now also preserves optional `headers`, shared warnings now include the
  AI SDK `deprecated` category, and this audit is tracked under
  `docs/workstreams/shared-type-surface-alignment/`.
- Shared tooling/runtime helpers now align much more closely with AI SDK `provider-utils`:
  `siumai::tooling` now exposes `ToolExecutionOptions`, `ToolExecutionResult`, streamed
  `execute_tool(...)`, `ToolSet`, `tool(...)`, `dynamic_tool(...)`, and
  `is_executable_tool(...)`; `ExecutableTool` now supports both one-shot and streamed execution
  bindings while normalizing streamed outputs into `preliminary` / `final` results, the extras
  orchestrator now reuses the shared execution-result type instead of owning a parallel one,
  direct local tool execution paths now also forward shared execution options into
  `ExecutableTools` including `tool_call_id`, projected `ModelMessage`s when representable, and
  shared `context`; runtime input callbacks now also project from the same shared execution
  contract (`onInputStart` over `ToolExecutionOptions`, `onInputDelta` / `onInputAvailable` over
  shared `ModelMessage` / `context` / `abort_signal`, and a dedicated
  `ToolNeedsApprovalContext` for approval checks), streaming orchestrator cancellation now reaches
  both runtime callbacks and local tool execution, approval-continuation of approved local tools
  now also reuses the current shared message history instead of falling back to empty runtime
  messages, and stable tool schemas now expose builders/accessors for `title`, `inputExamples`,
  `strict`, function-tool
  `providerOptions`, and provider-defined-tool
  `providerOptions`; public facade compile/run coverage now also locks the shared runtime carrier
  types (`ToolExecutionStream`, approval/input callback contexts, and runtime metadata accessors),
  and the follow-up audit explicitly records that upstream TypeScript-only `InferTool*`
  conditional helpers are intentionally not mirrored as a fake Rust generic surface. This slice is
  tracked under
  `docs/workstreams/provider-utils-tooling-runtime-alignment/`.
- Amazon Bedrock now has provider-owned image generation aligned with the AI SDK
  `image()` / `imageModel()` surface: builder/config-first/registry/public paths all converge on
  the real `/model/{id}/invoke` image runtime, Bedrock now advertises `image_generation`
  capability, `amazon.nova-canvas-v1:0` keeps the audited `max_images_per_call = 5` default, and
  Bedrock image responses now normalize onto stable base64 image outputs with moderation/error
  handling.
- Bedrock image alignment now also has a dedicated workstream under
  `docs/workstreams/bedrock-image-alignment/`, documenting the audited upstream runtime surface
  and the intentional decision to keep image-only Bedrock provider options private until the
  upstream package exports a public image option type.
- Stream/metadata parity hardening now has a dedicated workstream under
  `docs/workstreams/stream-metadata-parity-hardening/`, covering idempotent textual shadow replay,
  Perplexity hosted-search usage parity, and Gemini/Vertex reasoning stream compatibility.
- Groq browser-search parity now has a dedicated workstream under
  `docs/workstreams/groq-browser-search-alignment/`, documenting the AI SDK reference behavior,
  the compat warning allowlist design, and the provider-owned middleware injection strategy.
- Groq package-surface parity now also has a dedicated workstream under
  `docs/workstreams/groq-package-surface-alignment/`, separating the wider `@ai-sdk/groq`
  typed/model audit from the earlier browser-search-only runtime fix.
- Groq response-metadata parity now also mirrors the upstream `@ai-sdk/groq`
  `get-response-metadata.ts` contract on the provider-owned Rust helper surface: Groq chat
  responses now keep stable `id` / `modelId` / `timestamp` metadata available across non-stream,
  stream-end, config-first, and registry/runtime paths.
- Chat/text/completion streaming now has a runtime-only `includeRawChunks` request lane:
  stable `StreamRequestOptions` lives outside provider wire payloads, `ChatRequest` and
  `CompletionRequest` carry it as runtime-only state, and `siumai::text::StreamOptions` plus
  `siumai::completion::StreamOptions` now map `include_raw_chunks` to that lane instead of
  overloading `providerOptions`.
- The shared OpenAI Responses stream bridge now upgrades more legacy V3 payloads directly onto
  the stable runtime `Part` lane: when provider-prefixed custom events already carry canonical
  `raw`, `custom`, `file`, or `reasoning-file` stream-part JSON, the bridge now emits
  `ChatStreamEvent::Part` instead of preserving them as loose provider-scoped `Custom` events.
- DeepInfra text-family custom base-URL handling now matches the audited
  `repo-ref/ai/packages/deepinfra/src/deepinfra-provider.ts` contract more closely: compat
  DeepInfra config/builder paths now normalize root, `/openai`, and `/inference` inputs onto the
  shared `/openai` text-family prefix; top-level builder/provider/config/registry streaming paths
  now emit equivalent `/openai/chat/completions` requests when callers pass a root base URL; and
  the public-path regression suite now also pins that `includeRawChunks` stays runtime-only plus
  that finish-time `metadataExtractor` merging survives on those DeepInfra stream lanes.
- DeepInfra package-surface aliases are now a bit more complete on the Rust side:
  `provider_ext::deepinfra` now also exports provider-scoped
  `DeepInfraChatModelId` / `DeepInfraCompletionModelId` /
  `DeepInfraEmbeddingModelId` / `DeepInfraImageModelId` string aliases, and both provider-crate
  plus top-level public-surface tests now lock those names.
- `provider_ext::deepinfra` now also exposes package-level unified-provider entry helpers:
  `deepinfra()` and `create_deepinfra()` both return the unified `SiumaiBuilder` surface, while
  the module docs now make it explicit that `DeepInfraClient` / `DeepInfraConfig` are the
  lower-level compat text-family aliases rather than the full hybrid provider entrypoint.
- Stable text-completion family support aligned with AI SDK `completionModel()`:
  `CompletionRequest` / `CompletionResponse`, core `CompletionModel` capability/model contracts,
  registry `completion_model(...)` handles, and facade `siumai::completion::{complete, stream,
  stream_with_cancel}` are now public.
- Stable high-level file upload support aligned with AI SDK `uploadFile()`:
  `siumai::upload_file(...)` and `siumai::files::upload(...)` now expose public
  `UploadFileOptions`, `UploadFileResult`, and `UploadFileProviderMetadata`, auto-detect request
  media types from bytes, reject URL inputs, return canonical `providerReference`, accept shared
  `DataContent` plus direct byte/string carriers, and ship built-in adapters for the current
  file-capable unified/provider clients.
- Stable high-level file uploads now also expose canonical `providerOptions` like AI SDK
  `FilesV4`: shared `FileUploadRequest` plus `UploadFileOptions` carry provider-owned upload
  knobs, OpenAI/Azure honor provider-scoped `purpose` / `expiresAfter`, and Gemini honors
  `displayName` plus polling controls on the upload path.
- Anthropic beta file management now also reuses the shared AI SDK-style file contract end-to-end:
  `AnthropicFiles` / `AnthropicClient` implement `FileManagementCapability`, upload/list/
  retrieve/delete now use shared file-management request/result structs directly, the high-level
  helper no longer needs an Anthropic-only upload bridge, and the redundant provider-local file
  wrapper layer has been removed. This slice is tracked under
  `docs/workstreams/anthropic-files-shared-contract-alignment/`.
- Stable high-level skill upload support aligned with AI SDK `uploadSkill()`:
  `siumai::upload_skill(...)` and `siumai::skills::upload(...)` now expose public
  `UploadSkillFile`, `UploadSkillOptions`, `UploadSkillResult`, and
  `UploadSkillProviderMetadata`; a shared `SkillsCapability` now also bridges `Siumai`,
  `LanguageModelHandle`, and provider clients/resources through one stable upload interface;
  Anthropic also resolves latest-version metadata via
  `/skills/{id}/versions/{version}`, and OpenAI mirrors the audited
  `unsupported { feature: "displayTitle" }` warning behavior.
- Stable AI SDK-style UI-message support is now available on the public Rust surface:
  shared `UiMessage` / `UiMessagePart` / `UiToolPart` types live under `siumai::types::*`,
  `siumai::ui::{validate_ui_messages, convert_to_model_messages, convert_to_chat_request}`
  exposes the conversion helper lane, the top-level `siumai::types` module is restored, and
  stable `tool-approval-response` parts now preserve optional `providerExecuted`. The unified
  prelude now also exports these UI message structures plus passive `UiMessageChunk` stream-event
  carriers, `UI_MESSAGE_STREAM_HEADERS`, and `UiMessageStreamOptions` /
  `UIMessageStreamOptions`, mirroring the AI SDK `ui-message-stream` data shapes and the
  serializable `stream-text-result.ts` UI stream options without claiming frontend stream runtime
  helpers or function-valued callbacks. `siumai::ui::safe_validate_ui_messages(...)` and
  `SafeValidateUiMessagesResult` / `SafeValidateUIMessagesResult` now mirror the upstream
  `safeValidateUIMessages` result-union path for callers that prefer non-throwing validation.
  Passive UI client configuration shapes are also importable from `prelude::unified`:
  `CreateUIMessage`, `ChatRequestOptions`, `ChatStatus`, `ChatState`, `ChatInit`,
  `ChatTransportSendMessagesOptions`, `ChatTransportReconnectToStreamOptions`,
  `HttpChatTransportInitOptions`, passive prepare-request input/result carriers,
  `CompletionRequestOptions`, `UseCompletionOptions`, `RequestCredentials`,
  `CompletionStreamProtocol`, and the schema-map aliases `UIDataPartSchemas` /
  `UIDataTypesToSchemas` / `InferUIDataParts`. Upstream UI part/tool alias names such as
  `UIMessage`, `UIMessagePart`, `TextUIPart`, `DataUIPart`, `ToolUIPart`, `DynamicToolUIPart`,
  `UITool`, and `UITools` now also resolve to the existing Rust UI carriers. The pure
  `ui-messages.ts` helper surface is also importable as Rust functions, including `is_text_ui_part`,
  `is_tool_ui_part`, `get_tool_name`, and the last-assistant-message completion checks for tool
  calls and approval responses. The remaining non-runtime inference/export aliases from
  `ui-messages.ts` and `ui-message-stream/index.ts` are now available too:
  `InferUIMessage*`, `UIMessageChunk`, `InferUIMessageChunk`, `DataUIMessageChunk`, and
  `is_data_ui_message_chunk`. Runtime callbacks, transport classes, custom `fetch`, and hook state
  machines remain intentionally deferred.
- Public AI SDK package facades now also expose package-level provider entry helpers directly on
  their own namespaces: compat-promoted wrappers
  `provider_ext::{mistral,perplexity,fireworks,moonshotai,deepinfra}` plus the audited
  provider-owned facades (`openai`, `anthropic`, `azure`, `bedrock`, `cohere`, `togetherai`,
  `google_vertex::vertex`, `groq`, `xai`, `deepseek`) now mirror the upstream
  `provider` + `createProvider` export pattern instead of forcing callers back to unrelated root
  namespaces.
- Stable prompt-owned `ToolApprovalResponse` values now also expose Rust-idiomatic builder helpers
  for optional `reason` and `providerExecuted`, keeping the public shared prompt surface more
  symmetric with `ToolCall` / `ToolResult` without changing the underlying AI SDK wire contract.
- Stable prompt/content shared structs now also expose first-class Rust builders/accessors for
  upstream `providerOptions` on text/image/file/reasoning/custom/tool parts and all four
  model-message variants, and prompt `ToolCallPart` now also has a dedicated
  `with_provider_executed(...)` helper instead of requiring direct field mutation. Those prompt
  structs now also follow the broader shared-type convention with single-provider convenience
  helpers `with_provider_option(...)` and `provider_option(...)`.
- Runtime `ContentPart::Image` now also preserves optional shared `mediaType`, so prompt
  `ImagePart.mediaType` no longer gets dropped when projecting `ModelMessage` values back into the
  richer chat runtime; bridges that already know image MIME types now retain them instead of
  defaulting immediately to `image/jpeg`.
- Shared `ToolResultOutput` and nested `ToolResultContentPart` provider-option helpers now also
  follow that same convention: the stable Rust surface keeps the older
  `provider_options()` / `provider_options_mut()` names, but now also exposes
  `provider_options_map*`, `with_provider_options_map(...)`, and `provider_option(...)`.
- Prompt `ImagePart` and `FilePart` now also expose focused field-level builders for shared
  optional metadata (`with_media_type(...)` and `with_filename(...)`) so common prompt payloads no
  longer require direct struct-field mutation on the stable Rust surface.
- Nested tool-result `file-url` content parts now also preserve the upstream optional
  `mediaType` field on the stable Rust surface, with serde roundtrip support and a focused
  `with_media_type(...)` builder.
- Stable prompt/content modeling now includes Vercel-aligned `custom`, `reasoning-file`, and explicit tool-result content variants, and provider-backed tool-result parts now preserve the upstream distinction between deprecated `file-id` / `image-file-id` and canonical `file-reference` / `image-file-reference`. Rust serde roundtrips now keep the correct tags and payload keys (`fileId` vs `providerReference`), and OpenAI Responses tool-output normalization now emits canonical provider-reference maps.
- Stable prompt/content modeling now also includes first-class user `providerReference` support
  for `image` / `file` parts through shared `ProviderReference` and `FilePartSource`, with
  builder helpers for both part-level and message-level construction.
- OpenAI Responses request/history alignment now matches the upstream AI SDK approval boundary:
  only provider-executed `tool-approval-response` parts serialize to
  `mcp_approval_response`, request-bridge inspection now reports non-provider-executed approval
  responses as lossy for that target, and normalization restores `providerExecuted: true` from the
  corresponding OpenAI wire item.
- Shared prompt execution validation now also mirrors the upstream AI SDK split between prompt
  standardization and provider-facing history checks: Rust now exposes dedicated
  `MissingToolResultsError` and `PromptExecutionError`, `Prompt::standardize()` remains
  shape-only, and `Prompt::standardize_for_execution()` / `Prompt::to_chat_request()` enforce the
  non-provider-executed tool-call result rule before runtime request conversion.
- Shared `DataContent` is now reusable across more public Rust helper families instead of being
  effectively trapped inside prompt-only APIs: `SttRequest`, `AudioTranslationRequest`,
  `ImageEditInput`, `VideoGenerationInput`, and `files::UploadFileData` now expose direct
  `from_data_content(...)` construction lanes, audio/image/video payload wrappers now also convert
  to and from shared `DataContent`, and this slice is tracked under
  `docs/workstreams/shared-data-content-surface-alignment/`.
- Shared `DataContent` decoding failures now also surface through a stable
  `InvalidDataContentError` instead of leaking `base64::DecodeError` from the shared public
  payload accessors; `DataContent`, audio/image/video file payload wrappers, and the corresponding
  request helper accessors now all use that error lane, and invalid base64 file uploads now reuse
  the same semantic message path. This slice is tracked under
  `docs/workstreams/data-content-error-surface-alignment/`.
- `files::UploadFileData` now also matches the upstream AI SDK `uploadFile` input boundary more
  closely by removing the explicit `Url` variant; URL-like string inputs are still detected and
  rejected at runtime with the same unsupported-upload message, but the stable Rust type shape no
  longer implies that URL uploads are a first-class supported input form. This slice is tracked
  under `docs/workstreams/upload-file-input-shape-alignment/`.
- `files::upload(...)` now also accepts shared `DataContent` directly instead of forcing callers to
  first materialize `UploadFileData`; the helper remains compatible with existing `UploadFileData`
  and direct byte inputs, but the public call boundary is now much closer in spirit to the AI SDK
  `uploadFile` helper. This slice is tracked under
  `docs/workstreams/upload-file-call-boundary-alignment/`.
- The now-redundant `files::UploadFileData` compatibility wrapper has been removed entirely; upload
  helpers now rely on shared `DataContent` plus direct byte/string conversions, while preserving
  the same runtime rejection of URL-like string uploads. This slice is tracked under
  `docs/workstreams/upload-file-wrapper-removal-alignment/`.
- High-level upload-file result shaping now also matches the upstream AI SDK helper more closely:
  shared `FileUploadRequest` / `FileObject` filenames are optional, missing filenames are no
  longer normalized to `blob`, `UploadFileResult.filename` / `media_type` are no longer backfilled
  from request-time fallbacks, and helper `providerMetadata` is now limited to provider-owned
  extra fields instead of injected generic file bookkeeping. This slice is tracked under
  `docs/workstreams/upload-file-result-surface-alignment/`.
- Shared image edit typing now exposes AI SDK-style multi-input `images[]` + `mask` semantics through
  public `ImageEditInput` and `ImageEditFileData` types on the extensions/facade surface.
- Shared image request typing now exposes top-level `aspectRatio` across generation/edit/variation
  plus shared `seed` support across the same request family, bringing the stable image call-option
  surface much closer to AI SDK `ImageModelV4CallOptions`.
- Stable unified image helper support is now available on the public Rust surface:
  `GenerateImageRequest` plus `siumai::image::{generate_image, edit, variation}` bridge one
  AI SDK-style request/helper lane onto the current generation/edit/variation runtimes, while the
  older split request structs remain available as compatibility surfaces.
- Stable image helper batching now also aligns with AI SDK `maxImagesPerCall`: object-safe
  `max_images_per_call()` metadata lives on `ImageGenerationCapability` / `ImageModelV3`,
  `siumai::image::GenerateOptions` now accepts `max_images_per_call`, and
  `siumai::image::{generate, edit, variation, generate_image}` split larger `count` requests
  across explicit limits or audited provider defaults while preserving per-call metadata and
  response envelopes under `metadata._siumai` for multi-call aggregation. Successful-but-empty
  helper runs now also return `LlmError::NoImageGenerated` with final response metadata instead of
  silently returning an empty image list.
- Image provider-option parity now has a dedicated workstream under
  `docs/workstreams/ai-sdk-structural-alignment/image-provider-option-surface-parity.md`,
  covering Gemini/Google image aliases, unified image request-ext coverage, and merge semantics on
  the audited TogetherAI/xAI/Vertex/Gemini provider lanes.
- Audited provider-owned image option surfaces now also align more closely with AI SDK on the
  public Rust facade: `provider_ext::{gemini,google}` expose `GeminiImageOptions`,
  `GoogleImageModelOptions`, and deprecated `GoogleGenerativeAIImageProviderOptions`; image request
  ext traits on Gemini/xAI/TogetherAI/Google Vertex now also cover the stable unified
  `GenerateImageRequest`; and those typed helpers now merge onto existing provider-owned
  `providerOptions` objects instead of overwriting sibling raw fields.
- Google Vertex image typed options now also expose the main audited generation-field subset from
  AI SDK `GoogleVertexImageModelOptions`: `personGeneration`, `safetySetting`, `addWatermark`,
  `storageUri`, and `sampleImageSize`, so callers no longer need raw `providerOptions.vertex`
  objects for those settings.
- Shared image variation typing now also uses a typed file/url image input instead of a raw
  byte-only field, aligning the stable variation request shape more closely with AI SDK
  `ImageModelV4File`.
- Shared image edit/variation input typing now also exposes per-input `providerOptions` on typed
  file/url image inputs, closing another structural gap against AI SDK `ImageModelV4File`.
- Shared video generation typing now exposes AI SDK-style typed file/url inputs through public
  `VideoGenerationInput`, plus canonical `count` (`n`), `fps`, and `seed` request knobs in place
  of the older raw `seed_image` / `seed_video` byte fields.
- Shared video input typing now also exposes per-input `providerOptions` on typed file/url inputs,
  bringing the stable `VideoGenerationInput` shape closer to AI SDK `VideoModelV4File`.
- Task-oriented video model family support is now available on the public Rust surface:
  `siumai-core` now exposes `VideoModelV3` / `VideoModelV4` / `VideoModel`,
  `siumai-registry` now exposes dedicated `video_model(...)` construction plus `VideoModelHandle`
  with native build-context propagation and LRU/TTL caching, and `siumai::video::{create_task,
  query_task}` provides the stable facade helper lane while the older `LanguageModelHandle` video
  capability remains as a compatibility bridge.
- The public Rust video facade now also exposes high-level polling helpers:
  `siumai::video::wait_for_task(...)` polls task-based providers to completion, and
  `siumai::video::generate(...)` submits and polls one or more video tasks while preserving
  Rust-first task semantics. The helper now batches by explicit or model-default
  `max_videos_per_call`, returns final `GeneratedVideo` assets separately from completed task
  responses, and preserves per-call plus aggregated provider metadata for multi-call execution.
- Final generated-video assets now also have explicit materialization helpers closer in role to AI
  SDK `GeneratedFile`: `GeneratedVideo::materialize(...)` and
  `GenerateVideoResult::{materialize_video,materialize_videos,into_materialized}` plus
  `siumai::video::generate_materialized(...)` download URL-backed videos on demand, preserve
  byte/base64 accessors on the materialized file representation, and keep provider-reference-only
  assets as an explicit unsupported path instead of pretending they are generically downloadable.
- The high-level Rust video helper now also matches AI SDK `experimental_generateVideo()` more
  closely at the result boundary: `siumai::video::generate(...)` now auto-materializes URL-backed
  final videos by default, `GenerateVideoResult` / `GenerateMaterializedVideoResult` now expose an
  AI SDK-style first `video` alongside `videos`, `GeneratedVideo` now has direct `bytes()` /
  `base64()` accessors for already-inline assets, and callers can still opt out of default URL
  downloads or pass helper-level download `HttpConfig`. Non-downloadable schemes such as current
  Vertex `gs://...` outputs now stay URL-backed with a warning instead of failing helper-level
  materialization. This slice is tracked under
  `docs/workstreams/video-generate-result-materialization-alignment/`.
- Provider-reference-only final video assets now also have a shared provider-owned materialization
  hook on the task-oriented video family: `MaterializedVideoAsset` now lives on the shared video
  type surface, `VideoGenerationCapability` / `VideoModelV3` expose
  `materialize_video_reference(...)`, `siumai::video::generate(...)` best-effort materializes
  those references through the same model-capability dispatch path, and audited Gemini/MiniMaxi
  providers now reuse their existing file-management runtimes on that path. This slice is tracked
  under `docs/workstreams/video-provider-reference-materialization-alignment/`.
- Stable video task-status payloads now also promote provider-owned final assets onto canonical
  `providerReference`, prefer camelCase serde output (`taskId`, `fileId`, `videoUrl`,
  `providerReference`, `videoWidth`, `videoHeight`, `baseResp`) while still accepting snake_case
  compatibility input, and `siumai::video` now consumes that canonical reference before falling
  back to legacy `fileId` inference. Top-level public-path parity now also locks the audited query
  split: Gemini returns canonical `providerReference`, while Vertex stays on raw `videoUrl`
  without fabricating a shared provider reference, and the feature-gated Gemini provider-local
  video regression lane now compiles again under the real `google` test feature. Direct
  `GeminiVideo::new(...)` helper construction now also reuses `GeminiConfig.http_transport` for
  task polling when no explicit transport override is passed. Video-family metadata readers now
  also accept upstream `google-vertex` alias roots on the read path while preserving the stable
  Rust aggregation root under `vertex`.
- Video-family batching and result shaping now align more closely with AI SDK
  `experimental_generateVideo()`: object-safe `max_videos_per_call()` metadata now lives on
  `VideoGenerationCapability` / `VideoModelV3` / registry video handles, audited defaults are
  exposed for Gemini/Vertex/xAI/MiniMaxi, and Gemini/Vertex polling metadata now keeps provider-
  owned `videos[]` entries so the facade can recover multi-video final assets instead of exposing
  only raw task responses. Successful-but-empty video runs now also return
  `LlmError::NoVideoGenerated` with final response metadata instead of collapsing that case into a
  generic parse failure, and aggregated video `provider_metadata` now preserves provider-root
  fields beyond `videos[]` / `tasks[]` during multi-call merges.
- Shared video request typing now also aligns more closely with AI SDK `GenerateVideoPrompt`:
  `VideoGenerationRequest.prompt` is now optional, `VideoGenerationRequest::new_without_prompt(...)`
  supports image-only flows on the stable surface, Gemini/Vertex accept prompt-less image-to-video
  requests on their provider-owned runtimes, xAI/MiniMaxi now fail fast when callers omit a
  prompt on routes that still require one, and the stable public surface now exposes
  `VideoGenerationPrompt` plus the AI SDK-auditable alias `GenerateVideoPrompt` for the same
  text-or-image prompt union.
- Video model family alignment now also has a dedicated workstream under
  `docs/workstreams/video-model-family-alignment/`, documenting the deliberate Rust-first
  task-based contract and the remaining provider-owned download gap against the upstream helper.
- Shared AI SDK-style video metadata is now exposed on the stable Rust surface:
  `VideoModelProviderMetadata` and `VideoModelResponseMetadata` live on
  `siumai::types::*` / `siumai::prelude::unified::*`, and `siumai::video::{GenerateVideoResult,
  GenerateMaterializedVideoResult, GenerateVideoResponseMetadata}` now provide best-effort
  accessors that project task-oriented video helper responses onto that shared AI SDK metadata
  view without hiding the underlying create/query lifecycle.
- Shared transcription and audio-translation typing now uses a canonical `audio` input plus
  `mediaType` / `providerOptions`, replacing the older stable `audio_data | file_path` split and
  bringing the request surface much closer to AI SDK `TranscriptionModelV4CallOptions`.
- Stable speech/transcription helper semantics now align more closely with AI SDK
  `generateSpeech()` / `transcribe()`: `TtsResponse` and `SttResponse` now preserve best-effort
  final `response` metadata, the raw stable audio response structs now also expose optional
  `warnings` plus `provider_metadata`, the shared audio executor carries that envelope across the
  audited provider paths, and the public facades now return richer helper result objects
  (`speech::SpeechResult` / `speech::GenerateSpeechResult` and
  `transcription::TranscriptionResult`) while still preserving compatibility mirrors for the older
  Rust fields. Successful-but-empty `siumai::speech::synthesize(...)` /
  `siumai::transcription::transcribe(...)` calls now also return `LlmError::NoSpeechGenerated` /
  `LlmError::NoTranscriptGenerated` instead of silently returning empty audio/text.
- Stable speech request typing now also closes the last obvious AI SDK shared call-option gap:
  `TtsRequest` carries first-class `instructions` and `language`, exposes
  `with_output_format(...)` as an AI SDK-style alias for `with_format(...)`, and the shared
  OpenAI-family audio transformer now consumes unified `instructions` directly instead of requiring
  provider-owned `providerOptions` for that common speech field. Native OpenAI/Azure speech also
  now warns and falls back to `mp3` for unsupported `outputFormat` values, while `language`
  surfaces an explicit warning instead of being silently dropped.
- Stable speech/transcription result metadata now also preserves AI SDK-style optional `request`
  envelopes: shared audio execution captures the final JSON request body on HTTP JSON routes,
  `TtsResponse` / `SttResponse` plus `speech::SpeechResult` / `transcription::TranscriptionResult`
  expose that best-effort `request.body`, and the audited OpenAI speech path now returns the same
  debuggable request payload that upstream `OpenAISpeechModel.doGenerate()` exposes.
- Shared transcription request typing now also matches AI SDK `TranscriptionModelV4CallOptions`
  more strictly: `SttRequest` no longer carries top-level `language` or
  `timestamp_granularities`, OpenAI-family multipart shaping now reads those knobs only from
  provider-owned `providerOptions` / escape hatches, and provider-owned typed request ext helpers
  now exist for OpenAI and Groq transcription options so callers can stay on typed surfaces
  without reintroducing shared-structure drift.
- DeepInfra now has a dedicated workstream under
  `docs/workstreams/deepinfra-unified-provider-surface/`, documenting the chosen first-class
  provider architecture and remaining follow-up audit scope.
- Vertex MaaS now also has a dedicated workstream under
  `docs/workstreams/vertex-maas-unified-provider-surface/`, documenting the chosen first-class
  provider architecture, Google auth semantics, and remaining follow-up scope.
- Cohere now also has a dedicated workstream under
  `docs/workstreams/cohere-unified-provider-surface/`, documenting the unified native `/v2`
  provider design, validation, and remaining parity audit scope.
- Fireworks now also has a dedicated workstream under
  `docs/workstreams/fireworks-unified-provider-surface/`, documenting the chosen hybrid provider
  architecture, provider-owned image routes, and remaining parity audit scope.
- Ollama and MiniMaxi now also have dedicated workstreams under
  `docs/workstreams/ollama-unified-provider-surface/` and
  `docs/workstreams/minimaxi-unified-provider-surface/`, documenting the chosen package-shape
  cleanup and single-source model/catalog alignment.
- DeepSeek, Ollama, and MiniMaxi now expose provider-owned curated model surfaces on the public
  facade: `provider_ext::deepseek::{chat, model_sets}`, `provider_ext::ollama::{chat, embedding,
  model_sets}`, and `provider_ext::minimaxi::{chat, speech, video, music, image, model_sets}`.
- DeepSeek now also exposes AI SDK-style `DeepSeekLanguageModelOptions` with deprecated
  `DeepSeekChatOptions` migration coverage on the provider-owned typed surface.
- DeepSeek now also exposes `DeepSeekErrorData` on the provider-owned/public facade boundary,
  matching the audited `@ai-sdk/deepseek` package export instead of leaving provider error
  envelopes accessible only through the generic OpenAI-compatible layer.
- TogetherAI now also exposes `TogetherAIErrorData` on the provider-owned/public facade boundary,
  aligning the audited package-level error envelope without widening the Rust surface to
  TypeScript-only `ProviderSettings` or `VERSION` exports.
- OpenAI-compatible package alignment now also exposes exact-case `OpenAICompatible*` public
  aliases alongside the existing `OpenAiCompatible*` Rust spellings, so the shared compat facade
  matches `repo-ref/ai/packages/openai-compatible/src/index.ts` more directly without breaking
  existing imports.
- xAI, Groq, and Amazon Bedrock now also expose AI SDK-style provider-option aliases on the
  provider-owned/public facade boundary, including upstream-style deprecated migration aliases
  where the audited AI SDK packages still export them.
- Anthropic package-surface naming is now closer to the audited AI SDK indices:
  `provider_ext::anthropic` exposes `AnthropicLanguageModelOptions`, deprecated
  `AnthropicProviderOptions`, `AnthropicMessageMetadata`, `AnthropicUsageIteration`, and
  `AnthropicToolOptions`; Anthropic request/tool helper writes now merge onto existing
  `providerOptions.anthropic` objects; `AnthropicMessageMetadata` is now a dedicated narrow typed
  struct rather than a thin alias to the wider `AnthropicMetadata` helper; the Anthropic tool path
  now forwards `eagerInputStreaming`; and the stable Bedrock facade conditionally mirrors the
  upstream `AnthropicProviderOptions` cross-export when both `anthropic` and `bedrock` features
  are enabled. The audit is documented under
  `docs/workstreams/anthropic-package-surface-alignment/`.
- Anthropic fixture-backed typed metadata alignment is now tighter too: the audited
  `anthropic-web-search-tool.1` non-stream and stream-end fixtures now explicitly lock
  `AnthropicMessageMetadata` to the AI SDK message-metadata shape (`usage` plus explicit `null`
  `stopSequence` / `iterations` / `container` / `contextManagement`) instead of relying only on
  lower-level unit tests and synthetic public roundtrips, and the narrow message-metadata
  container/skill surface now uses dedicated required-field structs rather than inheriting the
  wider helper container's optional-field looseness.
- Groq now also exposes AI SDK-style `browser_search()` provider tools on the public Rust facade
  through `provider_ext::groq::{tools, provider_tools}`.
- Azure now also mirrors the audited `@ai-sdk/azure` option-alias surface on both the
  provider-owned and public facade boundary:
  `OpenAILanguageModel{Chat,Responses}Options` plus deprecated
  `OpenAI{ChatLanguageModelOptions,ResponsesProviderOptions}`. `with_azure_options(...)` now also
  merges into existing `providerOptions.azure` objects instead of replacing sibling raw fields.
- OpenAI-compatible vendor-view public data structures now also align more closely with the
  audited AI SDK package indices: `provider_ext::openai_compatible` now exposes generic
  `OpenAiCompatible{Chat,Completion,Embedding,Image}ModelId`, generic typed option structs for
  chat/completion/embedding, the deprecated AI SDK migration aliases for those generic options,
  `OpenAiCompatibleErrorData`, and `with_openai_compatible_options(...)` for the shared
  `providerOptions.openaiCompatible` namespace. `provider_ext::deepinfra` re-exports
  `DeepInfraErrorData`, `provider_ext::fireworks` re-exports `FireworksErrorData` plus
  `FireworksEmbeddingModelId` / `FireworksImageModelId`, and `provider_ext::moonshotai`
  re-exports `MoonshotAIChatModelId`. Rust intentionally keeps those model ids as stable
  `String` aliases instead of inventing a larger frozen enum surface beyond the curated model
  constants, and the generic JS provider-function/settings exports remain represented by the
  existing Rust `OpenAiCompatibleBuilder` / `OpenAiCompatibleConfig` / `OpenAiCompatibleClient`
  story rather than a fake callable provider type. The shared generic helper lane now reaches all
  three relevant request families: chat/completion/embedding all have typed request helpers under
  the same `openaiCompatible` namespace, and the generic chat options also have no-network
  builder/provider/config/registry parity on the real OpenRouter public path, so this surface is
  no longer compile-only on the top-level facade.
- Groq now also exposes AI SDK-style `GroqTranscriptionModelOptions` on the provider-owned/public
  facade boundary. The provider-owned transcription helper surface now accepts
  `language` / `responseFormat` / `timestampGranularities`, and Groq STT requests and responses
  keep the matching runtime fields (`language`, `duration`, `segments`, `x_groq`) instead of
  dropping them after lowering.
- Native OpenAI now also exposes the main AI SDK-style typed option surface on the
  provider-owned/public facade boundary:
  `OpenAILanguageModel{Chat,Responses,Completion}Options`,
  `OpenAIEmbeddingModelOptions`, `OpenAISpeechModelOptions`,
  `OpenAITranscriptionModelOptions`, `OpenAIFilesOptions`, plus the upstream deprecated
  migration aliases `OpenAIChatLanguageModelOptions` and `OpenAIResponsesProviderOptions`.
- Google Vertex now also exposes the audited AI SDK-style typed option surface on the
  provider-owned/public facade boundary: `GoogleVertexEmbeddingModelOptions`,
  `GoogleVertexImageModelOptions`, deprecated `GoogleVertexImageProviderOptions`,
  `GoogleVertexReferenceImage`, `GoogleVertexVideoModelOptions`, deprecated
  `GoogleVertexVideoProviderOptions`, and `GoogleVertexVideoModelId`. The native Vertex package
  now also owns a real Veo task runtime on `:predictLongRunning` / `:fetchPredictOperation`,
  public/path and lower-contract tests lock video create/query parity, and the provider catalog
  plus native provider metadata now advertise Vertex video support and curated `veo-*` model ids.
- Google now also has a dedicated package-alignment workstream under
  `docs/workstreams/google-package-surface-alignment/`, documenting the audited
  `@ai-sdk/google` package boundary, the Google-branded typed surface layered on top of the
  provider-owned Gemini implementation, and the intentional Rust-side deferrals for TypeScript-only
  provider factory/settings exports plus task-based video polling ownership.
- OpenAI-compatible package-surface parity now also has a dedicated workstream under
  `docs/workstreams/openai-compatible-package-surface-alignment/`, documenting the audited
  `@ai-sdk/openai-compatible` package boundary and the intentional Rust-side deferrals for
  TypeScript-only provider factory/settings exports plus callable image-model wrappers.
- The provider-owned/public Google facade is now much closer to the audited `@ai-sdk/google`
  package surface: `provider_ext::google::{options::*, metadata::*, *}` now exposes
  `GoogleLanguageModelOptions`, `GoogleEmbeddingModelOptions`, `GoogleVideoModelOptions`,
  `GoogleVideoModelId`, `GoogleFilesUploadOptions`, `GoogleProviderMetadata`,
  `GoogleProviderSettings`, and `GoogleErrorData` plus the upstream deprecated aliases;
  Google-branded request/upload helpers now lower `serviceTier`, `streamFunctionCallArguments`,
  embedding `outputDimensionality` / `taskType` / positional multimodal `content`, and video
  `negativePrompt` / `personGeneration` / `referenceImages` onto the real provider-owned runtime;
  response metadata now also keeps typed `promptFeedback` alongside `usageMetadata`,
  `finishMessage`, and `serviceTier`; `Provider::google()` / `Siumai::builder().google()` plus
  `provider_ext::google::{google, create_google}` now also mirror the upstream package-level
  `google` / `createGoogle` entry naming instead of forcing callers onto the internal `gemini`
  name; `provider_ext::google` now also exposes the package `VERSION` constant plus a deprecated
  `create_google_generative_ai()` alias for the audited `createGoogleGenerativeAI` root export;
  `Provider::google()` / `Provider::gemini()` now also mirror the audited non-callable family
  helper names for chat, embedding, image, and video model selection; and
  `provider_ext::google::{chat, embedding, image, video, model_sets}` now exposes grouped audited
  Google model ids for direct package-surface diffing; the provider-owned Gemini support lists now
  also include `gemini-embedding-2-preview` and the newer audited Veo / Gemini package ids instead
  of lagging behind the public facade; `GoogleProviderSettings` is now also a dedicated
  provider-level input struct with `with_api_key`, `with_base_url`, `with_headers`, `with_fetch`,
  `with_generate_id`, `with_name`, and `into_builder*` helpers instead of a misleading
  `GeminiConfig` alias; `Provider::google()` / `Provider::gemini()` now also expose a direct
  `files()` member that returns the provider-owned `GeminiFiles` capability, matching the audited
  `google.files()` resource-construction story more closely;
  Gemini builder configuration now honors upstream `GOOGLE_GENERATIVE_AI_API_KEY` as the primary
  Google package environment-variable fallback while keeping legacy `GEMINI_API_KEY`
  compatibility; the Google package-alignment docs now explicitly record that Rust still
  intentionally does not fabricate a callable `GoogleProvider`; and Gemini option normalization now
  preserves `null` entries inside Google embedding `content[]` instead of collapsing text-only
  positions.
- Google package-surface alignment now also closes the previously deferred upstream `generateId`
  hook: `GoogleProviderSettings::with_generate_id(...)`, `GeminiBuilder::with_generate_id(...)`,
  and the public `SharedIdGenerator` facade now thread custom stable ID generation through
  `GeminiConfig`, and Gemini response/streaming transformers consume it for provider-owned
  tool-call, tool-result, and source ids instead of hard-coded local id generation.
- Google package-surface alignment now also closes the upstream `name` hook honestly:
  `GoogleProviderSettings::with_name(...)`, `GeminiBuilder::name(...)`, and
  `GeminiConfig::with_provider_name(...)` now carry a provider-facing display label through the
  Google package path; `Provider::google()` defaults that label to `google.generative-ai` while
  `Provider::gemini()` keeps `gemini`, `GeminiClient::provider_name()` and
  `GeminiFiles::provider_name()` now expose the resolved label, and provider-facing Gemini error
  text can use it without changing canonical `provider_id`, `providerReference`, or
  `providerMetadata` roots.
- Provider-owned public wrapper surfaces are now also more structurally uniform: the audited
  first-class `provider_ext::<provider>` paths now consistently re-export their native
  `*Builder` types alongside `*Client` / `*Config`, making the Rust facade closer to the audited
  AI SDK package entry-point shape and easier to diff against `repo-ref/ai`.
- MoonshotAI now also has a dedicated package-alignment workstream under
  `docs/workstreams/moonshotai-package-surface-alignment/`, documenting the canonical `moonshotai`
  wrapper contract and the intentional chat-only boundary inherited from `repo-ref/ai`.
- Mistral now also has a dedicated package-alignment workstream under
  `docs/workstreams/mistral-package-surface-alignment/`, documenting the audited `chat +
  embedding` wrapper boundary inherited from `repo-ref/ai`.
- Perplexity now also has a dedicated package-alignment workstream under
  `docs/workstreams/perplexity-package-surface-alignment/`, documenting the public typed option
  surface and its explicit lowering onto the Perplexity wire contract.
- xAI now also has a dedicated package-alignment workstream under
  `docs/workstreams/xai-package-surface-alignment/`, documenting the audited `@ai-sdk/xai`
  export boundary, provider-owned file-upload lane, and the intentional Rust-side deferrals for
  TypeScript-only factory/settings exports.
- xAI package-surface parity now also closes the remaining audited video/files option drift:
  `XaiVideoOptions` now covers upstream `mode` and `referenceImageUrls`, the public xAI
  video/files option structs now serialize the AI SDK-facing camelCase shape while still accepting
  legacy snake_case aliases, and the provider-owned xAI video runtime now routes
  `mode = "extend-video"` to `/videos/extensions` while lowering
  `mode = "reference-to-video"` onto the normal generation path with `reference_images`.
- xAI package-surface parity now also closes the remaining public `providerOptions.xai` naming
  drift on the text/search lanes: `XaiChatOptions`, `XaiResponsesOptions`,
  `XaiSearchParameters`, and discriminated search-source structs now serialize the audited
  AI SDK-facing camelCase fields while still accepting legacy snake_case aliases, and the
  provider-owned xAI config/runtime path keeps lowering those public shapes onto native snake_case
  wire keys.
- Amazon Bedrock embedding parity now also has a dedicated workstream under
  `docs/workstreams/bedrock-embedding-alignment/`, documenting the audited AI SDK embedding
  export/runtime contract, provider-owned embedding standard, and registry/public-path parity
  coverage.
- The public MoonshotAI wrapper surface is now much closer to the audited `@ai-sdk/moonshotai`
  package: `provider_ext::moonshotai::{MoonshotAIClient, MoonshotAIConfig, model_sets,
  recommended}`, typed `MoonshotAIChatOptions` / `MoonshotAILanguageModelOptions`, and
  `MoonshotAIChatRequestExt` are now available on the Rust facade.

### Changed

- Experimental streaming V4 payload aliases now use explicit `LanguageModelV4Stream*` names
  (`LanguageModelV4StreamToolCall`, `LanguageModelV4StreamFile`, `LanguageModelV4StreamUsage`,
  etc.) so they no longer collide with the standalone provider-result `LanguageModelV4*` data
  structures exported from `siumai::types`.
- AI SDK structural-alignment documentation now records the bounded root-export audit: `generateText`
  is covered by the single-step Rust projection, `streamText` remains passive-result parity until
  a real Rust stream-result runtime exists, AI SDK `text-stream` response helpers are covered by
  the Axum extras adapter, and browser UI / Vercel Gateway exports such as `AbstractChat`,
  `callCompletionApi`, `convertFileListToFileUIParts`, `gateway`, and `createGateway` are
  intentionally deferred instead of represented by fake root exports.
- OpenAI and Anthropic provider-owned `skills()` resources now consume the shared
  `SkillUploadRequest` / `SkillUploadResult` contract directly; the redundant public
  `OpenAiSkill*` / `AnthropicSkill*` wrapper types and bespoke `upload(...)` resource methods
  have been removed so the provider resource surface matches the audited AI SDK `SkillsV4` split
  more honestly.
- Public streaming examples, migration snippets, and gateway bridge samples now treat stable
  `Part(TextDelta)` / `PartWithReplay(TextDelta)` as first-class streamed text instead of
  teaching legacy-only `ContentDelta` consumers.
- Additional public streaming samples now also consume the stable semantic lane by default:
  advanced middleware, Anthropic web-search streaming, MiniMaxi basic streaming, OpenAI
  Responses streaming tools/websocket examples, registry quickstart, and the custom-provider
  implementation sample now all read stable `Part(TextDelta)` / `PartWithReplay(TextDelta)`
  first, and the MiniMaxi sample also reads stable `ReasoningDelta` before falling back to legacy
  `ThinkingDelta`.
- The public image model family now also exposes AI SDK-style `ImageModelV4` on core,
  `siumai::image`, and unified-prelude boundaries while keeping `ImageModel` / `ImageModelV3`
  as compatibility aliases.
- Groq's provider-owned PlayAI speech models now live in a provider-extension catalog that is
  separate from the AI SDK-aligned chat/transcription model groups, making the public Rust surface
  clearer about which constants mirror `@ai-sdk/groq` and which remain Rust-only extensions.
- `provider_ext::groq::{options::*, *}` now keep the AI SDK-aligned option lane centered on
  `GroqLanguageModelOptions` and `GroqTranscriptionModelOptions`; the concrete provider-owned
  audio escape hatches `GroqSttOptions` / `GroqTtsOptions` remain available only under
  `provider_ext::groq::ext::audio_options::*`.
- `provider_ext::groq::*` now also re-exports `GroqBuilder`, keeping the provider-owned Groq
  construction lane visible on the stable facade alongside `GroqClient` / `GroqConfig`.

### Fixed

- OpenAI-compatible DeepSeek, Fireworks, Mistral, Perplexity, and xAI presets now declare their
  canonical AI SDK API-key environment variables for config-first construction.
- OpenAI-compatible Groq now advertises its AI SDK transcription surface with a default
  transcription model.
- OpenAI-compatible usage conversion now applies the audited AI SDK provider-specific semantics for
  DeepSeek prompt-cache hits, MoonshotAI top-level cache counters, Groq reasoning without prompt
  cache normalization, Qwen/Alibaba cache-write counters, and xAI chat/Responses non-inclusive
  cache and reasoning totals while still preserving the original provider usage payload in
  `Usage.raw`.
- DeepSeek OpenAI-compatible response metadata now mirrors AI SDK's prompt-cache usage fields as
  `providerMetadata.deepseek.promptCacheHitTokens` / `promptCacheMissTokens`, and the typed
  DeepSeek metadata helper exposes those fields.
- OpenAI-compatible finish-reason conversion now applies AI SDK provider-specific mappings for
  DeepSeek `insufficient_system_resource`, Mistral `model_length`, Perplexity's narrower
  stop/length contract, and Cohere's uppercase stop reasons while continuing to preserve the raw
  provider finish reason.
- OpenAI Responses response encoding now replays `raw_finish_reason` as
  `incomplete_details.reason` for incomplete outputs and marks those responses as
  `response.incomplete`, so `max_output_tokens`, `content_filter`, and future provider-specific
  incomplete reasons round-trip without false bridge loss reports.
- Completion responses now expose provider-native raw finish reasons on OpenAI, Azure, and
  OpenAI-compatible non-streaming completion paths, matching the existing AI SDK-style streaming
  fidelity where `finishReason.raw` preserves the original `choices[0].finish_reason` string.
- Ollama chat conversion now preserves native `done_reason` as `raw_finish_reason` on
  non-streaming and streaming responses, uses that raw value for `length`/provider-specific
  finish mapping, and replays it as `done_reason` when serializing Ollama JSONL stream endings.
- Experimental response bridge loss reports now treat replayable raw finish reasons as exact for
  OpenAI Chat Completions, Anthropic Messages, and Gemini GenerateContent targets instead of
  flagging `finish_reason` as lossy after the encoder has already preserved the provider-native
  value.
- Gemini GenerateContent response parsing and protocol reserialization now preserve provider-native
  `finishReason` strings on the non-streaming path and prefer those raw values when replaying JSON
  or SSE responses, matching the existing streaming raw-finish fidelity and avoiding lossy
  `PROHIBITED_CONTENT` -> `SAFETY` style round-trips.
- AI SDK-style video polling controls are now honored by the shared high-level video helper:
  `siumai::video::generate(...)` consumes provider-owned `pollIntervalMs` / `pollTimeoutMs` from
  Vertex, Gemini/Google, and xAI video provider options through a new `VideoPollingOptions`
  capability hook instead of incorrectly treating those runtime-only controls as ignored
  task-submission fields.
- Google Vertex express-mode authentication now wins consistently when `GOOGLE_VERTEX_API_KEY`
  supplies the API key, suppressing token-provider auth just like the audited AI SDK node wrapper
  does when an effective API key is present.
- Anthropic native structured-output alignment is now tighter against the audited
  `@ai-sdk/anthropic` contract: native JSON Schema output lowers to `output_config.format`
  instead of the deprecated `output_format` field, `output_config.{format,effort,task_budget}`
  now merge without clobbering each other, request normalization/bridge round-trips restore
  `taskBudget` from the same native shape, `container.skills` now preserves the audited public
  split between Anthropic `skillId` and custom `providerReference`, and stream-time
  structured-output mode selection no longer falls back to the JSON-tool path merely because
  tools are present.
- Anthropic typed option parity is now tighter too: public/provider/shared builder surfaces now
  expose `taskBudget` and `inferenceGeo` helpers consistently, the typed effort enum now includes
  upstream `xhigh`, adaptive thinking now preserves `display` through request normalization/body
  finalization, and Anthropic requests lower those fields onto native `output_config.task_budget`,
  `inference_geo`, and `thinking.display` instead of silently dropping them.
- `@ai-sdk/anthropic` package-surface parity now also includes the package-level container carry-
  forward helper: `provider_ext::anthropic::{find_anthropic_container_id_from_last_step,
  forward_anthropic_container_id_from_last_step}` mirrors upstream
  `forwardAnthropicContainerIdFromLastStep(...)` for prepare-step/provider-metadata workflows.
- OpenAI Responses MCP fixture parity now matches the audited AI SDK request contract again:
  provider-executed `tool-approval-response` fixtures explicitly set `providerExecuted: true`,
  restoring exact `mcp_approval_response` request-fixture coverage and the full
  `openai_responses_*` regression sweep.
- Shared `FinishReason` serde now serializes `Other(...)` as the AI SDK string-union shape while
  still accepting the legacy object form on read, fixing failed OpenAI Responses streams where
  stable `finishReason.unified` must be `"other"` instead of a Rust enum object.
- OpenAI-family speech request shaping now actually lowers shared `TtsRequest.language` into the
  JSON body when a provider's audited defaults opt into that field, so the stable typed speech
  surface no longer carries a dead `language` knob on those paths.
- OpenAI/Azure/xAI file uploads now merge builder/config default `ProviderOptionsMap` values into
  `FileUploadRequest.provider_options`, so default provider-owned file knobs such as OpenAI
  `purpose` / `expiresAfter` and xAI file path options no longer work for other non-chat families
  while being silently skipped on files.
- Azure Responses request shaping now handles schema-less `ResponseFormat::JsonObject` on
  reasoning-model requests, restoring exhaustive provider-crate compilation and matching the
  OpenAI Responses `text.format = { type: "json_object" }` lowering.
- The built-in OpenAI-compatible `groq` preset now defaults `supportsStructuredOutputs = true`,
  matching the audited AI SDK package behavior so Groq JSON Schema requests no longer degrade to
  `response_format = { "type": "json_object" }` by default.
- OpenAI Responses request conversion now omits tool-role messages whose parts are all
  intentionally skipped, preventing false `Tool message missing tool result` errors on AI
  SDK-shaped histories that contain only non-provider-executed approval responses in a tool
  message.
- OpenAI typed control options now also mirror the audited AI SDK surface more closely:
  `OpenAILanguageModelChatOptions` now exposes `systemMessageMode`, and
  `OpenAILanguageModelResponsesOptions` now exposes `systemMessageMode`, `forceReasoning`, and
  `contextManagement`; public OpenAI/Azure option re-exports include the new typed controls, and
  `/responses` request shaping now lowers `contextManagement[].compactThreshold` to native
  `context_management[].compact_threshold` while keeping control-only fields off the wire body.
- OpenAI-compatible chat streaming now matches AI SDK `doStream()` error semantics more closely:
  explicit top-level `{"error": ...}` chunks and invalid JSON chunks now emit stable `error` plus
  error `finish` / `StreamEnd` semantics instead of surfacing only transport parse failures, and
  the shared compat stream audit now also pins finish-time `acceptedPredictionTokens` /
  `rejectedPredictionTokens` plus public metadata-extractor merging on the streaming path.
- OpenAI-compatible chat streaming now also matches the audited AI SDK ordering/terminal behavior
  more closely on two remaining stream edges: if one chunk carries both reasoning and text, the
  stable `reasoning-*` lane now opens before `text-*`, and explicit
  `finish_reason = "tool_calls"` chunks now finalize pending stable `tool-input-end` /
  `tool-call` parts including empty-input tool calls without duplicating already-completed tool
  calls on later empty chunks.
- Anthropic-on-Vertex structured-output defaults now also match the audited AI SDK wrapper
  semantics more closely: JSON-schema requests default to the reserved `json` tool fallback on the
  Vertex wrapper path instead of drifting back to Anthropic's native model-family heuristic, and
  the stream converter now receives that same effective mode so request shaping and
  streamed/non-stream JSON extraction stay aligned across the new audited Vertex Anthropic model
  ids.
- OpenAI-compatible generic tool warnings can now defer selected provider-defined tool ids to
  provider-owned middleware, allowing Groq `browser_search` to emit AI SDK-style
  unsupported-model warnings instead of the older generic compat warning.
- Groq GPT-OSS chat models now receive the native `{ "type": "browser_search" }` wire tool when a
  request includes `groq.browser_search`, while unsupported models skip wire injection and return
  the same warning details as AI SDK.
- Groq typed chat option parity is now tighter around the audited `@ai-sdk/groq` surface:
  `service_tier` accepts `performance`, `reasoning_effort` accepts `low|medium|high`, and the
  built-in Groq `KIMI_K2_INSTRUCT` constant now uses
  `moonshotai/kimi-k2-instruct-0905` instead of the decommissioned model id.
- Groq provider-owned typed options now also expose the remaining audited `@ai-sdk/groq`
  language-model option fields (`parallelToolCalls`, `user`, `structuredOutputs`,
  `strictJsonSchema`) on the public Rust surface, serialize them in AI SDK-style camelCase
  `providerOptions.groq`, and still normalize them back to Groq wire fields before transport.
- Groq's built-in model catalog is now much closer to the audited `@ai-sdk/groq` package:
  missing current chat ids such as `gemma2-9b-it`, `llama-guard-3-8b`, `llama3-{8b,70b}-8192`,
  `qwen-qwq-32b`, `qwen-2.5-32b`, and `deepseek-r1-distill-qwen-32b` are restored, while obsolete
  system/vision/tool-use preview ids are removed from the public Groq catalog.
- Groq provider construction now also matches the audited `@ai-sdk/groq` settings contract more
  closely: the compat `groq` preset resolves `GROQ_API_KEY` by default, provider-owned
  `GroqConfig` now exposes `from_env()` / `with_api_key(...)`, and `GroqBuilder` now accepts the
  AI SDK-style `headers(...)` alias in addition to the existing Rust-native HTTP configuration
  helpers.
- OpenAI-compatible provider option lookup now consistently accepts both raw provider ids and
  canonical AI SDK package ids across chat/completion/image request shaping. Alias pairs such as
  `together` / `togetherai` and `moonshot` / `moonshotai` no longer diverge by capability path
  when merging `providerOptions`.
- The public xAI package surface is now closer to the audited `@ai-sdk/xai` index:
  `provider_ext::xai::{XaiErrorData, XaiVideoModelId}` are now available on the stable facade,
  `provider_ext::xai::options::*` now also exports `XaiFilesOptions`, and the provider-owned
  `XaiClient` now exposes file management so `siumai::files::upload(...)` can execute through the
  xAI wrapper path with typed multipart lowering for `teamId` (plus provider-native
  `filePath -> file_path` support on the Rust upload lane).
- Amazon Bedrock embedding support is now closer to the audited
  `@ai-sdk/amazon-bedrock` package/runtime contract: `provider_ext::bedrock::{options::*, *}`
  now also exposes `AmazonBedrockEmbeddingModelOptions` plus typed Bedrock embedding helpers,
  `BedrockClient` now implements provider-owned embedding over the real `/model/{id}/invoke`
  route for Titan/Nova/Cohere model families, and builder/config-first/registry/public paths now
  all agree on the same request shape instead of failing before transport.
- Provider-crate feature-gated regression coverage now compiles against the current shared
  request/response structs across the audited provider set: xAI file/video tests were updated for
  optional upload filenames and `XaiVideoRequestExt`, MiniMaxi tests now distinguish video
  `prompt: Option<String>` from music `prompt: String`, and Azure native completion metadata now
  populates the shared `ResponseMetadata.headers` field.
- Protocol and top-level `all-features` regression coverage now also compile against the same
  shared contracts: Anthropic protocol streaming fixtures plus top-level
  experimental-bridge/transcoding tests now instantiate `ResponseMetadata.headers`, and the
  remaining MiniMaxi/public retry upload tests now follow the shared optional-filename file
  surface.

- Anthropic public streaming roundtrip coverage now explicitly guards GitHub issue `#17`:
  `siumai::protocol::anthropic::streaming::AnthropicEventConverter` keeps
  `cache_read_input_tokens`, `cache_creation_input_tokens`, `service_tier`, and
  `server_tool_use` intact across the public encode/decode path, and the fix is documented under
  `docs/workstreams/ai-sdk-structural-alignment/anthropic-extended-usage-roundtrip.md`.
- Anthropic Messages request fixtures now pin AI SDK-style message/part request option handling
  more tightly: message-level `providerOptions.anthropic.cacheControl` lowering and part-level
  document `providerOptions.anthropic.{citations,title,context}` are now fixture-backed on the
  request boundary, and the Anthropic fixture baselines were refreshed so
  `usage.raw.cache_creation` stays preserved where the current stable parser already returns it.
- OpenAI-compatible chat response parity is now tighter around Gemini thought signatures:
  direct response fixtures pin finalized tool-call
  `extra_content.google.thought_signature -> providerMetadata.{provider}.thoughtSignature`, and a
  no-network OpenRouter public-path regression test now locks the same metadata through
  `Siumai` / provider / config / registry entrypoints.
- Experimental stream bridge remapping is now stable-part aware: the upgraded
  `LanguageModelV3StreamPart` overlay exposes public runtime adapters in both directions
  (`from_runtime_part` and `to_runtime_part`), bridge primitive remappers now rewrite tool
  ids/names on direct `Part` / `PartWithReplay` events instead of only legacy `ToolCallDelta`,
  and stale OpenAI Responses `rawItem` replay payloads are dropped when a semantic remap would
  otherwise leave replay metadata inconsistent with the stable tool part.
- OpenAI Responses SSE conversion now emits more AI SDK-stable parts directly: `response.custom_tool_call_input.*`
  now becomes stable `tool-input-*`, web-search citations emit stable `source`, and buffered
  `response.failed` termination emits stable `finish` instead of provider-scoped custom events.
- Public OpenAI Responses, Anthropic, and Gemini streaming extension helpers now read stable
  `Part` / `PartWithReplay` events first and only fall back to legacy custom-event shadows, so
  provider-specific helper APIs stay aligned with the upgraded runtime stream contract.
- The typed stream-part downgrade boundary is now explicit: serializers call
  `LanguageModelV3StreamPart::to_protocol_custom_event(...)` as the canonical provider-wire
  lowering hook, while `to_custom_event(...)` remains only as a thin compatibility alias.
- Axum SSE stable-part export is now an explicit contract too: `event: part` always carries a
  `{ part, replay }` JSON envelope, with `replay: null` for plain runtime parts and populated
  replay hints only on `PartWithReplay`.
- Extras tool-loop gateway streaming now also injects stable `Part(ToolResult)` events for
  locally executed tools before the legacy `gateway:tool-result` compatibility custom event, so
  downstream protocol serializers can stay on the semantic part lane.
- Extras Axum plain-text streaming now also reads stable `Part(TextDelta)` /
  `PartWithReplay(TextDelta)` events, so semantic-only streams no longer lose text by depending
  on legacy `ContentDelta` shadows.
- Extras `stream_object`, tool-loop assistant-history accumulation, and streamed orchestrator
  fallback now also consume stable `TextDelta` parts directly, so semantic-only streams no longer
  lose structured-output text, follow-up assistant history, or fallback `StepResult.text()`.
- Shared stream wrappers now also treat stable `Part(TextDelta)` / `PartWithReplay(TextDelta)` as
  existing text when deciding whether to synthesize fallback legacy deltas, so semantic-only
  streams no longer get duplicate tail text from `StreamFactory` or `SimulateStreamingMiddleware`.
- Gemini streaming no longer emits parser-side `gemini:reasoning` custom shadows on top of stable
  `reasoning-*` parts, reducing runtime dual-lane duplication while keeping serializer-side custom
  compatibility handling intact.
- Gemini GenerateContent streaming now aligns more closely with the audited `@ai-sdk/google`
  `doStream()` contract on the runtime lane: parser-side output now includes stable
  `stream-start`, non-tool `text-*`, `file` / `reasoning-file`, successful `finish`, request-opt-in
  `raw`, and stable `error` parts for top-level `{"error": ...}` payloads plus invalid JSON
  chunks; request-aware Gemini/Vertex stream transformers now forward
  `ChatRequest.stream_options.include_raw_chunks`; same-protocol Gemini SSE replay now also lowers
  direct stable `Error` parts back into provider error envelopes; and EOF fallback closes active
  text/reasoning lanes before emitting `finish(unknown)` + `StreamEnd` when Google omits a
  terminal finish reason.
- Mixed stable/legacy Gemini text and reasoning streams are now deduplicated by first source in
  `StreamProcessor`, so stable `TextDelta` / `ReasoningDelta` plus compatibility shadows no longer
  double-count final text or thinking on direct runtime consumers.
- The shared OpenAI Responses bridge now upgrades only stable-shape legacy custom payloads and no
  longer keeps bespoke `gemini:*` / `anthropic:*` event-type special cases from the removed parser
  shadow era.
- DeepSeek provider-owned streaming now matches the audited AI SDK `@ai-sdk/deepseek`
  contract across unified/provider/config/registry/public entrypoints:
  non-text capabilities stay intentionally unavailable on the native DeepSeek package surface, and
  provider-owned chat streams now always emit `stream_options.include_usage = true` on the wire
  instead of diverging between `Siumai::builder()` and `Provider::deepseek()` / config-first
  clients.
- JSON stream end-event synthesis is now deferred until the upstream transport actually reaches
  EOF, so stateful converters keep their accumulated terminal response content on clean shutdown.
  This fixes Bedrock reserved-JSON structured-output extraction when the stream ends without an
  explicit terminal event and keeps `StreamEnd.response` aligned with the text deltas already seen
  on the stream.
- The shared OpenAI-compatible Fireworks preset now keeps the audited AI SDK completion boundary on
  the config/registry path as well: built-in compat metadata advertises `completion`, generic
  config-first Fireworks clients no longer diverge from `Siumai::builder().openai().fireworks()`,
  and public parity guards now pin the real `/completions` route across siumai/provider/config/
  registry entrypoints.
- The shared OpenAI-compatible TogetherAI/Together and DeepInfra presets now also advertise
  `completion` explicitly in static provider metadata, so raw compat preset inspection matches the
  audited AI SDK package boundary instead of relying on inferred completion support alone.
- The public compat package facades now also expose provider-scoped `Client/Config` aliases for
  the audited AI SDK-style wrappers `mistral`, `perplexity`, `fireworks`, and `deepinfra`, making
  the Rust package boundary easier to compare against per-provider package exports. TypeScript-only
  package exports such as `*ProviderSettings` and per-package `VERSION` remain intentionally
  deferred on these compat-wrapped facades because Rust already uses `Config` / builder surfaces
  as the stable provider-settings contract.
- MoonshotAI OpenAI-compatible alignment now follows the audited AI SDK package contract more
  closely: canonical public/runtime id is `moonshotai`, the historical `moonshot` id is kept only
  as a hidden migration alias, built-in examples/docs now use `moonshotai` / `.moonshotai()`,
  request normalization now maps `thinking.budgetTokens` and `reasoningHistory` onto Moonshot's
  wire shape, and curated/default model ids now match the audited Kimi subset more closely while
  completion/image/embedding stay intentionally unsupported on the wrapper boundary.
- The public Fireworks typed option surface is now closer to the audited AI SDK package exports:
  `FireworksEmbeddingModelOptions` is available as the explicit empty embedding option object, and
  the upstream deprecated aliases `FireworksProviderOptions` plus
  `FireworksEmbeddingProviderOptions` now exist on the Rust facade for migration coverage.
- Anthropic-on-Vertex structured-output and reasoning streams now preserve both stable runtime-part
  semantics and the older public textual delta contract: indexed Anthropic `text` / `thinking`
  blocks replay compatible `ContentDelta` / `ThinkingDelta` shadows again, and metadata-only
  redacted thinking placeholders no longer surface as empty reasoning strings through the shared
  response/message helpers.
- Anthropic AI SDK alignment now covers the latest audited provider-defined tool versions and 2026
  web-tool injection semantics: shared/provider surfaces include `web_search_20260209`,
  `web_fetch_20260209`, `code_execution_20260120`, and `computer_20251124`, request headers now
  inject the same beta tokens as `repo-ref/ai`, and implicit provider-executed `code_execution`
  calls are marked `dynamic` on the same conditions as the upstream Anthropic package.
- Anthropic Messages reverse request bridging now restores AI SDK-shaped request data more
  faithfully: request-level `providerOptions.anthropic` once again recovers
  `thinking.budgetTokens`, `cacheControl`, `metadata.userId`, `mcpServers`, `container`,
  `contextManagement`, and `speed`, and provider-defined tool args such as `maxUses` /
  `userLocation` no longer stay in raw wire snake_case after normalization.
- Extras orchestrator tool-approval continuity now follows the audited AI SDK flow more closely:
  first-turn `tool-approval-response` messages are collected only from the last `role=tool`
  message, approved local tools execute before the next model call, denied approvals synthesize
  `execution-denied`, provider-executed approvals are forwarded back into the next prompt, and
  denied provider approvals now also carry `output.providerOptions.openai.approvalId` for
  correlation.
- Shared response-side `providerMetadata` now uses one AI SDK-style provider-rooted map across the
  public Rust surface: chat/completion responses, content parts, stable stream parts, file-upload
  results, and skill-upload results all converge on `provider_id -> object` semantics instead of
  ad hoc nested map layouts. UI `providerMetadata` intentionally remains on the request-side
  `providerOptions` story because upstream `convertToModelMessages()` treats it that way.
- Stable tool-result content now emits the canonical AI SDK names for provider-owned file/image
  references: `file-reference` / `image-file-reference` with `providerReference`. OpenAI tool
  message normalization and tool-message JSON-string preservation were tightened around that
  canonical shape while keeping legacy `file-id` / `image-file-id` inputs as compatibility aliases.
- Stable UI-message validation is stricter on the AI SDK-aligned tool state machine:
  `validate_ui_messages()` now rejects invalid `UiToolPart` combinations for `approval`,
  `output`, `errorText`, `rawInput`, `resultProviderMetadata`, and `preliminary` based on the
  selected tool state, while preserving the existing wide-struct serde/public compatibility story.
- Stable UI tool parts now also expose a typed AI SDK-style state-discriminated overlay:
  `UiToolInvocation` / `UiToolInvocationState` model the valid tool lifecycle payloads directly,
  and `UiToolPart::invocation()` / `UiToolPart::from_invocation(...)` bridge that typed view with
  the existing serde-compatible `UiToolPart`.
- Stable UI-message validation now also has a schema-aware lane closer to AI SDK
  `validateUIMessages(...)`: `validate_ui_messages_with_schemas(...)` can validate message
  metadata, `data-*` parts, and static tool input/output payloads against caller-supplied schemas
  without forcing `siumai-core` itself to depend on a concrete JSON Schema engine.
- Stable UI-message conversion now also matches the audited AI SDK tool-error split:
  provider-executed `output-error` parts become `error-json`, while local tool-message
  `output-error` parts stay on `error-text`.
- Stable UI-message conversion now also has a runtime tool-output mapping lane closer to AI SDK
  `convertToModelMessages({ tools })`: `ExecutableTool` / `ExecutableTools` can carry
  `to_model_output` mappers, and `convert_to_model_messages_with_tooling` /
  `convert_to_chat_request_with_tooling` apply them on `output-available` tool results.
- Shared function-tool structures are closer to AI SDK `Tool.inputSchema/outputSchema` now:
  `ToolFunction` now serializes the stable portable shape with `inputSchema`, still accepts legacy
  `parameters` / `input_schema` on input, stores optional `outputSchema` metadata, `Tool` /
  `ExecutableTool` expose schema-oriented helpers, and OpenAI Responses request normalization
  preserves AI SDK-style `inputSchema` / `outputSchema` / `inputExamples` fields without changing
  existing provider wire shaping.
- Shared provider-defined tool structures now also carry optional AI SDK-style
  `supportsDeferredResults`, and the audited Anthropic deferred-result tool factories mark that
  metadata explicitly on the stable portable tool shape.
- High-level deferred provider-tool behavior is now aligned on the audited orchestration paths:
  `siumai-extras` orchestrator and gateway tool-loop flows keep pending
  provider-executed `supportsDeferredResults` calls alive across steps, accept later provider
  `tool-result` turns without a same-step tool-call, and surface those response-native results
  through `StepResult.tool_results` / `tool_result_views()`.
- Local AI SDK-style tool runtime semantics are now much closer on the audited extras path:
  `ExecutableTool` carries runtime-only `dynamic`, `contextSchema`, `needsApproval`, and
  `onInputStart` / `onInputDelta` / `onInputAvailable` metadata, `ToolResolver` can expose that
  metadata without changing existing execution signatures, non-stream and stream orchestrators now
  surface local `tool-approval-request` parts when runtime approval is required without the legacy
  Rust callback, streamed/local input lifecycle callbacks are invoked on the audited local-tool
  loop, runtime-dynamic tool flags now propagate into `StepResult.dynamic_tool_calls()` /
  `dynamic_tool_results()`, and provider-executed deferred tools stay excluded from local
  execution.
- `ChatResponse` now exposes `tool_results()` / `has_tool_results()` so higher-level runtime code
  can consume provider-native response `tool-result` parts directly instead of only looking at
  locally executed tool messages.
- Provider catalog/model-surface drift is tighter across focused providers: DeepSeek, Ollama, and
  MiniMaxi catalog output now reuses provider-owned curated model sources instead of handwritten
  arrays, and MiniMaxi stable stream finish-part metadata now also rekeys to
  `provider_metadata["minimaxi"]` instead of leaking the borrowed Anthropic namespace.
- The structural-alignment cleanup also removed the remaining nested response-metadata fixtures and
  shims around that contract: `siumai-extras::StepResult`, OpenAI/OpenAI-compatible/Anthropic
  bridge tests, OpenAI Responses round-trip fixtures, Anthropic thinking helpers, and the
  gateway loss-policy example now all use the same provider-rooted `providerMetadata` object
  semantics.
- The wider provider/helper surface now also matches that provider-rooted metadata contract:
  Gemini/Vertex, Azure completion, Bedrock, Ollama, Anthropic skills/streaming, DeepSeek, Groq,
  xAI, and MiniMaxi typed metadata helpers no longer assume nested `HashMap<String, HashMap<...>>`
  roots, and the audited `cargo check -p siumai --all-features` lane is green again after the
  response-side metadata cleanup.
- Anthropic/DeepSeek/Gemini/OpenAI helper alignment now follows the audited AI SDK provider-root
  rules more closely as well: Anthropic and DeepSeek custom provider ids read request options from
  the runtime namespace and emit response metadata under that same resolved root, Gemini now uses
  runtime `google|vertex` precedence for request options and `thoughtSignature` replay, and
  OpenAI typed metadata helpers now expose keyed accessors for explicit provider-root reads.
- The built-in Perplexity compat preset now also exposes AI SDK-shaped typed provider metadata on
  the Rust surface: `providerMetadata.perplexity` now uses canonical `images.imageUrl|originUrl`,
  `usage.citationTokens|numSearchQueries`, and `cost.*` fields instead of forwarding raw
  snake_case `usage/images` fragments directly.
- The public Perplexity typed option surface now stays package-level instead of leaking the wire
  contract: `PerplexityOptions` / `PerplexityWebSearchOptions` serialize with AI SDK-style
  camelCase on the Rust facade, legacy snake_case input remains accepted as a compatibility alias,
  and the shared compat request boundary explicitly lowers known fields onto Perplexity's
  snake_case wire payload.
- The dedicated Mistral wrapper boundary is now documented explicitly as well: Siumai keeps the
  audited `@ai-sdk/mistral` split of `chat + embedding`, preserves camelCase public typed
  language-model options with explicit wire lowering in the compat layer, and continues to reject
  completion/image on that wrapper path.
- The provider-owned Groq wrapper now preserves JSON Schema structured outputs on the
  config-first stream/chat path by opting the compat runtime into structured-output support
  instead of silently downgrading `response_format` back to `{ "type": "json_object" }`.
- Native OpenAI speech/transcription typed provider options now drive real request behavior more
  closely: TTS provider options accept `speed`, transcription provider options accept
  `language` / `timestampGranularities`, and the OpenAI audio path now accepts both camelCase and
  snake_case option keys instead of requiring one fragile internal JSON spelling.
- Shared structure follow-up fixes now also keep adjacent AI SDK-aligned types consistent on the
  provider path: Ollama/Cohere follow the newer `FilePartSource` split, Anthropic JSON response
  shaping now handles `MessageContent::Json`, and Google/Gemini streaming fixture coverage now
  accepts the stable runtime `Part` channel instead of assuming only legacy custom events.
- Stream metadata parity is now tighter across mixed compatibility lanes:
  shared textual shadow replay no longer duplicates legacy deltas when a converter already emits
  them, Perplexity typed hosted-search metadata now includes `usage.reasoningTokens`, and
  Gemini/Vertex reasoning streams now expose AI SDK-style `reasoning-*` custom events while
  Gemini GenerateContent bridge round-trips suppress duplicate reasoning deltas from mixed
  `Part + Custom` input.
- Gemini protocol prompt/response conversion now consistently honors the newer `FilePartSource`
  split after the provider-reference refactor: image/file branches no longer mix `MediaSource`
  with `FilePartSource`, and Gemini feature builds are green again on the audited
  multi-feature lane.
- Stable tool-result content parsing now accepts the newer AI SDK provider-reference aliases
  `file-reference` / `image-file-reference` plus `providerReference` payload keys alongside the
  older `file-id` / `image-file-id` compatibility shape.
- OpenAI/Anthropic request bridging now treats provider-owned file ids as canonical stable
  `providerReference` values instead of re-encoding them through legacy base64/file-prefix
  compatibility lanes, and the AI SDK structural-alignment workstream docs now mark prompt-side
  provider references as complete.
- Injectable HTTP transport parity now also covers non-stream `GET` / binary `GET` execution:
  `execute_get_request` and `execute_get_binary` no longer bypass custom transports, so
  provider-owned resource helpers such as Anthropic skill-version metadata fetches inherit the
  same custom fetch/retry semantics as POST and multipart paths.
- Extras orchestrator high-level result/control flow is now closer to AI SDK:
  `StepResult` and `OrchestratorFinishEvent` carry a stable open-JSON `context`,
  `prepare_step` can now read/update that context on both non-stream and stream paths,
  context-aware tool execution hooks were added with backward-compatible `ToolResolver`
  defaults, and the first streamed step now also honors `prepare_step`, `tool_choice`, and
  `active_tools` instead of only later follow-up steps. `prepare_step` can now also swap the
  `LanguageModel` for an individual step, `StepResult.text()` now concatenates all generated
  text parts instead of returning only the first one, and standardized extras projections now
  expose static/dynamic tool-call and tool-result views with resolved tool inputs.
- Anthropic request-side provider options now track the audited AI SDK surface much more closely:
  typed `thinking` (`adaptive | enabled | disabled`), `sendReasoning`,
  `disableParallelToolUse`, `cacheControl`, `metadata.userId`, `mcpServers`,
  `contextManagement`, `toolStreaming`, `effort`, `speed`, `anthropicBeta`, and typed
  `container.skills` now flow through builder/config/public-facade defaults without raw JSON
  shims, and the plain `anthropic-standard` protocol build no longer assumes
  `structured-messages` just to compile JSON replay code paths.
- Anthropic enabled-thinking request shaping now also matches the upstream semantic rule more
  closely: final request bodies consistently add `thinkingBudget` onto `max_tokens` before
  model-cap capping, including the older legacy-specific Anthropic thinking path that previously
  skipped that adjustment.
- TogetherAI now follows the AI SDK single-provider story on the public Siumai surface more
  closely: canonical `togetherai` is a unified provider id for
  chat/completion/embedding/speech/transcription plus provider-owned image and native rerank,
  `Provider::togetherai()` now resolves through the normal unified builder path, the default text
  model is the chat default instead of the rerank default, the unified image lane now uses
  TogetherAI's provider-owned `/images/generations` contract for both generation and edit
  (`image_url` edits, no mask-based edits), public typed `TogetherAiImageOptions` mirrors the
  AI SDK image-option lane, the lower-level explicit compat escape hatch is now
  `Provider::openai().togetherai_openai_compatible()`, and the older public `together` builder
  alias is retired.
- TogetherAI's explicit compat escape hatch now also stays conflict-free on the unified builder
  surface: `Siumai::builder().openai().togetherai_openai_compatible()` no longer trips native
  TogetherAI feature gates in `openai`-only builds because the public shortcut now routes through
  the hidden compat alias internally while keeping `togetherai` as the native provider id.
- TogetherAI catalog/default-model data now reflects the full unified provider story instead of a
  chat+rerank fragment: the registry catalog lists embedding/image/speech/transcription/rerank
  family defaults for canonical `togetherai`, and the shared compat default-model helpers now
  expose public speech/transcription getters for unified-provider consumers.
- TogetherAI's public typed surface now mirrors the audited AI SDK package more closely as well:
  `provider_ext::togetherai` exposes curated `chat/completion/embedding/image/rerank` model
  constants plus AI SDK-style `TogetherAIImageModelOptions` /
  `TogetherAIRerankingModelOptions` aliases, with deprecated compatibility aliases kept for
  migration-side audits and the older Rust `TogetherAi*` names preserved as compatibility aliases.
- TogetherAI provider-catalog output now also includes the audited curated
  chat/completion/embedding/image/rerank model subset instead of listing only family defaults,
  while still keeping speech/transcription defaults visible for the extra unified families that
  exist on the Rust side.
- DeepInfra now also follows the AI SDK single-provider story on the public Siumai surface:
  canonical `deepinfra` is a first-class built-in provider id, `Provider::deepinfra()` /
  `Siumai::builder().deepinfra()` build a unified client, chat/completion/embedding reuse the
  shared OpenAI-compatible runtime, image generation/edit use provider-owned
  `/inference/{model}` and `/openai/images/edits`, and DeepInfra-specific reasoning/completion
  usage totals are normalized before entering the stable `Usage` layer.
- The public DeepInfra surface now also exposes curated
  `provider_ext::deepinfra::{chat, completion, embedding, image, model_sets}` constants, and the
  provider catalog reuses that same audited DeepInfra subset instead of listing only defaults.
- Vertex MaaS now also follows the AI SDK single-provider story on the public Siumai surface:
  canonical `vertex-maas` is a first-class built-in provider id, `Provider::vertex_maas()` /
  `Siumai::builder().vertex_maas()` build a unified client, chat/completion/embedding reuse the
  shared OpenAI-compatible runtime on the Vertex `/endpoints/openapi` base URL derived from
  `project + location`, and Google Bearer auth now works through token providers or explicit
  `Authorization` headers without requiring a fake non-empty API key.
- The public Vertex MaaS surface now also exposes curated
  `provider_ext::vertex_maas::{chat, completion, embedding, model_sets}` constants, and the
  provider catalog reuses that same audited MaaS subset instead of hardcoding a second copy of the
  model list.
- Cohere now also follows the AI SDK single-provider story on the public Siumai surface:
  canonical `cohere` is a native unified `/v2` provider id for chat/embedding/rerank,
  `Provider::cohere()` / `Siumai::builder().cohere()` build one native client, public/registry
  metadata and catalog now advertise the unified capability set, the old rerank-only native story
  is removed, and the canonical native path now requires explicit model ids instead of inheriting
  a rerank-biased provider default.
- The public Cohere typed surface now also mirrors the audited `@ai-sdk/cohere` package more
  closely: `provider_ext::cohere` exposes curated `chat` / `embedding` / `rerank` model-constant
  modules plus AI SDK-style `CohereLanguageModelOptions`, `CohereEmbeddingModelOptions`, and
  `CohereRerankingModelOptions` aliases, while deprecated migration aliases remain available for
  side-by-side checks.
- Native Cohere chat now also mirrors the audited AI SDK request/stream warning behavior more
  closely: provider-defined tools are filtered from `/v2/chat` requests but still surface as
  stable `unsupported { feature: "provider-defined tool <id>" }` warnings, and the Cohere stream
  lane now emits a stable `StreamStart` part plus runtime `raw` chunks when
  `includeRawChunks` is enabled.
- Fireworks now also follows the AI SDK single-provider story on the public Siumai surface:
  canonical `fireworks` is a first-class built-in provider id, `Provider::fireworks()` /
  `Siumai::builder().fireworks()` build a unified client, chat/completion/embedding/transcription
  keep reusing the shared OpenAI-compatible runtime, image generation/edit now route through
  provider-owned Fireworks workflow and `image_generation` endpoints, default image/edit fallback
  models are aligned with the audited AI SDK package, and the compat chat path now normalizes the
  Fireworks-specific request-shaping quirks for `thinking.budgetTokens`, `reasoningHistory`, and
  Fireworks reasoning-effort levels. The public typed request surface now also exposes
  `provider_ext::fireworks::{FireworksChatOptions, FireworksLanguageModelOptions,
  FireworksChatRequestExt}` so the audited AI SDK Fireworks language-model option shape no longer
  needs raw `providerOptions.fireworks` JSON. The audited AI SDK Fireworks curated
  chat/completion/embedding/image model subsets are now also promoted into public Rust constants
  and reused by the provider catalog instead of being duplicated as ad hoc string lists.
- Mistral and Perplexity now also follow the audited AI SDK package-owned surface more closely on
  the public compat path: `provider_ext::mistral::{MistralChatOptions,
  MistralLanguageModelOptions, MistralChatRequestExt}` now expose typed Mistral chat options
  instead of raw `providerOptions.mistral` JSON, Mistral request shaping now normalizes
  `safePrompt`, document limits, `structuredOutputs`, `parallelToolCalls`, and
  `reasoningEffort` onto the wire contract with the same default structured-output posture as the
  AI SDK package, and provider-owned curated Mistral/Perplexity model constants now back compat
  defaults plus the public catalog so stale Perplexity legacy ids no longer leak as canonical
- Native Bedrock request prompt conversion now also mirrors the audited AI SDK converter more
  closely: message-level `cachePoint` survives on system/user/tool/assistant blocks, user
  `file` parts map to Bedrock `document` / `image` blocks with typed Bedrock citations and
  upstream-style filename stripping, assistant reasoning replays `signature` / `redactedData`
  from canonical request-side `providerOptions`, tool-result `content` now supports `text` plus
  `image-data`, and request-side Mistral tool ids normalize on both tool calls and tool results.
- Native Bedrock now also exposes a typed public replay path for reasoning metadata:
  `provider_ext::bedrock::BedrockContentPartExt` reads typed `signature` / `redactedData` from
  response reasoning parts, and `assistant_message_with_reasoning_metadata(...)` converts those
  fields back into replayable request-side `providerOptions.bedrock`.
  defaults.
- Shared image execution now materializes `data:` / `http(s):` URL-backed edit and variation
  inputs at the executor boundary before synchronous OpenAI/OpenAI-compatible/Vertex
  multipart/inline transformers run, and public-path parity coverage now locks that behavior across
  facade/provider/config/registry surfaces.
- Vertex Imagen variation execution now follows the shared image-variation surface end-to-end:
  the native Vertex client advertises variation support again, variation requests now shape into
  the Imagen `:predict` contract instead of failing as unsupported, and parity coverage now locks
  builder/config/registry/public-path behavior for data-url-backed variation inputs.
- OpenAI-compatible streaming EOF fallback now keeps the stable AI SDK-style semantic suffix on the
  direct `Part` lane: when a stream ends without an explicit terminal `finish_reason`, active
  text/reasoning parts are closed, unfinished tool-call lifecycles are finalized, and a stable
  `finish` part is emitted before the legacy `StreamEnd`.
- OpenAI-compatible and Anthropic chat streaming now also honor AI SDK `includeRawChunks` on the
  main audited text/chat path: runtime `raw` parts are emitted on every parsed chunk, and the
  first chunk keeps the upstream ordering `stream-start -> raw -> response-metadata -> ...`.
- Native Bedrock Converse JSON streaming now also aligns its first audited stream preamble more
  closely with the AI SDK Bedrock provider: the first parsed chunk emits `stream-start`, stable
  `stream-start`, stable `response-metadata`, runtime-opt-in `raw`, then Bedrock content/tool
  deltas; Bedrock provider error envelopes now surface stable `error` parts on that same lane; and
  terminal streamed `ChatResponse` values now preserve the default model, request warnings,
  `raw_finish_reason`, and Bedrock `stopSequence`.
- Native Bedrock chat/stream response shaping now also aligns much more closely with the audited
  AI SDK provider payloads: Converse JSON streaming emits stable `text-*`, `reasoning-*`,
  `tool-input-*`, `tool-call`, and terminal `finish` parts while keeping legacy shadow events;
  finish parts now preserve Bedrock usage/provider metadata such as `trace`, `performanceConfig`,
  `serviceTier`, `cacheWriteInputTokens`, `cacheDetails`, `stopSequence`, and
  `isJsonResponseFromTool`; non-stream chat responses now retain reasoning provider metadata,
  default-model identity, request warnings, and Mistral tool-call-id normalization.
- Native Bedrock request shaping now also aligns much more closely with the audited AI SDK
  provider options surface: typed `provider_ext::bedrock::{BedrockReasoningConfig,
  BedrockReasoningEffort, BedrockReasoningType, BedrockServiceTier}` now accompany
  `BedrockChatOptions`, unknown top-level `providerOptions.bedrock` passthrough fields are
  preserved on the request body, Anthropic Bedrock requests now derive
  `additionalModelResponseFieldPaths`, `thinking`, `anthropic_beta`, and top-level
  `serviceTier`, `maxReasoningEffort` now maps onto the same provider-specific Bedrock fields as
  upstream (`output_config.effort`, `reasoning_effort`, or nested `reasoningConfig`), and
  Anthropic structured JSON output now uses native `additionalModelRequestFields.output_config`
  when the audited Bedrock/Anthropic route supports it or Bedrock thinking is enabled.
- OpenAI-compatible, Anthropic, Gemini, native OpenAI completion, native Cohere, native Bedrock,
  and Ollama streaming now also preserve first-chunk parse-failure lifecycles more consistently:
  OpenAI-compatible, Anthropic, native OpenAI completion, and native Cohere emit `stream-start`
  before optional runtime `raw` and the parse error, native Bedrock now keeps its first-chunk
  preamble before optional runtime `raw` and the parse error, and Gemini plus Ollama now emit
  `stream-start` before the parse error instead of skipping the stream lifecycle start entirely.
- Anthropic Messages SSE now also aligns one more rare AI SDK stream edge: provider-specific
  `compaction` / `compaction_delta` blocks map onto stable `text-*` parts with
  `providerMetadata.anthropic.type = "compaction"`, aggregate into final stream text, and
  same-protocol Anthropic replay preserves the `compaction` block type instead of degrading it to
  plain text.
- Anthropic Messages SSE now also matches the audited deferred tool-call path more closely:
  preloaded `message_start.message.content[*].tool_use` blocks emit the stable
  `tool-input-start -> tool-input-delta -> tool-input-end -> tool-call` lifecycle, and Anthropic
  tool-call `caller` metadata is preserved under `providerMetadata.anthropic.caller`.
- Anthropic same-protocol streaming replay now also preserves tool-call `caller` metadata when
  stable `ToolCall` parts are serialized back into Anthropic SSE `tool_use` blocks, instead of
  dropping those provider-specific replay hints on the stream bridge path.
- Anthropic Messages SSE finish metadata now also matches the audited container update semantics:
  non-terminal `message_delta.container` updates survive through `message_stop`, and later
  `message_delta` frames without `container` clear older container state instead of leaking stale
  message-start metadata into final `finish` / `StreamEnd` provider metadata.
- Anthropic provider metadata now also exposes the remaining audited message-level fields more
  faithfully: typed Anthropic responses surface `stopSequence` and `iterations`, non-stream
  Anthropic responses map `usage.iterations` onto top-level AI SDK-style provider metadata, and
  streaming `message_delta.stop_sequence` now follows the same latest-delta-wins finish metadata
  semantics as upstream `repo-ref/ai`.
- Anthropic response-side `contextManagement` metadata is now surfaced as a typed
  `appliedEdits` union on `AnthropicMetadata`, and the raw-to-camelCase response mapping now also
  preserves the audited `compact_20260112` edit branch instead of silently dropping compaction
  edits on that response path.
- Anthropic custom provider ids now follow the audited AI SDK request/response contract more
  closely: request shaping merges canonical `providerOptions.anthropic` with provider-owned custom
  keys, custom keys override canonical fields when both are present, typed Anthropic metadata
  accessors now support custom roots, and top-level non-stream / finish / stream-end
  `providerMetadata` duplicates onto the custom provider root only when that custom request key
  was actually used.
- Anthropic response content and provider metadata now also match the audited AI SDK source/null
  shape more closely: non-stream text citations and `web_search_tool_result` blocks emit stable
  `source` parts, request-scoped citation documents now feed the non-stream response transformer
  for document citation `title` / `filename` / `mediaType` resolution, Anthropic source ids now
  use stable `id-*` generation across stream and non-stream source paths, and absent
  `container` / `contextManagement` metadata now stays observable as explicit `null` on
  non-stream and final stream-end Anthropic message metadata instead of silently disappearing.
- OpenAI-compatible, native OpenAI, and native Azure completion streaming now also honor AI SDK
  `includeRawChunks` on the audited `/completions` path: stable `stream-start`, `raw`,
  `response-metadata`, `text-*`, and terminal `finish` parts are emitted while preserving legacy
  `ContentDelta` / `StreamEnd` compatibility.
- `siumai::prelude::unified::*` now also re-exports `CompletionCapability`, and audited
  completion public-path parity fixtures now pin native `openai` / `azure` plus first-class
  wrapper `togetherai` / `deepinfra` / `vertex-maas` / `fireworks` generate+stream routes so
  runtime-only `includeRawChunks` never leaks onto the provider wire payload.
- The stable runtime stream-request structure is now exported on the normal public facade too:
  `StreamRequestOptions` is available from `siumai::prelude::unified::*`,
  `siumai::completion::*`, and `siumai::text::*`, and the public compile surface now explicitly
  locks both `CompletionCapability` and request-level `includeRawChunks` construction.
- Lower-contract OpenAI-compatible URL audit coverage now also locks the canonical
  chat/embedding/completion endpoints for `togetherai`, `deepinfra`, and `vertex-maas`, and the
  Vertex MaaS audit now uses the real project/location-derived `/endpoints/openapi` base URL
  instead of the placeholder compat config URL.
- TogetherAI provider settings now also match the audited AI SDK env-compat contract more closely:
  canonical `togetherai` and the native rerank typed client prefer `TOGETHER_API_KEY` but still
  accept the deprecated `TOGETHER_AI_API_KEY` alias on config-first, builder, and registry paths.
- Native Cohere embedding execution now also matches the audited AI SDK runtime guards more
  closely: `outputDimension` is finite-valued and one `/v2/embed` call now fails fast if it carries
  more than `96` inputs.
- The broader Google Vertex compatibility builder now matches the AI SDK provider/model split more
  closely: `Siumai::builder().vertex()` and `Siumai::builder().anthropic_vertex()` no longer
  inject provider-wide default models and now require explicit model ids, aligning the unified
  facade with AI SDK's family-specific `languageModel()` / `embeddingModel()` / `imageModel()`
  construction pattern.
- Stable provider typing now also treats the broader Google Vertex wrapper family as first-class:
  `ProviderType::{Vertex, AnthropicVertex, VertexMaas}` now drive provider catalog, retry, and
  validator behavior instead of degrading `vertex` / `anthropic-vertex` into `Custom(...)`.
- The provider-owned Google Vertex facade now also exposes curated
  `provider_ext::google_vertex::{chat, embedding, image, model_sets}` plus
  `provider_ext::anthropic_vertex::{chat, model_sets}` constants, and the provider catalog reuses
  those same curated subsets instead of hardcoding a second copy of the `vertex` /
  `anthropic-vertex` model lists.
- Stable provider typing now also treats `azure`, `cohere`, `togetherai`, and `bedrock` as
  first-class: `ProviderType::{Azure, Cohere, TogetherAi, Bedrock}` now drive provider catalog,
  retry, validator, and client metadata behavior instead of degrading those built-in providers to
  `Custom(...)`.
- Stable provider typing now also treats the next AI SDK-packaged OpenAI-compatible providers as
  first-class: `ProviderType::{Mistral, Fireworks, Perplexity}` now drive provider catalog, retry,
  validator, and client metadata behavior instead of degrading those audited provider-package ids
  to `Custom(...)`, while their execution still reuses the shared OpenAI-compatible runtime.
- AI SDK-packaged OpenAI-compatible providers now also keep the correct public completion boundary:
  `mistral` and `perplexity` stay chat-only on the public/runtime capability surface, while
  `fireworks` keeps completion-family support.
- Canonical public-path guards now also lock that chat-only completion boundary explicitly:
  `Siumai::builder().mistral()/perplexity()`, `Provider::mistral()/perplexity()`, config-first
  compat clients, and registry `completion_model(...)` all keep completion disabled or rejected.
- OpenAI/Azure family routing variants now also collapse back to their canonical provider identity:
  `openai-chat`, `openai-responses`, and `azure-chat` no longer leak as fake custom providers in
  provider typing or provider-catalog lookups.
- OpenAI-compatible completion execution now uses the real `/completions` generate and SSE paths
  instead of chat-only shims, reuses the shared runtime stream lane, applies AI SDK-style prompt
  materialization and unsupported warnings, normalizes audited completion provider options, and
  preserves streaming `include_usage` / response metadata / provider metadata behavior on the new
  family surface.
- Native OpenAI and Azure completion execution now also uses the real `/completions` generate and
  SSE paths, registry-native OpenAI/Azure factories now advertise completion-family support, Azure
  keeps its deployment `api-version` URL semantics on the completion path, and compat/native
  completion responses now preserve raw completion `choices[0].logprobs` metadata instead of
  reusing chat-only extraction rules.
- Stable `Usage` now exposes AI SDK-style `inputTokens` / `outputTokens` / `raw`, and OpenAI-compatible, OpenAI Responses, Anthropic, and Gemini protocol paths now round-trip richer usage breakdowns instead of rebuilding provider-specific partial views.
- `Usage` now treats AI SDK-style usage as the canonical stable storage layer. Legacy `prompt/completion/total` counts remain available only through compatibility accessors/serde, and the public/examples/tests surface has been migrated off direct field access.
- The typed stable stream-part layer in `siumai-core` is now a V4-capable superset that includes first-class `custom` and `reasoning-file` parts, and OpenAI-compatible reserialization now degrades those unsupported V4-only parts into explicit text in `AsText` mode instead of silently dropping them.
- The upgraded typed stream-part overlay now also exposes public `LanguageModelV4*` aliases, so new code and docs can use AI SDK-aligned naming without depending on the historical `LanguageModelV3*` compatibility names.
- The runtime streaming contract now includes a first-class `ChatStreamEvent::Part(ChatStreamPart)` semantic channel plus a separate runtime replay carrier for protocol-only hints, and the main stream processor plus OpenAI/OpenAI-compatible/Anthropic/Gemini serializers now bridge that richer part model instead of forcing major V4 stream semantics through provider-scoped `Custom` payloads.
- OpenAI Responses and Anthropic SSE serializers now normalize runtime `ChatStreamEvent::Part(ChatStreamPart)` values before taking protocol serialization state locks, which fixes direct stable-part replay hangs caused by recursive lock re-entry.
- OpenAI Responses, Anthropic, Gemini, and OpenAI-compatible parser paths now consume that stable part channel directly for their main AI SDK-aligned stream semantics. OpenAI Responses provider-hosted tool / MCP / approval replay now rides the runtime replay carrier instead of loose `rawItem` / `outputIndex` custom payloads, and its SSE converter now also maps `response.custom_tool_call_input.*` to stable `tool-input-*`, web-search citations to stable `source`, and buffered `response.failed` termination to stable `finish`. Anthropic now emits runtime parts for `stream-start`, `response-metadata`, `text-*`, provider-hosted `server_tool_use` / MCP `tool-*`, standard local `tool-input-*` / `tool-call`, `reasoning-*`, `source`, and successful `finish` semantics, and OpenAI-compatible chat chunks now emit lifecycle parts for `stream-start`, `response-metadata`, `text-*`, `reasoning-*`, and `finish` while keeping legacy deltas in parallel for compatibility. Anthropic `signature_delta` and `redacted_thinking` now also stay on that stable lane through reasoning-part `providerMetadata`, matching AI SDK stream behavior instead of relying on provider-scoped custom events.
- OpenAI-compatible tool streaming now emits stable `tool-input-start` / `tool-input-delta` / `tool-input-end` / `tool-call` parts before the legacy shadow deltas, chat-completions reserialization now deduplicates mixed stable+legacy tool streams with first-source-wins semantics, and `StreamProcessor` now preserves final stable `tool-call` parts instead of dropping them during final response assembly.
- OpenAI-compatible chat responses now map URL citations from non-stream `message.annotations` and streaming `delta.annotations` into stable `source` parts, and stable URL `source` parts now round-trip back into chat-completions `annotations` during SSE reserialization.
- OpenAI-compatible exact alignment coverage now also pins those citation semantics on the public paths: non-stream chat-response fixtures lock `text -> tool-call -> source(url)` ordering for `message.annotations`, same-protocol chat-completions roundtrip tests lock `delta.annotations -> source(url) -> delta.annotations`, and the compat chat-response fixture suite now asserts the canonical AI SDK-style `Usage.inputTokens/outputTokens/raw` shape instead of only legacy totals.
- OpenAI-compatible same-protocol chat-completions roundtrip coverage now also preserves public-path `response-metadata`, terminal streamed `logprobs`, AI SDK-style `acceptedPredictionTokens` / `rejectedPredictionTokens` mirrored from `completion_tokens_details`, and terminal response-envelope `system_fingerprint` / `service_tier` fidelity. The bridge now prefers the richer `StreamEnd` envelope over earlier finish parts for chat-completions terminal chunks, and the terminal serializer maps stable finish-part/provider metadata logprobs back into the canonical chat-completions `choices[].logprobs.content` shape.
- Stable runtime stream types such as `ChatStreamPart`, `ChatStreamToolCall`, and related replay metadata are now re-exported through the normal streaming/prelude surface, and the current high-level/gateway code paths no longer need `__private::types` for the main stable stream contract.
- Anthropic streaming now preserves extended usage fields such as `cache_creation_input_tokens`, `cache_read_input_tokens`, `server_tool_use`, and `service_tier` across decode/encode round-trips, and terminal `Usage.raw` now keeps the full provider-native Anthropic usage object instead of a trimmed subset.
- Anthropic Messages response parsing now keeps the full provider-native `usage` object on both `Usage.raw` and `provider_metadata.anthropic.usage`, so AI SDK-style raw-usage snapshots preserve nested/forward-compatible Anthropic fields instead of only the audited subset.
- Stable `Usage.merge()` now follows AI SDK `addLanguageModelUsage()` semantics more closely: token totals/details are aggregated, but merged usages drop provider-native `raw` instead of recursively synthesizing a combined raw payload that has no stable cross-provider meaning.
- Extras orchestrator/agent `total_usage` aggregation now follows AI SDK `totalUsage` semantics more closely as well: step usage totals are summed across steps, but aggregated results never preserve per-step provider-native `Usage.raw`, including the single-step case.
- Extras orchestrator/agent/workflow completion surfaces now also align more closely with AI SDK
  `onFinish` / stream result semantics: `on_finish` receives an explicit finish event carrying the
  final response, last step, full `steps`, and aggregated `total_usage`, the extras orchestration
  family now binds `LanguageModel` instead of bare `ChatCapability` so step results can rely on
  stable model identity, `StepResult` now also carries stable `call_id`, stable
  `model { provider, model_id }`, explicit `step_number`, step-scoped `request` / `response`,
  telemetry `function_id` / `metadata`, and audited-provider-backed `raw_finish_reason`,
  `StreamOrchestration` now resolves `total_usage`, and the basic stream path no longer defaults
  to an empty `steps` list.
- Stable response/result finish metadata is now closer to AI SDK across the audited paths:
  `ChatResponse` exposes `raw_finish_reason`, shared OpenAI-compatible chat/stream decoding plus
  OpenAI Responses decoding propagate provider-native raw finish reasons where available, native
  Bedrock/Cohere chat paths preserve their raw stop reasons, audited OpenAI/OpenAI-compatible/Azure
  completion streams carry raw finish reasons into terminal `ChatResponse`, and extras
  `StepResult.raw_finish_reason` now forwards that stable response field instead of staying empty.
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
- OpenAI-compatible streaming now also falls back to the request model when those early Azure
  model-router placeholder chunks omit `model`, so stable `stream-start.model` is no longer lost
  on that audited compat path.
- OpenAI-compatible response metadata extraction now follows an AI SDK-style provider-owned policy instead of a shared compat-layer whitelist: `OpenAiStandardAdapter` / `ConfigurableAdapter` opt specific providers into `sources` / `logprobs` / prediction-token metadata, Perplexity keeps its hosted-search extras as a provider-specific special case, and generic OpenAI-compatible providers no longer infer those metadata fields by default.
- OpenAI-compatible chat request/response shaping now also matches the audited AI SDK raw/camelCase provider-key contract more closely: provider-owned passthrough options merge raw + camelCase keys with camelCase taking precedence, non-stream and stream-finish provider metadata keep the resolved request-side namespace key with an explicit provider root, and Gemini-compatible `extra_content.google.thought_signature` now survives on compat tool calls as `providerMetadata.{provider}.thoughtSignature`.
- OpenAI-compatible public config/builder surfaces now also expose an AI SDK-style response `metadataExtractor` hook through `ResponseMetadataExtractor`, `OpenAiCompatibleConfig::with_metadata_extractor(...)`, and `OpenAiCompatibleBuilder::with_metadata_extractor(...)`, so callers can extend provider metadata without reimplementing the whole compat adapter.
- OpenAI-compatible public package/facade exports now also mirror the audited AI SDK names more
  closely: `provider_ext::openai_compatible` now re-exports `MetadataExtractor` alongside the
  existing Rust-named `ResponseMetadataExtractor`, and also exposes a generic
  `ProviderErrorStructure<T>` helper for serde-based provider error decoding, message extraction,
  and optional retryability predicates.
- OpenAI-compatible provider-level request settings now align more closely with AI SDK `openai-compatible`: chat streaming omits `stream_options.include_usage` by default unless callers opt in through `OpenAiCompatibleConfig::with_include_usage(true)` / `OpenAiCompatibleBuilder::with_include_usage(true)`, public `queryParams`-style URL settings now flow through compat chat / embeddings / image generation-edit-variation / audio / rerank / model-listing routes, and the final compat chat body can now be customized through a public `RequestBodyTransformer` hook that mirrors AI SDK `transformRequestBody`.
- OpenAI-compatible public config/builder/runtime surfaces now also expose an explicit provider-level `supportsStructuredOutputs` policy aligned with AI SDK semantics: compat chat now defaults to downgrading JSON Schema outputs to `response_format = { "type": "json_object" }` while emitting a stable `unsupported { feature: "responseFormat" }` warning on chat responses, and callers can opt back into wire-level `json_schema` by setting `supportsStructuredOutputs = true`.
- OpenAI-compatible chat request shaping now also honors AI SDK-style known compat provider options from both canonical `providerOptions.openaiCompatible` and provider-owned keys: `user`, `reasoningEffort`, `textVerbosity`, and `strictJsonSchema` are mapped onto the final wire body (`user`, `reasoning_effort`, `verbosity`, `response_format.json_schema.strict`) instead of leaking through as raw compatibility keys.
- OpenAI-compatible image request shaping now follows the AI SDK image provider-options lane more closely: compat image generation/edit/variation no longer hardcode `providerOptions.openai|azure`, provider-owned image options now merge from deprecated `openai-compatible`, canonical `openaiCompatible`, and provider-owned keys, and image generation now surfaces a stable `unsupported { feature: "seed" }` warning instead of silently dropping `seed`.
- OpenAI and OpenAI-compatible image warning semantics now also cover top-level `aspectRatio` /
  `seed` consistently across generation, edit, and variation requests, matching the AI SDK-style
  unsupported-warning lane instead of leaving edit/variation as silent drops.
- OpenAI-compatible chat responses now also surface AI SDK-style warnings for provider-defined tools on the default runtime path: provider-defined tools are still filtered out of Chat Completions requests, and successful chat responses now include `unsupported { feature: "provider-defined tool <id>" }` warnings without requiring callers to install a custom middleware.
- OpenAI-compatible chat responses now also surface the AI SDK deprecation warning for legacy `providerOptions['openai-compatible']`: the deprecated key still works for audited compat chat options, but successful chat responses now include `other { message: "The 'openai-compatible' key in providerOptions is deprecated. Use 'openaiCompatible' instead." }` on the default runtime path.
- Unified warnings now expose AI SDK-style `unsupported` / `compatibility` shapes through a compatibility-superset model, and `systemMessageMode=remove` is reported through the `compatibility` warning type instead of a generic message.
- Unified `source` and `tool-approval-*` parts now preserve document/approval fields needed for closer AI SDK parity, including source `mediaType` / `filename` / `providerMetadata`, approval request `providerMetadata`, and approval response `reason`.
- Stable `source` parts now use a stricter URL/document union shape instead of a loose `sourceType + optional fields` bag, while preserving the same wire-level `sourceType` serialization.
- OpenAI Responses request conversion now forwards approval reasons, while Gemini and Anthropic source fallback paths now handle document-style source parts without assuming URL-only payloads.
- OpenAI/Azure Responses non-stream response parsing now matches AI SDK response content shape more closely: assistant `message.content[*].output_text` is always preserved as structured text parts on the stable boundary, including plain and empty text, so typed `providerMetadata.{openai|azure}.itemId` is no longer lost behind the single-text fast path, and the canonical OpenAI Responses JSON encoder now uses that text-part metadata instead of consuming or duplicating a legacy top-level response `itemId`. The same OpenAI Responses response-side alignment sweep also preserves `responseId` / `serviceTier`, text-part `phase` / raw `annotations`, and document citation `type` / `index` across exact JSON and SSE roundtrips.
- xAI Responses now follows the audited `repo-ref/ai` boundary more closely: non-stream text/source parts stay metadata-free, reasoning parts use `providerMetadata.xai.itemId` instead of the shared OpenAI namespace, top-level response `provider_metadata` is omitted, and xAI SSE reasoning parts now also carry `providerMetadata.xai.itemId` while `text-*` / `finish` remain metadata-free. The xAI SSE converter also backfills a missing `reasoning-start` before `reasoning-end` when upstream closes a reasoning item without an earlier start event.
- xAI request-side parity moved closer to the audited AI SDK split as well: the shared Responses request transformer now maps xAI `reasoningEffort` / `reasoningSummary`, `topLogprobs -> logprobs=true`, `previousResponseId`, and `store=false -> include += reasoning.encrypted_content`, assistant xAI message ids no longer collapse into OpenAI-style `item_reference`, assistant xAI tool calls now emit stable ids plus `status: "completed"`, and the OpenAI-compatible xAI chat path now normalizes supported chat fields while stripping Responses-only knobs (`reasoningSummary`, `previousResponseId`, `include`, `store`) before hitting `/chat/completions`.
- xAI's compat-backed chat configuration now opts into structured outputs by default, so JSON Schema
  response formats stay on the canonical `json_schema` wire shape instead of being downgraded to
  `json_object` through the generic OpenAI-compatible fallback policy.
- xAI chat typed options now also align more closely with `repo-ref/ai`: `parallel_function_calling` is exposed end-to-end on the typed surface, deprecated `xHandles` input now normalizes to wire `included_x_handles` instead of leaking as `x_handles`, and `with_default_search()` now matches the upstream `maxSearchResults=20` default.
- The provider-owned xAI option model is now split to match the audited AI SDK structure more closely: chat-only knobs live on `XaiChatOptions`, Responses-only knobs live on `XaiResponsesOptions`, and the main reasoning/include slots now use enum-backed typed wrappers instead of raw `String` / `Vec<String>` bags while preserving forward-compatible string passthrough for newly introduced upstream values.
- xAI search-source typing now follows the AI SDK structure more closely: `SearchSource` is modeled as a discriminated union over `web` / `news` / `x` / `rss` instead of a single permissive field bag, while deprecated `xHandles` input still normalizes to `included_x_handles`.
- xAI Responses tool preparation now matches the audited AI SDK `packages/xai/src/tool/*` and `responses/xai-responses-prepare-tools.ts` surface more closely: public Rust tool factories now cover `web_search`, `x_search`, `code_execution`, `view_image`, `view_x_video`, `file_search`, and `mcp`; xAI tool args now serialize to the expected snake_case request shape (`allowed_domains`, `allowed_x_handles`, `vector_store_ids`, `server_url`, etc.); unknown xAI provider-defined tools are no longer forwarded blindly; and xAI server-side provider tools are no longer forced through invalid Responses `tool_choice` payloads.
- The xAI provider-defined tool surface now also exposes typed Rust arg models and factory-style helpers for the audited AI SDK tool configs: `WebSearchArgs`, `XSearchArgs`, `FileSearchArgs`, and `McpArgs`, plus `web_search_with(...)`, `x_search_with(...)`, `file_search_with(...)`, and `mcp_server_with(...)`, so callers no longer need raw `.with_args(json)` for the main xAI hosted-tool path.
- xAI Responses SSE custom-tool streaming now matches the audited `repo-ref/ai/packages/xai/src/responses/xai-responses-language-model.ts` flow more closely: xAI `custom_tool_call` items (`x_search`, `view_x_video`) defer the finalized `tool-input-start` / `tool-input-delta` / `tool-input-end` plus `tool-call` emission until `response.output_item.done`, `response.custom_tool_call_input.*` is treated as input buffering instead of a second public event lane, and the fixture-backed xAI stream regression suite now covers `web_search`, `file_search`, and `x_search` on the stable part boundary.
- xAI public model constants were refreshed to match the current AI SDK reference set more closely, including `grok-4-1-fast-*`, `grok-4-fast-*`, `grok-4.20-*`, `grok-code-fast-1`, and `grok-3-mini-latest`.
- xAI provider-owned image/video parity now follows the audited AI SDK split much more closely: typed `XaiImageOptions` / `XaiVideoOptions` plus request ext traits are public, xAI native image generation/edit now use `/images/generations` and `/images/edits`, xAI native video create/query now use `/videos/generations|edits` and `GET /videos/{request_id}`, registry/native metadata/public-path parity now expose xAI image generation and video task support, and the shared video request/response types now carry AI SDK-style `providerOptions`, per-request `HttpConfig`, `aspectRatio`, `videoUrl`, metadata, warnings, and response envelopes.
- Local transcription file loading now happens in helper APIs such as `transcribe_file(...)` and
  `translate_file(...)` instead of leaking `file_path` into the stable request contract, and the
  OpenAI/OpenAI-compatible audio shaping paths now consume the canonical shared audio input
  directly.
- Shared transcription/audio-translation requests now also require `mediaType` on the stable Rust
  surface, so constructors, helpers, examples, and OpenAI/OpenAI-compatible multipart shaping now
  align with the AI SDK required-input contract instead of treating media type as optional.
- Built-in OpenAI-compatible `openrouter` and `perplexity` presets now default structured outputs
  to the schema-preserving path, so their public builder/config/registry surfaces no longer fall
  back to generic `json_object` formatting when AI SDK-aligned JSON Schema outputs are requested.
- OpenAI-compatible chat response decoding now also preserves provider-native legacy
  `raw_finish_reason = "function_call"` while still normalizing the stable finish reason to
  `tool_calls`.
- OpenAI Responses exact request/response roundtrip now also preserves the audited stable tool
  structure more closely: provider-executed tool calls/results keep stable `dynamic` plus
  tool-result `input`, and hosted dynamic `local_shell` / `shell` / `apply_patch` items now
  bridge back to native Responses tool item types instead of degrading to generic function calls.
- Shared URL joining now preserves query strings when appending paths, fixing query-bearing
  OpenAI/OpenAI-compatible endpoint composition such as OpenAI Files listing on `/files?...`
  without corrupting the request path.
- Shared image edit/provider boundaries now align more closely with AI SDK image-model semantics:
  `ImageEditRequest` carries typed multi-input `images[]` plus typed `mask`, xAI native edit now
  emits single-input `image` vs multi-input `images`, and OpenAI/OpenAI-compatible multipart edit
  plus Vertex inline edit now accept multiple file-backed source images. URL-backed edit inputs are
  already supported on xAI and are explicitly rejected on multipart/inline provider paths until a
  shared async materialization layer exists.
- Gemini / Google Imagen, Vertex Imagen, and xAI native image request shaping now also consume the
  canonical shared top-level `aspectRatio` / `seed` fields on their supported paths, reducing the
  remaining provider-specific image gaps to runtime materialization and a few incomplete variation
  edges rather than missing shared request structure.
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
- 鈥淎udio鈥?is no longer a first-class unified family: prefer `SpeechCapability` (TTS) and `TranscriptionCapability` (STT).
  - For OpenAI SSE audio/transcript streaming, use provider extensions: `siumai::provider_ext::openai::{speech_streaming, transcription_streaming}`.
- OpenAI鈥檚 public API does not expose a rerank endpoint.
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
- M1 鈥渃ore trio鈥?smoke scripts (fixture audit + transcoding + tool-loop gateway):
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
- Correct model propagation for OpenAI鈥慶ompatible in unified builder
- Env var loading for OpenAI鈥慶ompatible (`{PROVIDER_ID}_API_KEY`)
- Default/alias model handling across providers

## [0.11.0-beta.2] - 2025-11-08

### Added

- MiniMaxi provider support with multi-modal capabilities (text, speech, image generation).
- **Gemini File Search (RAG) support** - Provider-specific implementation for Gemini's File Search API
  - File Search Store management (create, list, get, delete)
  - Example: `siumai/examples/04-provider-specific/google/file_search.rs`

## [0.11.0-beta.1] - 2025-10-28

This beta delivers a major refactor of module layout, execution/streaming, and provider integration. Design inspired by Cherry Studio鈥檚 transformer design and the Vercel AI SDK鈥檚 adapter architecture.

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
- Unified streaming events (start/delta/usage/end); improved UTF鈥?-safe chunking and tag extraction.
- Unified retry facade (`retry_api`) with idempotency and 401 token refresh retry.
- OpenAI鈥慶ompatible providers consolidated via adapter; consistent transformers/executors paths.
- Clippy cleanups; boxed large enum variants internally (minor internal breaking).

### Removed
- Top-level `examples/` moved to `siumai/examples/`.
- Removed obsolete `docs/openapi.documented.yml`.

### Fixed
- Ensure `before_send_hook` is correctly applied across providers.
- UTF鈥? safety: tag extraction, string slicing, streaming chunk boundaries, and token masking.
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
- OpenAI鈥慶ompatible via Builder: `.api_key(..)` or `{PROVIDER_ID}_API_KEY`
- OpenAI鈥慶ompatible via Registry: reads `{PROVIDER_ID}_API_KEY` (e.g., `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`)

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


