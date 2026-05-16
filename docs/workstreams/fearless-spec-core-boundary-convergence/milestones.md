# Fearless Spec/Core Boundary Convergence - Milestones

Last updated: 2026-05-16

## FSCBC-M0 - Workstream Created

Acceptance criteria:

- Design, TODO, milestone, and audit documents exist.
- The docs index links to this workstream.
- The workstream states the target boundaries for `siumai-spec`, `siumai-core`, bridge, protocol,
  registry, and facade crates.

Status: done

## FSCBC-M1 - Spec Runtime Leakage Removed

Acceptance criteria:

- `siumai-spec` no longer owns runtime cancellation handles.
- `siumai-spec` no longer requires async runtime dependencies for pure data contracts.
- A source guard prevents runtime-only imports from returning to `siumai-spec`.
- `cargo check -p siumai-spec --no-default-features` passes.

Status: done

Notes:

- `CancelHandle` now lives in `siumai-core::types`, with core-owned aliases for
  `RequestOptions`, `LanguageModelV4CallOptions`, and deprecated `CallSettings`.
- `siumai-spec` keeps those option carriers generic over the abort handle and defaults to `()`, so
  the spec crate no longer needs cancellation runtime dependencies.
- Runtime `AudioStream` now lives in `siumai-core::types`; `siumai-spec` keeps only
  `AudioStreamEvent` data.
- `tokio-util` and `futures` were removed from `siumai-spec`.
- `siumai-spec::spec_purity_boundary_test` rejects `tokio-util`, `futures`, `CancellationToken`,
  `CancelHandle`, runtime stream aliases, and `siumai_core` source leaks.
- `siumai-spec::LlmError` now stays passive too: retry eligibility, status-code/category
  classification, user-facing messages, recovery suggestions, retry delay, and max-attempt policy
  moved to `siumai-core::error::{ErrorCategory,LlmErrorExt}`.
- `siumai-spec::spec_purity_boundary_test::spec_error_type_does_not_own_runtime_policy_helpers`
  prevents runtime/presentation error policy helpers from returning to the spec crate.
- `siumai-spec::HttpConfig` no longer owns runtime environment policy. The
  `SIUMAI_STREAM_DISABLE_COMPRESSION` default is resolved by
  `siumai-core::defaults::http::config_default()`, while the spec type keeps only a deterministic
  passive default value and no longer exposes a runtime-default constructor.
- `siumai-spec::spec_purity_boundary_test::spec_source_does_not_define_runtime_handles_or_streams`
  now rejects `std::env`, `CARGO_PKG_VERSION`, and `runtime_default` usage under
  `siumai-spec/src`.
- `siumai-spec::http_config_boundary_test` locks the passive `HttpConfig::default()` /
  `HttpConfig::empty()` / builder semantics so runtime defaults cannot silently move back into spec
  data construction.
- Request-level header override fixtures in core executors, selected provider helpers, Google
  Vertex context tests, Ollama/OpenAI unit tests, and public examples now use `HttpConfig::empty()`;
  config-first provider constructors remain the runtime-default path.
- `siumai-core::core_provider_boundary_test::runtime_http_defaults_do_not_use_passive_http_config_default`
  scans production sources across core, protocol, provider, registry, extras, and facade crates so
  passive `HttpConfig::default()` cannot return as a runtime-default shortcut.
- `rg "HttpConfig::default\(" -n -g "*.rs" --glob "!target/**"` now only finds the intentional
  spec boundary test and the guard text.
- `ChatResponse` docs now use pure `siumai-spec` data construction examples for metadata,
  conversation-history conversion, and response-id transport instead of facade builders, provider
  extension traits, or concrete client execution snippets.
- `siumai-spec::spec_purity_boundary_test::spec_docs_do_not_teach_facade_or_provider_runtime_construction`
  guards `siumai-spec/src` against facade/prelude imports, `Siumai::builder`, provider extension
  paths, provider builder calls, API-key builder calls, and client chat execution snippets.
- Verified commands for this slice:
  - `cargo check -p siumai-spec --no-default-features`
  - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test http_config_boundary_test --no-default-features --no-fail-fast`
  - `cargo fmt --package siumai-spec --check`
  - `cargo check -p siumai-spec --no-default-features`
  - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`
  - `cargo check -p siumai-core --no-default-features`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo fmt -p siumai-spec -p siumai-core --check`
  - `cargo check -p siumai-registry --tests --features openai,anthropic,google --no-default-features`
  - `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
  - `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test tooling_runtime_public_surface_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test openai_sse_streaming_alignment_test --features openai --no-default-features --no-fail-fast`

Additional error-policy convergence validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo check -p siumai-core --no-default-features`
- `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --test spec_purity_boundary_test spec_error_type_does_not_own_runtime_policy_helpers --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib error::policy --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib error::helpers --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib retry --no-default-features --no-fail-fast`
- `cargo check -p siumai-protocol-anthropic --no-default-features --features anthropic-standard`
- `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard standards::anthropic::errors standards::anthropic::utils::errors --no-fail-fast`
- `cargo check -p siumai-extras --no-default-features --features server,openai,anthropic,google`
- `cargo check -p siumai-extras --example openai-responses-gateway --no-default-features --features server,openai,anthropic,google`
- `cargo check -p siumai-registry --tests --features openai,anthropic,google --no-default-features`
- `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
- `cargo nextest run -p siumai-core --lib defaults builder --no-default-features --no-fail-fast`

Follow-up facade/prelude cleanup:

- `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features` now
  passes after facade tests stopped relying on `prelude::unified::*` for low-level streaming
  internals (`SseEventConverter`, `StreamProcessor`, `ProcessedEvent`, `ChatByteStream`) and
  file-upload provider hooks.
- Tests now import low-level streaming internals from `siumai::experimental::streaming` and
  upload-provider hooks from `siumai::files`, matching the narrowed stable prelude boundary.

## FSCBC-M2 - AI SDK Surface Split

Acceptance criteria:

- Oversized AI SDK type modules are split by responsibility.
- Prompt/input data, response/result data, UI message data, and runtime helpers no longer live in one
  mixed ownership module.
- Public imports have migration notes where paths changed.
- Focused spec and facade checks pass.

Status: in progress

Notes:

- The AI SDK surface now lives as `siumai-spec/src/types/ai_sdk/mod.rs` instead of the old
  single-file `siumai-spec/src/types/ai_sdk.rs` module.
- Shared primitives were split into `siumai-spec/src/types/ai_sdk/shared.rs`.
- Call option carriers were split into `siumai-spec/src/types/ai_sdk/call_options.rs`, keeping
  `LanguageModelV4CallOptions`, `RequestOptions`, `LanguageModelReasoning`,
  `LanguageModelCallOptions`, and deprecated `CallSettings` behind the stable `ai_sdk` re-export.
- Passive error carriers were split into `siumai-spec/src/types/ai_sdk/errors.rs`, keeping
  `AISDKError`, `APICallError`, provider/model lookup errors, no-output/no-media errors,
  UI message conversion errors, retry errors, and validation errors behind the stable `ai_sdk`
  re-export.
- Generated file carriers were split into `siumai-spec/src/types/ai_sdk/generated_files.rs`,
  keeping `GeneratedFile`, `DefaultGeneratedFileWithType`, `GeneratedAudioFile`, and
  `DefaultGeneratedAudioFileWithType` behind the stable `ai_sdk` re-export.
- Response metadata carriers were split into
  `siumai-spec/src/types/ai_sdk/response_metadata.rs`, keeping `ImageModelResponseMetadata`,
  `VideoModelResponseMetadata`, `SpeechModelResponseMetadata`, and
  `TranscriptionModelResponseMetadata` behind the stable `ai_sdk` re-export.
- Usage carriers and helpers were split into `siumai-spec/src/types/ai_sdk/usage.rs`, keeping
  `LanguageModelV4Usage`, `LanguageModelUsage`, `EmbeddingModelUsage`, `ImageModelUsage`,
  `add_language_model_usage`, and `add_image_model_usage` behind the stable `ai_sdk` re-export.
- Embedding result and event carriers were split into
  `siumai-spec/src/types/ai_sdk/embedding.rs`, keeping `ModelCallResponseData`, `EmbedResult`,
  `EmbedManyResult`, `EmbedStartEvent`, `EmbedEndEvent`, and embedding model-call events behind the
  stable `ai_sdk` re-export.
- Rerank result and event carriers were split into `siumai-spec/src/types/ai_sdk/rerank.rs`,
  keeping `RerankResponseMetadata`, `RerankResult`, `RerankStartEvent`, `RerankEndEvent`, and
  reranking model-call events behind the stable `ai_sdk` re-export.
- Language-model request/response metadata carriers were split into
  `siumai-spec/src/types/ai_sdk/language_model_metadata.rs`, keeping
  `LanguageModelRequestMetadata`, `LanguageModelResponseMetadata`,
  `LanguageModelV4ResponseMetadata`, `LanguageModelV4GenerateResponseMetadata`, and
  `LanguageModelV4StreamResponseMetadata` behind the stable `ai_sdk` re-export.
- Language-model V4 result envelopes were split into
  `siumai-spec/src/types/ai_sdk/language_model_results.rs`, keeping
  `LanguageModelV4FinishReason`, `LanguageModelV4GenerateResult`, and
  `LanguageModelV4StreamResult` behind the stable `ai_sdk` re-export.
- Language-model V4 prompt/content projections were split under
  `siumai-spec/src/types/ai_sdk/language_model_v4/`, while
  `siumai-spec/src/types/ai_sdk/language_model_v4.rs` remains a thin re-export shell:
  - `language_model_v4/shared.rs` owns shared V4 data carriers, type markers, provider
    option/metadata object-shape validators, and custom-kind validation.
  - `language_model_v4/prompt.rs` owns request-side prompt parts, message projections, and
    `prepare_language_model_v4_prompt`.
  - `language_model_v4/content.rs` owns response-side generated text/reasoning/source/file/tool
    content and `LanguageModelV4Content`.
- Source citation carriers were split into `siumai-spec/src/types/ai_sdk/source.rs`, keeping
  `Source`, `ImageModelProviderMetadata`, and `VideoModelProviderMetadata` behind the stable
  `ai_sdk` re-export.
- Structured-output result and event carriers were split into
  `siumai-spec/src/types/ai_sdk/generate_object.rs`, keeping
  `GenerateObjectOutputStrategy`, `GenerateObjectResponseMetadata`, `GenerateObjectStartEvent`,
  `GenerateObjectStepStartEvent`, `GenerateObjectStepEndEvent`, and `GenerateObjectEndEvent`
  behind the stable `ai_sdk` re-export.
- Text-generation result and callback event carriers were split into
  `siumai-spec/src/types/ai_sdk/generate_text.rs`, keeping `GenerateTextContentPart`,
  `GenerateTextResult`, `GenerateTextStepResult`, `GenerateTextStartEvent`,
  `GenerateTextStepStartEvent`, `GenerateTextEndEvent`, `PrepareStepOptions`,
  `PrepareStepResult`, and `StreamTextChunkEvent` behind the stable `ai_sdk` re-export.
- Output content and tool-output carriers were split into
  `siumai-spec/src/types/ai_sdk/output_parts.rs`, keeping `TextOutput`, `CustomOutput`,
  `FileOutput`, `ReasoningOutput`, `ToolCall`, `ToolResult`, `ToolError`, `ToolOutput`,
  `ToolOutputDenied`, and tool approval output parts behind the stable `ai_sdk` re-export.
- Flow-control helpers were split into `siumai-spec/src/types/ai_sdk/flow_control.rs`, keeping
  `StopCondition`, active-tool filtering, and `pruneMessages` data rules/helpers behind the stable
  `ai_sdk` re-export.
- Tool lifecycle carriers were split into `siumai-spec/src/types/ai_sdk/tool_lifecycle.rs`,
  keeping approval configuration, repair callback data, execution callback events, and deprecated
  callback aliases behind the stable `ai_sdk` re-export.
- Timeout carriers and helper accessors were split into `siumai-spec/src/types/ai_sdk/timeout.rs`,
  keeping `TimeoutConfiguration`, `TimeoutConfigurationSettings`, and `get_*_timeout_ms` behind
  the stable `ai_sdk` re-export.
- Object-stream part carriers were split into `siumai-spec/src/types/ai_sdk/object_stream.rs`,
  keeping `ObjectStreamObjectPart`, `ObjectStreamTextDeltaPart`, `ObjectStreamErrorPart`,
  `ObjectStreamFinishPart`, and `ObjectStreamPart` behind the stable `ai_sdk` re-export.
- Text-stream and language-model stream part carriers were split into
  `siumai-spec/src/types/ai_sdk/text_stream.rs`, keeping `TextStreamPart`,
  `LanguageModelStreamPart`, their concrete part structs, and experimental language-model stream
  aliases behind the stable `ai_sdk` re-export.
- Media result envelopes were split into `siumai-spec/src/types/ai_sdk/media_results.rs`, keeping
  `GenerateImageResult`, `GenerateVideoResult`, `SpeechResult`, `TranscriptionResult`, and their
  experimental compatibility aliases behind the stable `ai_sdk` re-export.
- UI message aliases, helper predicates, chat transport options, completion options, and
  `UI_MESSAGE_STREAM_HEADERS` were split into `siumai-spec/src/types/ai_sdk/ui_message.rs`, keeping
  the stable `ai_sdk` re-export.
- UI message stream chunk carriers were split into
  `siumai-spec/src/types/ai_sdk/ui_message_chunks.rs`, keeping `UiMessageChunk`,
  `UIMessageChunk`, `DataUIMessageChunk`, `InferUIMessageChunk`, and concrete
  `UiMessage*Chunk` structs behind the stable `ai_sdk` re-export.
- `siumai-spec::ai_sdk_module_boundary_test` guards the directory-module shape and shared
  primitive, call-option, embedding, error, flow-control, generate-object, generate-text,
  generated-file, language-model metadata, language-model result, nested language-model V4
  shared/prompt/content, media-result, object-stream, output-part, rerank, response-metadata,
  source, text-stream, timeout, tool-lifecycle, UI-message, UI-message-chunk, and usage splits. It
  also keeps the V4 prompt projection source free of response-side provider metadata and the V4
  generated content projection source free of request-side provider options. `ai_sdk/mod.rs` stays
  a thin re-export shell with no concrete type/function/impl definitions in production code, and
  the same thin-shell guard applies to `ai_sdk/language_model_v4.rs`.
- `siumai-spec::tools_boundary_test::provider_defined_tool_data_surface_remains_passive` now covers
  both `siumai-spec/src/tools.rs` and `siumai-spec/src/types/tools/**`, keeping provider-defined
  tool helpers and stable tool data carriers free of runtime execution, HTTP, core tooling
  execution, provider crate, or protocol crate dependencies.
- `siumai-spec::video_boundary_test` now covers `siumai-spec/src/types/video.rs`, keeping video
  generation request/response/status carriers free of provider/runtime execution, polling,
  download, HTTP client/server, thread/process, core, provider crate, or protocol crate behavior.
  It also locks `VideoGenerationRequest::with_header(...)` to `HttpConfig::empty()` so request
  header overrides do not pull runtime HTTP defaults back into spec data construction.
- `siumai-spec::request_carrier_boundary_test` now covers non-`ai_sdk` audio, completion,
  embedding, files, image, rerank, and skills request surfaces. These model-family/upload request
  carriers stay passive data contracts, and completion/embedding/file-list/rerank/skill-upload
  header helpers are locked to `HttpConfig::empty()` request overrides.
- `siumai-spec::chat_provider_helper_boundary_test` now guards the removal of the historical
  `ChatMessageBuilder` Anthropic cache/document helpers. The spec chat builder no longer owns
  provider-specific request helpers, and the guard prevents those removed methods, new
  provider-prefixed chat builder helpers, Anthropic request option literals, response metadata
  reads, provider/protocol crate dependencies, or runtime behavior from entering the spec chat
  builder.
- `siumai-provider-anthropic::providers::anthropic::ext::AnthropicChatMessageExt` now provides the
  provider-owned replacement path for Anthropic message cache/document request helpers. The facade
  exposes it through `siumai::provider_ext::anthropic`, and the Anthropic prompt-caching example now
  uses the provider-owned extension trait on built `ChatMessage` values instead of teaching the
  removed `ChatMessageBuilder::cache_control(...)` method.
- Facade cache-control macro arms now route through a narrow private helper and are guarded against
  calling the removed `ChatMessageBuilder::cache_control(...)` path again.
- Verified commands for this slice:
  - `cargo check -p siumai-spec --no-default-features`
  - `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`
  - `cargo fmt -p siumai-spec --check`
  - `git diff --check`
  - `cargo nextest run -p siumai-spec --test tools_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test video_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --test http_config_boundary_test --test video_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test request_carrier_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --test http_config_boundary_test --test request_carrier_boundary_test --test video_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --test spec_purity_boundary_test --test tools_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::chat_message --no-default-features --features anthropic --no-fail-fast`
  - `cargo check -p siumai --example prompt-caching --features anthropic --no-default-features`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_macros_only_create_request_side_empty_provider_options --features anthropic --no-default-features --no-fail-fast`

## FSCBC-M3 - Prompt And Response Content Views Separated

Acceptance criteria:

- Request-side provider options and response-side provider metadata have explicit boundaries.
- Shared content data is reused through adapters or common data components, not by mixing prompt and
  response semantics into one public shape.
- Tests cover representative request construction and response parsing paths.

Status: in progress

Notes:

- The legacy stable `ContentPart` union is now explicitly audited as a dual-use compatibility
  carrier for the variants that still contain both request-side `providerOptions` and response-side
  `providerMetadata`: `Text`, `Image`, `Audio`, `File`, `ReasoningFile`, `Custom`, `ToolCall`,
  `ToolApprovalRequest`, `ToolResult`, and `Reasoning`.
- `ContentPart::Source` remains response-side citation data with `providerMetadata` only, while
  `ContentPart::ToolApprovalResponse` remains request/decision-side data with `providerOptions`
  only.
- `siumai-spec::content_projection_boundary_test::legacy_content_part_dual_provider_maps_stay_explicitly_audited`
  turns that audit into a source guard so new dual-use `ContentPart` variants cannot appear
  silently.
- `content-part-construction-audit.md` now records the direct `ContentPart` construction search
  baseline, guarded paths, lower-priority false-positive buckets, and the remaining high-value
  candidates for Track C follow-up work.
- The refreshed direct-construction scan now classifies `siumai-bridge/src/customize.rs` as
  primitive-only bridge customization, with
  `bridge_customization_source_stays_primitive_only` preventing built-in remappers from becoming
  a provider option/metadata side channel.
- The same scan now classifies `siumai-core/src/streaming/{builder.rs,factory.rs}` as
  provider-agnostic stream helper paths, with
  `core_stream_helpers_only_initialize_empty_provider_metadata` allowing only empty
  `provider_metadata: None` initialization in those helpers.
- `siumai-core/src/utils/streaming_tool_call.rs` is now guarded as a provider-agnostic tracker:
  provider metadata may only arrive through caller-supplied generic callbacks, and core must not
  hard-code provider metadata namespaces or read request provider options there.
- `siumai-core/src/structured_output.rs` is now guarded as provider-agnostic JSON extraction and
  stream-response consolidation: it may generically merge response metadata, but it must not
  inspect request provider options or provider-specific metadata namespaces.
- AI SDK V4 prompt projections remain directional: `prompt.rs` carries request-side
  `providerOptions` and `siumai-spec::ai_sdk_module_boundary_test` rejects response-side
  `providerMetadata` / `provider_metadata` there.
- AI SDK V4 generated content projections remain directional: `content.rs` carries response-side
  `providerMetadata` and `siumai-spec::ai_sdk_module_boundary_test` rejects request-side
  `providerOptions` / `provider_options` there.
- Existing `ChatMessage`/`Prompt` narrowing adapters already reject response-side
  `providerMetadata` when converting legacy `ContentPart` values into prompt/model-message shapes.
  `content_projection_boundary_test::prompt_projection_rejects_response_side_provider_metadata_on_legacy_content_parts`
  guards representative legacy text, image, file, reasoning, custom, tool-call, and tool-result
  parts.
- Gemini request transformer post-processing is now guarded as request-side code: it may read
  request `provider_options` but must not read response-side `provider_metadata` /
  `providerMetadata`.
- Anthropic's mixed request/response transformer now has split direction guards, keeping the
  request transformer away from response metadata and the response transformer away from request
  provider options.
- AI SDK UI message conversion in `siumai-core::ui` is explicitly treated as an adapter boundary:
  UI-layer `providerMetadata`, `callProviderMetadata`, and `resultProviderMetadata` are request
  metadata names inherited from AI SDK UI messages, so the conversion normalizes them into
  request-side `provider_options` and leaves legacy `ContentPart::provider_metadata` unset for
  model-message request parts.
- `siumai-core::ui` unit tests guard that normalization for text, reasoning, custom, file,
  reasoning-file, tool-call, and tool-result parts.
- `siumai-core::ui` now routes UI-to-request legacy `ContentPart` creation through local
  `ui_request_*` adapter helpers, with
  `ui_conversion_centralizes_legacy_request_content_constructors` guarding that response-side
  provider metadata remains centralized and empty on request parts.
- Gemini request conversion no longer backfills `thoughtSignature` or `thought` from legacy
  `ContentPart::provider_metadata`; request replay uses `provider_options` only and is covered by
  `request_conversion_ignores_legacy_provider_metadata_thought_signature` plus the source guard
  `request_conversion_source_only_ignores_legacy_provider_metadata_fields`.
- Anthropic request conversion no longer backfills reasoning replay `signature` / `redactedData`
  from legacy `ContentPart::provider_metadata`; request replay uses `provider_options` only and is
  covered by `assistant_reasoning_ignores_legacy_provider_metadata_signature_for_request_replay`
  plus the source guard `request_conversion_source_does_not_read_legacy_provider_metadata_fields`.
- Anthropic request content conversion also guards `convert_message_content` with
  `request_content_conversion_source_does_not_read_legacy_provider_metadata_fields`, so document,
  file, and tool-result request settings cannot be restored from legacy response metadata.
- Anthropic prompt-cache request building now has
  `cache_request_builder_source_does_not_read_legacy_provider_metadata`, so cache block construction
  cannot learn from legacy response-side provider metadata.
- Anthropic prepare-step container replay remains intentionally response-metadata driven at the
  step-history level: `find_anthropic_container_id_from_last_step(...)` reads prior-step
  `ProviderMetadataMap` values, and `forward_anthropic_container_id_from_last_step(...)` emits
  next-step request-side `ProviderOptionsMap` overrides. The
  `prepare_step_source_only_bridges_response_metadata_to_request_provider_options` guard prevents
  that helper from becoming a legacy `ContentPart::provider_metadata` replay path.
- Anthropic hosted-tool extension helpers are now direction-guarded in
  `providers::anthropic::ext::tools`: `with_anthropic_tool_options(...)` remains request-side
  provider option construction, while hosted-tool `ChatStreamPart` / custom-event projections
  remain response/stream metadata views and do not read request provider options.
- Gemini tools extension helpers now guard source/custom-event projection with
  `gemini_tools_extension_source_does_not_read_request_provider_options`, so response/stream source
  helper code cannot start reading request-side provider options.
- OpenAI Responses request conversion already follows the same direction for image detail and
  assistant tool item ids, with existing tests proving legacy part provider metadata is ignored and
  a source guard keeping the request implementation from reintroducing direct legacy reads. The
  guard also requires request-side enum destructuring to keep legacy `provider_metadata: _` ignored
  instead of binding it for replay.
- OpenAI Chat and OpenAI-compatible request conversion utilities now have
  `openai_chat_request_conversion_source_does_not_read_legacy_provider_metadata_fields`, keeping
  shared message/content request serialization on canonical `provider_options` while preventing
  legacy response-side `ContentPart::provider_metadata` reads from returning.
- Alibaba/Qwen cache-control request shaping in
  `siumai-protocol-openai::standards::openai::compat::alibaba_cache_control` now has
  `alibaba_cache_control_source_does_not_read_legacy_provider_metadata`, so cache-control request
  shaping remains request-side `providerOptions` only.
- OpenAI-compatible protocol transformers now split mixed files by direction. Request
  transformation is guarded by
  `openai_compatible_request_transformer_source_does_not_read_legacy_provider_metadata_fields`,
  while chat response parsing is guarded by
  `openai_compatible_chat_response_source_does_not_emit_request_provider_options`.
- OpenAI-compatible streaming replay is guarded by
  `openai_compatible_streaming_source_does_not_emit_request_provider_options`, so stream
  conversion can normalize response provider metadata namespaces without emitting request-side
  provider options.
- OpenAI Responses response parsing is guarded by
  `responses_response_transformer_source_does_not_emit_request_provider_options`, keeping output
  item metadata response-side while legacy response `ContentPart::provider_options` remains empty.
- The protocol-owned OpenAI typed provider metadata view now has
  `openai_provider_metadata_source_does_not_read_request_provider_options`.
- Provider-owned OpenAI legacy completions are also split by source guard:
  `completion_request_source_does_not_read_legacy_provider_metadata_fields` keeps prompt/body
  construction away from legacy response metadata, and
  `completion_response_and_stream_source_do_not_emit_request_provider_options` keeps completion
  response/stream output away from request provider options.
- Anthropic response parsing is now guarded by
  `anthropic_parse_response_content_source_does_not_emit_request_provider_options`, keeping
  response `ContentPart` construction on empty default request options while still surfacing
  citations, tool metadata, and sources.
- Gemini request normalization in `siumai-bridge` now restores thought signatures into request-side
  `provider_options.google` and leaves legacy `ContentPart::provider_metadata` empty, covered by
  `gemini_request_normalization_source_uses_provider_options_for_thought_signature` and
  `gemini_generate_content_request_normalization_roundtrip_preserves_core_projection`.
- `siumai-bridge::request` also has
  `request_normalization_source_never_populates_legacy_provider_metadata`, ensuring request
  normalization does not read legacy `providerMetadata` / `provider_metadata` JSON keys and only
  sets legacy `provider_metadata` to `None`.
- `siumai-bridge::request` now routes request-normalized legacy `ContentPart` creation through
  request-side adapter helpers. OpenAI Responses, OpenAI Chat Completions, Anthropic Messages, and
  Gemini Generate Content parsing no longer scatter direct dual-use carrier construction across
  protocol branches.
- `request_normalization_centralizes_legacy_request_content_constructors` guards that the
  request-normalized `provider_metadata: None` writes stay inside the adapter helper block, except
  for the plain-text collapse match that reads the field while preserving text-only messages.
- Direct OpenAI Responses ↔ Anthropic Messages request bridge pairs are now source-guarded as
  request-side bridge code, so pair-specific reasoning replay cannot start reading legacy
  response-side provider metadata.
- `siumai-provider-google-vertex::standards::vertex_gemini_image` now routes synthetic Gemini image
  edit/variation file parts through a provider-owned request adapter helper. The
  `vertex_gemini_image_request_content_construction_is_centralized` source guard and
  `image_input_part_maps_provider_options_without_provider_metadata` behavior test keep provider
  options on the request side while leaving legacy response metadata empty.
- `siumai-provider-google-vertex::providers::vertex::video` now source-guards its task-based video
  request/response split. Vertex video request construction may read request-side
  `providerOptions.vertex`, while operation status response parsing may emit Vertex video metadata
  without reading request provider options.
- `siumai-provider-amazon-bedrock::standards::bedrock::chat` now source-guards its mixed
  request/response file. Request conversion may read only request-side `provider_options` for
  Bedrock document, cache-point, and reasoning replay settings, while response and streaming
  parsing may only initialize legacy `ContentPart::provider_options` with the empty default when
  producing response-side reasoning parts.
- `siumai-provider-amazon-bedrock::standards::bedrock::embedding` now source-guards the non-chat
  embedding request/response transformer split. Request transformation may read
  `providerOptions.bedrock`, while response embedding parsing cannot read request provider
  options.
- `siumai-provider-minimaxi::providers::minimaxi::spec` now source-guards the Anthropic-protocol
  adapter split. Response/stream metadata re-keying from `anthropic` to `minimaxi` is kept away
  from request-side provider options, and MiniMaxi request option resolution is kept away from
  response-side provider metadata.
- `siumai-provider-minimaxi::providers::minimaxi::video` now source-guards its non-chat video task
  request/response helper split. Request body construction may read `providerOptions.minimaxi`,
  while task creation/query response parsing cannot read request provider options.
- Google Vertex, Gemini, Bedrock, and MiniMaxi typed provider metadata extension modules now
  source-guard their response-side role. The provider-owned `provider_metadata::*` views can
  extract typed metadata from `ChatResponse` / `ContentPart`, but cannot grow request provider
  option readers.
- The protocol-owned Anthropic typed metadata view now has the same direction guard:
  `anthropic_provider_metadata_source_does_not_read_request_provider_options` keeps
  `ChatResponse` / `ContentPart` metadata helpers response-side only.
- `siumai-protocol-anthropic::standards::anthropic::thinking` is now guarded as a mixed
  protocol helper: `ThinkingConfig` request construction cannot read response metadata, while
  response-side thinking projection cannot read request provider options.
- `siumai-core::execution::executors::stream_json` is now guarded as a provider-agnostic
  line-delimited JSON streaming executor. Provider-specific stream parsing and metadata projection
  stay delegated to injected JSON converters.
- Core stable family adapters, trait shells, and runtime tooling contracts are now guarded as
  provider-map-neutral surfaces by
  `core_family_contract_and_tooling_sources_do_not_handle_provider_maps`.
- `siumai-core::execution::middleware::samples` is now guarded as provider-neutral sample
  middleware. Synthetic stream text parts may only initialize empty `provider_metadata`, while
  provider options and concrete provider namespaces remain outside this sample layer.
- `siumai-core::utils::provider_options` is guarded as the generic request-side provider option
  parser: it may parse `ProviderOptionsMap` through schemas but cannot read response metadata or
  hard-code concrete provider namespaces.
- `siumai-core::custom_provider::guide` no longer teaches direct legacy `ContentPart` variant
  matching for provider JSON serialization; provider-specific content mapping remains outside core
  instead of living in documentation examples.
- Anthropic, OpenAI, Anthropic-on-Vertex, and MiniMaxi provider client/spec files now have focused
  source guards for request-side provider-option routing and default merging. These guards allow
  provider-owned request options while rejecting response metadata reads as request input.
- `siumai-provider-gemini::providers::gemini::client` is classified as a test-only structured-output
  stream metadata preservation hit; production metadata handling remains covered by Gemini
  transformer and provider metadata guards.
- Remaining facade hits are now classified and guarded: request-message macros may only initialize
  empty request-side provider options, speech/transcription/structured-output helpers project
  response metadata into high-level results without reading request provider options, and video
  keeps high-level polling options separate from legacy provider option maps.
- `siumai-core::streaming::StreamProcessor` is classified as provider-agnostic response
  consolidation. Its final response `ContentPart` construction can preserve response-side stream
  metadata, but the source guard prevents it from reading request-side `providerOptions` or
  provider option maps.
- `siumai-protocol-gemini::standards::gemini::transformers::response` now source-guards response
  `ContentPart` construction. Gemini thought signatures may be preserved in response-side
  `provider_metadata`, while legacy `provider_options` fields on response content remain empty
  defaults only.
- `siumai-bridge::response::tests::response_and_stream_bridge_sources_do_not_emit_request_provider_options`
  keeps response/stream bridge production sources free of request-side `providerOptions` /
  `provider_options`, so response serialization and stream replay cannot reintroduce request
  metadata carriers.
- `ContentPart` docs now state its compatibility status and steer new provider-facing projections
  toward the separated AI SDK V4 prompt/content modules.
- Non-V4 stable prompt projection now has named helpers in `siumai-spec::types::prompt` for both
  stable `ChatMessage` to `ModelMessage` narrowing and prompt `ModelMessage` back to stable
  `ChatMessage` projection. `content_projection_boundary_test` keeps the non-V4 prompt types free
  of response-side provider metadata, verifies legacy response metadata is rejected on narrowing,
  and checks prompt-to-legacy projection emits no response metadata.
- Non-V4 response projection is now explicitly classified as the existing AI SDK output surface
  (`GenerateTextContentPart` plus `output_parts.rs` carriers). The same boundary test now guards
  those response-side carriers against request provider options and keeps response metadata carriers
  explicit instead of inventing a new duplicate generated-content family.
- Response-owned legacy `ContentPart` subsets now have explicit fallible projection helpers into
  `GenerateTextContentPart`. The helpers preserve response-side `providerMetadata`, ignore legacy
  request-side `providerOptions`, and reject ambiguous legacy shapes rather than producing lossy
  generated output.
- The facade `siumai::text::generate_text` projection path delegates content mapping to the
  spec-owned helper and keeps only the documented legacy tool-result-without-input fallback local,
  so the facade no longer owns a duplicate content projection matrix.
- OpenAI Responses SSE converter sources are guarded as response/stream conversion code, preventing
  stream parsing and replay metadata projection from reading request-side provider options.
- Anthropic and Gemini streaming response parsers and serializers are now source-guarded so stream
  metadata projection/replay may read response-side provider metadata without reading request-side
  provider options.
- OpenAI, Anthropic, and Gemini gateway/proxy JSON response encoders are now classified as
  response-side encoders. Their source guards prevent those encoders from reading request-side
  `providerOptions` / `provider_options` while still allowing response metadata serialization.
- OpenAI Responses provider extension stream/custom-event projection is guarded separately from the
  request helper in the same file, so response projection can inspect stream/provider metadata
  without reading request `provider_options`.
- OpenAI non-chat audio request/response handling is now guarded as a mixed provider-owned path:
  audio request construction may read request-side `provider_options_map` and `extra_params`, but
  cannot read legacy response metadata or `providerMetadata` JSON fields as request input.
- OpenAI WebSocket session request/recovery handling is now guarded separately from the HTTP
  client path. The session may mutate request `provider_options_map` for warm-up and retry, but it
  cannot read legacy provider metadata as request input.
- OpenAI chat request entry and response-id cancel wrapping now have a source guard so default
  request `provider_options_map` merging cannot grow legacy provider metadata reads.
- OpenAI provider-owned file and skill upload extensions now have source guards. File upload
  request validation/option merging cannot read legacy provider metadata, and skill upload response
  projection can emit response-side provider metadata without reading request provider options.
- OpenAI, Anthropic, and Gemini protocol chat standard wrappers now have source guards that keep
  wrapper responsibilities directional: request-side adapter/context extraction does not read
  legacy response metadata, and response/stream wrapper sections do not read request provider
  options.
- Verified commands for this slice:
  - `cargo fmt --package siumai-spec --check`
  - `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --lib text --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-default-features --features openai --no-fail-fast`
  - `cargo check -p siumai-spec --no-default-features`
  - `cargo check -p siumai --no-default-features --features openai`
  - `cargo fmt --package siumai-spec --check`
  - `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
  - `cargo check -p siumai-spec --no-default-features`
  - `cargo check -p siumai-spec --no-default-features`
  - `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
  - `cargo fmt --package siumai-spec --check`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_sse_converter_sources_do_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard openai_json_response_encoder_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_streaming_parser_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_streaming_serialize_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_request_transformer_source_does_not_read_response_provider_metadata anthropic_response_transformer_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_json_response_encoder_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_request_transformer_source_does_not_read_response_provider_metadata --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_streaming_parser_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_streaming_serialize_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_json_response_encoder_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-protocol-openai --no-default-features --features openai-standard`
  - `cargo check -p siumai-protocol-anthropic --no-default-features --features anthropic-standard`
  - `cargo check -p siumai-protocol-gemini --no-default-features --features google`
  - `cargo fmt --package siumai-protocol-openai --package siumai-protocol-anthropic --package siumai-protocol-gemini --check`
  - `cargo fmt --package siumai-provider-openai --check`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai responses_stream_event_projection_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai chat_request_path_does_not_read_legacy_response_metadata_maps --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai audio_mixed_request_response_path_does_not_read_legacy_response_metadata --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai file_upload_request_path_does_not_read_legacy_provider_metadata skill_upload_request_and_response_paths_keep_provider_maps_directional --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai-websocket websocket_session_request_mutation_does_not_read_legacy_metadata_maps --no-fail-fast`
  - `cargo fmt --package siumai-protocol-openai --check`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard chat_wrapper_keeps_request_response_stream_maps_directional --no-fail-fast`
  - `cargo fmt --package siumai-protocol-anthropic --package siumai-protocol-gemini --check`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard chat_wrapper_keeps_request_and_response_provider_maps_directional --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google chat_wrapper_keeps_request_response_stream_maps_directional --no-fail-fast`
  - `cargo fmt --package siumai-protocol-anthropic --check`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard request_content_conversion_source_does_not_read_legacy_provider_metadata_fields --no-fail-fast`
  - `cargo check -p siumai-protocol-anthropic --no-default-features --features anthropic-standard`
  - `cargo fmt --package siumai-provider-anthropic --check`
  - `cargo nextest run -p siumai-provider-anthropic --no-default-features --features anthropic prepare_step_source_only_bridges_response_metadata_to_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-provider-anthropic --no-default-features --features anthropic tool_options_extension_source_does_not_read_response_provider_metadata stream_event_projection_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-provider-anthropic --no-default-features --features anthropic`
  - `cargo fmt --package siumai-core --check`
  - `cargo nextest run -p siumai-core --lib ui --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-bridge --no-default-features --features openai,anthropic request_bridge_pair_sources_do_not_read_legacy_provider_metadata --no-fail-fast`
  - `cargo fmt --package siumai-bridge --check`
  - `cargo nextest run -p siumai-bridge --no-default-features --features openai,anthropic,google response_and_stream_bridge_sources_do_not_emit_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-bridge --no-default-features --features openai,anthropic,google`
  - `cargo nextest run -p siumai-bridge --no-default-features --features openai,anthropic,google request::tests --no-fail-fast`
  - `cargo fmt --package siumai-provider-google-vertex --check`
  - `cargo nextest run -p siumai-provider-google-vertex --no-default-features vertex_gemini_image --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex video_request_and_response_paths_keep_provider_maps_directional --no-fail-fast`
  - `cargo fmt --package siumai-provider-amazon-bedrock --check`
  - `cargo nextest run -p siumai-provider-amazon-bedrock --no-default-features --features bedrock request_conversion_source_does_not_read_legacy_provider_metadata_fields response_and_stream_source_do_not_emit_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-provider-amazon-bedrock --no-default-features --features bedrock embedding_request_and_response_transformers_keep_provider_maps_directional --no-fail-fast`
  - `cargo check -p siumai-provider-amazon-bedrock --no-default-features --features bedrock`
  - `cargo fmt --package siumai-provider-minimaxi --check`
  - `cargo nextest run -p siumai-provider-minimaxi --no-default-features --features minimaxi response_metadata_normalization_source_does_not_read_request_provider_options request_option_resolution_source_does_not_read_response_provider_metadata --no-fail-fast`
  - `cargo nextest run -p siumai-provider-minimaxi --no-default-features --features minimaxi video_request_and_response_paths_keep_provider_maps_directional --no-fail-fast`
  - `cargo check -p siumai-provider-minimaxi --no-default-features --features minimaxi`
  - `cargo fmt --package siumai-provider-google-vertex --package siumai-provider-minimaxi --check`
  - `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex vertex_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-provider-minimaxi --no-default-features --features minimaxi minimaxi_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-provider-google-vertex -p siumai-provider-minimaxi --no-default-features --features "siumai-provider-google-vertex/google-vertex siumai-provider-minimaxi/minimaxi"`
  - `cargo fmt --package siumai-provider-gemini --package siumai-provider-amazon-bedrock --check`
  - `cargo nextest run -p siumai-provider-gemini --no-default-features --features google gemini_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-provider-amazon-bedrock --no-default-features --features bedrock bedrock_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-provider-gemini -p siumai-provider-amazon-bedrock --no-default-features --features "siumai-provider-gemini/google siumai-provider-amazon-bedrock/bedrock"`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
  - `cargo fmt --package siumai-core --check`
  - `cargo nextest run -p siumai-core --lib streaming::processor::tests::stream_processor_source_does_not_read_request_provider_options --no-default-features --no-fail-fast`
  - `cargo check -p siumai-core --no-default-features`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard openai_chat_request_conversion_source_does_not_read_legacy_provider_metadata_fields --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard openai_provider_metadata_source_does_not_read_request_provider_options openai_compatible_request_transformer_source_does_not_read_legacy_provider_metadata_fields openai_compatible_chat_response_source_does_not_emit_request_provider_options openai_compatible_streaming_source_does_not_emit_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_response_transformer_source_does_not_emit_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai completion_request_source_does_not_read_legacy_provider_metadata_fields completion_response_and_stream_source_do_not_emit_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-provider-openai --no-default-features --features openai`
  - `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_parse_response_content_source_does_not_emit_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-protocol-anthropic --no-default-features --features anthropic-standard`
  - `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_response_content_source_does_not_emit_request_provider_options --no-fail-fast`
  - `cargo nextest run -p siumai-bridge --no-default-features --features openai,anthropic,google response_and_stream_bridge_sources_do_not_emit_request_provider_options --no-fail-fast`
  - `cargo check -p siumai-bridge --no-default-features --features openai,anthropic,google`
  - `cargo fmt --package siumai-core --package siumai-protocol-anthropic --check`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_json_stream_executor_does_not_handle_provider_maps --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic --lib standards::anthropic::thinking --no-default-features --features anthropic-standard --no-fail-fast`
  - `cargo fmt --package siumai-core --check`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_family_contract_and_tooling_sources_do_not_handle_provider_maps core_sample_streaming_middleware_only_initializes_empty_provider_metadata core_provider_options_parser_stays_request_only_and_provider_agnostic --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo fmt --package siumai-core --check`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_provider_agnostic_docs_do_not_describe_core_as_openai_compatible core_source_does_not_use_provider_model_fixture_literals --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo fmt --package siumai-provider-anthropic --package siumai-provider-openai --package siumai-provider-google-vertex --package siumai-provider-minimaxi --check`
  - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::spec::tests::anthropic_spec_request_option_routing_does_not_read_response_metadata providers::anthropic::client::tests::anthropic_client_middleware_request_option_checks_do_not_read_response_metadata --no-default-features --features anthropic --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --lib providers::openai::spec::tests::openai_spec_request_option_routing_does_not_read_response_metadata providers::openai::spec::tests::openai_spec_provider_metadata_key_selection_does_not_read_request_options providers::openai::client::tests::openai_client_default_provider_option_merging_does_not_read_response_metadata --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --lib providers::anthropic_vertex::client::tests::vertex_anthropic_client_default_provider_options_do_not_read_response_metadata --no-default-features --features google-vertex --no-fail-fast`
  - `cargo nextest run -p siumai-provider-minimaxi --lib providers::minimaxi::client::tests::minimaxi_client_request_option_merging_does_not_read_response_metadata --no-default-features --features minimaxi --no-fail-fast`
  - `cargo nextest run -p siumai-provider-gemini --lib providers::gemini::client --no-default-features --features google --no-fail-fast`
  - `cargo fmt --package siumai --check`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_macros_only_create_request_side_empty_provider_options facade_audio_and_structured_helpers_do_not_read_request_provider_options facade_video_metadata_projection_avoids_legacy_request_provider_options --no-default-features --features openai,anthropic,google --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-default-features --features openai,anthropic,google --no-fail-fast`

- `siumai-registry/src/registry/entry/boundary_tests.rs` now rejects both generic
  `compat_*_client_with_ctx(...)` calls and `.as_*_capability()` downcasts inside primary stable
  family execution sections.
- Whole-handle guards cover completion, embedding, rerank, and video handles; section-level guards
  cover language chat execution, image generation, speech synthesis, and transcription because those
  files still intentionally contain extension-only compatibility branches.
- Extension-only gaps currently remain behind explicit compatibility or extension naming:
  image edit/variation, speech/transcription streaming/extras, file management, skills, and music.
- Verified command:
  - `cargo nextest run -p siumai-registry registry::entry::boundary_tests::stable_registry_handles_do_not_use_compat_client_paths_for_primary_family_execution --no-default-features --no-fail-fast`

## FSCBC-M4 - Core Provider-Specific Residue Removed

Acceptance criteria:

- `siumai-core` no longer owns provider-specific stream bridge logic.
- Provider-specific custom event serialization lives in bridge, protocol, or provider crates.
- Provider defaults and hosted tool factories live in provider or registry ownership unless they are
  proven provider-agnostic.
- Boundary tests prevent provider-specific code from returning to `siumai-core`.

Status: in progress

Notes:

- `siumai-core::core_provider_boundary_test` now includes a strict source guard that rejects
  provider-specific bridge targets, bridge target catalogs, and stream custom-event serializers
  anywhere under `siumai-core/src`.
- `OpenAiResponsesStreamPartsBridge` moved out of `siumai-core/src/streaming/bridge.rs` and now
  lives in `siumai-bridge::stream` behind the `openai` feature.
- `BridgeTarget`, `BridgeOptions`, `BridgeReport`, bridge contexts, customization hooks, primitive
  remappers, and loss-policy traits moved from `siumai-core/src/bridge/mod.rs` to
  `siumai-bridge/src/contracts.rs` and remain available through the bridge crate's top-level
  exports.
- The bridge-specific OpenAI Responses SSE integration checks moved from
  `siumai-protocol-openai` into `siumai-bridge`, so the OpenAI protocol crate no longer reaches
  through its private `siumai_core::streaming` alias for cross-protocol bridge behavior.
- `StreamPartNamespace` and `TypedStreamPart::to_protocol_custom_event` were removed from
  `siumai-core`; OpenAI, Anthropic, and Gemini custom-event prefix mapping now lives in each
  protocol serializer.
- Provider-hosted tool constructors moved out of `siumai-core::hosted_tools`. OpenAI, Anthropic,
  and Google/Gemini constructors now live in the matching protocol crates and are re-exported by
  provider crates plus the stable `siumai::hosted_tools::*` facade path.
- `siumai-core::core_provider_boundary_test` rejects a returning `src/hosted_tools` directory or
  `pub mod hosted_tools` declaration.
- `siumai::facade_architecture_boundary_test` rejects re-exporting hosted tool constructors from
  `siumai-core` and requires the protocol-owned re-export chain.
- Provider-specific default URL/model/timeout constants were removed from `siumai-core::defaults`.
  Core defaults now cover provider-agnostic HTTP, timeout, streaming, model parameter, and profile
  values only; provider crates and registry factories own provider endpoint/model defaults.
- `siumai-core::core_provider_boundary_test` rejects a returning `defaults::providers` module and
  known provider default fragments in `siumai-core/src/defaults.rs`.
- Vertex URL helpers moved from `siumai-core::auth::vertex` to
  `siumai-provider-google-vertex::auth::vertex`. The facade path
  `siumai::experimental::auth::vertex` remains available behind the `google-vertex` feature and now
  re-exports the provider-owned module.
- `siumai-core::core_provider_boundary_test` rejects a returning core `auth::vertex` module and
  known Vertex URL helper fragments under `siumai-core/src`.
- Google Cloud ADC and service-account token provider implementations moved from `siumai-core::auth`
  to `siumai-provider-google-vertex::auth::{adc,service_account}`. `siumai-core::auth` now keeps
  only the provider-agnostic `TokenProvider` contract and `StaticTokenProvider` helper.
- The facade path `siumai::experimental::auth::{adc,service_account}` remains available behind the
  `gcp` feature and now re-exports provider-owned modules.
- `siumai-core::core_provider_boundary_test` rejects returning core ADC/service-account modules and
  known Google Cloud auth implementation fragments under `siumai-core/src`.
- Provider-specific `HookBuilder` body presets were removed from `siumai-core`. The experimental
  `with_openai_base()` and `with_anthropic_base()` shortcuts had no production call sites and
  encoded protocol-shaped request bodies in the provider-agnostic runtime crate. Custom hooks should
  now provide an explicit `with_chat_body_builder(...)` closure or use provider/protocol-owned
  helpers when such helpers exist.
- `siumai-core::core_provider_boundary_test` rejects returning provider-specific HookBuilder body
  presets.
- `ProviderSpec` no longer contains core-owned provider-shaped fallback endpoint strings. Default
  fallible route methods now return an explicit unsupported-route error instead of constructing
  OpenAI-style paths from `ProviderContext::base_url`.
- Core executors now use fallible `try_chat_url`, `try_embedding_url`, `try_image_url`,
  `try_image_edit_url`, `try_image_variation_url`, and `try_rerank_url` route resolution.
  `ProviderSpec` no longer exposes the historical string-returning `*_url(...)` hook methods.
- `ProviderSpec` default fallible route methods now return `UnsupportedOperation` directly instead
  of bridging through historical string-returning hooks.
- Custom provider examples and facade retry test specs now implement `try_*_url(...)` directly, so
  extension-facing guidance follows the provider-owned fallible route contract.
- `siumai-core::core_provider_boundary_test` rejects returning legacy route fallback helpers or
  provider-shaped endpoint strings to `ProviderSpec`.
- Gemini protocol/provider route specs now implement only fallible route methods. Their historical
  explicit string-returning route implementations were removed.
- `siumai-core::core_provider_boundary_test::migrated_gemini_route_specs_do_not_reintroduce_legacy_string_hooks`
  guards the migrated Gemini files against direct legacy route hook definitions.
- OpenAI protocol standard route specs (`chat`, `embedding`, `image`, and `rerank`) now implement
  only fallible route methods; their explicit string-returning route implementations were removed.
- `siumai-core::core_provider_boundary_test::migrated_openai_protocol_route_specs_do_not_reintroduce_legacy_string_hooks`
  guards the migrated OpenAI protocol standard files against direct legacy route hook definitions.
- OpenAI provider specs (`OpenAiSpec` and `OpenAiSpecWithRerank`) now implement only fallible route
  methods; their explicit string-returning route implementations were removed.
- `siumai-core::core_provider_boundary_test::migrated_openai_provider_route_specs_do_not_reintroduce_legacy_string_hooks`
  guards the migrated OpenAI provider spec file against direct legacy route hook definitions.
- OpenAI-compatible protocol spec (`OpenAiCompatibleSpecWithAdapter`) now implements only fallible
  route methods and applies query/request URL settings inside `try_*_url(...)`.
- The OpenAI protocol route guard now also covers the OpenAI-compatible protocol spec and its
  model listing/retrieve routes.
- Anthropic protocol/provider route specs, including Anthropic-on-Vertex, now implement only
  fallible route methods. Their explicit string-returning route implementations were removed.
- `siumai-core::core_provider_boundary_test::migrated_anthropic_route_specs_do_not_reintroduce_legacy_string_hooks`
  guards the migrated Anthropic route files against direct legacy route hook definitions.
- DeepSeek and Groq OpenAI-compatible provider wrappers now delegate only through
  `try_chat_url(...)`, without direct `chat_url(...)` implementations.
- Cohere native chat/embedding/rerank route specs and TogetherAI rerank route specs now implement
  only fallible route methods.
- `siumai-core::core_provider_boundary_test::migrated_openai_compatible_provider_route_specs_do_not_reintroduce_legacy_string_hooks`
  and
  `siumai-core::core_provider_boundary_test::migrated_rerank_provider_route_specs_do_not_reintroduce_legacy_string_hooks`
  guard the migrated DeepSeek/Groq/Cohere/TogetherAI route files against direct legacy route hook
  definitions.
- Ollama chat/model-listing/embedding specs and MiniMaxi chat/image specs now implement only
  fallible route methods.
- `siumai-core::core_provider_boundary_test::migrated_local_and_multi_surface_provider_route_specs_do_not_reintroduce_legacy_string_hooks`
  guards the migrated Ollama and MiniMaxi route files against direct legacy route hook definitions.
- Amazon Bedrock chat/embedding/image/rerank specs and Azure OpenAI chat/embedding/image specs now
  implement only fallible route methods.
- `siumai-core::core_provider_boundary_test::migrated_bedrock_and_azure_route_specs_do_not_reintroduce_legacy_string_hooks`
  guards the migrated Bedrock and Azure route files against direct legacy route hook definitions.
- Google Vertex generative AI, embedding, Gemini image, and Imagen route specs now implement only
  fallible route methods and no longer call protocol legacy string hooks.
- `siumai-core::core_provider_boundary_test::migrated_google_vertex_route_specs_do_not_reintroduce_legacy_string_hooks`
  guards the migrated Google Vertex route files against direct legacy route hook definitions and
  calls.
- Workspace provider/protocol specs that own route hooks now also own fallible route resolution.
  OpenAI, Anthropic, Gemini, Azure OpenAI, Groq, DeepSeek, Ollama, Cohere, Amazon Bedrock,
  TogetherAI, MiniMaxi, and Google Vertex implement `try_*_url(...)` directly or delegate to the
  protocol-owned fallible route hook.
- `siumai-core::core_provider_boundary_test::provider_specs_do_not_reintroduce_legacy_string_route_hooks`
  scans `siumai-provider-*` and `siumai-protocol-*` `ProviderSpec` impls and rejects historical
  string-returning route hooks outright.
- Production request paths for model listing/retrieve and direct chat helpers now use fallible
  `try_*_url(...)` route resolution in OpenAI, OpenAI-compatible, Anthropic, Gemini, Ollama, and
  Anthropic-on-Vertex code.
- `siumai-core::core_provider_boundary_test::production_request_paths_use_fallible_provider_routes`
  guards the migrated production files against direct `*_url(...)` calls.
- Facade fixture tests and provider/protocol route assertions now call `try_*_url(...).unwrap()`
  instead of asserting through the historical string-returning hooks.
- `siumai-core::core_provider_boundary_test::route_fixture_tests_use_fallible_provider_routes`
  guards facade fixture tests and provider/protocol test modules against direct `*_url(...)` calls.
- Core executor test fixtures and embedded examples now define/call only fallible route hooks, so
  executor-local guidance no longer teaches `chat_url(...)`, `embedding_url(...)`, image route, or
  rerank route hooks as the primary route contract.
- `siumai-core::core_provider_boundary_test::core_executor_tests_and_docs_use_fallible_route_hooks`
  guards `siumai-core/src/execution` Rust sources against direct legacy route definitions or calls.
- `docs/migration/migration-0.11.0-beta.7.md` now includes a custom `ProviderSpec` route migration
  note that maps removed string-returning `*_url(...)` hooks to fallible `try_*_url(...)` hooks.
- `siumai/examples/06-extensibility/README.md` now teaches `try_chat_url() -> fallible API endpoint`
  in the ProviderSpec pattern diagram.
- Provider-agnostic `siumai-core` comments and examples no longer present completion endpoints,
  stream transformers, standards helpers, builder helper presets, tracing examples,
  custom-provider examples, or URL helper tests as OpenAI-compatible core behavior.
- `siumai-core::core_provider_boundary_test` guards those docs/examples against reintroducing
  OpenAI-compatible wording or concrete OpenAI API URLs.
- `siumai-core::StreamFactory` no longer owns provider/protocol SSE terminal marker recognition.
  `SseEventConverter` and `StreamChunkTransformer` now expose `is_stream_end_event(&Event) -> bool`,
  and OpenAI-compatible, OpenAI Responses, OpenAI legacy completions, Anthropic, Gemini, xAI, and
  MiniMaxi wrapper paths delegate explicit marker recognition to protocol/provider-owned
  converters.
- `SseJsonStreamConfig::new(...)` now defaults to no protocol-specific done markers; OpenAI SSE
  JSON helpers configure `[DONE]` explicitly. Core HTTP tracing now finalizes from the synthetic
  `siumai_stream_end` event instead of inspecting provider payload text.
- `siumai-core::streaming::encoder` now has a source guard preventing provider option or provider
  metadata maps from entering core stream encoding and synthetic terminal event construction.
- `siumai-core` reasoning extraction middleware no longer owns concrete provider/model tag routing.
  It extracts response metadata from generic `thinking` / `reasoning` keys and is source-guarded
  against reintroducing provider names in core middleware.
- `siumai-core` system-message warning middleware no longer probes concrete provider fallback
  namespaces. It reads only the provider option namespace supplied by the caller, and automatic
  middleware wiring passes through the configured provider namespace.
- `siumai-core::execution::middleware::lm::language_model` middleware docs and tests now use
  provider-neutral model/provider IDs with a source guard.
- `siumai-core::execution::executors::image` test fixtures now use provider-neutral namespace and
  model names instead of concrete provider examples.
- `siumai-core::utils::chat_request` and `siumai-core::execution::http::headers` tests now use
  provider-neutral namespaces/header names with source guards while production code remains generic.
- `siumai-core::execution::executors::files` now source-guards the upload runtime path against
  concrete provider literals or response metadata reads.
- `siumai-core::core_provider_boundary_test::core_stream_factory_does_not_own_provider_sse_end_markers`
  guards `StreamFactory`, SSE JSON defaults, and HTTP tracing against reintroducing core-owned
  provider/protocol SSE marker checks.
- Core streaming tests/docs now use provider-neutral fixture model names and URLs. Provider metadata
  key assertions remain only as generic carrier-shape coverage.
- `siumai-core::core_provider_boundary_test::core_streaming_runtime_tests_do_not_use_provider_model_fixtures`
  guards streaming runtime sources against reintroducing provider model or endpoint fixtures.
- `siumai-core::retry_api` generic HTTP error classification now avoids provider-specific
  request-id headers in its fallback classifier. Provider-specific HTTP error/request-id handling
  remains provider-owned through `ProviderSpec::classify_http_error(...)`.
- Core docs/tests and carrier fixtures now use provider-neutral namespaces, model IDs, URLs,
  custom event kinds, and provider references across transformer, trait, telemetry, structured
  output, UI conversion, streaming, retry, and utility modules.
- A broad `siumai-core/src` source scan for concrete provider/model fixtures now returns no
  matches for the audited provider/model literal pattern.
- `siumai-core::core_provider_boundary_test::core_source_does_not_use_provider_model_fixture_literals`
  makes that scan a case-insensitive source guard across core production code, embedded tests, and
  rustdoc examples.
- Verified command for this guard:
  - `cargo check -p siumai-core --no-default-features`
  - `cargo check -p siumai-core --features gcp --no-default-features`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib streaming::sse_json --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib streaming::encoder stream_encoder_source_does_not_read_provider_option_or_metadata_maps --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib execution::middleware::presets::extract_reasoning::tests --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib execution::middleware::presets::system_message_mode_warning::tests --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib execution::middleware::auto::tests --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib execution::middleware::lm::language_model::tests --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test openai_chat_messages_fixtures_alignment_test --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib execution::executors::image::tests --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib utils::chat_request::tests --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib execution::http::headers::tests --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib execution::executors::files::tests::files_executor_upload_path_stays_provider_agnostic --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib streaming --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test streaming_tests factory_injection --features openai --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic -p siumai-protocol-gemini --no-default-features --features "siumai-protocol-anthropic/anthropic siumai-protocol-gemini/google" --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai --no-fail-fast`
  - `cargo check -p siumai-provider-xai -p siumai-provider-minimaxi --no-default-features --features "siumai-provider-xai/xai siumai-provider-minimaxi/minimaxi"`
  - `cargo fmt --package siumai-core --package siumai-protocol-openai --package siumai-protocol-anthropic --package siumai-protocol-gemini --package siumai-provider-openai --package siumai-provider-xai --package siumai-provider-minimaxi --package siumai --check`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_executor_tests_and_docs_use_fallible_route_hooks --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test route_fixture_tests_use_fallible_provider_routes --no-default-features --no-fail-fast`
  - `cargo check -p siumai-core --no-default-features`
  - `cargo fmt --package siumai-core --check`
  - `cargo check -p siumai --example custom_provider_spec --example complete_custom_provider --example testing_executors --features openai --no-default-features`
  - `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock,xai --no-default-features`
  - `cargo check -p siumai-provider-anthropic -p siumai-provider-gemini -p siumai-provider-google-vertex -p siumai-provider-ollama -p siumai-provider-openai -p siumai-provider-openai-compatible --no-default-features --features "siumai-provider-anthropic/anthropic siumai-provider-gemini/google siumai-provider-google-vertex/google-vertex siumai-provider-ollama/ollama siumai-provider-openai/openai siumai-provider-openai-compatible/openai-standard"`
  - `rg -n "chat_url\\(\\)|embedding_url\\(\\)|image_url\\(\\)|rerank_url\\(\\)|models_url\\(\\)|model_url\\(\\)|fn chat_url\\(" docs siumai\\examples -g "*.md" -g "*.rs"`
  - `cargo check -p siumai --example custom_provider_spec --example complete_custom_provider --example testing_executors --features openai --no-default-features`
  - `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock,xai --no-default-features`
  - `cargo nextest run -p siumai --test retry_401_tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock,xai --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini -p siumai-provider-gemini --no-default-features --features "siumai-provider-gemini/google" --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic -p siumai-provider-anthropic --no-default-features --features "siumai-provider-anthropic/anthropic" --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex --no-fail-fast`
  - `cargo fmt --package siumai-core --package siumai-protocol-anthropic --package siumai-provider-anthropic --package siumai-provider-google-vertex --check`
  - `cargo nextest run -p siumai-provider-deepseek -p siumai-provider-groq -p siumai-provider-cohere -p siumai-provider-togetherai --no-default-features --features "siumai-provider-deepseek/deepseek siumai-provider-groq/groq siumai-provider-cohere/cohere siumai-provider-togetherai/togetherai" --no-fail-fast`
  - `cargo fmt --package siumai-core --package siumai-provider-deepseek --package siumai-provider-groq --package siumai-provider-cohere --package siumai-provider-togetherai --check`
  - `cargo nextest run -p siumai-provider-ollama -p siumai-provider-minimaxi --no-default-features --features "siumai-provider-ollama/ollama siumai-provider-minimaxi/minimaxi" --no-fail-fast`
  - `cargo fmt --package siumai-core --package siumai-provider-ollama --package siumai-provider-minimaxi --check`
  - `cargo nextest run -p siumai-provider-amazon-bedrock -p siumai-provider-azure --no-default-features --features "siumai-provider-amazon-bedrock/bedrock siumai-provider-azure/azure" --no-fail-fast`
  - `cargo fmt --package siumai-core --package siumai-provider-amazon-bedrock --package siumai-provider-azure --check`
  - `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex --no-fail-fast`
  - `cargo fmt --package siumai-core --package siumai-provider-google-vertex --check`
  - `cargo nextest run -p siumai-core --no-default-features --no-fail-fast`
  - `cargo test -p siumai-core --no-default-features --doc hook_builder`
  - `rg -n '"(anthropic|gemini|google|openai|minimaxi|bedrock|azure|deepseek|groq|ollama|cohere|togetherai|xai)"|anthropic-|gemini-|openai-|gpt-|claude-|api\.anthropic|api\.openai|generativelanguage\.googleapis|aiplatform\.googleapis' siumai-core/src -g "*.rs"`
  - `rg -n -i 'openai|anthropic|gemini|google|azure|bedrock|groq|ollama|cohere|deepseek|minimaxi|togetherai|xai|gpt-|claude-|api\.anthropic|api\.openai|generativelanguage\.googleapis|aiplatform\.googleapis|x-openai-request-id|x-goog-request-id' siumai-core/src -g "*.rs"`
  - `cargo fmt --package siumai-core --check`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_source_does_not_use_provider_model_fixture_literals --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --lib --no-default-features --no-fail-fast`
  - `git diff --check -- <touched siumai-core files>`
  - `cargo check -p siumai-provider-google-vertex --no-default-features --features google-vertex`
  - `cargo check -p siumai-provider-google-vertex --no-default-features --features google-vertex,gcp`
  - `cargo check -p siumai-provider-openai-compatible --no-default-features --features openai-standard`
  - `cargo check -p siumai-registry --tests --features google-vertex --no-default-features`
  - `cargo check -p siumai-registry --tests --features google-vertex,gcp --no-default-features`
  - `cargo check -p siumai --tests --features google-vertex,gcp --no-default-features`
  - `cargo nextest run -p siumai-provider-google-vertex --test gcp_auth_alignment_test --no-default-features --features google-vertex,gcp --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex,gcp --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai-compatible --no-default-features --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai --test service_account_provider_test --features google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test vertex_maas_openai_compat_url_alignment_test --features google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test google_vertex_builder_alignment_test --features google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test anthropic_vertex_builder_alignment_test --features google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-bridge --no-default-features --features openai,anthropic,google --no-fail-fast`
  - `cargo check -p siumai-protocol-openai --no-default-features`
  - `cargo check -p siumai-protocol-anthropic --no-default-features`
  - `cargo check -p siumai-protocol-gemini --no-default-features`
  - `cargo check -p siumai-provider-openai --no-default-features --features openai`
  - `cargo check -p siumai-provider-anthropic --no-default-features --features anthropic`
  - `cargo check -p siumai-provider-gemini --no-default-features --features google`
  - `cargo check -p siumai-provider-google-vertex --no-default-features --features google-vertex`
  - `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex --no-default-features`
  - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai --no-fail-fast`
  - `cargo check -p siumai-provider-amazon-bedrock -p siumai-provider-azure -p siumai-provider-groq -p siumai-provider-deepseek -p siumai-provider-ollama -p siumai-provider-cohere -p siumai-provider-togetherai -p siumai-provider-minimaxi -p siumai-provider-google-vertex --no-default-features --features "siumai-provider-amazon-bedrock/bedrock siumai-provider-azure/azure siumai-provider-groq/groq siumai-provider-deepseek/deepseek siumai-provider-ollama/ollama siumai-provider-cohere/cohere siumai-provider-togetherai/togetherai siumai-provider-minimaxi/minimaxi siumai-provider-google-vertex/google-vertex"`
  - `cargo nextest run -p siumai-provider-amazon-bedrock -p siumai-provider-azure -p siumai-provider-groq -p siumai-provider-deepseek -p siumai-provider-ollama -p siumai-provider-cohere -p siumai-provider-togetherai -p siumai-provider-minimaxi -p siumai-provider-google-vertex --no-default-features --features "siumai-provider-amazon-bedrock/bedrock siumai-provider-azure/azure siumai-provider-groq/groq siumai-provider-deepseek/deepseek siumai-provider-ollama/ollama siumai-provider-cohere/cohere siumai-provider-togetherai/togetherai siumai-provider-minimaxi/minimaxi siumai-provider-google-vertex/google-vertex" --no-fail-fast`
  - `cargo check -p siumai-provider-anthropic -p siumai-provider-gemini -p siumai-provider-google-vertex -p siumai-provider-ollama -p siumai-provider-openai -p siumai-provider-openai-compatible --no-default-features --features "siumai-provider-anthropic/anthropic siumai-provider-gemini/google siumai-provider-google-vertex/google-vertex siumai-provider-ollama/ollama siumai-provider-openai/openai siumai-provider-openai-compatible/openai-standard"`
  - `cargo nextest run -p siumai-provider-anthropic -p siumai-provider-gemini -p siumai-provider-google-vertex -p siumai-provider-ollama -p siumai-provider-openai -p siumai-provider-openai-compatible --no-default-features --features "siumai-provider-anthropic/anthropic siumai-provider-gemini/google siumai-provider-google-vertex/google-vertex siumai-provider-ollama/ollama siumai-provider-openai/openai siumai-provider-openai-compatible/openai-standard" --no-fail-fast`
  - `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock --no-default-features`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test bedrock_chat_stream_alignment_test --features bedrock --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test mock_api_tests minimaxi --features minimaxi --no-default-features --no-fail-fast`
  - `cargo fmt --package siumai-provider-amazon-bedrock --package siumai-provider-azure --package siumai-provider-groq --package siumai-provider-deepseek --package siumai-provider-ollama --package siumai-provider-cohere --package siumai-provider-togetherai --package siumai-provider-minimaxi --package siumai-provider-google-vertex --check`
  - `cargo fmt --package siumai --check`
  - `cargo fmt --package siumai-core --check`
  - `git diff --check`

## FSCBC-M5 - `LlmClient` Compatibility Boundary Tightened

Acceptance criteria:

- Stable family handles do not call `as_*_capability()` or generic compatibility clients for primary
  execution.
- Extension-only gaps are named as compatibility or provider extension surfaces.
- Registry tests prove stable family paths use native family model construction.
- Public docs describe `LlmClient` as compatibility-only where relevant.

Status: complete

Notes:

- Stable family handle sources remain guarded against `.as_*_capability()` downcasts and
  `compat_*_client*` primary execution paths.
- `siumai-core::ClientWrapper` no longer exposes provider-named constructor aliases; advanced
  boxed-client wrapping uses the provider-agnostic `ClientWrapper::new(...)`.
- Extension-only gaps remain explicitly classified: image edit/variation,
  speech/transcription streaming and extras, file management, skills, and music.
- `docs/migration/migration-0.11.0-beta.7.md` now documents generic `LlmClient` factory paths as
  migration/extension-only compatibility and points new code to family methods such as
  `language_model_text_with_ctx(...)`.
- `docs/architecture/public-surface.md` and `docs/architecture/registry-without-builtins.md` now
  describe registry construction as family-first and keep generic `LlmClient` construction behind
  explicit `compat_*_client(...)` / `compat_*_client_with_ctx(...)` methods.
- `siumai-registry::factory_architecture_boundary_test::public_docs_classify_generic_llm_client_factory_paths_as_migration_only`
  guards the public docs against treating generic `LlmClient` factory paths as the default route.
- Verified commands:
  - `cargo fmt --package siumai-registry --check`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_client_wrapper_does_not_expose_provider_specific_constructors --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test debug_implementation_test --no-default-features --features openai --no-fail-fast`

## FSCBC-M6 - Facade And Registry Exports Narrowed

Acceptance criteria:

- Stable preludes exclude compatibility-only names unless explicitly documented.
- Registry exports do not mirror broad core/provider implementation modules.
- Compatibility-only names live under `compat` or documented `experimental` paths.
- Architecture and migration docs are updated for public surface changes.

Status: in progress

Notes:

- `siumai::facade_architecture_boundary_test` already rejects compatibility construction aliases
  from `prelude::unified`.
- `siumai::facade_architecture_boundary_test` now also audits the remaining
  `prelude::unified` compatibility/runtime aliases. After the deprecated AI SDK parity alias move,
  only `CancelHandle` remains on that allowlist because it is an active runtime cancellation type.
- The historical root `siumai::types::*` catch-all path has been removed. The compatibility
  catch-all type namespace is now explicit under `siumai::compat::types::*` and
  `siumai::prelude::compat::types::*`, and `compatibility-audit.md` records it as non-target stable
  surface.
- `siumai/src/lib.rs`, `siumai/src/compat.rs`, and `docs/architecture/public-surface.md` now use
  the same language for that path: explicit compatibility, not the stable facade target.
- `prelude::unified` no longer mirrors `siumai_core::streaming::*`. It keeps stable stream
  consumption types, while low-level converters, factories, encoders, and typed bridge stream parts
  are available through `siumai::experimental::streaming` for advanced integrations.
- Retry API controls are now scoped to `siumai::retry_api::*` instead of direct
  `prelude::unified::*` names. This keeps retry policy/runtime controls available while preventing
  future retry internals from entering the stable family prelude by accident.
- The stable unified prelude no longer mirrors the whole runtime `tooling` module. AI SDK-style
  tool helper names remain direct prelude imports, but broader runtime tooling APIs are explicit
  `siumai::tooling::*` imports.
- `prelude::unified` no longer exports execution middleware internals. `LanguageModelMiddleware`
  and related middleware builders remain available from
  `siumai::experimental::execution::middleware::*` for custom provider and advanced integration
  code.
- `prelude::unified::registry::*` now exports `BuildContext` and `ProviderBuildOverrides` so
  custom factory implementers can use the stable family-first registry trait signatures directly.
- `ProviderFactory` is no longer exported from top-level `prelude::unified::*`; custom factory
  code should import it from the scoped `prelude::unified::registry::*` surface.
- The unused root `siumai::registry_global` alias has been removed; callers use
  `registry::global()` or `siumai::prelude::unified::registry::global()` explicitly.
- The unused facade root `siumai::provider_catalog::*` mirror has been removed; advanced catalog
  code imports `siumai_registry::provider_catalog::*` explicitly while normal application code uses
  registry family handles.
- The OpenAI-compatible provider-list macro is no longer re-exported from the facade root; registry
  or provider glue imports it directly from `siumai_provider_openai_compatible`.
- File and skill upload helper types/functions are no longer exported as direct
  `prelude::unified::*` names. Stable imports use `siumai::files::*`, `siumai::skills::*`, or the
  root helper functions instead.
- The facade file-upload main flow now delegates provider-specific compatibility warnings to a
  helper and is source-guarded against provider-specific policy literals returning to the generic
  upload path.
- The facade skill-upload payload adapter is source-guarded to keep provider-specific skill policy
  in provider APIs, and the skill helper alignment test imports stable data carriers explicitly
  through `siumai::types`.
- OpenAI and Gemini protocol file transformers are now source-guarded so request-side upload
  helpers and response-side file object conversion stay in separate directional sections.
- Anthropic provider-owned file and skill upload resources are source-guarded so unsupported
  request `provider_options` remain explicit request-side compatibility choices while provider
  metadata/result projection remains response-only.
- Anthropic thinking replay is guarded as an explicit cross-step exception: request-only thinking
  helpers must not read response metadata, while `assistant_message_with_thinking_metadata(...)`
  remains the scoped response-metadata-to-request-options bridge for reasoning replay.
- Anthropic structured-output helper and OpenAI structured-output request config now have
  request-side source guards; the OpenAI response validator is kept as an explicit response-side
  parser with focused behavior coverage.
- Anthropic and Anthropic-on-Vertex request option extension helpers now have request-only source
  guards, keeping typed provider-option convenience APIs from reading response metadata.
- OpenAI-compatible request routing and provider option normalization now rejects response metadata
  reads at source level while keeping runtime-provider option behavior covered by the compat spec
  module tests.
- Bedrock extension helpers are split by guard: normal request helpers must stay request-only,
  while `assistant_message_with_reasoning_metadata(...)` remains the explicit cross-step exception
  for projecting Bedrock reasoning `signature` / `redactedData` into request provider options.
- Gemini chat/embedding/image/video and MiniMaxi chat/structured/thinking/TTS/video request
  extension helpers are source-guarded so provider-owned request builders cannot start reading
  response metadata.
- Vertex embedding, Imagen, and video request option extension helpers now have the same
  request-only source guard.
- `parse_json_event_stream(...)` is no longer exported as a direct `prelude::unified::*` name. The
  explicit root helper remains for callers that need JSON/SSE parsing.
- Deprecated AI SDK parity names (`CallSettings`, `Experimental_*` result aliases,
  `experimental_filter_active_tools`, and `step_count_is`) moved out of `prelude::unified` and into
  `siumai::compat` / `prelude::compat`. Stable code should use the non-experimental names.
- `StreamingToolCall*` helper aliases moved out of `prelude::unified` and into `siumai::compat` /
  `prelude::compat`. The historical root aliases were removed, so the stable facade root and
  Vercel-aligned prelude no longer export this low-level provider-utils helper family.
- Provider-specific builder construction is now explicitly classified through
  `siumai::compat::Provider` and `siumai::prelude::compat::Provider`. The historical root
  `siumai::Provider` alias has been removed, so migration-oriented builder imports must use the
  explicit compat path. The implementation body now lives under `siumai::compat`;
  `siumai::compat::{Siumai,SiumaiBuilder}` also binds directly to registry-owned types, and the
  root `siumai::provider::*` builder-era compatibility shim has been removed. Facade production
  code now routes upload helper impls through `crate::compat::Siumai`, and facade tests/examples now
  use explicit `siumai::compat` or stable registry paths instead of treating `siumai::provider::*`
  as the default construction import. Ordinary tests, the large provider public-path parity suite,
  and public-surface import coverage all use `siumai::compat::Provider`; no test/example allowlist
  remains for root `siumai::Provider` or root `siumai::provider::*`.
- The root `siumai::builder::*` builder-base shim has also been removed. The explicit
  `siumai::compat::builder::*` path now binds directly to `siumai-core`, and
  `siumai::compat::Provider` imports `siumai_core::builder::BuilderBase` directly instead of
  routing through a facade builder module.
  Runnable examples are also guarded against teaching the removed root alias.
- Provider extension package helpers that return `SiumaiBuilder` now bind directly to
  `siumai-registry`'s builder type instead of routing through the facade `provider` shim.
  Provider extension helpers also avoid the facade root `Provider` alias and use the explicit
  compat path when centralized builder construction is still needed.
- The facade boundary test now prevents legacy root `siumai::traits`, `siumai::error`, and
  `siumai::streaming` modules from returning as broad facade re-exports.
- Facade tests no longer use `prelude::unified::*` as a backdoor for low-level streaming internals.
  Transcoding and stream alignment fixtures import converters/byte streams/processors explicitly
  from `siumai::experimental::streaming`; file upload helper tests import `FileUploadProvider`
  from `siumai::files`.
- `siumai-registry` root now keeps only the small custom-factory contract surface and no longer
  mirrors broad `siumai-core` implementation/family modules such as `embedding`, `image`, `video`,
  `retry_api`, `hosted_tools`, or `custom_provider`.
- `siumai-registry::factory_architecture_boundary_test` guards the narrowed registry root surface.
- `siumai::facade_architecture_boundary_test::content_part_provider_map_audit_covers_high_value_production_hits`
  now turns the Track C direct `ContentPart` / provider-map scan into a manifest coverage guard.
  New high-value production hits must be added to `content-part-construction-audit.md`, explicitly
  classified as low-priority/false-positive buckets, or removed before the facade boundary suite
  passes.
- `siumai::facade_architecture_boundary_test::stable_unified_prelude_keeps_non_family_extension_types_scoped`
  now guards the Track F split between the stable family prelude and non-family extension
  capabilities. `prelude::unified` must not directly export extension capability traits or
  extension-only request/response types; those remain under `siumai::extensions::*` and
  `prelude::extensions::*`.
- `siumai::facade_architecture_boundary_test::broad_facade_types_path_is_explicit_compat_only`
  prevents the root `siumai::types::*` path from returning, keeps the broad type namespace under
  explicit compat paths, and prevents the stable unified prelude from regressing to a glob mirror of
  `siumai_core::types::*`. Tier A type exports remain curated.
- Verified commands:
  - `cargo check -p siumai-registry --no-default-features`
  - `cargo check -p siumai-registry --example no_builtins_custom_factory --no-default-features`
  - `cargo check -p siumai-registry --tests --features openai,anthropic,google --no-default-features`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
  - `cargo nextest run -p siumai --test integration_tests --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test files_upload_helper_alignment_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --lib files::tests::upload_via_file_management_keeps_provider_policy_delegated_to_helpers --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai --test files_upload_helper_alignment_test --no-default-features --features openai,anthropic,google,minimaxi --no-fail-fast`
  - `cargo nextest run -p siumai --lib skills::tests::upload_helper_keeps_provider_policy_delegated_to_api --no-default-features --features openai,anthropic --no-fail-fast`
  - `cargo nextest run -p siumai --test skills_upload_helper_alignment_test --no-default-features --features openai,anthropic --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --lib standards::openai::files::tests::openai_files_request_and_response_transformers_keep_maps_directional --no-default-features --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini --lib standards::gemini::transformers::files::files_tests::gemini_files_request_and_response_paths_keep_maps_directional --no-default-features --features google --no-fail-fast`
  - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::files::tests::anthropic_files_request_and_response_paths_keep_maps_directional --no-default-features --features anthropic --no-fail-fast`
  - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::skills::tests::anthropic_skills_request_and_response_paths_keep_maps_directional --no-default-features --features anthropic --no-fail-fast`
  - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::thinking::tests --no-default-features --features anthropic --no-fail-fast`
  - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::structured_output --no-default-features --features anthropic --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --lib providers::openai::structured_output --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::request_options --no-default-features --features anthropic --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --lib providers::anthropic_vertex::ext::request_options --no-default-features --features google-vertex --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --lib standards::openai::compat::spec::tests::openai_compatible_spec_request_option_source_does_not_read_response_metadata --no-default-features --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --lib standards::openai::compat::spec --no-default-features --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_streaming_tool_call_tracker_only_uses_callback_provider_metadata --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_structured_output_helpers_only_merge_generic_response_metadata --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-provider-amazon-bedrock --lib providers::bedrock::ext --no-default-features --features bedrock --no-fail-fast`
  - `cargo nextest run -p siumai-provider-gemini --lib providers::gemini::ext::request_options::tests::gemini_request_option_extension_source_does_not_read_response_metadata --no-default-features --features google --no-fail-fast`
  - `cargo nextest run -p siumai-provider-gemini --lib providers::gemini::ext --no-default-features --features google --no-fail-fast`
  - `cargo nextest run -p siumai-provider-minimaxi --lib providers::minimaxi::ext::request_options::tests::minimaxi_request_option_extension_source_does_not_read_response_metadata --no-default-features --features minimaxi --no-fail-fast`
  - `cargo nextest run -p siumai-provider-minimaxi --lib providers::minimaxi::ext --no-default-features --features minimaxi --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --lib providers::vertex::ext::embedding::tests::vertex_embedding_request_option_extension_source_does_not_read_response_metadata --no-default-features --features google-vertex --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --lib providers::vertex::ext::imagen::tests::vertex_imagen_request_option_extension_source_does_not_read_response_metadata --no-default-features --features google-vertex --no-fail-fast`
  - `cargo nextest run -p siumai-provider-google-vertex --lib providers::vertex::ext::video::tests::vertex_video_request_option_extension_source_does_not_read_response_metadata --no-default-features --features google-vertex --no-fail-fast`
  - `cargo nextest run -p siumai --test gemini_generate_content_stream_bridge_roundtrip_fixtures_alignment_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo fmt --package siumai --check`
  - `cargo check -p siumai --tests --features all-providers,gcp --no-default-features`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test content_part_provider_map_audit_covers_high_value_production_hits --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test stable_unified_prelude_keeps_non_family_extension_types_scoped --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test broad_facade_types_path_is_explicit_compat_only --features openai,anthropic,google --no-default-features --no-fail-fast`

## FSCBC-M7 - Final Validation

Acceptance criteria:

- `todo.md` has no remaining `[ ]` or `[~]` work items except explicitly deferred items with
  rationale.
- Focused `cargo nextest` runs pass for affected crates.
- Formatting and whitespace checks pass.
- Workstream notes record all intentional breaking removals or public migration paths.

Status: pending
