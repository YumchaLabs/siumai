# Fearless Spec/Core Boundary Convergence - TODO

Last updated: 2026-05-16

Status: Closed. Remaining broader `ContentPart` replacement work is deferred to a future
compatibility-breaking design, not part of this spec/core boundary workstream.

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[!]` blocked
- `[d]` deferred with rationale

## Track A - Audit And Guards

- [x] Create the workstream documents and add them to the docs index.
- [x] Add a spec purity guard that rejects runtime-only cancellation imports in `siumai-spec`.
- [x] Add a core boundary guard that rejects provider-specific bridge targets or serializers under
  `siumai-core/src`.
  - `siumai-core::core_provider_boundary_test` now rejects provider-specific bridge targets,
    bridge target catalogs, and protocol custom-event serializers anywhere under `siumai-core/src`.
  - `OpenAiResponsesStreamPartsBridge` has moved from `siumai-core` to
    `siumai-bridge::stream`, so OpenAI Responses stream-parts adaptation is no longer owned by the
    provider-agnostic runtime crate.
  - `StreamPartNamespace` and `to_protocol_custom_event` have been removed from `siumai-core`;
    protocol-specific custom stream event formatting now lives in the OpenAI, Anthropic, and Gemini
    protocol serializers.
  - `BridgeTarget`, `BridgeOptions`, bridge reports, contexts, hooks, primitive remappers, and loss
    policy traits moved out of `siumai-core::bridge` and are now owned by `siumai-bridge`.
- [x] Add a registry guard that prevents stable family handles from executing through
  `compat_*_client*` or `as_*_capability()` paths.
  - `siumai-registry::registry::entry::boundary_tests` rejects generic compatibility clients and
    `LlmClient` capability downcasts in primary family execution sections.
- [x] Add a facade guard that prevents compatibility-only construction aliases from entering stable
  preludes.
  - Existing `siumai::facade_architecture_boundary_test` rejects `Provider`, `SiumaiBuilder`, and
    deprecated experimental helpers from `prelude::unified`.
  - `stable_unified_prelude_keeps_only_audited_compatibility_and_runtime_aliases` also keeps the
    remaining runtime bridge aliases tied to explicit compatibility-audit entries.
- [x] Add a short audit note for every kept compatibility surface that explains why it still exists.
  - The broad facade type path `siumai::types::*` has been removed from the facade root; the
    migration-only catch-all namespace now lives under `siumai::compat::types::*` and
    `siumai::prelude::compat::types::*`.
  - Audited `prelude::unified` compatibility aliases are now documented.
  - The legacy `ContentPart` dual `providerOptions` / `providerMetadata` carrier is now documented
    in `compatibility-audit.md` with owner, replacement, removal window, and validation commands.
  - The kept `siumai-spec::tools` / `siumai-spec::types::tools` provider-defined tool surface is
    now documented as passive data constructors/carriers, with a guard covering both the helper
    module and stable tool data types.

## Track B - Spec Purity

- [x] Move `CancelHandle` and runtime cancellation semantics out of `siumai-spec`.
- [x] Remove async runtime dependencies from `siumai-spec` when they are no longer needed.
  - `tokio-util` was removed with `CancelHandle`.
  - `futures` was removed by moving the runtime `AudioStream` alias to `siumai-core`.
- [x] Split `siumai-spec/src/types/ai_sdk.rs` into responsibility-focused modules.
  - The surface now lives as the `siumai-spec/src/types/ai_sdk/` directory module.
  - Shared primitives (`JSONValue`, `CallWarning`, provider option/metadata aliases,
    `TelemetryOptions`) live in `ai_sdk/shared.rs`.
  - Call option carriers (`LanguageModelV4CallOptions`, `RequestOptions`,
    `LanguageModelReasoning`, `LanguageModelCallOptions`, and deprecated `CallSettings`) live in
    `ai_sdk/call_options.rs`.
  - Passive error carriers (`AISDKError`, `APICallError`, provider/model lookup errors,
    no-output/no-media errors, UI message conversion errors, retry errors, and validation errors)
    live in `ai_sdk/errors.rs`.
  - Generated file carriers (`GeneratedFile`, `DefaultGeneratedFileWithType`,
    `GeneratedAudioFile`, and `DefaultGeneratedAudioFileWithType`) live in
    `ai_sdk/generated_files.rs`.
  - Response metadata carriers (`ImageModelResponseMetadata`, `VideoModelResponseMetadata`,
    `SpeechModelResponseMetadata`, and `TranscriptionModelResponseMetadata`) live in
    `ai_sdk/response_metadata.rs`.
  - Usage carriers and helpers (`LanguageModelV4Usage`, `LanguageModelUsage`,
    `EmbeddingModelUsage`, `ImageModelUsage`, `add_language_model_usage`, and
    `add_image_model_usage`) live in `ai_sdk/usage.rs`.
  - Embedding result and event carriers (`ModelCallResponseData`, `EmbedResult`,
    `EmbedManyResult`, `EmbedStartEvent`, `EmbedEndEvent`, and embedding model-call events) live in
    `ai_sdk/embedding.rs`.
  - Rerank result and event carriers (`RerankResponseMetadata`, `RerankResult`,
    `RerankStartEvent`, `RerankEndEvent`, and reranking model-call events) live in
    `ai_sdk/rerank.rs`.
  - Language-model request/response metadata carriers (`LanguageModelRequestMetadata`,
    `LanguageModelResponseMetadata`, `LanguageModelV4ResponseMetadata`,
    `LanguageModelV4GenerateResponseMetadata`, and `LanguageModelV4StreamResponseMetadata`) live in
    `ai_sdk/language_model_metadata.rs`.
  - Language-model V4 result envelopes (`LanguageModelV4FinishReason`,
    `LanguageModelV4GenerateResult`, and `LanguageModelV4StreamResult`) live in
    `ai_sdk/language_model_results.rs`.
  - Language-model V4 prompt/content projection ownership now lives under
    `ai_sdk/language_model_v4/`, with `language_model_v4.rs` kept as a thin re-export shell:
    - shared V4 data and serde validators (`LanguageModelV4DataContent`,
      `LanguageModelV4GeneratedFileData`, `LanguageModelV4FilePartData`, type markers,
      provider option/metadata object-shape helpers, and custom-kind validation) live in
      `ai_sdk/language_model_v4/shared.rs`.
    - request-side prompt/message projections (`LanguageModelV4Prompt`,
      `LanguageModelV4*Part`, `LanguageModelV4*Message`, and
      `prepare_language_model_v4_prompt`) live in `ai_sdk/language_model_v4/prompt.rs`.
    - response-side generated content projections (`LanguageModelV4Text`,
      `LanguageModelV4Reasoning`, `LanguageModelV4Source`, generated file/tool content, and
      `LanguageModelV4Content`) live in `ai_sdk/language_model_v4/content.rs`.
  - Source citation carriers (`Source`, `ImageModelProviderMetadata`, and
    `VideoModelProviderMetadata`) live in `ai_sdk/source.rs`.
  - Structured-output result and event carriers (`GenerateObjectOutputStrategy`,
    `GenerateObjectResponseMetadata`, `GenerateObjectStartEvent`, `GenerateObjectStepStartEvent`,
    `GenerateObjectStepEndEvent`, and `GenerateObjectEndEvent`) live in
    `ai_sdk/generate_object.rs`.
  - Text-generation result and callback event carriers (`GenerateTextContentPart`,
    `GenerateTextResult`, `GenerateTextStepResult`, `GenerateTextStartEvent`,
    `GenerateTextStepStartEvent`, `GenerateTextEndEvent`, `PrepareStepOptions`,
    `PrepareStepResult`, and `StreamTextChunkEvent`) live in `ai_sdk/generate_text.rs`.
  - Output content and tool-output carriers (`TextOutput`, `CustomOutput`, `FileOutput`,
    `ReasoningOutput`, `ToolCall`, `ToolResult`, `ToolError`, `ToolOutput`,
    `ToolOutputDenied`, and tool approval output parts) live in `ai_sdk/output_parts.rs`.
  - Flow-control helpers (`StopCondition`, active-tool filtering, and `pruneMessages` data
    rules/helpers) live in `ai_sdk/flow_control.rs`.
  - Tool lifecycle carriers (approval configuration, repair callback data, execution callback
    events, and deprecated callback aliases) live in `ai_sdk/tool_lifecycle.rs`.
  - Timeout carriers and helper accessors (`TimeoutConfiguration`, `TimeoutConfigurationSettings`,
    and `get_*_timeout_ms`) live in `ai_sdk/timeout.rs`.
  - Object-stream part carriers (`ObjectStreamObjectPart`, `ObjectStreamTextDeltaPart`,
    `ObjectStreamErrorPart`, `ObjectStreamFinishPart`, and `ObjectStreamPart`) live in
    `ai_sdk/object_stream.rs`.
  - Text-stream and language-model stream part carriers (`TextStreamPart`,
    `LanguageModelStreamPart`, their concrete part structs, and experimental language-model stream
    aliases) live in `ai_sdk/text_stream.rs`.
  - Media result envelopes (`GenerateImageResult`, `GenerateVideoResult`, `SpeechResult`,
    `TranscriptionResult`, and experimental compatibility aliases) live in
    `ai_sdk/media_results.rs`.
  - UI message aliases, helper predicates, chat transport options, completion options, and
    `UI_MESSAGE_STREAM_HEADERS` live in `ai_sdk/ui_message.rs`.
  - UI message stream chunk carriers (`UiMessageChunk`, `UIMessageChunk`, `DataUIMessageChunk`,
    `InferUIMessageChunk`, and concrete `UiMessage*Chunk` structs) live in
    `ai_sdk/ui_message_chunks.rs`.
  - `ai_sdk_module_boundary_test` prevents the surface from collapsing back to a single file and
    guards the nested V4 shared/prompt/content split, including prompt-side request metadata and
    content-side response metadata direction. The shell module is also kept to pure `mod`/`pub use`
    re-exports so new concrete types do not creep back into `mod.rs` or the V4 shell.
- [x] Keep prompt/input data, response/result data, UI message data, and runtime helper concepts in
  separate modules or crates.
  - AI SDK V4 prompt and generated content projections are now physically separate.
  - Non-V4 stable prompt projection now has named helpers in `siumai-spec::types::prompt`:
    `project_chat_message_to_prompt_message`, `project_chat_messages_to_prompt_messages`,
    `project_prompt_message_to_chat_message`, and `project_prompt_messages_to_chat_messages`.
    `content_projection_boundary_test` keeps these prompt projection types request-side only and
    verifies prompt-to-legacy conversions do not emit response-side provider metadata.
  - Non-V4 response projection is now classified as the existing AI SDK output surface
    (`GenerateTextContentPart` plus `output_parts.rs` carriers) rather than a new generated-content
    family. Explicit fallible helpers now project response-owned legacy subsets into
    `GenerateTextContentPart`, while ambiguous legacy carriers remain errors instead of being
    silently treated as generated output.
  - The facade `siumai::text::generate_text` projection path now delegates response-content mapping
    to the spec-owned helper, with only the documented legacy tool-result-without-input fallback
    still handled locally.
  - UI message data and UI message stream chunks are already separate modules.
  - Runtime cancellation/stream aliases have been moved out of `siumai-spec`.
  - Closeout: the direct-construction scan is guarded by
    `content_part_provider_map_audit_covers_high_value_production_hits`; remaining broader
    non-V4 `ContentPart` replacement work is an explicit deferred follow-up, not an open
    spec/core boundary task.
- [x] Review all spec types for hidden execution policy, transport policy, or provider construction
  semantics.
  - `siumai-spec::LlmError` no longer owns retry, status-code, category, user-message, recovery
    suggestion, or retry-delay policy helpers. Those runtime/presentation decisions now live in
    `siumai-core::error::{ErrorCategory,LlmErrorExt}` while the spec crate keeps only the passive
    error data shape and constructors.
  - `spec_purity_boundary_test::spec_error_type_does_not_own_runtime_policy_helpers` prevents those
    policy helpers from returning to the spec crate.
  - `siumai-spec::HttpConfig` no longer reads `SIUMAI_STREAM_DISABLE_COMPRESSION` from the process
    environment. The spec crate keeps only a deterministic passive default; runtime default
    construction now lives entirely in `siumai-core::defaults::http::config_default()`.
  - Core request/executor tests use `HttpConfig::empty()` for request-level header overrides, so
    test fixtures no longer imply that `HttpConfig::default()` means runtime defaults.
  - Provider helper tests and public examples that only need per-request/header overrides now use
    `HttpConfig::empty()` as well; runtime defaults remain in provider/config-first constructors.
  - `core_provider_boundary_test::runtime_http_defaults_do_not_use_passive_http_config_default`
    now scans production sources across core, protocol, provider, registry, extras, and facade
    crates to prevent `HttpConfig::default()` from being used as a runtime-default shortcut.
  - Workspace-wide `HttpConfig::default()` references are intentionally limited to the spec boundary
    test and guard text; empty/request-level fixtures use `HttpConfig::empty()`.
  - `spec_purity_boundary_test::spec_source_does_not_define_runtime_handles_or_streams` now rejects
    `std::env`, `CARGO_PKG_VERSION`, and `runtime_default` usage in `siumai-spec/src`, preventing
    runtime environment, versioned user-agent, or runtime-default policy from returning to spec data
    carriers.
  - `ChatResponse` docs now use pure data construction examples instead of teaching facade builder,
    provider extension, or concrete client execution paths from inside `siumai-spec`.
  - `spec_purity_boundary_test::spec_docs_do_not_teach_facade_or_provider_runtime_construction`
    rejects facade/prelude imports, `Siumai::builder`, provider extension paths, provider builder
    calls, API-key builder calls, and client chat execution snippets under `siumai-spec/src`.
  - `tools_boundary_test::provider_defined_tool_data_surface_remains_passive` now scans
    `siumai-spec/src/tools.rs` and `siumai-spec/src/types/tools/**` so provider-defined tool
    helpers and data carriers cannot grow runtime execution, HTTP client/server, thread/process,
    provider crate, protocol crate, or core tooling execution dependencies.
  - `video_boundary_test::video_generation_surface_remains_passive_data_contract` now guards
    `siumai-spec/src/types/video.rs` so video request/response/status carriers cannot grow
    runtime polling, HTTP client/server, thread/process, provider crate, protocol crate, or core
    execution dependencies.
  - `video_boundary_test::video_request_header_helper_uses_empty_http_override_config` keeps
    `VideoGenerationRequest::with_header(...)` tied to `HttpConfig::empty()` so per-request video
    header overrides do not become a backdoor for runtime HTTP defaults in `siumai-spec`.
  - `request_carrier_boundary_test::non_ai_sdk_request_carriers_remain_passive_data_contracts`
    scans the non-`ai_sdk` audio, completion, embedding, files, image, rerank, and skills request
    surfaces so model-family/upload request carriers cannot grow runtime execution, HTTP
    client/server, provider crate, protocol crate, or core dependencies.
  - `request_carrier_boundary_test::request_header_helpers_use_empty_http_override_config` keeps
    completion, embedding, file-list, rerank, and skill-upload header helpers on
    `HttpConfig::empty()` request overrides instead of runtime defaults.
  - `chat_provider_helper_boundary_test` documents and guards the removal of the historical
    `ChatMessageBuilder` Anthropic cache/document helper methods. The spec chat builder no longer
    exposes provider-specific request helpers and must not grow new provider-prefixed helpers,
    Anthropic request option literals, response-metadata reads, provider/protocol dependencies, or
    runtime behavior.
  - `chat_message_production_source_does_not_embed_concrete_provider_names` now also guards
    `siumai-spec::ChatMessage` production code against concrete provider namespace literals such
    as `anthropic`, `openai`, or `gemini`; provider-specific message helpers must live in provider
    extension crates.
  - `siumai-provider-anthropic::providers::anthropic::ext::AnthropicChatMessageExt` now provides
    the provider-owned replacement surface for Anthropic message cache/document helpers, and the
    facade exports it through `siumai::provider_ext::anthropic`.
  - The Anthropic prompt-caching example now demonstrates the provider-owned message extension
    path instead of teaching the removed `ChatMessageBuilder::cache_control(...)` method.
  - Facade cache-control macro arms now route through a narrow private helper instead of calling
    the removed `ChatMessageBuilder::cache_control(...)` method, with a facade boundary guard
    preventing that teaching path from returning.
  - Bridge/protocol Anthropic cache/document tests no longer consume the compatibility spec builder
    methods. They now use explicit request-side `providerOptions.anthropic.*` fixtures or the
    legacy passive `MessageMetadata.cache_control` field only when validating compatibility
    behavior. `low_level_crates_do_not_consume_anthropic_chat_builder_compat_helpers` guards
    bridge, Anthropic protocol, facade, and examples against reintroducing those calls.
  - The historical `ChatMessageBuilder::{cache_control,cache_control_for_part,
    cache_control_for_parts,anthropic_document_citations_for_part,
    anthropic_document_metadata_for_part}` methods have been removed from `siumai-spec`; the
    provider-owned `AnthropicChatMessageExt` path is now the migration target.
- [x] Run `cargo check -p siumai-spec --no-default-features` after each spec slice.
  - Latest error-policy slice verified `cargo check -p siumai-spec --no-default-features`.
  - Latest HTTP runtime-default slice verified:
    - `cargo check -p siumai-spec --no-default-features`
    - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-spec --test http_config_boundary_test --no-default-features --no-fail-fast`
    - `cargo check -p siumai-core --no-default-features`
    - `cargo nextest run -p siumai-core --lib defaults --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --lib execution::http::client defaults --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --lib utils::chat_request execution::executors::embedding execution::executors::files execution::executors::image --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --lib execution::executors::chat --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-provider-anthropic --no-default-features --features anthropic providers::anthropic::tokens providers::anthropic::files providers::anthropic::skills providers::anthropic::message_batches --no-fail-fast`
    - `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex providers::vertex::context providers::anthropic_vertex::context --no-fail-fast`
    - `cargo nextest run -p siumai-provider-ollama --no-default-features --features ollama providers::ollama::chat providers::ollama::embeddings providers::ollama::model_listing --no-fail-fast`
    - `cargo nextest run -p siumai-provider-openai --no-default-features --features openai providers::openai::rerank --no-fail-fast`
    - `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google standards::gemini::transformers --no-fail-fast`
    - `cargo nextest run -p siumai-provider-anthropic --no-default-features --features anthropic providers::anthropic::client --no-fail-fast`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai --test google_vertex_imagen_headers_alignment_test --features google-vertex --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai --test provider_public_path_parity_test --features openai,anthropic,google,google-vertex --no-default-features --no-fail-fast`
    - `cargo check -p siumai-registry --tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock,xai --no-default-features`
    - `cargo check -p siumai --example complex-request --features openai --no-default-features`
    - `cargo check -p siumai --example bedrock-chat --example bedrock-rerank --features bedrock --no-default-features`
    - `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
  - Latest spec-doc purity slice verified:
    - `cargo fmt --package siumai-spec --check`
    - `cargo check -p siumai-spec --no-default-features`
    - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`
  - Latest AI SDK V4 metadata-direction guard slice verified:
    - `cargo fmt --package siumai-spec --check`
    - `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
    - `cargo check -p siumai-spec --no-default-features`
  - Latest spec tool passive-data guard slice verified:
    - `cargo fmt --package siumai-spec --check`
    - `cargo nextest run -p siumai-spec --test tools_boundary_test --no-default-features --no-fail-fast`
  - Latest video passive-data guard slice verified:
    - `cargo fmt --package siumai-spec --check`
    - `cargo nextest run -p siumai-spec --test video_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --test http_config_boundary_test --test video_boundary_test --no-default-features --no-fail-fast`
    - `cargo check -p siumai-spec --no-default-features`
  - Latest request-carrier passive-data guard slice verified:
    - `cargo fmt --package siumai-spec --check`
    - `cargo nextest run -p siumai-spec --test request_carrier_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --test http_config_boundary_test --test request_carrier_boundary_test --test video_boundary_test --no-default-features --no-fail-fast`
    - `cargo check -p siumai-spec --no-default-features`
  - Latest chat provider-helper compatibility guard slice verified:
    - `cargo fmt --package siumai-spec --check`
    - `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --test spec_purity_boundary_test --test tools_boundary_test --no-default-features --no-fail-fast`
    - `cargo check -p siumai-spec --no-default-features`
  - Latest provider-owned Anthropic chat-message extension slice verified:
    - `cargo fmt --package siumai-provider-anthropic --package siumai --check`
    - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::chat_message --no-default-features --features anthropic --no-fail-fast`
    - `cargo check -p siumai --example prompt-caching --features anthropic --no-default-features`
    - `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_macros_only_create_request_side_empty_provider_options --features anthropic --no-default-features --no-fail-fast`
  - Latest Anthropic compatibility-helper consumer cleanup verified:
    - `cargo fmt --package siumai-spec --package siumai-bridge --package siumai-protocol-anthropic --check`
    - `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-bridge --lib anthropic_part_cache_paths_follow_canonical_part_provider_options --no-default-features --features anthropic,openai --no-fail-fast`
    - `cargo nextest run -p siumai-bridge --lib anthropic_bridge_reports_cache_breakpoints_beyond_limit anthropic_bridge_reports_part_cache_breakpoints_from_canonical_provider_options --no-default-features --features anthropic,openai --no-fail-fast`
    - `cargo nextest run -p siumai-protocol-anthropic --lib standards::anthropic::utils::messages standards::anthropic::chat --no-default-features --features anthropic-standard --no-fail-fast`
  - Latest Anthropic spec builder helper removal slice verified:
    - `cargo fmt --package siumai-spec --check`
    - `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --no-default-features --no-fail-fast`
    - `cargo check -p siumai-spec --no-default-features`

## Track C - Prompt And Response Content Separation

- [x] Audit content part types that carry both `providerOptions` and `providerMetadata`.
  - The legacy stable `ContentPart` union is explicitly classified as a dual-use compatibility
    carrier for `Text`, `Image`, `Audio`, `File`, `ReasoningFile`, `Custom`, `ToolCall`,
    `ToolApprovalRequest`, `ToolResult`, and `Reasoning`.
  - `ContentPart::Source` is audited as response-side citation data with `providerMetadata` only.
  - `ContentPart::ToolApprovalResponse` is audited as request/decision-side data with
    `providerOptions` only.
  - `content_projection_boundary_test::legacy_content_part_dual_provider_maps_stay_explicitly_audited`
    prevents new dual `providerOptions`/`providerMetadata` `ContentPart` variants from appearing
    without updating this Track C audit.
  - `content-part-construction-audit.md` now classifies high-value direct construction and
    metadata/options usage paths by request-side, response-side, workflow, typed-view, and
    remaining-candidate ownership.
  - A refreshed direct-construction scan now classifies `siumai-bridge/src/customize.rs` as
    primitive-only bridge customization shared by request, response, and stream paths.
    `bridge_customization_source_stays_primitive_only` prevents the built-in remapper from
    becoming a provider option/metadata channel.
  - The same scan now classifies `siumai-core/src/streaming/{builder.rs,factory.rs}` as
    provider-agnostic stream helper paths. `core_stream_helpers_only_initialize_empty_provider_metadata`
    permits only empty `provider_metadata: None` initialization in those core helpers.
  - `siumai-core/src/utils/streaming_tool_call.rs` is now classified as a provider-agnostic
    streaming tool-call tracker. It may forward metadata produced by caller-supplied callbacks, but
    `core_streaming_tool_call_tracker_only_uses_callback_provider_metadata` prevents core from
    hard-coding provider metadata namespaces or reading request provider options.
  - `siumai-core/src/structured_output.rs` is now classified as provider-agnostic JSON extraction
    and stream-response consolidation. It may generically merge response `provider_metadata`, but
    `core_structured_output_helpers_only_merge_generic_response_metadata` prevents provider
    namespaces or request provider options from entering the helper.
  - `siumai-core/src/execution/executors/stream_json.rs` is now classified as a provider-agnostic
    line-delimited JSON streaming executor. `core_json_stream_executor_does_not_handle_provider_maps`
    keeps provider-specific stream parsing delegated to injected converters instead of letting core
    read provider maps directly.
  - `siumai-protocol-anthropic/src/standards/anthropic/thinking.rs` is now classified as a mixed
    Anthropic protocol helper with split request and response boundaries. The request config source
    stays free of response metadata, while the response projection helper stays free of request
    provider options.
  - Core stable family contract adapters (`text`, `completion`, `speech`, `transcription`), core
    trait shells, and runtime tooling contracts are now guarded as provider-map-neutral surfaces by
    `core_family_contract_and_tooling_sources_do_not_handle_provider_maps`.
  - `siumai-core/src/execution/middleware/samples.rs` is now classified as provider-neutral sample
    streaming middleware. It may synthesize typed stream text parts with empty
    `provider_metadata: None`, but it cannot read provider options or concrete provider namespaces.
  - `siumai-core/src/utils/provider_options.rs` is now classified as a generic request-side
    provider-option schema parser. Its source guard allows `ProviderOptionsMap` parsing while
    rejecting response metadata reads and concrete provider namespaces.
  - `siumai-core/src/custom_provider/guide.rs` no longer teaches direct legacy `ContentPart`
    variant matching for provider JSON serialization; documentation now points custom providers to
    own provider-specific content mapping outside core.
  - Anthropic, OpenAI, Anthropic-on-Vertex, and MiniMaxi provider client/spec files now have
    focused source guards for request-side provider-option routing/default merging. Those guards
    allow provider-owned request option handling while rejecting response metadata reads as request
    input.
  - `siumai-provider-gemini/src/providers/gemini/client.rs` is classified as a test-only broad
    scan hit for structured-output stream metadata preservation; production Gemini metadata
    handling remains covered by transformer and provider metadata guards.
  - Remaining facade hits are now classified and guarded: macros can only initialize empty
    request-side provider options, speech/transcription/structured-output helpers can only project
    response metadata into high-level results, and video keeps high-level polling options separate
    from legacy provider option maps.
- [x] Introduce explicit prompt-side and response-side projections where the same type encourages
  request/response concern mixing.
  - AI SDK V4 prompt projections under `types/ai_sdk/language_model_v4/prompt.rs` carry
    request-side `providerOptions` and are source-guarded against response-side
    `providerMetadata` / `provider_metadata`.
  - AI SDK V4 generated content projections under `types/ai_sdk/language_model_v4/content.rs`
    carry response-side `providerMetadata` and are source-guarded against request-side
    `providerOptions` / `provider_options`.
  - Current decision: use an adapter-first migration and defer any broader non-V4 stable
    prompt/content projection until remaining direct `ContentPart` construction paths have a clear
    migration target.
- [x] Add adapters for shared content data instead of reusing response-only metadata in request
  construction.
  - Existing `ChatMessage`/`Prompt` narrowing adapters already reject response-side
    `providerMetadata` when converting legacy `ContentPart` values into prompt/model-message
    shapes.
  - `content_projection_boundary_test::prompt_projection_rejects_response_side_provider_metadata_on_legacy_content_parts`
    now guards representative legacy text, image, file, reasoning, custom, tool-call, and
    tool-result parts so response metadata cannot leak into request construction.
  - `siumai-core::ui` intentionally treats AI SDK UI `providerMetadata`,
    `callProviderMetadata`, and `resultProviderMetadata` as UI-layer request metadata. Conversion
    into stable model messages normalizes those maps into request-side `provider_options` and keeps
    legacy `ContentPart::provider_metadata` empty on request construction paths.
  - `siumai-core::ui` unit tests now guard normal UI parts and tool parts against leaking
    UI-layer provider metadata into response-side legacy `ContentPart::provider_metadata`.
  - `siumai-core::ui` request conversion now centralizes legacy `ContentPart` construction through
    UI request adapter helpers, and
    `ui_conversion_centralizes_legacy_request_content_constructors` prevents new scattered
    request-side `provider_metadata: None` writes from returning to UI conversion code.
  - Gemini request conversion now reads thought signatures / thought flags only from
    request-side `provider_options`, not from legacy `ContentPart::provider_metadata`;
    `request_conversion_source_only_ignores_legacy_provider_metadata_fields` guards the source
    shape against future direct legacy reads.
  - Gemini request transformer post-processing now has
    `gemini_request_transformer_source_does_not_read_response_provider_metadata`, so
    provider-option mapping in the transformer cannot start reading response-side
    `provider_metadata` / `providerMetadata`.
  - Anthropic request conversion now replays assistant reasoning signatures/redacted data only from
    request-side `provider_options`, not from legacy `ContentPart::provider_metadata`;
    `request_conversion_source_does_not_read_legacy_provider_metadata_fields` guards the source
    shape against future direct legacy reads.
  - Anthropic mixed request/response transformer now has split source guards:
    `anthropic_request_transformer_source_does_not_read_response_provider_metadata` keeps request
    transformation away from response metadata, while
    `anthropic_response_transformer_source_does_not_read_request_provider_options` keeps response
    transformation away from request options.
  - Anthropic request content conversion now also has
    `request_content_conversion_source_does_not_read_legacy_provider_metadata_fields`, guarding the
    `convert_message_content` implementation against reading legacy `providerMetadata` /
    `provider_metadata` for document, file, or tool-result request settings.
  - Anthropic prompt-cache request building now has
    `cache_request_builder_source_does_not_read_legacy_provider_metadata`, so cache block
    construction cannot learn from legacy response-side provider metadata.
  - Anthropic structured-output extension and OpenAI structured-output request config now have
    request-side source guards. OpenAI's response validator remains explicitly response-side and is
    covered by a focused behavior test.
  - Anthropic and Anthropic-on-Vertex request option extension helpers now have request-only source
    guards, so typed provider option convenience APIs cannot read response metadata.
  - OpenAI-compatible request routing and provider option normalization now has a source guard that
    rejects response metadata reads while preserving existing runtime-provider option behavior.
  - Bedrock extension helpers now split request-only helpers from the explicit reasoning replay
    bridge. The bridge remains a scoped cross-step exception that only projects Bedrock reasoning
    `signature` / `redactedData` metadata into request-side provider options.
  - Gemini embedding/image/video request extension helpers now each have request-only source
    guards, so modality-specific convenience APIs cannot start reading response metadata while
    populating request `provider_options_map`.
  - MiniMaxi structured-output, thinking, TTS, and video extension helpers now have request-only
    source guards, covering both direct request option traits and typed request builders.
  - `siumai-provider-anthropic` prepare-step container replay is explicitly classified as a
    response-level metadata workflow, not a legacy `ContentPart` metadata request replay.
    `prepare_step_source_only_bridges_response_metadata_to_request_provider_options` guards that it
    reads prior-step `ProviderMetadataMap` values only to produce next-step request-side
    `ProviderOptionsMap` overrides.
  - `siumai-provider-anthropic` tools extension now guards its mixed helper file by direction:
    `with_anthropic_tool_options(...)` remains request-side `providerOptions.anthropic` builder
    logic, while hosted-tool stream/custom-event projections remain response/stream metadata
    views and cannot read request provider options.
  - `siumai-provider-gemini` tools extension now guards source/custom-event projection with
    `gemini_tools_extension_source_does_not_read_request_provider_options`, so response/stream
    source helper code cannot start reading request-side provider options.
  - OpenAI Responses request conversion already ignores legacy part provider metadata for image
    detail and assistant tool item ids; the existing protocol tests cover that direction, and
    `request_transformer_source_does_not_read_legacy_provider_metadata_fields` keeps the request
    implementation from reintroducing direct legacy reads. The guard also rejects replacing ignored
    `provider_metadata: _` destructuring with a bound legacy metadata variable.
  - OpenAI Responses provider extension stream/custom-event projection now has
    `responses_stream_event_projection_source_does_not_read_request_provider_options`, so
    response-side projection can still read response metadata without learning from request-side
    `provider_options`.
  - OpenAI audio client request/response handling now has
    `audio_mixed_request_response_path_does_not_read_legacy_response_metadata`, so non-chat audio
    request construction can still read request-side `provider_options_map` / `extra_params`
    without learning from legacy response metadata.
  - OpenAI WebSocket session request/recovery handling now has
    `websocket_session_request_mutation_does_not_read_legacy_metadata_maps`, so the
    provider-specific session may mutate request `provider_options_map` for warm-up and recovery
    without using legacy provider metadata as request input.
  - OpenAI chat request entry now has
    `chat_request_path_does_not_read_legacy_response_metadata_maps`, so default
    `provider_options_map` merging and response-id cancel wrapping stay away from legacy provider
    metadata.
  - OpenAI file and skill provider extensions now have source guards:
    `file_upload_request_path_does_not_read_legacy_provider_metadata` keeps file upload request
    options request-side, while
    `skill_upload_request_and_response_paths_keep_provider_maps_directional` keeps skill upload
    request handling away from response metadata and skill upload response projection away from
    request provider options.
  - OpenAI Chat and OpenAI-compatible request conversion in
    `siumai-protocol-openai::standards::openai::utils` now has
    `openai_chat_request_conversion_source_does_not_read_legacy_provider_metadata_fields`, so
    shared message/content utilities can keep reading request-side `provider_options` without
    learning from legacy response-side `ContentPart::provider_metadata`.
  - Alibaba/Qwen cache-control request shaping in
    `siumai-protocol-openai::standards::openai::compat::alibaba_cache_control` now has
    `alibaba_cache_control_source_does_not_read_legacy_provider_metadata`, so cache-control
    request shaping remains request-side `providerOptions` only.
  - OpenAI-compatible protocol transformers are now split by source guard:
    `openai_compatible_request_transformer_source_does_not_read_legacy_provider_metadata_fields`
    keeps request construction away from legacy response metadata, while
    `openai_compatible_chat_response_source_does_not_emit_request_provider_options` keeps
    response `ContentPart` options empty.
  - OpenAI-compatible streaming replay now has
    `openai_compatible_streaming_source_does_not_emit_request_provider_options`, preventing stream
    conversion from emitting request-side `providerOptions` while allowing provider metadata
    namespace normalization.
  - OpenAI chat standard wrapper now has
    `chat_wrapper_keeps_request_response_stream_maps_directional`, so request, response, and stream
    wrapper sections remain directional while the known provider-metadata namespace selection stays
    explicit in `resolve_provider_metadata_key(...)`.
  - Anthropic chat standard wrapper now has
    `chat_wrapper_keeps_request_and_response_provider_maps_directional`, so request-side citation
    document extraction can read `ContentPart::File.provider_options` while response and stream
    wrapper sections stay free of request provider option reads.
  - Gemini chat standard wrapper now has
    `chat_wrapper_keeps_request_response_stream_maps_directional`, so request, response, and stream
    adapter sections remain split and delegate provider-specific mapping to the dedicated
    transformers.
  - OpenAI Responses response parsing now has
    `responses_response_transformer_source_does_not_emit_request_provider_options`, so Responses
    output items can carry response-side `provider_metadata` without reintroducing request options.
  - OpenAI Responses SSE conversion now has a directory-level source guard,
    `responses_sse_converter_sources_do_not_read_request_provider_options`, so stream parsing and
    replay metadata projection cannot start reading request-side `provider_options`.
  - The protocol-owned OpenAI typed provider metadata view now has
    `openai_provider_metadata_source_does_not_read_request_provider_options`, matching the
    response-side typed-view guards for Anthropic, Gemini, Bedrock, MiniMaxi, and Vertex.
  - Provider-owned OpenAI legacy completions are guarded in
    `completion_request_source_does_not_read_legacy_provider_metadata_fields` and
    `completion_response_and_stream_source_do_not_emit_request_provider_options`, separating
    completion request option handling from completion response/stream metadata emission.
  - Anthropic response parsing now has
    `anthropic_parse_response_content_source_does_not_emit_request_provider_options`, so response
    `ContentPart` construction keeps `provider_options` empty while still surfacing Anthropic
    citations, tool metadata, and sources.
  - Anthropic streaming response parsing now has
    `anthropic_streaming_parser_source_does_not_read_request_provider_options`, keeping SSE
    response-to-stream-part conversion away from request-side `provider_options`.
  - Anthropic streaming response serialization now has
    `anthropic_streaming_serialize_source_does_not_read_request_provider_options`, keeping replay
    of Anthropic stream metadata away from request-side `provider_options`.
  - Gemini request normalization in `siumai-bridge` now restores `thoughtSignature` replay hints as
    `provider_options.google.thoughtSignature` and keeps legacy `ContentPart::provider_metadata`
    empty; `gemini_request_normalization_source_uses_provider_options_for_thought_signature` guards
    against routing it back through response metadata.
  - `siumai-bridge::request` now also has
    `request_normalization_source_never_populates_legacy_provider_metadata`, so request
    normalization does not read legacy `providerMetadata` / `provider_metadata` JSON keys and can
    only leave legacy `provider_metadata` explicitly empty.
  - `siumai-bridge::request` request parsing now centralizes legacy `ContentPart` construction
    through request-side adapter helpers, so OpenAI Responses, OpenAI Chat Completions, Anthropic
    Messages, and Gemini Generate Content normalization no longer scatter dual-use carrier
    construction across protocol branches.
  - `request_normalization_centralizes_legacy_request_content_constructors` guards that
    request-side legacy provider metadata construction stays inside those adapters, with the only
    outside occurrence being the plain-text collapse match.
  - Direct OpenAI Responses ↔ Anthropic Messages request bridge pairs now have
    `request_bridge_pair_sources_do_not_read_legacy_provider_metadata`, so pair-specific reasoning
    replay can use request-side `provider_options` without reading legacy response metadata.
  - `siumai-provider-google-vertex` Vertex Gemini image edit/variation now builds synthetic
    Gemini file parts through a provider-owned request adapter helper, so image model request
    inputs keep provider options request-side and do not scatter legacy `provider_metadata` writes.
  - `vertex_gemini_image_request_content_construction_is_centralized` and
    `image_input_part_maps_provider_options_without_provider_metadata` guard that provider-owned
    request slice.
  - `siumai-provider-google-vertex` Vertex video task helpers now have
    `video_request_and_response_paths_keep_provider_maps_directional`, so request-side Vertex video
    options stay in request construction while response status parsing can emit Vertex video
    metadata without reading request provider options.
  - `siumai-provider-amazon-bedrock` chat request conversion now has
    `request_conversion_source_does_not_read_legacy_provider_metadata_fields`, keeping Bedrock
    document, cache-point, and reasoning replay settings on request-side `provider_options`.
    `response_and_stream_source_do_not_emit_request_provider_options` guards the matching
    response/stream parser so Bedrock response metadata cannot start emitting request-side
    `providerOptions`.
  - `siumai-provider-amazon-bedrock` embedding request/response transformation now has
    `embedding_request_and_response_transformers_keep_provider_maps_directional`, keeping
    `providerOptions.bedrock` reads request-side while response embedding parsing stays free of
    request provider options.
  - `siumai-provider-minimaxi` now guards its Anthropic-protocol adapter split:
    response/stream metadata re-keying from `anthropic` to `minimaxi` cannot read request-side
    provider options, and MiniMaxi request option resolution cannot read response-side
    `providerMetadata` / `provider_metadata`.
  - `siumai-provider-minimaxi` video task helpers now have
    `video_request_and_response_paths_keep_provider_maps_directional`, so video task request body
    construction can read `providerOptions.minimaxi` while task creation/query response parsing
    remains free of request provider options.
  - Google Vertex, Gemini, Bedrock, and MiniMaxi typed `provider_metadata::*` extension modules now
    have source guards that prevent response-side typed metadata views from reading request-side
    `providerOptions` / `provider_options` / `provider_options_map`.
  - The protocol-owned Anthropic typed provider metadata view now has the same response-side guard:
    `anthropic_provider_metadata_source_does_not_read_request_provider_options` prevents
    `ChatResponse` / `ContentPart` metadata helpers from reading request provider options.
  - `siumai-core::streaming::StreamProcessor` is classified as provider-agnostic response
    consolidation; its source guard prevents the final-response `ContentPart` accumulator from
    reading request-side `providerOptions` or provider option maps.
  - Gemini response parsing is source-guarded by
    `gemini_response_content_source_does_not_emit_request_provider_options`; response
    `ContentPart` constructors may preserve Gemini thought signatures in response-side
    `provider_metadata`, but their legacy `provider_options` fields must remain empty defaults.
  - Gemini streaming response parsing now has
    `gemini_streaming_parser_source_does_not_read_request_provider_options`, so Generate Content
    SSE events can project thought signatures into stream metadata without reading request options.
  - Gemini streaming response serialization now has
    `gemini_streaming_serialize_source_does_not_read_request_provider_options`, so thought
    signature replay in streamed Gemini wire output can read response metadata but not request
    provider options.
  - `siumai-bridge::response::tests::response_and_stream_bridge_sources_do_not_emit_request_provider_options`
    now guards response/stream production bridge sources so request-side `providerOptions` /
    `provider_options` cannot leak back into response serialization or stream replay code.
  - OpenAI, Anthropic, and Gemini gateway/proxy JSON response encoders now have source guards that
    keep those response-side encoders from reading request-side `providerOptions` /
    `provider_options` while still allowing response metadata serialization.
  - Main protocol and bridge request serialization paths now have behavior coverage plus source
    guards against direct legacy `providerMetadata` replay reads. Removing dual fields from the
    stable `ContentPart` surface remains a future compatibility-breaking follow-up after the
    adapter-first migration has wider replacement coverage.
- [x] Update docs and examples to use the canonical prompt and response content shapes.
  - `ContentPart` docs now explicitly call out the legacy dual-use compatibility status and point
    new provider-facing projections to AI SDK V4 prompt/content modules.
  - `facade_architecture_boundary_test::content_part_provider_map_audit_covers_high_value_production_hits`
    now replays the high-value Track C direct-construction scan against
    `content-part-construction-audit.md`, so new production `ContentPart::`,
    `provider_metadata:`, or `provider_options:` hits in core/bridge/protocol/provider/facade code
    must be classified before they can land.

## Track D - Core Runtime Boundary

- [x] Move provider-specific model defaults out of `siumai-core` into provider or registry crates.
  - `siumai-core::defaults` now only owns provider-agnostic HTTP, timeout, streaming, model
    parameter, and profile defaults.
  - Unused OpenAI, Anthropic, SiliconFlow, and Groq URL/model/timeout constants were removed from
    core. Provider crates and registry factories remain the owners for provider endpoint and model
    defaults.
  - `siumai-core::core_provider_boundary_test` rejects a returning `defaults::providers` module and
    known provider default fragments in `siumai-core/src/defaults.rs`.
- [x] Move provider-specific Vertex URL helpers out of `siumai-core::auth`.
  - `siumai-provider-google-vertex::auth::vertex` now owns Vertex publisher, Google Vertex,
    Anthropic-on-Vertex, express-mode, and Vertex MaaS URL helpers.
  - The facade compatibility path `siumai::experimental::auth::vertex` now re-exports the
    provider-owned module behind the `google-vertex` feature.
  - `siumai-core::core_provider_boundary_test` rejects a returning `auth::vertex` module and known
    Vertex URL helper fragments under `siumai-core/src`.
- [x] Move provider-specific Google Cloud ADC/service-account token providers out of
  `siumai-core::auth`.
  - `siumai-provider-google-vertex::auth::{adc,service_account}` now owns Google Cloud ADC and
    service-account token provider implementations.
  - `siumai-core::auth` keeps only the provider-agnostic `TokenProvider` contract and
    `StaticTokenProvider` test/basic helper.
  - The facade compatibility path `siumai::experimental::auth::{adc,service_account}` now
    re-exports the provider-owned modules behind the `gcp` feature.
  - `siumai-core::core_provider_boundary_test` rejects a returning core ADC/service-account module
    and known Google Cloud auth implementation fragments under `siumai-core/src`.
- [x] Remove provider-specific request body presets from `siumai-core::HookBuilder`.
  - `HookBuilder` now only accepts caller-supplied `with_chat_body_builder` functions for chat
    request body construction.
  - The old experimental `with_openai_base()` and `with_anthropic_base()` helpers were removed
    instead of migrated because no production call sites existed and protocol-shaped request bodies
    belong to provider/protocol ownership.
  - `siumai-core::core_provider_boundary_test` rejects returning provider-specific HookBuilder body
    presets.
- [x] Migrate `ProviderSpec` provider-shaped route fallbacks toward provider-owned route
  resolution.
  - `ProviderSpec` no longer contains core-owned provider-shaped fallback endpoint strings such as
    `chat/completions`, `images/generations`, or `models/{model_id}`.
  - Core executors now call fallible `try_*_url(...)` route methods, and `ProviderSpec` no longer
    exposes the historical string-returning `*_url(...)` hook methods.
  - `ProviderSpec` default fallible route methods now return `UnsupportedOperation` directly
    instead of calling or adapting through historical string hooks.
  - Custom provider examples and facade retry test specs now implement `try_*_url(...)` directly,
    so extension guidance no longer teaches the old hook as the primary route contract.
  - `siumai-core::core_provider_boundary_test` rejects returning legacy route fallback helpers and
    provider-shaped endpoint strings or historical string route hook methods to `ProviderSpec`.
  - Gemini protocol/provider route specs have removed their explicit string-returning `*_url(...)`
    implementations and now implement only the fallible `try_*_url(...)` route path.
  - `migrated_gemini_route_specs_do_not_reintroduce_legacy_string_hooks` prevents the migrated
    Gemini route specs from reintroducing direct string hook implementations.
  - OpenAI protocol standard route specs (`chat`, `embedding`, `image`, and `rerank`) have removed
    their explicit string-returning `*_url(...)` implementations and now implement only the fallible
    `try_*_url(...)` route path.
  - `migrated_openai_protocol_route_specs_do_not_reintroduce_legacy_string_hooks` prevents migrated
    OpenAI protocol standard specs from reintroducing direct string hook implementations.
  - OpenAI provider specs (`OpenAiSpec` and `OpenAiSpecWithRerank`) have removed their explicit
    string-returning route implementations and now implement only the fallible `try_*_url(...)`
    route path.
  - `migrated_openai_provider_route_specs_do_not_reintroduce_legacy_string_hooks` prevents migrated
    OpenAI provider specs from reintroducing direct string hook implementations.
  - OpenAI-compatible protocol spec (`OpenAiCompatibleSpecWithAdapter`) has removed its explicit
    string-returning route implementations and now applies URL settings inside fallible
    `try_*_url(...)` methods.
  - `migrated_openai_protocol_route_specs_do_not_reintroduce_legacy_string_hooks` now also guards
    the OpenAI-compatible protocol spec, including model listing/retrieve routes.
  - Anthropic protocol/provider route specs, including Anthropic-on-Vertex, have removed their
    explicit string-returning route implementations and now implement only the fallible
    `try_*_url(...)` route path.
  - `migrated_anthropic_route_specs_do_not_reintroduce_legacy_string_hooks` prevents the migrated
    Anthropic route specs from reintroducing direct string hook implementations.
  - DeepSeek and Groq OpenAI-compatible provider route wrappers now delegate only through
    `try_chat_url(...)`, with no direct `chat_url(...)` implementation.
  - Cohere native chat/embedding/rerank route specs and TogetherAI rerank route specs now implement
    only fallible route methods.
  - `migrated_openai_compatible_provider_route_specs_do_not_reintroduce_legacy_string_hooks` and
    `migrated_rerank_provider_route_specs_do_not_reintroduce_legacy_string_hooks` guard those
    migrated provider slices against direct legacy route hook definitions.
  - Ollama chat/model-listing/embedding specs and MiniMaxi chat/image specs now implement only
    fallible route methods.
  - `migrated_local_and_multi_surface_provider_route_specs_do_not_reintroduce_legacy_string_hooks`
    guards the migrated Ollama and MiniMaxi route specs against direct legacy route hook
    definitions.
  - Amazon Bedrock chat/embedding/image/rerank specs and Azure OpenAI chat/embedding/image specs now
    implement only fallible route methods.
  - `migrated_bedrock_and_azure_route_specs_do_not_reintroduce_legacy_string_hooks` guards the
    migrated Bedrock and Azure route specs against direct legacy route hook definitions.
  - Google Vertex generative AI, embedding, Gemini image, and Imagen route specs now implement only
    fallible route methods and no longer call protocol legacy string hooks.
  - `migrated_google_vertex_route_specs_do_not_reintroduce_legacy_string_hooks` guards the migrated
    Google Vertex route specs against direct legacy route hook definitions and calls.
  - OpenAI, Anthropic, Gemini, Azure OpenAI, Groq, DeepSeek, Ollama, Cohere, Amazon Bedrock,
    TogetherAI, MiniMaxi, and Google Vertex provider/protocol specs now implement fallible
    `try_*_url(...)` route hooks directly or delegate to protocol-owned fallible route hooks.
  - `provider_specs_do_not_reintroduce_legacy_string_route_hooks` scans provider/protocol
    `ProviderSpec` impls and rejects historical string-returning route hooks outright.
  - Production request paths for OpenAI, OpenAI-compatible providers, Anthropic models, Gemini
    models, Ollama chat/model listing, and Anthropic-on-Vertex model listing now call
    `try_*_url(...)` instead of string-returning route hooks.
  - `production_request_paths_use_fallible_provider_routes` guards those migrated client/model
    paths against regressing to `*_url(...)`.
  - Facade fixture tests and provider/protocol route assertions now call `try_*_url(...).unwrap()`
    so the test suite exercises the provider-owned fallible route path.
  - `route_fixture_tests_use_fallible_provider_routes` guards facade fixture tests plus
    provider/protocol test modules against reintroducing direct `*_url(...)` calls.
  - Core executor test fixtures and embedded examples now define/call only fallible route hooks, so
    executor-local guidance no longer teaches `chat_url(...)`, `embedding_url(...)`, image route, or
    rerank route hooks as the primary contract.
  - `core_executor_tests_and_docs_use_fallible_route_hooks` guards `siumai-core/src/execution`
    Rust sources against reintroducing legacy string-returning route definitions or calls.
  - `docs/migration/migration-0.11.0-beta.7.md` documents the downstream migration from
    string-returning custom `ProviderSpec` route hooks to fallible `try_*_url(...)` hooks.
- [x] Remove provider-specific compatibility wording from provider-agnostic core docs and examples.
  - `siumai-core` comments and examples no longer describe completion, stream transformers,
    protocol standards, builder helpers, tracing examples, custom-provider examples, or URL helper
    tests as OpenAI-compatible core behavior.
  - `siumai-core::core_provider_boundary_test` now guards those provider-agnostic docs/examples
    against reintroducing OpenAI-compatible wording or concrete OpenAI API URLs.
- [x] Move provider hosted tool factory ownership out of `siumai-core` or document the canonical
  provider-agnostic contract that remains.
  - OpenAI, Anthropic, and Google/Gemini hosted tool constructors moved from
    `siumai-core::hosted_tools` to the matching protocol crates:
    `siumai-protocol-openai::hosted_tools`,
    `siumai-protocol-anthropic::hosted_tools`, and
    `siumai-protocol-gemini::hosted_tools`.
  - Provider crates and the `siumai::hosted_tools::*` facade path now re-export protocol-owned
    constructors; `siumai-core` no longer exposes a `hosted_tools` module.
  - Facade boundary tests reject `pub use siumai_core::hosted_tools` and require protocol-owned
    hosted tool re-exports.
- [x] Move OpenAI Responses stream bridge residue out of `siumai-core`.
- [x] Move provider-specific custom event serialization out of `siumai-core`.
- [x] Keep only provider-agnostic streaming carriers and runtime contracts in `siumai-core`.
  - `SseEventConverter` and `StreamChunkTransformer` now expose
    `is_stream_end_event(&Event) -> bool`, so protocol/provider converters own explicit SSE
    terminal markers.
  - `StreamFactory` now delegates marker recognition to that contract and only owns generic final
    event draining and HTTP metadata propagation.
  - `SseJsonStreamConfig::new(...)` now defaults to no provider/protocol done markers. OpenAI
    speech/transcription SSE helpers configure `[DONE]` explicitly through
    `with_done_markers(...)`.
  - Core HTTP tracing now finalizes streams from the core-owned synthetic `siumai_stream_end` event
    instead of inspecting provider payload text.
  - `siumai-core::streaming::encoder` is now source-guarded to keep provider option/metadata maps
    out of core stream encoding and synthetic terminal event construction.
  - `siumai-core` reasoning extraction middleware is now provider-agnostic: concrete provider/model
    tag routing was removed from core, provider metadata extraction uses generic `thinking` /
    `reasoning` keys, and a source guard prevents provider names returning to the middleware.
  - `siumai-core` system-message warning middleware now receives the provider option namespace from
    caller wiring instead of probing concrete provider fallback names, and `auto.rs` passes the
    configured provider namespace through to the middleware.
  - `siumai-core::execution::middleware::lm::language_model` middleware docs and tests now use
    provider-neutral model/provider IDs with a source guard.
  - `siumai-core::execution::executors::image` test fixtures now use provider-neutral namespace and
    model names instead of concrete provider examples.
  - `siumai-core::utils::chat_request` and `siumai-core::execution::http::headers` tests now use
    provider-neutral namespaces/header names with source guards, while production code remains
    generic over provider option maps and HTTP header names.
  - `siumai-core::execution::executors::files` now source-guards the upload runtime path against
    concrete provider literals or response metadata reads.
  - `siumai-core::retry_api` generic HTTP error classification now collects only provider-agnostic
    request/trace header names; provider-specific request-id headers belong in provider-owned
    `ProviderSpec::classify_http_error(...)` hooks.
  - Remaining core docs/tests and carrier fixtures now use provider-neutral namespaces, model IDs,
    URLs, custom event kinds, and provider references across transformers, traits, telemetry,
    structured output, UI conversion, streaming, and utility modules.
  - `core_provider_boundary_test::core_source_does_not_use_provider_model_fixture_literals` now
    turns the broad `siumai-core/src` provider/model literal scan into a case-insensitive source
    guard across production code, embedded tests, and rustdoc examples.
  - `core_provider_boundary_test::core_stream_factory_does_not_own_provider_sse_end_markers`
    prevents `StreamFactory` from reintroducing concrete provider/protocol SSE end marker checks.
  - `core_provider_boundary_test::core_streaming_runtime_tests_do_not_use_provider_model_fixtures`
    keeps core streaming tests/docs on provider-neutral fixture model names and URLs.
- [x] Run focused `siumai-core`, bridge, protocol, and affected provider tests after each move.
  - Current SSE marker ownership slice verified `siumai-core`, OpenAI/Anthropic/Gemini protocol,
    OpenAI provider, xAI/MiniMaxi checks, and facade stream factory injection tests.
  - Latest provider-neutral core fixture slice verified `siumai-core` lib tests and the broad
    `siumai-core/src` provider/model literal source guard.

## Track E - `LlmClient` Compatibility Reduction

- [x] Audit all `as_*_capability()` call sites.
- [x] Classify each call site as stable family path, extension-only compatibility, or legacy public
  compatibility.
- [x] Remove downcast usage from stable family paths.
  - Primary language, completion, embedding, image generation, speech, transcription, video, and
    rerank family execution paths are guarded against `compat_*_client*` and
    `.as_*_capability()` calls.
  - `siumai-core::ClientWrapper` no longer exposes provider-named constructor aliases; advanced
    boxed-client wrapping uses the provider-agnostic `ClientWrapper::new(...)`.
- [x] Keep extension-only gaps behind explicit `compat_*` or provider extension naming.
  - Image edit/variation, speech/transcription streaming/extras, files, skills, and music remain
    compatibility or extension-only gaps until they become first-class family models.
- [x] Add migration notes for public generic-client paths that remain temporarily available.
  - `docs/migration/migration-0.11.0-beta.7.md` now classifies generic `LlmClient` factory paths as
    migration/extension-only compatibility and points new factories to family methods such as
    `language_model_text_with_ctx(...)`.
  - `docs/architecture/public-surface.md` and `docs/architecture/registry-without-builtins.md` now
    describe registry construction as family-first and keep generic `LlmClient` construction behind
    explicit `compat_*_client(...)` / `compat_*_client_with_ctx(...)` methods.
  - `factory_architecture_boundary_test::public_docs_classify_generic_llm_client_factory_paths_as_migration_only`
    guards those public docs against losing the migration-only classification.

## Track F - Facade And Registry Surface

- [x] Narrow registry re-exports so `siumai-registry` does not mirror implementation crates.
  - `siumai-registry` root now keeps the custom-factory contract surface (`LlmClient`, `LlmError`,
    `error`, `streaming`, `text`, `traits`, `types`) and no longer mirrors broad `siumai-core`
    family/helper modules such as `embedding`, `image`, `video`, `retry_api`, `hosted_tools`, or
    `custom_provider`.
  - `factory_architecture_boundary_test` guards this root surface.
- [x] Narrow facade re-exports so broad type compatibility paths and stable preludes expose intentional stable
  data only.
  - The historical root `siumai::types::*` path was removed. Migration code that intentionally
    needs a broad type namespace now imports `siumai::compat::types::*` or
    `siumai::prelude::compat::types::*`.
  - `siumai/src/lib.rs`, `siumai/src/compat.rs`, and `docs/architecture/public-surface.md` now
    describe the catch-all type namespace as explicit compatibility surface rather than shared
    stable facade data.
  - `prelude::unified` is now guarded against falling back to `siumai_core::types::*`; stable type
    exports must stay an explicit curated list while broad type imports remain under compat.
  - `facade_architecture_boundary_test` prevents legacy root `traits`, `error`, and `streaming`
    modules from returning to the facade.
  - `prelude::unified` no longer mirrors `siumai_core::streaming::*`; stable stream consumption
    types remain there, while converter/factory/encoder/typed-bridge internals are explicitly
    imported from `siumai::experimental::streaming`.
  - Facade stream and transcoding fixture tests now import low-level streaming internals such as
    `SseEventConverter`, `ChatByteStream`, `StreamProcessor`, and `ProcessedEvent` from
    `siumai::experimental::streaming` instead of relying on `prelude::unified::*`.
  - Retry API controls such as `RetryOptions`, `RetryPolicy`, and `retry_with(...)` are no longer
    direct `prelude::unified::*` names; callers import the scoped runtime module
    `siumai::retry_api::*` instead.
  - `prelude::unified` no longer mirrors the whole runtime `tooling` module. Stable AI SDK-style
    tool helper names remain direct prelude imports, while broader runtime tooling APIs use
    `siumai::tooling::*`.
  - `prelude::unified` no longer exports execution middleware internals such as
    `LanguageModelMiddleware`; middleware imports now use
    `siumai::experimental::execution::middleware::*`.
  - `prelude::unified::registry::*` now also exports `BuildContext` and
    `ProviderBuildOverrides`, so custom `ProviderFactory` implementations can use the stable
    family-first `*_with_ctx(...)` method signatures without depending on private registry paths.
  - `ProviderFactory` is now scoped to `prelude::unified::registry::*` instead of the top-level
    `prelude::unified::*`, keeping registry contracts grouped under the registry surface.
  - The unused root `siumai::registry_global` alias has been removed; callers use
    `registry::global()` or `siumai::prelude::unified::registry::global()` explicitly.
  - The unused historical `siumai::prelude::registry::*` mirror has been removed; callers use
    `siumai::prelude::unified::registry::*` or root `siumai::registry::*`.
  - The unused facade root `siumai::provider_catalog::*` mirror has been removed; advanced catalog
    code imports `siumai_registry::provider_catalog::*` explicitly.
  - The OpenAI-compatible provider-list macro is no longer re-exported from the facade root; registry
    or provider glue imports it directly from `siumai_provider_openai_compatible`.
  - File/skill upload helper types and functions are no longer direct `prelude::unified::*` names;
    they remain available through explicit `siumai::files::*`, `siumai::skills::*`, and root helper
    function paths.
  - Non-family extension capability traits and request/response types now have a facade boundary
    guard that keeps them scoped to `siumai::extensions::*` / `prelude::extensions::*` instead of
    direct `prelude::unified::*` imports.
  - Facade file-upload helper tests now import `FileUploadProvider` through `siumai::files`, so the
    stable prelude is no longer required to carry upload-provider internals for tests to compile.
  - Facade file-upload warning compatibility policy is now delegated to a dedicated helper, and the
    main upload flow is source-guarded against reintroducing provider-specific policy literals.
  - Facade skill-upload helper tests now import stable data carriers explicitly through
    `siumai::types`, and `siumai/src/skills.rs` is source-guarded so the payload adapter does not
    grow provider-specific policy branches.
  - OpenAI and Gemini protocol file transformers are now source-guarded so request-side file
    upload helpers stay separate from response-side file object conversion.
  - Anthropic provider-owned file and skill upload resources are now source-guarded so unsupported
    request `provider_options` stay explicit on the request side while response metadata projection
    remains response-only.
  - Anthropic thinking replay is now documented and guarded as an explicit cross-step exception:
    normal thinking request helpers stay request-only, while
    `assistant_message_with_thinking_metadata(...)` is the scoped response-metadata-to-request-options
    bridge for reasoning replay.
  - Gemini chat/embedding/image/video and MiniMaxi chat/structured/thinking/TTS/video request
    extension helpers are now source-guarded so provider-owned request builders cannot start
    reading response metadata.
  - Vertex embedding, Imagen, and video request option extension helpers are now source-guarded
    with the same request-only boundary.
  - `parse_json_event_stream(...)` is no longer a direct `prelude::unified::*` name; explicit root
    helper and `experimental::streaming` paths remain available for JSON/SSE parser use cases.
  - Low-level utility helper families such as download/header/settings/JSON parse/provider-option
    helpers are no longer direct `prelude::unified::*` names; explicit facade root imports remain
    available for opt-in utility users.
  - `prelude::unified` no longer exports deprecated AI SDK parity names such as `CallSettings`,
    `Experimental_*` result aliases, `experimental_filter_active_tools`, or `step_count_is`.
    Those migration spellings now live in `siumai::compat` / `prelude::compat`.
- [x] Move compatibility-only names under `siumai::compat`, `prelude::compat`, or documented
  `experimental` paths.
  - Current high-risk unified-prelude aliases are guarded by an allowlist and compatibility audit
    entry before any breaking split.
  - `StreamingToolCall*` helper aliases have moved out of `prelude::unified` and the facade root;
    migration imports now use `siumai::compat` / `prelude::compat`.
  - Provider-specific builder construction is now explicitly available from `siumai::compat::Provider`
    and `siumai::prelude::compat::Provider`; the historical root `siumai::Provider` alias has been
    removed so builder-style construction is explicit compat-only.
  - The `Provider` implementation body now lives under `siumai::compat`; the facade root no longer
    re-exports it.
  - `siumai::compat::{Siumai,SiumaiBuilder}` now re-export registry-owned types directly, and
    the root `siumai::provider::*` builder-era compatibility shim has been removed.
  - `siumai::compat::builder::*` now re-exports core builder base internals directly, and the root
    `siumai::builder::*` shim has been removed.
  - Facade production code routes upload helper impls through `crate::compat::Siumai`, and no
    facade source file routes through `crate::provider::{Siumai,SiumaiBuilder}`.
  - Facade tests and examples no longer use `siumai::provider::{Siumai,SiumaiBuilder}` directly;
    the boundary guard keeps future coverage on explicit `siumai::compat` or stable registry paths
    and prevents the removed root shim from returning.
  - Runnable examples are also guarded against reintroducing the removed root `siumai::Provider`
    alias as a taught construction path.
  - Ordinary facade tests now use `siumai::compat::Provider` for builder-era construction coverage;
    the root `siumai::Provider` alias is no longer allowlisted.
  - The large `provider_public_path_parity_test` suite now imports `siumai::compat::Provider`
    instead of the facade root alias, and `public_surface_imports_test` now does the same.
  - Provider extension package helpers that return `SiumaiBuilder` now bind directly to the
    registry-owned builder instead of routing through `crate::provider::SiumaiBuilder`.
  - Provider extension helpers no longer call the facade root `crate::Provider` alias; remaining
    centralized builder construction goes through explicit `crate::compat::Provider`.
  - `siumai::compat::Provider` no longer routes provider builders through `crate::builder`; it
    imports `siumai_core::builder::BuilderBase` directly.
  - Execution middleware contracts are documented as `experimental` instead of stable unified
    prelude exports.
  - Deprecated AI SDK parity names and helper spellings are now explicit compat imports rather
    than stable unified prelude names.
- [x] Update `docs/architecture/public-surface.md` when stable facade exports change.
  - Latest Track F slice documents removal of root `StreamingToolCall*` aliases and the explicit
    `siumai::compat` / `prelude::compat` migration paths.
  - Latest Track F provider parity cleanup documents that large parity tests use the explicit
    `siumai::compat::Provider` path.
  - Latest Track F root Provider cleanup documents removal of root `siumai::Provider` and the
    explicit `siumai::compat::Provider` migration path.
  - Latest Track F provider shim cleanup documents removal of root `siumai::provider::*` and the
    explicit `siumai::compat::{Siumai,SiumaiBuilder}` migration path.
  - Latest Track F builder shim cleanup documents removal of root `siumai::builder::*` and the
    explicit `siumai::compat::builder::*` migration path.
  - Latest Track F registry-global cleanup documents removal of root `siumai::registry_global` and
    the scoped `registry::global()` replacement.
  - Latest Track F prelude-registry cleanup documents removal of the historical
    `siumai::prelude::registry::*` mirror and the unified/root registry replacements.
  - Latest Track F provider-catalog cleanup documents removal of root
    `siumai::provider_catalog::*` and the registry-owned replacement path.
  - Latest Track F OpenAI-compatible provider-list macro cleanup documents removal of the facade
    root macro re-export and the provider-owned replacement path.
  - Latest Track F low-level utility prelude cleanup documents explicit root imports for
    download/header/settings/JSON parse/provider-option helper names.
- [x] Update migration docs for any public breaking removals.
  - `migration-0.11.0-beta.7.md` now includes the root `StreamingToolCall*` alias removal and
    replacement imports.
  - `migration-0.11.0-beta.7.md` now includes the root `siumai::provider::*` shim removal and
    replacement imports.
  - `migration-0.11.0-beta.7.md` now includes the root `siumai::builder::*` shim removal and
    replacement imports.
  - `migration-0.11.0-beta.7.md` now includes the root `siumai::registry_global` alias removal and
    replacement import/call paths.
  - `migration-0.11.0-beta.7.md` now includes the historical
    `siumai::prelude::registry::*` mirror removal and replacement import paths.
  - `migration-0.11.0-beta.7.md` now includes the root `siumai::provider_catalog::*` mirror removal
    and replacement import path.
  - `migration-0.11.0-beta.7.md` now includes the root
    `siumai::siumai_for_each_openai_compatible_provider` re-export removal and direct provider-crate
    import path.
  - `migration-0.11.0-beta.7.md` now includes the low-level utility helper removal from
    `prelude::unified::*` and the explicit root import path.

## Track G - Documentation And Final Validation

- [x] Keep `design.md`, `milestones.md`, and this TODO updated after each completed slice.
- [x] Update architecture docs when crate ownership changes.
- [x] Run `cargo fmt` for touched Rust crates.
  - Latest error-policy slice formatted `siumai-spec`, `siumai-core`,
    `siumai-protocol-anthropic`, `siumai-extras`, and `siumai`.
  - Latest facade/prelude import-cleanup slice verified `cargo fmt --package siumai --check`.
  - Latest provider-neutral core fixture slice verified `cargo fmt --package siumai-core --check`.
  - Latest bridge customization guard slice verified `cargo fmt --package siumai-bridge --check`.
  - Latest core stream helper guard slice verified `cargo fmt --package siumai-core --check`.
  - Latest Anthropic cache request guard slice verified
    `cargo fmt --package siumai-protocol-anthropic --check`.
  - Latest OpenAI-compatible Alibaba cache request guard slice verified
    `cargo fmt --package siumai-protocol-openai --check`.
  - Latest Gemini tools extension guard slice verified
    `cargo fmt --package siumai-provider-gemini --check`.
  - Latest Gemini modality request extension guard slice verified
    `cargo fmt --package siumai-provider-gemini --check`.
  - Latest MiniMaxi extension request guard slice verified
    `cargo fmt --package siumai-provider-minimaxi --check`.
  - Latest structured-output provider helper guard slice verified
    `cargo fmt --package siumai-provider-anthropic --package siumai-provider-openai --check`.
  - Latest Bedrock extension guard slice verified
    `cargo fmt --package siumai-provider-amazon-bedrock --check`.
  - Latest Anthropic request option extension guard slice verified
    `cargo fmt --package siumai-provider-anthropic --package siumai-provider-google-vertex --check`.
  - Latest OpenAI-compatible spec request-option guard slice verified
    `cargo fmt --package siumai-protocol-openai --check`.
  - Latest core streaming tool-call metadata callback guard slice verified
    `cargo fmt --package siumai-core --check`.
  - Latest core structured-output metadata merge guard slice verified
    `cargo fmt --package siumai-core --check`.
  - Latest Anthropic thinking/core JSON stream guard slice verified
    `cargo fmt --package siumai-core --package siumai-protocol-anthropic --check`.
  - Latest core family/tooling/provider-options guard slice verified
    `cargo fmt --package siumai-core --check`.
  - Latest Track C audit coverage guard slice verified `cargo fmt --package siumai --check`.
  - Latest Track F non-family extension prelude guard slice verified
    `cargo fmt --package siumai --check`.
  - Latest Track F broad facade type glob guard slice verified
    `cargo fmt --package siumai --check`.
  - Closeout verified:
    - `cargo fmt --package siumai-spec --package siumai-core --package siumai-registry --package siumai --check`
- [x] Run focused `cargo nextest` validation for affected crates.
  - Latest error-policy slice verified `siumai-spec`, `siumai-core`, and
    `siumai-protocol-anthropic` focused tests.
  - Latest facade/prelude import-cleanup slice verified `siumai` integration, file-upload helper,
    and Gemini stream-bridge fixture tests.
  - Latest provider-neutral core fixture slice verified
    `cargo nextest run -p siumai-core --lib --no-default-features --no-fail-fast`.
  - Latest provider-neutral core fixture slice also verified
    `cargo nextest run -p siumai-core --test core_provider_boundary_test core_source_does_not_use_provider_model_fixture_literals --no-default-features --no-fail-fast`.
  - Latest bridge customization guard slice verified
    `cargo nextest run -p siumai-bridge --lib bridge_customization_source_stays_primitive_only --no-default-features --no-fail-fast`.
  - Latest bridge customization guard slice also verified
    `cargo nextest run -p siumai-bridge --lib --no-default-features --no-fail-fast`.
  - Latest core stream helper guard slice verified
    `cargo nextest run -p siumai-core --test core_provider_boundary_test core_stream_helpers_only_initialize_empty_provider_metadata --no-default-features --no-fail-fast`.
  - Latest core stream helper guard slice also verified
    `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`.
  - Latest Anthropic cache request guard slice verified
    `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard cache_request_builder_source_does_not_read_legacy_provider_metadata --no-fail-fast`.
  - Latest Anthropic cache request guard slice also verified
    `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard standards::anthropic::cache --no-fail-fast`.
  - Latest OpenAI-compatible Alibaba cache request guard slice verified
    `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard alibaba_cache_control_source_does_not_read_legacy_provider_metadata --no-fail-fast`.
  - Latest OpenAI-compatible Alibaba cache request guard slice also verified
    `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard standards::openai::compat::alibaba_cache_control --no-fail-fast`.
  - Latest Gemini tools extension guard slice verified
    `cargo nextest run -p siumai-provider-gemini --no-default-features --features google gemini_tools_extension_source_does_not_read_request_provider_options --no-fail-fast`.
  - Latest Gemini tools extension guard slice also verified
    `cargo nextest run -p siumai-provider-gemini --no-default-features --features google providers::gemini::ext::tools --no-fail-fast`.
  - Latest Gemini modality request extension guard slice verified
    `cargo nextest run -p siumai-provider-gemini --lib providers::gemini::ext --no-default-features --features google --no-fail-fast`.
  - Latest MiniMaxi extension request guard slice verified
    `cargo nextest run -p siumai-provider-minimaxi --lib providers::minimaxi::ext --no-default-features --features minimaxi --no-fail-fast`.
  - Latest structured-output provider helper guard slice verified:
    - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::structured_output --no-default-features --features anthropic --no-fail-fast`
    - `cargo nextest run -p siumai-provider-openai --lib providers::openai::structured_output --no-default-features --features openai --no-fail-fast`
  - Latest Bedrock extension guard slice verified
    `cargo nextest run -p siumai-provider-amazon-bedrock --lib providers::bedrock::ext --no-default-features --features bedrock --no-fail-fast`.
  - Latest Anthropic request option extension guard slice verified:
    - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::request_options --no-default-features --features anthropic --no-fail-fast`
    - `cargo nextest run -p siumai-provider-google-vertex --lib providers::anthropic_vertex::ext::request_options --no-default-features --features google-vertex --no-fail-fast`
  - Latest OpenAI-compatible spec request-option guard slice verified:
    - `cargo nextest run -p siumai-protocol-openai --lib standards::openai::compat::spec::tests::openai_compatible_spec_request_option_source_does_not_read_response_metadata --no-default-features --features openai-standard --no-fail-fast`
    - `cargo nextest run -p siumai-protocol-openai --lib standards::openai::compat::spec --no-default-features --features openai-standard --no-fail-fast`
  - Latest core streaming tool-call metadata callback guard slice verified:
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_streaming_tool_call_tracker_only_uses_callback_provider_metadata --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - Latest core structured-output metadata merge guard slice verified:
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_structured_output_helpers_only_merge_generic_response_metadata --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - Latest Anthropic thinking/core JSON stream guard slice verified:
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_json_stream_executor_does_not_handle_provider_maps --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-protocol-anthropic --lib standards::anthropic::thinking --no-default-features --features anthropic-standard --no-fail-fast`
  - Latest core family/tooling/provider-options guard slice verified:
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_family_contract_and_tooling_sources_do_not_handle_provider_maps core_sample_streaming_middleware_only_initializes_empty_provider_metadata core_provider_options_parser_stays_request_only_and_provider_agnostic --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - Latest custom provider guide cleanup verified:
    - `cargo fmt --package siumai-core --check`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test core_provider_agnostic_docs_do_not_describe_core_as_openai_compatible core_source_does_not_use_provider_model_fixture_literals --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - Latest provider client/spec request-option guard slice verified:
    - `cargo fmt --package siumai-provider-anthropic --package siumai-provider-openai --package siumai-provider-google-vertex --package siumai-provider-minimaxi --check`
    - `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::spec::tests::anthropic_spec_request_option_routing_does_not_read_response_metadata providers::anthropic::client::tests::anthropic_client_middleware_request_option_checks_do_not_read_response_metadata --no-default-features --features anthropic --no-fail-fast`
    - `cargo nextest run -p siumai-provider-openai --lib providers::openai::spec::tests::openai_spec_request_option_routing_does_not_read_response_metadata providers::openai::spec::tests::openai_spec_provider_metadata_key_selection_does_not_read_request_options providers::openai::client::tests::openai_client_default_provider_option_merging_does_not_read_response_metadata --no-default-features --features openai --no-fail-fast`
    - `cargo nextest run -p siumai-provider-google-vertex --lib providers::anthropic_vertex::client::tests::vertex_anthropic_client_default_provider_options_do_not_read_response_metadata --no-default-features --features google-vertex --no-fail-fast`
    - `cargo nextest run -p siumai-provider-minimaxi --lib providers::minimaxi::client::tests::minimaxi_client_request_option_merging_does_not_read_response_metadata --no-default-features --features minimaxi --no-fail-fast`
    - `cargo nextest run -p siumai-provider-gemini --lib providers::gemini::client --no-default-features --features google --no-fail-fast`
  - Latest facade provider-map direction guard slice verified:
    - `cargo fmt --package siumai --check`
    - `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_macros_only_create_request_side_empty_provider_options facade_audio_and_structured_helpers_do_not_read_request_provider_options facade_video_metadata_projection_avoids_legacy_request_provider_options --no-default-features --features openai,anthropic,google --no-fail-fast`
    - `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-default-features --features openai,anthropic,google --no-fail-fast`
  - Latest Track C audit coverage guard slice verified:
    - `cargo nextest run -p siumai --test facade_architecture_boundary_test content_part_provider_map_audit_covers_high_value_production_hits --no-default-features --features openai,anthropic,google --no-fail-fast`
  - Latest Track F non-family extension prelude guard slice verified:
    - `cargo nextest run -p siumai --test facade_architecture_boundary_test stable_unified_prelude_keeps_non_family_extension_types_scoped --no-default-features --features openai,anthropic,google --no-fail-fast`
  - Latest Track F broad facade type glob guard slice verified:
    - `cargo nextest run -p siumai --test facade_architecture_boundary_test broad_facade_types_path_is_explicit_compat_only --no-default-features --features openai,anthropic,google --no-fail-fast`
  - Closeout verified:
    - `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --test content_projection_boundary_test --test spec_purity_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`
    - `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- [x] Run `git diff --check` before closing the workstream.
  - Latest provider-neutral core fixture slice verified `git diff --check -- <touched siumai-core files>`;
    only existing LF/CRLF warnings were reported.
  - Closeout verified `git diff --check` with no whitespace errors.
