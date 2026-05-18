# Track C Direct ContentPart Construction Audit

Last updated: 2026-05-16

This audit tracks direct `ContentPart` construction and legacy
`provider_options` / `provider_metadata` usage while Track C converges prompt-side and
response-side content boundaries.

## Purpose

The legacy stable `ContentPart` carrier still has variants that contain both request-side
`providerOptions` and response-side `providerMetadata`. Until a broader non-V4 prompt/content
projection exists, every direct construction path should be classified so request construction does
not learn from response metadata by accident.

## Search Baseline

Current review baseline:

```text
rg -l "ContentPart::|provider_metadata:|provider_options:" \
  siumai-spec/src siumai-core/src siumai-bridge/src \
  siumai-protocol-openai/src siumai-protocol-anthropic/src siumai-protocol-gemini/src \
  siumai-provider-amazon-bedrock/src siumai-provider-anthropic/src \
  siumai-provider-google-vertex/src siumai-provider-minimaxi/src \
  siumai-provider-openai/src siumai-provider-gemini/src siumai/src -g "*.rs"

rg -n "ContentPart::(Text|Image|Audio|File|Reasoning|ReasoningFile|ToolCall|ToolResult|Custom|Source|ToolApproval)|provider_metadata:|provider_options:" \
  siumai-core/src siumai-bridge/src siumai-protocol-openai/src \
  siumai-protocol-anthropic/src siumai-protocol-gemini/src \
  siumai-provider-amazon-bedrock/src siumai-provider-anthropic/src \
  siumai-provider-google-vertex/src siumai-provider-minimaxi/src \
  siumai-provider-openai/src siumai-provider-gemini/src -g "*.rs"
```

The raw search intentionally overmatches type definitions, inline tests, provider option type docs,
and response-only typed metadata helpers. This file classifies the high-value production paths so
future passes do not repeat the same broad scan.

`facade_architecture_boundary_test::content_part_provider_map_audit_covers_high_value_production_hits`
now replays the high-value subset of this scan in CI. Any production source under the core, bridge,
protocol, provider, or facade crates that directly constructs `ContentPart` values or provider-map
fields must be listed in the guarded-path table, or it must fit one of the low-priority buckets
below. This keeps the audit from becoming stale while the legacy dual-use carrier still exists.

## Classification Rules

- Request serializers may read `provider_options` / `providerOptions`.
- Request serializers must not read legacy `ContentPart::provider_metadata` or
  `providerMetadata` JSON fields for replay, cache, detail, or vendor knobs.
- Response parsers and stream replay may emit `provider_metadata` / `providerMetadata`.
- Response parsers and stream replay must not emit request-side `providerOptions` except for the
  empty default required by legacy `ContentPart` response variants.
- Provider metadata typed extension modules are response-side views only.
- Cross-step workflow helpers may read response-level `ProviderMetadataMap` only when they emit
  next-step request-side `ProviderOptionsMap` and do not touch legacy `ContentPart` metadata.

## Guarded Paths

| Path | Classification | Guard evidence |
| --- | --- | --- |
| `siumai-spec/src/types/prompt.rs` | request-side non-V4 stable prompt projection | `content_projection_boundary_test` keeps prompt projection items free of response metadata, exposes named projection helpers, rejects legacy response metadata on narrowing, and checks prompt-to-legacy conversions emit no response metadata |
| `siumai-spec/src/types/ai_sdk/generate_text.rs` | response-side non-V4 generated text content projection | `content_projection_boundary_test` keeps `GenerateTextContentPart` and final reasoning output projection free of request-side provider options, and verifies the named response projection helpers preserve metadata while rejecting ambiguous legacy carriers |
| `siumai-spec/src/types/ai_sdk/output_parts.rs` | response-side non-V4 generated output part projection | `content_projection_boundary_test` keeps output parts free of request-side provider options and verifies response-side metadata carriers stay explicit |
| `siumai/src/text.rs` | facade generateText result projection | `facade_architecture_boundary_test::generate_text_projection_delegates_content_part_mapping_to_spec` keeps facade output projection delegated to the spec-owned response projection helper, with only the documented legacy tool-result-without-input fallback left local |
| `siumai/src/files.rs` | facade file upload helper flow and compatibility warning projection | `upload_via_file_management_keeps_provider_policy_delegated_to_helpers` keeps provider-specific upload policy out of the main facade flow and delegated to helper/provider-owned paths |
| `siumai/src/skills.rs` | facade skill upload helper payload adapter | `upload_helper_keeps_provider_policy_delegated_to_api` keeps provider-specific skill upload policy out of the facade payload adapter and delegated to provider APIs |
| `siumai-spec/src/types/ai_sdk/language_model_v4/prompt.rs` | request-side V4 prompt projection | `ai_sdk_module_boundary_test` rejects response metadata terms in prompt projection |
| `siumai-spec/src/types/ai_sdk/language_model_v4/content.rs` | response-side V4 generated content projection | `ai_sdk_module_boundary_test` rejects request options terms in content projection |
| `siumai-core/src/ui.rs` | UI request adapter | UI tests keep AI SDK UI metadata normalized into request `provider_options` and centralize legacy construction |
| `siumai-core/src/utils/chat_request.rs` | provider-agnostic chat request normalization | `chat_request_tests_use_provider_neutral_option_namespaces` keeps default/request provider options merge tests on neutral namespaces while production code treats the map generically |
| `siumai-core/src/execution/middleware/presets/extract_reasoning.rs` | provider-agnostic reasoning extraction middleware | `extract_reasoning_middleware_source_stays_provider_agnostic` keeps concrete provider/model routing out of core and extracts metadata from generic keys only |
| `siumai-core/src/execution/middleware/presets/system_message_mode_warning.rs` | provider-agnostic request warning middleware | `system_message_mode_warning_source_stays_provider_agnostic` keeps concrete provider fallback namespaces out of core and reads only the injected provider option namespace |
| `siumai-core/src/execution/middleware/auto.rs` | provider-agnostic middleware wiring | `automatic_middleware_source_stays_provider_agnostic` keeps automatic middleware selection from hard-coding concrete providers or models and passes the configured provider option namespace into request warning middleware |
| `siumai-core/src/execution/executors/files.rs` | provider-agnostic files HTTP executor | `files_executor_upload_path_stays_provider_agnostic` keeps concrete provider literals and response metadata out of the core upload runtime |
| `siumai-bridge/src/request/*` | protocol-body to `ChatRequest` request normalization | request normalization guards reject legacy provider metadata reads and centralize request constructors |
| `siumai-bridge/src/request/normalize.rs` | protocol-body to `ChatRequest` request normalization implementation | covered by bridge request normalization guards |
| `siumai-bridge/src/request/primitives.rs` | request-side bridge primitive helpers | covered by bridge request normalization guards |
| `siumai-bridge/src/request/pairs/*` | direct cross-protocol request bridge pairs | `request_bridge_pair_sources_do_not_read_legacy_provider_metadata` keeps OpenAI Responses ↔ Anthropic Messages pair bridges request-side |
| `siumai-bridge/src/request/pairs/anthropic_messages_to_openai_responses.rs` | Anthropic Messages to OpenAI Responses request bridge pair | `request_bridge_pair_sources_do_not_read_legacy_provider_metadata` |
| `siumai-bridge/src/request/pairs/openai_responses_to_anthropic_messages.rs` | OpenAI Responses to Anthropic Messages request bridge pair | `request_bridge_pair_sources_do_not_read_legacy_provider_metadata` |
| `siumai-bridge/src/customize.rs` | primitive-only bridge customization/remapping shared by request, response, and stream paths | `bridge_customization_source_stays_primitive_only` keeps built-in remappers away from request/response provider maps |
| `siumai-bridge/src/response/*`, `siumai-bridge/src/stream/*` | response serialization and stream replay | `response_and_stream_bridge_sources_do_not_emit_request_provider_options` |
| `siumai-bridge/src/response/inspect.rs` | bridge response capability inspection | covered by response/stream bridge source guard |
| `siumai-protocol-openai/src/standards/openai/utils.rs` | OpenAI Chat/OpenAI-compatible request serializers | `openai_chat_request_conversion_source_does_not_read_legacy_provider_metadata_fields` |
| `siumai-protocol-openai/src/standards/openai/compat/alibaba_cache_control.rs` | OpenAI-compatible Alibaba/Qwen request-side cache-control mapper | `alibaba_cache_control_source_does_not_read_legacy_provider_metadata` keeps cache-control request shaping on request provider options |
| `siumai-protocol-openai/src/standards/openai/compat/transformers.rs` | mixed OpenAI-compatible request/response transformer | request metadata-read guard plus response provider-options emission guard |
| `siumai-protocol-openai/src/standards/openai/compat/streaming.rs` | OpenAI-compatible streaming response/replay converter | `openai_compatible_streaming_source_does_not_emit_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/transformers/request/responses.rs` | OpenAI Responses request serializer | request transformer source guard rejects legacy provider metadata reads and bound `provider_metadata` |
| `siumai-protocol-openai/src/standards/openai/transformers/response/responses.rs` | OpenAI Responses response parser | `responses_response_transformer_source_does_not_emit_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/files.rs` | OpenAI file upload/list/retrieve transformer | `openai_files_request_and_response_transformers_keep_maps_directional` |
| `siumai-protocol-openai/src/standards/openai/responses_sse/converter/*` | OpenAI Responses SSE response/stream converter | `responses_sse_converter_sources_do_not_read_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/responses_sse/converter/convert.rs` | OpenAI Responses SSE response/stream converter content projection | `responses_sse_converter_sources_do_not_read_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/responses_sse/converter/serialize.rs` | OpenAI Responses SSE response/stream serializer | `responses_sse_converter_sources_do_not_read_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/responses_sse/converter/sse.rs` | OpenAI Responses SSE event converter | `responses_sse_converter_sources_do_not_read_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/responses_sse/converter/tool_events.rs` | OpenAI Responses SSE tool-event projection helper | `responses_sse_converter_sources_do_not_read_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/json_response.rs` | OpenAI gateway/proxy JSON response encoder | `openai_json_response_encoder_source_does_not_read_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/completion_request.rs` | OpenAI legacy completions request serializer | completion prompt materialization accepts text-only legacy content and reads request-side provider options only when building completion request bodies |
| `siumai-protocol-openai/src/standards/openai/completion_metadata.rs` | OpenAI legacy completions response metadata helper | response-only provider metadata extraction for raw logprobs/sources and stream finish metadata |
| `siumai-protocol-openai/src/provider_metadata/openai.rs` | protocol-owned response-side typed metadata view | `openai_provider_metadata_source_does_not_read_request_provider_options` |
| `siumai-protocol-openai/src/standards/openai/compat/spec.rs` | OpenAI-compatible request routing and provider option normalization | `openai_compatible_spec_request_option_source_does_not_read_response_metadata` |
| `siumai-provider-openai/src/providers/openai/ext/responses.rs` | provider extension helper for OpenAI Responses stream/custom event projection | `responses_stream_event_projection_source_does_not_read_request_provider_options` |
| `siumai-protocol-gemini/src/standards/gemini/convert.rs` | Gemini request serializer | request conversion source guard keeps thought replay on request `provider_options` |
| `siumai-protocol-gemini/src/standards/gemini/transformers/request.rs` | Gemini request transformer | `gemini_request_transformer_source_does_not_read_response_provider_metadata` |
| `siumai-protocol-gemini/src/standards/gemini/transformers/response.rs` | Gemini response parser | `gemini_response_content_source_does_not_emit_request_provider_options` |
| `siumai-protocol-gemini/src/standards/gemini/transformers/files.rs` | Gemini file upload/list/retrieve transformer | `gemini_files_request_and_response_paths_keep_maps_directional` |
| `siumai-protocol-gemini/src/standards/gemini/streaming/mod.rs` | Gemini streaming response parser | `gemini_streaming_parser_source_does_not_read_request_provider_options` |
| `siumai-protocol-gemini/src/standards/gemini/streaming/serialize.rs` | Gemini streaming response serializer | `gemini_streaming_serialize_source_does_not_read_request_provider_options` |
| `siumai-protocol-gemini/src/standards/gemini/json_response.rs` | Gemini gateway/proxy JSON response encoder | `gemini_json_response_encoder_source_does_not_read_request_provider_options` |
| `siumai-provider-gemini/src/providers/gemini/interactions.rs` | provider-owned Google Interactions language-model shell and inline behavior tests | production shell delegates request, response, runtime, and stream mapping to split Interactions modules; inline tests cover signature/history replay and agent/model request boundaries |
| `siumai-provider-gemini/src/providers/gemini/interactions/request.rs` | provider-owned Google Interactions request serializer | request conversion reads request-side `provider_options` plus the explicit Interactions signature/id replay metadata needed by the `/interactions` API; `google_interactions_request_*` tests cover model, agent, tool, media, signature, and previous-interaction shaping |
| `siumai-provider-gemini/src/providers/gemini/interactions/response.rs` | provider-owned Google Interactions response parser | response parsing emits stable `ContentPart` values with empty request provider options and Google response metadata; `google_interactions_response_*` tests cover text, reasoning, tools, files, sources, usage, finish reason, and provider metadata |
| `siumai-provider-gemini/src/providers/gemini/interactions/runtime.rs` | provider-owned Google Interactions non-stream execution and polling | runtime parses request provider options for polling controls but delegates content construction to request/response helpers; `google_interactions_non_stream_*` tests cover POST, polling, missing id, and timeout behavior |
| `siumai-provider-gemini/src/providers/gemini/interactions/stream.rs` | provider-owned Google Interactions SSE parser and resumable agent stream runtime | stream conversion emits response-side stable parts/metadata and delegates request body shaping to the request helper; `google_interactions_stream*` tests cover model streaming, agent reconnect, cancellation, sources, tool parts, usage, and finish metadata |
| `siumai-provider-gemini/src/providers/gemini/ext/request_options.rs` | provider-owned Gemini request option extension helper | `gemini_request_option_extension_source_does_not_read_response_metadata` |
| `siumai-provider-gemini/src/providers/gemini/ext/embedding_options.rs` | provider-owned Google embedding request option extension helper | `google_embedding_request_extension_source_does_not_read_response_metadata` |
| `siumai-provider-gemini/src/providers/gemini/ext/image_options.rs` | provider-owned Gemini/Google image request option extension helper | `gemini_image_request_extension_source_does_not_read_response_metadata` |
| `siumai-provider-gemini/src/providers/gemini/ext/video_options.rs` | provider-owned Google video request option extension helper | `google_video_request_extension_source_does_not_read_response_metadata` |
| `siumai-provider-gemini/src/providers/gemini/ext/tools.rs` | provider-owned Gemini tools response/stream extension view | `gemini_tools_extension_source_does_not_read_request_provider_options` keeps source/custom-event projection away from request options |
| `siumai-protocol-anthropic/src/standards/anthropic/utils/messages.rs` | Anthropic message request serializer | request conversion source guard keeps reasoning replay on request `provider_options` |
| `siumai-protocol-anthropic/src/standards/anthropic/utils/content.rs` | Anthropic content request serializer | `request_content_conversion_source_does_not_read_legacy_provider_metadata_fields` |
| `siumai-protocol-anthropic/src/standards/anthropic/cache.rs` | Anthropic request-side prompt cache block builder | `cache_request_builder_source_does_not_read_legacy_provider_metadata` prevents cache request construction from reading response metadata |
| `siumai-protocol-anthropic/src/standards/anthropic/transformers.rs` | mixed Anthropic request/response transformer | request-side metadata-read guard plus response-side provider-options read guard |
| `siumai-protocol-anthropic/src/standards/anthropic/utils/parse.rs` | Anthropic response parser | `anthropic_parse_response_content_source_does_not_emit_request_provider_options` |
| `siumai-protocol-anthropic/src/standards/anthropic/streaming/mod.rs` | Anthropic streaming response parser | `anthropic_streaming_parser_source_does_not_read_request_provider_options` |
| `siumai-protocol-anthropic/src/standards/anthropic/streaming/serialize.rs` | Anthropic streaming response serializer | `anthropic_streaming_serialize_source_does_not_read_request_provider_options` |
| `siumai-protocol-anthropic/src/standards/anthropic/json_response.rs` | Anthropic gateway/proxy JSON response encoder | `anthropic_json_response_encoder_source_does_not_read_request_provider_options` |
| `siumai-protocol-anthropic/src/standards/anthropic/thinking.rs` | mixed Anthropic thinking request config and response metadata projection helper | split guards keep request config away from response metadata and response projection away from request provider options |
| `siumai-provider-google-vertex/src/standards/vertex_gemini_image.rs` | provider-owned synthetic Gemini image request builder | request file-part construction guard plus behavior test |
| `siumai-provider-google-vertex/src/providers/vertex/video.rs` | Google Vertex video task request/body and response metadata helper | `video_request_and_response_paths_keep_provider_maps_directional` |
| `siumai-provider-google-vertex/src/providers/vertex/ext/embedding.rs` | provider-owned Vertex embedding request option extension helper | `vertex_embedding_request_option_extension_source_does_not_read_response_metadata` |
| `siumai-provider-google-vertex/src/providers/vertex/ext/imagen.rs` | provider-owned Vertex Imagen request option extension helper | `vertex_imagen_request_option_extension_source_does_not_read_response_metadata` |
| `siumai-provider-google-vertex/src/providers/vertex/ext/video.rs` | provider-owned Vertex video request option extension helper | `vertex_video_request_option_extension_source_does_not_read_response_metadata` |
| `siumai-provider-google-vertex/src/providers/anthropic_vertex/ext/request_options.rs` | provider-owned Anthropic-on-Vertex request option extension helper | `vertex_anthropic_request_option_extension_source_does_not_read_response_metadata` |
| `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` | mixed Bedrock request/response transformer | request-side metadata read guard and response/stream provider-options emission guard |
| `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/streaming.rs` | Bedrock streaming response parser | covered by `response_and_stream_source_do_not_emit_request_provider_options`; stream code may emit response metadata and empty legacy provider option defaults only |
| `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/tests.rs` | Bedrock chat transformer and streaming tests | inline guard and behavior tests for request/response provider-map directionality |
| `siumai-provider-amazon-bedrock/src/standards/bedrock/embedding.rs` | mixed Bedrock embedding request/response transformer | `embedding_request_and_response_transformers_keep_provider_maps_directional` |
| `siumai-provider-amazon-bedrock/src/providers/bedrock/ext.rs` | provider-owned Bedrock request helpers plus explicit reasoning metadata replay bridge | request-only guard plus `bedrock_reasoning_replay_bridge_stays_explicit_cross_step_exception` |
| `siumai-provider-anthropic/src/providers/anthropic/prepare_step.rs` | response-level workflow metadata to request options | `prepare_step_source_only_bridges_response_metadata_to_request_provider_options` |
| `siumai-provider-anthropic/src/providers/anthropic/files.rs` | provider-owned Anthropic file upload/list/retrieve extension | `anthropic_files_request_and_response_paths_keep_maps_directional` |
| `siumai-provider-anthropic/src/providers/anthropic/skills.rs` | provider-owned Anthropic skill upload extension | `anthropic_skills_request_and_response_paths_keep_maps_directional` |
| `siumai-provider-anthropic/src/providers/anthropic/ext/chat_message.rs` | provider-owned Anthropic chat-message request option extension helper | `anthropic_chat_message_extension_source_stays_request_side` |
| `siumai-provider-anthropic/src/providers/anthropic/ext/request_options.rs` | provider-owned Anthropic request option extension helper | `anthropic_request_option_extension_source_does_not_read_response_metadata` |
| `siumai-provider-anthropic/src/providers/anthropic/ext/structured_output.rs` | provider-owned Anthropic structured-output request helper | `anthropic_structured_output_extension_source_does_not_read_response_metadata` |
| `siumai-provider-anthropic/src/providers/anthropic/ext/thinking.rs` | explicit Anthropic thinking metadata replay bridge | request-only guard plus `thinking_replay_bridge_stays_explicit_cross_step_exception` |
| `siumai-provider-anthropic/src/providers/anthropic/ext/tools.rs` | request tool-options builder plus response/stream tool metadata projection | two direction guards split request builder from stream projection |
| `siumai-provider-minimaxi/src/providers/minimaxi/spec.rs` | Anthropic-protocol adapter with response metadata re-keying and request options alias | two direction guards split metadata normalization from request option resolution |
| `siumai-provider-minimaxi/src/providers/minimaxi/ext/request_options.rs` | provider-owned MiniMaxi request option extension helper | `minimaxi_request_option_extension_source_does_not_read_response_metadata` |
| `siumai-provider-minimaxi/src/providers/minimaxi/ext/structured_output.rs` | provider-owned MiniMaxi structured-output request helper | `minimaxi_structured_output_extension_source_does_not_read_response_metadata` |
| `siumai-provider-minimaxi/src/providers/minimaxi/ext/thinking.rs` | provider-owned MiniMaxi thinking request helper | `minimaxi_thinking_extension_source_does_not_read_response_metadata` |
| `siumai-provider-minimaxi/src/providers/minimaxi/ext/tts_options.rs` | provider-owned MiniMaxi TTS request option extension helper | `minimaxi_tts_request_extension_source_does_not_read_response_metadata` |
| `siumai-provider-minimaxi/src/providers/minimaxi/ext/tts.rs` | provider-owned MiniMaxi TTS request builder | `minimaxi_tts_builder_source_does_not_read_response_metadata` |
| `siumai-provider-minimaxi/src/providers/minimaxi/ext/video_options.rs` | provider-owned MiniMaxi video request option extension helper | `minimaxi_video_request_extension_source_does_not_read_response_metadata` |
| `siumai-provider-minimaxi/src/providers/minimaxi/ext/video.rs` | provider-owned MiniMaxi video request builder | `minimaxi_video_builder_source_does_not_read_response_metadata` |
| `siumai-provider-minimaxi/src/providers/minimaxi/video.rs` | MiniMaxi video task request/body and response query helper | `video_request_and_response_paths_keep_provider_maps_directional` |
| `siumai-provider-google-vertex/src/provider_metadata/vertex.rs` | response-side typed metadata view | source guard rejects request provider options reads |
| `siumai-provider-minimaxi/src/provider_metadata/minimaxi.rs` | response-side typed metadata view | source guard rejects request provider options reads |
| `siumai-provider-gemini/src/provider_metadata/gemini.rs` | response-side typed metadata view | source guard rejects request provider options reads |
| `siumai-provider-amazon-bedrock/src/provider_metadata/bedrock.rs` | response-side typed reasoning metadata view | source guard rejects request provider options reads |
| `siumai-protocol-anthropic/src/provider_metadata/anthropic.rs` | protocol-owned response-side typed metadata view | `anthropic_provider_metadata_source_does_not_read_request_provider_options` |
| `siumai-protocol-anthropic/src/standards/anthropic/provider_metadata.rs` | compatibility re-export for protocol-owned Anthropic typed metadata view | covered by `siumai-protocol-anthropic/src/provider_metadata/anthropic.rs` guard |
| `siumai-provider-openai/src/providers/openai/client/chat.rs` | provider-owned OpenAI chat request entry and response-id cancel wrapper | `chat_request_path_does_not_read_legacy_response_metadata_maps` |
| `siumai-provider-openai/src/providers/openai/client/completion.rs` | provider-owned OpenAI legacy completion request/response path | request metadata-read guard plus response/stream provider-options emission guard |
| `siumai-provider-openai/src/providers/openai/client/audio.rs` | provider-owned OpenAI non-chat audio request/response path | `audio_mixed_request_response_path_does_not_read_legacy_response_metadata` |
| `siumai-provider-openai/src/providers/openai/client.rs` | provider-owned OpenAI client default provider-option merging | `openai_client_default_provider_option_merging_does_not_read_response_metadata` keeps non-chat request default merging away from response metadata |
| `siumai-provider-openai/src/providers/openai/spec.rs` | provider-owned OpenAI request option routing and Responses API metadata key selection | request option guard plus provider metadata key selection guard split request provider options from response metadata namespace choice |
| `siumai-provider-openai/src/providers/openai/structured_output.rs` | provider-owned OpenAI structured-output request config plus response validator | request-config source guard plus response validator test |
| `siumai-provider-openai/src/providers/openai/files.rs` | provider-owned OpenAI file upload/resource extension | `file_upload_request_path_does_not_read_legacy_provider_metadata` |
| `siumai-provider-openai/src/providers/openai/skills.rs` | provider-owned OpenAI skill upload extension | `skill_upload_request_and_response_paths_keep_provider_maps_directional` |
| `siumai-provider-openai/src/providers/openai/websocket_session.rs` | provider-specific OpenAI WebSocket session request/recovery helper | `websocket_session_request_mutation_does_not_read_legacy_metadata_maps` |
| `siumai-protocol-openai/src/standards/openai/chat.rs` | OpenAI chat standard wrapper with request/response/stream adapter split | `chat_wrapper_keeps_request_response_stream_maps_directional` |
| `siumai-protocol-anthropic/src/standards/anthropic/chat.rs` | Anthropic chat standard wrapper with request citation extraction and response/stream adapters | `chat_wrapper_keeps_request_and_response_provider_maps_directional` |
| `siumai-protocol-gemini/src/standards/gemini/chat.rs` | Gemini chat standard wrapper with request/response/stream adapter split | `chat_wrapper_keeps_request_response_stream_maps_directional` |
| `siumai-core/src/streaming/{builder.rs,factory.rs}` | provider-agnostic stream event helpers and fallback text-delta synthesis | `core_stream_helpers_only_initialize_empty_provider_metadata` permits only empty `provider_metadata: None` initialization in core stream helpers |
| `siumai-core/src/streaming/factory.rs` | provider-agnostic stream event helper and fallback text-delta synthesis | covered by `core_stream_helpers_only_initialize_empty_provider_metadata` |
| `siumai-core/src/execution/executors/stream_json.rs` | provider-agnostic line-delimited JSON streaming executor | `core_json_stream_executor_does_not_handle_provider_maps` delegates provider-specific parsing and metadata projection to injected converters |
| `siumai-core/src/execution/middleware/samples.rs` | provider-neutral sample streaming middleware with synthetic text parts | `core_sample_streaming_middleware_only_initializes_empty_provider_metadata` permits only empty `provider_metadata: None` on synthetic stream parts |
| `siumai-core/src/utils/provider_options.rs` | generic request-side provider option schema parser | `core_provider_options_parser_stays_request_only_and_provider_agnostic` keeps it away from response metadata and concrete provider namespaces |
| `siumai-core/src/completion.rs` | stable completion family contract adapter | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps family contracts provider-map-neutral |
| `siumai-core/src/custom_provider/guide.rs` | documentation-only custom provider guide | no direct legacy `ContentPart` pattern matching remains; guide steers provider-specific content serialization out of core |
| `siumai-core/src/text.rs` | stable text/language family contract adapter | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps family contracts provider-map-neutral |
| `siumai-core/src/speech.rs` | stable speech family contract adapter | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps family contracts provider-map-neutral |
| `siumai-core/src/transcription.rs` | stable transcription family contract adapter | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps family contracts provider-map-neutral |
| `siumai-core/src/tooling.rs` | runtime tool execution contracts and adapters | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps tooling contracts from handling provider maps directly |
| `siumai-core/src/traits.rs` | stable capability trait re-export shell | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps trait surfaces provider-map-neutral |
| `siumai-core/src/traits/audio.rs` | compatibility audio family trait contract | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps trait surfaces provider-map-neutral |
| `siumai-core/src/traits/speech.rs` | speech family trait contract | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps trait surfaces provider-map-neutral |
| `siumai-core/src/traits/transcription.rs` | transcription family trait contract | `core_family_contract_and_tooling_sources_do_not_handle_provider_maps` keeps trait surfaces provider-map-neutral |
| `siumai-core/src/streaming/encoder.rs` | provider-agnostic stream encoder and terminal event synthesis | `stream_encoder_source_does_not_read_provider_option_or_metadata_maps` |
| `siumai-core/src/streaming/processor.rs` | provider-agnostic response consolidation | source guard rejects request provider options reads |
| `siumai-core/src/utils/streaming_tool_call.rs` | provider-agnostic streaming tool-call tracker with caller-supplied metadata callbacks | `core_streaming_tool_call_tracker_only_uses_callback_provider_metadata` |
| `siumai-core/src/structured_output.rs` | provider-agnostic JSON extraction and stream-response consolidation helper | `core_structured_output_helpers_only_merge_generic_response_metadata` |
| `siumai-provider-anthropic/src/providers/anthropic/spec.rs` | provider-owned Anthropic request beta/header routing | `anthropic_spec_request_option_routing_does_not_read_response_metadata` keeps beta routing on request provider options |
| `siumai-provider-anthropic/src/providers/anthropic/client.rs` | provider-owned Anthropic request warning middleware | `anthropic_client_middleware_request_option_checks_do_not_read_response_metadata` keeps post-generate warnings from reading response metadata as request input |
| `siumai-provider-google-vertex/src/providers/anthropic_vertex/client.rs` | provider-owned Anthropic-on-Vertex default provider-option merging | `vertex_anthropic_client_default_provider_options_do_not_read_response_metadata` keeps defaults request-side |
| `siumai-provider-minimaxi/src/providers/minimaxi/client.rs` | provider-owned MiniMaxi default provider-option merging and TTS response wrapper | `minimaxi_client_request_option_merging_does_not_read_response_metadata`; TTS wrapper only initializes empty response metadata |
| `siumai-provider-gemini/src/providers/gemini/client.rs` | provider-owned Gemini client tests for structured-output stream metadata preservation | production paths already covered by Gemini transformer/provider metadata guards; remaining scan hit is test-only metadata assertion |
| `siumai/src/macros.rs` | facade request-message convenience macros | `facade_macros_only_create_request_side_empty_provider_options` permits only empty request `provider_options` initialization and the tool-result constructor |
| `siumai/src/speech.rs` | facade speech helper response metadata projection | `facade_audio_and_structured_helpers_do_not_read_request_provider_options` keeps high-level result projection away from request provider options and local `ContentPart` mapping |
| `siumai/src/transcription.rs` | facade transcription helper response metadata projection | `facade_audio_and_structured_helpers_do_not_read_request_provider_options` keeps high-level result projection away from request provider options and local `ContentPart` mapping |
| `siumai/src/structured_output.rs` | facade structured-output helper response metadata projection | `facade_audio_and_structured_helpers_do_not_read_request_provider_options` keeps high-level result projection away from request provider options and local `ContentPart` mapping |
| `siumai/src/video.rs` | facade video task polling and response metadata aggregation | `facade_video_metadata_projection_avoids_legacy_request_provider_options` keeps high-level polling options separate from legacy request provider option maps |
| `siumai-bridge/src/stream/openai_responses_parts_bridge.rs` | OpenAI Responses stream parts to stable content replay | included in `response_and_stream_bridge_sources_do_not_emit_request_provider_options` |

## Refreshed Scan Notes

The post-projection-helper scan surfaced `siumai/src/text.rs` as a high-value duplicate projection
path. That facade path now delegates response `ContentPart` mapping to the spec-owned helper and is
guarded. OpenAI, Anthropic, and Gemini gateway/proxy JSON response encoders are also guarded as
response-side encoders that may read response metadata but must not read request-side provider
options, the OpenAI provider Responses stream/custom-event projection helper is now guarded as
a response-side projection path, and the OpenAI non-chat audio client is guarded as a mixed
request/response path that may read request-side audio provider options but must not read legacy
response metadata as request input. The remaining broad scan hits are either already classified
guarded paths, Anthropic protocol thinking helpers with split request/response guards, the
Anthropic standards provider-metadata compatibility re-export, core JSON streaming executor tests
whose production executor delegates metadata projection to injected converters, OpenAI provider
chat/audio/WebSocket request and recovery helpers that now reject
legacy metadata reads, OpenAI/Anthropic/Gemini protocol chat wrappers that keep request and
response directional concerns separate, AI SDK passive output/stream/UI data carriers, inline
tests, or provider/protocol response parsers that still need case-by-case lossiness review before
they can move away from legacy `ContentPart`.

## Lower-Priority Or False-Positive Buckets

- `siumai-spec/src/types/**`: mostly carrier definitions and compatibility docs; covered by
  content projection and AI SDK module boundary tests.
- `siumai-core/src/streaming/stream_part.rs`: stream carrier conversions and tests, not provider
  request serialization.
- `siumai-bridge/src/contracts.rs`: bridge diagnostics/control-plane types that record carried
  provider metadata in reports, not request content construction or response metadata replay.
- `siumai-provider-*/src/provider_options/**`: request option type definitions and docs, not
  response metadata reads.
- `siumai-protocol-openai/src/standards/openai/compat/metadata.rs`: OpenAI-compatible metadata
  namespace helper. It may inspect request `ProviderOptionsMap` only to choose the provider
  metadata namespace/key, not to construct `ContentPart` values or replay response metadata.
- `siumai-provider-openai/src/providers/openai/ext/responses.rs`: provider extension helper for
  OpenAI Responses stream/custom event projection. Request-side `chat_via_responses_api(...)`
  remains a legitimate request helper, while the guarded projection section keeps response
  streaming metadata separate from request `provider_options`.
- `siumai-provider-openai/src/providers/openai/middleware/responses_input_warnings.rs`: request
  warning middleware that inspects request-side `provider_options` and `ContentPart` data as
  policy input rather than as a response-metadata replay path.
- Inline test modules in protocol/provider files: useful behavior coverage but should not drive
  production-source classification.

## Non-V4 Response Projection Decision

First-phase response projection should reuse the existing AI SDK-style non-V4 output surface instead
of introducing another `GeneratedContentPart` family:

- `GenerateTextContentPart` is already the generated content union for high-level text helpers.
- `TextOutput`, `CustomOutput`, `FileOutput`, `ReasoningOutput`, `ReasoningFileOutput`, `ToolCall`,
  `ToolResult`, and `ToolError` already carry response-side `providerMetadata`.
- `ChatResponse.content: MessageContent` remains the stable compatibility carrier until protocol and
  provider response parsers can project into output parts without losing unsupported legacy shapes.
- `GenerateTextStepReasoningPart` is a documented cross-step replay exception: it converts response
  metadata into request-side provider options for the next step and is not the final generated
  content projection.
- `project_response_content_part_to_generate_text_content_part`,
  `project_response_content_to_generate_text_content_parts`, and
  `project_chat_response_to_generate_text_content_parts` provide the first fallible migration path
  for response-owned legacy content. They only map lossless response-side subsets and reject
  ambiguous legacy carriers such as `Image`, `Audio`, URL-backed generated files, tool approval
  parts without the original tool call, and tool results without original input.

## Next Slice

Recommended next implementation slice:

1. Use the audit coverage guard as the baseline for any new `ContentPart` or provider-map source
   hit: either add a guarded-path row, classify it as a low-priority bucket, or refactor it away.
2. Identify provider/protocol response parsers that can safely project through
   `GenerateTextContentPart` without losing provider-specific response data.
3. Keep parsers on legacy `ContentPart` where output projection would be lossy, but document those
   paths as compatibility surfaces with removal or narrowing criteria.
