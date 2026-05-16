# Fearless Spec/Core Boundary Convergence - Compatibility Audit

Last updated: 2026-05-16

This audit is the staging area for compatibility decisions made during the workstream. Every broad
surface kept during refactor should have a reason, owner, and removal or stabilization path.

## Audit Rules

- Keep compatibility only when it has a public migration purpose, an extension-only gap, or a
  documented removal window.
- Prefer moving misplaced code to the owning crate over adding forwarding aliases.
- Add a source guard before deleting a pattern that is likely to return.
- Record focused validation commands for each completed slice.

## Initial Findings

| Surface | Current risk | Target owner | Default action |
| --- | --- | --- | --- |
| `siumai-spec::tools` provider-defined factories | The spec crate owns a broad provider-defined tool constructor surface, which is easy to mistake for runtime or provider implementation code. | Passive data-constructor surface owned by `siumai-spec`, with provider/runtime behavior kept in provider/protocol/facade crates | Keep it passive only, add a guard against runtime/provider dependencies, and avoid moving execution logic into spec. |
| `siumai-spec::types::video` generation and task-status carriers | Task/status helpers can drift into polling, download, HTTP client, or provider execution behavior because video APIs are naturally async at the provider level. | Passive request/response/status data owned by `siumai-spec`; polling/download/execution belongs in provider, protocol, core runtime, or facade family code. | Keep the data carriers and small projections, but guard against runtime/provider dependencies and keep request headers as passive `HttpConfig::empty()` overrides. |
| Non-`ai_sdk` request carriers with provider options and HTTP overrides | Model-family/upload request types can silently turn into execution helpers because they already carry provider options and per-request HTTP config. | Passive request data in `siumai-spec`; execution, polling, upload/download transport, retry, and provider defaults belong outside spec. | Keep request constructors/builders passive, guard them against runtime/provider dependencies, and require header helpers to start from `HttpConfig::empty()`. |
| Removed `ChatMessageBuilder` Anthropic cache/document helper methods | Provider-prefixed convenience methods lived in the spec chat builder and wrote `providerOptions.anthropic.*`, making it easy to add more provider-specific request helpers to spec. | Anthropic provider extension surface, exposed through `siumai::provider_ext::anthropic`. | Removed from spec after internal consumers migrated; guard against reintroduction. |
| `siumai-spec` runtime cancellation handles | Spec crate carries runtime semantics. | `siumai-core` or facade/runtime helper module | Move out of spec and add a purity guard. |
| `siumai-spec/src/types/ai_sdk.rs` | Mixed prompt, response, UI, and runtime concerns in one surface. | Split spec modules plus runtime-owned helpers | Split by responsibility, not by cosmetic size. |
| Shared content parts with request options and response metadata | Prompt construction and response parsing concerns can leak into each other. | Prompt/response projections or adapters | Separate views and keep shared data components explicit. |
| Provider-specific stream bridge residue in `siumai-core` | Core becomes a hidden protocol implementation center. | `siumai-bridge`, `siumai-protocol-*`, or provider crates | Relocate and add core boundary tests. |
| Provider defaults and hosted tool factories in `siumai-core` | Provider ownership is obscured by core helpers. | Provider crates or registry | Move unless the contract is truly provider-agnostic. |
| `LlmClient` downcasts in stable execution paths | Compatibility umbrella can become the real runtime center again. | Explicit `compat_*` modules or provider extensions | Remove from stable family paths. |
| Broad facade and registry re-exports | Stable public surface mirrors internal crate structure. | Intentional facade modules and registry APIs | Narrow exports and document migration paths. |

## Compatibility Decision Template

Use this template when a compatibility surface is reviewed:

```text
Surface:
Owner:
Current users:
Canonical replacement:
Keep, move, or remove:
Migration note needed:
Removal window:
Validation:
```

## Completed Decisions

### `CancelHandle`

Surface: `CancelHandle`

Owner: `siumai-core::types`

Current users: core streaming cancellation helpers, facade request options, tool runtime options,
and public import compatibility through `siumai_core::types` / `siumai::types`.

Canonical replacement: use the core-owned `CancelHandle` through `siumai_core::types` or the facade.
`siumai-spec` option carriers are now generic over the abort handle and default to `()`.

Keep, move, or remove: moved out of `siumai-spec`; kept as a core runtime type.

Migration note needed: yes if users imported `siumai_spec::types::CancelHandle` directly.

Removal window: direct spec import is removed in this fearless refactor slice.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-core --no-default-features`
- `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`
- `cargo check -p siumai-registry --tests --features openai,anthropic,google --no-default-features`
- `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test openai_sse_streaming_alignment_test --features openai --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-registry --tests --features openai,anthropic,google --no-default-features`
- `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
- `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test tooling_runtime_public_surface_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### `siumai-spec::tools` provider-defined factories

Surface: `siumai-spec::tools::{provider_defined_tool, openai, anthropic, google, groq, xai}`,
`siumai-spec::types::tools::{Tool,ProviderDefinedTool,LanguageModelV4ProviderTool}`, and the
`Tool::provider_defined*` helper family that delegates to those constructors.

Owner: `siumai-spec`, as passive tool-schema/data constructors only.

Current users: provider-defined tool metadata tests, `Tool::provider_defined_id(...)`, provider
facade examples, and provider extension helper paths that need passive `Tool::ProviderDefined`
values.

Canonical replacement: keep the passive constructors in spec, but never add runtime execution,
provider HTTP, or provider crate dependencies there. If a future slice wants to move these factories,
it should do so only after replacing all call sites with a clearer owning crate path.

Keep, move, or remove: keep for now as a passive spec-level constructor/data surface; guard it so
both the helper module and the stable tool data carriers stay data-only.

Migration note needed: no current source migration. The concern is boundary clarity, not a breaking
path change.

Removal window: not set.

Validation:

- `cargo nextest run -p siumai-spec --test tools_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-spec --no-default-features`

### `siumai-spec::types::video` generation carriers

Surface: `VideoGenerationInput`, `VideoGenerationPrompt`, `VideoGenerationRequest`,
`VideoGenerationResponse`, `VideoTaskStatusResponse`, and `MaterializedVideoAsset`.

Owner: `siumai-spec`, as passive video request/response/status data contracts only.

Current users: provider video adapters, facade video helpers, provider-specific video option
extensions, and AI SDK-aligned media result surfaces that need a shared video prompt/result shape.

Canonical replacement: keep importing the shared data carriers through `siumai_spec::types::*`,
`siumai_core::types`, or the facade. Provider task creation, polling, materialized download, HTTP
client construction, retry behavior, and model/provider defaults must stay in provider/protocol,
core runtime, registry, or facade family code.

Keep, move, or remove: keep as passive spec data. `effective_provider_reference(...)` is an
allowed data projection from a legacy `file_id` into `ProviderReference`; it must not become a
provider lookup or runtime fetch path. `with_header(...)` remains a request-level override helper
that starts from `HttpConfig::empty()` instead of `HttpConfig::default()`.

Migration note needed: no current public path change. The decision clarifies ownership and prevents
future runtime behavior from being added to the spec crate.

Removal window: not set.

Validation:

- `cargo nextest run -p siumai-spec --test video_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --test http_config_boundary_test --test video_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-spec --no-default-features`

### Non-`ai_sdk` request carriers with HTTP overrides

Surface: `TtsRequest`, `SttRequest`, `AudioTranslationRequest`, `CompletionRequest`,
`EmbeddingRequest`, `FileUploadRequest`, `FileListQuery`, `ImageGenerationRequest`,
`GenerateImageRequest`, image edit/variation request carriers, `RerankRequest`, and
`SkillUploadRequest`.

Owner: `siumai-spec`, as passive model-family/upload request data contracts only.

Current users: core family executors, provider/protocol transformers, facade helpers, examples, and
provider-specific request option extension helpers.

Canonical replacement: keep request data construction in spec. Runtime execution, HTTP transport,
retry policy, task polling, file/skill upload or download behavior, provider defaults, and
provider-specific protocol conversion belong in core, bridge, protocol, provider, registry, or
facade family code.

Keep, move, or remove: keep as passive spec request carriers. Header convenience helpers are
allowed only as per-request data overrides that start from `HttpConfig::empty()`. They must not use
`HttpConfig::default()` because runtime defaults are owned by `siumai-core::defaults`.

Migration note needed: no current public path change. This decision is an architectural guardrail
for future changes.

Removal window: not set.

Validation:

- `cargo nextest run -p siumai-spec --test request_carrier_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --test http_config_boundary_test --test request_carrier_boundary_test --test video_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-spec --no-default-features`

### Removed `ChatMessageBuilder` Anthropic cache/document helpers

Surface: `ChatMessageBuilder::{cache_control,cache_control_for_part,cache_control_for_parts,
anthropic_document_citations_for_part,anthropic_document_metadata_for_part}`.

Owner: removed from `siumai-spec`. The owner for Anthropic provider-specific request helpers is the
Anthropic provider extension surface, exposed through `siumai::provider_ext::anthropic`.

Current users: none inside this workspace. Internal bridge/protocol tests, the facade Anthropic
prompt-cache example, and facade cache-control macro arms have migrated away from consuming these
methods. Bridge/protocol tests now construct explicit request-side
`providerOptions.anthropic.*` data or legacy passive `MessageMetadata.cache_control` fixtures where
they are intentionally validating compatibility behavior.

Canonical replacement: use the provider-owned
`siumai::provider_ext::anthropic::AnthropicChatMessageExt` trait, which adds
`with_anthropic_cache_control(...)`, `with_anthropic_part_cache_control(...)`,
`with_anthropic_parts_cache_control(...)`,
`with_anthropic_document_citations_for_part(...)`, and
`with_anthropic_document_metadata_for_part(...)` on built `ChatMessage` values. This writes the same
request-side `providerOptions.anthropic.*` data without adding new provider-specific helpers to the
spec chat builder.

Keep, move, or remove: removed. The source guard now prevents these helpers, any new
provider-prefixed chat builder helpers, Anthropic request option literals, and concrete provider
namespace literals in `ChatMessage` production code from returning to `siumai-spec`.

Migration note needed: yes. The migration guide now points historical builder-helper users to
`AnthropicChatMessageExt`.

Removal window: completed in this fearless refactor slice.

Validation:

- `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-bridge --lib anthropic_part_cache_paths_follow_canonical_part_provider_options --no-default-features --features anthropic,openai --no-fail-fast`
- `cargo nextest run -p siumai-bridge --lib anthropic_bridge_reports_cache_breakpoints_beyond_limit anthropic_bridge_reports_part_cache_breakpoints_from_canonical_provider_options --no-default-features --features anthropic,openai --no-fail-fast`
- `cargo nextest run -p siumai-protocol-anthropic --lib standards::anthropic::utils::messages standards::anthropic::chat --no-default-features --features anthropic-standard --no-fail-fast`
- `cargo nextest run -p siumai-spec --test chat_provider_helper_boundary_test --test spec_purity_boundary_test --test tools_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-provider-anthropic --lib providers::anthropic::ext::chat_message --no-default-features --features anthropic --no-fail-fast`
- `cargo check -p siumai --example prompt-caching --features anthropic --no-default-features`
- `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_macros_only_create_request_side_empty_provider_options --features anthropic --no-default-features --no-fail-fast`

### `AudioStream`

Surface: `AudioStream`

Owner: `siumai-core::types`

Current users: audio traits, registry audio handles, facade public surface, provider tests, and
streaming TTS/STT examples.

Canonical replacement: use the core-owned runtime `AudioStream` alias through `siumai_core::types`
or the facade. `siumai-spec` keeps the serializable/passive event payload as `AudioStreamEvent`.

Keep, move, or remove: moved out of `siumai-spec`; kept as a core runtime type.

Migration note needed: yes if users imported `siumai_spec::types::AudioStream` directly.

Removal window: direct spec import is removed in this fearless refactor slice.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test spec_purity_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-core --no-default-features`

### AI SDK call option carriers

Surface: `LanguageModelV4CallOptions`, `RequestOptions`, `LanguageModelReasoning`,
`LanguageModelCallOptions`, and deprecated `CallSettings`

Owner: `siumai-spec::types::ai_sdk::call_options`

Current users: AI SDK-aligned spec types, facade/core type aliases for runtime abort handles, and
public imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`,
`siumai_core::types`, or the facade. The physical owner is now `ai_sdk/call_options.rs`; direct
file layout is not part of the public API.

Keep, move, or remove: kept as pure spec data carriers and moved out of the oversized `ai_sdk`
module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: deprecated `CallSettings` remains as compatibility while callers migrate to
`LanguageModelCallOptions` plus `RequestOptions`.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK passive error carriers

Surface: `AISDKError`, `APICallError`, `EmptyResponseBodyError`, `InvalidPromptError`,
`InvalidResponseDataError`, provider/model lookup errors, no-output/no-media errors, UI message
conversion errors, retry errors, and validation errors

Owner: `siumai-spec::types::ai_sdk::errors`

Current users: AI SDK-aligned spec types, error construction tests, UI message conversion helpers,
stream validation helpers, generation result helpers, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/errors.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec error data and moved out of the oversized `ai_sdk`
module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none for these passive errors.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK generated file carriers

Surface: `GeneratedFile`, `DefaultGeneratedFile`, `DefaultGeneratedFileWithType`,
`Experimental_GeneratedImage`, `GeneratedAudioFile`, `DefaultGeneratedAudioFile`, and
`DefaultGeneratedAudioFileWithType`

Owner: `siumai-spec::types::ai_sdk::generated_files`

Current users: text helper output parts, language-model V4 file projections, image/video/speech
result envelopes, stream parts, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/generated_files.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec file data and moved out of the oversized `ai_sdk` module
file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK embedding result and event carriers

Surface: `ModelCallResponseData`, `EmbedValue`, `EmbedOutput`, `EmbedResponseData`,
`EmbedResult`, `EmbedManyResult`, `EmbedStartEvent`, `EmbedEndEvent`,
`EmbeddingModelCallStartEvent`, and `EmbeddingModelCallEndEvent`

Owner: `siumai-spec::types::ai_sdk::embedding`

Current users: embedding result envelopes, embedding callback event payloads, rerank response
payloads that reuse `ModelCallResponseData`, tests, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/embedding.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec embedding/result/event data and moved out of the
oversized `ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`

### `siumai-core` provider-specific bridge residue

Surface: `BridgeTarget`

Owner: `siumai-bridge` plus the relevant `siumai-protocol-*` crates. `siumai-core` retains only
provider-agnostic runtime stream carriers and shared execution primitives.

Current users: bridge gateway helpers, protocol bridge fixtures, `siumai-bridge` target dispatch,
protocol stream serializers, and public experimental imports through the facade.

Canonical replacement: `BridgeTarget`, `BridgeOptions`, bridge reports, contexts, customization
hooks, primitive remappers, and loss-policy traits now live at the top level of `siumai_bridge`.
The facade compatibility path `siumai::experimental::bridge::*` continues to re-export the bridge
crate. `OpenAiResponsesStreamPartsBridge` now lives at
`siumai_bridge::stream::OpenAiResponsesStreamPartsBridge` behind the `openai` feature and remains
available through `siumai::experimental::streaming::OpenAiResponsesStreamPartsBridge` when the
facade `openai` feature is enabled. `StreamPartNamespace` and `to_protocol_custom_event` have no
core replacement; protocol serializers now own their custom event prefix mappings directly.

Keep, move, or remove: move. The OpenAI Responses stream-parts bridge, bridge customization
contracts, target catalog, and provider-specific custom event serialization have moved out of
`siumai-core`. The source guard is now strict rather than allowlist-based.

Migration note needed: yes. Direct imports from `siumai_core::bridge::*` must move to
`siumai_bridge::*` or the facade's `siumai::experimental::bridge::*` compatibility re-export. For
`OpenAiResponsesStreamPartsBridge`, use
`siumai_bridge::stream::OpenAiResponsesStreamPartsBridge` directly or the facade's
`siumai::experimental::streaming::OpenAiResponsesStreamPartsBridge` compatibility re-export.
For `StreamPartNamespace`, use the target protocol serializer instead of formatting core stream
parts into provider-prefixed custom events directly.

Removal window: during the core/provider boundary convergence phase.

Validation:

- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`

### SSE stream-end marker ownership

Surface: `siumai-core::streaming::StreamFactory`, `SseEventConverter`,
`StreamChunkTransformer`, and protocol/provider SSE converters that historically relied on core
recognizing `[DONE]`.

Owner: protocol/provider converters. `siumai-core` owns only the SSE transport orchestration:
parsing event frames, delegating conversion, draining final events, and preserving HTTP metadata.
Concrete stream-end marker values belong to the wire protocol implementation.

Current users: OpenAI-compatible chat/completions, OpenAI Responses, OpenAI legacy completions,
Anthropic, Gemini, xAI Responses, and facade stream factory injection tests that use SSE
marker-style streams. Cohere-style streams that finalize on disconnect keep using
`finalize_on_disconnect()` and do not need a marker predicate.

Canonical replacement: converters implement `is_stream_end_event(&Event) -> bool` when their wire
protocol has an explicit terminal SSE event. `StreamFactory` calls that contract and then drains
`handle_stream_end_events()`; it no longer recognizes concrete marker payloads directly.
`TransformerConverter` and provider-specific `StreamChunkTransformer` wrappers delegate the
predicate to their inner protocol converters. `SseJsonStreamConfig::new(...)` now defaults to no
done markers; provider helpers such as OpenAI speech/transcription SSE configure their marker list
explicitly.

Keep, move, or remove: move. Core keeps the trait method and generic final-event draining logic,
but concrete markers such as OpenAI-style `[DONE]` are owned by protocol/provider crates.
Core streaming runtime tests and docs now use provider-neutral fixture model names and URLs; provider
metadata key tests remain only as carrier-shape coverage.

Migration note needed: no user-facing API migration. Custom SSE converters that depend on marker
events should implement `is_stream_end_event(...)`; otherwise marker frames are treated as normal
events.

Removal window: completed in this workstream slice for `StreamFactory`, `SseJsonStreamConfig`
defaults, and core HTTP tracing marker recognition.

Validation:

- `cargo check -p siumai-core --no-default-features`
- `cargo check -p siumai-protocol-openai -p siumai-protocol-anthropic -p siumai-protocol-gemini --no-default-features --features openai-standard`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib streaming::sse_json --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib streaming --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test streaming_tests factory_injection --features openai --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard --no-fail-fast`
- `cargo nextest run -p siumai-protocol-anthropic -p siumai-protocol-gemini --no-default-features --features "siumai-protocol-anthropic/anthropic siumai-protocol-gemini/google" --no-fail-fast`
- `cargo nextest run -p siumai-provider-openai --no-default-features --features openai --no-fail-fast`
- `cargo check -p siumai-provider-xai -p siumai-provider-minimaxi --no-default-features --features "siumai-provider-xai/xai siumai-provider-minimaxi/minimaxi"`
- `cargo fmt --package siumai-core --package siumai-protocol-openai --package siumai-protocol-anthropic --package siumai-protocol-gemini --package siumai-provider-openai --package siumai-provider-xai --package siumai-provider-minimaxi --package siumai --check`

### AI SDK language-model V4 prompt and generated content projections

Surface: `LanguageModelV4Prompt`, V4 prompt parts/messages,
`prepare_language_model_v4_prompt`, `LanguageModelV4DataContent`,
`LanguageModelV4GeneratedFileData`, `LanguageModelV4FilePartData`, `LanguageModelV4Text`,
`LanguageModelV4Reasoning`, `LanguageModelV4Source`, V4 generated file/tool content, and
`LanguageModelV4Content`

Owner: `siumai-spec::types::ai_sdk::language_model_v4`

Current users: V4 call options, language-model V4 generate/stream result envelopes, prompt
projection tests, generated content serde tests, provider metadata validation tests, and public
imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now split into `ai_sdk/language_model_v4/shared.rs`,
`ai_sdk/language_model_v4/prompt.rs`, and `ai_sdk/language_model_v4/content.rs`; the public path
remains stable.

Keep, move, or remove: kept as pure spec projection/data shapes and moved out of the oversized
`ai_sdk` module file. Prompt-side request `providerOptions` and response-side generated
`providerMetadata` now live in different physical modules.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`

### Legacy `ContentPart` provider option/metadata maps

Surface: legacy stable `ContentPart` variants that still carry both request-side
`provider_options` / `providerOptions` and response-side `provider_metadata` /
`providerMetadata`.

Owner: `siumai-spec` owns the compatibility carrier shape. Directional conversion belongs to
adapter/protocol owners: `siumai-core::ui` for AI SDK UI message conversion,
`siumai-bridge::request` for protocol-body normalization into `ChatRequest`, and
`siumai-protocol-*` for wire request/response conversion.

Current users: stable chat message builders, AI SDK UI message conversion, bridge request
normalizers, protocol request serializers, response parsers, and downstream code that still
constructs legacy multimodal `ContentPart` values directly.

Canonical replacement: prompt/request projections should carry request-side `providerOptions`;
generated/response projections should carry response-side `providerMetadata`. The AI SDK V4
`language_model_v4::{prompt,content}` split is the current concrete projection. Until a broader
stable non-V4 projection ships, legacy `ContentPart` is kept only as an explicitly audited dual-use
compatibility carrier.

Keep, move, or remove: keep temporarily, but request serializers must not treat legacy
`ContentPart::provider_metadata` as input. UI-layer `providerMetadata`, `callProviderMetadata`, and
`resultProviderMetadata` are normalized into request-side `provider_options` at the
`siumai-core::ui` adapter boundary, and UI request conversion now centralizes legacy `ContentPart`
construction behind `ui_request_*` helpers. Gemini, Anthropic, and OpenAI Responses request
conversion now read replay/item/detail request data from canonical `provider_options` only. OpenAI
Chat and OpenAI-compatible shared request utilities are source-guarded the same way, so direct
message/content serialization cannot read legacy response-side `ContentPart::provider_metadata`.
OpenAI-compatible protocol transformers and streaming replay are now guarded by direction as well:
request transformation cannot read legacy response metadata, while response/stream conversion
cannot emit request-side provider options. OpenAI Responses response parsing and the protocol-owned
OpenAI typed provider metadata view both have response-side guards. Provider-owned OpenAI legacy
completions are split similarly between request prompt/body guards and response/stream guards.
Anthropic response parsing now follows the same response-side rule: it can surface citations, tool
metadata, and sources while keeping legacy response `ContentPart::provider_options` empty.
OpenAI Responses also has a source guard that rejects direct legacy `provider_metadata` reads in
its request transformer implementation, including accidental bindings of ignored
`provider_metadata: _` fields. Request normalization in `siumai-bridge` is source-guarded so it
does not read legacy `providerMetadata` / `provider_metadata` JSON keys and legacy
`provider_metadata` is only ever explicitly set to `None`. Gemini request normalization in
`siumai-bridge` restores thought signatures into `provider_options.google` and keeps legacy
`provider_metadata` empty. Anthropic prepare-step container replay is an explicit response-level
workflow exception: it reads prior-step `ProviderMetadataMap` values and writes next-step
`ProviderOptionsMap` overrides, without touching legacy `ContentPart` metadata or parsing
`providerMetadata` JSON fields directly. Anthropic hosted-tool extension helpers are also split by
direction: `with_anthropic_tool_options(...)` writes request-side tool provider options, while
hosted-tool stream/custom-event projections read only response/stream metadata. Bridge request
normalization now centralizes legacy `ContentPart` construction behind request-side adapter
helpers, and the source guard verifies that any request-normalized `provider_metadata: None` writes
stay in that helper block except for the plain-text collapse match. Vertex Gemini image
edit/variation request synthesis in
`siumai-provider-google-vertex` also uses a provider-owned request adapter helper for image input
file parts. Amazon Bedrock chat request conversion is now source-guarded so request settings for
documents, cache points, and reasoning replay come only from request-side `provider_options`.
Bedrock response and stream parsing is guarded separately so it can emit response-side
`provider_metadata` while only initializing legacy response `ContentPart::provider_options` with
the empty default. MiniMaxi's Anthropic-protocol adapter is also split by direction: response and
stream metadata re-keying from `anthropic` to `minimaxi` is response-side only, while request option
resolution reads `provider_options_map` only and keeps the legacy `anthropic` request-options key
as a compatibility alias rather than a response metadata replay path. Google Vertex, Gemini,
Bedrock, and MiniMaxi typed provider metadata extension modules are response-side views only: they
can parse `ChatResponse` / `ContentPart` provider metadata, but source guards prevent them from
reading request provider options. The protocol-owned Anthropic typed provider metadata view follows
the same response-side-only rule. `siumai-core::streaming::StreamProcessor` is classified as
provider-agnostic response consolidation: it can preserve response-side stream metadata while
building final `ContentPart` values, but it must not read request provider options. Gemini
response parsing follows the same response-side rule: thought signatures can be preserved in
`provider_metadata`, while legacy response `ContentPart::provider_options` fields remain empty
defaults only.

Migration note needed: yes when a broader stable prompt/content projection replaces direct
`ContentPart` construction. For now the type docs steer provider-facing prompt/content work toward
the split AI SDK V4 projection modules.

Removal window: after a stable prompt/content projection or migration adapter exists for the
remaining non-V4 `ContentPart` construction paths.

Validation:

- `cargo nextest run -p siumai-core --lib ui --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib ui_conversion_centralizes_legacy_request_content_constructors --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google request_conversion_source_only_ignores_legacy_provider_metadata_fields --no-fail-fast`
- `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google request_conversion_ignores_legacy_provider_metadata_thought_signature --no-fail-fast`
- `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard request_conversion_source_does_not_read_legacy_provider_metadata_fields --no-fail-fast`
- `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard assistant_reasoning_ignores_legacy_provider_metadata_signature_for_request_replay --no-fail-fast`
- `cargo nextest run -p siumai-provider-anthropic --no-default-features --features anthropic prepare_step_source_only_bridges_response_metadata_to_request_provider_options --no-fail-fast`
- `cargo nextest run -p siumai-provider-anthropic --no-default-features --features anthropic tool_options_extension_source_does_not_read_response_provider_metadata stream_event_projection_source_does_not_read_request_provider_options --no-fail-fast`
- `cargo check -p siumai-provider-anthropic --no-default-features --features anthropic`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard openai_chat_request_conversion_source_does_not_read_legacy_provider_metadata_fields --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard openai_provider_metadata_source_does_not_read_request_provider_options openai_compatible_request_transformer_source_does_not_read_legacy_provider_metadata_fields openai_compatible_chat_response_source_does_not_emit_request_provider_options openai_compatible_streaming_source_does_not_emit_request_provider_options --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_response_transformer_source_does_not_emit_request_provider_options --no-fail-fast`
- `cargo check -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses request_transformer_source_does_not_read_legacy_provider_metadata_fields --no-fail-fast`
- `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_parse_response_content_source_does_not_emit_request_provider_options --no-fail-fast`
- `cargo check -p siumai-protocol-anthropic --no-default-features --features anthropic-standard`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_transform_chat_ignores_legacy_image_provider_metadata_without_provider_options --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_transform_chat_ignores_assistant_tool_call_legacy_metadata_item_id --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_transform_chat_prefers_image_provider_options_over_legacy_provider_metadata --no-fail-fast`
- `cargo nextest run -p siumai-bridge --no-default-features --features google request_normalization_source_never_populates_legacy_provider_metadata --no-fail-fast`
- `cargo nextest run -p siumai-bridge --no-default-features --features openai,anthropic,google request_normalization_centralizes_legacy_request_content_constructors --no-fail-fast`
- `cargo nextest run -p siumai-bridge --no-default-features --features google gemini_request_normalization_source_uses_provider_options_for_thought_signature --no-fail-fast`
- `cargo nextest run -p siumai-bridge --no-default-features --features google gemini_generate_content_request_normalization_roundtrip_preserves_core_projection --no-fail-fast`
- `cargo nextest run -p siumai-bridge --no-default-features --features google request::tests --no-fail-fast`
- `cargo nextest run -p siumai-bridge --no-default-features --features openai,anthropic,google request::tests --no-fail-fast`
- `cargo nextest run -p siumai-provider-google-vertex --no-default-features vertex_gemini_image --no-fail-fast`
- `cargo nextest run -p siumai-provider-amazon-bedrock --no-default-features --features bedrock request_conversion_source_does_not_read_legacy_provider_metadata_fields response_and_stream_source_do_not_emit_request_provider_options --no-fail-fast`
- `cargo check -p siumai-provider-amazon-bedrock --no-default-features --features bedrock`
- `cargo nextest run -p siumai-provider-minimaxi --no-default-features --features minimaxi response_metadata_normalization_source_does_not_read_request_provider_options request_option_resolution_source_does_not_read_response_provider_metadata --no-fail-fast`
- `cargo check -p siumai-provider-minimaxi --no-default-features --features minimaxi`
- `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex vertex_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
- `cargo nextest run -p siumai-provider-minimaxi --no-default-features --features minimaxi minimaxi_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
- `cargo check -p siumai-provider-google-vertex -p siumai-provider-minimaxi --no-default-features --features "siumai-provider-google-vertex/google-vertex siumai-provider-minimaxi/minimaxi"`
- `cargo nextest run -p siumai-provider-gemini --no-default-features --features google gemini_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
- `cargo nextest run -p siumai-provider-amazon-bedrock --no-default-features --features bedrock bedrock_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
- `cargo check -p siumai-provider-gemini -p siumai-provider-amazon-bedrock --no-default-features --features "siumai-provider-gemini/google siumai-provider-amazon-bedrock/bedrock"`
- `cargo nextest run -p siumai-provider-openai --no-default-features --features openai completion_request_source_does_not_read_legacy_provider_metadata_fields completion_response_and_stream_source_do_not_emit_request_provider_options --no-fail-fast`
- `cargo check -p siumai-provider-openai --no-default-features --features openai`
- `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_provider_metadata_source_does_not_read_request_provider_options --no-fail-fast`
- `cargo nextest run -p siumai-core --lib streaming::processor::tests::stream_processor_source_does_not_read_request_provider_options --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_response_content_source_does_not_emit_request_provider_options --no-fail-fast`
- `cargo check -p siumai-core --no-default-features`
- `cargo fmt --package siumai-protocol-openai --check`
- `cargo fmt --package siumai-bridge --check`
- `cargo fmt --package siumai-provider-anthropic --check`
- `cargo fmt --package siumai-provider-google-vertex --check`
- `cargo fmt --package siumai-provider-amazon-bedrock --check`
- `cargo fmt --package siumai-provider-minimaxi --check`
- `cargo fmt --package siumai-provider-gemini --check`

### Provider-hosted tool constructors

Surface: `siumai::hosted_tools::{openai,anthropic,google}` plus provider package extension paths
such as `siumai::provider_ext::openai::hosted_tools`,
`siumai::provider_ext::anthropic::hosted_tools`, and
`siumai::provider_ext::gemini::hosted_tools`

Owner: protocol crates own the provider tool-id shapes:
`siumai-protocol-openai::hosted_tools`,
`siumai-protocol-anthropic::hosted_tools`, and
`siumai-protocol-gemini::hosted_tools`. Provider crates and the facade re-export those constructors.

Current users: facade examples, provider extension modules, OpenAI/Anthropic/Google public-surface
compile checks, Google Vertex hosted tool helpers, and `siumai-extras` hosted tool re-exports.

Canonical replacement: keep public imports through `siumai::hosted_tools::<provider>` or
provider-specific extension paths. Internal crates should import from the owning protocol crate or
provider package instead of `siumai-core`.

Keep, move, or remove: moved out of `siumai-core`. The public facade paths are kept for source
compatibility, but `siumai-core` no longer owns provider-specific hosted tool factories.

Migration note needed: no for normal facade users because `siumai::hosted_tools::*` remains
available. Yes for internal crates that imported `siumai_core::hosted_tools::*` directly; use the
matching protocol crate.

Removal window: direct `siumai_core::hosted_tools` imports are removed in this fearless refactor
slice.

Validation:

- `cargo check -p siumai-core --no-default-features`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo check -p siumai-protocol-openai --no-default-features`
- `cargo check -p siumai-protocol-anthropic --no-default-features`
- `cargo check -p siumai-protocol-gemini --no-default-features`
- `cargo check -p siumai-provider-openai --no-default-features --features openai`
- `cargo check -p siumai-provider-anthropic --no-default-features --features anthropic`
- `cargo check -p siumai-provider-gemini --no-default-features --features google`
- `cargo check -p siumai-provider-google-vertex --no-default-features --features google-vertex`
- `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex --no-default-features`

### Provider-specific core defaults

Surface: `siumai_core::defaults::providers::{openai,anthropic,siliconflow,groq}`

Owner: provider crates and registry factories. `siumai-core::defaults` owns only provider-agnostic
runtime defaults such as HTTP, timeout, streaming, model parameter, and profile values.

Current users: none found in production source during the boundary-convergence audit. The constants
were historical residue in core and duplicated provider-owned defaults.

Canonical replacement: use the owning provider crate's config/default constants or registry factory
defaults for endpoint/model selection. Do not import provider URL/model defaults from
`siumai-core`.

Keep, move, or remove: remove from `siumai-core`. Provider-specific defaults remain owned by
provider packages or registry factories where they participate in provider construction.

Migration note needed: only for direct internal users of `siumai_core::defaults::providers`, and no
current production call sites were found.

Removal window: removed in this fearless refactor slice.

Validation:

- `cargo check -p siumai-core --no-default-features`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`

### Provider-specific HookBuilder body presets

Surface: `siumai_core::execution::transformers::hook_builder::HookBuilder::{with_openai_base,with_anthropic_base}`

Owner: provider or protocol crates if a provider-specific preset is needed. `siumai-core` owns only
the provider-agnostic hook composition contract.

Current users: none found outside the `hook_builder.rs` doc example and implementation during the
boundary-convergence audit.

Canonical replacement: pass an explicit closure through `HookBuilder::with_chat_body_builder(...)`
or use a provider/protocol-owned helper when one exists. The hook composer should not choose an
OpenAI- or Anthropic-shaped request body on behalf of all core users.

Keep, move, or remove: remove from `siumai-core`. The methods were experimental shortcuts and
encoded wire-shape assumptions in the provider-agnostic runtime crate.

Migration note needed: yes for experimental users who called these methods directly. Inline the
request body builder closure or depend on a provider/protocol helper once one is introduced.

Removal window: removed in this fearless refactor slice.

Validation:

- `rg -n "with_openai_base|with_anthropic_base|openai_base_chat_body|anthropic_base_chat_body|HookBuilder"`
- `cargo check -p siumai-core --no-default-features`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --no-default-features --no-fail-fast`
- `cargo test -p siumai-core --no-default-features --doc hook_builder`

### Removed `ProviderSpec` string route hooks

Surface: removed `ProviderSpec::{chat_url,embedding_url,image_url,image_edit_url,image_variation_url,rerank_url,models_url,model_url}` hook methods

Owner: route resolution should be provider/protocol-owned. `siumai-core` should own only the
provider-agnostic execution contract and the `ProviderContext` data passed into route resolvers.

Current users: no workspace code should call or implement the historical string-returning
`ProviderSpec` route hooks. Custom provider examples, workspace provider/protocol specs, facade
fixtures, and provider/protocol route tests now use the matching `try_*_url(...)` hooks.

Canonical replacement: provider-owned specs should implement route resolution explicitly. Core
executors now consume fallible `try_*_url(...)` methods so unsupported capabilities can return
`UnsupportedOperation` without core carrying provider-shaped endpoint strings. Custom provider
examples and test specs should implement `try_*_url(...)` directly.

Keep, move, or remove: remove the string-returning `*_url(...)` methods from `ProviderSpec`. The
OpenAI-shaped default endpoint strings and `legacy_provider_route(...)` fallback helper were removed
from `siumai-core`. Workspace provider/protocol specs that expose route hooks have been migrated to
own fallible route resolution directly or delegate to protocol-owned fallible route hooks.
Production client/model-listing paths that previously consumed `*_url(...)` directly now call
`try_*_url(...)` in the migrated OpenAI, OpenAI-compatible, Anthropic, Gemini, Ollama, and
Anthropic-on-Vertex code. Facade fixture tests and provider/protocol route assertions now also use
`try_*_url(...).unwrap()`, with a source guard to prevent the old assertion style from returning.
`ProviderSpec` default fallible route methods now return `UnsupportedOperation` directly. Core
executor test fixtures and embedded examples now also define/call only fallible route hooks, with a
source guard covering `siumai-core/src/execution` so new executor-local examples do not re-teach the
old route contract. Gemini protocol/provider route specs have removed their explicit
string-returning route implementations and now use the fallible route
methods as the only concrete implementation path. OpenAI protocol standard route specs (`chat`,
`embedding`, `image`, and `rerank`) have also removed their explicit string-returning route
implementations. OpenAI provider specs (`OpenAiSpec` and `OpenAiSpecWithRerank`) have removed their
explicit string-returning route implementations as well. The OpenAI-compatible protocol spec now
applies URL settings inside fallible `try_*_url(...)` methods and no longer implements direct
string-returning route hooks. Anthropic protocol/provider route specs, including
Anthropic-on-Vertex, now follow the same concrete implementation pattern and are guarded against
direct legacy route hook definitions. DeepSeek and Groq OpenAI-compatible provider wrappers now
delegate only through `try_chat_url(...)`; Cohere native chat/embedding/rerank route specs and
TogetherAI rerank route specs now also implement only fallible route methods. Ollama
chat/model-listing/embedding specs and MiniMaxi chat/image specs now follow the same fallible-only
implementation pattern. Amazon Bedrock chat/embedding/image/rerank specs and Azure OpenAI
chat/embedding/image specs have also removed their direct string-returning route implementations.
Google Vertex generative AI, embedding, Gemini image, and Imagen route specs now implement only
fallible route methods and no longer call protocol legacy string hooks.

Migration note needed: done. `docs/migration/migration-0.11.0-beta.7.md` includes the downstream
custom `ProviderSpec` migration from string-returning `*_url(...)` hooks to fallible
`try_*_url(...)` hooks.

Removal window: removed in this workstream. Downstream migration guidance now points custom
providers to `try_*_url(...)`.

Validation:

- `rg -n "legacy_provider_route|chat/completions|images/generations|images/edits|images/variations|models/\\{model_id\\}" siumai-core/src/core/provider_spec.rs`
- `rg -n "fn (chat_url|embedding_url|image_url|image_edit_url|image_variation_url|rerank_url|models_url|model_url)\\s*\\(" -g "*.rs"`
- `rg -n "\\.(chat_url|embedding_url|image_url|image_edit_url|image_variation_url|rerank_url|models_url|model_url)\\(" -g "*.rs"`
- `cargo check -p siumai-core --no-default-features`
- `cargo fmt --package siumai-core --check`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test core_executor_tests_and_docs_use_fallible_route_hooks --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test route_fixture_tests_use_fallible_provider_routes --no-default-features --no-fail-fast`
- `cargo check -p siumai --example custom_provider_spec --example complete_custom_provider --example testing_executors --features openai --no-default-features`
- `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock,xai --no-default-features`
- `cargo check -p siumai-provider-anthropic -p siumai-provider-gemini -p siumai-provider-google-vertex -p siumai-provider-ollama -p siumai-provider-openai -p siumai-provider-openai-compatible --no-default-features --features "siumai-provider-anthropic/anthropic siumai-provider-gemini/google siumai-provider-google-vertex/google-vertex siumai-provider-ollama/ollama siumai-provider-openai/openai siumai-provider-openai-compatible/openai-standard"`
- `rg -n "chat_url\\(\\)|embedding_url\\(\\)|image_url\\(\\)|rerank_url\\(\\)|models_url\\(\\)|model_url\\(\\)|fn chat_url\\(" docs siumai\\examples -g "*.md" -g "*.rs"` (expected remaining match: the migration guide's `Before` example)
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
- `cargo check -p siumai-core --features gcp --no-default-features`
- `cargo nextest run -p siumai-core --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-provider-openai --no-default-features --features openai --no-fail-fast`
- `cargo check -p siumai-provider-amazon-bedrock -p siumai-provider-azure -p siumai-provider-groq -p siumai-provider-deepseek -p siumai-provider-ollama -p siumai-provider-cohere -p siumai-provider-togetherai -p siumai-provider-minimaxi -p siumai-provider-google-vertex --no-default-features --features "siumai-provider-amazon-bedrock/bedrock siumai-provider-azure/azure siumai-provider-groq/groq siumai-provider-deepseek/deepseek siumai-provider-ollama/ollama siumai-provider-cohere/cohere siumai-provider-togetherai/togetherai siumai-provider-minimaxi/minimaxi siumai-provider-google-vertex/google-vertex"`
- `cargo nextest run -p siumai-provider-amazon-bedrock -p siumai-provider-azure -p siumai-provider-groq -p siumai-provider-deepseek -p siumai-provider-ollama -p siumai-provider-cohere -p siumai-provider-togetherai -p siumai-provider-minimaxi -p siumai-provider-google-vertex --no-default-features --features "siumai-provider-amazon-bedrock/bedrock siumai-provider-azure/azure siumai-provider-groq/groq siumai-provider-deepseek/deepseek siumai-provider-ollama/ollama siumai-provider-cohere/cohere siumai-provider-togetherai/togetherai siumai-provider-minimaxi/minimaxi siumai-provider-google-vertex/google-vertex" --no-fail-fast`
- `cargo check -p siumai-provider-anthropic -p siumai-provider-gemini -p siumai-provider-google-vertex -p siumai-provider-ollama -p siumai-provider-openai -p siumai-provider-openai-compatible --no-default-features --features "siumai-provider-anthropic/anthropic siumai-provider-gemini/google siumai-provider-google-vertex/google-vertex siumai-provider-ollama/ollama siumai-provider-openai/openai siumai-provider-openai-compatible/openai-standard"`
- `cargo nextest run -p siumai-provider-anthropic -p siumai-provider-gemini -p siumai-provider-google-vertex -p siumai-provider-ollama -p siumai-provider-openai -p siumai-provider-openai-compatible --no-default-features --features "siumai-provider-anthropic/anthropic siumai-provider-gemini/google siumai-provider-google-vertex/google-vertex siumai-provider-ollama/ollama siumai-provider-openai/openai siumai-provider-openai-compatible/openai-standard" --no-fail-fast`
- `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock --no-default-features`
- `cargo nextest run -p siumai --test bedrock_chat_stream_alignment_test --features bedrock --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test mock_api_tests minimaxi --features minimaxi --no-default-features --no-fail-fast`

### Generic `LlmClient` registry factory compatibility paths

Surface: `ProviderFactory::compat_*_client(...)`,
`ProviderFactory::compat_*_client_with_ctx(...)`, downstream `Arc<dyn LlmClient>` values, and
extension code that calls `.as_*_capability()` on a generic client.

Owner: `siumai-registry`. Provider crates own concrete model families; registry owns the temporary
generic-client compatibility construction surface and the policy that keeps it out of stable family
execution paths.

Current users: migration code that still expects a generic `LlmClient`, custom factories that have
not yet implemented native family methods, and extension-only gaps that do not have first-class
family models yet: files, skills, music, image edit/variation, and speech/transcription streaming
or extras.

Canonical replacement: stable code should construct family handles directly through
`language_model_text_with_ctx(...)`, `completion_model_family_with_ctx(...)`,
`embedding_model_family_with_ctx(...)`, `image_model_family_with_ctx(...)`,
`reranking_model_family_with_ctx(...)`, `speech_model_family_with_ctx(...)`,
`transcription_model_family_with_ctx(...)`, and `video_model_family_with_ctx(...)`. Facade callers
should normally enter through `registry::global().language_model(...)` and related family APIs
rather than asking for a generic client.

Keep, move, or remove: keep the explicit `compat_*_client(...)` and
`compat_*_client_with_ctx(...)` methods temporarily as migration and extension-only escape hatches.
Do not allow stable family handle execution to call those methods or downcast through
`.as_*_capability()`.

Migration note needed: done. `docs/migration/migration-0.11.0-beta.7.md`,
`docs/architecture/public-surface.md`, and `docs/architecture/registry-without-builtins.md` now
classify generic `LlmClient` factory paths as migration or extension-only compatibility and point
new construction code to family-first registry methods.

Removal window: deferred until remaining extension-only gaps have first-class family models or
explicit provider extension namespaces. Public stable family execution is already guarded against
using these paths.

Validation:

- `cargo fmt --package siumai-registry --check`
- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`

### `ClientWrapper` provider-named constructor aliases

Surface: `ClientWrapper::openai(...)`, `ClientWrapper::anthropic(...)`,
`ClientWrapper::gemini(...)`, `ClientWrapper::groq(...)`, `ClientWrapper::xai(...)`, and
`ClientWrapper::ollama(...)`.

Owner: `siumai-core::client::ClientWrapper` owns only provider-agnostic boxed-client wrapping.
Provider-specific construction belongs in provider crates, registry factories, or facade compat
builders.

Current users: one facade debug test used `ClientWrapper::openai(...)` only as a synonym for
`ClientWrapper::new(...)`.

Canonical replacement: use `ClientWrapper::new(Box<dyn LlmClient>)` for advanced dynamic dispatch.
Provider-specific clients should be constructed before wrapping, through provider/facade/registry
owned APIs.

Keep, move, or remove: remove. These aliases did not encode behavior and only reintroduced
provider names into core.

Migration note needed: yes. `docs/migration/migration-0.11.0-beta.7.md` now documents the
`ClientWrapper::new(...)` replacement.

Removal window: completed in this workstream slice.

Validation:

- `cargo nextest run -p siumai-core --test core_provider_boundary_test core_client_wrapper_does_not_expose_provider_specific_constructors --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test debug_implementation_test --no-default-features --features openai --no-fail-fast`

### Vertex URL helper ownership

Surface: `siumai_core::auth::vertex::*`, including `vertex_base_url`,
`google_vertex_base_url`, `google_vertex_anthropic_base_url`,
`GOOGLE_VERTEX_EXPRESS_BASE_URL`, and `google_vertex_maas_base_url`

Owner: `siumai-provider-google-vertex::auth::vertex`. The facade compatibility path
`siumai::experimental::auth::vertex` re-exports the provider-owned module behind the
`google-vertex` feature.

Current users: Google Vertex provider builders/settings, Anthropic-on-Vertex builders/settings,
registry Google Vertex factories, Vertex MaaS factory construction, facade Vertex alignment tests,
and Google Vertex examples.

Canonical replacement: import provider-owned URL helpers from
`siumai-provider-google-vertex::auth::vertex` for provider/registry internals, or from
`siumai::experimental::auth::vertex` for facade-level compatibility code. Do not import Vertex URL
construction helpers from `siumai-core`.

Keep, move, or remove: move out of `siumai-core`. `siumai-core::auth` keeps only the generic
`TokenProvider` contract, `StaticTokenProvider`, and optional GCP token-provider implementations.

Migration note needed: yes for direct imports from `siumai_core::auth::vertex::*`.

Removal window: direct core Vertex URL helpers are removed in this fearless refactor slice.

Validation:

- `cargo check -p siumai-core --no-default-features`
- `cargo check -p siumai-provider-google-vertex --no-default-features --features google-vertex`
- `cargo check -p siumai-provider-openai-compatible --no-default-features --features openai-standard`
- `cargo check -p siumai-registry --tests --features google-vertex --no-default-features`
- `cargo check -p siumai --tests --features google-vertex,gcp --no-default-features`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex --no-fail-fast`
- `cargo nextest run -p siumai-provider-openai-compatible --no-default-features --features openai-standard --no-fail-fast`
- `cargo nextest run -p siumai --test vertex_maas_openai_compat_url_alignment_test --features google-vertex,gcp --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test google_vertex_builder_alignment_test --features google-vertex,gcp --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test anthropic_vertex_builder_alignment_test --features google-vertex,gcp --no-default-features --no-fail-fast`

### Google Cloud auth provider ownership

Surface: `siumai_core::auth::{adc,service_account}` with `AdcTokenProvider`,
`ServiceAccountCredentials`, and `ServiceAccountTokenProvider`

Owner: `siumai-provider-google-vertex::auth::{adc,service_account}`. The facade compatibility path
`siumai::experimental::auth::{adc,service_account}` re-exports the provider-owned modules behind
the `gcp` feature.

Current users: Google Vertex and Anthropic-on-Vertex provider builders, registry Google/Vertex auth
fallbacks, Vertex MaaS auth fallback, facade service-account tests, and Google Vertex examples.

Canonical replacement: import Google Cloud auth implementations from
`siumai-provider-google-vertex::auth::{adc,service_account}` for provider/registry internals, or
from `siumai::experimental::auth::{adc,service_account}` for facade-level compatibility code.
Continue using `siumai_core::auth::TokenProvider` for the provider-agnostic trait contract.

Keep, move, or remove: move implementations out of `siumai-core`. `siumai-core::auth` keeps the
generic `TokenProvider` contract plus `StaticTokenProvider` only.

Migration note needed: yes for direct imports from `siumai_core::auth::adc::*` or
`siumai_core::auth::service_account::*`.

Removal window: direct core Google Cloud auth provider implementations are removed in this fearless
refactor slice.

Validation:

- `cargo check -p siumai-core --features gcp --no-default-features`
- `cargo check -p siumai-provider-google-vertex --no-default-features --features google-vertex,gcp`
- `cargo check -p siumai-registry --tests --features google-vertex,gcp --no-default-features`
- `cargo check -p siumai --tests --features google-vertex,gcp --no-default-features`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-provider-google-vertex --test gcp_auth_alignment_test --no-default-features --features google-vertex,gcp --no-fail-fast`
- `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex,gcp --no-fail-fast`
- `cargo nextest run -p siumai --test service_account_provider_test --features google-vertex,gcp --no-default-features --no-fail-fast`

### AI SDK rerank result and event carriers

Surface: `RerankResponseMetadata`, `RerankRanking`, `RerankResult`, `RerankStartEvent`,
`RerankEndEvent`, `RerankingModelCallStartEvent`, `RerankingModelCallRanking`, and
`RerankingModelCallEndEvent`

Owner: `siumai-spec::types::ai_sdk::rerank`

Current users: rerank result envelopes, rerank callback event payloads, response metadata
conversion tests, and public imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/rerank.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec rerank result/event data and moved out of the oversized
`ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`

### AI SDK response metadata carriers

Surface: `ImageModelResponseMetadata`, `VideoModelResponseMetadata`,
`SpeechModelResponseMetadata`, and `TranscriptionModelResponseMetadata`

Owner: `siumai-spec::types::ai_sdk::response_metadata`

Current users: modality result envelopes, generation error carriers, response metadata conversion
tests, and public imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/response_metadata.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec response metadata and moved out of the oversized `ai_sdk`
module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK language-model metadata carriers

Surface: `LanguageModelRequestMetadata`, `LanguageModelResponseMetadata`,
`LanguageModelV4RequestMetadata`, `LanguageModelV4ResponseMetadata`,
`LanguageModelV4GenerateResponseMetadata`, and `LanguageModelV4StreamResponseMetadata`

Owner: `siumai-spec::types::ai_sdk::language_model_metadata`

Current users: generate-text/object response envelopes, language-model V4 generate/stream result
envelopes, model-call callback payloads, response metadata conversion tests, and public imports
through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/language_model_metadata.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec request/response metadata data and moved out of the
oversized `ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK media result envelopes

Surface: `GenerateImageResult`, `Experimental_GenerateImageResult`, `GenerateVideoResult`,
`SpeechResult`, `Experimental_SpeechResult`, `TranscriptionSegment`, `TranscriptionResult`, and
`Experimental_TranscriptionResult`

Owner: `siumai-spec::types::ai_sdk::media_results`

Current users: image/video/speech/transcription helper result tests, generated file carriers,
modality response metadata carriers, usage carriers, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/media_results.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec result envelopes and moved out of the oversized
`ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none for the stable result names. Experimental aliases remain compatibility
exports while upstream AI SDK keeps those names.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK language-model V4 result envelopes

Surface: `LanguageModelV4FinishReason`, `LanguageModelV4GenerateResult`, and
`LanguageModelV4StreamResult`

Owner: `siumai-spec::types::ai_sdk::language_model_results`

Current users: language-model V4 generate result tests, provider content projections, usage
carriers, request/response metadata carriers, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/language_model_results.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec result envelopes and moved out of the oversized
`ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK source citation carriers

Surface: `Source`, `ImageModelProviderMetadata`, and `VideoModelProviderMetadata`

Owner: `siumai-spec::types::ai_sdk::source`

Current users: generate-text content parts, text-stream source parts, language-model V4 source
projection helpers, source serde tests, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/source.rs`; the public path remains stable.

Keep, move, or remove: kept as pure source citation data and moved out of the oversized `ai_sdk`
module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK generate-object result and event carriers

Surface: `GenerateObjectOutputStrategy`, `GenerateObjectResponseMetadata`,
`GenerateObjectStartEvent`, `GenerateObjectStepStartEvent`, `GenerateObjectStepEndEvent`, and
`GenerateObjectEndEvent`

Owner: `siumai-spec::types::ai_sdk::generate_object`

Current users: structured-output result envelopes, object stream tests, callback event payload
tests, request/response metadata carriers, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/generate_object.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec structured-output result/event data and moved out of the
oversized `ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK generate-text result and callback event carriers

Surface: `GenerateTextContentPart`, `ResponseMessage`, `GenerateTextReasoningPart`,
`GenerateTextStepReasoningPart`, `GenerateTextModelInfo`, `GenerateTextResponseMetadata`,
`GenerateTextStepResult`, `StepResult`, `DefaultStepResult`, `GenerateTextResult`,
`GenerateTextStartEvent`, `GenerateTextStepStartEvent`, `PrepareStepOptions`,
`PrepareStepResult`, `GenerateTextStepEndEvent`, `GenerateTextEndEvent`,
`StreamTextLifecycleChunkType`, `StreamTextLifecycleChunk`, `StreamTextChunk`, and
`StreamTextChunkEvent`

Owner: `siumai-spec::types::ai_sdk::generate_text`

Current users: generate-text result tests, stop-condition helpers, tool approval and repair
callback payloads, stream text callback payloads, usage and request/response metadata carriers, and
public imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/generate_text.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec result/event data and moved out of the oversized
`ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK output content and tool-output carriers

Surface: `TextOutput`, `CustomOutput`, `FileOutput`, `ReasoningOutput`,
`ReasoningFileOutput`, `ToolCall`, `ToolResult`, `ToolError`, `ToolOutput`,
`ToolOutputDenied`, `ToolApprovalRequestOutput`, and `ToolApprovalResponseOutput`

Owner: `siumai-spec::types::ai_sdk::output_parts`

Current users: generate-text content parts, text-stream parts, language-model V4 content
projection helpers, tool execution callback payloads, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/output_parts.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec output/tool data and moved out of the oversized
`ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK flow-control helpers

Surface: `StopCondition`, `is_step_count`, `step_count_is`, `is_loop_finished`,
`has_tool_call`, `is_stop_condition_met`, `filter_active_tools`,
`experimental_filter_active_tools`, `PruneMessagesOptions`, `PruneToolCallRule`, and
`prune_messages`

Owner: `siumai-spec::types::ai_sdk::flow_control`

Current users: generate-text step control tests, active-tool filtering tests, message pruning
tests, and public imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/flow_control.rs`; the public path remains stable.

Keep, move, or remove: kept as pure symbolic data/helpers and moved out of the oversized
`ai_sdk` module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: deprecated `step_count_is` remains only as source compatibility for upstream naming.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK tool lifecycle carriers

Surface: `CallbackModelInfo`, `ToolApprovalStatus`, `ToolApprovalConfiguration`,
`ToolApprovalDecisionContext`, tool-call repair carriers, tool execution callback events, and
deprecated callback event aliases

Owner: `siumai-spec::types::ai_sdk::tool_lifecycle`

Current users: generate-text callback event tests, tool approval/repair tests, tool execution
event tests, and public imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/tool_lifecycle.rs`; the public path remains stable.

Keep, move, or remove: kept as passive callback/event data and moved out of the oversized
`ai_sdk` module file.

Migration note needed: no public path change for normal users; deprecated aliases should remain
documented as compatibility-only.

Removal window: deprecated callback aliases remain temporarily while callers migrate to the
canonical event names.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK timeout carriers

Surface: `TimeoutConfigurationSettings`, `TimeoutConfiguration`, `get_total_timeout_ms`,
`get_step_timeout_ms`, `get_chunk_timeout_ms`, and `get_tool_timeout_ms`

Owner: `siumai-spec::types::ai_sdk::timeout`

Current users: request option helpers, timeout helper tests, and public imports through the stable
`siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/timeout.rs`; the public path remains stable.

Keep, move, or remove: kept as pure timeout data/helpers and moved out of the oversized `ai_sdk`
module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK object-stream part carriers

Surface: `ObjectStreamObjectPart`, `ObjectStreamTextDeltaPart`, `ObjectStreamErrorPart`,
`ObjectStreamFinishPart`, and `ObjectStreamPart`

Owner: `siumai-spec::types::ai_sdk::object_stream`

Current users: object stream serde tests, structured-output streaming consumers, usage and
response metadata carriers, and public imports through the stable `siumai_spec::types::ai_sdk::*`
surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/object_stream.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec stream data and moved out of the oversized `ai_sdk`
module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK text-stream and language-model stream part carriers

Surface: `TextStreamTextStartPart`, `TextStreamTextDeltaPart`, `TextStreamTextEndPart`,
`TextStreamReasoningStartPart`, `TextStreamReasoningDeltaPart`,
`TextStreamReasoningEndPart`, `TextStreamCustomPart`, `TextStreamToolInputStartPart`,
`TextStreamToolInputDeltaPart`, `TextStreamToolInputEndPart`, `TextStreamFilePart`,
`TextStreamReasoningFilePart`, `TextStreamStartStepPart`, `TextStreamFinishStepPart`,
`TextStreamStartPart`, `TextStreamFinishPart`, `TextStreamAbortPart`, `TextStreamErrorPart`,
`TextStreamRawPart`, `TextStreamPart`, `LanguageModelStreamModelCallStartPart`,
`LanguageModelStreamModelCallEndPart`, `LanguageModelStreamModelCallResponseMetadataPart`,
`LanguageModelStreamPart`, `ExperimentalLanguageModelStreamPart`, and
`Experimental_LanguageModelStreamPart`

Owner: `siumai-spec::types::ai_sdk::text_stream`

Current users: stream text result tests, language-model stream tests, `StreamTextChunkEvent`,
provider stream projections, request/response metadata carriers, usage carriers, and public imports
through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/text_stream.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec stream data and moved out of the oversized `ai_sdk`
module file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none for stable names. Experimental aliases remain compatibility exports while
upstream AI SDK keeps those names.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK UI message carriers

Surface: `UI_MESSAGE_STREAM_HEADERS`, UI message aliases, helper predicates,
`CreateUIMessage`, chat transport option carriers, completion option carriers, and
`UIMessageStreamOptions`

Owner: `siumai-spec::types::ai_sdk::ui_message`

Current users: UI message serde/helper tests, chat transport configuration carriers, completion
hook option tests, and public imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/ui_message.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec/UI data and moved out of the oversized `ai_sdk` module
file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none for stable names.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK UI message chunk carriers

Surface: `UiMessageChunk`, `UIMessageChunk`, `DataUIMessageChunk`, `InferUIMessageChunk`,
`is_data_ui_message_chunk`, and concrete `UiMessage*Chunk` structs

Owner: `siumai-spec::types::ai_sdk::ui_message_chunks`

Current users: UI message stream serde tests, stream transport compatibility helpers, and public
imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/ui_message_chunks.rs`; the public path remains stable.

Keep, move, or remove: kept as pure stream data and moved out of the oversized `ai_sdk` module
file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`

### AI SDK usage carriers

Surface: `LanguageModelV4InputTokens`, `LanguageModelV4OutputTokens`, `LanguageModelV4Usage`,
`LanguageModelInputTokenDetails`, `LanguageModelOutputTokenDetails`, `LanguageModelUsage`,
`EmbeddingModelUsage`, `ImageModelUsage`, `add_language_model_usage`, and `add_image_model_usage`

Owner: `siumai-spec::types::ai_sdk::usage`

Current users: language-model V4 generate/stream results, generate-text/object results,
embedding/image result envelopes, callback event payloads, usage conversion tests, and public
imports through the stable `siumai_spec::types::ai_sdk::*` surface.

Canonical replacement: continue importing through `siumai_spec::types::ai_sdk::*`. The physical
owner is now `ai_sdk/usage.rs`; the public path remains stable.

Keep, move, or remove: kept as pure spec usage data and moved out of the oversized `ai_sdk` module
file.

Migration note needed: no public path change for normal users; only internal source ownership
changed.

Removal window: none.

Validation:

- `cargo check -p siumai-spec --no-default-features`
- `cargo nextest run -p siumai-spec --test ai_sdk_module_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --no-default-features --no-fail-fast`

### Facade broad type path

Surface: `siumai::compat::types::*` and `siumai::prelude::compat::types::*`.
The historical root `siumai::types::*` path was removed from the facade root.

Owner: facade crate compatibility surface; canonical type ownership remains in `siumai-core`,
`siumai-spec`, and extension/provider crates.

Current users: migration-oriented facade imports that intentionally need a single catch-all type
namespace while older code is moved to stable family, extension, provider extension, or owning-crate
paths.

Canonical replacement: prefer `siumai::prelude::unified::*` for the stable family surface,
`siumai::prelude::extensions::*` for non-family extension types, and
`siumai::provider_ext::<provider>::*` for provider-specific APIs. Internal crates should import from
the owning crate instead of the facade path. If a migration temporarily needs the broad namespace,
use `siumai::compat::types::*` or `siumai::prelude::compat::types::*`.

Keep, move, or remove: root path removed. Keep the explicit compat paths temporarily because they
make migration-only broad imports visible and separate them from the stable facade root. They are
intentionally not treated as the future stable surface because they mirror broad
`siumai-core::types::*`.

Migration note needed: done in `docs/migration/migration-0.11.0-beta.7.md`.

Removal window: no earlier than `0.12.0` for the explicit compat catch-all namespace.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade execution middleware advanced surface

Surface: `LanguageModelMiddleware`, `MiddlewareBuilder`, `NamedMiddleware`, and related
`siumai-core::execution::middleware` contracts, including preset helpers such as
`ReasoningTagPresets::for_model(...)` and `SystemMessageModeWarningMiddleware`.

Owner: `siumai-core` owns the provider-agnostic middleware contracts and execution hooks. The
facade exposes them through `siumai::experimental::execution::middleware` for custom providers,
registry wiring, and advanced integration code.

Current users: middleware override tests, provider builder consistency tests, custom-provider
implementations, and downstream users that intentionally install model-level middleware.

Canonical replacement: import middleware contracts from
`siumai::experimental::execution::middleware::*`. Stable application code should consume model
families through `prelude::unified` rather than implementing execution middleware from the stable
prelude.

Keep, move, or remove: move out of `prelude::unified`; keep the explicit experimental facade path
because middleware is an advanced execution extension point rather than a stable model-family data
type. Provider-specific reasoning tag routing was removed from `siumai-core`; callers that need a
provider-specific tag choose it explicitly with preset helpers such as
`ExtractReasoningMiddleware::with_tag(ReasoningTagPresets::thought())`, or a provider/facade
extension can provide a provider-owned default. `SystemMessageModeWarningMiddleware` now receives
the provider option namespace at construction time, so core can emit the Vercel-aligned warning
without knowing concrete provider fallback names; provider/facade wiring owns aliases or
multi-namespace policy.

Migration note needed: yes for code that imported `LanguageModelMiddleware` from
`prelude::unified::*`, and for code that relied on provider/model-name heuristics in
`ReasoningTagPresets::for_model(...)`. Manual `SystemMessageModeWarningMiddleware` construction
must also pass the provider option namespace explicitly.

Removal window: completed for `prelude::unified` during this Track F slice. The experimental path
remains.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib execution::middleware::presets::extract_reasoning::tests --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib execution::middleware::presets::system_message_mode_warning::tests --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib execution::middleware::auto::tests --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test openai_chat_messages_fixtures_alignment_test --no-default-features --features openai --no-fail-fast`

### Facade provider builder compatibility entry

Surface: `siumai::compat::Provider` and `siumai::provider::*`

Owner: `siumai::compat::Provider` owns the compatibility entry type. The facade root no longer
re-exports that type; concrete provider builder implementations remain provider-owned.
`siumai::provider::*` is a builder-era facade shim that directly re-exports registry-owned `Siumai`
/ `SiumaiBuilder` types.

Current users: historical provider-specific builder construction and provider parity tests that
intentionally import `Provider::openai()` style builders from `siumai::compat` or
`siumai::prelude::compat`.

Canonical replacement: prefer registry model handles from
`siumai::prelude::unified::registry::*` or config-first provider clients for stable construction.
When builder-style construction is intentionally needed during migration, import
`siumai::compat::Provider` or `siumai::prelude::compat::Provider` explicitly.

Keep, move, or remove: move the implementation body out of `siumai/src/lib.rs` and into the
explicit compat surface, then remove the root `siumai::Provider` re-export. This slice removed the root `siumai::Provider` re-export; the explicit compatibility path now owns the entry type, and examples, public-surface tests, and migration docs use `siumai::compat::Provider` or `siumai::prelude::compat::Provider`.

Compat re-exports now bind directly to registry-owned `Siumai` / `SiumaiBuilder` types rather than
routing through the facade `provider` shim. The shim remains public for source compatibility but is
hidden from generated facade docs, and facade production code now binds upload helper impls through
`crate::compat::Siumai` instead of `crate::provider::Siumai`. Facade tests and examples now use
`siumai::compat::*` or stable registry paths instead of `siumai::provider::*`, with a source guard
to prevent the shim from returning as the default test/example construction path. Ordinary tests,
the large provider public-path parity suite, and public-surface import coverage now use
`siumai::compat::Provider` for builder-era construction coverage. The facade boundary guard no
longer allowlists the root `siumai::Provider` alias.

Provider extension package helpers that return `SiumaiBuilder` also bind directly to the
registry-owned type. The facade boundary guard rejects `crate::provider::SiumaiBuilder` references
under `siumai/src/provider_ext`. Provider extension helpers also avoid the root `crate::Provider`
alias; when they need the centralized compatibility builder entry they use the explicit
`crate::compat::Provider` path.

Migration note needed: completed in `docs/migration/migration-0.11.0-beta.7.md`; replace
`siumai::Provider` with `siumai::compat::Provider` or `siumai::prelude::compat::Provider`.

Removal window: completed for the facade root alias in this Track F slice. The explicit compat path
remains time-bounded alongside provider-owned config/client constructors and registry model handles.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo check -p siumai --test provider_public_path_parity_test --features openai,anthropic,google --no-default-features`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade low-level streaming exports

Surface: `SseEventConverter`, `JsonEventConverter`, `StreamFactory`, `ChatByteStream`,
`TypedStreamPart`, `UnsupportedStreamPartBehavior`, stream encoders, processors, and other
low-level `siumai-core::streaming` implementation exports.

Owner: `siumai-core` owns the provider-agnostic runtime contracts and implementation helpers.
The facade exposes them through `siumai::experimental::streaming` for provider, gateway,
transcoding, and custom-stream integrations.

Current users: protocol bridge tests, provider fixture alignment tests, transcoding tests, custom
stream factory injection tests, and downstream advanced integrations that implement or serialize
stream converters.

Canonical replacement: keep stable stream consumption types such as `ChatStream`,
`ChatStreamEvent`, `ChatStreamPart`, and `ChatStreamHandle` in `prelude::unified`. Import
converter/factory/encoder/typed-bridge stream internals from `siumai::experimental::streaming::*`.

Keep, move, or remove: move out of `prelude::unified`; keep the explicit experimental facade path
for advanced integrations. The facade must not mirror `siumai_core::streaming::*` through the
stable prelude.

Migration note needed: yes before publishing this as a breaking change for code that imported
converter/factory internals from `prelude::unified`.

Removal window: completed for `prelude::unified` during this Track F slice. Root experimental paths
remain.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Core provider-agnostic docs/examples residue

Surface: comments and examples in `siumai-core/src/completion.rs`,
`siumai-core/src/execution/transformers/stream.rs`, `siumai-core/src/standards/mod.rs`,
`siumai-core/src/utils/builder_helpers.rs`, `siumai-core/src/custom_provider/guide.rs`,
`siumai-core/src/observability/tracing/README.md`, and `siumai-core/src/utils/url.rs`.

Owner: `siumai-core` provider-agnostic runtime documentation and examples.

Current users: maintainers reading core docs/examples and core unit tests that exercise generic URL
helpers.

Canonical replacement: describe these surfaces as provider-agnostic core behavior. Concrete
provider wire endpoints, OpenAI-compatible compatibility language, and concrete provider URLs
belong in provider/protocol crates or explicitly marked compatibility docs.

Keep, move, or remove: remove provider-specific wording from core docs/examples. Route fallback debt
has been resolved by removing provider-shaped fallback helpers, default endpoint strings, and the
historical string-returning `ProviderSpec` route hooks from core.

The follow-up provider-neutral fixture slice also removed concrete provider/model examples from
core transformer docs, traits, telemetry comments, custom provider guide examples, tool-name
mapping tests, structured-output stream-end fixtures, UI conversion fixtures, streaming carrier
tests, provider reference/option utility tests, URL helper tests, and the generic retry fallback
classifier. The retry fallback now collects only generic request/trace headers; concrete provider
request-id headers are expected to live behind provider-owned
`ProviderSpec::classify_http_error(...)` hooks when needed. The broad scan is now enforced by
`core_provider_boundary_test::core_source_does_not_use_provider_model_fixture_literals`.

Migration note needed: no public API migration; this is documentation/example boundary cleanup.

Removal window: immediate for docs/examples. The route hook cleanup now uses the fallible-only
`try_*_url(...)` contract tracked above.

Validation:

- `cargo check -p siumai-core --no-default-features`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --no-default-features --no-fail-fast`
- `cargo fmt --package siumai-core --check`
- `rg -n '"(anthropic|gemini|google|openai|minimaxi|bedrock|azure|deepseek|groq|ollama|cohere|togetherai|xai)"|anthropic-|gemini-|openai-|gpt-|claude-|api\.anthropic|api\.openai|generativelanguage\.googleapis|aiplatform\.googleapis' siumai-core/src -g "*.rs"`
- `rg -n -i 'openai|anthropic|gemini|google|azure|bedrock|groq|ollama|cohere|deepseek|minimaxi|togetherai|xai|gpt-|claude-|api\.anthropic|api\.openai|generativelanguage\.googleapis|aiplatform\.googleapis|x-openai-request-id|x-goog-request-id' siumai-core/src -g "*.rs"`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test core_source_does_not_use_provider_model_fixture_literals --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --lib --no-default-features --no-fail-fast`
- `git diff --check -- <touched siumai-core files>`

### Facade streaming tool-call compatibility helpers

Surface: `StreamingToolCallDelta`, `StreamingToolCallFunctionDelta`,
`StreamingToolCallTracker`, `StreamingToolCallTrackerOptions`, and
`StreamingToolCallTypeValidation`

Owner: compatibility facade plus the core utility implementation while this helper remains public.
The long-term owner should be either a provider/protocol package or a documented compat namespace
because the helper models indexed provider streaming deltas rather than a stable model-family API.

Current users: public-surface import tests and downstream users that imported these helper aliases
from the historical root/facade surface. Workspace provider streaming implementations do not depend
on this helper.

Root aliases were removed during the Track F facade cleanup so the explicit compat surface is now
the only migration import path.

Canonical replacement: use `siumai::compat::*` or `siumai::prelude::compat::*` for
migration-oriented imports. Stable application code should use `prelude::unified` model-family
streams rather than constructing provider streaming deltas manually.

Keep, move, or remove: moved out of `prelude::unified` and the facade root; keep only explicit
`compat` / `prelude::compat` aliases for source-compatible migration imports. Root aliases were
removed during the Track F facade cleanup so the stable root no longer carries this low-level
provider-utils helper family.

Migration note needed: complete for root-alias removal; still needed before moving the
implementation to a protocol crate.

Removal window: completed for root and `prelude::unified` aliases during Track F. Candidate future
path is to migrate the implementation to the OpenAI protocol package or a dedicated compatibility
module and leave only explicit compat re-exports.

Validation:

- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade compat AI SDK parity aliases

Surface: `CallSettings`, `Experimental_GenerateImageResult`, `Experimental_GeneratedImage`,
`Experimental_LanguageModelStreamPart`, `ExperimentalLanguageModelStreamPart`,
`Experimental_SpeechResult`, `Experimental_TranscriptionResult`,
`experimental_filter_active_tools`, and `step_count_is`

Owner: facade crate compatibility surface; physical ownership remains in `siumai-core::types` or
the AI SDK-aligned spec modules re-exported through core.

Current users: source-compatible AI SDK imports, public-surface compile tests, and downstream code
that imported deprecated upstream AI SDK names from the facade.

Canonical replacement: prefer the stable non-experimental names where they exist:
`GenerateImageResult`, `GeneratedImage`, `LanguageModelStreamPart`, `SpeechResult`,
`TranscriptionResult`, `filter_active_tools`, and `is_step_count`. Prefer `LanguageModelCallOptions`
plus `RequestOptions` over deprecated `CallSettings`.

Keep, move, or remove: move out of `prelude::unified`; keep them temporarily in
`siumai::compat` / `prelude::compat` because the names match upstream AI SDK compatibility and are
useful during migration. The stable unified prelude should use non-experimental names.

Migration note needed: yes for users that imported these aliases from `prelude::unified`.
New documentation should steer users away from experimental aliases and deprecated helper names.

Removal window: completed for `prelude::unified` during this Track F slice. Removal from
`siumai::compat` is not set.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade JSON event stream parser helper path

Surface: `parse_json_event_stream(...)`

Owner: core streaming utility re-exported by the facade root.

Current users: tooling/public-surface tests and downstream integrations that parse SSE `data:`
JSON payloads from byte streams.

Canonical replacement: call the explicit root helper as `siumai::parse_json_event_stream(...)`.
Advanced stream integration code can import lower-level parser/converter utilities from
`siumai::experimental::streaming::*`.

Keep, move, or remove: move out of direct `prelude::unified`; keep the explicit root helper. It is
a low-level JSON/SSE parser helper, not a stable model-family prelude name.

Migration note needed: yes for code that imported `parse_json_event_stream` from
`prelude::unified::*`.

Removal window: completed for direct `prelude::unified` export during this Track F slice.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade file and skill upload helper paths

Surface: `UploadFileApi`, `UploadFileOptions`, `UploadFileResult`, `UploadSkillApi`,
`UploadSkillFile`, `UploadSkillOptions`, `UploadSkillResult`, `upload_file`, and `upload_skill`.

Owner: facade explicit helper modules `siumai::files` and `siumai::skills`; lower capability
contracts remain in `siumai-core`.

Current users: upload helper tests, provider file/skill resources, and downstream code using
AI SDK-style file or skill upload helpers.

Canonical replacement: import helper types from `siumai::files::*` and `siumai::skills::*`, or call
the root helper functions as `siumai::upload_file(...)` and `siumai::upload_skill(...)`.
`prelude::unified` keeps `files` and `skills` modules available for navigation but does not export
the upload helper names directly.

Keep, move, or remove: move direct helper names out of `prelude::unified`; keep explicit stable
module and root helper paths. File and skill upload are extension/helper surfaces, not one of the
six stable model-family prelude names.

Migration note needed: yes for code that imported direct upload helper names from
`prelude::unified::*`.

Removal window: completed for direct `prelude::unified` names during this Track F slice.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade retry API helper paths

Surface: `RetryOptions`, `RetryBackend`, `RetryPolicy`, `BackoffRetryExecutor`, `retry`,
`retry_with`, `maybe_retry`, `classify_http_error`, `backoff_executor_for_provider`,
`backoff_options_for_provider`, and `retry_for_provider`.

Owner: facade explicit runtime module `siumai::retry_api`; provider-agnostic retry types and helpers
remain owned by `siumai-core::retry_api`, while provider-tuned defaults remain in the facade so
`siumai-core` stays provider-agnostic.

Current users: per-call family option structs, builder/registry retry configuration, retry-focused
tests, and downstream code that directly controls retry behavior.

Canonical replacement: import direct retry controls from `siumai::retry_api::*`, for example
`use siumai::retry_api::{RetryOptions, RetryPolicy, retry_with};`. Stable family helpers and
request option structs continue to accept `RetryOptions` where retry is a call option.

Keep, move, or remove: move direct helper names out of `prelude::unified`; keep the explicit stable
runtime module. Retry policy control is a cross-cutting runtime concern, not one of the six stable
model-family prelude names.

Migration note needed: yes for code that imported direct retry names from `prelude::unified::*`.

Removal window: completed for direct `prelude::unified` names during this Track F slice.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test retry_wrapping_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test streaming_tests chat_stream_connect_retry --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test streaming_tests http_connect_timeout_retry --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade runtime tooling module path

Surface: `prelude::unified::tooling` module re-export.

Owner: facade explicit runtime module `siumai::tooling`; physical implementation remains in
`siumai-core::tooling`.

Current users: public tooling runtime tests and downstream integrations that need advanced tool
execution contexts, stream execution helpers, or runtime metadata.

Canonical replacement: import the broader runtime module explicitly through `siumai::tooling::*`.
Stable AI SDK-style helper names such as `tool`, `dynamic_tool`, `ToolExecutionOptions`,
`ToolExecutionResult`, `ToolSet`, `ExecutableTool`, `ExecutableTools`, `execute_tool`, and
`model_messages_from_chat_messages` remain direct `prelude::unified::*` imports.

Keep, move, or remove: remove the whole-module re-export from `prelude::unified`; keep explicit
root module and direct stable helper names. This prevents the stable prelude from mirroring future
tooling runtime internals while preserving the audited AI SDK root helper surface.

Migration note needed: no broad user migration expected because direct helper names remain in the
stable prelude and advanced code already imports `siumai::tooling::*` explicitly. Add a migration
note only if downstream code relied on the module name from `prelude::unified::*`.

Removal window: completed for direct `prelude::unified` module re-export during this Track F slice.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test tooling_runtime_public_surface_test --features openai,anthropic,google --no-default-features --no-fail-fast`

### Facade unified prelude runtime compatibility aliases

Surface: `CancelHandle`

Owner: facade crate runtime surface; physical ownership remains in `siumai-core::types`.

Current users: request option builders, stream cancellation helpers, public-surface compile tests,
and downstream code that uses cancellation through the stable facade.

Canonical replacement: keep using `CancelHandle` when request cancellation is needed. It is not a
deprecated AI SDK spelling; it is a runtime cancellation handle that moved out of `siumai-spec` and
is now owned by core.

Keep, move, or remove: keep temporarily in `prelude::unified` while cancellation remains part of the
stable request option surface. It is guarded by
`stable_unified_prelude_keeps_only_audited_compatibility_and_runtime_aliases`, so new
compatibility/runtime aliases cannot enter the stable prelude without being classified here first.

Migration note needed: no for the facade path. The spec-to-core movement is documented separately.

Removal window: not set.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
