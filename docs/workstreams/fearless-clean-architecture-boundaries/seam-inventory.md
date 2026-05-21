# Fearless Clean Architecture Boundaries — Seam Inventory

Status: Active
Last updated: 2026-05-21

## Purpose

FCAB-020 freezes the current architecture seams before the fearless refactor moves code. The
inventory is intentionally source-guard oriented: every high-risk seam below has either an existing
guard, a refreshed guard in this task, or a named follow-up task that must add a stronger guard after
the code moves.

The target architecture is inspired by `repo-ref/ai/packages/provider`,
`repo-ref/ai/packages/provider-utils`, and `repo-ref/ai/packages/openai-compatible`: stable provider
interfaces stay small, compatibility paths are explicit, protocol conversion is protocol-owned, and
provider-specific behavior does not leak into core.

## Current Guard Map

| Seam | Existing or refreshed guard | Current proof | Gap for later task |
| --- | --- | --- | --- |
| Registry family construction vs compatibility clients | `siumai-registry/src/registry/entry/boundary_tests.rs` | Stable family handles do not execute through `compat_*_client_with_ctx` or `LlmClient` capability downcasts. FCAB-020 guards that `ProviderFactory` generic client methods remain explicitly named `compat_*` compatibility aliases. FCAB-030 adds `ProviderFamilyFactory`, `ProviderCompatibilityFactory`, and `ProviderExtensionFactory` facets so registry execution can depend on narrower boundaries while custom providers keep implementing `ProviderFactory`. | FCAB-040 should convert built-in provider factories toward the narrowed story and remove redundant provider glue. |
| Built-in provider factory projection glue | `siumai-registry/tests/factory_architecture_boundary_test.rs`, `siumai-registry/src/registry/factories/*.rs` | FCAB-040 centralizes typed-client `Arc` projection in helper methods for OpenAI, OpenAI-compatible, Azure, Anthropic, Gemini, Google Vertex, Google Vertex MaaS/xAI, Groq, xAI, DeepSeek, DeepInfra, Fireworks, TogetherAI, MiniMaxi, Ollama, Cohere, Bedrock, and Anthropic Vertex. Factory methods now delegate to helpers instead of repeating `build_*_with_ctx` + `Arc::new` or routing same-family methods through `compat_language_client_with_ctx`. | Extension-only compat adapters remain explicit. FCAB-050/060 can now deepen OpenAI-compatible runtime/vendor ownership without chasing repeated registry projection glue. |
| Core neutrality and provider-utils ownership | `siumai-core/tests/core_provider_boundary_test.rs`, `siumai-provider-utils/src/*`, `provider-utils-classification.md` | `siumai-core` has no registry/facade/provider/protocol dependencies, production core does not import provider/protocol crates, route specs use fallible hooks, and core does not own provider-specific bridge contracts. FCAB-090 introduced `siumai-provider-utils` as the provider/protocol utility seam. FCAB-100 moved the remaining spec-only AI SDK-style utility implementations there and classified leftover core-owned utilities: `cancel` is stable core runtime, `streaming_tool_call` is explicit compat. Matching moved `siumai-core::utils::*` modules are compatibility aliases. | FCAB-120 should decide whether the facade root keeps explicit low-level utility imports or demotes more of them after the compatibility window. |
| Directional content and legacy compatibility carriers | `siumai/tests/facade_architecture_boundary_test.rs`, `siumai/tests/public_surface_imports_test.rs`, `docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md`, `docs/workstreams/fearless-content-part-boundary-split/direct-content-part-scan.md` | FCAB-070 makes request prompt parts, generated output parts, and legacy compatibility content visibly separate through `content::{prompt, output, compat}` namespaces. FCAB-080 moves high-value production response construction for OpenAI-compatible chat, Anthropic streaming, Cohere, and Ollama behind named `response_content` adapters, while `content-part-adapter-audit.md` classifies the remaining direct appearances. | FCAB-090/100 should keep core/provider utility movement from reintroducing broad legacy content imports. FCAB-110 owns bridge target wire parsing. FCAB-120 owns facade/public-surface tightening. |
| OpenAI-compatible protocol/runtime/vendor split | `siumai-protocol-openai/tests/openai_compat_boundary_test.rs`, `siumai-registry/tests/factory_architecture_boundary_test.rs`, `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/tests.rs`, `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs`, `siumai-provider-openai/src/providers/openai/client.rs` | The OpenAI-compatible client shell stays split into runtime, compatibility, and type modules. FCAB-050 moved `/completions` response conversion and SSE parser state to protocol-owned `siumai-protocol-openai/src/standards/openai/compat/completion.rs`. The OpenAI-compatible provider runtime delegates response conversion and stream parsing through protocol conversion types and no longer owns a local `completion/streaming.rs` module. FCAB-060 moved TogetherAI image request/response mapping and HTTP execution into `siumai-provider-togetherai::providers::togetherai::image`; the registry factory now only builds/projects provider-owned image and rerank clients plus the shared OpenAI-compatible text/audio runtime. Source guards `openai_compatible_completion_streaming_conversion_is_protocol_owned` and `togetherai_provider_crate_owns_image_runtime` prevent those responsibilities from drifting back into provider runtime or registry glue. | Native OpenAI still keeps provider-specific completion conversion intentionally until a native-provider seam task says otherwise. Further vendor-specific long tails should be split only when a provider adds truly distinct runtime behavior. |
| Bridge target adapters | `siumai-bridge/src/request/normalize.rs`, `siumai-bridge/src/request/normalize/gemini_generate_content.rs`, `siumai-protocol-gemini/src/standards/gemini/request_bridge.rs`, `siumai-bridge/src/response`, `siumai-bridge/src/stream`, `siumai-core/tests/core_provider_boundary_test.rs::core_does_not_own_provider_specific_bridge_contracts`, `siumai/tests/facade_architecture_boundary_test.rs::experimental_bridge_is_owned_by_bridge_crate_and_reexported_by_facade` | Core does not define `BridgeTarget`, the facade re-exports the dedicated bridge crate, and FCAB-110 moved Gemini GenerateContent request JSON normalization into `siumai-protocol-gemini::standards::gemini::request_bridge`. The bridge keeps public wrappers, loss reports, policy, lifecycle, customization, dispatch, and thin compatibility shims. | Remaining bridge-owned OpenAI/Anthropic normalization and direct-pair modules should move only when their bridge loss/replay policy can stay out of protocol crates. FCAB-120 owns facade/export tightening. |
| Facade and public surface | `siumai/tests/facade_architecture_boundary_test.rs`, `siumai/tests/public_surface_imports_test.rs`, `docs/architecture/public-surface.md` | FCAB-120 makes facade root utility helpers provider-utils-backed, moves `ToolNameMapping` ownership to `siumai-provider-utils`, keeps only a narrow AI SDK-style helper subset in `prelude::unified`, and replaces the experimental grouped core mirror with named advanced modules. Retained broad exports are documented as explicit namespaces only. | FCAB-130 should align family taxonomy now that facade paths are tightened. |
| Family taxonomy | `siumai/tests/public_surface_imports_test.rs::public_surface_video_family_imports_compile`, `siumai/tests/facade_architecture_boundary_test.rs::family_taxonomy_documents_video_as_stable_and_music_as_extension_only`, `docs/workstreams/video-model-family-alignment/` | FCAB-130 makes Video the seventh stable task-oriented family across docs, facade comments, registry handles, and ADR policy. Music remains extension-only with no stable `MusicModel` or registry `music_model(...)` handle. | FCAB-140 should keep the integration gate from drifting back to the historical six-family wording. |

## Current Leak Inventory

### Registry and compatibility construction

- `ProviderFactory` currently lives in `siumai-registry/src/registry/entry/factory.rs` and remains
  the source-compatible custom-provider implementation trait.
- FCAB-030 introduced `ProviderFamilyFactory`, `ProviderCompatibilityFactory`, and
  `ProviderExtensionFactory` facets so registry handles can call stable family construction without
  depending on the whole legacy `LlmClient` compatibility surface.
- Stable handles in `siumai-registry/src/registry/entry/handles` already prefer native family model
  paths for text, embedding, reranking, video, and primary audio/image execution.
- Remaining compatibility handle paths are extension-only: image edit/variation and audio
  speech/transcription extension helpers.

### OpenAI-compatible packages

- `siumai-protocol-openai` owns OpenAI and OpenAI-compatible wire conversion tests and fixtures.
- FCAB-050 moved OpenAI-compatible completion response conversion and SSE stream conversion to
  protocol-owned `standards::openai::compat::completion`.
- `siumai-provider-openai-compatible` owns the reusable OpenAI-compatible HTTP execution,
  capability routing, and promoted vendor behavior, while delegating completion protocol conversion
  to `siumai-protocol-openai`.
- `siumai-provider-openai` owns native OpenAI provider behavior.
- Compatibility re-export paths under
  `providers::openai_compatible::{adapter,streaming,transformers,types,registry}` remain documented
  compatibility shims, not new implementation homes.
- Vendor packages such as Groq, xAI, DeepSeek, and TogetherAI converge on presets, quirks, typed
  options, and metadata, not duplicate shared protocol conversion logic.
- TogetherAI is a hybrid provider: text/audio reuse the OpenAI-compatible runtime, while image and
  rerank are provider-owned runtime surfaces in `siumai-provider-togetherai`. The registry factory
  composes those clients but no longer owns TogetherAI image request mapping, response parsing, or
  HTTP execution.

### Directional content

- `ContentPart` remains a legacy compatibility carrier and still appears in production paths across
  core, bridge, protocol, provider, and facade crates.
- FCAB-070 added directional content namespaces: `content::prompt` for request/model-message
  content, `content::output` for generated response output and lossless projection helpers, and
  `content::compat` / `compat::content` for legacy serde-facing carriers.
- Stable `prelude::unified` exposes only `prompt` and `output` navigation modules; it does not
  expose `ContentPart` or the legacy compat content module.
- FCAB-080 moved high-value response-side legacy construction behind named adapters:
  - `siumai-protocol-openai/src/standards/openai/compat/response_content.rs`
  - `siumai-protocol-anthropic/src/standards/anthropic/streaming/response_content.rs`
  - `siumai-provider-cohere/src/standards/cohere/response_content.rs`
  - `siumai-provider-ollama/src/standards/ollama/response_content.rs`
- Existing adapters for Anthropic parse, Gemini response, Google Interactions response, OpenAI
  Responses, and Bedrock remain the canonical homes for their response-side legacy constructors.
- `content-part-adapter-audit.md` classifies remaining direct `ContentPart` appearances as
  request-side serialization, bridge inspection/stream shims, metadata views, test-only fixtures, or
  FCAB-090/100 core streaming aggregation follow-up work.

### Core and provider-utils

- FCAB-090 chose a real `siumai-provider-utils` crate instead of a deep internal module because the
  helpers already had multiple provider/protocol callers and the dependency direction can stay clean:
  `siumai-provider-utils` depends on `siumai-spec`, not `siumai-core`.
- Historical high-churn utility modules under `siumai-core/src/utils`, plus the related
  `siumai-core/src/streaming`, `siumai-core/src/execution`, `siumai-core/src/retry`, and
  `siumai-core/src/encoding` seams, were audited before the split so the core/provider-utils
  boundary remains discoverable.
- `siumai-provider-utils` now owns the AI SDK-style spec-only adapter-helper surface:
  `builder_helpers`, `chat_request`, `data`, `download`, `error_message`, `headers`, `id`,
  `json_instruction`, `json_parse`, `mime`, `option`, `provider_options`, `provider_reference`,
  `reasoning`, `runtime`, `serial_job`, `settings`, `url`, `utf8_decoder`, and
  `validate_types`.
- Provider, protocol, and registry crates import those helpers through internal
  `crate::provider_utils` aliases, making the provider-utils seam visible without publicly mirroring
  the new crate from provider crates.
- Matching `siumai-core::utils::*` modules remain temporary compatibility aliases so older internal
  and downstream paths do not break during this workstream. FCAB-100 records the module-level
  classification in `provider-utils-classification.md`.
- `siumai-core::utils::cancel` remains a stable core runtime utility because it owns
  `CancelHandle`, `ChatStream`, and `ChatStreamHandle` wiring. `streaming_tool_call` remains an
  explicit compatibility helper because it assembles core stream parts and is exposed through
  `siumai::compat::*`, not the stable facade root.

### Bridge target adapters

- `siumai-bridge` currently owns `BridgeTarget` and target-specific request/response/stream
  normalization for OpenAI, Anthropic, and Gemini-compatible targets.
- This is acceptable as the baseline, but it is not the desired final ownership boundary: protocol
  packages should own wire parsing where feasible.

### Facade and family taxonomy

- FCAB-120 narrows `siumai/src/lib.rs` so the stable unified prelude keeps family-first types,
  schema/ID/tool helpers, UI/tool runtime helpers, and navigation modules, while low-level
  provider-utils helpers require explicit facade root imports.
- `ToolNameMapping` and `create_tool_name_mapping` now live in `siumai-provider-utils::standards`;
  `siumai-core::standards` keeps only a compatibility re-export.
- `experimental::{defaults,execution,observability,params,retry,utils}` are named advanced modules
  instead of a broad grouped `siumai_core` mirror.
- Retained broad globs are limited to explicit namespaces documented in
  `docs/architecture/public-surface.md`: protocol, hosted tools, directional content, compatibility
  prelude, and experimental advanced modules.
- FCAB-130 promotes Video to the documented stable taxonomy as the seventh task-oriented family.
  The stable family list is Language, Embedding, Image, Rerank, Speech, Transcription, and Video.
  Registry exposes `video_model(...)` / `VideoModelHandle` and no `music_model(...)` handle.
- Music remains extension-only through `MusicGenerationCapability`, provider extensions, and
  compatibility language-handle delegation until a future ADR defines a first-class family.

## Parallelism After FCAB-020

- FCAB-030 should start first because it defines the registry/factory contract used by later provider
  rewires.
- FCAB-050 can run in parallel with FCAB-030 if the worker stays inside
  `siumai-protocol-openai`, `siumai-provider-openai-compatible`, and `siumai-provider-openai`.
- FCAB-070 can run in parallel with registry work if it stays focused on spec/core/facade content
  exports and does not migrate protocol/provider direct construction yet.
- FCAB-090 can start now that registry and directional-content adapter seams are stable; it should
  keep provider-utils movement separate from bridge-target ownership and facade finalization.
- FCAB-110 should wait for FCAB-080 so bridge target adapter moves do not fight the content-direction
  migration.
