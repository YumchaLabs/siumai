# Fearless Architecture Convergence - Compatibility Audit

Last updated: 2026-05-16

## Scope

This audit tracks compatibility surfaces that still depend on the legacy `LlmClient` capability
umbrella or deprecated facade aliases.

The goal is not to delete every compatibility path at once. The goal is to keep stable family
execution paths free of legacy downcasts, while making the remaining compatibility paths explicit
and time-bounded.

## `as_*_capability()` Call-Site Categories

### Stable Family Paths - Remove Or Guard

Status: guarded.

Stable family registry handles should execute through family model factories, not through
`LlmClient` downcasts.

Current guard:

- `siumai-registry::factory_architecture_boundary_test::registry_family_handles_keep_llm_client_downcasts_isolated`

Current boundary:

- `registry/entry/handles/{audio,completion,embedding,image,rerank,video}.rs` must not call
  `.as_*_capability()`.
- New stable family handle code should use `ProviderFactory::*_model_family_with_ctx(...)`.

### Registry Compatibility Adapters - Keep, Isolated

Status: keep until `ProviderFactory` no longer needs generic-client defaults.

Location:

- `siumai-registry/src/registry/entry/compat_client.rs`

Reason:

- This module adapts legacy `Arc<dyn LlmClient>` clients into family model traits.
- It is the correct isolation point for old provider factories that have not yet implemented native
  family model construction.

Deletion condition:

- Built-in providers implement native family methods for all declared stable families.
- `ProviderFactory` generic-client construction methods are renamed or moved behind an explicit
  compatibility trait.

Current convergence step:

- `ProviderFactory` exposes explicit `compat_*_client*` aliases for generic `LlmClient`
  construction.
- Registry family adapter defaults and extension-only generic-client paths now call those
  `compat_*` aliases.
- `SiumaiBuilder`'s historical `Arc<dyn LlmClient>` construction path now calls
  `compat_*_client_with_ctx` explicitly instead of the older `*_model_with_ctx` names.
- Production provider factories implement `compat_*_client*` methods for generic-client
  construction paths.
- The old generic `*_model*` and `*_model_with_ctx` trait methods have been removed. Generic
  `LlmClient` construction now uses explicit `compat_*_client*` names only.
- Bedrock and Cohere now use native family factory overrides for declared stable families, so those
  registry handles no longer depend on the default `ClientBacked*Model` compatibility bridge.
- Anthropic Vertex now uses a typed registry builder and native text-family factory override, so
  its language handle no longer depends on the default `ClientBackedLanguageModel` bridge.
- OpenAI and Azure now use native completion-family factory overrides for their declared
  completion capability, so those handles no longer depend on the default
  `ClientBackedCompletionModel` bridge.
- Gemini and Google Vertex now use native video-family factory overrides for their declared video
  surface, so those handles no longer depend on the default `ClientBackedVideoModel` bridge.
- MiniMaxi and xAI now use native video-family factory overrides for their declared video
  capability, so their video handles no longer depend on the default `ClientBackedVideoModel`
  bridge.
- `provider_factories_use_explicit_compat_for_generic_self_calls` guards provider factory source
  files against reintroducing old generic `self.*_model_with_ctx(...)` self-calls or old
  `async fn *_model_with_ctx(...)` generic-client overrides.
- `production_factories_do_not_override_legacy_generic_language_method` guards production provider
  factory source files against reintroducing direct `async fn language_model(...)` overrides.
- `production_factories_with_declared_family_surfaces_use_native_family_overrides` guards
  production provider factory source files against relying on default `ClientBacked*Model` bridge
  methods for declared stable family surfaces.
- `compatibility_audit_does_not_keep_removed_providerfactory_methods_alive` guards this audit
  against describing the removed generic factory methods as deprecated-but-kept wrappers.
- Custom factory guidance now points at native family methods first:
  `docs/architecture/registry-without-builtins.md` and
  `siumai-registry/examples/no_builtins_custom_factory.rs` use `language_model_text_with_ctx` as the
  primary construction method, while the generic-client example path uses
  `compat_language_client`.

### Language-Handle Extension Paths - Isolated Extension Adapters

Status: isolated behind registry-owned extension adapters.

Location:

- `siumai-registry/src/registry/entry/handles/language.rs`
- `siumai-registry/src/registry/entry/extension_adapters.rs`

Current seam:

- `LanguageModelHandle` calls explicit `ProviderFactory` extension methods for file, skill, and
  music surfaces.
- Default factory methods adapt legacy `Arc<dyn LlmClient>` clients through registry-owned
  `ClientBacked*Capability` adapters.
- The handle no longer owns direct `compat_language_client_with_ctx(...)` calls or
  `as_*_capability()` downcasts for these extension implementations.

Reason:

- File, skill, and music surfaces are still extension-style APIs in the registry language handle.
- Chat and video execution already use family model paths.
- Keeping the compatibility adapter behind the provider-factory seam gives providers a native
  override point later without changing handle code.

Deletion condition:

- File, skill, and music become first-class family handles or provider-owned native extension
  factory overrides cover the built-in providers that declare those surfaces.

### Facade `Siumai` Wrapper - Compatibility Surface

Status: keep, but do not treat as the target architecture.

Locations:

- `siumai-registry/src/provider/siumai/*.rs`
- `siumai-registry/src/provider/siumai.rs`

Reason:

- `Siumai` preserves historical method-style APIs over a boxed `LlmClient`.
- New code should prefer `registry::global().*_model(...)` plus family helper modules.

Deletion condition:

- Migration docs and public examples no longer recommend `Siumai` method dispatch.
- Provider-specific features are reachable through family handles or explicit extension handles.

### Provider Composite Clients - Compat-Only Isolated

Status: isolated as compatibility adapters.

Locations include:

- `siumai-registry/src/registry/factories/deepinfra.rs`
- `siumai-registry/src/registry/factories/fireworks.rs`
- `siumai-registry/src/registry/factories/togetherai.rs`

Reason:

- These provider factories compose shared OpenAI-compatible text clients with provider-owned image,
  rerank, audio, or video clients.
- Their `LlmClient` implementations expose multiple capability views as a compatibility bridge.

Current seam:

- The private composite wrappers are named `*CompatCompositeClient`.
- `compat_language_client_with_ctx(...)` is the only construction point for those wrappers.
- Stable family methods construct native family objects directly and are source-guarded against
  composite-client construction, `compat_*_client_with_ctx(...)` self-calls, and `LlmClient`
  capability downcasts.

Current guard:

- `siumai-registry::factory_architecture_boundary_test::hybrid_provider_composite_clients_are_compat_only_adapters`

Deletion condition:

- Factory methods construct each family through provider-owned native family objects directly.
- Composite `LlmClient` wrappers are no longer needed for stable family dispatch.
- Historical method-style `Siumai` and generic `LlmClient` compatibility paths are either removed
  or rewritten to compose family models without exposing capability downcasts.

### Trait Declarations, Proxies, And Tests - Keep

Status: keep.

Locations include:

- `siumai-core/src/client.rs`
- `siumai-core/src/custom_provider/mod.rs`
- `siumai-registry/src/provider/proxies.rs`
- registry and provider contract tests

Reason:

- Trait declarations define the compatibility contract.
- Proxies and tests verify legacy capability behavior while migration remains supported.

## Deprecated Facade Allows

### `siumai/src/lib.rs`

Status: audited.

Current kept uses:

- `prelude::unified` re-exports deprecated experimental helper spellings for AI SDK parity:
  `experimental_generate_image`, `experimental_generate_speech`, and `experimental_transcribe`.
- `prelude::unified` re-exports deprecated experimental result/type aliases for compatibility.
- The provider-builder smoke test uses `#[allow(deprecated)]` while `Siumai::builder()` remains a
  time-bounded compatibility convenience.

Rule:

- Do not add new deprecated names to `prelude::unified` unless a migration note explains why the
  alias must remain visible there.

## Deprecated Public Alias Categorization

Status: categorized.

This table tracks the public deprecated aliases found in the facade, registry, and core crates. The
category is the current architecture decision, not a promise to remove everything in the same
release.

| Surface | Category | Migration / deletion note |
| --- | --- | --- |
| `Siumai::builder()` | keep, time-bounded | Compatibility-only unified builder. New code should use `registry::global().language_model("provider:model")`, other family handles, or config-first provider clients. Planned removal remains no earlier than `0.12.0`. |
| `siumai::compat::{Siumai, SiumaiBuilder, builder::*}` | keep, time-bounded | Explicit migration import surface for method-style code. Do not re-export these names as the default stable prelude path. |
| `SiumaiBuilder::provider(...)` | removed | Use `.provider_id(...)` or provider-specific helper methods while migrating existing builder code. |
| `SiumaiBuilder::vision(...)`, `Siumai::vision_capability()`, `VisionCapability`, `VisionCapabilityProxy` | removed | Dedicated vision is not a stable family. Use multimodal chat messages for image understanding and image-generation family APIs for image creation. Removed by `docs/workstreams/fearless-vision-compat-removal/`. |
| `experimental_generate_image`, `experimental_generate_speech`, `experimental_transcribe`, `experimental_generate_video` | keep, move out of recommendations | Deprecated AI SDK import-spelling aliases. Prefer `generate_image`, `synthesize`, `transcribe`, and `generate`. Keep aliases only for migration/import parity. |
| `create_google_generative_ai()` | keep, move out of recommendations | Deprecated analogue of AI SDK `createGoogleGenerativeAI()`. Prefer `create_google()`. |
| `SpeechModelHandle::text_to_speech(...)` | removed | Use `SpeechModel::synthesize(...)` or the `siumai::speech::synthesize(...)` helper. Keep `AudioCapability::text_to_speech(...)` only for explicit legacy audio compatibility. |
| `execute_json_request_with_headers(...)` and HTTP JSON static-header helpers | removed | Use `execute_json_request` with `HttpExecutionConfig`; static headers should be provided by a `ProviderSpec`. |
| `siumai_core::utils::vertex` | removed | Use facade `siumai::experimental::auth::vertex` or provider-owned `siumai-provider-google-vertex::auth::vertex`; the intermediate `siumai_core::auth::vertex` target was later removed. |
| Provider extension deprecated option/metadata aliases | keep, time-bounded | Keep upstream/Rust spelling migration aliases inside provider-owned extension modules only. Do not add them to the stable prelude. |

Guardrails:

- `siumai-registry::factory_architecture_boundary_test::compatibility_audit_categorizes_public_deprecated_surfaces`
  verifies that the known deprecated public surfaces remain categorized in this audit.
- `siumai-registry::factory_architecture_boundary_test::public_docs_do_not_recommend_compatibility_surfaces_as_default`
  keeps current README/architecture/migration docs from recommending `Siumai::builder()` or
  `LlmClient` as the default construction/invocation path.

## Facade Bridge Export

Status: implementation moved; facade path remains as compatibility re-export.

Current locations:

- `siumai-bridge/src/lib.rs`
- `siumai-bridge/src/*`
- public path: `siumai::experimental::bridge::*`
- direct consumer path: `siumai_bridge::*`

Decision:

- Keep the public facade path for compatibility.
- Own the implementation in the dedicated `siumai-bridge` crate.
- Make `siumai-extras` runtime bridge code consume `siumai-bridge` directly.
- Do not move the implementation directly into `siumai-extras` while extras depends on the facade.
- Do not move gateway/protocol conversion logic into `siumai-core`.

Current guard:

- `siumai::facade_architecture_boundary_test::experimental_bridge_is_owned_by_bridge_crate_and_reexported_by_facade`
- `siumai-extras::bridge_architecture_boundary_test::runtime_bridge_code_imports_dedicated_bridge_crate`

Deletion condition:

- Public docs and examples migrate away from the facade bridge path where direct `siumai_bridge`
  imports are more honest.
- A future compatibility-removal window is documented for `siumai::experimental::bridge::*` if the
  facade path should be retired.
