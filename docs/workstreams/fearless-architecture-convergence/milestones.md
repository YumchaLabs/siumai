# Fearless Architecture Convergence - Milestones

Last updated: 2026-05-16

## FAC-M0 - Workstream Boundary Locked

Acceptance criteria:

- Design document exists.
- TODO document lists deletion candidates and sequencing.
- Milestones document defines acceptance criteria for each convergence phase.
- Existing V4 workstream remains intact; this workstream focuses on cleanup and convergence.

Status: done

## FAC-M1 - Low-Risk Redundancy Cleanup

Acceptance criteria:

- Stale comments that reference already-deleted types are removed.
- Dead or misleading compatibility notes are replaced with current migration language.
- No public API behavior changes.
- `cargo fmt --check` passes for touched crates.

Status: done

Notes:

- Stale `UnifiedLlmClient` removal comments were removed from `siumai-core/src/client.rs`; the
  historical changelog entry remains as release history.
- Stale provider-parameter module comments were updated so provider-specific options point to
  provider crates and `provider_ext`, not `siumai-core::params`.
- Redundant "already removed" comments were removed from parameter and OpenAI-compatible streaming
  internals.
- Redundant "already removed" comments were also removed from Ollama, Gemini, Anthropic, and
  OpenAI protocol/provider internals; current module docs now describe current responsibilities
  instead of old deleted types.
- Builder-facing module docs in `siumai-registry::provider` and the facade `Provider` docs now
  describe builder-style construction as compatibility/convenience, not the primary architecture.
- Touched crates passed focused `cargo fmt --check` slices after the cleanup.

## FAC-M2 - OpenAI-Compatible Protocol Ownership

Acceptance criteria:

- OpenAI-compatible adapter/config/provider-registry/streaming/transformer modules live in
  `siumai-protocol-openai`.
- `siumai-core` no longer owns OpenAI-compatible provider-specific protocol modules.
- Temporary compatibility re-exports are documented and isolated.
- Direct imports from `siumai_core::standards::openai::compat::*` are removed from non-test code.
- Protocol and OpenAI-compatible provider tests pass.

Status: done

Notes:

- Provider-options/metadata key helpers now live in
  `siumai-protocol-openai::standards::openai::compat::metadata`, and protocol/provider callers no
  longer import those helpers from `siumai-core`.
- DeepInfra base-URL normalization helpers now live in
  `siumai-protocol-openai::standards::openai::compat::base_url`, and the DeepInfra registry factory
  no longer imports those helpers from `siumai-core`.
- `cargo nextest run -p siumai-protocol-openai --features openai-standard --no-fail-fast` passed
  after the metadata/base-url helper move.
- `cargo nextest run -p siumai-registry --features deepinfra --no-fail-fast` passed after the
  DeepInfra base-url import move.
- `adapter`, `base_url`, `metadata`, `openai_config`, `provider_registry`, `streaming`,
  `transformers`, and their focused tests now live under
  `siumai-protocol-openai/src/standards/openai/compat`.
- `siumai-core/src/standards/openai/compat` no longer owns Rust source files and
  `siumai-core/src/standards/openai/mod.rs` no longer exposes `compat`.
- Runtime callers were moved to the protocol-owned path, including `siumai` experimental bridge,
  registry DeepInfra/OpenAI-compatible factories, OpenAI-compatible provider metadata helpers, and
  xAI request fixture alignment tests.
- `siumai-protocol-openai/tests/openai_compat_boundary_test.rs` locks the boundary by asserting
  that direct `siumai_core::standards::openai::compat` imports and core compat source ownership do
  not reappear.
- Verified commands for this slice:
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai-compatible --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-registry --features openai --no-fail-fast`
  - `cargo nextest run -p siumai --features openai --no-fail-fast`
  - `cargo nextest run -p siumai-core --features openai --no-fail-fast`

## FAC-M3 - Family-First ProviderFactory

Acceptance criteria:

- `ProviderFactory` primary methods return family model trait objects.
- Generic `LlmClient` factory methods are renamed or moved behind compatibility defaults.
- Registry model handles no longer use capability downcasts for stable families when a native
  family factory exists.
- No new provider feature depends only on `LlmClient`.

Status: done

Notes:

- The legacy generic-client family bridge implementations were removed after native family factory
  methods covered declared built-in provider surfaces. `registry/entry/compat_client.rs` no longer
  exists.
- `ProviderFactory` now has explicit `compat_*_client*` generic-client aliases. Registry fallback
  adapters and extension-only handle paths call those aliases, while older generic
  `*_model*_with_ctx` trait methods have been removed.
- The historical `SiumaiBuilder` wrapper now uses the explicit `compat_*_client_with_ctx` aliases
  when it must construct an `Arc<dyn LlmClient>`, and
  `default_client_builder_uses_explicit_compat_factory_methods` guards that boundary.
- Production provider factories now implement `compat_*_client_with_ctx` for generic-client
  construction paths. The older `*_model_with_ctx` names are no longer trait methods or production
  override points.
- Production provider factories now implement `compat_language_client(...)` instead of overriding
  the removed generic `language_model(...)` method. Generic-client construction now uses
  `compat_*_client*` aliases only.
- Bedrock and Cohere now provide native family factory overrides for their declared stable
  families, returning provider-owned typed clients instead of generic family compatibility bridges.
- Anthropic Vertex now exposes a typed registry builder and native text-family factory override,
  so chat-family registry handles no longer depend on a generic family compatibility bridge for
  that provider.
- OpenAI and Azure now expose native completion-family factory overrides, so their declared
  completion family handles no longer depend on a family compatibility bridge.
- Gemini and Google Vertex now expose native video-family factory overrides, so their declared
  video family handles no longer depend on a family compatibility bridge.
- MiniMaxi and xAI now expose native video-family factory overrides for their declared video
  capability, replacing generic family compatibility bridges on those registry handles.
- `provider_factories_use_explicit_compat_for_generic_self_calls` now guards production provider
  factories against reintroducing old generic self-call names or old `async fn *_model_with_ctx`
  override names.
- `production_factories_do_not_override_legacy_generic_language_method` now guards production
  provider factories against reintroducing direct `async fn language_model(...)` overrides.
- `production_factories_with_declared_family_surfaces_use_native_family_overrides` now guards
  production provider factories against relying on generic family compatibility bridges for
  declared stable family surfaces.
- `compatibility_audit_does_not_keep_removed_providerfactory_methods_alive` now guards the
  convergence docs against claiming removed generic factory methods are merely deprecated wrappers.
- The no-builtins custom `ProviderFactory` example now implements the native
  `language_model_text_with_ctx` family method and keeps generic-client construction behind
  `compat_language_client`. `docs/architecture/registry-without-builtins.md` now teaches that
  shape.
- `siumai-registry::factory_architecture_boundary_test` prevents the no-builtins custom factory
  example and architecture doc from drifting back to `LlmClient`-first guidance.
- `ProviderFactory` now lists stable family-returning methods before their generic `LlmClient`
  compatibility entry points, and `factory_architecture_boundary_test` locks that ordering for all
  stable family/generic method pairs.
- Registry architecture boundary tests now include a no-network source guard,
  `registry_family_handles_keep_llm_client_downcasts_isolated`, so stable family handles cannot
  reintroduce `LlmClient` capability downcasts on primary family execution paths.
- `registry_client_backed_family_model_adapters_are_removed` guards against reintroducing the
  removed `registry/entry/compat_client.rs` module or `ClientBacked*Model` family adapters.
- Verified commands for this slice:
  - `cargo fmt -p siumai-registry --check`
  - `cargo check -p siumai-registry --features openai --no-default-features`
  - `cargo nextest run -p siumai-registry --features openai --no-fail-fast`
  - `cargo check -p siumai-registry --features "cohere,bedrock" --no-default-features`
  - `cargo nextest run -p siumai-registry --features "cohere,bedrock" --no-default-features --no-fail-fast`
  - `cargo check -p siumai-registry --features google-vertex --no-default-features`
  - `cargo nextest run -p siumai-registry --features google-vertex --no-default-features --no-fail-fast`
  - `cargo check -p siumai-registry --features "xai,minimaxi" --no-default-features`
  - `cargo nextest run -p siumai-registry --features "xai,minimaxi" --no-default-features --no-fail-fast`
  - `cargo check -p siumai-registry --all-features`
  - `cargo check -p siumai-registry --example no_builtins_custom_factory --no-default-features`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`
  - `cargo check -p siumai-registry --no-default-features`
  - `cargo nextest run -p siumai-registry provider_factories_use_explicit_compat_for_generic_self_calls production_factories_with_declared_family_surfaces_use_native_family_overrides --all-features --no-fail-fast`

## FAC-M4 - Registry Responsibility Split

Acceptance criteria:

- `registry/entry.rs` is split into focused modules.
- Build-context resolution is isolated from handle execution.
- Cache entries and cache keys are isolated from provider factory contracts.
- Compatibility `LlmClient` adapter code is visibly separated from family model handles.
- Registry tests remain no-network and family-scoped.

Status: done

Notes:

- `registry/entry.rs` no longer owns `ClientBacked*Model` family compatibility adapters; the
  transitional `registry/entry/compat_client.rs` module has been removed.
- Registry build-context and provider override merging now live in
  `registry/entry/build_context.rs`; `entry.rs` re-exports `BuildContext` and
  `ProviderBuildOverrides` to preserve public paths.
- Registry cache-entry ownership now lives in `registry/entry/cache.rs`; duplicate per-family TTL
  entry implementations were collapsed into one generic timed family-model cache entry.
- The `ProviderFactory` contract now lives in `registry/entry/factory.rs`; `entry.rs` re-exports
  it so existing `siumai_registry::registry::entry::ProviderFactory` imports remain stable.
- Focused handle modules now live under `registry/entry/handles`: audio, completion, embedding,
  image, language, reranking, and video handles were moved out of `entry.rs` while preserving the
  existing `entry::*ModelHandle` public imports.
- Shared video handle defaults now live in `registry/entry/handles/video_support.rs`, so both the
  language-handle compatibility video path and the video-family handle use one audited helper.
- Shared registry test fixtures now live in `registry/entry/test_support.rs`.
- Build-context propagation tests now live in `registry/entry/build_context_tests.rs`.
- Cache behavior tests now live in `registry/entry/cache_tests.rs`; the shared
  `TEST_BUILD_COUNT` serialization lock moved into `test_support.rs` so split test modules do not
  race each other.
- Alias normalization tests now live in `registry/entry/alias_tests.rs`, and registry interceptor
  propagation tests now live in `registry/entry/interceptor_tests.rs`.
- Language-model handle and factory tests now live in `registry/entry/language_tests.rs`.
- Reranking registry-handle tests now live in `registry/entry/rerank_tests.rs`.
- `registry/entry/tests.rs` now only keeps the builtins/default-registry smoke check, while the
  remaining family and handle tests live in dedicated module files.
- `registry/entry.rs` is now focused on registry options, factory lookup, handle construction, and
  cache wiring instead of owning handle execution bodies or shared test fixture implementations.
- `cargo nextest run -p siumai-registry -p siumai --no-fail-fast` passed after the registry and
  facade splits.
- `cargo nextest run -p siumai-registry --features openai --no-fail-fast` passed after the
  build-context/cache/factory/audio-handle/completion-handle/embedding-handle/image-handle/
  rerank-handle/video-handle split.
- `cargo nextest run -p siumai-registry --features openai --no-fail-fast` passed again after the
  language-handle and shared-video-helper split.
- `cargo nextest run -p siumai-registry --features openai --no-fail-fast` passed with 192 tests
  after moving shared test fixtures and adding the factory compatibility source guards.
- `cargo nextest run -p siumai-registry registry::entry::cache_tests --no-default-features --no-fail-fast`
  passed after moving LRU/TTL cache behavior tests out of root `entry/tests.rs`.
- `cargo nextest run -p siumai-registry registry::entry::build_context_tests --no-default-features --no-fail-fast`
  passed after moving build-context propagation tests out of root `entry/tests.rs`.
- `cargo nextest run -p siumai-registry registry::entry::build_context_tests::build_context_resolves_google_token_provider_with_backward_compatibility --features google --no-default-features --no-fail-fast`
  passed for the Google token-provider compatibility branch.

## FAC-M5 - Facade Slimming

Acceptance criteria:

- `siumai/src/lib.rs` mostly declares modules and stable re-exports.
- Provider extension module bodies are moved into dedicated files.
- `experimental_bridge` ownership is decided and documented.
- Compatibility preludes remain explicit and time-bounded.

Status: done

Notes:

- `siumai/src/lib.rs` no longer contains provider extension module bodies. The crate root now
  declares `pub mod provider_ext;`, keeps `siumai::providers` as a thin alias, and provider-specific
  extension surfaces live under `siumai/src/provider_ext/*.rs`.
- The Gemini model-id catalog now lives in `siumai/src/provider_ext/gemini/models.rs`; the public
  `siumai::provider_ext::gemini::{models, chat, embedding, image, model_sets, video}` paths remain
  stable while `provider_ext/gemini.rs` stays focused on provider-owned re-export, option, metadata,
  and resource glue.
- `experimental_bridge` implementation now lives in the dedicated `siumai-bridge` crate. The
  `siumai` facade keeps `siumai::experimental::bridge::*` as a compatibility re-export, and
  `siumai-extras` runtime bridge code imports from `siumai_bridge` directly.
- `facade_architecture_boundary_test` guards both facade slimming decisions: provider extension
  bodies must stay outside `siumai/src/lib.rs`, and the bridge implementation must stay outside the
  facade crate. It also prevents the Gemini model-id catalog from returning to the provider
  re-export glue file.
- `siumai-extras::bridge_architecture_boundary_test` guards the extras dependency direction so
  runtime bridge code does not import through the facade.
- `cargo nextest run -p siumai-registry -p siumai --no-fail-fast` passed after the registry and
  facade splits.
- Verified commands for the bridge crate split:
  - `cargo check -p siumai-bridge --no-default-features`
  - `cargo check -p siumai-bridge --features openai --no-default-features`
  - `cargo check -p siumai-bridge --features anthropic --no-default-features`
  - `cargo check -p siumai-bridge --features google --no-default-features`
  - `cargo check -p siumai-bridge --features "openai,anthropic,google" --no-default-features`
  - `cargo check -p siumai --features openai --no-default-features`
  - `cargo check -p siumai-extras --no-default-features --features "server,openai,anthropic,google"`
  - `cargo nextest run -p siumai-bridge --no-default-features --features "openai,anthropic,google" --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai-extras --test bridge_architecture_boundary_test --no-default-features --features openai --no-fail-fast`
- Verified commands for the Gemini provider-extension catalog split:
  - `cargo fmt -p siumai --check`
  - `cargo check -p siumai --features google --no-default-features`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-default-features --features google --no-fail-fast`

## FAC-M6 - Compatibility Surface Audit

Acceptance criteria:

- Every deprecated public alias is categorized as keep, move, or remove.
- Removals have migration notes.
- Compatibility-only builder paths compile down to canonical config-first construction.
- Public docs no longer recommend compatibility surfaces except for migration.

Status: done

Notes:

- `siumai/src/lib.rs` deprecated-allow blocks were audited. The remaining uses are explicit
  compatibility surfaces: AI SDK experimental import-spelling aliases in `prelude::unified` and the
  provider-builder smoke test while `Siumai::builder()` remains time-bounded compatibility.
- `compatibility-audit.md` categorizes `as_*_capability()` call sites into stable family paths,
  isolated registry adapters, extension-only language-handle paths, facade compatibility wrappers,
  provider composite clients, and test/proxy declarations.
- `compatibility-audit.md` now also categorizes known deprecated public aliases into keep, move, or
  remove decisions, with migration notes for each removal/move target.
- `docs/migration/migration-0.11.0-beta.5.md` no longer presents `Siumai::builder()` or
  `LlmClient` as current recommended paths; historical snippets are labeled as compatibility-only
  migration shapes and current guidance points at registry/config-first construction.
- `siumai-registry::provider::build::default_client_builder_uses_explicit_compat_factory_methods`
  guards the compatibility-only builder path so it delegates through explicit
  `compat_*_client_with_ctx` factory methods instead of old generic factory names.
- `siumai-registry::factory_architecture_boundary_test` now verifies both that deprecated public
  surfaces remain categorized in the compatibility audit and that current README/architecture/
  migration docs do not recommend `Siumai::builder()` or `LlmClient` as default guidance.
- Verified commands for this slice:
  - `cargo fmt -p siumai-registry --check`
  - `cargo check -p siumai-registry --no-default-features`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`

## Completion Audit - 2026-05-13

Objective deliverables and evidence:

- Workstream planning exists and is linked:
  - `docs/workstreams/fearless-architecture-convergence/{design.md,todo.md,milestones.md,compatibility-audit.md}`
  - `docs/README.md` links to the active workstream.
  - `factory_architecture_boundary_test` guards the docs index link.
- OpenAI-compatible protocol ownership moved out of core:
  - `siumai-protocol-openai/src/standards/openai/compat/*` owns adapter/config/registry/
    streaming/transformer modules.
  - `siumai-core/src/standards/openai/compat` is removed.
  - `openai_compat_boundary_test` guards against core ownership/import regression.
- `ProviderFactory` is family-first:
  - stable family methods return family model trait objects and appear before legacy generic
    `LlmClient` entry points.
  - old generic `*_model*` trait methods are removed.
  - production factories implement `compat_*_client*` methods instead of overriding removed
    generic `language_model(...)`.
  - no-builtins custom factory docs/example use `language_model_text_with_ctx` first and
    `compat_language_client` for generic-client compatibility.
- Registry responsibilities are split:
  - `registry/entry.rs` delegates factory contract, build context, cache, compatibility adapters,
    family handles, and tests to focused modules under `registry/entry/*`.
  - public paths remain stable through re-exports.
- Facade is slim:
  - `siumai/src/lib.rs` declares `provider_ext` externally and keeps `siumai::providers` as a thin
    alias.
  - provider extension bodies live under `siumai/src/provider_ext/*`.
  - Gemini model-id constants live in `siumai/src/provider_ext/gemini/models.rs`.
  - bridge implementation lives in `siumai-bridge`; `siumai` keeps only compatibility re-exports.
- Compatibility and redundant code are isolated:
  - `ClientBacked*Model` family adapters are removed; extension-only capability adapters live in
    `registry/entry/extension_adapters.rs`.
  - stable family handles are guarded against `LlmClient` capability downcasts.
  - deprecated public aliases are categorized in `compatibility-audit.md`.
  - current public docs no longer recommend `Siumai::builder()` or `LlmClient` as default guidance.
- Workstream status is closed:
  - FAC-M0 through FAC-M6 are `Status: done`.
  - `todo.md` has no remaining `[ ]`, `[~]`, or `[-]` work items.

Final verification commands:

- Per-package format check loop over every workspace package:
  - `cargo metadata --no-deps --format-version 1 | ConvertFrom-Json | Select-Object -ExpandProperty packages | ForEach-Object { $_.name } | Sort-Object -Unique`
  - `cargo fmt -p <package> --check` for each listed package
  - Note: `cargo fmt --all --check` hits Windows `os error 206` in this workspace, so the
    per-package loop is the equivalent final formatting gate.
- `git diff --check`
- `cargo check --workspace --all-features`
- `cargo check -p siumai-core --features openai --no-default-features`
- `cargo check -p siumai-registry --no-default-features`
- `cargo check -p siumai-registry --all-features`
- `cargo check -p siumai-registry --example no_builtins_custom_factory --no-default-features`
- `cargo nextest run -p siumai-protocol-openai --test openai_compat_boundary_test --features openai-standard --no-fail-fast`
- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-registry provider_factories_use_explicit_compat_for_generic_self_calls production_factories_with_declared_family_surfaces_use_native_family_overrides --all-features --no-fail-fast`
- `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-default-features --features google --no-fail-fast`
- `cargo nextest run -p siumai-extras --test bridge_architecture_boundary_test --no-default-features --features openai --no-fail-fast`
