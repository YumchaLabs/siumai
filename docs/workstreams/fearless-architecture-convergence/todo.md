# Fearless Architecture Convergence - TODO

Last updated: 2026-05-16

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Boundary and Audit

- [x] Create workstream design document.
- [x] Create milestone document.
- [x] Create TODO document.
- [x] Add a short README link from the architecture index if the repo grows one.
  - `factory_architecture_boundary_test` now locks the docs index link to the active workstream
- [x] Audit all direct non-test imports of `siumai_core::standards::openai::compat`.
  - direct imports were removed; protocol-owned callers now use
    `siumai_protocol_openai::standards::openai::compat`
- [x] Audit all `as_*_capability()` call sites and categorize them:
  - stable family path to remove
  - extension-only path to keep for now
  - compatibility path to isolate
  - audit recorded in `compatibility-audit.md`
- [x] Audit all `#[allow(deprecated)]` blocks in `siumai/src/lib.rs`.
  - keep `prelude::unified` deprecated experimental helper/type aliases for AI SDK import-spelling
    compatibility until a documented removal version exists
  - keep the local provider-builder smoke test allow while `Siumai::builder()` remains a
    compatibility convenience
  - do not add new deprecated exports to the stable prelude without a migration note

## Track B - Low-Risk Redundancy Cleanup

- [x] Remove stale comments in `siumai-core/src/client.rs` that reference already-deleted
  `UnifiedLlmClient`.
- [x] Replace stale provider-parameter comments that still imply provider-specific params belong in
  `siumai-core::params`.
- [x] Remove redundant removal comments for already-deleted parameter/streaming compatibility code.
- [x] Remove redundant deletion-marker comments from Ollama/Gemini/Anthropic/OpenAI protocol and
  provider internals.
- [x] Replace outdated comments that still describe builders as architectural entry points.
  - `siumai-registry::provider` and `siumai::Provider` docs now describe builder-style paths as
    compatibility/convenience surfaces, not the target architecture
- [x] Remove redundant compatibility notes that are duplicated in README, ADRs, and module docs.
  - trimmed duplicate compatibility wording in `docs/architecture/registry-without-builtins.md`
    and `docs/architecture/provider-extensions.md`
  - removed duplicate legacy crate-name mentions from `docs/adr/0002-provider-crates-by-provider.md`
  - removed a repeated legacy alias sentence from the OpenAI-like reuse section in the same ADR
  - updated `factory_architecture_boundary_test` to match the revised compatibility wording
  - current remaining compatibility/legacy mentions in architecture and ADR docs carry migration
    guidance, historical rationale, or explicit demotion policy rather than duplicated default
    recommendation text
- [x] Keep compatibility docs where they carry migration information.
  - `docs/migration/migration-0.11.0-beta.5.md` now keeps historical builder snippets explicitly
    labeled as compatibility/migration-only and points current guidance to registry/config-first
    construction.

## Track C - OpenAI-Compatible Protocol Move

- [x] Move `base_url` helpers from core compat into `siumai-protocol-openai`.
- [x] Move metadata-key helpers into `siumai-protocol-openai`.
- [x] Move `adapter` into `siumai-protocol-openai`.
- [x] Move `openai_config` into `siumai-protocol-openai`.
- [x] Move `provider_registry` into `siumai-protocol-openai`.
- [x] Move `streaming` and its tests into `siumai-protocol-openai`.
- [x] Move `transformers` and its tests into `siumai-protocol-openai`.
- [x] Remove core compat re-exports instead of extending the transitional path.
- [x] Update imports in:
  - `siumai-registry`
  - `siumai`
  - `siumai-provider-openai-compatible`
  - OpenAI-compatible provider tests

## Track D - Family-First Factory Cleanup

- [x] Make family-returning `ProviderFactory` methods the required construction contract.
  - `ProviderFactory` trait docs now describe generic `LlmClient` construction as legacy
    compatibility only; explicit `compat_*_client*` names are the only generic-client factory
    methods
  - `factory_architecture_boundary_test` now locks the trait doc wording alongside the family-first
    custom factory example and README/docs index links
  - `language_model_text_with_ctx` now appears before the explicit `compat_language_client` entry
    point in `ProviderFactory`, making the primary family path the first thing readers see
  - `factory_architecture_boundary_test` now also locks that family-first ordering in the trait
    source text itself
  - `cargo check -p siumai-registry --no-default-features` passed after the trait doc/order change
  - `cargo check -p siumai-registry --features builtins --no-default-features` also passed for the
    builtins-enabled default registry path after the stable family ordering pass
  - `completion_model_family_with_ctx` now appears before the generic `completion_model` entry
    point as well, and the boundary test locks that second family ordering too
  - `embedding_model_family_with_ctx` and `image_model_family_with_ctx` now also appear before their
    generic client entry points, with boundary-test coverage for both orderings
  - speech, transcription, video, and reranking family methods now also appear before their generic
    client entry points; the boundary test locks all stable family/generic ordering pairs
- [x] Rename generic-client methods to make their compatibility role explicit.
  - `ProviderFactory` now exposes `compat_*_client*` aliases for generic `LlmClient`
    construction
  - registry-internal generic-client fallbacks now call the explicit `compat_*` methods
  - the historical `SiumaiBuilder` wrapper now calls `compat_*_client_with_ctx` explicitly when it
    must build an `Arc<dyn LlmClient>`
  - production provider factories now implement `compat_*_client_with_ctx` for generic-client
    construction paths; old `*_model_with_ctx` names have been removed
  - production provider factories no longer self-call old `*_model_with_ctx` names from their
    generic-client wrapper/fallback paths
  - production provider factories now implement `compat_language_client(...)` instead of
    overriding removed `language_model(...)`
  - old generic-client `*_model*` trait methods have been removed; use `compat_*_client*` aliases
    or native family factory methods
  - Bedrock and Cohere now return provider-owned typed clients from native family factory methods
    instead of relying on the default `ClientBacked*Model` bridge for declared stable families
  - Anthropic Vertex now has a typed client builder and native text-family factory path instead of
    relying on the default `ClientBackedLanguageModel` bridge
  - OpenAI and Azure now expose native completion-family factory paths instead of relying on the
    default `ClientBackedCompletionModel` bridge for declared completion capability
  - Gemini and Google Vertex now expose native video-family factory paths instead of relying on the
    default `ClientBackedVideoModel` bridge for their declared video surface
  - MiniMaxi and xAI now expose native video-family factory paths for their declared video
    capability instead of relying on the default `ClientBackedVideoModel` bridge
  - old `*_model*_with_ctx` generic-client methods remain for source compatibility until the
    public trait contract can be tightened
- [x] Move `ClientBacked*Model` adapters into a dedicated compatibility module.
- [x] Remove family handle execution paths that downcast through `LlmClient` when a family model
  path exists.
  - stable registry handles now use family factory methods directly
  - remaining language-handle downcasts are extension-only file/skills/music paths
- [x] Add compile/no-network coverage proving family factories are the default path.
  - registry handle tests already panic if legacy generic-client builders are used where native
    family factories exist
  - `registry_family_handles_keep_llm_client_downcasts_isolated` prevents stable family handles
    from adding new `LlmClient` capability downcasts; it now lives in the integration architecture
    boundary test instead of root `entry/tests.rs`
  - `default_client_builder_uses_explicit_compat_factory_methods` prevents the historical
    `SiumaiBuilder` wrapper from calling old generic factory method names directly
  - `provider_factories_use_explicit_compat_for_generic_self_calls` prevents production provider
    factories from reintroducing old generic `self.*_model_with_ctx(...)` self-calls or
    `async fn *_model_with_ctx(...)` generic-client overrides
  - `production_factories_do_not_override_legacy_generic_language_method` prevents production
    provider factories from reintroducing direct `async fn language_model(...)` overrides
  - `production_factories_with_declared_family_surfaces_use_native_family_overrides` prevents
    production provider factories from relying on default `ClientBacked*Model` bridges for declared
    stable family surfaces
  - `compatibility_audit_does_not_keep_removed_providerfactory_methods_alive` prevents the
    architecture convergence docs from describing removed generic factory methods as deprecated
    wrappers
- [x] Move custom `ProviderFactory` guidance and runnable no-builtins example to family-model-first
  construction.
  - `docs/architecture/registry-without-builtins.md` now describes `*_family_with_ctx` methods as
    the primary custom factory contract
  - `siumai-registry/examples/no_builtins_custom_factory.rs` implements
    `language_model_text_with_ctx` and keeps generic-client construction behind
    `compat_language_client`
  - `siumai-registry/tests/factory_architecture_boundary_test.rs` prevents the no-builtins example
    and architecture doc from drifting back to `LlmClient`-first guidance

## Track E - Registry Split

- [x] Split `registry/entry.rs` into modules:
  - `factory.rs` (done; public path preserved through re-export)
  - `build_context.rs` (done; public paths preserved through re-exports)
  - `handles/language.rs` (done)
  - `handles/completion.rs` (done)
  - `handles/embedding.rs` (done)
  - `handles/image.rs` (done)
  - `handles/rerank.rs` (done)
  - `handles/video.rs` (done)
  - `handles/video_support.rs` (done; shared video handle defaults)
  - `handles/audio.rs` (done)
  - `cache.rs` (done; duplicate TTL cache entries collapsed)
  - `compat_client.rs` (done)
- [x] Move `ClientBacked*Model` compatibility adapters to `registry/entry/compat_client.rs`.
- [x] Move `BuildContext` and `ProviderBuildOverrides` to `registry/entry/build_context.rs`.
- [x] Move family model cache entries to `registry/entry/cache.rs`.
- [x] Move `ProviderFactory` to `registry/entry/factory.rs`.
- [x] Move shared registry test fixtures to `registry/entry/test_support.rs`.
- [x] Move audio, completion, embedding, image, language, reranking, and video handles into
  `registry/entry/handles`.
- [x] Move tests into matching module files.
  - family handle tests are split by family; root registry tests now only keep the builtins/default-registry smoke check in `entry/tests.rs`
  - build-context propagation tests now live in `registry/entry/build_context_tests.rs`
  - cache behavior tests now live in `registry/entry/cache_tests.rs`
  - `TEST_BUILD_COUNT` serialization now lives in `registry/entry/test_support.rs` so split test
    modules share one lock
  - alias normalization tests now live in `registry/entry/alias_tests.rs`
  - registry interceptor propagation tests now live in `registry/entry/interceptor_tests.rs`
  - language-model handle and factory tests now live in `registry/entry/language_tests.rs`
  - reranking registry-handle tests now live in `registry/entry/rerank_tests.rs`
- [x] Keep public paths stable through `pub use`.

## Track F - Facade Split

- [x] Move `provider_ext::openai` body out of `siumai/src/lib.rs`.
- [x] Move OpenAI-compatible vendor extension modules out of `siumai/src/lib.rs`.
- [x] Move Google/Gemini/Vertex extension modules out of `siumai/src/lib.rs`.
- [x] Move Anthropic/Anthropic-Vertex extension modules out of `siumai/src/lib.rs`.
- [x] Move remaining provider extension modules out of `siumai/src/lib.rs`.
- [x] Keep `siumai::providers` alias as a thin public alias only.
- [x] Continue splitting provider extension files only when a provider module grows beyond
  provider-owned re-export glue.
  - Gemini model-id constants now live in `siumai/src/provider_ext/gemini/models.rs`; the public
    `siumai::provider_ext::gemini::{models, chat, embedding, image, model_sets, video}` paths stay
    stable.
  - Current audit stops here: the remaining provider extension files are still provider-owned
    re-export/options/resource glue and should only split further when cohesive implementation
    bodies appear.
- [x] Decide whether `experimental_bridge` belongs in `siumai-extras` or a dedicated bridge crate.
  - decision: target a dedicated bridge crate; `siumai-extras` is a consumer and currently depends
    on the `siumai` facade, so moving bridge ownership directly into extras would create the wrong
    dependency direction
  - `siumai::experimental::bridge::*` remains as a compatibility facade re-export
- [x] Move `experimental_bridge` implementation into the dedicated bridge crate and keep the facade
  path as a compatibility re-export.
- [x] Make `siumai-extras` runtime bridge code consume `siumai-bridge` directly instead of importing
  bridge APIs through the `siumai` facade.

## Track G - Validation

- [x] Add focused import-boundary tests or source checks for no new direct
  `siumai_core::standards::openai::compat` imports outside compatibility re-exports.
- [x] Run `cargo nextest run -p siumai-core` after core cleanup slices.
  - latest slice: `cargo nextest run -p siumai-core --features openai --no-fail-fast`
- [x] Run `cargo nextest run -p siumai-protocol-openai` after protocol moves.
  - latest slice: `cargo nextest run -p siumai-protocol-openai --features openai-standard --no-fail-fast`
- [x] Run `cargo nextest run -p siumai-registry` after registry splits.
  - latest slice: `cargo nextest run -p siumai-registry --features openai --no-fail-fast`
    (192 tests)
- [x] Run all-feature registry type checks after cross-provider factory edits.
  - latest slice: `cargo check -p siumai-registry --all-features`
- [x] Run custom factory no-builtins checks after family-first guidance updates.
  - latest slice:
    `cargo check -p siumai-registry --example no_builtins_custom_factory --no-default-features`
  - latest slice:
    `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --no-fail-fast`
- [x] Run split cache test checks after moving cache behavior tests out of root `entry/tests.rs`.
  - latest slice:
    `cargo nextest run -p siumai-registry registry::entry::cache_tests --no-default-features --no-fail-fast`
- [x] Run split build-context test checks after moving build-context propagation tests out of root
  `entry/tests.rs`.
  - latest slice:
    `cargo nextest run -p siumai-registry registry::entry::build_context_tests --no-default-features --no-fail-fast`
  - latest slice:
    `cargo nextest run -p siumai-registry registry::entry::build_context_tests::build_context_resolves_google_token_provider_with_backward_compatibility --features google --no-default-features --no-fail-fast`
- [x] Run `cargo nextest run -p siumai` after facade splits.
- [x] Add source guards for facade slimming boundaries.
  - `facade_architecture_boundary_test` prevents provider extension bodies from returning to
    `siumai/src/lib.rs`
  - `facade_architecture_boundary_test` verifies that the bridge implementation is owned by
    `siumai-bridge` and that the facade only re-exports it
  - `facade_architecture_boundary_test` verifies that Gemini model-id constants stay in
    `siumai/src/provider_ext/gemini/models.rs`
- [x] Add source guard for `siumai-extras` bridge dependency direction.
  - `bridge_architecture_boundary_test` prevents runtime extras code from importing bridge APIs
    through `siumai::experimental::bridge`
- [x] Add source guards for compatibility-surface audit drift.
  - `factory_architecture_boundary_test` verifies that the known deprecated public surfaces remain
    categorized in `compatibility-audit.md`
  - `factory_architecture_boundary_test` prevents current README/architecture/migration docs from
    recommending `Siumai::builder()` or `LlmClient` as default public guidance
- [x] Run `cargo fmt --check` before each commit candidate.
