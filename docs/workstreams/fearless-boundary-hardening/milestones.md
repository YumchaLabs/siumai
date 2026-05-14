# Fearless Boundary Hardening - Milestones

Last updated: 2026-05-14

## FBH-M0 - Workstream Policy Locked

Acceptance criteria:

- Design, TODO, and milestone documents exist.
- The docs state that unnecessary compatibility and redundant code may be removed during the
  fearless-refactor phase.
- Removal criteria are explicit: canonical replacement, coverage, and migration documentation.
- The docs index links to this workstream.

Status: done

## FBH-M1 - Core Protocol Residue Removed

Acceptance criteria:

- `siumai-core` no longer owns provider-specific OpenAI wire-format conversion helpers.
- Remaining `siumai-core::standards` modules are either provider-agnostic or explicitly justified.
- Protocol-owned helpers live in `siumai-protocol-*` crates.
- Boundary tests prevent protocol/provider-specific code from returning to `siumai-core`.

Status: done

Notes:

- OpenAI message conversion, wire type, finish reason, and usage helpers now live in
  `siumai-protocol-openai::standards::openai`.
- `siumai-core::standards::openai` was removed; `siumai-core::standards` now keeps only the
  provider-agnostic tool-name mapping helper.
- `siumai-protocol-openai::openai_compat_boundary_test` prevents direct
  `siumai_core::standards::openai` imports and verifies that `siumai-core/src/standards/openai`
  does not own Rust protocol files.
- Verified commands for this slice:
  - `cargo check -p siumai-core --features openai --no-default-features`
  - `cargo check -p siumai-protocol-openai --features openai-standard --no-default-features`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard --no-fail-fast`

## FBH-M2 - Provider And Protocol Re-Exports Narrowed

Acceptance criteria:

- Provider crates no longer broadly re-export `siumai_core` modules unless the export is a documented
  public compatibility surface.
- Protocol crates import required core modules explicitly instead of acting as wide `siumai-core`
  mirrors.
- Any kept compatibility re-export has a migration note or removal condition.
- Source guards cover the narrowed boundary.

Status: done

Notes:

- Protocol crates now keep `siumai_core` module aliases as crate-private implementation details
  instead of public API mirrors.
- Provider crates now keep `siumai_core` module aliases and `builder` helpers crate-private.
- Public tests and registry code that used provider/protocol crates as `siumai-core` aliases now
  import shared core types, traits, and `BuilderBase` from `siumai_core`.
- The legacy `siumai-provider-openai-compatible` crate remains a protocol compatibility alias, but
  its provider implementation owns only private core aliases.
- `siumai-protocol-openai::openai_compat_boundary_test` now rejects broad public core mirrors in
  protocol/provider crate roots.
- Verified commands for this slice:
  - `cargo check -p siumai-protocol-openai --features openai-standard --no-default-features`
  - `cargo check -p siumai-protocol-anthropic --features anthropic-standard --no-default-features`
  - `cargo check -p siumai-protocol-gemini --features google --no-default-features`
  - `cargo check -p siumai-provider-openai-compatible --features openai-standard --no-default-features`
  - `cargo check -p siumai-provider-openai --features openai --no-default-features`
  - `cargo check -p siumai-provider-anthropic --features anthropic --no-default-features`
  - `cargo check -p siumai-provider-gemini --features google --no-default-features`
  - `cargo check -p siumai-provider-azure --features azure --no-default-features`
  - `cargo check -p siumai-provider-groq --features groq --no-default-features`
  - `cargo check -p siumai-provider-xai --features xai --no-default-features`
  - `cargo check -p siumai-provider-deepseek --features deepseek --no-default-features`
  - `cargo check -p siumai-provider-minimaxi --features minimaxi --no-default-features`
  - `cargo check -p siumai-provider-ollama --features ollama --no-default-features`
  - `cargo check -p siumai-provider-cohere --features cohere --no-default-features`
  - `cargo check -p siumai-provider-togetherai --features togetherai --no-default-features`
  - `cargo check -p siumai-provider-amazon-bedrock --features bedrock --no-default-features`
  - `cargo check -p siumai-provider-google-vertex --features google-vertex --no-default-features`
  - `cargo check -p siumai-registry --features openai,anthropic,google --no-default-features`
  - `cargo check -p siumai --features openai,anthropic,google --no-default-features`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-anthropic --test anthropic_streaming_feature_surface_test --features anthropic-standard --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-gemini --test gemini_streaming_feature_surface_test --features google --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --test openai_completion_stream_parse_error_test --features openai --no-fail-fast`
  - `cargo nextest run -p siumai-provider-groq --test config_first_http_convenience_test --features groq --no-fail-fast`
  - `cargo nextest run -p siumai-provider-deepseek --test config_first_http_convenience_test --features deepseek --no-fail-fast`
  - `cargo nextest run -p siumai-provider-xai --test config_first_http_convenience_test --features xai --no-fail-fast`

## FBH-M3 - Registry Compatibility Adapters Reduced

Acceptance criteria:

- Stable family registry handles use native family factory paths.
- Stable family defaults reject missing native family factory paths instead of silently adapting
  generic compatibility clients.
- Extension-only gaps remain behind explicit `compat_*_client*` methods.
- Deprecated generic factory wrappers have removal notes and are removed when the compatibility
  window allows it.
- Registry boundary tests prove stable family paths do not use `LlmClient` downcasts.

Status: done

Notes:

- Stable registry handle primary execution now goes through native family factory paths:
  language, completion, embedding, image generation, speech synthesis, transcription,
  reranking, and video.
- `EmbeddingModelHandle::embed_with_config(...)` no longer probes
  `compat_embedding_client_with_ctx(...)` before the native embedding-family path.
- The default `ProviderFactory` family methods now return `UnsupportedOperation` when a provider
  does not expose a native family model path.
- The old default bridge module was removed. Providers that implement stable families must expose
  native family factory methods; compatibility generic clients remain only as explicit
  `compat_*_client*` entry points for historical and extension-only surfaces.
- `ProviderFactory` no longer exposes deprecated generic `*_model(...)` or `*_model_with_ctx(...)`
  wrapper methods. Remaining generic client construction must use explicit `compat_*_client*`
  methods.
- Extension-only compat bridges remain isolated for capabilities that do not yet have first-class
  family construction paths: language file/skill/music, image edit/variation, speech streaming and
  voice listing, and transcription streaming/translation/language listing.
- `siumai-registry::registry::entry::boundary_tests` now proves stable handle primary execution
  does not call `compat_*_client_with_ctx(...)`.
- Verified commands for this slice:
  - `cargo nextest run -p siumai-registry registry::entry::boundary_tests --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-registry registry::entry --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-registry registry::factories::contract_tests --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test middleware_override_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo check -p siumai-registry --features openai,anthropic,google --no-default-features`
  - `cargo check -p siumai-registry --tests --features openai,anthropic,google --no-default-features`
  - `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
  - `cargo fmt -p siumai-registry --check`

## FBH-M4 - Facade Surface Hardened

Acceptance criteria:

- `siumai/src/lib.rs` remains a declaration and re-export facade, not an implementation hub.
- Stable preludes exclude compatibility-only names unless they are intentionally documented.
- Deprecated aliases are removed when no longer protected by migration policy.
- Facade boundary tests cover provider-extension and bridge ownership.

Status: done

Notes:

- `siumai::prelude::unified` no longer re-exports compatibility construction aliases:
  root `Provider`, hidden `provider::Siumai`, or the deprecated experimental helper aliases.
- Explicit compatibility paths remain available under `siumai::compat` and
  `siumai::prelude::compat` for the documented migration window.
- Public import tests now use explicit family/root paths for deprecated experimental helper aliases.
- `facade_architecture_boundary_test` now guards the stable unified prelude against reintroducing
  compatibility-only construction aliases.
- Verified commands for this slice:
  - `cargo check -p siumai --tests --features openai,anthropic,google --no-default-features`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google --no-default-features --no-fail-fast`

## FBH-M5 - Documentation And Examples Converged

Acceptance criteria:

- README, architecture docs, and examples point to registry-first or config-first construction.
- Removed public compatibility paths have migration notes.
- Changelog or migration docs record intentional breaking removals.
- Docs do not describe deleted compatibility surfaces as current architecture.

Status: done

Notes:

- `README.md` already presents registry-first and config-first examples before the builder
  compatibility section, and keeps `Siumai::builder()` framed as temporary compatibility.
- `docs/architecture/public-surface.md` now explicitly documents that `prelude::unified` excludes
  compatibility construction aliases and that builder aliases live under `siumai::compat` /
  `prelude::compat` for migration-only use.
- `docs/architecture/public-surface.md` now points Vertex helpers to
  `siumai::experimental::auth::vertex`, not the removed `experimental::utils::vertex` path.
- The deprecated `siumai_core::utils::vertex` forwarding module was removed after its migration
  target had already been documented.
- Verified commands for this slice:
  - `cargo check -p siumai-core --features gcp --no-default-features`
  - `cargo check -p siumai-registry --tests --features google-vertex,gcp --no-default-features`
  - `cargo check -p siumai --tests --features google-vertex,gcp --no-default-features`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test google_vertex_provider_base_url_alignment_test --features google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --features google-vertex,gcp --no-default-features --no-fail-fast`

## FBH-M6 - Final Validation

Acceptance criteria:

- Focused tests pass for all affected crates.
- Boundary tests pass for core, registry, facade, bridge, and affected providers.
- Formatting and whitespace checks pass.
- `todo.md` has no remaining `[ ]` or `[~]` items, except explicitly deferred items with rationale.

Status: done

Notes:

- `todo.md` has no remaining incomplete workstream tasks. The only remaining `[ ]` / `[~]`
  strings are the status legend and this milestone's acceptance-criteria wording.
- `cargo fmt --all --check` was attempted first, but Windows returned `os error 206` (command line
  too long). Package-scoped formatting checks were used for all affected packages instead.
- Verified commands for this slice:
  - `cargo fmt -p siumai-core -p siumai-protocol-openai -p siumai-protocol-anthropic -p siumai-protocol-gemini -p siumai-registry -p siumai --check`
  - `cargo check -p siumai-core --features openai,gcp --no-default-features`
  - `cargo check -p siumai-protocol-openai --features openai-standard --no-default-features`
  - `cargo check -p siumai-registry --tests --features openai,anthropic,google,google-vertex,gcp --no-default-features`
  - `cargo check -p siumai --tests --features openai,anthropic,google,google-vertex,gcp --no-default-features`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard --no-fail-fast`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,anthropic,google,google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-registry registry::entry::boundary_tests --features openai,anthropic,google,google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google,google-vertex,gcp --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --features openai,anthropic,google,google-vertex,gcp --no-default-features --no-fail-fast`
  - `git diff --check`
