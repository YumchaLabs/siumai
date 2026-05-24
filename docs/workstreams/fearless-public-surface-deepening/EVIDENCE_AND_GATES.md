# Fearless Public Surface Deepening — Evidence And Gates

Status: Active
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
cargo nextest run -p siumai --test public_surface_imports_test public_surface_tooling_imports_compile --no-default-features --features openai --no-fail-fast
```

This proves the first executable slice preserves `siumai::tooling` public imports after replacing a
wildcard facade mirror with an explicit curated surface.

## Gate Set

### Facade Tooling Gate

```powershell
cargo check -p siumai --tests --no-default-features --features openai
cargo nextest run -p siumai --test tooling_runtime_public_surface_test public_surface_tooling_runtime_contract_compiles --no-default-features --features openai --no-fail-fast
cargo nextest run -p siumai --test public_surface_imports_test public_surface_tooling_imports_compile --no-default-features --features openai --no-fail-fast
```

Proves the first facade tightening slice compiles and preserves runtime-tool import behavior.

### Core Gate

```powershell
cargo check -p siumai-core --tests --no-default-features
```

Use this when touching core streaming or compatibility shims.

### Public Surface Gate

```powershell
cargo nextest run -p siumai --test public_surface_imports_test --no-default-features --features openai --no-fail-fast
```

Use narrower filters during iteration and record the exact filter.

### Facade Root Namespace Gate

```powershell
cargo nextest run -p siumai --test facade_architecture_boundary_test facade_root_splits_public_namespace_modules --no-default-features --features openai --no-fail-fast
```

Use this when splitting public namespace modules out of `siumai/src/lib.rs`.

### Broader Closeout Gate

Prefer a focused package matrix over full workspace commands in this Windows terminal when full
workspace formatting or smoke scripts hit documented path-length or WSL/bash issues.

### Review Gate

Before accepting a major task or lane closeout, run or perform a review focused on:

- public path preservation,
- source guard accuracy,
- compatibility shim retention/removal reasoning,
- and whether deferred work should become a child workstream.

## Evidence Anchors

- `docs/workstreams/fearless-public-surface-deepening/DESIGN.md`
- `docs/workstreams/fearless-public-surface-deepening/TODO.md`
- `docs/workstreams/fearless-public-surface-deepening/MILESTONES.md`
- `docs/architecture/public-surface.md`
- `docs/adr/0001-vercel-aligned-modular-split.md`
- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`

## Evidence Log

| Date | Task | Command / Evidence | Result | What it proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | FPSD-010 | Workstream docs opened; `git diff --check -- docs/workstreams/fearless-public-surface-deepening docs/workstreams/INDEX.md`. | Pass | Establishes the durable lane and first executable target. |
| 2026-05-25 | FPSD-020 | `cargo check -p siumai --tests --no-default-features --features openai`. | Pass | Proves the facade crate and tests compile after replacing the tooling wildcard mirror with explicit re-exports. |
| 2026-05-25 | FPSD-020 | `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_tooling_module_exports_an_explicit_runtime_surface stable_unified_prelude_does_not_mirror_tooling_runtime_module --no-default-features --features openai --no-fail-fast`. | Pass: 2 tests run, 2 passed. | Proves `siumai::tooling` stays explicit and `prelude::unified` does not mirror the full runtime tooling module. |
| 2026-05-25 | FPSD-020 | `cargo nextest run -p siumai --test tooling_runtime_public_surface_test public_surface_tooling_runtime_contract_compiles --no-default-features --features openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Proves the broader runtime tooling contract still compiles from the public facade. |
| 2026-05-25 | FPSD-020 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_tooling_imports_compile --no-default-features --features openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Proves documented `siumai::tooling` imports still work. |
| 2026-05-25 | FPSD-020 | `git diff --check`. | Pass; Git reported expected LF-to-CRLF working-copy warnings only. | Proves touched source/docs have no whitespace-error diff. |
| 2026-05-25 | FPSD-030 | Split `siumai::{hosted_tools,protocol,content,extensions}` from `siumai/src/lib.rs` into named source files. | Pass | Reduces facade root coupling while preserving existing public paths; `prelude` and `experimental` remain in `lib.rs` for follow-up slices. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_root_splits_public_namespace_modules legacy_content_part_has_explicit_compat_namespace directional_content_namespaces_are_visible_and_compat_is_explicit stable_unified_prelude_keeps_non_family_extension_types_scoped hosted_tools_facade_reexports_protocol_owned_constructors facade_root_and_experimental_exports_are_owner_backed_and_scoped --no-default-features --features openai --no-fail-fast`. | Pass: 6 tests run, 6 passed. | Proves the split root namespace files keep owner-backed exports and existing facade architecture guards still apply. |
| 2026-05-25 | FPSD-030 | `cargo check -p siumai --tests --no-default-features --features openai`. | Pass | Proves the facade crate and tests compile after the namespace split. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_directional_content_namespaces_compile public_surface_extensions_imports_compile public_surface_openai_provider_ext_compiles --no-default-features --features openai --no-fail-fast`. | Pass: 3 tests run, 3 passed. | Proves `siumai::content`, `siumai::extensions`, and OpenAI hosted-tool imports still compile through public paths. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_protocol_openai_compiles --no-default-features --features openai,protocol-openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Proves the split `siumai::protocol::openai` facade still compiles with the required provider feature enabled. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_protocol_openai_compiles --no-default-features --features protocol-openai --no-fail-fast`. | Expected fail: `build.rs` requires at least one provider feature. | Documents the pre-existing facade feature constraint; use `openai,protocol-openai` for this gate. |
| 2026-05-25 | FPSD-030 | Split `siumai::experimental` from `siumai/src/lib.rs` into `siumai/src/experimental.rs`. | In progress | Further reduces facade root coupling while preserving advanced `siumai::experimental::*` paths; `prelude` remains in `lib.rs` for a follow-up slice. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test facade_architecture_boundary_test experimental_bridge_is_owned_by_bridge_crate_and_reexported_by_facade stable_unified_prelude_does_not_mirror_core_streaming_internals stable_unified_prelude_does_not_export_middleware_internals facade_root_and_experimental_exports_are_owner_backed_and_scoped facade_generic_client_paths_are_explicit_compatibility_exports facade_root_splits_public_namespace_modules --no-default-features --features openai --no-fail-fast`. | Pass: 6 tests run, 6 passed. | Proves `siumai::experimental` remains a named advanced facade path after moving its body out of the root. |
| 2026-05-25 | FPSD-030 | `cargo check -p siumai --tests --no-default-features --features openai`. | Pass | Proves the facade crate and tests compile after the `experimental` split. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_directional_content_namespaces_compile public_surface_extensions_imports_compile public_surface_openai_provider_ext_compiles public_surface_protocol_openai_compiles --no-default-features --features openai,protocol-openai --no-fail-fast`. | Pass: 4 tests run, 4 passed. | Rechecks touched facade paths plus OpenAI protocol/provider imports after the `experimental` split. |
| 2026-05-25 | FPSD-030 | Split `siumai::prelude` from `siumai/src/lib.rs` into `siumai/src/prelude.rs`. | Pass | Completes facade root aggregation split while preserving unified, compatibility, extension, and registry prelude paths. |
| 2026-05-25 | FPSD-030 | `cargo check -p siumai --tests --no-default-features --features openai`. | Pass | Proves the facade crate and tests compile after the `prelude` split. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_root_splits_public_namespace_modules stable_unified_prelude_excludes_compatibility_construction_aliases legacy_content_part_has_explicit_compat_namespace directional_content_namespaces_are_visible_and_compat_is_explicit stable_unified_prelude_does_not_export_legacy_content_part stable_unified_prelude_keeps_only_audited_compatibility_and_runtime_aliases broad_facade_types_path_is_explicit_compat_only stable_registry_prelude_exports_factory_signature_types stable_unified_prelude_scopes_non_family_upload_helpers stable_unified_prelude_keeps_non_family_extension_types_scoped family_taxonomy_documents_video_as_stable_and_music_as_extension_only --no-default-features --features openai --no-fail-fast`. | Pass: 11 tests run, 11 passed. | Proves the split prelude remains curated and guarded outside `lib.rs`. |
| 2026-05-25 | FPSD-030 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_directional_content_namespaces_compile public_surface_unified_prelude_excludes_legacy_content_part public_surface_extensions_imports_compile public_surface_protocol_openai_compiles public_surface_openai_provider_ext_compiles --no-default-features --features openai,protocol-openai --no-fail-fast`. | Pass: 5 tests run, 5 passed. | Proves public prelude/content/extensions/protocol imports still compile after the root split. |
| 2026-05-25 | FPSD-040 | Split `siumai::image` workflow helpers into `siumai/src/image/workflow.rs` and result projection helpers into `siumai/src/image/projection.rs`. | Pass | Reduces image facade root coupling while preserving stable `siumai::image::*` functions and type re-exports. |
| 2026-05-25 | FPSD-040 | `cargo check -p siumai --tests --no-default-features --features openai`. | Pass | Proves the facade crate and tests compile after the image helper split. |
| 2026-05-25 | FPSD-040 | `cargo nextest run -p siumai --test facade_architecture_boundary_test image_facade_splits_workflow_and_projection_helpers --no-default-features --features openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Guards that `siumai/src/image.rs` stays a root facade over named workflow/projection helpers. |
| 2026-05-25 | FPSD-040 | `cargo nextest run -p siumai image::tests --no-default-features --features openai --no-fail-fast`. | Pass: 14 tests run, 14 passed. | Proves image generation/edit/variation dispatch, batching, no-image errors, and result projection still behave after the split. |
| 2026-05-25 | FPSD-040 | `cargo nextest run -p siumai --test public_surface_imports_test public_family_helpers_compile_against_stable_family_models --no-default-features --features openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Proves stable family imports still compile through `siumai::image::*`. |
| 2026-05-25 | FPSD-050 | Split `siumai::video` workflow helpers into `siumai/src/video/workflow.rs`, generated-video extraction/materialization helpers into `siumai/src/video/materialization.rs`, and AI SDK projection/provider metadata helpers into `siumai/src/video/projection.rs`. | Pass | Reduces video facade root coupling while preserving stable `siumai::video::*` functions and type re-exports. |
| 2026-05-25 | FPSD-050 | `cargo check -p siumai --tests --no-default-features --features openai`. | Pass | Proves the facade crate and tests compile after the video helper split. |
| 2026-05-25 | FPSD-050 | `cargo nextest run -p siumai --test facade_architecture_boundary_test video_facade_splits_workflow_materialization_and_projection_helpers --no-default-features --features openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Guards that `siumai/src/video.rs` stays a root facade over named workflow/materialization/projection helpers. |
| 2026-05-25 | FPSD-050 | `cargo nextest run -p siumai video::tests --no-default-features --features openai --no-fail-fast`. | Pass: 20 tests run, 20 passed. | Proves video task polling, batching, URL/provider-reference materialization, metadata aggregation, and AI SDK projection still behave after the split. |
| 2026-05-25 | FPSD-050 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_video_family_imports_compile public_family_helpers_compile_against_stable_family_models --no-default-features --features openai --no-fail-fast`. | Pass: 2 tests run, 2 passed. | Proves stable video family imports and family helper imports still compile through `siumai::video::*`. |
| 2026-05-25 | FPSD-050 | `git diff --check`. | Pass; Git reported expected LF-to-CRLF working-copy warnings only. | Proves touched source/docs have no whitespace-error diff before commit. |
