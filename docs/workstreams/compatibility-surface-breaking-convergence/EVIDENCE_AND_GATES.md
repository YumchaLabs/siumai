# Compatibility Surface Breaking Convergence — Evidence And Gates

Status: Active
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
cargo nextest run -p siumai --test public_surface_imports_test public_surface_compat_imports_compile --no-default-features --features openai --no-fail-fast
```

This proves the current explicit facade compatibility imports compile before broad type narrowing.

## Gate Set

### Facade Compat Types Gate

```powershell
cargo check -p siumai --tests --no-default-features --features openai
cargo nextest run -p siumai --test facade_architecture_boundary_test broad_facade_types_path_is_explicit_compat_only --no-default-features --features openai --no-fail-fast
cargo nextest run -p siumai --test public_surface_imports_test public_surface_compat_imports_compile public_surface_compat_prelude_imports_compile --no-default-features --features openai --no-fail-fast
```

### Core Generic Client Gate

```powershell
cargo check -p siumai-core --tests --no-default-features
cargo nextest run -p siumai-core --test core_provider_boundary_test llm_client_is_physically_scoped_under_compat_module --no-default-features --no-fail-fast
```

### Registry Compatibility Factory Gate

```powershell
cargo check -p siumai-registry --tests --no-default-features --features openai
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --features openai --no-fail-fast
```

### ContentPart Boundary Gate

Use targeted content boundary tests selected during CSBC-050, plus public import tests for any
public namespace movement.

### Review Gate

Before accepting a breaking or near-breaking slice, check:

- ADR-0007 / ADR-0008 compliance;
- public migration docs;
- public compile tests for kept paths;
- source guards for removed or narrowed paths.

## Evidence Anchors

- `docs/workstreams/fearless-public-surface-deepening/compatibility-shim-audit.md`
- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `docs/architecture/public-surface.md`
- `docs/migration/migration-0.11.0-beta.7.md`

## Evidence Log

| Date | Task | Command / Evidence | Result | What it proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | CSBC-010 | Workstream docs opened. | Pass | Establishes the compatibility breaking-convergence lane and first executable target. |
| 2026-05-25 | CSBC-010 | `git diff --check -- docs/workstreams/compatibility-surface-breaking-convergence docs/workstreams/INDEX.md`. | Pass | Proves the new workstream docs and index have no whitespace-error diff. |
| 2026-05-25 | CSBC-020 | Narrowed `siumai::compat::types` and `siumai::prelude::compat::types`, moving the historical catch-all mirror under `legacy_all`. | Pass | Reduces the default compatibility type namespace while preserving a last-resort migration escape hatch. |
| 2026-05-25 | CSBC-020 | `cargo fmt --package siumai`. | Pass | Formats touched facade source and tests. |
| 2026-05-25 | CSBC-020 | `cargo check -p siumai --tests --no-default-features --features openai`. | Pass | Proves the facade crate and tests compile after narrowing compat type exports. |
| 2026-05-25 | CSBC-020 | `cargo nextest run -p siumai --test facade_architecture_boundary_test broad_facade_types_path_is_explicit_compat_only public_surface_deepening_compatibility_audit_classifies_remaining_shims stable_unified_prelude_excludes_compatibility_construction_aliases --no-default-features --features openai --no-fail-fast`. | Pass: 3 tests run, 3 passed. | Guards that the facade compat type namespace is narrowed and still documented. |
| 2026-05-25 | CSBC-020 | `cargo nextest run -p siumai --test public_surface_imports_test public_surface_compat_imports_compile public_surface_compat_prelude_imports_compile --no-default-features --features openai --no-fail-fast`. | Pass: 2 tests run, 2 passed. | Proves retained compat type imports and the nested `legacy_all` escape hatch compile. |
| 2026-05-25 | CSBC-020 | `git diff --check`. | Pass; Git reported expected LF-to-CRLF working-copy warnings only. | Proves touched source/docs have no whitespace-error diff before commit. |
| 2026-05-25 | CSBC-030 | Deprecated `siumai_core::client` and `siumai_core::core::client` aliases with ADR-0007 removal guidance. | Pass | Makes lower-level generic-client aliases visibly transitional while preserving compatibility. |
| 2026-05-25 | CSBC-030 | `cargo fmt --package siumai-core`. | Pass | Formats touched core source and tests. |
| 2026-05-25 | CSBC-030 | `cargo check -p siumai-core --tests --no-default-features`. | Pass | Proves core aliases and tests compile after adding deprecation annotations. |
| 2026-05-25 | CSBC-030 | `cargo nextest run -p siumai-core --test core_provider_boundary_test llm_client_is_physically_scoped_under_compat_module --no-default-features --no-fail-fast`. | Pass: 1 test run, 1 passed. | Guards physical generic-client ownership under `compat::client` and prevents production alias consumption. |
| 2026-05-25 | CSBC-030 | `git diff --check`. | Pass; Git reported expected LF-to-CRLF working-copy warnings only. | Proves touched source/docs have no whitespace-error diff before commit. |
| 2026-05-25 | CSBC-040 | Moved image, speech, and transcription extras construction behind `ProviderExtensionFactory`; removed stored `ProviderCompatibilityFactory` from `ProviderFactoryFacets` and stable image/audio handles. | Pass | Reduces stable registry-family dependency on generic-client compatibility construction. |
| 2026-05-25 | CSBC-040 | `cargo fmt --package siumai-registry`. | Pass | Formats touched registry source and tests. |
| 2026-05-25 | CSBC-040 | `cargo check -p siumai-registry --tests --no-default-features --features openai`. | Pass | Proves registry source and tests compile after moving extras to the extension facet. |
| 2026-05-25 | CSBC-040 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test public_docs_classify_generic_llm_client_factory_paths_as_migration_only --no-default-features --features openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Proves public docs still classify generic-client factory paths as migration-only. |
| 2026-05-25 | CSBC-040 | `cargo nextest run -p siumai-registry registry_handles_depend_on_narrow_factory_facets stable_registry_handles_do_not_use_compat_client_paths_for_primary_family_execution remaining_registry_handle_compat_paths_are_extension_only provider_factory_facets_split_stable_compat_and_extension_execution --no-default-features --features openai --no-fail-fast`. | Pass: 4 tests run, 4 passed. | Guards that registry handles use family/extension facets and do not route primary family execution through compatibility clients. |
