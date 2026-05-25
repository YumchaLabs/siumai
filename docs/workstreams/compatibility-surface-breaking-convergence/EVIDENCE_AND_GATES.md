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
