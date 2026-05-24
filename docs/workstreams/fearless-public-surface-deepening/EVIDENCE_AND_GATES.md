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
