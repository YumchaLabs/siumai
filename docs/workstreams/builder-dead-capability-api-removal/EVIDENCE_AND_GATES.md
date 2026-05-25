# Builder Dead Capability API Removal — Evidence And Gates

Status: Closed
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
rg -n "capabilities: Vec<String>|with_capability\\(|pub fn with_audio\\(self\\)|pub fn with_embedding\\(self\\)|pub fn with_image_generation\\(self\\)|capabilities_count" siumai-registry/src/provider/siumai_builder.rs
```

This shows the builder-owned capability strings and helper methods that must disappear. It is
intentionally scoped to `SiumaiBuilder`; `ProviderCapabilities` fluent methods are a different,
valid provider metadata surface.

## Gate Set

### Source Boundary Gate

```powershell
cargo nextest run -p siumai-registry --test builder_architecture_boundary_test builder_does_not_expose_noop_capability_flags --no-default-features --features openai --no-fail-fast
```

### Registry Compile Gate

```powershell
cargo check -p siumai-registry --tests --no-default-features --features openai
```

### Formatting / Diff Gate

```powershell
cargo fmt --package siumai-registry
git diff --check -- CHANGELOG.md docs/architecture/public-surface.md docs/workstreams/builder-dead-capability-api-removal docs/workstreams/INDEX.md siumai-registry/src/provider/siumai_builder.rs siumai-registry/tests/builder_architecture_boundary_test.rs
```

## Evidence Log

| Date | Task | Command / Evidence | Result | What it proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | BDCA-010 | Source inventory in `siumai-registry/src/provider/siumai_builder.rs`. | Pass | Found that builder capability strings were write-only and only surfaced through `Debug` as `capabilities_count`. |
| 2026-05-25 | BDCA-020 | `cargo nextest run -p siumai-registry --test builder_architecture_boundary_test builder_does_not_expose_noop_capability_flags --no-default-features --features openai --no-fail-fast` before implementation. | Expected fail: `capabilities: Vec<String>`. | Proves the source guard detects the existing no-op builder capability surface. |
| 2026-05-25 | BDCA-030 | Removed `capabilities: Vec<String>`, its initializer, `with_capability()`, `with_audio()`, `with_embedding()`, `with_image_generation()`, and `capabilities_count`. | Pass | Removes the false builder capability control plane without touching provider metadata capability APIs. |
| 2026-05-25 | BDCA-020/BDCA-030 | `cargo nextest run -p siumai-registry --test builder_architecture_boundary_test builder_does_not_expose_noop_capability_flags --no-default-features --features openai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Guards that `SiumaiBuilder` does not reintroduce write-only capability flags. |
| 2026-05-25 | BDCA-030 | `cargo check -p siumai-registry --tests --no-default-features --features openai`. | Pass | Proves the registry crate compiles after the builder API removal under the touched feature set. |
| 2026-05-25 | BDCA-040 | `cargo fmt --package siumai-registry -- --check`. | Pass | Proves the touched registry source and test are formatted. |
| 2026-05-25 | BDCA-040 | `cargo check -p siumai-registry --tests --all-features`. | Pass | Proves all registry feature-gated builder code still compiles after the public API removal. |
| 2026-05-25 | BDCA-040 | `rg -n "capabilities: Vec<String>|capabilities: Vec::new\\(\\)|pub fn with_capability<|pub fn with_audio\\(self\\) -> Self|pub fn with_embedding\\(self\\) -> Self|pub fn with_image_generation\\(self\\) -> Self|capabilities_count" siumai-registry/src/provider/siumai_builder.rs`. | Pass: no matches. | Confirms the removed builder-only surface is absent from `SiumaiBuilder`. |
| 2026-05-25 | BDCA-040 | `git diff --check -- CHANGELOG.md docs/architecture/public-surface.md docs/workstreams/builder-dead-capability-api-removal docs/workstreams/INDEX.md siumai-registry/src/provider/siumai_builder.rs siumai-registry/tests/builder_architecture_boundary_test.rs`. | Pass; Git reported expected LF-to-CRLF working-copy warnings. | Confirms touched docs and code diffs have no whitespace-error diff. |

## Closeout Gate Matrix

| Area | Evidence | Result |
| --- | --- | --- |
| Source guard | `builder_does_not_expose_noop_capability_flags` under `openai`. | Pass |
| Compile gate | `cargo check -p siumai-registry --tests --no-default-features --features openai`. | Pass |
| Feature coverage | `cargo check -p siumai-registry --tests --all-features`. | Pass |
| Migration docs | `CHANGELOG.md` and `docs/architecture/public-surface.md`. | Pass |
| Diff hygiene | `git diff --check` for touched docs, builder source, and boundary test. | Pass |

## Broader Gates Not Run

- Full workspace nextest was not run. This slice removes no-op methods from `SiumaiBuilder`; the
  focused source guard plus `siumai-registry` all-features compile gate cover the changed API
  surface without exercising unrelated provider runtime fixtures.

## Residual Risks

- This is a public API removal. The migration note must be explicit that capability selection now
  belongs to provider/family/extension APIs rather than builder capability flags.
- The guard is source-level because Rust compile-fail tests would add heavier test harness
  complexity for this narrow deletion.
