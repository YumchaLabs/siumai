# Registry Typed Builder Isolation — Evidence And Gates

Status: Closed
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
rg -n "crate::registry::factory::(build_openai_compatible_typed_client|build_gemini_typed_client|build_anthropic_vertex_typed_client|build_google_vertex_typed_client|OpenAiChatApiMode)" siumai-registry/src/registry/factories
```

Before implementation this showed production provider factories depending on typed helpers through
the legacy `registry::factory` module. After closeout it should return no production matches.

## Gate Set

### Source Boundary Gate

```powershell
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test production_factories_use_internal_typed_builders_not_legacy_factory_module --no-default-features --features openai,google,google-vertex,togetherai,deepinfra --no-fail-fast
```

### Registry Compile Gate

```powershell
cargo check -p siumai-registry --tests --all-features
```

### Formatting / Diff Gate

```powershell
cargo fmt --package siumai-registry -- --check
git diff --check -- CHANGELOG.md docs/architecture/public-surface.md docs/workstreams/registry-typed-builder-isolation docs/workstreams/INDEX.md siumai-registry/src/registry siumai-registry/tests/factory_architecture_boundary_test.rs
```

## Evidence Log

| Date | Task | Command / Evidence | Result | What it proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | RTBI-010 | Source inventory of typed helper calls through `crate::registry::factory`. | Pass | Captured that production factory paths mixed typed construction with the legacy public factory module before isolation. |
| 2026-05-25 | RTBI-020 | `git diff -U0 -- siumai-registry/src/registry/factories \| rg -n "^-.*crate::registry::factory::(build_openai_compatible_typed_client\|build_gemini_typed_client\|build_anthropic_vertex_typed_client\|build_google_vertex_typed_client\|OpenAiChatApiMode)"`. | Pass: removed legacy typed helper paths from production factories. | Proves the source guard targets real pre-isolation production calls without relying on old-session test output. |
| 2026-05-25 | RTBI-030 | Added `siumai-registry/src/registry/typed_builders.rs`, changed production factories to `crate::registry::typed_builders::*`, and kept `registry::factory` wrappers. | Pass | Moves typed helper implementation behind an internal registry module while preserving compatibility paths. |
| 2026-05-25 | RTBI-030 | `rg -n "crate::registry::factory::(build_openai_compatible_typed_client\|build_gemini_typed_client\|build_anthropic_vertex_typed_client\|build_google_vertex_typed_client\|OpenAiChatApiMode)" siumai-registry/src/registry/factories -S`. | Pass: no production matches. | Confirms production provider factories no longer depend on the legacy public module path for typed helper construction. |
| 2026-05-25 | RTBI-020/RTBI-030 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test production_factories_use_internal_typed_builders_not_legacy_factory_module --no-default-features --features openai,google,google-vertex,togetherai,deepinfra --no-fail-fast`. | Pass: 1 test run, 1 passed. | Guards that production provider factories do not call typed helpers through `registry::factory`. |
| 2026-05-25 | RTBI-030 | `cargo check -p siumai-registry --tests --all-features`. | Pass | Proves all feature-gated registry typed builder users compile after the internal-module extraction. |
| 2026-05-25 | RTBI-040 | `cargo fmt --package siumai-registry`; then `cargo fmt --package siumai-registry -- --check`. | Pass | Formats and verifies the touched registry source and boundary test. |
| 2026-05-25 | RTBI-040 | `git diff --check -- CHANGELOG.md docs/architecture/public-surface.md docs/workstreams/registry-typed-builder-isolation docs/workstreams/INDEX.md siumai-registry/src/registry siumai-registry/tests/factory_architecture_boundary_test.rs`. | Pass; Git reported expected LF-to-CRLF working-copy warnings. | Confirms touched docs, registry source, and boundary test diffs have no whitespace-error diff. |

## Closeout Gate Matrix

| Area | Evidence | Result |
| --- | --- | --- |
| Source guard | `production_factories_use_internal_typed_builders_not_legacy_factory_module` under `openai,google,google-vertex,togetherai,deepinfra`. | Pass |
| Compile gate | `cargo check -p siumai-registry --tests --all-features`. | Pass |
| Compatibility wrappers | `registry::factory` delegates typed helper paths to `registry::typed_builders`. | Pass |
| Migration docs | `CHANGELOG.md` and `docs/architecture/public-surface.md`. | Pass |
| Diff hygiene | `git diff --check` for touched docs, registry source, and boundary test. | Pass |

## Broader Gates Not Run

- Full workspace nextest was not run. This slice changes registry module ownership and feature-gated
  construction call paths; the focused source guard plus `siumai-registry` all-features compile gate
  cover the changed surface without exercising unrelated provider runtime fixtures.

## Residual Risks

- Public typed helper paths may exist in older direct-import code. This lane should keep wrappers
  in `registry::factory` unless a separate breaking decision removes them.
- The all-features compile gate remains important for future edits because typed helper users are
  feature-gated across OpenAI-compatible, Gemini, and Vertex providers.
