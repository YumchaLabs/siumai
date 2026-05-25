# Registry Typed Builder Isolation — TODO

Status: Closed
Last updated: 2026-05-25

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[-]` intentionally deferred or split

## M0 — Scope And Evidence Freeze

- [x] RTBI-010 [owner=codex] [deps=none] [scope=docs/workstreams/registry-typed-builder-isolation]
  Goal: Open the lane and record why typed helper implementation should move out of
  `registry::factory`.
  Validation: workstream docs and `docs/workstreams/INDEX.md` agree.
  Review: planner self-review.
  Evidence: workstream docs.
  Handoff: DONE. Workstream docs opened and scoped to typed helper implementation isolation, not
  deletion of deprecated generic-client helpers.

## M1 — Source Guard

- [x] RTBI-020 [owner=codex] [deps=RTBI-010] [scope=siumai-registry/tests]
  Goal: Add a focused guard that rejects production provider factory calls to
  `crate::registry::factory::{typed helpers}` and checks that `registry::factory` is compatibility
  wrapper-only for typed helper paths.
  Validation:
  `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test production_factories_use_internal_typed_builders_not_legacy_factory_module --no-default-features --features openai,google,google-vertex,togetherai,deepinfra --no-fail-fast`
  Review: registry construction-boundary review.
  Evidence: failing-then-passing source guard.
  Handoff: DONE. Added `production_factories_use_internal_typed_builders_not_legacy_factory_module`.

## M2 — Typed Builder Isolation

- [x] RTBI-030 [owner=codex] [deps=RTBI-020] [scope=siumai-registry/src/registry]
  Goal: Move typed builder implementation into an internal module and migrate production factory
  call sites away from `registry::factory`.
  Validation:
  `cargo check -p siumai-registry --tests --all-features`
  Review: provider factory ownership review.
  Evidence: production factory calls use the internal typed builder module.
  Handoff: DONE. Added internal `registry::typed_builders`, retained compatibility wrappers in
  `registry::factory`, and moved production factory call sites to the internal module.

## M3 — Docs And Closeout

- [x] RTBI-040 [owner=codex] [deps=RTBI-020,RTBI-030] [scope=CHANGELOG.md,docs/workstreams/registry-typed-builder-isolation]
  Goal: Record migration/architecture notes, evidence, and closeout status.
  Validation:
  - `cargo fmt --package siumai-registry -- --check`
  - focused gates from `EVIDENCE_AND_GATES.md`
  - `git diff --check -- CHANGELOG.md docs/architecture/public-surface.md docs/workstreams/registry-typed-builder-isolation docs/workstreams/INDEX.md siumai-registry/src/registry siumai-registry/tests/factory_architecture_boundary_test.rs`
  Review: final self-review.
  Evidence: `EVIDENCE_AND_GATES.md`.
  Handoff: DONE. Architecture notes, changelog note, evidence, and closeout docs were updated.
