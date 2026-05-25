# Builder Dead Capability API Removal — TODO

Status: Closed
Last updated: 2026-05-25

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[-]` intentionally deferred or split

## M0 — Scope And Evidence Freeze

- [x] BDCA-010 [owner=codex] [deps=none] [scope=docs/workstreams/builder-dead-capability-api-removal]
  Goal: Open the lane, record the no-op capability inventory, and define validation gates.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, `HANDOFF.md`, and `docs/workstreams/INDEX.md` agree.
  Review: planner self-review.
  Evidence: workstream docs.
  Handoff: DONE. Workstream docs opened, indexed, and scoped to builder-only no-op capability
  flags.

## M1 — Source Guard

- [x] BDCA-020 [owner=codex] [deps=BDCA-010] [scope=siumai-registry/tests]
  Goal: Add a focused source guard that rejects the removed builder capability storage and no-op
  helper methods.
  Validation:
  `cargo nextest run -p siumai-registry --test builder_architecture_boundary_test builder_does_not_expose_noop_capability_flags --no-default-features --features openai --no-fail-fast`
  Review: registry public-surface boundary review.
  Evidence: failing-then-passing guard test.
  Handoff: DONE. Added `builder_does_not_expose_noop_capability_flags`; it failed before removal
  and passed after BDCA-030.

## M2 — API Removal

- [x] BDCA-030 [owner=codex] [deps=BDCA-020] [scope=siumai-registry/src/provider/siumai_builder.rs]
  Goal: Remove the write-only `capabilities` field and the no-op public builder helpers.
  Validation:
  `cargo check -p siumai-registry --tests --no-default-features --features openai`
  Review: builder construction review.
  Evidence: no remaining builder capability storage or helper method definitions.
  Handoff: DONE. Removed the write-only field, initializer, helper methods, and debug-only count.

## M3 — Migration Notes And Closeout

- [x] BDCA-040 [owner=codex] [deps=BDCA-020,BDCA-030] [scope=CHANGELOG.md,docs/workstreams/builder-dead-capability-api-removal]
  Goal: Update migration notes, record evidence, and close or split follow-on work.
  Validation:
  - `cargo fmt --package siumai-registry`
  - focused gates from `EVIDENCE_AND_GATES.md`
  - `git diff --check -- CHANGELOG.md docs/workstreams/builder-dead-capability-api-removal docs/workstreams/INDEX.md siumai-registry/src/provider/siumai_builder.rs siumai-registry/tests/builder_architecture_boundary_test.rs`
  Review: final self-review.
  Evidence: `EVIDENCE_AND_GATES.md`.
  Handoff: DONE. Migration notes, architecture note, evidence, and closeout docs were updated.
