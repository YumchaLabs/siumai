# Fearless Public Surface Deepening — Milestones

Status: Active
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

Exit criteria:

- The problem and target state are explicit.
- Related ADRs, architecture docs, and closed workstreams are linked.
- Tasks are split into independently validatable slices.

Primary evidence:

- `docs/workstreams/fearless-public-surface-deepening/DESIGN.md`
- `docs/workstreams/fearless-public-surface-deepening/TODO.md`

## M1 — Facade Surface Tightening

Exit criteria:

- `siumai::tooling` no longer wildcard-mirrors `siumai_core::tooling`.
- Facade public compile tests prove the curated surface still works.
- The source guard explains why tooling remains explicit instead of prelude-wide.

Primary gates:

- `cargo check -p siumai --tests --no-default-features --features openai`
- tooling public-surface nextest filters.

## M2 — Large Facade Module Deepening

Exit criteria:

- `siumai/src/lib.rs` is reduced to a clearer public root over named facade Modules, or deferred
  with specific blockers.
- `siumai::image` and `siumai::video` are split by workflow or helper role while keeping their root
  import surfaces intact, or split into narrower child lanes.

Primary gates:

- focused facade architecture guards;
- public-surface import guards for touched Modules.

## M3 — Core Streaming Deepening

Exit criteria:

- At least one core streaming Module split lands with behavior-preserving tests, or the work is
  split into a dedicated streaming child lane.
- Provider-map neutrality guards cover the new source files.

Primary gates:

- `cargo check -p siumai-core --tests --no-default-features`
- focused `siumai-core` streaming nextest filters.

## M4 — Compatibility Shim Audit

Exit criteria:

- Remaining compatibility shims are classified as retained, removed, or split into a future
  breaking-change lane.
- Retained shims have documented reasons and source guards.
- Removed shims have migration notes and public compile coverage.

Primary gates:

- facade architecture guard filters;
- public compatibility import compile tests.

## M5 — Closeout

Exit criteria:

- Evidence is recorded for all completed slices.
- Deferred work is explicit.
- `WORKSTREAM.json`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, and `HANDOFF.md` agree.
