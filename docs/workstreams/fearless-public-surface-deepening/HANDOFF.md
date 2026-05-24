# Fearless Public Surface Deepening — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

This workstream is newly opened as a follow-on to the closed FCAB and module-deepening lanes. It
tracks the next public-surface and runtime deepening queue:

- narrow `siumai::tooling`,
- split facade aggregation in `siumai/src/lib.rs`,
- split large `image` and `video` facade Modules,
- perform one or more core streaming deepening slices,
- and audit remaining compatibility shims.

## Active Task

- Task ID: FPSD-010
- Owner: planner
- Files: `docs/workstreams/fearless-public-surface-deepening/*`
- Validation: document consistency and `git diff --check` for this directory
- Status: NEEDS_CONTEXT until docs are committed
- Review: planner self-review
- Evidence: `EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Opened a new active follow-on lane instead of reopening `fearless-clean-architecture-boundaries`,
  because that umbrella lane is closed and its handoff says new work should be concrete follow-ons.
- Chose `FPSD-020` (`siumai::tooling` explicit facade exports) as the first executable task because
  `siumai-core::tooling` was just split and public compile guards already exist.
- Compatibility shim deletion is intentionally last because ADR-0007 and ADR-0008 require migration
  evidence before removing compatibility paths.

## Blockers

- No blocker currently.

## Next Recommended Action

1. Finish FPSD-010 by verifying the workstream docs.
2. Commit the workstream opening docs.
3. Start FPSD-020 by replacing `siumai/src/tooling.rs` wildcard export with explicit re-exports.
