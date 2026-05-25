# Compatibility Surface Breaking Convergence — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

This lane is newly opened from the closed public-surface deepening compatibility audit.

## Active Task

- Task ID: CSBC-020
- Owner: codex
- Files: `siumai/src/compat.rs`, `siumai/src/prelude.rs`, `siumai/tests`, docs.
- Validation: facade compat type gate.
- Status: READY
- Evidence: `EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Completed CSBC-010 by opening and indexing this workstream.
- Scope is intentionally limited to compatibility-surface breaking convergence.
- First executable implementation slice is `CSBC-020`, broad facade compat type narrowing.
- Alias deletion and ContentPart namespace movement are not assumed safe; they must satisfy
  ADR-0007 / ADR-0008 conditions or be split.

## Blockers

- No blocker currently.

## Next Recommended Action

1. Start CSBC-020 by inventorying `siumai_core::types` names currently consumed through
   `siumai::compat::types` and `siumai::prelude::compat::types`.
