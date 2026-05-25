# Compatibility Surface Breaking Convergence — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

This lane is newly opened from the closed public-surface deepening compatibility audit.

## Active Task

- Task ID: CSBC-030
- Owner: codex
- Files: `siumai-core/src`, `siumai-core/tests`, docs.
- Validation: core generic client gate.
- Status: READY
- Evidence: `EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Completed CSBC-010 by opening and indexing this workstream.
- Scope is intentionally limited to compatibility-surface breaking convergence.
- First executable implementation slice is `CSBC-020`, broad facade compat type narrowing.
- Alias deletion and ContentPart namespace movement are not assumed safe; they must satisfy
  ADR-0007 / ADR-0008 conditions or be split.
- Started CSBC-020 by narrowing `siumai::compat::types` / `prelude::compat::types` to common legacy
  names and moving the full historical mirror to nested `legacy_all`.
- Completed CSBC-020 by keeping only `ChatMessage`, `Tool`, `StopCondition`, and `Warning` in the
  default compat type namespace, moving the full historical mirror to `legacy_all`, and updating
  public-surface and migration docs.

## Blockers

- No blocker currently.

## Next Recommended Action

1. Start CSBC-030 by inventorying internal and test uses of `siumai_core::client` /
   `siumai_core::core::client`.
