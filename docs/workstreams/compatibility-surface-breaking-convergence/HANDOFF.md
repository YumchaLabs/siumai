# Compatibility Surface Breaking Convergence — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

This lane is newly opened from the closed public-surface deepening compatibility audit.

## Active Task

- Task ID: CSBC-040
- Owner: codex
- Files: `siumai-registry/src`, `siumai-registry/tests`, docs.
- Validation: registry compatibility factory gate.
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
- Started CSBC-030 by confirming no production code consumes `siumai_core::client` /
  `siumai_core::core::client` beyond the alias modules and guards.
- Completed CSBC-030 by deprecating the lower-level core client aliases, documenting
  `siumai_core::compat::client` as the preferred core-level import, and adding a production source
  guard against consuming the aliases.

## Blockers

- No blocker currently.

## Next Recommended Action

1. Start CSBC-040 by inventorying stable-family and extension uses of
   `ProviderCompatibilityFactory` / `compat_*_client*` in `siumai-registry`.
