# Compatibility Surface Breaking Convergence — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

This lane is newly opened from the closed public-surface deepening compatibility audit.

## Active Task

- Task ID: CSBC-050
- Owner: codex
- Files: `siumai-core`, `siumai-spec`, `siumai`, docs, tests.
- Validation: content boundary and public import gates selected during task execution.
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
- Completed CSBC-040 by moving image, speech, and transcription extras construction behind
  `ProviderExtensionFactory`, removing `ProviderCompatibilityFactory` storage from stable
  image/audio handles, and shrinking `ProviderFactoryFacets` to family/extension facets only.
  `ProviderCompatibilityFactory` remains available through the explicit
  `compatibility_facet_from_provider_factory(...)` adapter used by `SiumaiBuilder` generic-client
  migration construction.

## Blockers

- No blocker currently.

## Next Recommended Action

1. Start CSBC-050 by reading ADR-0008 and inventorying current public `ContentPart` /
   `legacy_content` import paths before choosing whether a breaking namespace move is safe.
