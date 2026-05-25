# Compatibility Surface Breaking Convergence — Handoff

Status: Closed
Last updated: 2026-05-25

## Current State

This lane is closed. It was opened from the closed public-surface deepening compatibility audit to
narrow remaining compatibility surfaces and record any unsafe breaking removals.

## Active Task

None.

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
- Completed CSBC-050 with DONE_WITH_CONCERNS. The facade-level `ContentPart` compatibility break is
  already complete, but moving the low-level `siumai-spec::types::ContentPart` /
  `siumai-core::types::ContentPart` root paths remains blocked by serde-facing `ChatMessage` /
  `ChatResponse`, provider/protocol response parity, and missing full-root fixture coverage. The
  blockers are recorded in `CSBC-050-content-part-decision.md` and guarded in
  `siumai-spec/tests/content_projection_boundary_test.rs`.

## Blockers

- No blocker currently.

## Residual Risks

- Low-level core generic-client aliases remain as deprecated migration aliases until ADR-0007
  conditions are met.
- `ProviderCompatibilityFactory` / `ProviderFactory::compat_*_client*` remain explicit migration
  seams for method-style generic-client construction.
- Low-level `ContentPart` root paths remain until ADR-0008 root-move fixture/parity prerequisites
  are complete.

## Next Recommended Action

No active task remains in this lane. Start a new narrow follow-on only for a concrete residual risk:
core alias deletion, method-style/generic-client retirement, or full `ContentPart` root namespace
movement.
