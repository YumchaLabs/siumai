# Native Extension And Compat Retirement — Handoff

Status: Closed
Last updated: 2026-05-25

## Current State

This lane is closed. It follows the closed compatibility-surface breaking-convergence lane and
tracks three residual refactor areas requested for continued fearless refactoring:

- provider-specific native extension factory convergence;
- method-style / generic-client retirement planning;
- ADR-0008 low-level root `ContentPart` namespace move preparation.

## Active Task

None.

## Decisions Since Last Update

- Opened a new follow-on instead of reopening the closed compatibility-surface lane.
- Split the work into inventory first, implementation second. The first source edit should be chosen
  from evidence about existing native provider extension clients.
- Kept `LlmClient` deletion and low-level `ContentPart` root movement out of immediate scope unless
  their ADR gates are satisfied.
- Completed NECR-010 by creating the workstream docs, indexing the lane, and validating the doc
  diff.
- Completed NECR-020 by inventorying extension-facet fallbacks and selecting image extras for
  DeepInfra, Fireworks, and TogetherAI as the first native override set.
- Completed NECR-030 by overriding `image_extras_with_ctx(...)` in those three provider factories
  and adding `hybrid_provider_image_extras_use_native_extension_clients` as a source guard.
- Completed NECR-040 by adding a source guard that confines `ProviderCompatibilityFactory` to
  `registry/entry/factory.rs` and the historical `SiumaiBuilder` method-style compatibility path,
  then documenting ADR-0007 deletion gates.
- Completed NECR-050 by adding `adr_0008_root_content_part_move_has_serde_parity_fixture_gate`.
  The gate locks root/compat `ContentPart` serialization inside `ChatMessage` and `ChatResponse`
  and records the current externally tagged `MessageContent::MultiModal` and top-level
  `provider_metadata` response metadata shapes.
- Completed NECR-060 by closing the lane and splitting remaining removals into follow-ons.

## Blockers

- No blocker currently.

## Next Recommended Action

No active task remains in this lane. Open a new follow-on for broader provider/protocol fixture
parity, additional native extension overrides, or the eventual breaking low-level root
`ContentPart` namespace movement.
