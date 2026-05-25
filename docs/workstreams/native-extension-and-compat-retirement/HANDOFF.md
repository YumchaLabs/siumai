# Native Extension And Compat Retirement — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

This lane is active. It follows the closed compatibility-surface breaking-convergence lane and
tracks three residual refactor areas requested for continued fearless refactoring:

- provider-specific native extension factory convergence;
- method-style / generic-client retirement planning;
- ADR-0008 low-level root `ContentPart` namespace move preparation.

## Active Task

NECR-040.

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

## Blockers

- No blocker currently.

## Next Recommended Action

Continue with NECR-040. The next useful step is to make method-style/generic-client retirement
criteria source-enforced, building on the remaining `ProviderCompatibilityFactory` and
`compat_*_client_with_ctx(...)` call-site inventory.
