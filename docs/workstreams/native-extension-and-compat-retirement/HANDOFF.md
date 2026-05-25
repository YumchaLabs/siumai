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

NECR-020.

## Decisions Since Last Update

- Opened a new follow-on instead of reopening the closed compatibility-surface lane.
- Split the work into inventory first, implementation second. The first source edit should be chosen
  from evidence about existing native provider extension clients.
- Kept `LlmClient` deletion and low-level `ContentPart` root movement out of immediate scope unless
  their ADR gates are satisfied.
- Completed NECR-010 by creating the workstream docs, indexing the lane, and validating the doc
  diff.

## Blockers

- No blocker currently.

## Next Recommended Action

Run NECR-020 inventory against `siumai-registry/src` before selecting the first provider-native
extension override.
