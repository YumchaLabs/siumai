# Fearless Provider Composite Client Isolation - Handoff

Last updated: 2026-05-16

## Current State

The workstream is closed.

## Next Step

No immediate continuation is required for this lane.

## Notes

- The wrappers should not be deleted in this lane because historical method-style `Siumai` and
  generic `LlmClient` compatibility paths still need them.
- The important invariant is that stable family methods construct native family objects directly
  and never route through the composite wrapper.
- A later compatibility-removal lane can revisit whether `compat_language_client_with_ctx(...)`
  should compose family models differently or disappear after method-style compatibility is removed.
