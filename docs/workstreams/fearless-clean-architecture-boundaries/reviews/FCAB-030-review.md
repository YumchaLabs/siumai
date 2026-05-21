# FCAB-030 Review

Status: Accepted
Date: 2026-05-21
Scope: Registry factory facet split and handle rewiring.

## Workstream Compliance

- No blocking findings.
- The task goal is satisfied: stable family construction, compatibility `LlmClient` construction,
  and extension-capability construction are now represented by separate facets:
  `ProviderFamilyFactory`, `ProviderCompatibilityFactory`, and `ProviderExtensionFactory`.
- Custom provider integration remains source-compatible because downstream providers still
  implement `ProviderFactory`; the narrower facets are blanket-implemented from it.
- The task stayed in the FCAB-030 scope: registry entry factory, registry handles, SiumaiBuilder
  compatibility construction, source guards, and architecture docs.

## Code Quality

- No blocking findings.
- The facet methods use `build_*` names to avoid method-resolution ambiguity with existing
  `ProviderFactory` methods.
- Stable registry handles now call family facet methods. Compatibility calls remain isolated to
  extension-only image/audio paths and SiumaiBuilder's historical generic-client construction.
- The new source guard verifies that the family facet contains no `LlmClient` or `compat_` surface,
  while compatibility and extension facets stay explicit.

## Missing Gates

- No missing gates for FCAB-030.
- `cargo nextest run -p siumai-registry --no-fail-fast registry_entry` was attempted but selected
  zero tests because the actual test filter is `registry::entry`; the correct `registry::entry` gate
  passed and is recorded in `EVIDENCE_AND_GATES.md`.

## Residual Risk

- Built-in provider factories still implement the wide `ProviderFactory` methods directly. FCAB-040
  should now convert provider factory internals and delete redundant compatibility glue where native
  family objects exist.
- Public source compatibility is intentionally preserved. If a future release wants to make the
  family facet the public custom-provider trait, that should be a separate migration/ADR decision.
