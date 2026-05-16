# Architecture Decision Records (ADR)

This folder contains Architecture Decision Records for the fearless refactor.

## Index

- `0001-vercel-aligned-modular-split.md` — Adopt a Vercel-aligned modular split (interfaces / provider-utils / providers) adapted for Rust.
- `0002-provider-crates-by-provider.md` — Split provider implementations into provider crates (provider-first). (The historical umbrella `siumai-providers` is now legacy and removed from the workspace.)
- `0003-provider-ext-export-policy.md` — Define a scoped export policy for `siumai::provider_ext::<provider>` to reduce accidental coupling.
- `0004-experimental-surface-policy.md` — Define stability tiers and the policy for `siumai::experimental::*`.
- `0005-builder-retention-and-convergence-policy.md` — Keep builders as ergonomic wrappers over canonical config-first construction.
- `0006-family-model-first-trait-policy.md` — Make family model traits the primary execution contracts.
- `0007-llmclient-demotion-policy.md` — Demote `LlmClient` to a compatibility abstraction.
- `0008-legacy-content-part-compatibility-boundary.md` — Classify legacy `ContentPart` as a compatibility carrier until a later breaking slice can move it.

## Conventions

- ADRs are written in English.
- Status starts as **Proposed** and moves to **Accepted** once the team agrees.
- Each ADR records:
  - context/problem statement
  - options considered (with trade-offs)
  - decision
  - consequences and migration plan
