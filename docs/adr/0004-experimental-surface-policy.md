---
status: Proposed
date: 2026-01-04
---

# ADR 0004: Experimental surface policy (`siumai::experimental`)

## Context

During the Alpha.5 split-crate refactor, we need to balance:

- a small, stable facade surface for most users
- enough low-level building blocks for advanced integrations and debugging
- the freedom to refactor internal architecture without freezing internal module layouts

Without an explicit policy, low-level internals tend to leak into “stable-looking” paths, which
creates accidental coupling and makes refactors riskier.

## Decision

We define three stability tiers:

### Tier A: Stable unified surface

`siumai::prelude::unified::*` is the recommended stable surface for most users.
It represents the Vercel-aligned unified model families and should remain source-stable within the
Alpha series as much as practical.

### Tier B: Stable provider extension roots (scoped)

`siumai::provider_ext::<provider>` is a stable module root for provider-specific features.
To reduce accidental coupling, provider extensions are scoped into:

- `options::*` — typed provider options + request extension traits
- `metadata::*` — typed response metadata extraction helpers (when supported)
- `resources::*` — provider-specific resources not covered by unified families (when applicable)
- `ext::*` — non-unified escape hatches (helpers/parsers/adapters); stable root, but APIs may evolve faster

### Tier C: Experimental low-level building blocks

`siumai::experimental::*` is explicitly **unstable**.

It may contain executors, wiring, auth/middleware/interceptors, protocol helpers, or provider internals.
These APIs are allowed to change, move, or be removed across minor/beta releases without deprecation.

Users should only depend on `experimental` when they:

- build custom providers or deep integrations
- need access to observability/debug hooks not part of the stable surfaces
- accept higher migration cost in exchange for low-level control

## Consequences

- We prefer moving “internal” utilities to `experimental` instead of expanding the stable facade.
- New features should default to Tier A or Tier B only when they have clear semantics and broad demand.
- CI should include a public-surface guard test and facade example compilation to prevent accidental
  breaking changes in Tier A/B.
