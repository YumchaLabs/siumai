---
status: Proposed
date: 2026-01-04
---

# ADR 0003: `provider_ext` export policy (scoped, discoverable, low-coupling)

## Context

During the Alpha.5 split-crate refactor, we want the `siumai` facade to have a small, stable surface.
Provider-specific APIs must remain accessible without encouraging accidental cross-layer coupling.

Historically, `siumai::provider_ext::<provider>::*` tended to accumulate:

- typed provider options (request-side)
- typed metadata (response-side)
- non-unified escape hatches (provider resources, helpers, streaming parsers)

When everything is flattened into `*`, it becomes easy to import too much by accident.
This increases coupling and makes refactors riskier.

## Options considered

1) **Keep everything flattened at `siumai::provider_ext::<provider>::*`**
- Pros: shortest imports
- Cons: accidental imports, hard to audit, coupling drift over time

2) **Expose only provider crates (no facade re-exports)**
- Pros: maximum explicitness
- Cons: facade users must learn many crate/module paths; harder migrations; less “Vercel-aligned” ergonomics

3) **Scoped submodules under `provider_ext` (chosen)**
- Pros: discoverable, avoids accidental imports, keeps stable module roots
- Cons: slightly longer import paths; requires migrating docs/examples

## Decision

Adopt a scoped and structured export policy under `siumai::provider_ext::<provider>`:

- `options::*` — typed provider options and request extension traits
- `metadata::*` — typed response metadata extraction helpers (when supported)
- `ext::*` — non-unified escape hatches (provider helpers, streaming event parsers, etc.)
- `resources::*` — provider-specific clients/resources not covered by unified model families (when applicable)

Additionally:

- `ext::*` is **not** flattened into `siumai::provider_ext::<provider>::*`.
- We may still re-export a small set of “common” provider items at the provider module root
  (e.g., `*Client`, `*Config`, and the most common typed option/metadata types) for ergonomics.

## Consequences

- Docs and examples should prefer structured imports:
  - `use siumai::provider_ext::openai::{options::*, metadata::*};`
  - `use siumai::provider_ext::openai::ext::*;` (only when needed)
- CI should include guardrails that compile facade examples and verify import stability.

## Migration plan

- Update docs and examples to use the structured submodules.
- Keep the provider module roots stable (`siumai::provider_ext::<provider>`) as the primary entrypoint.
