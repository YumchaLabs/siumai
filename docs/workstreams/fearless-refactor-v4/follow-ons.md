# Fearless Refactor V4 - Follow-ons

Last updated: 2026-05-17

The V4 core architecture workstream is closed. Future work should start from this backlog only when a
concrete product, provider, behavior, or public API gap appears. Do not reopen V4 for mechanical
cleanup.

## Follow-on Boundaries

### Typed Metadata Boundary Pass

Driver: `typed-metadata-boundary-matrix.md`.

Open a new workstream only when provider evidence shows a stable response-side contract that should
move from raw `provider_metadata` into a typed escape hatch. Current explicit non-goals remain:
Azure and Google Vertex stream/event payloads stay raw unless a stronger response-level contract
emerges.

### Provider Capability Alignment Pass

Driver: `provider-capability-alignment-matrix.md`.

Open a new workstream only when a provider's advertised public facade diverges from real capability
behavior, or when a deferred family becomes backed by provider documentation plus no-network request
boundary evidence. Current `Deferred` cells are not V4 blockers.

### Hosted Search Stable Surface

Driver: `hosted-search-surface.md`.

Keep OpenRouter, Perplexity, and xAI on provider-owned typed extensions for now. Reconsider a Stable
cross-provider hosted-search surface only after at least three providers converge on both request and
response semantics.

### OpenAI-compatible Internal Boundaries

Driver: concrete bugs only.

Track J is closed. New OpenAI-compatible work should start from behavior drift, public API drift, or
a real ownership/coupling problem. File size alone is not a trigger.

## Closeout Rule

Before opening a follow-on, write down:

- the observed failing behavior or missing public contract,
- the matrix row or provider evidence that justifies it,
- the smallest affected provider/package set,
- the no-network gate that will prove the fix.
