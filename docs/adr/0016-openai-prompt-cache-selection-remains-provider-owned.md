# ADR-0016: OpenAI Prompt-Cache Selection Remains Provider-Owned

## Status

Accepted

## Date

2026-08-11

## Context

OpenAI exposes explicit prompt-cache breakpoints on message content together with request-level
cache controls. The service, not the client, decides whether an eligible breakpoint is read from an
existing cache entry, written as a new entry, or ignored for the current request.

An earlier Siumai design represented two client-side roles: retained historical markers and current
write candidates. It also derived mode-specific local write budgets and required one role to precede
the other. That model attempted to predict service behavior from one request snapshot. It could not
prove whether an apparently old marker was still eligible, whether an upstream conversation store
had already retained it, or which eligible marker the service would write. It also made Siumai's
public API depend on product limits that OpenAI can revise independently of the wire shape.

Siumai still needs stable validation for annotation ownership, duplicate annotations on one node,
request/body bounds, cache-control JSON shape, and protected request fields. Those invariants do not
require assigning a read/write role to each breakpoint.

## Decision

Siumai exposes one node-scoped OpenAI breakpoint annotation:

```rust
OpenAiContentOptions::prompt_cache_breakpoint()
```

Every annotated content node projects the same explicit wire-level breakpoint. Siumai does not:

- classify a breakpoint as historical, retained, or newly writable;
- require historical markers to precede write candidates;
- choose which eligible markers the provider will write;
- trim request history to a locally maintained marker window; or
- reject a request because a local model-name rule predicts different cache eligibility.

`OpenAiPromptCacheOptions` continues to own request-level mode and TTL. The separately documented
`OpenAiPromptCacheRetention` field remains available for wire fidelity while OpenAI deprecates it in
favor of TTL. Explicit caller intent reaches the wire or fails with a typed structural error; Siumai
does not silently rewrite it from model-name heuristics.

Transport and protocol bounds remain strict. A request can still fail locally for an invalid
annotation, malformed cache-control value, protected-field override, or aggregate resource limit.
These are Siumai safety and wire-shape invariants, not predictions about provider cache selection.

## Options considered

### Option A: Keep historical and write-candidate roles

Rejected. The roles are not independent wire semantics and require Siumai to predict mutable
provider cache state.

### Option B: Accept arbitrary untyped breakpoint JSON

Rejected. It would weaken annotation ownership, diagnostics, and protected request-shape checks
without improving forward compatibility.

### Option C: Use one typed wire-level breakpoint

Chosen. It preserves the provider capability while leaving cache selection with the only component
that knows the current cache state.

## Consequences

### Positive

- The public API matches the stable wire intent instead of a guessed cache lifecycle.
- Existing conversation history can retain markers without being reclassified by the caller.
- Product-limit changes do not require a new Siumai breakpoint role or ordering rule.
- Chat Completions and Responses share one annotation contract.

### Costs

- Siumai cannot promise which breakpoint a request will read or write.
- Applications that need cache-effect attribution must inspect returned usage telemetry and their
  own request history rather than a client-side marker role.
- Callers using the removed historical/write-candidate constructors must migrate to the single
  breakpoint constructor.

## Migration and verification

Replace both former marker constructors with
`OpenAiContentOptions::prompt_cache_breakpoint()`. Keep request-level mode, TTL, retention, and cache
key controls in `OpenAiResponsesOptions` or `OpenAiChatCompletionsOptions`.

Deterministic fixtures verify that every annotated node emits the same breakpoint shape, duplicate
annotations on one node fail, explicit request-level controls reach both protocol modes, and usage
telemetry remains unknown when the provider omits it. Live cache behavior is diagnostic evidence,
not a release gate.

## References

- https://developers.openai.com/api/docs/guides/prompt-caching
- https://developers.openai.com/api/reference/resources/responses/methods/create
- `docs/adr/0012-provider-annotations-follow-semantic-nodes.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/migration/siumai-next.md`
