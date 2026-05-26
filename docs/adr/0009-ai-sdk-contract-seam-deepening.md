# ADR 0009: AI SDK contract seam deepening

- Status: accepted
- Date: 2026-05-27

## Context

The AI SDK contract hardening workstream made the stream, diagnostics, tool ownership, usage, and
unsupported capability contracts explicit. That work closed the immediate ambiguity, but the follow-up
architecture review found that several important rules still live as repeated caller knowledge:

- OpenAI Responses stream state combines reasoning lifecycle, replay hints, terminal buffering,
  provider tool state, and serializer state in one wide module.
- Public provider metadata versus private diagnostics has clear markers, but public projection
  adapters do not all cross one executable seam.
- Provider-executed tool ownership is represented across constructors, prompt validation, runtime
  tooling factories, UI conversion, and OpenAI stream adapters.
- Unsupported capability behavior is explicit, but each executor still repeats the same guard
  choreography.
- Usage values have a strong type, but stream snapshot replacement versus multi-call aggregation is
  still caller knowledge.
- Stream final-response assembly is partially split out, but accumulation and terminal replay
  reconciliation still share too much implementation state.

ADR-0001 already requires a Vercel-aligned modular split, ADR-0006 makes family-model traits the
primary contracts, and ADR-0008 classifies legacy `ContentPart` as compatibility-only. This ADR
narrows the next architecture step: deepen the AI SDK contract seams without reopening those
decisions.

## Decision

Adopt AI SDK contract seam deepening as the next fearless refactor rule.

1. Contract rules that multiple callers must remember should move behind deep modules.
2. Public projection and private diagnostics must cross an executable seam before user-visible
   output is produced.
3. Provider-executed tool ownership must be owned by a named contract module, not inferred from
   constructor naming or scattered booleans.
4. Family capability gating must concentrate feature naming, unsupported behavior, warning/error
   projection, and provider fallback policy.
5. Stream usage handling must distinguish provider-call snapshot replacement from explicit
   orchestration aggregation through a named module.
6. OpenAI Responses stream replay, reasoning lifecycle, terminal buffering, and provider tool state
   may stay in the protocol crate, but their implementation should be internally deepened into named
   modules with smaller interfaces.

## Options considered

### Option A - Keep the contract hardening as documentation and tests only

Pros:

- Minimal churn after the previous workstream.
- No risk of introducing new public types too early.

Cons:

- The same subtle rules stay spread across adapters and executors.
- Future providers can bypass the rules by adding new branches.
- Tests remain broad fixture checks instead of focused module interface checks.

### Option B - Push every rule into `siumai-spec`

Pros:

- Strong central signal for downstream users.
- Fewer protocol-specific interpretations.

Cons:

- Protocol-specific replay and provider tool state would leak into the stable spec layer.
- Conflicts with ADR-0001 by moving provider/protocol behavior upward.
- Makes public compatibility harder if the internal shape needs to keep changing.

### Option C - Deepen seams in the owning crates

Pros:

- Keeps stable contracts in `siumai-spec`, runtime orchestration in `siumai-core`, and OpenAI wire
  behavior in `siumai-protocol-openai`.
- Converts repeated caller knowledge into named module interfaces.
- Allows old compatibility surfaces to remain while new internal modules become the test surface.

Cons:

- Requires multiple coordinated slices.
- Some shallow helpers will be deleted or demoted, which can cause short-term churn.

**Chosen**: Option C.

## Consequences

### Positive

- Higher leverage: one contract module can protect many adapters.
- Better locality: stream, diagnostics, tool ownership, usage, and capability bugs concentrate in
  smaller implementation areas.
- Cleaner tests: focused tests can target module interfaces instead of only large provider fixtures.
- Better AI-navigability: future agents can find the owner of each contract rule.

### Negative / costs

- The OpenAI Responses converter will need careful staged refactoring because it carries many
  provider-specific stream cases.
- Some compatibility helpers may survive temporarily until all callers move to the deepened module.
- The workstream must avoid broad public breaks unless a task explicitly proves the compatibility
  path is obsolete.

## Migration plan

Track implementation in `docs/workstreams/fearless-ai-sdk-seam-deepening/`.

The first pass should keep public behavior stable and deepen internal seams. After tests prove parity,
obsolete helpers, duplicated tests, and pass-through adapters may be deleted in the same task that
proves they are no longer useful.

## References

- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-part.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-result.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-usage.ts`
- `docs/adr/0001-vercel-aligned-modular-split.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `docs/workstreams/fearless-ai-sdk-contract-hardening/`
