# AI SDK Provider Interface Convergence - Design

Status: Active
Last updated: 2026-05-18

## Why This Lane Exists

Siumai has already moved through the V4 fearless refactor, provider package alignment, shared
metadata cleanup, completion-family alignment, upload helpers, and stream-part hardening. The
remaining risk is no longer a single misplaced module. The risk is architectural drift returning in
small pieces:

- a provider feature lands only on a compatibility `LlmClient` path,
- a provider package inherits capability from the shared OpenAI-compatible runtime that the upstream
  AI SDK package does not expose,
- protocol parsers emit major stream semantics through raw/custom escape hatches instead of the
  stable part lane,
- `siumai-core` grows provider-specific helpers again,
- old workstream notes remain useful but are not machine-trackable enough to guide the next cuts.

This lane is a program workstream. It coordinates the next fearless-refactor phase and splits code
changes into bounded vertical slices.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- Existing docs:
  - `docs/architecture/module-split-design.md`
  - `docs/workstreams/ai-sdk-structural-alignment/design.md`
  - `docs/workstreams/ai-sdk-structural-alignment/data-structure-matrix.md`
  - `docs/workstreams/fearless-refactor-v4/follow-ons.md`
  - `docs/workstreams/INDEX.md`
- AI SDK reference packages:
  - `repo-ref/ai/packages/provider`
  - `repo-ref/ai/packages/provider-utils`
  - `repo-ref/ai/packages/ai`
  - `repo-ref/ai/packages/openai`
  - `repo-ref/ai/packages/openai-compatible`
  - `repo-ref/ai/packages/anthropic`
  - `repo-ref/ai/packages/google`
  - `repo-ref/ai/packages/google-vertex`
  - `repo-ref/ai/packages/amazon-bedrock`
  - `repo-ref/ai/packages/{azure,cohere,deepinfra,deepseek,fireworks,groq,mistral,moonshotai,perplexity,togetherai,xai}`

## Problem

Siumai's target architecture is clear, but the enforcement surface is distributed across historical
workstreams. Future provider work needs one authoritative program that answers:

1. which crate owns each interface,
2. which compatibility paths are allowed,
3. which AI SDK semantic slots still need parity,
4. which provider-package differences are intentional,
5. which no-network gates prove each slice.

Without that program, the project can pass tests while slowly regressing toward the old broad-client
architecture.

## Target State

When this lane closes:

- `siumai-spec` is locked as serializable provider-agnostic contract space.
- `siumai-core` is locked as provider-agnostic runtime space.
- `siumai-registry` construction is family-model-first for stable families, with compatibility
  clients isolated behind explicit compat/extension seams.
- protocol crates own wire conversion and protocol stream state machines.
- provider crates own typed options, typed metadata helpers, provider-specific behavior, and
  package-surface parity.
- OpenAI-compatible vendors reuse the shared adapter only for capabilities their AI SDK package
  actually exposes or Siumai intentionally documents.
- stream semantics prefer `ChatStreamEvent::Part(ChatStreamPart)` for AI SDK-stable semantics.
- provider-by-provider parity is tracked in a matrix that can spawn child workstreams without
  reopening the whole program.

## In Scope

- Core/spec/registry/protocol/provider seam audits.
- Source guards that prevent provider-specific behavior from re-entering provider-agnostic crates.
- Registry/factory migration away from primary stable-family execution through `LlmClient`.
- Stream parser/serializer convergence on the stable part lane.
- OpenAI-compatible vendor capability and typed-option parity checks.
- Provider-package parity inventory against `repo-ref/ai`.
- Workstream status hygiene for child lanes that this program depends on.

## Out Of Scope

- A single mega PR touching every provider.
- Mechanical file moves that do not remove coupling or clarify ownership.
- Renaming public Rust APIs only to mimic TypeScript names.
- Removing user-facing compatibility APIs without a migration note or explicit removal version.
- Implementing frontend-only AI SDK surfaces such as React hooks.
- Adding fake provider packages for AI SDK packages whose runtime semantics Siumai does not own.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Family-model-first is still the architectural center. | High | ADR-0006 and ADR-0007 are accepted. | Reopen ADR-0006/0007 before doing code migration. |
| The next broad refactor should be a program lane, not one execution lane. | High | `fearless-refactor-v4/follow-ons.md` says not to reopen V4 for mechanical cleanup. | Split the program into a narrower child workstream. |
| AI SDK semantic parity matters more than exact package/folder mirroring. | High | ADR-0001 explicitly chooses a Rust-adapted Vercel-aligned split. | Add an ADR if a closer package mirror becomes necessary. |
| `ai-sdk-structural-alignment` remains useful but is not a clean active workstream. | High | `docs/workstreams/INDEX.md` marks it `unknown` and it lacks `WORKSTREAM.json`. | Normalize that lane instead of using this program as the index. |
| Remaining provider work is uneven by package. | High | Existing provider package workstreams and matrix notes show mixed closed/unknown state. | Build the inventory before assigning implementation slices. |

## Architecture Direction

The intended dependency direction remains:

```text
siumai facade
  -> siumai-registry / siumai-bridge / siumai-extras
  -> siumai-provider-* / siumai-protocol-*
  -> siumai-core
  -> siumai-spec
```

The intended execution direction for stable model families is:

```text
public helper or registry handle
  -> family model trait
  -> provider-owned implementation
  -> protocol adapter / HTTP execution
```

The compatibility direction remains allowed only when named as compatibility:

```text
compat entry point
  -> LlmClient
  -> optional capability downcast
  -> provider implementation
```

This program treats seams as enforceable interfaces:

- **Spec seam:** stable JSON/data contracts, provider maps, prompt/result/usage/warning/content
  structures.
- **Core seam:** runtime traits, errors, HTTP abstractions, middleware, stream processing, tooling
  primitives.
- **Registry seam:** lookup, factories, handles, caching, build context, compat adapters.
- **Protocol seam:** wire conversion, exact parser/serializer behavior, protocol stream machines.
- **Provider seam:** package-specific settings, typed options, typed metadata helpers, provider
  defaults, hosted tools, resources, and public provider extension modules.

## Execution Strategy

1. Keep this lane as the authoritative program ledger.
2. Execute one vertical slice at a time.
3. Prefer source guards before broad moves.
4. Open child workstreams only when a provider or seam needs its own durable lane.
5. Commit completed milestones with conventional commits once gates pass.

## Closeout Condition

This lane can close when:

- the seam inventory and provider parity matrix are complete enough to guide future work,
- core/spec/registry/protocol/provider guards cover the highest-risk regressions,
- stable-family registry execution no longer depends on generic-client downcasts except documented
  compatibility or extension lanes,
- stream stable-part migration gaps are either closed or split into child lanes,
- provider-package parity gaps are either closed, intentionally deferred, or tracked in child
  workstreams,
- evidence gates are recorded,
- and `WORKSTREAM.json` plus `HANDOFF.md` reflect the final state.
