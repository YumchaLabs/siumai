# ADR-0001: Vercel-aligned modular split (adapted for Rust)

## Status

Accepted

## Context

Siumai is in a fearless refactor phase (currently `0.11.0-beta.5`) and already split into:

- `siumai` (facade)
- `siumai-core` (types/runtime)
- `siumai-registry` (registry handles + factories)
- `siumai-extras` (orchestrator/telemetry/server/mcp utilities)

However, coupling remains high because `siumai-core` still contains provider-specific logic:

- (historical) protocol/compat mapping layers under `siumai-core/src/standards/*` (migrated into provider-owned crates/modules; core may still host protocol-level shared building blocks)
- provider-hosted tool factories under `siumai-core/src/hosted_tools/*`
- (historical) provider-specific option structs and metadata types under `siumai-core/src/types/provider_options/*` and `siumai-core/src/types/provider_metadata/*`
- (historical) a closed `ProviderOptions` enum transport that forced core changes when providers/features evolved

In beta.5, typed `providerOptions`/`providerMetadata` were moved to provider crates and the legacy closed
`ProviderOptions` enum transport was removed in favor of an open `provider_options_map`.

This makes compilation heavier, blurs ownership, and increases the cost of adding or evolving providers.

We want a split similar in *spirit* to the Vercel AI SDK:

- a small “provider interface” layer
- a shared “provider utils / protocol helpers” layer
- provider packages that own protocol details and typed provider options
- openai-compatible vendors should reuse a shared OpenAI-like protocol adapter rather than duplicating logic

At the same time, we are in Rust (not JS/TS), so we must optimize for:

- compile time and feature gating
- clear ownership and dependency direction
- avoiding an explosion of crates too early (keep migration incremental)

## Decision

Adopt a **Vercel-aligned layered architecture**, adapted for Rust:

1. **Interface & shared types (thin, stable)**
   - Keep the stable “6 model families” surface small and explicit.
   - Treat provider-specific features as extensions (hosted tools, provider options, provider_ext modules).
   - Replace “closed provider options enum” with an **open, provider-id keyed options map** (pass-through).

2. **Provider runtime utilities / protocol helpers**
   - Move protocol mapping and reusable parsing helpers out of the provider-agnostic “core types” layer.
   - Consolidate SSE parsing, retry, error mapping scaffolding, tool-call streaming helpers, etc. into a shared layer.

3. **Provider implementations own provider-specific details**
   - Provider-specific option structs (typed) live with the provider.
   - Provider-specific metadata types live with the provider.
   - “OpenAI-like” protocol support is shared (as a dedicated *family* crate, used by multiple providers).

4. **Registry stays optional for built-ins**
   - Registry should provide abstractions without forcing built-in providers unless enabled.
   - Built-ins are behind explicit features (already partially done via `siumai-registry` `builtins`).

This ADR does **not** require immediately creating many new crates. The MVP can be achieved by:

- introducing the open `provider_options` map in `siumai-core` requests
- moving typed provider options and protocol mapping modules from `siumai-core` into provider crates (provider-owned)
- tightening re-exports and public entry points

### OpenAI vs OpenAI-compatible (pragmatic constraint)

We avoid splitting OpenAI and OpenAI-compatible into separate published crates in the MVP.
Instead, we extract the shared OpenAI-like protocol layer into a single crate (which can later
also host the OpenAI provider implementation as we move toward provider-first crates):

- `siumai-provider-openai`

This crate owns the OpenAI-like “standard” (request/response mapping + streaming/tool-call helpers).

OpenAI-compatible vendors are treated as configuration entries (base URL / headers / error structure),
not as first-class provider crates, mirroring the Vercel AI SDK approach.

## Options considered

### Option A — Keep current split, only internal refactors

Pros:
- minimal churn
- least risk of breaking changes

Cons:
- `siumai-core` remains a “god crate” with provider coupling
- adding provider-specific features keeps touching core
- hard to enforce ownership boundaries long-term

### Option B — TanStack AI-style “core owns standards, providers are thin adapters”

Pros:
- very ergonomic API design by “activities” (capabilities)
- clear separation between “what you do” (activities) and “how you do it” (adapters)

Cons in Rust:
- TanStack relies heavily on tree-shaking; Rust relies on feature gating and crate boundaries
- likely to keep core large (which is the current pain point)
- less suitable when many providers share protocol layers (e.g., OpenAI-like vendors)

### Option C — Pure Vercel AI SDK clone (many packages/crates)

Pros:
- strongest enforcement of dependency direction
- excellent reuse for openai-compatible vendors

Cons:
- crate explosion risk during a refactor
- requires careful migration to avoid breaking downstream users

### Option D — Hybrid: Vercel-aligned layers + TanStack-style capability composition

Pros:
- best of both worlds
- providers can be implemented per-capability (chat/embed/image/tts/stt/rerank)
- protocol reuse remains possible (openai-like shared adapter)

Cons:
- needs discipline and conventions to avoid “two architectures at once”

**Chosen**: Option D (hybrid), implemented incrementally in a Vercel-aligned dependency direction.

## Consequences

### Positive

- Reduced coupling: provider-specific logic no longer lives in the provider-agnostic core layer.
- Faster iteration: adding provider-specific features does not require modifying core enums/types.
- Better ownership: each provider owns its typed options, metadata, and protocol mapping.
- Clearer extensibility: openai-compatible vendors reuse a shared OpenAI-like adapter.

### Negative / costs

- Migration work to move modules and adjust imports.
- Some breaking changes may be needed (especially around provider options).
- Temporary compatibility layers (dual representations) may be required during transition.

## Migration plan (high level)

See `docs/roadmap/roadmap-mvp.md` for the step-by-step milestones and acceptance criteria.

## References

- Vercel AI SDK split: `repo-ref/ai/packages/provider/package.json`, `repo-ref/ai/packages/provider-utils/package.json`, `repo-ref/ai/packages/openai-compatible/package.json`
- TanStack AI core/adapters: `repo-ref/tanstack_ai/packages/typescript/ai/package.json`, `repo-ref/tanstack_ai/packages/typescript/ai/src/activities/index.ts`
