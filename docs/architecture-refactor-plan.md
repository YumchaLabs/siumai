# Architecture Refactor Plan (Fearless Refactor)

This document is a working plan for the ongoing split-crate refactor (currently `0.11.0-beta.5`).
It is intentionally pragmatic: it focuses on reducing coupling and improving maintainability while
keeping the *user-facing* surface aligned with the Vercel AI SDK philosophy.

## Goals

- **Keep the stable surface small and consistent**: 6 model families only
  - Language (chat + streaming)
  - Embedding
  - Image
  - Rerank
  - Speech (TTS)
  - Transcription (STT)
- **Make provider-specific features explicitly opt-in**:
  - provider-hosted tools (`hosted_tools::*`)
  - provider options (pass-through `providerOptions`)
  - provider extension modules (`provider_ext::<provider>::*`)
- **Reduce internal coupling** by enforcing clear crate/module ownership:
  - Provider-agnostic types do not depend on provider protocols.
  - Protocol adapters do not leak into the unified surface.
  - Registry does not need built-in provider implementations unless requested.

## Current Workspace Layout (beta.5)

- `siumai` (facade): recommended entry for most users; exports prelude and feature flags.
- `siumai-core` (core/runtime): types, traits, execution runtime, retry, and streaming normalization (provider-agnostic).
- `siumai-registry` (registry): provider factories, registry handles; optional built-ins behind `builtins`.
- `siumai-registry` can be used as a pure abstraction layer; see `docs/registry-without-builtins.md`.
- `siumai-extras` (extras): orchestrator, schema helpers, telemetry, OpenTelemetry, server adapters, MCP.

Provider crates (feature-gated):

- `siumai-provider-openai` (OpenAI provider)
- `siumai-provider-openai-compatible` (OpenAI-like protocol standard shared by multiple providers)
- `siumai-provider-anthropic` (Anthropic provider)
- `siumai-provider-anthropic-compatible` (Anthropic Messages protocol standard shared by multiple providers)
- `siumai-provider-gemini`
- `siumai-provider-ollama`
- `siumai-provider-groq`
- `siumai-provider-xai`
- `siumai-provider-minimaxi`

This split is already a big step forward. The next step is **tightening boundaries** so that the
split *actually* reduces coupling and compilation cost.

## Key Coupling Problems (Observed)

1. **Provider-specific typed options/metadata must not live in `siumai-core`**
   - Historically, typed provider options/metadata lived in `siumai-core`, which forced core changes
     whenever providers evolved and increased compile cost.
   - In beta.5, typed `providerOptions` and typed `providerMetadata` were moved to provider crates and
     exposed via `siumai::provider_ext::<provider>::*`.

   `siumai-core` now only owns provider-agnostic transports:
   - `ProviderOptionsMap` (open JSON map keyed by provider id)
   - `CustomProviderOptions` (trait for converting to a `(provider_id, JSON)` entry)

2. **Provider options must be open (no closed enum transport)**
   - The legacy closed `ProviderOptions` enum transport has been removed (breaking change).
   - Providers parse options from `request.provider_options_map["<provider_id>"]`.

3. **Crate boundaries are blurred by blanket re-exports**
   - Provider crates and `siumai-registry` can accidentally re-export large portions of `siumai-core`.

   This makes it easy for internal modules and downstream users to accidentally “reach across layers”
   and increases migration cost when splitting further.

## Target Architecture (Vercel-aligned)

Think in “interfaces + runtime + providers”:

1. **Provider interface layer** (Vercel: `@ai-sdk/provider`)
   - Versioned model interfaces and shared types
   - Pass-through `providerOptions` and `providerMetadata` as `Map<provider_id, JSON object>`

2. **Provider runtime utilities** (Vercel: `@ai-sdk/provider-utils`)
   - HTTP helpers, retry, SSE parsing, stream framing, tracing hooks

3. **Provider packages**
   - Own provider-specific option structs + helpers
   - Own protocol mapping, endpoints, and extension APIs

4. **Facade + registry**
   - Facade: “nice” user API, unified prelude, feature aggregation
   - Registry: model resolution and caching, optional built-ins

In Rust, we can achieve the same outcome either via more crates (fine) or via stricter module
boundaries inside `siumai-core` + provider crates (also fine). The important part is **ownership**.

## Migration Plan (Incremental)

### Phase 1 — Introduce pass-through provider options (done)

- Requests carry `provider_options_map: ProviderOptionsMap`.
- Request types expose helpers like `with_provider_option(provider_id, json_value)`.
- Providers parse options from the map (no legacy enum fallback).

Outcome: provider-specific features evolve without editing core enums/types.

### Phase 2 — Move typed provider options/metadata out of `siumai-core` (done)

- Relocate typed provider option structs to provider-owned code (provider crates are preferred, e.g. `siumai-provider-openai`).
- Expose them via stable facade paths (e.g. `siumai::provider_ext::<provider>::*`) behind provider features.
- Deprecate/remove provider-specific request builder methods in `siumai-core` once the extension traits exist.

Outcome: `siumai-core` becomes truly provider-agnostic (types & runtime).

### Phase 3 — Extract protocol adapters / “standards” ownership

Decide one of:

- **Option A**: Keep `standards::<provider>` in provider crates (provider-owned).
- **Option B**: Create a dedicated “protocols” crate that providers depend on (shared, still not core).

Outcome: core compiles fast, providers own protocols.

### Phase 4 — Tighten public exports and enforce boundaries

- Avoid blanket re-exports from provider crates / `siumai-registry`.
- Keep public, stable entry points in:
  - `siumai::prelude::unified::*`
  - `siumai::prelude::extensions::*`
  - `siumai::provider_ext::<provider>::*`

Outcome: smaller API surface, fewer accidental dependencies.

## Non-goals

- No attempt to provide a UI/message framework like Vercel’s React/RSC packages.
- No “automatic” capability gating that blocks calls; capability info remains advisory.
