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
- `siumai-providers` (providers): built-in provider implementations and builders.
- `siumai-registry` (registry): provider factories, registry handles; optional built-ins behind `builtins`.
- `siumai-registry` can be used as a pure abstraction layer; see `docs/registry-without-builtins.md`.
- `siumai-extras` (extras): orchestrator, schema helpers, telemetry, OpenTelemetry, server adapters, MCP.

This split is already a big step forward. The next step is **tightening boundaries** so that the
split *actually* reduces coupling and compilation cost.

## Key Coupling Problems (Observed)

1. **Provider-specific types live in `siumai-core`**
   - `types::provider_options::*` contains provider-specific option structs and convenience APIs.
   - `types::provider_metadata::*` contains provider-specific metadata types.
   - `hosted_tools::*` contains provider-specific logic.

   This makes `siumai-core` “know” every provider even when a user wants a minimal build.
   As part of the beta.5 refactor, OpenAI typed options/metadata were moved out of `siumai-core`
   as the first “proof point” of provider-owned extensions.

2. **Provider options are modeled as a closed enum**
   - `ProviderOptions` enumerates OpenAI/Anthropic/Gemini/… variants.

   This is the opposite of the Vercel AI SDK strategy, where provider options are a **pass-through map**
   keyed by provider id (e.g. `Record<string, JSONObject>`), enabling new provider features without
   touching the core package.

3. **Crate boundaries are blurred by blanket re-exports**
   - `siumai-providers` and `siumai-registry` currently re-export large portions of `siumai-core`.

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
boundaries inside `siumai-core` + `siumai-providers` (also fine). The important part is **ownership**.

## Migration Plan (Incremental)

### Phase 1 — Introduce pass-through provider options (compat)

- Add a provider-agnostic map type:
  - `ProviderOptionsMap = HashMap<String, serde_json::Map<String, Value>>` (or equivalent)
- Add it to request types as an additional field (alongside the existing enum for now).
- Add helpers to set/get provider options by provider id.
- Providers read from the map first, then fall back to legacy enum (temporary bridge).

Outcome: provider-specific features can be shipped without editing the core enum.

### Phase 2 — Move provider-specific option structs out of `siumai-core`

- Relocate typed provider option structs to provider-owned code (provider crates are preferred, e.g. `siumai-provider-openai`).
- Expose them via stable facade paths (e.g. `siumai::provider_ext::<provider>::*`) behind provider features.
- Deprecate/remove provider-specific request builder methods in `siumai-core` once the extension traits exist.

Outcome: `siumai-core` becomes truly provider-agnostic (types & runtime).

### Phase 3 — Extract protocol adapters / “standards” ownership

Decide one of:

- **Option A**: Move `standards::<provider>` into `siumai-providers` (provider-owned).
- **Option B**: Create a dedicated “protocols” crate that providers depend on (shared, still not core).

Outcome: core compiles fast, providers own protocols.

### Phase 4 — Tighten public exports and enforce boundaries

- Avoid blanket re-exports from `siumai-providers` / `siumai-registry`.
- Keep public, stable entry points in:
  - `siumai::prelude::unified::*`
  - `siumai::prelude::extensions::*`
  - `siumai::provider_ext::<provider>::*`

Outcome: smaller API surface, fewer accidental dependencies.

## Non-goals

- No attempt to provide a UI/message framework like Vercel’s React/RSC packages.
- No “automatic” capability gating that blocks calls; capability info remains advisory.
