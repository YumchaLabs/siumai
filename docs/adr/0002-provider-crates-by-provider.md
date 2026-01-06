# ADR-0002: Split provider implementations into provider crates (provider-first)

## Status

Accepted (incremental rollout)

## Update (beta.5+)

The historical umbrella crate `siumai-providers` is now **legacy** and removed from the workspace.
The `siumai` facade and `siumai-registry` wire provider crates directly.

Protocol mapping is being moved out of provider crates into protocol crates to reduce coupling:

- New protocol crates: `siumai-protocol-gemini`
- Legacy protocol/family crates (kept for compatibility): `siumai-provider-openai-compatible`, `siumai-provider-anthropic-compatible`

## Context

Siumai is in a fearless refactor phase and is already split into:

- `siumai` (facade)
- `siumai-core` (provider-agnostic runtime + types)
- `siumai-registry` (registry + factories)
- `siumai-extras` (orchestrator/telemetry/server/mcp utilities)
- `siumai-provider-openai` (OpenAI provider implementation; depends on the OpenAI-like protocol family crate)
- `siumai-provider-openai-compatible` (OpenAI-like protocol family crate; reused by multiple providers)
- `siumai-provider-ollama` (Ollama provider + Ollama standard)
- `siumai-provider-anthropic` (Anthropic provider implementation; depends on the Anthropic Messages protocol family crate)
- `siumai-provider-anthropic-compatible` (Anthropic Messages protocol family crate)
- `siumai-provider-gemini` (Gemini provider implementation)
- `siumai-protocol-gemini` (Gemini protocol standard)
- `siumai-provider-google-vertex` (Vertex provider implementation)
- `siumai-provider-groq` (Groq provider; OpenAI-like)
- `siumai-provider-xai` (xAI provider; OpenAI-like)
- `siumai-provider-minimaxi` (MiniMaxi provider; Anthropic chat + OpenAI-like media endpoints)

We want to further reduce coupling and compile cost by making each provider implementation
an independently buildable unit, similar in spirit to how Vercel AI SDK splits providers.

Key constraints:

- Rust is feature-gated and compile-time oriented (no tree-shaking).
- We should avoid a “crate explosion” that slows refactors, but still enforce dependency direction.
- OpenAI-compatible vendors should be configuration presets on an OpenAI-like protocol family,
  not separate heavy implementations.

## Decision

Adopt a **provider-first crate split**:

1. Keep `siumai-core` strictly provider-agnostic.
2. Create provider crates for concrete implementations:
   - `siumai-provider-openai` (OpenAI provider implementation)
   - `siumai-provider-anthropic`
   - `siumai-provider-gemini`
   - `siumai-provider-ollama`
   - `siumai-provider-groq`
   - `siumai-provider-xai`
   - `siumai-provider-minimaxi`
3. Keep `siumai` as the **only** umbrella facade for:
   - ergonomic entry points (`Siumai::builder()`, `Provider::<provider>()`)
   - optional “all-providers” feature aggregation
   - stable re-exports (`prelude::*`, `provider_ext::*`)

### OpenAI-like family reuse

OpenAI and OpenAI-compatible vendors share substantial protocol behavior.
To avoid duplicated mapping/stream parsing logic:

- The OpenAI-like protocol adapter layer lives in `siumai-provider-openai-compatible` (family/protocol crate).
- OpenAI-compatible vendors remain “presets” (base URL / headers / quirks) rather than separate crates.

This keeps naming aligned with “provider crates” while still enabling reuse.

## Options considered

### Option A — Keep a separate “openai-like family” crate (current)

Pros:
- other providers can depend on it without depending on OpenAI provider code

Cons:
- naming ambiguity (`siumai-provider-openai` vs “OpenAI-like protocol”)
- harder to explain ownership to users (“is this provider or protocol?”)

### Option B — Make OpenAI provider crate also own the OpenAI-like protocol layer (not used in beta.5+)

Pros:
- naming stays provider-first (`siumai-provider-openai` is the OpenAI crate)
- a single place to fix protocol issues that affect vendors
- aligns with Vercel AI SDK’s “OpenAI + openai-compatible vendors” mental model

Cons:
- other providers reuse implies depending on the OpenAI provider crate (acceptable during refactor);
  if this becomes problematic, we can re-extract a lighter protocol-only crate later.

## Consequences

### Positive

- Clear ownership: each provider owns its implementation, options, metadata, and mapping code.
- Better feature gating: enabling a provider pulls only that provider’s crate.
- Faster iteration: provider changes do not ripple through unrelated providers.

### Negative / costs

- Migration work: move modules and update crate features/paths.
- Potential “public surface churn”: we must keep `siumai` facade stable via re-exports and deprecations.

## Migration plan (high level)

Add a new milestone (M6) after MVP:

1. Move OpenAI implementation into `siumai-provider-openai` (keeping stable public APIs via `siumai`).
2. Extract the next provider with the least shared surface (e.g. `ollama`).
3. Repeat for remaining providers.

Keep `cargo nextest` as the safety net during each extraction.
