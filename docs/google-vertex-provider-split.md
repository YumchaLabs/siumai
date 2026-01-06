# Google Vertex Provider Split (Vercel-aligned)

## Context

In Vercel AI SDK, the Google provider surface is split into:

- `@ai-sdk/google` (Gemini API)
- `@ai-sdk/google-vertex` (Vertex AI, including Imagen)

In `siumai` today (`0.11.0-beta.5`), Vertex-specific image support (Imagen via `:predict`) is routed
through the Gemini provider (`siumai-provider-gemini`) using the base URL heuristic
(`aiplatform.googleapis.com`) and model prefix (`imagen-*`).

This creates avoidable coupling:

- The Gemini crate contains Vertex-only protocol mapping.
- The unified provider id (`gemini`) becomes overloaded (Gemini API vs Vertex AI).
- Fixture alignment with Vercel is harder because “provider identity” is ambiguous.

This document proposes a split that matches Vercel’s granularity while keeping Rust ergonomics.

## Goals

- Make Vertex AI a separate provider package/crate.
- Keep `siumai-core` provider-agnostic.
- Reduce cross-module coupling by enforcing ownership:
  - Vertex Imagen request/response mapping lives in the Vertex provider crate.
  - Gemini API mapping lives in the Gemini provider crate.
- Keep fixtures and tests provider-scoped and reusable.

## Non-goals

- Moving every Vertex-hosted model (e.g., Anthropic on Vertex) immediately.
- Perfect API stability during the alpha/beta refactor phase.

## Proposed crate split

### New crate: `siumai-provider-google-vertex`

Owns:

- Vertex AI routing/spec for `:predict` family endpoints.
- Vertex Imagen mapping (`models/{imagen-...}:predict`).
- Vertex-specific typed provider options / helpers (future).

Public surface (facade exports):

- `siumai::provider_ext::google_vertex::*` (client + config)
- `siumai::experimental::providers::google_vertex::*` (internals)

### Existing crate: `siumai-provider-gemini`

Owns:

- Gemini API mapping (`generativelanguage.googleapis.com`).
- Gemini provider-specific options/metadata and hosted tools.

Stops owning:

- Vertex Imagen mapping and routing heuristics (moved to the Vertex crate).

## Provider identity & providerOptions key

Vercel’s `providerOptions` key for Google Vertex is `vertex` (even though the model provider string
is `google-vertex`). In `siumai`, `providerOptions` is defined as a provider-id keyed map.

To minimize ambiguity and align fixtures with Vercel, we use:

- Provider id: `vertex`
- Facade module path: `google_vertex`

Registry aliases can map:

- `"google-vertex"` -> `"vertex"`

## Migration plan (incremental)

1. Introduce `siumai-provider-google-vertex` with the Vertex Imagen standard + a minimal image client.
2. Update fixtures/tests to be routed through the Vertex provider (instead of Gemini).
3. Update examples/docs:
   - Move `vertex_imagen_edit` example under a Vertex feature flag.
4. Remove Vertex Imagen routing from `GeminiSpec` once downstream migration is complete.

## Test strategy

- Keep fixture-driven semantic tests for request/response mapping:
  - `siumai/tests/*_fixtures_alignment_test.rs`
- Add executor-level end-to-end tests to validate:
  - `warnings` for unsupported settings (e.g., `size`)
  - response envelope (`timestamp`, `modelId`, response headers)

