# Provider Split Alignment (Vercel AI SDK)

This document describes how `siumai` maps to the Vercel AI SDK package layout, and what we are
doing during the fearless refactor (alpha/beta split phase) to reduce coupling.

## Goals

- Make provider crates own only their provider-specific wiring (config, auth, base URLs, headers, tools).
- Keep protocol mapping (request/response + streaming conversion) in protocol crates.
- Avoid cross-provider code in a provider crate (e.g. Vertex-specific providers should live under the Vertex crate).
- Keep a stable facade surface in `siumai` while crates move internally.

## Mapping: Vercel -> siumai

- `@ai-sdk/provider` / `@ai-sdk/provider-utils`
  - -> `siumai-core` (runtime + types + executors + streaming + retry + interceptors)

- `@ai-sdk/openai`
  - -> `siumai-provider-openai` (OpenAI provider wiring)
  - -> `siumai-protocol-openai` (OpenAI(-like) protocol mapping, incl. Responses API)

- `@ai-sdk/openai-compatible`
  - -> `siumai-provider-openai-compatible` (vendor presets + adapter registry + routing)
  - -> `siumai-protocol-openai` (shared OpenAI-like protocol mapping)

- `@ai-sdk/anthropic`
  - -> `siumai-provider-anthropic` (Anthropic provider wiring)
  - -> `siumai-protocol-anthropic` (Messages protocol mapping)

- `@ai-sdk/google` (Gemini API)
  - -> `siumai-provider-gemini`
  - -> `siumai-protocol-gemini`

- `@ai-sdk/google-vertex`
  - -> `siumai-provider-google-vertex` (Vertex chat/embedding/imagen wiring)
  - -> `siumai-protocol-gemini` (Gemini protocol mapping reused for Vertex chat)
  - -> `siumai-protocol-anthropic` (optional: Anthropic on Vertex)

## Current Refactor Actions

### 1) De-duplicate OpenAI-compatible implementation

Problem:
- `siumai-provider-openai` historically hosted an `openai_compatible` module.
- `siumai-provider-openai-compatible` also hosts the same implementation.
- This duplication increases coupling and creates two sources of truth.

Action:
- Keep OpenAI-compatible providers **only** in `siumai-provider-openai-compatible`.
- `siumai-provider-openai` may depend on it for convenience methods, but does not own the implementation.

Status: Completed (beta.5 split phase).

### 2) Move Anthropic@Vertex under the Vertex provider crate

Problem:
- `siumai-provider-anthropic` contained a Vertex-specific provider (`anthropic_vertex`).
- This couples a provider crate to another provider's deployment environment.

Action:
- Move `Anthropic@Vertex` provider implementation to `siumai-provider-google-vertex::providers::anthropic_vertex`.
- Keep `siumai::provider_ext::anthropic_vertex` as a facade re-export for now.

Status: Completed (beta.5 split phase).

## Non-goals (for this iteration)

- Renaming crates (e.g. removing `*-compatible` legacy names) - deferred.
- Removing the temporary `pub use siumai_core::{...}` re-export patterns - deferred until module paths are stabilized.

