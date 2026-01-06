# Provider Granularity Alignment (Vercel AI SDK)

This document describes how `siumai` maps to the Vercel AI SDK package granularity and what we
intend to split next to reduce internal coupling during the alpha/beta fearless refactor.

## Why this exists

We already split “providers” into separate crates (`0.11.0-beta.5`), but some crates still mix two
concerns:

1. **Provider implementation** (HTTP client, auth, builder, middleware, runtime wiring)
2. **Protocol / standard mapping** (request/response schema, streaming event conversion, adapters)

In the Vercel AI SDK, those concerns are separated by a “provider utils” layer and provider packages
that tend to be protocol-focused. To keep Rust builds maintainable and reduce internal coupling, we
mirror that direction using **protocol crates** that provider crates depend on.

## Current mapping (beta.5)

| Vercel AI SDK | Meaning | siumai (today) |
| --- | --- | --- |
| `@ai-sdk/provider` | shared interfaces + types | `siumai-core::types`, `siumai-core::traits` |
| `@ai-sdk/provider-utils` | HTTP/streaming/retry utilities | `siumai-core::execution`, `siumai-core::retry`, `siumai-core::streaming` |
| `@ai-sdk/openai` | OpenAI provider | `siumai-provider-openai` |
| `@ai-sdk/anthropic` | Anthropic provider | `siumai-provider-anthropic` |
| `@ai-sdk/google` | Gemini API (Generative Language API) | `siumai-provider-gemini` |
| `@ai-sdk/google-vertex` | Vertex AI (incl. Imagen) | `siumai-provider-google-vertex` |
| `@ai-sdk/groq` | Groq (OpenAI-like) | `siumai-provider-groq` (reuses OpenAI-like family) |
| `@ai-sdk/xai` | xAI (OpenAI-like) | `siumai-provider-xai` (reuses OpenAI-like family) |

Notes:

- `siumai` is the facade crate (feature aggregation + stable preludes).
- `siumai-registry` owns the registry abstractions and optional built-in provider wiring.
- Some protocols are already factored into protocol crates:
  - Gemini: `siumai-protocol-gemini`
  - Vertex (Imagen): `siumai-protocol-vertex`
- Some protocols are factored as “family crates” (protocol crates with legacy names):
  - OpenAI-like: `siumai-provider-openai-compatible`
  - Anthropic Messages: `siumai-provider-anthropic-compatible`

## Target split: provider vs protocol

### Target rule (dependency direction)

`protocol crates` depend on `siumai-core`, but never on provider crates.

```text
siumai-core  <-  siumai-protocol-<x>  <-  siumai-provider-<x>  <-  siumai / siumai-registry
```

### What moves into protocol crates

Protocol crates should own:

- typed request/response schemas (serde structs/enums)
- request conversion (`ChatMessage`/`Tool` -> provider JSON)
- response parsing into unified types
- streaming event conversion (SSE chunk -> `ChatStreamEvent`)
- protocol-local helpers (headers building, URL/mime helpers)

Protocol crates must NOT own:

- HTTP clients / endpoints routing
- auth/token provider wiring
- provider-specific typed `providerOptions` / `providerMetadata` “nice” APIs
- registry factory / middleware injection

### Compatibility strategy (during beta)

To avoid breaking imports, provider crates should **re-export** the protocol module tree under the
existing paths (e.g. keep `siumai_provider_gemini::standards::gemini::*` working).

## Planned protocol crates (incremental)

Start with the highest-coupling provider first:

1. `siumai-protocol-gemini` (done)
2. `siumai-protocol-vertex` (done)
3. `siumai-protocol-ollama` (if/when Ollama standard grows beyond the provider implementation)
4. OpenAI-like and Anthropic Messages are already split as family crates; we can optionally rename
   them later, but renaming is a breaking/coordination cost and is not required for the refactor.

## Migration checklist (for each provider)

- [ ] Ensure protocol code does not depend on provider-owned typed options/metadata modules.
- [ ] Replace typed option parsing with JSON-based parsing from `providerOptions[provider_id]`.
- [ ] Keep provider-owned extension traits (typed helpers) in the provider crate.
- [ ] Re-export protocol modules from the provider crate to preserve paths.
- [ ] Run `cargo fmt` and `cargo nextest run` for the relevant feature set.
