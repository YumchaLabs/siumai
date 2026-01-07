# Vercel AI SDK Alignment (siumai alpha.5)

This document tracks how `siumai` aligns (conceptually and structurally) with the Vercel AI SDK (`repo-ref/ai`), and defines an actionable checklist for ongoing refactors and fixture/test parity.

## Goals

- Reduce internal coupling by enforcing provider/protocol boundaries.
- Align crate granularity with Vercel's package layout (provider-specific crates + shared utils).
- Keep a stable public surface for downstream users while refactoring aggressively.
- Incrementally port Vercel fixture tests to prevent regressions during the alpha split.

## Package ↔ crate mapping

| Vercel AI SDK package | Role | siumai crate(s) (recommended) | Notes |
| --- | --- | --- | --- |
| `@ai-sdk/provider` | Provider interfaces/types | `siumai-core` | Traits, request/response types, streaming primitives. |
| `@ai-sdk/provider-utils` | Shared helpers (HTTP, streaming, tooling) | `siumai-core`, `siumai-registry` | `siumai-registry` owns the Vercel-style `"provider:model"` registry handle. |
| `@ai-sdk/openai` | Native OpenAI provider | `siumai-provider-openai` | Owns native OpenAI endpoints; re-exports protocol mapping under `standards`. |
| `@ai-sdk/openai-compatible` | OpenAI-like family adapter | `siumai-protocol-openai` | Preferred name; legacy compatibility crate: `siumai-provider-openai-compatible`. |
| `@ai-sdk/anthropic` | Native Anthropic provider | `siumai-provider-anthropic` | Owns native Anthropic client; re-exports protocol mapping under `standards`. |
| `@ai-sdk/google` | Google provider | `siumai-provider-gemini` | Gemini (GenerateContent) mapping + client implementation. |
| `@ai-sdk/google-vertex` (conceptually) | Vertex AI provider | `siumai-provider-google-vertex` | Provider-owned standards live under this crate for now. |

## Naming conventions (Rust crates)

- **Protocol crates**: `siumai-protocol-<family>` (protocol mapping + protocol-owned metadata)
  - Current protocol family baseline: `openai`, `anthropic`, `gemini`
  - Stable facade imports: `siumai::protocol::{openai, anthropic, gemini}`
- **Provider crates**: `siumai-provider-<vendor>` (native client implementations)
  - Provider crates may depend on one or more protocol crates, but **should not** depend on other provider crates.
- **Legacy compatibility**: keep `siumai-provider-*-compatible` as transitional crate names until downstream migration is complete.

## Current status (alpha.5)

- Protocol facade is available at `siumai::protocol::*` to keep downstream imports stable.
- `siumai-protocol-openai` exists as the preferred OpenAI-like protocol crate name (now the real implementation).
- `siumai-protocol-anthropic` exists as the preferred Anthropic protocol crate name (now the real implementation).
- OpenAI-like dependent providers (`groq`, `xai`, `minimaxi`, and `siumai-registry`) are migrated to `siumai-protocol-openai`.

## Fixture/test parity checklist

### 1) Identify Vercel fixtures we want

Target categories (ordered by ROI):

1. **Request serialization**: JSON body + headers for chat/streaming/tool calling.
2. **Response parsing**: normal + streaming variants.
3. **Error mapping**: status codes and provider-specific error bodies.
4. **Tool surface**: provider-defined tools (e.g. web search).

### 2) Port strategy (Rust)

- Prefer `wiremock` for HTTP fixtures and deterministic assertions.
- Keep a one-to-one mapping between Vercel fixtures and Rust tests when possible.
- Place provider-specific tests in the provider crate, and protocol-only tests in the protocol crate.

### 3) Planned ports

- OpenAI: `chat`, `tools`, `streaming`, `embeddings`, `images` (where applicable).
- Anthropic: `messages`, `tools`, `streaming`, `thinking`, `prompt caching` (protocol-level where applicable).
- Gemini: `generateContent`, `streaming`, `tools`, `grounding`.
- Google Vertex: `imagen` (`edit/mask/referenceImages`) and auth/headers behavior.

## Open questions (design)

- Should tool factories mirror Vercel's `Tool::provider_defined("openai.web_search", "web_search")` exactly, or should we expose a higher-level Rust API that still serializes to the same wire format?
- How do we guarantee cross-provider tool interoperability without re-introducing provider-to-provider coupling?
