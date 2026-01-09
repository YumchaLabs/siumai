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
| `@ai-sdk/azure` | Azure OpenAI provider | `siumai-provider-azure` | Owns Azure OpenAI URL + header quirks; reuses `siumai-protocol-openai` mapping. |
| `@ai-sdk/openai-compatible` | OpenAI-like family adapter | `siumai-provider-openai-compatible` + `siumai-protocol-openai` | `siumai-provider-openai-compatible` hosts the vendor preset layer (DeepSeek/OpenRouter/etc) and re-exports the protocol surface from `siumai-protocol-openai`. |
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
- OpenAI-compatible vendor presets are hosted by `siumai-provider-openai-compatible` (migrated out of `siumai-protocol-openai`).
- `siumai::tools::openai::*` defaults match Vercel fixtures for tool names (`webSearch`, `fileSearch`, `generateImage`, `codeExecution`, `MCP`, etc.).
- Google Vertex builder surface includes Vercel-aligned aliases: `language_model(...)`, `embedding_model(...)` (deprecated: `text_embedding_model(...)`).
- Google Vertex builder supports Vercel-style custom fetch via `fetch(...)` (injects `HttpTransport` for non-stream JSON requests).
- Provider-defined tool factories are available under `siumai::tools::*` (implemented in `siumai-core::tools`) and serialize to the Vercel `{ type: "provider", id, name, args }` shape; `Tool::provider_defined(id, name)` remains the escape hatch for unknown tools.

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
- Google Vertex: `imagen` (fixtures for `edit/mask/referenceImages` and Imagen 4 parameters), plus auth/headers behavior.

## Open questions (design)

- Tool factories: we expose a higher-level Rust API (`siumai::tools::<provider>::...`) while preserving the exact wire shape; the `id` is authoritative (`provider.tool_type`), and the `name` is customizable for toolName mappings.
- How do we guarantee cross-provider tool interoperability without re-introducing provider-to-provider coupling?
