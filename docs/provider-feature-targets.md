# Provider Feature Targets (Alpha.5)

This document defines the **target capability matrix** for the Alpha.5 fearless refactor.
It complements `docs/provider-feature-alignment.md` (current state) by stating *where we want to go next*.

Guiding principle: **align modules first**, then iterate on field-level parity via fixtures and official API docs.

## Milestones (ordered by ROI)

### M1: “Core Trio” parity (OpenAI / Anthropic / Gemini)

Goal: make the “big three” interoperable for gateways and multi-provider apps.

Checklist: `docs/m1-core-trio-checklist.md`

Target:

- ✅ Language chat + streaming
- ✅ Function tools (tool-calls + tool-results)
- ✅ Provider-defined tools (web search / file search / code execution / MCP) via `Tool::ProviderDefined`
- ✅ Vercel-aligned v3 stream parts end-to-end for gateway use-cases:
  - tool-call / tool-result / tool-input-* / source / reasoning / finish / error
- ✅ Cross-protocol streaming transcoding for:
  - OpenAI Responses SSE
  - OpenAI Chat Completions SSE
  - Anthropic Messages SSE
  - Gemini GenerateContent SSE
- ✅ Tool-loop gateway pattern documented and supported in `siumai-extras` (keep 1 downstream SSE connection open across tool rounds)

Non-goals (M1):

- strict “lossless” mapping for every provider-specific stream edge case
- full compatibility for provider-specific non-chat endpoints

### M2: Google Vertex completeness (Gemini + Imagen)

Goal: make Vertex a first-class backend for both chat and images.

Target:

- ✅ Vertex Gemini chat + streaming parity with Gemini protocol mapping (already the design)
- ✅ Vertex Imagen:
  - generate
  - edit (mask/referenceImages/editMode)
  - response envelope / metadata parity

### M3: OpenAI-compatible vendor layer quality bar

Goal: keep “OpenAI-compatible” vendors predictable and discoverable.

Target:

- ✅ A stable preset capability view (tools/vision/embedding/rerank/image_generation/reasoning)
- ✅ Compatibility fallbacks for known vendor quirks (e.g. legacy tool-call fields)
- ◐ Optional: add a small “vendor smoke matrix” (a few golden fixtures per preset family)

### M4: Secondary providers (best-effort)

Goal: ensure “nice to have” providers don’t regress during refactor, without over-investing.

Target:

- ✅ Groq / xAI: chat + streaming + tools (OpenAI-like family)
- ✅ Ollama: chat + streaming (+ embedding)
- ✅ MiniMaxi: chat + streaming (+ speech + images)
- ✅ Cohere / TogetherAI: rerank
- ◐ Bedrock: chat + streaming + tools + rerank (auth is a separate workstream)

## Target matrix (Alpha.5)

Legend:
- ✅ required for Alpha.5 target
- ◐ best-effort / optional
- — out of scope

| Provider id | Language | Streaming | Tools | Provider tools | Vision | Embedding | Image | Rerank | Speech | Transcription | Files | Gateway readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ | ✅ | ✅ |
| `anthropic` | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | — | ✅ |
| `gemini` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | ✅ | ✅ (tool-loop) |
| `vertex` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | ✅ (tool-loop) |
| `azure` | ◐ | ◐ | ◐ | ◐ | — | ◐ | ◐ | — | ◐ | ◐ | ◐ | ◐ |
| `openai-compatible` (presets) | ✅ | ✅ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — | ◐ |
| others | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |

Notes:

- “Gateway readiness” = can expose multi-protocol SSE surfaces with acceptable fidelity (strict drop or lossy downgrade), and can optionally do tool-loop execution for providers that require next-step tool results.
- For `azure`, the target is intentionally best-effort in Alpha.5: we keep it green and aligned to existing fixtures, but we don’t treat it as a core parity driver.

## Acceptance criteria

For each ✅ cell we should have at least one of:

- a Vercel fixture parity test, or
- a deterministic mock-api test, or
- an integration test (only when fixture cannot represent the behavior).

And we must keep:

- `python scripts/audit_vercel_fixtures.py --ai-root ../ai --siumai-root .` green (no missing/drift)
