# Gemini Official API Alignment (GenerateContent + Streaming)

This document records **official API correctness checks** for the Gemini (Google Generative Language) API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/core-trio-module-alignment.md` (module mapping for the core trio)

## Sources (Google AI for Developers)

- Text generation (REST): <https://ai.google.dev/gemini-api/docs/text-generation?lang=rest>
- API overview: <https://ai.google.dev/gemini-api/docs/api-overview>
- API reference (GenerateContent + StreamGenerateContent): <https://ai.google.dev/api/generate-content>

## Local access note (contributors)

In some regions, `ai.google.dev` may be unstable.
This repo’s contributors commonly use a local HTTP proxy:

```bash
set HTTP_PROXY=http://127.0.0.1:10809
set HTTPS_PROXY=http://127.0.0.1:10809
```

## Siumai implementation (where to compare)

- Protocol mapping:
  - `siumai-protocol-gemini/src/standards/gemini/headers.rs`
  - `siumai-protocol-gemini/src/standards/gemini/chat.rs`
  - `siumai-protocol-gemini/src/standards/gemini/streaming.rs`
  - `siumai-protocol-gemini/src/standards/gemini/transformers.rs`
  - `siumai-protocol-gemini/src/standards/gemini/types/*`
- Provider wiring:
  - `siumai-provider-gemini/src/providers/gemini/*`
- Local OpenAPI schema reference (unofficial but helpful for shapes):
  - `docs/gemini_OPENAPI3_0.json`

## Official endpoint + required headers

From the REST examples in the official docs:

- Non-stream endpoint:
  - `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- Stream endpoint (SSE):
  - `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse`
- Required headers (official examples):
  - `x-goog-api-key: $GEMINI_API_KEY`
  - `content-type: application/json`

### Siumai mapping

- Default base URL: `https://generativelanguage.googleapis.com/v1beta`
  - `siumai-protocol-gemini/src/standards/gemini/types/config.rs`
- URL builders:
  - `siumai-protocol-gemini/src/standards/gemini/chat.rs`
- Header builder:
  - `siumai-protocol-gemini/src/standards/gemini/headers.rs`
  - Rule: if a user provides `Authorization: Bearer ...` in custom headers, Siumai does **not** inject `x-goog-api-key`.

## Request body (GenerateContentRequest)

The official docs describe (and the API reference expands) a request body centered around:

- `contents[]` with multi-part content (text, inline data, etc.)
- Optional generation config (e.g. temperature/max output tokens)
- Optional safety settings
- Optional tools/function calling (`tools`, `toolConfig`)

### Siumai mapping

- Unified request type: `siumai-core/src/types/chat.rs` (`ChatRequest`)
- Gemini transformer:
  - `siumai-protocol-gemini/src/standards/gemini/transformers.rs`
- Fixtures:
  - `siumai/tests/fixtures/google_generative_ai/*`

## Streaming protocol (SSE)

From the official REST docs:

- Streaming uses SSE frames (`text/event-stream`) with `data: { ... }` JSON payloads.
- The stream endpoint is the same request shape as non-stream, with a different method suffix (`:streamGenerateContent`).
- The server terminates the stream by closing the connection (no standardized `[DONE]` marker is documented).

### Siumai mapping

- Parser/bridge:
  - `siumai-protocol-gemini/src/standards/gemini/streaming.rs`
- Behavior:
  - Accepts `data: { ... }` frames and converts them into unified stream parts.
  - Ignores empty `data:` frames.
  - Treats `data: [DONE]` as a no-op marker for compatibility with proxies/gateways (even though it is not a documented Gemini requirement).
- Tests:
  - `siumai/tests/google_generative_ai_stream_fixtures_alignment_test.rs`
  - Gateway/transcoding:
    - `siumai/tests/transcoding_*_to_gemini_alignment_test.rs`

## Status

- Gemini (GenerateContent + SSE streaming) is treated as **Green** for parity and “practical official correctness”
  (base URL, auth headers, endpoints, request shapes, streaming protocol behavior).
