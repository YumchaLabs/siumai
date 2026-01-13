# Gemini Official API Alignment (GenerateContent + Streaming)

This document records **official API correctness checks** for the Gemini (Google Generative Language) API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/core-trio-module-alignment.md` (module mapping for the core trio)

## Sources (Google AI for Developers)

- Text generation (REST): <https://ai.google.dev/gemini-api/docs/text-generation?lang=rest>
- Embeddings: <https://ai.google.dev/gemini-api/docs/embeddings>
- Imagen (image generation API): <https://ai.google.dev/gemini-api/docs/imagen>
- Gemini image models (Nano Banana): <https://ai.google.dev/gemini-api/docs/image-generation>
- Video generation (Veo): <https://ai.google.dev/gemini-api/docs/video>
- Music generation (Lyria RealTime / Live API): <https://ai.google.dev/gemini-api/docs/music-generation>
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
- Optional logprobs export (`generationConfig.responseLogprobs`, `generationConfig.logprobs`)
- Optional safety settings
- Optional tools/function calling (`tools`, `toolConfig`)

### Siumai mapping

- Unified request type: `siumai-core/src/types/chat.rs` (`ChatRequest`)
- Gemini transformer:
  - `siumai-protocol-gemini/src/standards/gemini/transformers.rs`
- Fixtures:
  - `siumai/tests/fixtures/google_generative_ai/*`

## Logprobs (`responseLogprobs` / `logprobs`)

From the API reference:

- Request:
  - `generationConfig.responseLogprobs: boolean` enables logprobs export.
  - `generationConfig.logprobs: integer` controls top-K logprobs per decoding step (valid when `responseLogprobs == true`).
- Response:
  - `candidates[].avgLogprobs: number`
  - `candidates[].logprobsResult: LogprobsResult`

### Siumai mapping

- Request:
  - Protocol: `siumai-protocol-gemini/src/standards/gemini/types/generation.rs` (`GenerationConfig`)
  - Provider options: `siumai-provider-gemini/src/provider_options/gemini/mod.rs` (`GeminiOptions`)
- Response:
  - Protocol parsing: `siumai-protocol-gemini/src/standards/gemini/types/content.rs` (`Candidate.avg_logprobs`, `Candidate.logprobs_result`)
  - Exposed via `ChatResponse.provider_metadata` under the Vercel-aligned namespace key (`google` / `vertex`).
  - Typed access: `siumai-provider-gemini/src/provider_metadata/gemini.rs` (`GeminiMetadata`)

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

## Additional endpoints (Embeddings / Imagen / Tokens / Caching / Video)

This repo also aligns a few non-chat endpoints used by the official docs and/or Vercel AI SDK.

### Embeddings (`embedContent` / `batchEmbedContents`)

**Official endpoints**

- `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent`
- `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents`

**Siumai mapping**

- Protocol: `siumai-protocol-gemini/src/standards/gemini/embedding.rs`
- Transformer:
  - Request: `siumai-protocol-gemini/src/standards/gemini/transformers/request.rs` (`transform_embedding`)
  - Response: `siumai-protocol-gemini/src/standards/gemini/transformers/response.rs` (supports `usageMetadata`)
- Provider: `siumai-provider-gemini/src/providers/gemini/client/embedding.rs`

### Imagen (`predict`)

**Official endpoint**

- `POST https://generativelanguage.googleapis.com/v1beta/models/{imagen-model}:predict`

**Siumai mapping**

- Protocol router: `siumai-protocol-gemini/src/standards/gemini/image.rs` (routes `imagen-*` → `:predict`)
- Transformers:
  - Request: `siumai-protocol-gemini/src/standards/gemini/transformers/request.rs`
  - Response: `siumai-protocol-gemini/src/standards/gemini/transformers/response.rs` (parses `predictions[].bytesBase64Encoded`)

### Token counting (`countTokens`)

**Official endpoint**

- `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens`

**Siumai mapping**

- Provider-only helper: `siumai-provider-gemini/src/providers/gemini/tokens.rs`

### Cached contents (`/cachedContents`)

**Official endpoints**

- `POST /cachedContents`
- `GET /cachedContents`
- `GET /cachedContents/{id}`
- `PATCH /cachedContents/{id}` (with `updateMask`)
- `DELETE /cachedContents/{id}`

**Siumai mapping**

- Provider-only helper: `siumai-provider-gemini/src/providers/gemini/cached_contents.rs`

### Video generation (Veo / `predictLongRunning`)

**Official endpoints**

- `POST https://generativelanguage.googleapis.com/v1beta/models/{veo-model}:predictLongRunning`
- `GET  https://generativelanguage.googleapis.com/v1beta/{operation.name}` (poll until `done == true`)

**Siumai mapping**

- Provider-only helper: `siumai-provider-gemini/src/providers/gemini/video.rs`
- Unified capability:
  - `GeminiClient` implements `VideoGenerationCapability` via `siumai-provider-gemini/src/providers/gemini/client/video.rs`

**Download note**

The Veo operation response exposes a downloadable HTTPS URI (not a Files API resource).
For convenience, `GeminiFiles::get_file_content(...)` accepts such URIs directly when the input is `http(s)://...`.

### Music generation (Lyria RealTime)

The official music generation docs are built on the **Live API / WebSocket** (`client.live.music.connect(...)`).
Siumai currently does not implement the Live API client surface for Gemini, so this feature is treated as
**unsupported** at the unified `MusicGenerationCapability` layer for now.
