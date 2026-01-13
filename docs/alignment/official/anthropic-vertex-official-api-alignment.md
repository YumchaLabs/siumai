# Anthropic on Google Vertex Official API Alignment (Claude + RawPredict)

This document records **official API correctness checks** for Anthropic (Claude) on Google Vertex AI,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/official/google-vertex-official-api-alignment.md` (Vertex Gemini/Embeddings/Imagen audit)

## Sources (Google Cloud docs)

- Claude on Vertex AI (partner models): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude>
- PredictionService (RawPredict, REST reference):
  - `projects.locations.publishers.models.rawPredict` (v1): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/projects.locations.publishers.models/rawPredict>
  - `projects.locations.publishers.models.streamRawPredict` (v1): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/projects.locations.publishers.models/streamRawPredict>

## Local access note (contributors)

In some regions, `docs.cloud.google.com` may be unstable.
This repo’s contributors commonly use a local HTTP proxy:

```bash
set HTTP_PROXY=http://127.0.0.1:10809
set HTTPS_PROXY=http://127.0.0.1:10809
```

## Siumai implementation (where to compare)

- Provider wiring:
  - `siumai-provider-google-vertex/src/providers/anthropic_vertex/client.rs`
  - `siumai-provider-google-vertex/src/providers/anthropic_vertex/spec.rs`
- Reused Anthropic protocol mapping:
  - `siumai-protocol-anthropic/src/standards/anthropic/transformers.rs`
  - `siumai-protocol-anthropic/src/standards/anthropic/streaming.rs`

Vercel reference:

- `repo-ref/ai/packages/google-vertex/src/anthropic/*`

## Base URL + endpoints

Vercel-aligned (publisher model path):

- Base URL (enterprise mode):
  - `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/anthropic`
- Chat endpoints (per-model):
  - Non-stream: `${base}/models/{model}:rawPredict`
  - Stream: `${base}/models/{model}:streamRawPredict`

### Siumai mapping

- URL builder:
  - `siumai-provider-google-vertex/src/providers/anthropic_vertex/spec.rs` (`chat_url`)
- Behavior:
  - Supports both base forms:
    - base ends with `/publishers/anthropic` (preferred in Siumai)
    - base ends with `/publishers/anthropic/models` (Vercel-style)

## Authentication + headers (official)

From official examples:

- `Authorization: Bearer $(gcloud auth print-access-token)`
- `Content-Type: application/json` (Google examples often show `application/json; charset=utf-8`)

### Siumai mapping

- Header builder:
  - `siumai-provider-google-vertex/src/providers/anthropic_vertex/spec.rs` (`build_headers`)
- Note:
  - Vertex Anthropic uses standard Google auth; it does **not** use `x-api-key` / `anthropic-version` headers.

## Request body differences vs native Anthropic Messages

Vercel AI SDK behavior (Vertex Anthropic adapter):

- Removes `model` from the request body (model id is in the URL).
- Injects `anthropic_version: "vertex-2023-10-16"` as a request-body field.

### Siumai mapping

- Request transformer wrapper:
  - `siumai-provider-google-vertex/src/providers/anthropic_vertex/spec.rs` (`VertexAnthropicRequestTransformer`)
- Test coverage:
  - `siumai/tests/streaming/siumai_interceptor_request_assert_test.rs` (asserts `anthropic_version` injection and `model` omission)

## Streaming protocol

Siumai reuses the Anthropic SSE event mapping to parse and serialize streaming events.

- Streaming converter:
  - `siumai-protocol-anthropic/src/standards/anthropic/streaming.rs`

## Status

- Anthropic on Vertex (RawPredict + streamRawPredict) is treated as **Green** for official correctness:
  base URL composition, auth headers, request body shaping (`anthropic_version`), and streaming behavior.

## Non-applicable native Anthropic endpoints

The native Anthropic API includes additional endpoints such as:

- `POST /v1/messages/count_tokens`
- `POST /v1/messages/batches` (Message Batches) + `/results`

These endpoints are **not part of** the Vertex AI `rawPredict` / `streamRawPredict` surface for partner models.

### Vercel AI SDK

Vercel’s `vertexAnthropic` provider (`repo-ref/ai/packages/google-vertex/src/anthropic/*`) only targets:

- `:rawPredict` and `:streamRawPredict` for chat/streaming

and does not expose token counting or message batching helpers for Vertex Anthropic.

### Siumai

Siumai’s `siumai-provider-google-vertex` `anthropic_vertex` client mirrors the same scope:

- Chat + streaming via RawPredict/streamRawPredict (Green)
- No provider-only helpers for `count_tokens` / Message Batches on Vertex Anthropic (not supported by the underlying API)
