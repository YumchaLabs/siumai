# OpenAI Official API Alignment (Chat Completions + Responses + Streaming)

This document records **official API correctness checks** for the OpenAI APIs that Siumai supports,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/core-trio-module-alignment.md` (module mapping for the core trio)

## Sources (OpenAI official)

Primary (machine-readable, accessible even when the docs site is blocked):

- OpenAI OpenAPI (documented) spec (Stainless): <https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml>

Human-readable (may be Cloudflare-blocked in some regions):

- API reference (Chat Completions): <https://platform.openai.com/docs/api-reference/chat>
- API reference (Responses): <https://platform.openai.com/docs/api-reference/responses>
- Guide: Responses vs Chat Completions: <https://platform.openai.com/docs/guides/responses-vs-chat-completions>
- Guide: Streaming: <https://platform.openai.com/docs/guides/streaming-responses>

## Local access note (contributors)

In some regions, `platform.openai.com` can be blocked by Cloudflare challenges.
When that happens, use the Stainless OpenAPI spec above as the “official” source of truth,
and/or route through a local HTTP proxy:

```bash
set HTTP_PROXY=http://127.0.0.1:10809
set HTTPS_PROXY=http://127.0.0.1:10809
```

## Siumai implementation (where to compare)

- Protocol mapping:
  - `siumai-protocol-openai/src/standards/openai/headers.rs`
  - `siumai-protocol-openai/src/standards/openai/errors.rs`
  - `siumai-protocol-openai/src/standards/openai/transformers/request.rs`
  - `siumai-protocol-openai/src/standards/openai/transformers/stream.rs`
  - `siumai-protocol-openai/src/standards/openai/responses_sse.rs`
- OpenAI-compatible streaming (Chat Completions-style):
  - `siumai-core/src/standards/openai/compat/streaming.rs`
- Provider wiring:
  - `siumai-provider-openai/src/providers/openai/*`
  - `siumai-provider-azure/src/providers/azure_openai/*`

## Base URL + endpoints

From the OpenAPI spec `servers[0].url`:

- Base URL: `https://api.openai.com/v1`

Endpoints relevant to Siumai Alpha.5 parity:

- Chat Completions: `POST /chat/completions`
- Responses: `POST /responses`
- Embeddings: `POST /embeddings`
- Images: `POST /images/generations`

### Siumai mapping

- OpenAI provider defaults:
  - `siumai-provider-openai/src/providers/openai/config.rs` (`base_url = https://api.openai.com/v1`)
- Endpoint builders:
  - `siumai-protocol-openai/src/standards/openai/chat.rs`
  - `siumai-protocol-openai/src/standards/openai/transformers/request.rs`

## Required headers + auth

From the OpenAPI spec examples:

- `Authorization: Bearer $OPENAI_API_KEY`
- `Content-Type: application/json`

Notes:

- The Stainless spec does not explicitly list legacy org/project headers, but they are widely used:
  - `OpenAI-Organization: org_...`
  - `OpenAI-Project: proj_...`
- The spec also references `x-request-id` in examples; some deployments additionally emit `x-openai-request-id`.

### Siumai mapping

- Header construction:
  - `siumai-protocol-openai/src/standards/openai/headers.rs` (`build_openai_compatible_json_headers`)
  - OpenAI provider config supports `organization` + `project`:
    - `siumai-provider-openai/src/providers/openai/config.rs`
- Error/debug IDs:
  - `siumai-core/src/retry_api.rs` (`classify_http_error` extracts `x-request-id`, `x-openai-request-id`, etc.)

## Chat Completions request/response

Official concepts (see OpenAPI + docs):

- Request fields: `model`, `messages[]`, optional `tools[]`, optional `tool_choice`, optional `stream`
- Streaming: `text/event-stream` with JSON “chunk” objects; stream terminates with `data: [DONE]`
- `stream_options` exists and can include `{"include_usage": true}` (spec mentions this in streaming descriptions)

### Siumai mapping

- Request transformer:
  - `siumai-protocol-openai/src/standards/openai/transformers/request.rs`
- Streaming parsing/serialization (OpenAI-compatible SSE):
  - `siumai-core/src/standards/openai/compat/streaming.rs`
  - Provider SSE config uses `[DONE]` as a done marker:
    - `siumai-provider-openai/src/providers/openai/client/sse_helpers.rs`
- Fixtures/tests:
  - `siumai/tests/openai_*_fixtures_alignment_test.rs`
  - `siumai/tests/transcoding_*_to_openai_alignment_test.rs`

## Responses request/response

Official concepts (see OpenAPI + docs):

- Request fields: `model`, `input`, optional `tools[]`, optional `tool_choice`, optional `include[]`
- Streaming:
  - Enabled via `stream: true`
  - SSE frames may include `event:` plus JSON `data:` (the spec examples include `data: [DONE]`)
  - `stream_options` exists and can include usage, and the spec documents extra knobs (e.g. obfuscation)
- Event replay:
  - The spec documents query params like `starting_after` for the streaming “get response” endpoint

### Siumai mapping

- SSE parser/serializer:
  - `siumai-protocol-openai/src/standards/openai/responses_sse.rs`
- Tests:
  - `siumai/tests/openai_responses_*_stream_alignment_test.rs`
  - `siumai/tests/openai_*_fixtures_alignment_test.rs`

## Known gaps / out-of-scope (Alpha.5)

These are “official API surface areas” that exist in the OpenAPI spec but are not (yet) modeled by Siumai:

- Responses streaming replay controls (e.g. `starting_after`)
- Responses stream obfuscation control (`include_obfuscation`)

This is intentional for Alpha.5 because Siumai focuses on:

- end-to-end request mapping
- streaming parsing/serialization for gateway use-cases
- fixture parity with Vercel AI SDK behavior

## Status

- OpenAI (Chat Completions + Responses) is treated as **Green** for parity and “practical official correctness”
  (base URL, auth headers, endpoints, streaming protocol behavior, error envelope parsing).
