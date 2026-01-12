# OpenRouter Official API Alignment (OpenAI-compatible)

This document records **official API correctness checks** for OpenRouter’s OpenAI-compatible API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/openai-official-api-alignment.md` (shared OpenAI protocol family details)
- `docs/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)

## Sources (OpenRouter docs)

- OpenRouter API docs: <https://openrouter.ai/docs/api>

## Siumai implementation (where to compare)

OpenAI-compatible preset:

- Preset config: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs` (`openrouter`)
- Shared protocol family: `siumai-protocol-openai/src/standards/openai/*`

## Base URL + endpoints (official)

From the official OpenRouter API docs:

- Base URL: `https://openrouter.ai/api/v1`
- OpenAI-compatible endpoints include:
  - `POST /chat/completions`
  - `POST /embeddings`
  - `GET /models`

### Siumai mapping

- Default preset base URL: `https://openrouter.ai/api/v1`
  - Source: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs`
- Chat endpoint: `${base}/chat/completions`
  - Implementation: `siumai-protocol-openai/src/standards/openai/chat.rs` (OpenAI-compatible)
- Embeddings endpoint: `${base}/embeddings`
  - Implementation: `siumai-protocol-openai/src/standards/openai/embedding.rs` (OpenAI-compatible)
- Models endpoint: `${base}/models`
  - Implementation: `ProviderSpec::{models_url, model_url}` via the OpenAI family mapping

## Authentication + recommended headers (official)

OpenRouter uses OpenAI-style auth:

- `Authorization: Bearer <OPENROUTER_API_KEY>`

OpenRouter also documents optional attribution headers (recommended for rankings):

- `HTTP-Referer: <your site url>`
- `X-Title: <your app name>`

### Siumai mapping

- `Authorization: Bearer ...` is provided by the OpenAI-compatible headers builder.
- Optional attribution headers are passed through via:
  - `ProviderContext.http_extra_headers` (builder `.header(k, v)` / `.custom_headers(..)`)

## Known deltas / notes

- OpenRouter is treated as **OpenAI-compatible Chat Completions** for parity and gateway use-cases.
  If an endpoint is not documented as OpenAI-compatible (e.g. `/responses`), Siumai treats it as out-of-scope
  for the `openrouter` preset (use a provider that natively supports that surface).

## Tests (how correctness is locked down)

- `siumai/tests/openrouter_chat_request_alignment_test.rs`
- `siumai/tests/openrouter_http_error_alignment_test.rs`

## Status

- OpenRouter is treated as **Green** for endpoint/header correctness and OpenAI-compatible error envelope handling.

