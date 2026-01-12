# Perplexity Official API Alignment (Sonar Chat Completions)

This document records **official API correctness checks** for Perplexity’s Sonar Chat Completions API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/official/openai-official-api-alignment.md` (shared OpenAI protocol family details)

## Sources (Perplexity docs)

- API reference root: <https://docs.perplexity.ai/api-reference>
- Chat Completions: <https://docs.perplexity.ai/api-reference/chat-completions-post>

## Siumai implementation (where to compare)

OpenAI-compatible preset:

- Preset config: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs` (`perplexity`)
- Shared protocol family: `siumai-protocol-openai/src/standards/openai/*`

## Base URL + endpoints (official)

From the official API reference:

- Base URL: `https://api.perplexity.ai`
- Endpoints:
  - Chat Completions: `POST /chat/completions`
  - Async Chat Completions (out-of-scope): `POST /async/chat/completions` and related `GET` endpoints
  - Search API (out-of-scope): `POST /search`

### Siumai mapping

- Default preset base URL: `https://api.perplexity.ai`
- Chat endpoint: `${base}/chat/completions`
  - Implemented via the shared OpenAI-compatible spec.
- Async chat and Search API are **out-of-scope** for the OpenAI-compatible preset layer.

## Authentication (official)

Perplexity uses OpenAI-style bearer auth:

- `Authorization: Bearer <PPLX_API_KEY>`

### Siumai mapping

- `Authorization: Bearer ...` is provided by the OpenAI-compatible headers builder.
- Additional headers can be injected via `ProviderContext.http_extra_headers`.

## Streaming (official)

The API reference includes a `stream` parameter for chat completions.

### Siumai mapping

Siumai treats Perplexity as OpenAI-compatible Chat Completions streaming (SSE).

## Tests (how correctness is locked down)

- `siumai/tests/perplexity_openai_compat_url_alignment_test.rs`
- `siumai/tests/perplexity_openai_compat_error_alignment_test.rs`

## Status

- Perplexity is treated as **Green** for endpoint/header correctness and OpenAI-compatible error envelope handling.
