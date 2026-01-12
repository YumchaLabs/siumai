# SiliconFlow Official API Alignment (OpenAI-compatible)

This document records **official API correctness checks** for SiliconFlow’s OpenAI-compatible API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/openai-official-api-alignment.md` (shared OpenAI protocol family details)

## Sources (SiliconFlow docs)

- API overview: <https://docs.siliconflow.cn/docs/api>
- Chat Completions: <https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions>
- Embeddings: <https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings>
- Image generations: <https://docs.siliconflow.cn/cn/api-reference/images/images-generations>
- Rerank: <https://docs.siliconflow.cn/cn/api-reference/rerank/create-rerank>

## Siumai implementation (where to compare)

OpenAI-compatible preset:

- Preset config: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs` (`siliconflow`)
- Shared protocol family: `siumai-protocol-openai/src/standards/openai/*`

## Base URL + endpoints (official)

From the official API reference pages, the documented OpenAI-compatible base URL is:

- Base URL: `https://api.siliconflow.cn/v1`

Documented endpoints used by Siumai’s preset:

- Chat Completions: `POST /chat/completions`
- Embeddings: `POST /embeddings`
- Image generations: `POST /images/generations`
- Rerank: `POST /rerank`

### Siumai mapping

- Default preset base URL: `https://api.siliconflow.cn/v1`
- URL building is done via the shared OpenAI-compatible spec:
  - `siumai-protocol-openai/src/standards/openai/compat/spec.rs`

## Authentication (official)

SiliconFlow uses OpenAI-style bearer auth:

- `Authorization: Bearer <SILICONFLOW_API_KEY>`

### Siumai mapping

- `Authorization: Bearer ...` is provided by the OpenAI-compatible headers builder.
- Additional headers can be injected via `ProviderContext.http_extra_headers`.

## Tests (how correctness is locked down)

- `siumai/tests/siliconflow_openai_compat_url_alignment_test.rs`
- `siumai/tests/siliconflow_openai_compat_error_alignment_test.rs`

## Status

- SiliconFlow is treated as **Green** for endpoint/header correctness and OpenAI-compatible error envelope handling.

