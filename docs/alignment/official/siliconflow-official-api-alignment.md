# SiliconFlow Official API Alignment (OpenAI-compatible)

This document records **official API correctness checks** for SiliconFlow’s OpenAI-compatible API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/official/openai-official-api-alignment.md` (shared OpenAI protocol family details)

## Sources (SiliconFlow docs)

- API overview: <https://docs.siliconflow.cn/docs/api>
- Chat Completions: <https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions>
- Embeddings: <https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings>
- Image generations: <https://docs.siliconflow.cn/cn/api-reference/images/images-generations>
- Rerank: <https://docs.siliconflow.cn/cn/api-reference/rerank/create-rerank>
- Speech: <https://docs.siliconflow.cn/cn/api-reference/audio/create-speech>
- Audio transcriptions: <https://docs.siliconflow.cn/cn/api-reference/audio/create-audio-transcriptions>

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
- Speech: `POST /audio/speech`
- Audio transcriptions: `POST /audio/transcriptions`

### Siumai mapping

- Default preset base URL: `https://api.siliconflow.cn/v1`
- URL building is done via the shared OpenAI-compatible spec:
  - `siumai-protocol-openai/src/standards/openai/compat/spec.rs`
- Speech/transcription execution is routed through the shared OpenAI-compatible audio transformer and executor stack.

## Authentication (official)

SiliconFlow uses OpenAI-style bearer auth:

- `Authorization: Bearer <SILICONFLOW_API_KEY>`

### Siumai mapping

- `Authorization: Bearer ...` is provided by the OpenAI-compatible headers builder.
- Additional headers can be injected via `ProviderContext.http_extra_headers`.

## Tests (how correctness is locked down)

- `siumai/tests/siliconflow_openai_compat_url_alignment_test.rs`
- `siumai/tests/siliconflow_openai_compat_error_alignment_test.rs`
- `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs`
- `siumai-registry/src/registry/factories/contract_tests.rs`

- SiliconFlow speech/transcription capability enrollment is locked on the OpenAI-compatible factory path.
- SiliconFlow now also has direct no-network transport-boundary coverage for both `/audio/speech` request shaping and `/audio/transcriptions` multipart upload shaping.

## Status

- SiliconFlow is treated as **Green** for endpoint/header correctness and OpenAI-compatible error envelope handling.
- SiliconFlow speech/transcription family routing is now also **Green** on the shared compat path, with direct request-shape capture coverage for both speech and transcription.
