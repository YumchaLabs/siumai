# Fireworks Official API Alignment (OpenAI-compatible)

This document records **official API correctness checks** for Fireworks’ OpenAI-compatible inference API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/openai-official-api-alignment.md` (shared OpenAI protocol family details)

## Sources (Fireworks docs)

- API reference: <https://docs.fireworks.ai/api-reference>
- Inference API: <https://docs.fireworks.ai/api-reference/inference>

## Siumai implementation (where to compare)

OpenAI-compatible preset:

- Preset config: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs` (`fireworks`)
- Shared protocol family: `siumai-protocol-openai/src/standards/openai/*`

## Base URL + endpoints (official)

From the official Fireworks API reference:

- Base URL: `https://api.fireworks.ai/inference/v1`
- OpenAI-compatible endpoints referenced in the inference section include:
  - `POST /chat/completions`
  - `POST /embeddings`
  - `POST /responses` (documented by Fireworks, but not part of Siumai’s OpenAI-compatible preset yet)

### Siumai mapping

- Default preset base URL: `https://api.fireworks.ai/inference/v1`
  - Source: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs`
- Chat endpoint: `${base}/chat/completions`
- Embeddings endpoint: `${base}/embeddings`
- Responses endpoint: out-of-scope for the `fireworks` OpenAI-compatible preset (future work).

## Authentication (official)

Fireworks uses OpenAI-style bearer auth:

- `Authorization: Bearer <FIREWORKS_API_KEY>`

### Siumai mapping

- `Authorization: Bearer ...` is provided by the OpenAI-compatible headers builder.
- Additional headers can be injected via `ProviderContext.http_extra_headers`.

## Tests (how correctness is locked down)

- `siumai/tests/fireworks_openai_compat_url_alignment_test.rs`
- `siumai/tests/fireworks_openai_compat_error_alignment_test.rs`

## Status

- Fireworks is treated as **Green** for endpoint/header correctness and OpenAI-compatible error envelope handling.
- Fireworks `/responses` support is currently **Out-of-scope** for this preset.

