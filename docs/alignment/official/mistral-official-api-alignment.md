# Mistral Official API Alignment (OpenAI-Compatible)

This document records **official API correctness checks** for the Mistral API
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)

## Sources (Mistral official)

Primary (machine-readable):

- Mistral OpenAPI spec: <https://docs.mistral.ai/openapi.yaml>

Human-readable:

- API docs: <https://docs.mistral.ai/api/>

## Siumai implementation (where to compare)

Mistral is currently supported as an **OpenAI-compatible vendor preset**:

- Preset configuration:
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs` (`id = "mistral"`)
- Protocol mapping:
  - `siumai-protocol-openai/src/standards/openai/compat/*`
- ProviderSpec used at runtime:
  - `siumai-protocol-openai/src/standards/openai/compat/spec.rs` (`OpenAiCompatibleSpecWithAdapter`)

## Official endpoints (from OpenAPI)

The OpenAPI spec declares the server host as `https://api.mistral.ai`,
and all public endpoints are under the `/v1` prefix.

### Chat Completions

- Endpoint: `POST https://api.mistral.ai/v1/chat/completions`
- Streaming: response may be `text/event-stream`

Siumai mapping:

- Base URL (preset): `https://api.mistral.ai/v1`
- Chat endpoint: `${base_url}/chat/completions`
- URL parity tests:
  - `siumai/tests/mistral_openai_compat_url_alignment_test.rs`

### Embeddings

- Endpoint: `POST https://api.mistral.ai/v1/embeddings`

Siumai mapping:

- Embeddings endpoint: `${base_url}/embeddings`
- URL parity tests:
  - `siumai/tests/mistral_openai_compat_url_alignment_test.rs`

### Models

- List models: `GET https://api.mistral.ai/v1/models`
- Retrieve model: `GET https://api.mistral.ai/v1/models/{model_id}`

Siumai mapping:

- Models are handled via the shared OpenAI-compatible `ProviderSpec::{models_url, model_url}` path rules.

## Authentication + headers

The OpenAPI spec uses HTTP `bearer` auth.

Siumai mapping:

- Uses `Authorization: Bearer {API_KEY}` (OpenAI-compatible family behavior).
- The API key env var follows `{PROVIDER_ID}_API_KEY` conventions when possible.

## Errors

The Mistral OpenAPI spec does not comprehensively enumerate all error responses for every operation,
so Siumai enforces the shared OpenAI-compatible error envelope handling:

- Expected envelope: `{ "error": { "message": "...", "type": "...", "code": "..." } }`
- Implementation: `siumai-protocol-openai/src/standards/openai/errors.rs` (`classify_openai_compatible_http_error`)
- Tests:
  - `siumai/tests/mistral_openai_compat_error_alignment_test.rs`

## Status

- **Green**: base URL + endpoints + error envelope mapping are covered by focused tests.
- **Yellow**: provider-specific capabilities beyond OpenAI-compat (e.g. Mistral-specific streaming quirks,
  additional endpoints like audio transcription present in the OpenAPI) are not yet audited/covered.

