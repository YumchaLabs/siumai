# Fireworks Official API Alignment (OpenAI-compatible + Transcription)

This document records **official API correctness checks** for Fireworks’ OpenAI-compatible inference API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/official/openai-official-api-alignment.md` (shared OpenAI protocol family details)

## Sources (Fireworks docs)

- API reference: <https://docs.fireworks.ai/api-reference>
- Inference API: <https://docs.fireworks.ai/api-reference/inference>
- Audio transcription API: <https://docs.fireworks.ai/api-reference/audio-transcriptions>

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
- Audio endpoints documented separately include:
  - Base URL: `https://audio.fireworks.ai/v1`
  - `POST /audio/transcriptions`

### Siumai mapping

- Default preset base URL: `https://api.fireworks.ai/inference/v1`
  - Source: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs`
- Chat endpoint: `${base}/chat/completions`
- Embeddings endpoint: `${base}/embeddings`
- Transcription endpoint: `POST https://audio.fireworks.ai/v1/audio/transcriptions`
  - Siumai keeps this on the shared OpenAI-compatible transcription path, but resolves it through the documented dedicated audio host instead of the inference host.
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
- `siumai-core/src/execution/executors/http_request/tests.rs`
- `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs`
- `siumai-registry/src/registry/factories/contract_tests.rs`

- Core multipart request/bytes executors now have injected-transport coverage for final body capture plus 401 retry parity.
- Fireworks transcription now has both an explicit-base-url mock HTTP test and a no-network transport-boundary capture test for the default dedicated audio host.

## Status

- Fireworks is treated as **Green** for endpoint/header correctness and OpenAI-compatible error envelope handling, and now also **Green** for transcription-family alignment through the documented dedicated audio host.
- Fireworks multipart STT request-body parity is now also **Green** at the executor boundary: injected transports see the final multipart `Content-Type` and body bytes that Siumai emits for `/audio/transcriptions`.
- Fireworks TTS/speech support is currently **Out-of-scope** because the public docs currently expose transcription but not a matching standalone speech endpoint on the same compat track.
- Fireworks `/responses` support is currently **Out-of-scope** for this preset.
