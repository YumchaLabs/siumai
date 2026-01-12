# Groq Official API Alignment (OpenAI-compatible Chat + Audio)

This document records **official API correctness checks** for Groq’s OpenAI-compatible API surface,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/provider-dual-anchor-alignment-table.md` (Vercel + official pointers per provider family)

## Sources (Groq docs)

- OpenAI compatibility overview: <https://console.groq.com/docs/openai>
- API reference: <https://console.groq.com/docs/api-reference>

## Siumai implementation (where to compare)

- Provider:
  - `siumai-provider-groq/src/providers/groq/*`
- Shared OpenAI-like protocol family:
  - `siumai-protocol-openai/src/standards/openai/*`
  - `siumai-core/src/standards/openai/compat/*`

## Base URL + endpoints (official)

From the official OpenAI compatibility docs:

- Base URL: `https://api.groq.com/openai/v1`

From the API reference:

- Chat Completions:
  - `POST https://api.groq.com/openai/v1/chat/completions`
- Audio:
  - TTS: `POST https://api.groq.com/openai/v1/audio/speech`
  - STT: `POST https://api.groq.com/openai/v1/audio/transcriptions`

### Siumai mapping

- Default base URL:
  - `siumai-provider-groq/src/providers/groq/config.rs` (`GroqConfig::DEFAULT_BASE_URL`)
- Routing:
  - Chat: inherited from the OpenAI-compatible chat spec:
    - `siumai-provider-groq/src/providers/groq/spec.rs`
  - Audio endpoints:
    - `siumai-provider-groq/src/providers/groq/transformers.rs` (`tts_endpoint`, `stt_endpoint`)

## Authentication + required headers (official)

Groq’s OpenAI-compatible endpoints use the standard Bearer token scheme:

- `Authorization: Bearer $GROQ_API_KEY`
- `Content-Type: application/json` (for JSON requests)

### Siumai mapping

- Groq adapter ensures JSON content type and a stable user-agent:
  - `siumai-provider-groq/src/providers/groq/spec.rs` (`GroqOpenAiChatAdapter::build_headers`)
- The executor layer injects the `Authorization: Bearer ...` header via `ProviderContext.api_key`.

## Chat request mapping quirks (Groq-specific)

Groq is largely OpenAI Chat Completions compatible, but has a few documented/observed quirks that
Siumai intentionally normalizes for parity:

- The `developer` role is not accepted; Siumai downgrades it to `system`.
- `max_completion_tokens` is rewritten to `max_tokens`.
- `stream_options` is omitted for streaming requests.

### Siumai mapping

- Adapter logic:
  - `siumai-provider-groq/src/providers/groq/spec.rs` (`GroqOpenAiChatAdapter::transform_request`)
- Fixtures/tests:
  - `siumai/tests/groq_chat_request_fixtures_alignment_test.rs`
  - `siumai/tests/fixtures/groq/chat-requests/*`

## Error envelope (official)

Groq errors follow the OpenAI-style envelope:

- `{ "error": { "message": "...", "type": "...", "code": "..." } }`

### Siumai mapping

- Error classification:
  - `siumai-protocol-openai/src/standards/openai/errors.rs`
- Fixtures/tests:
  - `siumai/tests/groq_http_error_fixtures_alignment_test.rs`
  - `siumai/tests/fixtures/groq/errors/*`
  - `siumai/tests/mock_api/groq_mock_api_test.rs` (wiremock based on official examples)

## Status

- Groq is treated as **Green** for official correctness:
  base URL, endpoints, auth headers, chat/audio routing, OpenAI-style error envelope,
  and the Groq-specific request mapping quirks above.
