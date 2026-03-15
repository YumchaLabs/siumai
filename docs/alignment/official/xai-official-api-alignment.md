# xAI Official API Alignment (OpenAI-compatible Chat + Responses + TTS)

This document records **official API correctness checks** for the xAI Enterprise API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/provider-defined-tools-alignment.md` (tool id conventions + provider-defined tools)

## Sources (xAI docs)

- REST API Reference: <https://docs.x.ai/developers/api-reference>
- Text to Speech (Beta): <https://docs.x.ai/developers/model-capabilities/audio/text-to-speech>
- Voice Overview: <https://docs.x.ai/docs/guides/voice>

## Local access note (contributors)

In some regions, `docs.x.ai` may be blocked or protected by anti-bot checks.
This repo’s contributors commonly use a local HTTP proxy:

```bash
set HTTP_PROXY=http://127.0.0.1:10809
set HTTPS_PROXY=http://127.0.0.1:10809
```

## Siumai implementation (where to compare)

Provider (thin wrapper):

- `siumai-provider-xai/src/providers/xai/*`

OpenAI-compatible vendor layer (xAI preset):

- `siumai-provider-openai-compatible/src/providers/openai_compatible/*`

OpenAI protocol family (Chat + Responses + SSE):

- `siumai-protocol-openai/src/standards/openai/*`

## Base URL + endpoints (official)

From the official REST API Reference:

- Base for all routes: `https://api.x.ai`
- Authentication header: `Authorization: Bearer <your xAI API key>`
- Documented endpoints include:
  - Chat Completions: `/v1/chat/completions`
  - Responses: `/v1/responses`
  - Responses (retrieve/delete): `/v1/responses/{response_id}`
  - Models: `/v1/models` and `/v1/models/{model_id}`
  - Messages (Anthropic compatible): `/v1/messages`
  - Images: `/v1/images/generations`
  - Text to Speech (Beta): `/v1/tts`
  - Legacy completions (Anthropic compatible - legacy): `/v1/complete`

The same page states the API provides “full compatibility with the OpenAI REST API”.

### Siumai mapping

- Default xAI base URL (used by the OpenAI-compatible vendor preset):
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs` (`https://api.x.ai/v1`)
- Chat routing:
  - `POST {base_url}/chat/completions`
  - Fixtures: `siumai/tests/xai_chat_request_fixtures_alignment_test.rs`
- Responses routing:
  - `POST {base_url}/responses`
  - SSE mapping:
    - `siumai-protocol-openai/src/standards/openai/responses_sse.rs`
  - Fixtures/tests:
    - `siumai/tests/xai_responses_response_fixtures_alignment_test.rs`
    - `siumai/tests/xai_responses_*_stream_alignment_test.rs`
- Speech routing:
  - Provider-owned TTS path: `POST https://api.x.ai/v1/tts`
  - Siumai implementation: `siumai-provider-xai/src/providers/xai/audio.rs`
  - Registry factory path: `siumai-registry/src/registry/factories/xai.rs`
  - Contract tests:
    - `siumai-provider-xai/src/providers/xai/audio.rs`
    - `siumai-registry/src/registry/factories/contract_tests.rs`

## Provider-defined tools (xAI specifics)

xAI’s Responses API supports provider tools (e.g. search tools) that are represented as
provider-defined tool ids in Vercel AI SDK fixtures.

### Siumai mapping

- Tool naming + request mapping parity is locked down via fixtures:
  - `siumai/tests/xai_responses_request_tool_mapping_test.rs`
  - `siumai/tests/xai_responses_web_search_stream_alignment_test.rs`
  - `siumai/tests/xai_responses_x_search_stream_alignment_test.rs`

## Speech (official)

The current xAI audio docs expose a dedicated beta Text to Speech API:

- Endpoint: `POST https://api.x.ai/v1/tts`
- Response body: raw audio bytes
- Request body fields documented by xAI include `text`, `voice_id`, and `output_format`
- The same page also documents a streaming WebSocket variant at `wss://api.x.ai/v1/tts`

As of **2026-03-08**, the current public voice overview still centers on the Grok Voice Agent API and does not document a standalone REST speech-to-text endpoint that matches the shared OpenAI-compatible `/audio/transcriptions` adapter shape.

The voice docs do describe realtime transcription events inside the Voice Agent/WebSocket flow, but that is a different contract from a standalone REST transcription API.

### Siumai mapping

- xAI stays **outside** the shared OpenAI-compatible audio family on purpose.
- Siumai now exposes a **provider-owned speech-family path** for xAI via `XaiClient`, using `/v1/tts` directly.
- `speech` is available on registry-native and config-first xAI paths; `transcription` remains intentionally unsupported for now.
- Core no-network coverage is now safe for this path because JSON-bytes execution honors injected custom transports.
- Provider-owned unsupported semantics are now also locked explicitly:
  - `siumai-provider-xai/src/providers/xai/audio.rs`
  - `siumai-registry/src/registry/factories/contract_tests.rs`

## Error envelope (official)

xAI uses the OpenAI-style error envelope:

- `{ "error": { "message": "...", "type": "...", "code": "..." } }`

### Siumai mapping

- Error classification (shared OpenAI-compatible mapping):
  - `siumai-protocol-openai/src/standards/openai/errors.rs`
- Fixtures/tests:
  - `siumai/tests/xai_http_error_fixtures_alignment_test.rs`
  - `siumai/tests/fixtures/xai/errors/*`

## Status

- xAI is treated as **Green** for official correctness on chat/responses and **Green-with-scope-boundary** for audio: base URL and endpoints, auth header format, OpenAI-compatible error envelope, Responses API request/stream mapping, and the dedicated `/v1/tts` speech path are aligned with current public docs.
- As of **2026-03-08**, shared OpenAI-compatible STT/TTS enrollment remains intentionally out of scope until xAI publishes a matching standalone transcription contract; Siumai now also has explicit no-network rejection coverage so this boundary cannot regress silently.
