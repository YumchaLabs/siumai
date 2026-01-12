# Anthropic Official API Alignment (Messages + Streaming)

This document records **official API correctness checks** for the Anthropic Messages API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/core-trio-module-alignment.md` (module mapping for the core trio)

## Sources (Anthropic docs)

The legacy `docs.anthropic.com` host redirects to the Claude platform docs, so both are listed.

- Messages API:
  - <https://docs.anthropic.com/en/api/messages>
  - <https://platform.claude.com/docs/en/api/messages/create>
- Streaming (Messages SSE):
  - <https://docs.anthropic.com/en/api/messages-streaming>
  - <https://platform.claude.com/docs/en/api/messages-streaming>
- Models:
  - <https://platform.claude.com/docs/en/api/models/list>
- API overview (basic curl): <https://docs.anthropic.com/en/api/overview>
- Versioning policy: <https://docs.anthropic.com/en/api/versioning>
- Beta headers:
  - <https://docs.anthropic.com/en/api/beta-headers>
  - <https://platform.claude.com/docs/en/api/beta-headers>
- Errors:
  - <https://docs.anthropic.com/en/api/errors>
  - <https://platform.claude.com/docs/en/api/errors>

### Local access note (contributors)

In some regions, `docs.anthropic.com` may be blocked.
This repo’s contributors commonly use a local HTTP proxy:

```bash
set HTTP_PROXY=http://127.0.0.1:10809
set HTTPS_PROXY=http://127.0.0.1:10809
```

## Siumai implementation (where to compare)

- Protocol mapping:
  - `siumai-protocol-anthropic/src/standards/anthropic/transformers.rs`
  - `siumai-protocol-anthropic/src/standards/anthropic/streaming.rs`
  - `siumai-protocol-anthropic/src/standards/anthropic/errors.rs`
  - `siumai-protocol-anthropic/src/standards/anthropic/utils.rs`
- Provider wiring:
  - `siumai-provider-anthropic/src/providers/anthropic/*`

## Official endpoint + required headers

From the official “Create a Message” docs:

- Endpoint: `POST https://api.anthropic.com/v1/messages`
- Required headers:
  - `x-api-key: $ANTHROPIC_API_KEY`
  - `anthropic-version: 2023-06-01`
  - `content-type: application/json`

### Siumai mapping

- Header builder:
  - `siumai-protocol-anthropic/src/standards/anthropic/utils.rs` (`build_headers`)
- Tests:
  - `siumai/tests/mock_api/anthropic_mock_api_test.rs` (wiremock asserts)
  - `siumai/tests/anthropic_messages_custom_transport_alignment_test.rs` (transport/header plumbing)

## Models API (`/v1/models`)

From the official “Models” docs:

- List models: `GET https://api.anthropic.com/v1/models`
- Retrieve a model: `GET https://api.anthropic.com/v1/models/{model_id}`
- Pagination/query params:
  - `before_id`, `after_id`, `limit`
- Response envelope (list):
  - `data: [...]`, `first_id`, `last_id`, `has_more`

### Siumai mapping

- URL shaping (adds `/v1` when the configured base URL does not include it):
  - `siumai-provider-anthropic/src/providers/anthropic/spec.rs` (`models_url`, `model_url`)
- Implementation:
  - `siumai-provider-anthropic/src/providers/anthropic/models.rs`
  - `siumai-provider-anthropic/src/providers/anthropic/types.rs` (`AnthropicModelsResponse`, `AnthropicModelInfo`)

## Beta headers (`anthropic-beta`)

From the official “Beta headers” doc:

- Header name: `anthropic-beta`
- Multi-feature syntax: `anthropic-beta: feature1,feature2,feature3`

### Siumai mapping

- `ProviderContext.http_extra_headers` can carry `anthropic-beta`.
- Siumai also **auto-injects required betas** for certain features (Vercel-aligned),
  and merges/de-dupes values.
- Tests:
  - `siumai/tests/anthropic_messages_fixtures_alignment_test.rs`
  - `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`

## Request body (Messages)

From the official “Create a Message” docs:

- Required fields:
  - `model` (string)
  - `max_tokens` (int)
  - `messages` (array)
  - `stream` (optional boolean; enables SSE streaming)
  - `system` (optional; system prompt)
  - `tools` (optional array)
  - `tool_choice` (optional object)
  - `thinking` (optional object)
  - `stop_sequences` (optional array)

### Siumai mapping

- Unified request type: `siumai-core/src/types/chat.rs` (`ChatRequest`)
- Anthropic transformer:
  - `siumai-protocol-anthropic/src/standards/anthropic/transformers.rs`
- Fixtures:
  - `siumai/tests/fixtures/anthropic/messages/*`

## Streaming protocol (SSE)

From the official streaming doc (`/en/api/messages-streaming`):

- The server sends **named SSE events**, and each frame includes a matching JSON `type`.
- The documented event names include:
  - `message_start`
  - `content_block_start`
  - `content_block_delta`
  - `content_block_stop`
  - `message_delta`
  - `message_stop`
  - `ping`
  - `error`
- Versioning note (official): new event types may be added; clients should handle unknown types.
- The older `data: [DONE]` is not part of the modern protocol.

### Siumai mapping

- Parser/bridge:
  - `siumai-protocol-anthropic/src/standards/anthropic/streaming.rs`
- Behavior:
  - Uses `event.data` JSON `type` as the primary discriminator.
  - Ignores unknown/unsupported `type` values (returns no events).
  - Ignores `data == "[DONE]"` (compat with legacy emitters).
  - `ping` currently yields no unified event (safe no-op).
  - `content_block_delta` subtypes (official):
    - `text_delta` -> `content_delta` (and an `anthropic:text-delta` custom event when the block is a text block)
    - `input_json_delta` -> `tool_call_delta` (streaming tool input JSON)
    - `thinking_delta` -> `thinking_delta`
    - `signature_delta` -> captured and emitted as `anthropic:thinking-signature-delta` (also surfaced in stream-end metadata)
- Fixtures/tests:
  - `siumai/tests/fixtures/anthropic/messages-stream/*`
  - `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`
  - Cross-protocol gateway:
    - `docs/alignment/streaming-bridge-alignment.md`
    - `siumai/tests/transcoding_*_to_anthropic_alignment_test.rs`

## Error format + retryable overload

From the official Errors doc:

- Error envelopes are JSON and include an error type + message.
- Overload scenarios are described as `overloaded_error` (service overloaded, HTTP `529`).

### Siumai mapping

- Error classification:
  - `siumai-protocol-anthropic/src/standards/anthropic/errors.rs`
- Retry semantics:
  - `overloaded_error` is treated as retryable.
  - If an upstream proxy misreports the status code, Siumai still maps the error to `529`
    for stable downstream handling (Vercel parity).
- Tests:
  - Fixture-driven error alignment tests (see `docs/alignment/vercel-ai-fixtures-alignment.md`)

## Status

- **Green**: `POST /v1/messages` headers/body/streaming/errors are covered by fixture parity + targeted tests.
- **Yellow**: `/v1/models` correctness is implemented, but still needs fixture-driven parity tests.
