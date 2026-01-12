# Anthropic Official API Alignment (Messages + Streaming)

This document records **official API correctness checks** for the Anthropic Messages API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/core-trio-module-alignment.md` (module mapping for the core trio)

## Sources (Anthropic docs)

- Messages API: <https://docs.anthropic.com/en/api/messages>
- Streaming (Messages SSE): <https://docs.anthropic.com/en/api/messages-streaming>
- API overview (basic curl): <https://docs.anthropic.com/en/api/overview>
- Versioning policy: <https://docs.anthropic.com/en/api/versioning>
- Beta headers: <https://docs.anthropic.com/en/api/beta-headers>
- Errors: <https://docs.anthropic.com/en/api/errors>

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

From the official “Basic Example” (`/en/api/overview`):

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

From the official “Basic Example” (`/en/api/overview`):

- Required fields:
  - `model` (string)
  - `max_tokens` (int)
  - `messages` (array)

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
- Fixtures/tests:
  - `siumai/tests/fixtures/anthropic/messages-stream/*`
  - `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`
  - Cross-protocol gateway:
    - `docs/streaming-bridge-alignment.md`
    - `siumai/tests/transcoding_*_to_anthropic_alignment_test.rs`

## Error format + retryable overload

From the official Errors doc:

- Error envelopes are JSON and include an error type + message.
- Overload scenarios are described as `overloaded_error` (service overloaded).

### Siumai mapping

- Error classification:
  - `siumai-protocol-anthropic/src/standards/anthropic/errors.rs`
- Retry semantics:
  - `overloaded_error` is treated as retryable (Siumai uses a synthetic `529` internally for parity).
- Tests:
  - Fixture-driven error alignment tests (see `docs/vercel-ai-fixtures-alignment.md`)

## Status

- This area is treated as **Green**: official endpoint/headers/streaming/errors are covered by
  fixture parity + targeted tests.

