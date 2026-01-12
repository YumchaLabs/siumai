# xAI Official API Alignment (OpenAI-compatible Chat + Responses)

This document records **official API correctness checks** for the xAI Enterprise API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/provider-defined-tools-alignment.md` (tool id conventions + provider-defined tools)

## Sources (xAI docs)

- REST API Reference: <https://docs.x.ai/docs/api-reference>

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

## Provider-defined tools (xAI specifics)

xAI’s Responses API supports provider tools (e.g. search tools) that are represented as
provider-defined tool ids in Vercel AI SDK fixtures.

### Siumai mapping

- Tool naming + request mapping parity is locked down via fixtures:
  - `siumai/tests/xai_responses_request_tool_mapping_test.rs`
  - `siumai/tests/xai_responses_web_search_stream_alignment_test.rs`
  - `siumai/tests/xai_responses_x_search_stream_alignment_test.rs`

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

- xAI is treated as **Green** for official correctness:
  base URL and endpoints, auth header format, OpenAI-compatible error envelope, and
  Responses API request/stream mapping validated by fixtures.
