# Amazon Bedrock Official API Alignment (Converse + Rerank)

This document records **official API correctness checks** for Amazon Bedrock,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/provider-implementation-alignment.md` (global provider audit checklist)

## Sources (AWS docs)

- Bedrock Runtime API reference (Converse / ConverseStream): <https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html>
- Bedrock Agent Runtime API reference (Rerank): <https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html>

## Siumai implementation (where to compare)

- Provider crate: `siumai-provider-amazon-bedrock/src/*`
- Chat (Converse): `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs`
- Rerank: `siumai-provider-amazon-bedrock/src/standards/bedrock/rerank.rs`
- Error mapping: `siumai-provider-amazon-bedrock/src/standards/bedrock/errors.rs`

## Base URL + endpoints (official)

Bedrock uses region-scoped AWS endpoints. Two common services are involved:

- Bedrock Runtime (chat):
  - `POST /model/{modelId}/converse`
  - `POST /model/{modelId}/converse-stream`
- Bedrock Agent Runtime (rerank):
  - `POST /rerank`

### Siumai mapping

- Siumai treats `ProviderContext.base_url` as the fully-resolved service endpoint (user-provided).
- Chat URLs:
  - `BedrockChatSpec::chat_url` builds `/model/{model}/converse` or `/converse-stream`.
- Rerank URL:
  - `BedrockRerankSpec::rerank_url` builds `/rerank`.

## Authentication (official)

Most Bedrock HTTP requests require **AWS SigV4 signing**.

### Siumai behavior (intentional)

To keep the provider lightweight and fixture-aligned, Siumai does not implement SigV4 signing internally.

Supported auth strategies:

- Inject signed headers via `ProviderContext.http_extra_headers`.
- Optionally send a bearer token via `Authorization: Bearer ...` (best-effort; not the primary AWS path).

See:

- `BedrockChatSpec::build_headers`
- `BedrockRerankSpec::build_headers`

## Rerank model ARN mapping

Official rerank requests use a model identifier/ARN.

### Siumai mapping

Siumai builds a foundation-model ARN from:

- request `model` (treated as foundation model id)
- `providerOptions.bedrock.region` (defaults to `us-east-1` for fixtures)

See: `BedrockRerankRequestTransformer` in `siumai-provider-amazon-bedrock/src/standards/bedrock/rerank.rs`.

## Tests (how correctness is locked down)

- Chat request fixtures: `siumai/tests/bedrock_chat_request_fixtures_alignment_test.rs`
- Chat response: `siumai/tests/bedrock_chat_response_alignment_test.rs`
- Chat streaming: `siumai/tests/bedrock_chat_stream_alignment_test.rs`
- HTTP errors: `siumai/tests/bedrock_http_error_fixtures_alignment_test.rs`
- Rerank response: `siumai/tests/bedrock_rerank_response_alignment_test.rs`

## Status

- Bedrock request/response mapping is treated as **Green for Vercel fixture parity**.
- Auth is **Out-of-scope** (SigV4 must be provided by the caller via headers/middleware).

