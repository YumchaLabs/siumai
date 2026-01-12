# DeepSeek Official API Alignment (Chat Completions)

This document records **official API correctness checks** for DeepSeek’s chat API surface,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/openai-official-api-alignment.md` (shared OpenAI protocol family details)
- `docs/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)

## Sources (DeepSeek docs)

- DeepSeek API docs: <https://api-docs.deepseek.com/>

## Vercel reference (parity target)

- Provider package: `repo-ref/ai/packages/deepseek/src/*`
- Default base URL (Vercel): `https://api.deepseek.com`
- Chat endpoint (Vercel): `POST /chat/completions`
  - See: `repo-ref/ai/packages/deepseek/src/chat/deepseek-chat-language-model.ts`

## Siumai implementation (where to compare)

DeepSeek can be used in two equivalent ways:

1. OpenAI-compatible preset (recommended for parity):
   - Preset config: `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs` (`deepseek`)
   - Shared protocol family: `siumai-protocol-openai/src/standards/openai/*`
2. Thin provider wrapper (Vercel granularity):
   - `siumai-provider-deepseek/src/*` (wraps the OpenAI-compatible preset)

## Base URL + endpoint

Siumai follows Vercel’s base URL convention for DeepSeek:

- Default base URL: `https://api.deepseek.com`
- Endpoint: `POST {base}/chat/completions`

If you need a different prefix (self-hosted gateway, proxy, or a `/v1`-style mount), override it via:

- `Siumai::builder().openai().deepseek().base_url(\"...\")`
- or `registry` options / provider config.

## Authentication

DeepSeek uses OpenAI-style bearer auth:

- `Authorization: Bearer <DEEPSEEK_API_KEY>`

### Siumai mapping

DeepSeek uses the shared OpenAI-compatible headers builder from the OpenAI protocol family.

## Response + streaming parity (fixtures)

DeepSeek is aligned via Vercel fixtures copied into this repo:

- Fixtures: `siumai/tests/fixtures/deepseek/chat/*`
- Response parsing tests: `siumai/tests/deepseek_chat_response_alignment_test.rs`
- Streaming tests: `siumai/tests/deepseek_chat_stream_alignment_test.rs`

## Status

- DeepSeek is treated as **Green** for Vercel fixture parity (response + streaming).
- Default base URL is aligned to Vercel (`https://api.deepseek.com`).

