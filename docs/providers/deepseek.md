# DeepSeek Provider Support

- Provider identity: `deepseek`
- Technical platforms: `deepseek-api` and `deepseek-beta-api`
- Language modes: Chat Completions, Responses, and Anthropic-compatible Messages
- Evidence refreshed: 2026-08-14
- Provider crate: `siumai-provider-deepseek`

DeepSeek model identifiers are open caller input. Known model constants and profile catalog entries
are dated advisories, not runtime allowlists. Future, private, and routed model identifiers remain
callable when they satisfy the selected protocol's structural requirements.

## Responses API

The official Responses guide verified on 2026-08-14 lists both `deepseek-v4-flash` and
`deepseek-v4-pro` for the `model` field. Both are represented in the verified Responses profile and
the public advisory catalog. The runtime does not reject a model merely because it is absent from
that dated catalog.

Siumai retains provider-owned validation for the implemented Responses wire contract. Unsupported
stateful and service-owned controls fail before submission, while supported reasoning, logprobs,
caller identity, and provider-executed tool controls reach `POST /v1/responses`.

## Evidence

| Exact slice | Official source | Verified |
|---|---|---|
| Responses model IDs, request fields, tools, and response shape | https://api-docs.deepseek.com/guides/responses_api/ | 2026-08-14 |

Deterministic offline fixtures cover V4 Pro and a private future model reaching the final Responses
wire and decoding a completed response. A separate negative fixture preserves rejection of an
unsupported stateful Responses field before transport submission.
