# DeepSeek Provider Support

- Provider identity: `deepseek`
- Technical platforms: `deepseek-api` and `deepseek-beta-api`
- Language modes: Chat Completions, Responses, and Anthropic-compatible Messages
- Protocol evidence refreshed: 2026-08-14
- Latest model-ID evidence: 2026-10-25
- Provider crate: `siumai-provider-deepseek`

DeepSeek model identifiers are open caller input. Known model constants and profile catalog entries
are dated advisories, not runtime allowlists. Future, private, and routed model identifiers remain
callable when they satisfy the selected protocol's structural requirements.

## Model-ID advisories

As checked on 2026-10-25, the official model catalog names `deepseek-flash` as the current Flash
identifier. `DEEPSEEK_FLASH` is available in the provider's public `models` module and the facade's
`siumai::providers::deepseek::models` module. The existing ordinary Chat Completions, Responses,
and Messages advisory catalogs include this identifier for basic generation and streaming through
their existing codecs. The beta Chat catalog is unchanged; open model IDs remain constructible
there without a new named advisory claim.

The old `DEEPSEEK_V4_FLASH` constant retains the literal `deepseek-v4-flash`; the provider still
accepts that name for compatibility routing. V4 Pro's existing advisory state is unchanged.
Only the new Flash row carries the independent 2026-10-25 model-ID evidence. Existing model rows
and protocol compatibility claims retain their previous evidence dates. This refresh does not
add vision support, request parameters, or new protocol behavior.

## Responses API

The Responses protocol contract was verified on 2026-08-14. The model-ID advisory checked on
2026-10-25 adds `deepseek-flash` alongside the retained `deepseek-v4-flash` and `deepseek-v4-pro`
entries in the Responses profile and public advisory catalog. The runtime does not reject a model
merely because it is absent from that dated catalog.

Siumai retains provider-owned validation for the implemented Responses wire contract. Unsupported
stateful and service-owned controls fail before submission, while supported reasoning, logprobs,
caller identity, and provider-executed tool controls reach `POST /v1/responses`.

## Evidence

| Exact slice | Official source | Verified |
|---|---|---|
| Responses model IDs, request fields, tools, and response shape | https://api-docs.deepseek.com/guides/responses_api/ | 2026-08-14 |
| Flash model ID and retained legacy-name routing | https://api-docs.deepseek.com/quick_start/pricing/ | 2026-10-25 |
| Flash identifier in the existing Responses mode | https://api-docs.deepseek.com/guides/responses_api/ | 2026-10-25 |
| Flash identifier in the existing Messages mode | https://api-docs.deepseek.com/guides/anthropic_api/ | 2026-10-25 |

Deterministic offline fixtures cover the new Flash identifier reaching all three basic language
wire formats, plus V4 Pro and a private future model reaching the final Responses wire and decoding
a completed response. A separate negative fixture preserves rejection of an unsupported stateful
Responses field before transport submission.
