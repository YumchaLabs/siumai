# TogetherAI Official API Alignment (Rerank)

This document records **official API correctness checks** for TogetherAI Rerank,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)

## Sources (TogetherAI docs)

- Rerank: <https://docs.together.ai/reference/rerank-1>

## Siumai implementation (where to compare)

- Provider crate: `siumai-provider-togetherai/src/*`
- Standard mapping: `siumai-provider-togetherai/src/standards/togetherai/rerank.rs`
- Error mapping: `siumai-provider-togetherai/src/standards/togetherai/errors.rs`

## Base URL + endpoint (official)

- Base URL: `https://api.together.xyz/v1`
- Endpoint: `POST /rerank`

### Siumai mapping

- URL: `TogetherAiRerankSpec::rerank_url` uses `join_url(base_url, "/rerank")`
- Default `base_url` is declared in `siumai-registry/src/native_provider_metadata.rs`

## Authentication (official)

- `Authorization: Bearer <TOGETHER_API_KEY>`
- `Content-Type: application/json`

### Siumai mapping

- Headers: `TogetherAiRerankSpec::build_headers`
- Custom headers: `ProviderContext.http_extra_headers` merged in (user headers override defaults)

## Request/response mapping highlights

- Request fields supported directly:
  - `model`, `query`, `documents`, `top_n`
- Siumai always sets `return_documents=false` (Vercel-aligned, smaller payload).
- Provider options (`providerOptions.togetherai`) supported (best-effort):
  - `rankFields` / `rank_fields`

See: `TogetherAiRerankRequestTransformer` in `siumai-provider-togetherai/src/standards/togetherai/rerank.rs`.

## Tests (how correctness is locked down)

- Fixtures: `siumai/tests/fixtures/togetherai/*`
- Rerank fixtures: `siumai/tests/togetherai_rerank_fixtures_alignment_test.rs`
- HTTP errors: `siumai/tests/togetherai_http_error_fixtures_alignment_test.rs`

## Status

- TogetherAI rerank is treated as **Green** for fixture parity and endpoint/header correctness.
