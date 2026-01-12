# Cohere Official API Alignment (Rerank v2)

This document records **official API correctness checks** for Cohere Rerank,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/provider-implementation-alignment.md` (global provider audit checklist)

## Sources (Cohere docs)

- Rerank (v2): <https://docs.cohere.com/v2/reference/rerank>

## Siumai implementation (where to compare)

- Provider crate: `siumai-provider-cohere/src/*`
- Standard mapping: `siumai-provider-cohere/src/standards/cohere/rerank.rs`
- Error mapping: `siumai-provider-cohere/src/standards/cohere/errors.rs`

## Base URL + endpoint (official)

- Base URL: `https://api.cohere.com/v2`
- Endpoint: `POST /rerank`

### Siumai mapping

- URL: `CohereRerankSpec::rerank_url` uses `join_url(base_url, "/rerank")`
- Default `base_url` is declared in `siumai-registry/src/native_provider_metadata.rs`

## Authentication (official)

- `Authorization: Bearer <COHERE_API_KEY>`
- `Content-Type: application/json`

### Siumai mapping

- Headers: `CohereRerankSpec::build_headers`
- Custom headers: `ProviderContext.http_extra_headers` merged in (user headers override defaults)

## Request/response mapping highlights

- Request fields supported directly:
  - `model`, `query`, `documents`, `top_n`
- Provider options (`providerOptions.cohere`) supported (best-effort):
  - `maxTokensPerDoc` / `max_tokens_per_doc`
  - `priority`

See: `CohereRerankRequestTransformer` in `siumai-provider-cohere/src/standards/cohere/rerank.rs`.

## Tests (how correctness is locked down)

- Fixtures: `siumai/tests/fixtures/cohere/*`
- Rerank fixtures: `siumai/tests/cohere_rerank_fixtures_alignment_test.rs`
- HTTP errors: `siumai/tests/cohere_http_error_fixtures_alignment_test.rs`

## Status

- Cohere rerank is treated as **Green** for fixture parity and endpoint/header correctness.

