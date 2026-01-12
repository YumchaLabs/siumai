# Azure OpenAI Official API Alignment

This document records **official API correctness checks** for Azure OpenAI,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/alignment/official/openai-official-api-alignment.md` (shared OpenAI protocol family details)

## Sources (Microsoft docs)

- Azure OpenAI REST API reference (service docs): <https://learn.microsoft.com/azure/ai-services/openai/reference>

## Siumai implementation (where to compare)

- Provider: `siumai-provider-azure/src/providers/azure_openai/spec.rs`
- Shared protocol family (Chat + Responses + SSE): `siumai-protocol-openai/src/standards/openai/*`

## URL shape + api-version (official)

Azure OpenAI uses an Azure resource host and requires an `api-version` query parameter.

Vercel AI SDK (and Siumai’s parity layer) supports two URL routing modes:

- v1-style routing (default):
  - Base URL: `https://{resource}.openai.azure.com/openai`
  - Full URL: `${base}/v1{path}?api-version={apiVersion}`
- Deployment-based routing (legacy/compat):
  - Full URL: `${base}/deployments/{deploymentId}{path}?api-version={apiVersion}`

### Siumai mapping

- URL builder: `AzureOpenAiSpec::build_url` in `siumai-provider-azure/src/providers/azure_openai/spec.rs`
- Default `api_version`: `v1` (Vercel-aligned; can be overridden)
- Deployment-based URLs: `AzureUrlConfig.use_deployment_based_urls`

## Authentication (official)

- Azure uses `api-key: <key>` (not `Authorization: Bearer ...`) for most OpenAI-compatible surfaces.

### Siumai mapping

- Headers: `build_azure_openai_json_headers` in `siumai-provider-azure/src/providers/azure_openai/spec.rs`
- Custom headers: `ProviderContext.http_extra_headers` merged in (user headers override defaults)

## Endpoints covered by Siumai (parity focus)

Siumai routes Azure operations through the shared OpenAI protocol family (with Azure-specific URL/header rules):

- Chat Completions: `POST /chat/completions`
- Responses: `POST /responses`
- Embeddings: `POST /embeddings`
- Images: `POST /images/generations`
- Audio:
  - `POST /audio/speech`
  - `POST /audio/transcriptions`
- Files: `POST /files`, `GET /files`, `GET /files/{id}`, `DELETE /files/{id}` (+ `api-version`)

## Tests (how correctness is locked down)

- URL building: `siumai/tests/azure_openai_provider_url_fixtures_alignment_test.rs`
- Request mapping: `siumai/tests/azure_openai_provider_request_fixtures_alignment_test.rs`
- Streaming parity: `siumai/tests/azure_*_stream_alignment_test.rs`

## Status

- Azure OpenAI is treated as **Green** for Vercel parity (fixtures) and URL/header correctness.
- Remaining work (if needed): track Azure-specific Responses API preview deltas via additional fixtures.
