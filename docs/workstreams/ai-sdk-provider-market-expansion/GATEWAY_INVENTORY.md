# Vercel AI Gateway Contract Inventory

Status: Closed
Last updated: 2026-05-26

Task: PMX-020

## Summary

`@ai-sdk/gateway` is not an OpenAI-compatible provider preset. It is a remote implementation of the
AI SDK provider protocol with dedicated endpoints per model family. The package is pulled by the top-level
`ai` package and is the default provider when AI SDK resolves string model ids without a custom global provider.

Siumai should treat Gateway as a native provider boundary, not as an OpenAI-compatible vendor. The first viable
implementation slice should be language plus embedding, with typed Gateway provider options and no-network
request/stream tests. Image, video, rerank, speech, transcription, OIDC, and admin metadata calls should stay
explicit follow-ons unless the first slice proves cheap.

## Upstream Files

| Area | Upstream file |
| --- | --- |
| Provider factory and auth | `repo-ref/ai/packages/gateway/src/gateway-provider.ts` |
| Language family | `repo-ref/ai/packages/gateway/src/gateway-language-model.ts` |
| Embedding family | `repo-ref/ai/packages/gateway/src/gateway-embedding-model.ts` |
| Image family | `repo-ref/ai/packages/gateway/src/gateway-image-model.ts` |
| Video family | `repo-ref/ai/packages/gateway/src/gateway-video-model.ts` |
| Reranking family | `repo-ref/ai/packages/gateway/src/gateway-reranking-model.ts` |
| Speech family | `repo-ref/ai/packages/gateway/src/gateway-speech-model.ts` |
| Transcription family | `repo-ref/ai/packages/gateway/src/gateway-transcription-model.ts` |
| Request-scoped provider options | `repo-ref/ai/packages/gateway/src/gateway-provider-options.ts` |
| Model metadata resources | `repo-ref/ai/packages/gateway/src/gateway-fetch-metadata.ts` |
| Spend report resource | `repo-ref/ai/packages/gateway/src/gateway-spend-report.ts` |
| Generation info resource | `repo-ref/ai/packages/gateway/src/gateway-generation-info.ts` |
| Provider-defined tools | `repo-ref/ai/packages/gateway/src/gateway-tools.ts` |
| Header constants | `repo-ref/ai/packages/gateway/src/gateway-headers.ts` |

## Provider Identity And Settings

Provider id: `gateway`

Default base URL:

```text
https://ai-gateway.vercel.sh/v4/ai
```

Provider settings:

| AI SDK setting | Meaning | Siumai recommendation |
| --- | --- | --- |
| `baseURL` | Base URL prefix for model-family endpoints. | `GatewayConfig.base_url`, defaulting to upstream base URL. |
| `apiKey` | Bearer token, default env `AI_GATEWAY_API_KEY`. | Support explicit key and env lookup in first slice. |
| `teamIdOrSlug` | Team scope header. | Support as optional config field/header. |
| `headers` | Custom request headers. | Use Siumai `HttpConfig` / provider config header merge. |
| `fetch` | JS fetch override. | Map to Siumai transport override; no public fetch analogue needed. |
| `metadataCacheRefreshMillis` | Cache TTL for `getAvailableModels`. | Defer until metadata resources are implemented. |
| `_internal.currentDate` | Test-only clock. | No public Rust equivalent. |

Auth behavior:

- Prefer explicit `apiKey`, then `AI_GATEWAY_API_KEY`.
- Upstream falls back to Vercel OIDC when no API key is present.
- Initial Siumai support should defer OIDC because it depends on Vercel runtime environment behavior and is not needed
  for ordinary Rust usage. Emit a clear missing-API-key error instead.

Required provider-level headers:

| Header | Source |
| --- | --- |
| `Authorization: Bearer <token>` | API key or OIDC token |
| `ai-gateway-protocol-version: 0.0.1` | Fixed upstream constant |
| `ai-gateway-auth-method: api-key` or `oidc` | Auth mode |
| `x-vercel-ai-gateway-team: <team>` | Optional `teamIdOrSlug` |

Observability headers:

| Header | Upstream environment |
| --- | --- |
| `ai-o11y-deployment-id` | `VERCEL_DEPLOYMENT_ID` |
| `ai-o11y-environment` | `VERCEL_ENV` |
| `ai-o11y-region` | `VERCEL_REGION` |
| `ai-o11y-project-id` | `VERCEL_PROJECT_ID` |
| `ai-o11y-request-id` | Vercel request id helper |

Initial Siumai implementation can support explicit headers and defer automatic Vercel environment discovery.

## Model Family Endpoints

All model execution requests are JSON POSTs to a family endpoint under `baseURL`.

| Family | Endpoint | Required model headers | Initial Siumai priority |
| --- | --- | --- | --- |
| Language | `/language-model` | `ai-language-model-specification-version: 4`, `ai-language-model-id`, `ai-language-model-streaming` | P0 proof |
| Embedding | `/embedding-model` | `ai-embedding-model-specification-version: 4`, `ai-model-id` | P0 proof |
| Image | `/image-model` | `ai-image-model-specification-version: 4`, `ai-model-id` | Follow-on |
| Video | `/video-model` | `ai-video-model-specification-version: 4`, `ai-model-id`, `accept: text/event-stream` | Follow-on |
| Reranking | `/reranking-model` | `ai-reranking-model-specification-version: 4`, `ai-model-id` | Follow-on |
| Speech | `/speech-model` | `ai-speech-model-specification-version: 4`, `ai-model-id` | Follow-on |
| Transcription | `/transcription-model` | `ai-transcription-model-specification-version: 4`, `ai-model-id` | Follow-on |

Gateway model id unions in the local AI SDK reference:

| Family | Current typed ids | Notes |
| --- | ---: | --- |
| Language | 190 | Includes `openai/*`, `anthropic/*`, `google/*`, `mistral/*`, `deepseek/*`, `alibaba/*`, and others. |
| Embedding | 24 | Includes Alibaba, Amazon, Cohere, Google, Mistral, OpenAI, Voyage. |
| Image | 30 | Includes Google Imagen, OpenAI image models, Recraft, and others. |
| Video | 26 | Includes Google Veo and other video providers. |
| Reranking | 5 | Cohere and Voyage. |
| Speech | Open string | `string & {}` upstream. |
| Transcription | Open string | `string & {}` upstream. |

Do not mirror all Gateway model ids into a hard required Rust enum. Gateway is a router and supports opaque future ids.
Use string model ids with optional grouped constants for popular ids once implementation exists.

## Language Request And Response Shape

Language requests:

- Accept AI SDK `LanguageModelV4CallOptions`.
- Remove `abortSignal` before serialization.
- Base64-encode inline file part bytes.
- Merge user headers, provider auth headers, model config headers, and observability headers.
- For streaming, set `ai-language-model-streaming: true`; otherwise `false`.

Language non-stream response:

- Successful response is arbitrary JSON parsed as `LanguageModelV4GenerateResult`.
- Upstream returns response headers and raw response body to the caller.
- Warnings are currently always empty on request construction.

Language stream response:

- Successful response is SSE parsed as JSON `LanguageModelV4StreamPart`.
- Raw chunks are skipped unless `includeRawChunks` is requested.
- `response-metadata.timestamp` string values are converted to `Date`.

Siumai implication:

- This requires an AI SDK V4 provider-protocol request/stream serializer, not an OpenAI/Anthropic/Gemini wire serializer.
- Existing generated-output and stream-part alignment work should be reused where possible.
- The first proof should only claim support for the subset of Siumai request parts that already have a loss-aware
  AI SDK V4 representation.

## Embedding Request And Response Shape

Embedding request body:

```json
{
  "values": ["..."],
  "providerOptions": {}
}
```

Embedding response body:

```json
{
  "embeddings": [[0.1, 0.2]],
  "usage": { "tokens": 123 },
  "providerMetadata": {}
}
```

Siumai implication:

- Embedding is a good second family in the first proof because the request/response contract is simple and proves
  non-language registry family wiring.

## Request-Scoped Gateway Options

Upstream exposes these under Gateway provider options:

| Field | Type | Meaning |
| --- | --- | --- |
| `only` | `string[]` | Allowed provider slugs. |
| `order` | `string[]` | Provider fallback order. |
| `sort` | `cost | ttft | tps` | Dynamic provider sorting intent. |
| `user` | `string` | End-user attribution. |
| `tags` | `string[]` | Spend/report tags. |
| `models` | `string[]` | Fallback model ids. |
| `byok` | `Record<string, Array<Record<string, unknown>>>` | Request-scoped credentials. |
| `zeroDataRetention` | `bool` | Restrict to zero-data-retention providers. |
| `disallowPromptTraining` | `bool` | Restrict to providers that do not train on prompts. |
| `hipaaCompliant` | `bool` | Restrict to HIPAA-compliant providers. |
| `quotaEntityId` | `string` | Quota accounting entity. |
| `providerTimeouts.byok` | `Record<string, number>` | BYOK provider timeout in milliseconds. |
| `serviceTier` | `flex | priority` | Gateway-level service-tier intent. |

Siumai should expose these as typed `provider_ext::gateway::options::GatewayOptions` and request extension helpers.
The raw provider-options map can still carry unknown future fields through a compatibility escape hatch.

## Metadata And Admin Resources

Upstream provider methods:

| Method | Endpoint | Notes |
| --- | --- | --- |
| `getAvailableModels()` | `{baseURL}/config` | Cached for `metadataCacheRefreshMillis`, default 5 minutes. |
| `getCredits()` | `{origin}/v1/credits` | Uses base URL origin, not `/v4/ai`. |
| `getSpendReport(params)` | `{origin}/v1/report?...` | Query params are snake_case. |
| `getGenerationInfo({ id })` | `{origin}/v1/generation?id=...` | Returns detailed cost/latency/token data. |

Initial implementation should defer these as `resources::*` until language/embedding request execution is proven.
If implemented, keep them out of `prelude::unified` and expose through `provider_ext::gateway::resources`.

## Provider-Defined Tools

Gateway tools:

- `parallelSearch`
- `perplexitySearch`

Initial implementation should defer provider-defined Gateway tools. They are useful, but they depend on hosted
tool semantics and can be added after language streaming has a stable request/response path.

## Error Surface

Upstream exports typed errors:

- `GatewayAuthenticationError`
- `GatewayInvalidRequestError`
- `GatewayRateLimitError`
- `GatewayModelNotFoundError`
- `GatewayInternalServerError`
- `GatewayResponseError`
- `GatewayTimeoutError`

Initial Siumai support should map these into existing `LlmError` categories with preserved provider metadata.
Exact typed error structs can be added under `provider_ext::gateway::errors` only if users need stable matching.

## Siumai Ownership Recommendation

Preferred first implementation shape:

- New native provider crate: `siumai-provider-gateway`.
- Public facade root: `siumai::provider_ext::gateway`.
- Feature flag: `gateway`.
- Registry provider id: `gateway`.
- First model families:
  - language,
  - embedding.
- First typed options:
  - `GatewayOptions`,
  - `GatewayProviderSettings` / `GatewayConfig`,
  - request extension helpers for language and embedding provider options.

Why not OpenAI-compatible:

- Endpoint paths are AI SDK provider protocol paths, not `/chat/completions`, `/responses`, or `/embeddings`.
- Streaming emits AI SDK stream parts rather than OpenAI SSE frames.
- Gateway routing options are a provider-specific protocol layer.

Why not all families immediately:

- Image, video, speech, transcription, and reranking each have distinct request/response envelopes.
- Video uses SSE even for generate-style calls.
- Media families need careful byte/base64 handling and response metadata decisions.
- Language plus embedding proves the core provider-protocol transport without opening the whole media surface.

## PMX-030 Recommendation

Proceed with PMX-030 as a bounded proof only if the implementer can keep the slice to:

- config and auth headers,
- language non-stream and stream request path,
- embedding request path,
- typed Gateway provider options,
- registry and facade roots,
- no-network tests for URL/header/body/stream part pass-through.

Split a dedicated `vercel-gateway-provider-implementation` workstream if PMX-030 starts requiring:

- full model catalog generation,
- admin resource clients,
- provider-defined tools,
- media families,
- OIDC,
- or broad AI SDK V4 stream-part projection refactors.
