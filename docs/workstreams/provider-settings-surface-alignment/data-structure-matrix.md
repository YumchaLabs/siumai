# Provider Settings Surface Alignment - Data Structure Matrix

Last updated: 2026-04-24

Status legend:

- `supported` = exposed on the Rust package/provider surface and backed by real runtime behavior
- `deferred` = upstream field exists, but Siumai does not yet have an honest analogue

## OpenAI Compatible

Reference: `repo-ref/ai/packages/openai-compatible/src/openai-compatible-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `OpenAICompatibleProviderSettings.base_url`, required by `new(name, base_url)` |
| `name` | supported | `OpenAICompatibleProviderSettings.name`; Rust uses it as the generic provider label and prefixes the internal provider id only when it would collide with a built-in preset |
| `apiKey` | supported | `OpenAICompatibleProviderSettings.api_key`; optional like AI SDK, with unauthenticated generic gateways represented by `auth_required = false` |
| `headers` | supported | `OpenAICompatibleProviderSettings.headers` |
| `queryParams` | supported | `OpenAICompatibleProviderSettings.query_params` |
| `fetch` | supported | `OpenAICompatibleProviderSettings.fetch` |
| `includeUsage` | supported | `OpenAICompatibleProviderSettings.include_usage` |
| `supportsStructuredOutputs` | supported | `OpenAICompatibleProviderSettings.supports_structured_outputs` |
| `transformRequestBody` | supported | `OpenAICompatibleProviderSettings.transform_request_body`, via `RequestBodyTransformer` |
| `metadataExtractor` | supported | `OpenAICompatibleProviderSettings.metadata_extractor`, via `ResponseMetadataExtractor` |

## OpenAI

Reference: `repo-ref/ai/packages/openai/src/openai-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `OpenAIProviderSettings.base_url` |
| `apiKey` | supported | `OpenAIProviderSettings.api_key` |
| `organization` | supported | `OpenAIProviderSettings.organization` |
| `project` | supported | `OpenAIProviderSettings.project` |
| `headers` | supported | `OpenAIProviderSettings.headers` |
| `fetch` | supported | `OpenAIProviderSettings.fetch` |
| `name` | deferred | Siumai keeps canonical `openai` provider identity fixed; no honest display-only alias yet |

## Anthropic

Reference: `repo-ref/ai/packages/anthropic/src/anthropic-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `AnthropicProviderSettings.base_url` |
| `apiKey` | supported | `AnthropicProviderSettings.api_key` |
| `authToken` | supported | `AnthropicProviderSettings.auth_token`, emitted as `Authorization: Bearer ...` without forcing `x-api-key` |
| `headers` | supported | `AnthropicProviderSettings.headers` |
| `fetch` | supported | `AnthropicProviderSettings.fetch` |
| `generateId` | deferred | current Anthropic runtime does not own a comparable provider-level stable ID hook |
| `name` | deferred | Siumai keeps canonical `anthropic` provider identity fixed; no honest display-only alias yet |

## Google

Reference: `repo-ref/ai/packages/google/src/google-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `GoogleProviderSettings.base_url` |
| `apiKey` | supported | `GoogleProviderSettings.api_key`, with `GOOGLE_GENERATIVE_AI_API_KEY` fallback |
| `headers` | supported | `GoogleProviderSettings.headers`; Rust models static headers directly |
| `fetch` | supported | `GoogleProviderSettings.fetch` |
| `generateId` | supported | `GoogleProviderSettings.generate_id`, shared by chat/video ID generation paths |
| `name` | supported | `GoogleProviderSettings.name`, defaulting to `google.generative-ai` |

## Azure OpenAI

Reference: `repo-ref/ai/packages/azure/src/azure-openai-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `resourceName` | supported | `AzureOpenAIProviderSettings.resource_name` -> derived Azure base URL |
| `baseURL` | supported | `AzureOpenAIProviderSettings.base_url` |
| `apiKey` | supported | `AzureOpenAIProviderSettings.api_key` |
| `headers` | supported | `AzureOpenAIProviderSettings.headers` |
| `fetch` | supported | `AzureOpenAIProviderSettings.fetch` |
| `apiVersion` | supported | `AzureOpenAIProviderSettings.api_version` |
| `useDeploymentBasedUrls` | supported | `AzureOpenAIProviderSettings.use_deployment_based_urls` |

## Amazon Bedrock

Reference: `repo-ref/ai/packages/amazon-bedrock/src/bedrock-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `region` | supported | `AmazonBedrockProviderSettings.region` |
| `apiKey` | supported | `AmazonBedrockProviderSettings.api_key` |
| `baseURL` | supported | `AmazonBedrockProviderSettings.base_url` |
| `headers` | supported | `AmazonBedrockProviderSettings.headers` |
| `fetch` | supported | `AmazonBedrockProviderSettings.fetch` |
| `accessKeyId` | deferred | no first-class SigV4 credential carrier on the Rust Bedrock runtime yet |
| `secretAccessKey` | deferred | same as above |
| `sessionToken` | deferred | same as above |
| `credentialProvider` | deferred | no first-class async AWS credential-provider abstraction yet |
| `generateId` | deferred | current Bedrock runtime does not own a comparable provider-level ID hook |

## Cohere

Reference: `repo-ref/ai/packages/cohere/src/cohere-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `CohereProviderSettings.base_url` |
| `apiKey` | supported | `CohereProviderSettings.api_key` |
| `headers` | supported | `CohereProviderSettings.headers` |
| `fetch` | supported | `CohereProviderSettings.fetch` |
| `generateId` | deferred | current Cohere runtime does not own a comparable provider-level stable ID hook |

## DeepSeek

Reference: `repo-ref/ai/packages/deepseek/src/deepseek-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `apiKey` | supported | `DeepSeekProviderSettings.api_key` |
| `baseURL` | supported | `DeepSeekProviderSettings.base_url` |
| `headers` | supported | `DeepSeekProviderSettings.headers` |
| `fetch` | supported | `DeepSeekProviderSettings.fetch` |

## TogetherAI

Reference: `repo-ref/ai/packages/togetherai/src/togetherai-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `apiKey` | supported | `TogetherAIProviderSettings.api_key` |
| `baseURL` | supported | `TogetherAIProviderSettings.base_url` |
| `headers` | supported | `TogetherAIProviderSettings.headers` |
| `fetch` | supported | `TogetherAIProviderSettings.fetch` |

## DeepInfra

Reference: `repo-ref/ai/packages/deepinfra/src/deepinfra-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `apiKey` | supported | `DeepInfraProviderSettings.api_key` |
| `baseURL` | supported | `DeepInfraProviderSettings.base_url` -> text-family runtime base URL normalized with `/openai` |
| `headers` | supported | `DeepInfraProviderSettings.headers` |
| `fetch` | supported | `DeepInfraProviderSettings.fetch` |

## Google Vertex MaaS

Reference:

- `repo-ref/ai/packages/google-vertex/src/maas/google-vertex-maas-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/maas/google-vertex-maas-provider-node.ts`
- `repo-ref/ai/packages/google-vertex/src/maas/google-vertex-maas-options.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `project` | supported | `GoogleVertexMaasProviderSettings.project`, with `GOOGLE_VERTEX_PROJECT` fallback |
| `location` | supported | `GoogleVertexMaasProviderSettings.location`, with `GOOGLE_VERTEX_LOCATION` then `global` fallback |
| `baseURL` | supported | `GoogleVertexMaasProviderSettings.base_url`; otherwise derived as `/v1/projects/{project}/locations/{location}/endpoints/openapi` |
| `headers` | supported | `GoogleVertexMaasProviderSettings.headers`; Rust supports static headers directly on config/builder |
| `fetch` | supported | `GoogleVertexMaasProviderSettings.fetch` |
| `googleAuthOptions` | Rust analogue supported | `GoogleVertexMaasProviderSettings.token_provider`; this intentionally does not model Node's `google-auth-library` options object |

## Google Vertex

Reference:

- `repo-ref/ai/packages/google-vertex/src/google-vertex-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-provider-node.ts`
- `repo-ref/ai/packages/google-vertex/src/edge/google-vertex-provider-edge.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `apiKey` | supported | `GoogleVertexProviderSettings.api_key`, with `GOOGLE_VERTEX_API_KEY` fallback and express-mode base URL behavior |
| `location` | supported | `GoogleVertexProviderSettings.location`, with `GOOGLE_VERTEX_LOCATION` fallback |
| `project` | supported | `GoogleVertexProviderSettings.project`, with `GOOGLE_VERTEX_PROJECT` fallback |
| `headers` | supported | `GoogleVertexProviderSettings.headers`; Rust models static headers directly |
| `fetch` | supported | `GoogleVertexProviderSettings.fetch` |
| `generateId` | supported | `GoogleVertexProviderSettings.generate_id`, shared by chat/image/video ID generation paths |
| `baseURL` | supported | `GoogleVertexProviderSettings.base_url`; otherwise derived from express mode or project/location enterprise mode |
| `googleAuthOptions` | Rust analogue supported | `GoogleVertexProviderSettings.token_provider`; effective API key / express mode intentionally suppresses token-provider auth to match the Node wrapper |
| `googleCredentials` | Rust analogue supported | same `token_provider` analogue for the edge wrapper credential object |

## Google Vertex Anthropic

Reference:

- `repo-ref/ai/packages/google-vertex/src/anthropic/google-vertex-anthropic-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/anthropic/google-vertex-anthropic-provider-node.ts`
- `repo-ref/ai/packages/google-vertex/src/anthropic/edge/google-vertex-anthropic-provider-edge.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `project` | supported | `GoogleVertexAnthropicProviderSettings.project`, with `GOOGLE_VERTEX_PROJECT` fallback through the builder/config path |
| `location` | supported | `GoogleVertexAnthropicProviderSettings.location`, with `GOOGLE_VERTEX_LOCATION` fallback through the builder/config path |
| `baseURL` | supported | `GoogleVertexAnthropicProviderSettings.base_url`; otherwise derives the `/publishers/anthropic/models` base URL |
| `headers` | supported | `GoogleVertexAnthropicProviderSettings.headers`; Rust models static headers directly |
| `fetch` | supported | `GoogleVertexAnthropicProviderSettings.fetch` |
| `googleAuthOptions` | Rust analogue supported | `GoogleVertexAnthropicProviderSettings.token_provider`; this intentionally does not model Node's `google-auth-library` options object |
| `googleCredentials` | Rust analogue supported | same `token_provider` analogue for the edge wrapper credential object |

## MoonshotAI

Reference: `repo-ref/ai/packages/moonshotai/src/moonshotai-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `apiKey` | supported | `MoonshotAIProviderSettings.api_key` |
| `baseURL` | supported | `MoonshotAIProviderSettings.base_url` |
| `headers` | supported | `MoonshotAIProviderSettings.headers` |
| `fetch` | supported | `MoonshotAIProviderSettings.fetch` |
## Fireworks

Reference: `repo-ref/ai/packages/fireworks/src/fireworks-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `apiKey` | supported | `FireworksProviderSettings.api_key` |
| `baseURL` | supported | `FireworksProviderSettings.base_url` |
| `headers` | supported | `FireworksProviderSettings.headers` |
| `fetch` | supported | `FireworksProviderSettings.fetch` |
## Perplexity

Reference: `repo-ref/ai/packages/perplexity/src/perplexity-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `PerplexityProviderSettings.base_url` |
| `apiKey` | supported | `PerplexityProviderSettings.api_key` |
| `headers` | supported | `PerplexityProviderSettings.headers` |
| `fetch` | supported | `PerplexityProviderSettings.fetch` |

Note: upstream Perplexity imports `generateId` internally, but `generateId` is not a field on
`PerplexityProviderSettings`; Siumai should not track it as a deferred Perplexity settings gap.
## Mistral

Reference: `repo-ref/ai/packages/mistral/src/mistral-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `MistralProviderSettings.base_url` |
| `apiKey` | supported | `MistralProviderSettings.api_key` |
| `headers` | supported | `MistralProviderSettings.headers` |
| `fetch` | supported | `MistralProviderSettings.fetch` |
| `generateId` | deferred | shared OpenAI-compatible runtime does not yet own a comparable provider-level stable ID hook |
## Groq

Reference: `repo-ref/ai/packages/groq/src/groq-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `GroqProviderSettings.base_url` |
| `apiKey` | supported | `GroqProviderSettings.api_key` |
| `headers` | supported | `GroqProviderSettings.headers` |
| `fetch` | supported | `GroqProviderSettings.fetch` |
## xAI

Reference: `repo-ref/ai/packages/xai/src/xai-provider.ts`

| Upstream field | Rust status | Rust analogue / note |
| --- | --- | --- |
| `baseURL` | supported | `XaiProviderSettings.base_url` |
| `apiKey` | supported | `XaiProviderSettings.api_key` |
| `headers` | supported | `XaiProviderSettings.headers` |
| `fetch` | supported | `XaiProviderSettings.fetch` |

Note: upstream xAI imports `generateId` internally, but `generateId` is not a field on
`XaiProviderSettings`; Siumai should not track it as a deferred xAI settings gap.
## Package exports

The audited package-surface alignment now also exposes:

| Provider | Rust export |
| --- | --- |
| OpenAI | `provider_ext::openai::{OpenAIProviderSettings, VERSION}` |
| Anthropic | `provider_ext::anthropic::{AnthropicProviderSettings, VERSION}` |
| Azure | `provider_ext::azure::{AzureOpenAIProviderSettings, VERSION}` |
| Bedrock | `provider_ext::bedrock::{AmazonBedrockProviderSettings, VERSION}` |
| Cohere | `provider_ext::cohere::{CohereProviderSettings, VERSION}` |
| DeepSeek | `provider_ext::deepseek::{DeepSeekProviderSettings, VERSION}` |
| TogetherAI | `provider_ext::togetherai::{TogetherAIProviderSettings, VERSION}` |
| xAI | `provider_ext::xai::{XaiProviderSettings, VERSION}` |
| Groq | `provider_ext::groq::{GroqProviderSettings, VERSION}` |
| Google | `provider_ext::google::{GoogleProviderSettings, GoogleGenerativeAIProviderSettings, VERSION}` |
| Google Vertex | `provider_ext::google_vertex::{GoogleVertexProviderSettings, VERSION}` |
| Google Vertex Anthropic | `provider_ext::anthropic_vertex::{GoogleVertexAnthropicProviderSettings, VERSION}` |
| Mistral | `provider_ext::mistral::{MistralProviderSettings, VERSION}` |
| Perplexity | `provider_ext::perplexity::{PerplexityProviderSettings, VERSION}` |
| Fireworks | `provider_ext::fireworks::{FireworksProviderSettings, VERSION}` |
| MoonshotAI | `provider_ext::moonshotai::{MoonshotAIProviderSettings, VERSION}` |
| DeepInfra | `provider_ext::deepinfra::{DeepInfraProviderSettings, VERSION}` |
| Google Vertex MaaS | `provider_ext::vertex_maas::{GoogleVertexMaasProviderSettings, VERSION}` |
