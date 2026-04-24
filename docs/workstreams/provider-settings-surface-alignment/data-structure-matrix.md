# Provider Settings Surface Alignment - Data Structure Matrix

Last updated: 2026-04-24

Status legend:

- `supported` = exposed on the Rust package/provider surface and backed by real runtime behavior
- `deferred` = upstream field exists, but Siumai does not yet have an honest analogue

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
| Azure | `provider_ext::azure::{AzureOpenAIProviderSettings, VERSION}` |
| Bedrock | `provider_ext::bedrock::{AmazonBedrockProviderSettings, VERSION}` |
| Cohere | `provider_ext::cohere::{CohereProviderSettings, VERSION}` |
| DeepSeek | `provider_ext::deepseek::{DeepSeekProviderSettings, VERSION}` |
| TogetherAI | `provider_ext::togetherai::{TogetherAIProviderSettings, VERSION}` |
| xAI | `provider_ext::xai::{XaiProviderSettings, VERSION}` |
| Groq | `provider_ext::groq::{GroqProviderSettings, VERSION}` |
| Mistral | `provider_ext::mistral::{MistralProviderSettings, VERSION}` |
| Perplexity | `provider_ext::perplexity::{PerplexityProviderSettings, VERSION}` |
| Fireworks | `provider_ext::fireworks::{FireworksProviderSettings, VERSION}` |
| MoonshotAI | `provider_ext::moonshotai::{MoonshotAIProviderSettings, VERSION}` |
| DeepInfra | `provider_ext::deepinfra::{DeepInfraProviderSettings, VERSION}` |
