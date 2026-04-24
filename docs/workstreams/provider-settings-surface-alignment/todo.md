# Provider Settings Surface Alignment - TODO

Last updated: 2026-04-24

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## 0) Audit and scope

- [x] Lock the audited AI SDK provider reference files for OpenAI Compatible, OpenAI, Anthropic, Azure, Bedrock,
  Cohere, DeepSeek, TogetherAI, xAI, Groq, Mistral, Perplexity, Fireworks, MoonshotAI, DeepInfra,
  Google, Google Vertex, Google Vertex Anthropic, and Google Vertex MaaS.
- [x] Confirm the local provider-owned builder/config surfaces can already support most of the
  upstream package-level settings.
- [x] Separate supported fields from deferred fields before exposing any new Rust struct.

## 1) Native provider settings carriers

- [x] Add `OpenAIProviderSettings`.
- [x] Add `AnthropicProviderSettings`.
- [x] Add `AzureOpenAIProviderSettings`.
- [x] Add `AmazonBedrockProviderSettings`.
- [x] Add `CohereProviderSettings`.
- [x] Add `DeepSeekProviderSettings`.
- [x] Add `TogetherAIProviderSettings`.
- [x] Add `XaiProviderSettings`.
- [x] Add `GroqProviderSettings`.
- [x] Add generic compat-backed `OpenAICompatibleProviderSettings`.
- [x] Add compat-backed `MistralProviderSettings`.
- [x] Add compat-backed `PerplexityProviderSettings`.
- [x] Add compat-backed `FireworksProviderSettings`.
- [x] Add compat-backed `MoonshotAIProviderSettings`.
- [x] Add compat-backed `DeepInfraProviderSettings`.
- [x] Account for existing `GoogleProviderSettings`.
- [x] Account for existing `GoogleVertexProviderSettings`.
- [x] Account for existing `GoogleVertexAnthropicProviderSettings`.
- [x] Add compat-backed `GoogleVertexMaasProviderSettings`.
- [x] Keep the new settings carriers model-agnostic.
- [x] Expose `into_builder()`, `into_builder_for_model(...)`, and `into_config_for_model(...)`.

## 2) Provider-owned ergonomic gaps

- [x] Add OpenAI builder/config header helpers needed by the settings carrier.
- [x] Add Azure builder/config support for `resourceName` and header helpers.
- [x] Add Bedrock builder/config header helpers needed by the settings carrier.
- [x] Add Cohere builder/config header helpers needed by the settings carrier.
- [x] Add DeepSeek builder/config header helpers needed by the settings carrier.
- [x] Add TogetherAI builder/config header helpers needed by the settings carrier.
- [x] Add xAI builder/config header helpers needed by the settings carrier.
- [x] Reuse existing Groq builder/config header/fetch helpers for the settings carrier.

## 3) Public facade alignment

- [x] Re-export the new provider settings carriers from the provider-owned modules.
- [x] Re-export package `VERSION` from the provider-owned modules.
- [x] Mirror those exports on
  `siumai::provider_ext::{openai_compatible,openai,anthropic,azure,bedrock,cohere,deepseek,togetherai,xai,groq,mistral,perplexity,fireworks,moonshotai,deepinfra,vertex_maas}`.
- [x] Extend public-surface compile guards for the new exports.
- [x] Extend top-level public-path parity coverage for the new exports.

## 4) Documentation and changelog

- [x] Create a dedicated provider-settings workstream folder.
- [x] Record the supported/deferred field matrix.
- [x] Update changelog `Unreleased` sections for the affected crates.

## 5) Intentional deferrals

- [-] OpenAI `name`
- [-] Anthropic `name`
- [-] Anthropic `generateId`
- [-] Bedrock `accessKeyId`
- [-] Bedrock `secretAccessKey`
- [-] Bedrock `sessionToken`
- [-] Bedrock `credentialProvider`
- [-] Bedrock `generateId`
- [-] Cohere `generateId`
- [-] Mistral `generateId`
- [-] Google Vertex MaaS Node `googleAuthOptions` object shape
- [-] Google Vertex Node `googleAuthOptions` object shape
- [-] Google Vertex Edge `googleCredentials` object shape
- [-] Google Vertex Anthropic Node `googleAuthOptions` object shape
- [-] Google Vertex Anthropic Edge `googleCredentials` object shape
