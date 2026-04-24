# Provider Settings Surface Alignment - Milestones

Last updated: 2026-04-24

## PSA-M0 - Audit locked

Acceptance criteria:

- The upstream OpenAI Compatible / OpenAI / Anthropic / Azure / Bedrock / Cohere / DeepSeek / TogetherAI / xAI / Groq / Google / Google Vertex / Google Vertex Anthropic / Mistral / Perplexity / Fireworks / MoonshotAI / DeepInfra / Google Vertex MaaS provider settings
  references are recorded.
- The supported/deferred matrix is explicit.
- The workstream scope distinguishes package-shape parity from runtime parity.

Status: completed

## PSA-M1 - Supported provider settings exposed

Acceptance criteria:

- OpenAI Compatible, OpenAI, Anthropic, Azure, Bedrock, Cohere, DeepSeek, TogetherAI, xAI, Groq, Google, Google Vertex, Google Vertex Anthropic, Mistral, Perplexity, Fireworks, MoonshotAI, DeepInfra, and Google Vertex MaaS each expose a dedicated package-level
  provider settings carrier.
- Those settings carriers are model-agnostic.
- They convert into the provider-owned builder/config surfaces without special-case backdoors.

Current state:

- `OpenAIProviderSettings` is exposed on the provider-owned OpenAI module and the top-level
  `provider_ext::openai` facade.
- `AnthropicProviderSettings` is exposed on the provider-owned Anthropic module and the top-level
  `provider_ext::anthropic` facade.
- `AzureOpenAIProviderSettings` is exposed on the provider-owned Azure module and the top-level
  `provider_ext::azure` facade.
- `AmazonBedrockProviderSettings` is exposed on the provider-owned Bedrock module and the
  top-level `provider_ext::bedrock` facade.
- `CohereProviderSettings` is exposed on the provider-owned Cohere module and the top-level
  `provider_ext::cohere` facade.
- `DeepSeekProviderSettings` is exposed on the provider-owned DeepSeek module and the top-level
  `provider_ext::deepseek` facade.
- `TogetherAIProviderSettings` is exposed on the provider-owned TogetherAI module and the top-level
  `provider_ext::togetherai` facade.
- `XaiProviderSettings` is exposed on the provider-owned xAI module and the top-level
  `provider_ext::xai` facade.
- `GroqProviderSettings` is exposed on the provider-owned Groq module and the top-level
  `provider_ext::groq` facade.
- `GoogleProviderSettings` is exposed on the provider-owned Gemini/Google module and the
  top-level `provider_ext::google` facade.
- `GoogleVertexProviderSettings` is exposed on the provider-owned Vertex module and the top-level
  `provider_ext::google_vertex` facade.
- `GoogleVertexAnthropicProviderSettings` is exposed on the provider-owned Vertex Anthropic module
  and the top-level `provider_ext::anthropic_vertex` facade.
- `OpenAICompatibleProviderSettings` is exposed from the OpenAI-compatible module and the
  top-level `provider_ext::openai_compatible` facade.
- `MistralProviderSettings` is exposed from the OpenAI-compatible module and the top-level
  `provider_ext::mistral` facade.
- `PerplexityProviderSettings` is exposed from the OpenAI-compatible module and the top-level
  `provider_ext::perplexity` facade.
- `FireworksProviderSettings` is exposed from the OpenAI-compatible module and the top-level
  `provider_ext::fireworks` facade.
- `MoonshotAIProviderSettings` is exposed from the OpenAI-compatible module and the top-level
  `provider_ext::moonshotai` facade.
- `DeepInfraProviderSettings` is exposed from the OpenAI-compatible module and the top-level
  `provider_ext::deepinfra` facade.
- `GoogleVertexMaasProviderSettings` is exposed from the OpenAI-compatible module and the
  top-level `provider_ext::vertex_maas` facade.
- Each carrier now exposes `new()`, fluent `with_*` setters, `into_builder()`,
  `into_builder_for_model(...)`, and `into_config_for_model(...)`.

Status: completed

## PSA-M2 - Public/package boundary locked

Acceptance criteria:

- Provider-owned/public modules expose `VERSION` on the audited package boundaries.
- Public-surface compile tests cover the new settings carriers and `VERSION`.
- Public-path parity tests cover the top-level re-exported settings carriers.

Current state:

- `providers::{openai,azure_openai,bedrock}::VERSION` now exist.
- `provider_ext::{openai,anthropic,azure,bedrock}::VERSION` now exist.
- `provider_ext::cohere::VERSION` now exists.
- `provider_ext::{openai_compatible,deepseek,togetherai,xai,groq,google,google_vertex,anthropic_vertex,mistral,perplexity,fireworks,moonshotai,deepinfra,vertex_maas}::VERSION` now exist.
- Public compile/path tests now lock the new settings carriers on the top-level facade.

Status: completed
