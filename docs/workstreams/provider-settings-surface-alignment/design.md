# Provider Settings Surface Alignment - Design

Last updated: 2026-04-24

## Problem

Compared with the audited AI SDK package roots:

- `repo-ref/ai/packages/openai-compatible/src/openai-compatible-provider.ts`
- `repo-ref/ai/packages/openai/src/openai-provider.ts`
- `repo-ref/ai/packages/anthropic/src/anthropic-provider.ts`
- `repo-ref/ai/packages/azure/src/azure-openai-provider.ts`
- `repo-ref/ai/packages/amazon-bedrock/src/bedrock-provider.ts`
- `repo-ref/ai/packages/cohere/src/cohere-provider.ts`
- `repo-ref/ai/packages/deepseek/src/deepseek-provider.ts`
- `repo-ref/ai/packages/togetherai/src/togetherai-provider.ts`
- `repo-ref/ai/packages/google/src/google-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/anthropic/google-vertex-anthropic-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/maas/google-vertex-maas-provider-node.ts`

Siumai already had most runtime/build/config capabilities for provider-level construction, but the
package boundary still drifted in one important way:

- `openai-compatible`, `openai`, `azure`, `bedrock`, `cohere`, `deepseek`, and `togetherai` did not expose dedicated
  package-level `*ProviderSettings`
  carriers comparable to the upstream settings inputs
- `provider_ext::{openai,azure,bedrock}` therefore lagged the newer Google / Google Vertex
  package-surface story, where construction settings and `VERSION` were already exposed explicitly,
  and the Google / Vertex carriers needed to be recorded in the same matrix instead of living only
  in older structural-alignment notes
- some provider-level fields were supported by the real Rust runtime but only indirectly through
  builder/config methods, which made side-by-side structural diffing against `repo-ref/ai` noisy
- other upstream fields were not yet honestly supported by the Rust runtime, but that gap was not
  documented in one dedicated matrix

This was not a single runtime bug. It was a package-shape and data-structure gap:

- package-surface review against AI SDK required too much manual translation
- provider-construction parity was inconsistent across packages
- supported vs deferred provider-setting fields were not explicit enough

## Goals

- Expose provider-level settings carriers for the honest, currently supported subset of:
  `OpenAICompatibleProviderSettings`, `OpenAIProviderSettings`, `AnthropicProviderSettings`,
  `AzureOpenAIProviderSettings`,
  `AmazonBedrockProviderSettings`, `CohereProviderSettings`, `DeepSeekProviderSettings`,
  `TogetherAIProviderSettings`, `GoogleProviderSettings`, `GoogleVertexProviderSettings`,
  `GoogleVertexAnthropicProviderSettings`, and the audited compat-backed provider settings carriers
  tracked in the matrix, including `GoogleVertexMaasProviderSettings`.
- Keep those settings carriers model-agnostic and require model selection later through
  `into_builder_for_model(...)` / `into_config_for_model(...)`.
- Re-export the settings carriers and `VERSION` from the provider-owned modules and the top-level
  `siumai::provider_ext::*` facade.
- Fill any small builder/config ergonomic holes that are required to make the provider settings
  carriers honest and direct, such as `headers` / `header` helpers and Azure `resourceName`.
- Record the supported/deferred field matrix explicitly in this workstream.

## Non-goals

- Do not fabricate JavaScript-style callable provider objects on the Rust facade.
- Do not expose upstream fields that do not have an honest Rust analogue yet.
- Do not silently reinterpret Bedrock AWS credential-provider settings as generic header bags.
- Do not make `name` look supported on OpenAI while canonical Rust provider identity remains fixed.

## Chosen design

### 1. Add dedicated provider-settings carriers on the provider-owned surface

The provider-owned modules now expose:

- `providers::openai::OpenAIProviderSettings`
- `providers::anthropic::AnthropicProviderSettings`
- `providers::azure_openai::AzureOpenAIProviderSettings`
- `providers::bedrock::AmazonBedrockProviderSettings`
- `providers::cohere::CohereProviderSettings`
- `providers::deepseek::DeepSeekProviderSettings`
- `providers::togetherai::TogetherAIProviderSettings`
- `providers::gemini::GoogleProviderSettings`
- `providers::vertex::GoogleVertexProviderSettings`
- `providers::anthropic_vertex::GoogleVertexAnthropicProviderSettings`
- `providers::openai_compatible::{OpenAICompatibleProviderSettings, MistralProviderSettings,
  PerplexityProviderSettings, FireworksProviderSettings, MoonshotAIProviderSettings,
  DeepInfraProviderSettings, GoogleVertexMaasProviderSettings}`

Each carrier:

- is model-agnostic by design
- provides `new()`
- provides fluent `with_*` setters for the supported package-level fields
- exposes `into_builder()`
- exposes `into_builder_for_model(...)`
- exposes `into_config_for_model(...)`

This keeps the Rust construction story closer to the upstream package-level `createProvider(options)`
boundary without pretending Rust has a callable provider object.

### 2. Fill only the minimal builder/config gaps needed for honest support

To avoid settings carriers becoming special-case shims, the underlying provider-owned surfaces now
also expose the small missing helpers they genuinely support:

- OpenAI:
  - builder `headers(...)`, `header(...)`
  - config `with_headers(...)`, `with_header(...)`
- Azure:
  - builder `resource_name(...)`, `headers(...)`, `header(...)`
  - config `base_url_for_resource(...)`, `with_resource_name(...)`,
    `with_headers(...)`, `with_header(...)`
- Bedrock:
  - builder `headers(...)`, `header(...)`
  - config `with_headers(...)`, `with_header(...)`
- Cohere:
  - builder `headers(...)`, `header(...)`
  - config `with_headers(...)`, `with_header(...)`
- DeepSeek:
  - builder `headers(...)` alias over the existing custom-header path
  - config `with_headers(...)`, `with_header(...)`
- TogetherAI:
  - builder `headers(...)`, `header(...)`
  - config `with_headers(...)`, `with_header(...)`

This keeps provider settings carriers thin. They translate into real provider-owned construction
APIs instead of bypassing them.

### 3. Export `VERSION` on the audited package boundaries

The provider-owned modules now expose:

- `providers::openai::VERSION`
- `providers::anthropic::VERSION`
- `providers::azure_openai::VERSION`
- `providers::bedrock::VERSION`
- `providers::cohere::VERSION`
- `providers::deepseek::VERSION`
- `providers::togetherai::VERSION`
- `providers::gemini::VERSION`
- `providers::vertex::VERSION`
- `providers::anthropic_vertex::VERSION`
- `providers::openai_compatible::OPENAI_COMPATIBLE_VERSION`
- provider-scoped OpenAI-compatible package facades, including `provider_ext::vertex_maas::VERSION`

The top-level facade mirrors these on:

- `provider_ext::openai::VERSION`
- `provider_ext::azure::VERSION`
- `provider_ext::bedrock::VERSION`
- `provider_ext::cohere::VERSION`
- `provider_ext::deepseek::VERSION`
- `provider_ext::togetherai::VERSION`
- `provider_ext::google::VERSION`
- `provider_ext::google_vertex::VERSION`
- `provider_ext::anthropic_vertex::VERSION`
- `provider_ext::openai_compatible::VERSION`

This matches the audited package-root review story more closely and keeps package-surface checks
consistent with the Google / Google Vertex slices.

### 4. Keep deferred fields explicit instead of faking parity

This workstream intentionally does not expose unsupported fields merely to mirror TypeScript names.

Current deferred examples:

- OpenAI `name`
- Bedrock `accessKeyId`
- Bedrock `secretAccessKey`
- Bedrock `sessionToken`
- Bedrock `credentialProvider`
- Bedrock `generateId`
- Cohere `generateId`
- Google Vertex / Vertex Anthropic credential object shapes (`googleAuthOptions`,
  `googleCredentials`) beyond the Rust token-provider analogue

These are recorded in `data-structure-matrix.md` so future follow-up can target the real runtime
gaps instead of rediscovering them.

## Validation

This slice is locked by:

- provider-local unit tests in the new `settings.rs` files
- public-surface compile guards in `siumai/tests/public_surface_imports_test.rs`
- top-level public-path parity tests in `siumai/tests/provider_public_path_parity_test.rs`

## Remaining follow-up

- Revisit whether OpenAI should gain an honest provider-display-label surface without changing the
  canonical `openai` identity used by metadata, registry, and typed helper namespaces.
- Revisit Bedrock credential-provider parity only if Siumai grows a first-class SigV4 credential
  abstraction instead of generic header/transport escape hatches.
- Re-audit other AI SDK package facades to ensure provider-settings carriers are exposed wherever
  the Rust runtime already has honest support.
