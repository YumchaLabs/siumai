# Azure OpenAI Alias Parity

Last updated: 2026-04-14

## Reference

- `repo-ref/ai/packages/azure/src/index.ts`

## What changed

- `siumai-provider-azure` now re-exports the same AI SDK-style OpenAI option subset that the
  audited Azure package exposes:
  - `OpenAILanguageModelChatOptions`
  - `OpenAIChatLanguageModelOptions` (deprecated alias)
  - `OpenAILanguageModelResponsesOptions`
  - `OpenAIResponsesProviderOptions` (deprecated alias)
- `siumai::provider_ext::azure::options::*` and the top-level
  `siumai::provider_ext::azure::*` facade now expose the same names.
- `ChatRequest::with_azure_options(...)` now merges into the existing
  `providerOptions.azure` object instead of replacing it, so raw sibling keys and nested
  `responses_api` fields survive typed helper calls.

## Why this shape

The audited AI SDK Azure package does not define a separate Azure-specific language-model option
type. It re-exports the OpenAI chat/responses option types because Azure rides the same protocol
shape with Azure-owned URL/auth/runtime behavior.

Rust now follows that package-boundary decision:

- Azure-owned runtime/config/metadata types stay in `siumai-provider-azure`
- AI SDK-style chat/responses option aliases are forwarded from `siumai-provider-openai`
- request shaping still reads both snake_case and camelCase keys on the Azure path

## Validation

- provider-local request-ext tests now cover recursive merge behavior for `providerOptions.azure`
- public facade compile guards now cover the four Azure/OpenAI alias exports
- boundary tests now prove the alias-typed Azure options reach the final `/responses` body across
  `Siumai`, `Provider::azure()`, and config-first `AzureOpenAiClient`
