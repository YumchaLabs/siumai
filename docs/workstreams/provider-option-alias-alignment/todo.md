# Provider Option Alias Alignment - TODO

Last updated: 2026-04-14

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Provider crate aliases

- [x] Add AI SDK-style xAI option aliases on the provider-owned crate surface.
- [x] Add AI SDK-style Groq language-model option aliases on the provider-owned crate surface.
- [x] Add AI SDK-style Bedrock language-model and reranking option aliases on the provider-owned
  crate surface.

## Track B - Public facade alignment

- [x] Re-export the new aliases from `provider_ext::xai`.
- [x] Re-export the new aliases from `provider_ext::groq`.
- [x] Re-export the new aliases from `provider_ext::bedrock`.
- [x] Add facade compile guards for the new aliases.

## Follow-up

- [x] Revisit Azure option alias parity against `repo-ref/ai/packages/azure/src/index.ts`.
  - `siumai-provider-azure` and `siumai::provider_ext::azure` now expose
    `OpenAILanguageModel{Chat,Responses}Options` plus the upstream deprecated aliases
    `OpenAIChatLanguageModelOptions` and `OpenAIResponsesProviderOptions`
  - `with_azure_options(...)` now merges into existing `providerOptions.azure` objects instead of
    overwriting sibling raw fields
- [x] Add `GroqTranscriptionModelOptions` and close the associated runtime gap.
  - `siumai-provider-groq` and `siumai::provider_ext::groq` now expose
    `GroqTranscriptionModelOptions`
  - `GroqSttOptions` now covers `language` / `prompt` / `responseFormat` /
    `timestampGranularities`
  - Groq STT request shaping now forwards `language` and `timestamp_granularities[]`, and the
    provider-owned response path now preserves `language` / `duration` / `segments` /
    `x_groq`
