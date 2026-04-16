# Provider Option Alias Alignment - Design

Last updated: 2026-04-15

## Problem

Several provider-owned packages were already structurally close to the audited AI SDK package
indices in `repo-ref/ai`, but their typed option exports still drifted at the naming layer:

- `xai` exposed `XaiChatOptions` / `XaiResponsesOptions` / `XaiImageOptions` /
  `XaiVideoOptions`, while AI SDK exports AI-SDK-shaped aliases such as
  `XaiLanguageModelChatOptions` and deprecated `XaiProviderOptions`
- `groq` exposed `GroqOptions`, while AI SDK exports `GroqLanguageModelOptions` plus deprecated
  `GroqProviderOptions`
- `amazon-bedrock` exposed `BedrockChatOptions` / `BedrockRerankOptions`, while AI SDK exports
  `AmazonBedrockLanguageModelOptions` / `AmazonBedrockRerankingModelOptions` plus deprecated
  compatibility aliases

This was not a runtime bug, but it made package-surface comparison against `repo-ref/ai`
needlessly noisy and kept `provider_ext::*` farther from the audited provider-package contract
than necessary.

There was one notable exception after the initial alias pass:

- AI SDK `@ai-sdk/groq` also exports `GroqTranscriptionModelOptions`, but the Rust Groq STT helper
  surface initially lagged behind that shape (`language` / `timestampGranularities` were missing
  from the typed helper, request shaping dropped them, and the response path discarded
  `language` / `duration` / `segments`)

## Implemented design

### 1. Keep historical Rust-first names as the implementation anchor

Existing native names remain valid and canonical inside the provider crates:

- `XaiChatOptions`
- `XaiResponsesOptions`
- `XaiImageOptions`
- `XaiVideoOptions`
- `GroqOptions`
- `BedrockChatOptions`
- `BedrockRerankOptions`

This avoids churn for existing Rust callers and keeps the implementation vocabulary stable.

### 2. Add AI SDK-style alias types at the provider-owned boundary

The audited provider crates now also expose AI SDK-style alias names:

- `XaiLanguageModelChatOptions`
- `XaiLanguageModelResponsesOptions`
- `XaiImageModelOptions`
- `XaiVideoModelOptions`
- `GroqLanguageModelOptions`
- `AmazonBedrockLanguageModelOptions`
- `AmazonBedrockRerankingModelOptions`
- `OpenAILanguageModelChatOptions` on the Azure package surface
- `OpenAILanguageModelResponsesOptions` on the Azure package surface

Deprecated migration aliases are also available where upstream keeps them:

- `XaiProviderOptions`
- `XaiResponsesProviderOptions`
- `XaiImageProviderOptions`
- `XaiVideoProviderOptions`
- `GroqProviderOptions`
- `BedrockProviderOptions`
- `BedrockRerankingOptions`
- `OpenAIChatLanguageModelOptions` on the Azure package surface
- `OpenAIResponsesProviderOptions` on the Azure package surface

These aliases are intentionally thin type aliases rather than new structs, so the public surface
gets closer to AI SDK package exports without introducing a second runtime shape.

### 3. Re-export the same aliases on the stable public facade

`siumai::provider_ext::{xai, groq, bedrock, azure}` now re-export the same alias names under
`options::*` and at the top level.

That makes the public facade easier to diff against `repo-ref/ai/packages/*/src/index.ts`
without forcing callers to abandon the older Rust-first names immediately.

For Azure, the implementation deliberately keeps the runtime/config layer Azure-owned while
forwarding the audited OpenAI chat/responses option aliases, matching the AI SDK package split
more closely than inventing a second Azure-only language-model option type.

### 4. Treat Groq transcription alias parity as a runtime-backed alias, not a naming-only shim

`GroqTranscriptionModelOptions` is now exposed as an alias to the provider-owned `GroqSttOptions`,
but this alias is only considered valid because the runtime surface was brought up to the same
minimum semantic level:

- `GroqSttOptions` now includes `language` and `timestamp_granularities`
- the typed helper serializes AI SDK-style keys (`responseFormat`, `timestampGranularities`)
- Groq multipart request shaping now forwards `language` and `timestamp_granularities[]`
- Groq STT responses now retain `language`, `duration`, and raw `segments` / `x_groq` metadata

This keeps the Rust alias honest: callers do not get an AI SDK-shaped name that still lowers into
an older, incomplete runtime contract.

The concrete `GroqSttOptions` / `GroqTtsOptions` helpers remain available only through explicit
provider-owned escape hatches (`ext::audio_options`) rather than the main AI SDK-aligned
`provider_ext::groq::{options::*, *}` lane.

## Validation

The alignment is locked by:

- facade compile guards in `siumai/tests/public_surface_imports_test.rs`
- provider-local cargo checks on the affected crates
- a dedicated workstream note so the naming decision is explicit and durable

## Remaining follow-up

- The earlier Groq `browserSearch` helper/runtime gap is no longer part of this workstream; it was
  closed in the dedicated Groq browser-search/package-surface follow-up.
- Any remaining Groq drift now lives at the broader package/facade construction boundary
  (`repo-ref/ai/packages/groq/src/index.ts` vs. Rust `provider_ext::groq` exports), not at the
  alias-naming layer tracked here.
