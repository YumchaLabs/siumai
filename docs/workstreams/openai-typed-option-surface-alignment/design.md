# OpenAI Typed Option Surface Alignment - Design

Last updated: 2026-04-11

## Problem

Compared with `repo-ref/ai/packages/openai/src/index.ts`, Siumai's native OpenAI package still had
one obvious public-surface gap:

- public Rust users could reach historical provider-owned types such as `OpenAiOptions`,
  `ResponsesApiConfig`, `OpenAiEmbeddingOptions`, `OpenAiTtsOptions`, and `OpenAiSttOptions`
- but the AI SDK-style named typed exports were mostly absent:
  - `OpenAILanguageModelResponsesOptions`
  - `OpenAIResponsesProviderOptions`
  - `OpenAILanguageModelChatOptions`
  - `OpenAIChatLanguageModelOptions`
  - `OpenAILanguageModelCompletionOptions`
  - `OpenAIEmbeddingModelOptions`
  - `OpenAISpeechModelOptions`
  - `OpenAITranscriptionModelOptions`
  - `OpenAIFilesOptions`

This was more than a naming annoyance:

- it made package-surface diffing against `repo-ref/ai` noisy
- it forced callers back onto older Rust-first names even when they wanted AI SDK-like review
- audio/file typed option coverage was uneven enough that some AI SDK-shaped option names would
  have been misleading if added as thin aliases only

## Design

### 1. Keep Rust-first types as the implementation anchor

The existing native types remain valid and continue to anchor implementation internals:

- `OpenAiOptions`
- `ResponsesApiConfig`
- `OpenAiEmbeddingOptions`
- `OpenAiTtsOptions`
- `OpenAiSttOptions`

This avoids needless churn in already-working provider code and existing Rust call sites.

### 2. Add AI SDK-style named typed surfaces where the semantics are real

The provider-owned OpenAI options module now exposes AI SDK-style names directly:

- `OpenAILanguageModelChatOptions`
- `OpenAILanguageModelResponsesOptions`
- `OpenAILanguageModelCompletionOptions`
- `OpenAIEmbeddingModelOptions`
- `OpenAISpeechModelOptions`
- `OpenAITranscriptionModelOptions`
- `OpenAIFilesOptions`

Deprecated upstream migration aliases are also present where AI SDK still exports them:

- `OpenAIChatLanguageModelOptions`
- `OpenAIResponsesProviderOptions`

Important nuance:

- chat/options and completion/options are modeled as flat AI SDK-shaped provider option structs
- responses/options serializes an internal `responsesApi.enabled = true` envelope so a flat
  Rust AI SDK-style struct still routes onto the `/responses` lane in Siumai
- embedding/speech/transcription reuse existing provider-owned Rust types where they were already
  close enough to the audited AI SDK request shape

### 3. Prefer "named surface + real behavior" over fake alias parity

This pass intentionally avoided adding names that would compile but not work.

Concrete behavior follow-up landed together with the new types:

- TTS provider options now support provider-owned `speed`
- transcription provider options now support `language` and `timestampGranularities`
- OpenAI audio shaping now accepts camelCase and snake_case transcription keys
- file upload typed options continue to honor provider-scoped `purpose` / `expiresAfter`

That keeps the new surface useful instead of purely cosmetic.

### 4. Re-export the same names through the stable facade

The new OpenAI AI SDK-style names are re-exported through:

- `siumai_provider_openai::provider_options::openai::*`
- `siumai_provider_openai::providers::openai::*`
- `siumai::provider_ext::openai::options::*`
- `siumai::provider_ext::openai::*`

This keeps provider-crate and facade review aligned.

## Validation

This workstream is locked by:

- `siumai/tests/public_surface_imports_test.rs`
- provider-local option serialization tests
- `cargo check -p siumai-provider-openai --no-default-features --features openai`
- `cargo check -p siumai --features openai`
- `cargo nextest run -p siumai-provider-openai --no-default-features --features openai`
- `cargo nextest run -p siumai --test public_surface_imports_test --features openai`

## Remaining follow-up

- Google Vertex should receive the same comparison pass next, but only for types that map onto
  real provider-owned runtime structures today.
- OpenAI builder/config typed-default helpers still primarily accept historical `OpenAiOptions`;
  that is acceptable for now because request-level AI SDK-style typed surfaces are the audited
  priority lane.
