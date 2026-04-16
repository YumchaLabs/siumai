# Provider Option Alias Alignment - Milestones

Last updated: 2026-04-14

## POA-M0 - Scope locked

Acceptance criteria:

- the naming-only drift is separated from larger runtime/provider-surface refactors
- the pass is limited to low-risk alias additions and public-facade export cleanup

Status: completed

## POA-M1 - xAI alias surface aligned

Acceptance criteria:

- `siumai-provider-xai` exposes AI SDK-style language-model / responses / image / video option
  aliases
- deprecated AI SDK compatibility aliases are available for migration checks

Status: completed

## POA-M2 - Groq and Bedrock alias surface aligned

Acceptance criteria:

- `siumai-provider-groq` exposes `GroqLanguageModelOptions` plus deprecated
  `GroqProviderOptions`
- `siumai-provider-amazon-bedrock` exposes AI SDK-style language-model and reranking aliases plus
  deprecated compatibility aliases

Status: completed

## POA-M3 - Facade, docs, and changelog aligned

Acceptance criteria:

- `provider_ext::{xai, groq, bedrock}` re-export the new aliases
- public-surface compile guards cover the aliases
- workstream docs and unreleased changelogs record the change

Status: completed

## POA-M4 - Azure alias surface aligned

Acceptance criteria:

- `siumai-provider-azure` exposes the audited Azure/OpenAI alias subset:
  `OpenAILanguageModel{Chat,Responses}Options` plus deprecated migration aliases
- `provider_ext::azure` re-exports the same alias names
- `with_azure_options(...)` preserves existing raw provider-option siblings instead of replacing
  the whole object

Status: completed

## POA-M5 - Groq transcription alias parity aligned

Acceptance criteria:

- `siumai-provider-groq` exposes `GroqTranscriptionModelOptions`
- `provider_ext::groq::{options::*, *}` re-exports the same alias without leaking unrelated
  provider-owned audio helpers into the AI SDK-aligned options lane
- the concrete `GroqSttOptions` / `GroqTtsOptions` helpers remain available under
  `provider_ext::groq::ext::audio_options::*`
- Groq transcription typed options serialize AI SDK-style `responseFormat` /
  `timestampGranularities`
- Groq STT request/response handling now preserves the corresponding runtime fields instead of
  exposing a naming-only alias

Status: completed
