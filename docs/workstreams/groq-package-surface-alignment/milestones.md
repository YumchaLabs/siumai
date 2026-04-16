# Groq Package Surface Alignment - Milestones

Last updated: 2026-04-15

## GPS-M0 - Scope isolated from browser-search work

Acceptance criteria:

- the broader `@ai-sdk/groq` parity pass is tracked outside the browser-search workstream
- the new workstream is explicitly anchored to `repo-ref/ai/packages/groq/src/*`
- scope and non-goals are written down

Status: completed

## GPS-M1 - Typed option enums match current upstream set

Acceptance criteria:

- `GroqServiceTier` includes `auto`, `on_demand`, `performance`, and `flex`
- `GroqReasoningEffort` includes `none`, `default`, `low`, `medium`, and `high`
- validation and provider-owned tests lock the same accepted values

Status: completed

## GPS-M2 - Built-in model ids are refreshed

Acceptance criteria:

- Groq built-in model constants no longer reference decommissioned Kimi K2 ids
- the provider-owned model catalog tests cover the refreshed constant
- changelog entries mention the externally visible model-id update

Status: completed

## GPS-M3 - Remaining typed option audit is resolved

Acceptance criteria:

- each field in `repo-ref/ai/packages/groq/src/groq-chat-options.ts` is classified as:
  - already covered by shared/common Siumai request fields
  - exposed on the Groq typed option surface
  - intentionally deferred with a written reason
- missing stable typed fields are either implemented or explicitly deferred
- provider-owned audio escape hatches no longer leak into the AI SDK-aligned `options::*` lane

Status: completed

## GPS-M4 - Response/request shaping parity is stable

Acceptance criteria:

- remaining request-shaping and warning differences against audited upstream files are either fixed
  or explicitly deferred
- regression tests cover any new provider-owned behavior

Status: completed

## GPS-M5 - Model catalog matches the audited package boundary

Acceptance criteria:

- the Groq chat catalog matches the currently audited `@ai-sdk/groq` `GroqChatModelId` union
- obsolete legacy/system/vision/tool-use model ids that are no longer part of the upstream package
  surface are removed from the public Groq catalog
- Groq transcription ids remain aligned with `GroqTranscriptionModelId`
- provider-owned Groq TTS models stay supported, but are grouped outside the AI SDK-aligned chat
  catalog

Status: completed

## GPS-M6 - Provider-construction settings match the audited package contract

Acceptance criteria:

- the OpenAI-compatible `groq` preset uses `GROQ_API_KEY` as its canonical environment variable
- provider-owned Groq config/builder surfaces cover the stable upstream settings lane
  (`apiKey`, `baseURL`, `headers`, `fetch`) without introducing extra non-upstream abstractions as
  the primary path
- regression tests lock env fallback and AI SDK-style header aliasing on the provider-owned path

Status: completed
