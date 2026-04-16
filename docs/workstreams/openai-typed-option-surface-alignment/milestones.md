# OpenAI Typed Option Surface Alignment - Milestones

Last updated: 2026-04-11

## Completed

- Audited `repo-ref/ai/packages/openai/src/index.ts` and the referenced option modules against
  Siumai's native OpenAI provider-owned/public facade surface
- Added AI SDK-style OpenAI typed option exports to the provider-owned options module
- Re-exported the same names on `provider_ext::openai`
- Added public compile guards on the stable facade
- Added provider-local regression tests for the new option surfaces
- Extended OpenAI audio option handling so the newly exposed speech/transcription typed options
  map onto real request behavior instead of dead JSON

## Next

- Audit `google_vertex` typed option names against `repo-ref/ai/packages/google-vertex/src/index.ts`
- Revisit OpenAI typed builder/config defaults only if a real caller need appears
