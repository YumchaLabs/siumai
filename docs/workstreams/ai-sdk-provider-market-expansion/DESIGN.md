# AI SDK Provider Market Expansion

Status: Closed
Last updated: 2026-05-26

## Why This Lane Exists

Siumai already covers the mainstream direct LLM providers, but the Vercel AI SDK package ecosystem has shifted
toward provider routing, OpenAI-compatible vendor wrappers, and selected secondary providers with meaningful
npm adoption. This lane uses recent npm package download data and the local AI SDK reference checkout to choose
the next provider work by user impact instead of by catalog breadth alone.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0002-provider-crates-by-provider.md`
  - `docs/adr/0003-provider-ext-export-policy.md`
  - `docs/adr/0004-experimental-surface-policy.md`
- Existing docs:
  - `docs/architecture/public-surface.md`
  - `docs/alignment/provider-implementation-alignment.md`
  - `docs/alignment/provider-feature-alignment.md`
  - `docs/workstreams/fearless-refactor-v4/provider-capability-alignment-matrix.md`
- Reference sources:
  - `repo-ref/ai/packages/gateway`
  - `repo-ref/ai/packages/cerebras`
  - `repo-ref/ai/packages/mistral`
  - `repo-ref/ai/packages/azure`
  - `repo-ref/ai/packages/amazon-bedrock`
  - `repo-ref/ai/packages/google-vertex`
  - `repo-ref/ai/packages/deepgram`
  - `repo-ref/ai/packages/elevenlabs`
  - `repo-ref/ai/packages/fal`
  - `repo-ref/ai/packages/replicate`
- Related workstreams:
  - `docs/workstreams/provider-model-catalog-ai-sdk-refresh/`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/`
  - `docs/workstreams/mistral-package-surface-alignment/`
  - `docs/workstreams/cohere-unified-provider-surface/`
  - `docs/workstreams/togetherai-unified-provider-surface/`

## Market Snapshot

npm downloads were sampled with the official downloads API for `last-month`, which returned the complete
window `2026-04-25` through `2026-05-24`.

| AI SDK package | Downloads | Siumai position | Priority signal |
| --- | ---: | --- | --- |
| `@ai-sdk/gateway` | 45,995,757 | Missing as a Vercel remote gateway provider | P0 design/proof |
| `@ai-sdk/openai` | 26,415,755 | Native provider | Maintenance |
| `@ai-sdk/anthropic` | 23,989,010 | Native provider | Maintenance |
| `@ai-sdk/google` | 17,601,231 | Native Gemini/Google facade | Maintenance |
| `@ai-sdk/openai-compatible` | 11,051,338 | Shared provider crate and vendor presets | Maintenance plus preset quality |
| `@ai-sdk/google-vertex` | 6,609,651 | Native provider | Enterprise polish |
| `@ai-sdk/amazon-bedrock` | 6,285,133 | Native provider | Enterprise polish |
| `@ai-sdk/xai` | 5,841,870 | Native provider | Maintenance |
| `@ai-sdk/azure` | 5,364,901 | Native provider | Enterprise polish |
| `@ai-sdk/groq` | 4,401,992 | Native provider | Maintenance |
| `@ai-sdk/mistral` | 4,142,636 | OpenAI-compatible preset/facade | P1 parity audit |
| `@ai-sdk/deepseek` | 3,458,059 | Native plus compat preset | Maintenance |
| `@ai-sdk/cerebras` | 2,859,534 | Missing | P1 onboarding |
| `@ai-sdk/perplexity` | 2,712,137 | OpenAI-compatible vendor facade | Maintenance |
| `@ai-sdk/togetherai` | 2,090,503 | Native plus compat preset | Parity cleanup |
| `@ai-sdk/cohere` | 1,742,523 | Native provider | Parity cleanup |
| `@ai-sdk/fireworks` | 981,722 | OpenAI-compatible vendor facade | Maintenance |
| `@ai-sdk/deepgram` | 605,045 | Missing audio provider | P2 decision |
| `@ai-sdk/elevenlabs` | 533,132 | Missing audio provider | P2 decision |
| `@ai-sdk/deepinfra` | 433,919 | Curated compat preset | Keep deferred full catalog |
| `@ai-sdk/replicate` | 257,948 | Missing media provider | P2/P3 decision |
| `@ai-sdk/fal` | 192,339 | Missing media provider | P2/P3 decision |

## Problem

Provider breadth work has two failure modes:

- treating all AI SDK provider packages as equally important, which creates long-tail maintenance drag;
- treating current direct-provider coverage as "done", which misses high-impact shifts such as AI Gateway and
  high-download OpenAI-compatible wrappers.

Siumai needs a bounded provider-expansion lane that protects existing provider architecture while adding or
deepening only the provider surfaces that have clear market signal and manageable maintenance cost.

## Target State

When this workstream closes:

- Vercel AI Gateway has either a committed minimal provider implementation or a documented ADR-quality deferral
  with concrete blockers.
- Cerebras is supported through the lowest-coupling implementation that matches the AI SDK package behavior.
- Mistral's AI SDK package surface has been audited against Siumai's current OpenAI-compatible preset/facade,
  with concrete gaps closed or explicitly deferred.
- Enterprise provider polish for Azure, Bedrock, and Google Vertex is updated where high-value gaps are found.
- Audio/media provider expansion is decided with explicit priority boundaries rather than left as implicit backlog.
- DeepInfra full catalog parity remains an explicit deferred strategy unless new evidence changes the cost/benefit.

## In Scope

- New provider onboarding for high-download AI SDK packages when the implementation can be bounded.
- Provider package-surface audits for high-download packages already covered by a compat preset.
- Provider extension roots, registry wiring, model catalogs, and examples needed for adopted provider work.
- Focused no-network tests for request shape, URL/header behavior, public facade imports, and registry resolution.
- Documentation updates that explain priority, support boundaries, and intentional deferrals.

## Out Of Scope

- Full parity for every AI SDK provider package.
- Full DeepInfra catalog parity.
- Replacing Siumai's registry with Vercel AI Gateway semantics.
- Adding UI/framework packages such as `@ai-sdk/react`, `@ai-sdk/vue`, or `@ai-sdk/svelte`.
- Live integration tests that require paid credentials as mandatory release gates.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Gateway download volume is partly transitive through the `ai` package but still signals AI SDK's default-provider direction. | High | `repo-ref/ai/packages/ai/package.json` depends on `@ai-sdk/gateway`; `ai/src/model/resolve-model.ts` defaults to gateway. | Gateway priority may become P1 rather than P0, but the design audit is still justified. |
| Cerebras can be added with low coupling because the AI SDK package wraps OpenAI-compatible chat behavior. | High | `repo-ref/ai/packages/cerebras/src/cerebras-chat-language-model.ts` extends `OpenAICompatibleChatLanguageModel`. | If official API differences are larger than expected, split native onboarding into a follow-on. |
| Mistral may not need a native crate immediately. | Medium | Siumai has an OpenAI-compatible `mistral` facade and model catalog, while AI SDK has a dedicated package. | If API surface gaps are material, promote Mistral native provider work. |
| Enterprise polish is higher ROI than long-tail provider breadth. | High | Azure, Bedrock, and Vertex all exceed five million monthly downloads. | If user demand points elsewhere, reprioritize within TODO without widening scope. |
| Audio/media packages should be evaluated before implementation. | Medium | Deepgram/ElevenLabs are meaningful but below core LLM providers; Replicate/Fal are lower still. | If Siumai product direction shifts toward media, split a dedicated media provider workstream. |

## Architecture Direction

Keep provider ownership explicit:

- Native providers stay in dedicated `siumai-provider-*` crates only when they own protocol behavior beyond a
  simple OpenAI-compatible preset.
- OpenAI-compatible vendors start in `siumai-provider-openai-compatible` with provider-specific model catalogs,
  field mappings, and typed `provider_ext` facades before being promoted.
- Gateway should not be modeled as another OpenAI-compatible vendor. It uses AI SDK protocol-family endpoints
  such as `/language-model`, `/embedding-model`, `/image-model`, and request-scoped `providerOptions.gateway`.
- Public facade additions must stay under stable provider roots such as `siumai::provider_ext::<provider>` and
  avoid widening `prelude::unified` unless there is an explicit stable family reason.
- Every provider addition or promotion must have no-network tests before any credentialed example is considered.

## Closeout Condition

This lane can close when:

- Gateway, Cerebras, Mistral, enterprise polish, and audio/media decision tasks are complete or explicitly split;
- focused nextest/check gates pass for touched crates;
- catalog and public facade audits are updated when provider roots or model constants change;
- workstream docs record shipped behavior, deferred scope, and follow-on work.

Closeout: Achieved on 2026-05-26. The lane now records the shipped Gateway proof, Cerebras onboarding,
Mistral request-shape fixes, Bedrock alias polish, and the audio/media decision note with explicit follow-on
splits for Deepgram, ElevenLabs, Replicate, and Fal.
