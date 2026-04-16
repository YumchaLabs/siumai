# MoonshotAI Package Surface Alignment - Milestones

Last updated: 2026-04-13

## MPSA-M0 - Scope and upstream boundary locked

Acceptance criteria:

- the audited AI SDK MoonshotAI reference files are identified
- the package is explicitly classified as a chat/language-model wrapper, not a multi-family
  unified provider
- the canonical public id decision is documented

Current state:

- `repo-ref/ai/packages/moonshotai/src/index.ts`,
  `moonshotai-provider.ts`, and `moonshotai-chat-options.ts` are the main reference files
- the workstream now records that upstream MoonshotAI is chat-only on top of the shared
  OpenAI-compatible runtime
- canonical public/runtime id is `moonshotai`

Status: completed

## MPSA-M1 - Canonical identity and public entrypoints aligned

Acceptance criteria:

- public examples/docs use `moonshotai` instead of `moonshot`
- builder/facade entrypoints converge on `.moonshotai()`
- the historical `moonshot` id survives only as a hidden migration alias

Current state:

- config-first examples now build with `OpenAiCompatibleClient::from_builtin_env("moonshotai", ..)`
- builder examples/docs now use `.moonshotai()`
- shared compat config keeps `moonshot` only as a hidden alias that resolves back to canonical
  `moonshotai`

Status: completed

## MPSA-M2 - Typed options and request normalization aligned

Acceptance criteria:

- MoonshotAI typed request options exist on the provider-owned/public boundary
- request normalization matches the audited AI SDK wire-key behavior
- provider-key canonicalization works for both canonical and legacy alias inputs

Current state:

- `MoonshotAIChatOptions`, `MoonshotAILanguageModelOptions`, deprecated
  `MoonshotAIProviderOptions`, and `MoonshotAIChatRequestExt` now exist
- shared compat request normalization maps `thinking.budgetTokens -> thinking.budget_tokens` and
  `reasoningHistory -> reasoning_history`
- provider-key normalization rewrites `moonshot` onto `moonshotai` before transport

Status: completed

## MPSA-M3 - Model surface, docs, and tests aligned

Acceptance criteria:

- curated model/default ids follow the audited package subset
- public-path tests lock the canonical MoonshotAI boundary
- docs/workstreams/changelog explain the intentional chat-only design

Current state:

- curated model constants/defaults now follow the Kimi K2 + Moonshot V1 subset used by
  `repo-ref/ai/packages/moonshotai`
- public import/runtime/path tests now cover canonical id, alias fallback, URL alignment, and the
  intentional lack of completion/non-text families
- changelog plus dedicated workstream docs now record the canonical `moonshotai` surface and the
  intentional lack of a unified image/completion package story

Status: completed

## MPSA-M4 - Follow-up cleanup after downstream migration

Acceptance criteria:

- the remaining compatibility-only pieces are explicitly named
- future cleanup no longer depends on rediscovering why they exist

Current state:

- the only notable remaining compatibility bridge is the hidden low-level `moonshot` alias
- TypeScript-only `ProviderSettings` / `VERSION` exports remain intentionally deferred
- any future image/completion support is blocked on upstream AI SDK moving the package boundary,
  not on missing local glue code

Status: in progress
