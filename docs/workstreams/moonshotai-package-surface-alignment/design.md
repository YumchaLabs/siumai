# MoonshotAI Package Surface Alignment - Design

Last updated: 2026-04-13

## Problem

Compared with `repo-ref/ai/packages/moonshotai`, Siumai still had several public-surface drifts on
the Moonshot wrapper path:

- public examples and some docs still treated `moonshot` as the canonical built-in/provider id
- top-level builder ergonomics had already moved to `moonshotai()`, but the surrounding examples
  and compat docs had not caught up
- typed Moonshot request options were missing, which forced callers back onto raw
  `providerOptions["moonshotai"]` JSON for the audited `thinking` / `reasoningHistory` lane
- shared compat request normalization did not yet pin Moonshot's wire-key conversion
  (`thinking.budget_tokens`, `reasoning_history`) at the canonical builder/config/registry paths
- curated model/default constants still drifted from the current AI SDK package subset

There was also an architectural question to resolve explicitly:

- whether Moonshot should be promoted into a broader unified provider surface with completion,
  image, or embedding families

After re-checking `repo-ref/ai/packages/moonshotai/src/index.ts` and
`repo-ref/ai/packages/moonshotai/src/moonshotai-provider.ts`, the answer is no:
the upstream package is a dedicated chat/language-model wrapper on top of the shared
OpenAI-compatible runtime, not a multi-family unified provider package.

## Goals

- Make `moonshotai` the canonical public/runtime identity.
- Keep `moonshot` only as a hidden migration alias.
- Expose AI SDK-style Moonshot typed language-model options on the provider-owned/public surface.
- Align curated model/default ids with the audited AI SDK package subset.
- Keep the wrapper explicitly chat-only instead of inventing unsupported completion/image/embedding
  APIs.
- Document the migration and rationale in `docs/workstreams`.

## Non-goals

- Do not create a new native Moonshot transport/runtime outside the shared OpenAI-compatible stack.
- Do not invent `completionModel()`, `embeddingModel()`, or `imageModel()` support that the
  audited upstream package does not expose.
- Do not mirror TypeScript-only exports such as `MoonshotAIProviderSettings` or `VERSION` one for
  one on the Rust facade.

## Chosen design

### 1. Canonical id is `moonshotai`

Public examples, builder shortcuts, config-first built-ins, and provider-facade exports now treat
`moonshotai` as the canonical identity.

The historical `moonshot` id is retained only as a hidden compatibility bridge inside the shared
OpenAI-compatible preset/config machinery so older downstream config strings do not break
immediately.

### 2. Keep the shared OpenAI-compatible runtime

MoonshotAI remains a provider-owned wrapper over the shared OpenAI-compatible runtime.

This matches the AI SDK architecture:

- `createMoonshotAI()` / `moonshotai()` return a language-model provider
- request shaping applies Moonshot-specific transforms on top of the shared chat-completions path
- there is no separate native non-chat Moonshot package in the audited upstream reference

### 3. Typed request options live under `providerOptions["moonshotai"]`

The Rust surface now exposes:

- `MoonshotAIChatOptions`
- `MoonshotAILanguageModelOptions`
- deprecated `MoonshotAIProviderOptions`
- `MoonshotAIThinkingConfig`
- `MoonshotAIThinkingType`
- `MoonshotAIReasoningHistory`
- `MoonshotAIChatRequestExt`

These types mirror the audited AI SDK option lane while preserving Rust-first implementation
stability.

### 4. Wire normalization is explicit and canonical

The shared compat boundary now locks Moonshot-specific request normalization on the canonical
`moonshotai` root:

- `thinking.budgetTokens -> thinking.budget_tokens`
- `reasoningHistory -> reasoning_history`

Alias-aware provider-key normalization still accepts legacy `moonshot` input and rewrites it onto
the canonical `moonshotai` path before transport.

### 5. Public surface stays chat-only by design

MoonshotAI is intentionally not promoted to a broader multi-family unified provider package.

The audited upstream package exposes:

- `moonshotai(modelId)`
- `chatModel(modelId)`
- `languageModel(modelId)`

and explicitly rejects:

- `embeddingModel(modelId)`
- `imageModel(modelId)`

Siumai mirrors that contract by keeping completion/image/embedding/rerank/speech/transcription
unsupported on the MoonshotAI wrapper boundary.

### 6. Curated model ids follow the audited package subset

The public model/constants story is now centered on the current AI SDK Moonshot subset:

- `kimi-k2`
- `kimi-k2-0905`
- `kimi-k2-thinking`
- `kimi-k2-thinking-turbo`
- `kimi-k2-turbo`
- `kimi-k2.5`
- `moonshot-v1-{8k,32k,128k}`

The public facade exposes those constants through both:

- `models::openai_compatible::moonshotai::*`
- `provider_ext::moonshotai::{model_sets, recommended}`

## Validation

This workstream is locked by:

- provider-local option serialization/request-normalization tests in
  `siumai-provider-openai-compatible`
- public import/runtime guards in
  `siumai/tests/public_surface_imports_test.rs`,
  `siumai/tests/openai_compatible_preset_guards_test.rs`,
  `siumai/tests/provider_public_path_parity_test.rs`, and
  `siumai/tests/moonshotai_openai_compat_url_alignment_test.rs`
- example/doc updates that now use `moonshotai` / `.moonshotai()` as the canonical public path

## Remaining follow-up

- Decide later whether the hidden low-level `moonshot` alias should be deleted entirely after
  downstream migration.
- Re-audit the wrapper if the upstream AI SDK Moonshot package grows beyond its current
  chat/language-model-only boundary.
- Keep TypeScript-only exports such as `MoonshotAIProviderSettings` and `VERSION` intentionally
  deferred unless a broader Rust cross-provider pattern emerges first.
