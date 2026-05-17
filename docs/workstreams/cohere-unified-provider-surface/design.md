# Cohere Unified Provider Surface - Design

Last updated: 2026-04-10

## Historical problem

AI SDK models Cohere as one native provider surface:

- `languageModel()`
- `embeddingModel()`
- `rerankingModel()`

all rooted at the native Cohere `/v2` API.

Siumai previously drifted from that shape:

- the native `siumai-provider-cohere` crate was treated as rerank-only
- `Provider::cohere()` / `Siumai::builder().cohere()` inherited that rerank-only story
- native registry metadata intentionally advertised only `rerank`
- `registry.language_model("cohere:model")` was rejected
- embeddings were routed through the generic OpenAI-compatible Cohere preset on `/v1/embeddings`
- old tests/docs locked that split-runtime decision in place

That meant the public `cohere` provider identity was materially misaligned with AI SDK.

## Implemented design

### 1. Canonical id remains `cohere`

`cohere` stays the built-in provider id, but it now means the AI SDK-shaped unified provider
surface rather than a rerank-only native crate.

### 2. Native Cohere owns the `/v2` family set

The native provider crate is now the canonical runtime for:

- `/v2/chat`
- `/v2/embed`
- `/v2/rerank`

This replaces the older split where embeddings depended on the generic OpenAI-compatible Cohere
preset.

### 3. Registry and facade expose one unified provider story

`CohereProviderFactory` now builds one native Cohere client that reports:

- chat
- streaming
- tools
- embedding
- rerank

under the same `cohere` provider id.

That means:

- `Provider::cohere()`
- `Siumai::builder().cohere()`
- `registry.language_model("cohere:model")`
- `registry.embedding_model("cohere:model")`
- `registry.rerank_model("cohere:model")`

all participate in the same provider identity instead of mixing native and compat stories.

### 4. Provider-wide default model was retired

A unified Cohere surface cannot safely inherit the old rerank-biased provider-wide default model.

The canonical native path now requires an explicit model id, matching the AI SDK provider/model
split more closely and avoiding ambiguity between chat, embedding, and rerank families.

### 5. The OpenAI-compatible Cohere preset remains compatibility-only

`openai().compatible("cohere")` still exists as an opt-in compatibility path for users who need
that older OpenAI-compatible contract, but it is no longer the canonical public Cohere story.

## Public/runtime consequences

- native metadata now describes Cohere as a unified `/v2` provider
- provider catalog output now lists curated Cohere chat/embed/rerank models
- public facade exports the unified typed Cohere surface (`CohereClient`, chat/embed/rerank
  options, request extensions, thinking config, embedding input enums, curated
  `models::{chat, embedding, rerank, model_sets}`, and AI SDK-style option aliases such as
  `CohereLanguageModelOptions` / `CohereEmbeddingModelOptions` /
  `CohereRerankingModelOptions`)
- public facade exports Rust package analogues for `cohere`, `createCohere`,
  `CohereProviderSettings`, and `VERSION` as `cohere()`, `create_cohere()`,
  `CohereProviderSettings`, and `VERSION`
- public-path tests now assert positive unified behavior instead of intentional chat rejection
- boundary tests now pin explicit-model semantics and the separation between native `/v2` and
  compat `/v1`

## Validation

The implemented surface is currently locked by:

- native/registry contract tests for chat, embeddings, and rerank
- public-path parity tests across builder/provider/config/registry paths
- public-surface compile guards for the Cohere typed exports, curated model modules, package
  settings/version helpers, and AI SDK compatibility aliases
- boundary tests proving native `cohere` requires explicit models and remains distinct from the
  compatibility preset

## Remaining follow-up

- Continue auditing Cohere typed request/option parity against `repo-ref/ai/packages/cohere/src/*`
  as upstream adds fields or semantics.
- Keep the curated model catalog aligned with the audited AI SDK reference set.
- Keep `generateId` and non-index error-data mirrors intentionally deferred unless a cross-provider
  Rust pattern emerges. `CohereErrorData` exists in the upstream internals but is not currently
  re-exported from the audited package index, so it is not part of the Rust package-surface target.
