# Cohere Unified Provider Surface - Milestones

Last updated: 2026-04-10

## CHP-M0 - Scope locked

Acceptance criteria:

- the AI SDK Cohere provider surface is identified
- the historical Siumai drift is written down explicitly
- the target architecture chooses native `/v2` unification over compat-driven split routing

Status: completed

## CHP-M1 - Canonical provider identity is corrected

Acceptance criteria:

- `cohere` remains the built-in provider id
- native metadata/provider catalog no longer describe Cohere as rerank-only
- old docs that treated compat embedding as the canonical public story are superseded

Status: completed

## CHP-M2 - Native runtime supports the AI SDK family set

Acceptance criteria:

- `siumai-provider-cohere` can execute chat
- `siumai-provider-cohere` can execute embeddings
- rerank remains supported on the same native `/v2` base

Status: completed

## CHP-M3 - Unified public/runtime surface exists

Acceptance criteria:

- `Provider::cohere()` and `Siumai::builder().cohere()` expose one built provider identity
- registry `language_model("cohere:model")` is supported
- the same provider id also exposes embeddings and rerank

Status: completed

## CHP-M4 - Old split-surface assumptions are removed

Acceptance criteria:

- tests/docs that locked "chat unsupported" are replaced with positive unified coverage
- compat-only embedding is no longer treated as the preferred public Cohere path
- default-model behavior is no longer rerank-biased on the unified provider surface

Status: completed

## CHP-M5 - Ongoing parity maintenance

Acceptance criteria:

- Cohere typed options stay aligned with the audited AI SDK provider package
- provider-owned curated `chat` / `embedding` / `rerank` model constants are exposed on the public
  Rust surface
- AI SDK-style option aliases (`CohereLanguageModelOptions`,
  `CohereEmbeddingModelOptions`, `CohereRerankingModelOptions`) exist with migration coverage
- curated model documentation stays in sync with the chosen reference set

Status: completed
