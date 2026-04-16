# Cohere Unified Provider Surface - TODO

Last updated: 2026-04-10

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Canonical provider identity

- [x] Audit the AI SDK Cohere provider surface in `repo-ref/ai/packages/cohere/src/*`.
- [x] Record the old Siumai drift and choose the target architecture.
- [x] Update native provider metadata so Cohere is no longer described as rerank-only.
- [x] Update provider catalog output and provider typing expectations for the unified Cohere story.

## Track B - Native runtime architecture

- [x] Expand `siumai-provider-cohere` beyond `/v2/rerank`.
- [x] Add native `/v2/chat` request/response execution.
- [x] Add native `/v2/embed` request/response execution.
- [x] Keep `/v2/rerank` behavior and typed rerank options intact during the expansion.
- [-] Keep the OpenAI-compatible Cohere embedding route as the canonical public API.
  - rejected because AI SDK models Cohere as a native `/v2` provider, not as an
    OpenAI-compatible preset

## Track C - Unified builder and registry surface

- [x] Refactor `CohereProviderFactory` so `language_model("cohere:model")` is no longer rejected.
- [x] Expose one built `cohere` client with chat/embedding/rerank capabilities.
- [x] Remove the provider-wide default model fallback and require explicit model ids.

## Track D - Regression coverage

- [x] Replace rerank-only boundary-guard tests with unified positive-path coverage.
- [x] Add public-path parity tests across `Siumai::builder().cohere()`, `Provider::cohere()`, and
  config-first clients.
- [x] Add registry contract tests for chat, embedding, and rerank routing on one provider id.
- [x] Add public-surface compile/runtime guards for the expanded Cohere provider-ext API.
  - compile guards now also pin `provider_ext::cohere::{chat, embedding, rerank, model_sets}` plus
    AI SDK-style option aliases on the facade crate itself

## Track E - Docs and migration

- [x] Create a dedicated workstream folder in `docs/workstreams/`.
- [x] Update structural-alignment docs now that Cohere is no longer Red.
- [x] Update unreleased changelog sections for the runtime/public-surface changes.
- [x] Rewrite older docs that taught the rerank-only/compat-embedding split as the desired end
  state.

## Follow-up audit

- [x] Keep auditing Cohere typed options and request/response extensions against
  `repo-ref/ai/packages/cohere/src/*`.
  - typed chat `thinking`, embedding `inputType` / `truncate` / `outputDimension`, and rerank
    `maxTokensPerDoc` / `priority` now match the audited AI SDK option surface
  - `provider_ext::cohere` now also exposes curated `models::{chat, embedding, rerank, model_sets}`
    constants plus AI SDK-style option aliases (`CohereLanguageModelOptions`,
    `CohereEmbeddingModelOptions`, `CohereRerankingModelOptions`) with deprecated migration aliases
    retained for side-by-side compile checks
  - native chat now also mirrors the audited AI SDK provider-defined-tool warning behavior:
    unsupported provider tools are filtered from the Cohere wire request but still surface as
    stable `unsupported { feature: "provider-defined tool <id>" }` warnings
  - native chat streaming now also emits stable `ChatStreamPart::StreamStart { warnings }` and
    runtime `raw` parts when `includeRawChunks` is enabled, so the Cohere stream lane is closer to
    the audited AI SDK `stream-start -> raw -> response-metadata -> ...` structure
- [x] Enforce the audited AI SDK runtime batch-size guard on native embeddings.
  - native `cohere` embedding requests now fail fast when a single `/v2/embed` call carries more
    than `96` inputs, matching `repo-ref/ai/packages/cohere/src/cohere-embedding-model.ts`
- [-] Mirror TypeScript-only package exports such as `CohereProviderSettings`,
  `CohereErrorData`, or `VERSION` one-for-one on the Rust side.
  - deferred intentionally because the Rust provider facade uses `CohereConfig`,
    `CohereBuilder`, and `Provider::cohere()` as the stable settings/construction story, and other
    aligned provider packages do not expose separate TS-style settings/error-data aliases either
- [ ] Revisit the curated model list whenever the audited AI SDK reference set changes.
