# Video Model Family Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Upstream audit

- [x] Audit `repo-ref/ai/packages/provider/src/video-model/v4/video-model-v4.ts`.
- [x] Audit `repo-ref/ai/packages/provider/src/video-model/v4/video-model-v4-call-options.ts`.
- [x] Audit `repo-ref/ai/packages/provider/src/video-model/v4/video-model-v4-result.ts`.
- [x] Audit `repo-ref/ai/packages/ai/src/registry/provider-registry.ts`.
- [x] Audit `repo-ref/ai/packages/ai/src/generate-video/generate-video.ts`.

## Track B - Core family-model surface

- [x] Add a dedicated task-oriented video family in `siumai-core`.
- [x] Keep AI SDK-auditable naming through `VideoModelV4` while preserving Rust-first
  `create_task(...)` / `query_task(...)` semantics.
- [x] Bridge existing `VideoGenerationCapability` implementations into the new family through a
  blanket adapter.
- [x] Re-export the new core video module from `siumai-core`.

## Track C - Registry and factory alignment

- [x] Add dedicated `ProviderFactory` video family constructors.
- [x] Add `ProviderRegistryHandle::video_model("provider:model")`.
- [x] Add `VideoModelHandle` with LRU/TTL caching.
- [x] Preserve `BuildContext` propagation on the dedicated registry video lane instead of relying
  on legacy generic-client-only construction.
- [x] Backfill missing request-local `model` values from the registry/default model id on task
  creation.

## Track D - Public facade alignment

- [x] Add `siumai::video`.
- [x] Expose stable `create_task(...)` and `query_task(...)` helpers.
- [x] Expose `wait_for_task(...)` as the explicit polling helper.
- [x] Expose `generate(...)` as a high-level create-and-poll helper without copying the
  TypeScript callable-model shape.
- [x] Keep `generate_materialized(...)` as the explicit `MaterializedVideo` normalization helper
  alongside the now auto-materializing `generate(...)` path.
- [x] Re-export the core `VideoModel*` family traits plus shared request/result types on the
  facade surface.
- [x] Add compile/public-path/registry coverage for the new public video lane.

## Track E - Docs and changelog

- [x] Create a dedicated `docs/workstreams/video-model-family-alignment/` folder.
- [x] Update `docs/workstreams/ai-sdk-structural-alignment/data-structure-matrix.md`.
- [x] Update older Google Vertex / Fearless Refactor workstreams that still described video as an
  extension-only registry story.
- [x] Update `CHANGELOG.md` `Unreleased` with the new video family-model surface.

## Track F - Intentional deferrals

- [-] Do not fabricate AI SDK-style `doGenerate(...)` on the core Rust video family while the real
  runtime remains task-oriented.
- [-] Do not pretend the registry/facade layer already owns a provider-agnostic final-file result
  contract equivalent to AI SDK `VideoModelV4Result`.

## Track G - Follow-up

- [x] Add stable video-family metadata for `maxVideosPerCall`.
- [x] Make `generate(...)` honor explicit or model-default `maxVideosPerCall` instead of always
  splitting into single-video tasks.
- [x] Refactor `GenerateVideoResult` to expose final generated-video assets separately from the
  underlying completed task responses.
- [x] Preserve provider-owned per-video metadata on the audited multi-video providers
  (Gemini/Vertex) so the facade can surface final assets without flattening them back into raw task
  responses.
- [x] Add explicit materialization helpers for final generated videos, including URL-backed
  downloads plus byte/base64 accessors on the materialized file representation.
- [x] Let the shared `VideoGenerationRequest` represent AI SDK-style image-only video prompts where
  `prompt` can be omitted.
- [x] Add a stable AI SDK-style prompt union shape (`VideoGenerationPrompt` /
  `GenerateVideoPrompt`) plus request constructors/helpers built on that union.
- [x] Return a specialized empty-result video error (`LlmError::NoVideoGenerated`) with final
  response metadata instead of treating successful-but-empty video runs as generic parse failures.
- [x] Add shared AI SDK-style video metadata carriers plus facade accessors.
  - stable `VideoModelProviderMetadata` / `VideoModelResponseMetadata` now live on the shared
    type surface
  - `GenerateVideoResponseMetadata` now exposes best-effort create/query/logical-call projections
  - `GenerateVideoResult` / `GenerateMaterializedVideoResult` now expose
    `video_model_responses()` for the AI SDK-style response list view
- [x] Make high-level `generate(...)` auto-materialize URL-backed final videos by default, with
  explicit opt-out plus helper-level download `HttpConfig`.
- [x] Expose AI SDK-style first `video` fields on `GenerateVideoResult` /
  `GenerateMaterializedVideoResult`, and add direct `GeneratedVideo::bytes()` /
  `GeneratedVideo::base64()` accessors for already-inline assets.
- [x] Add provider-owned download adapters for the currently audited provider-reference-only video
  paths (Gemini and MiniMaxi) through the shared video-family capability surface.
- [x] Promote task-status final-asset references onto canonical `providerReference` while keeping
  legacy `fileId` compatibility and camelCase stable serde output.
- [x] Lock top-level public-path parity around the current task-query split: Gemini returns
  canonical `providerReference`, while Vertex stays on raw `videoUrl` without fabricating a shared
  provider reference.
- [x] Restore the feature-gated provider-local Gemini video regression lane so `cargo test
  --features google` actually compiles and enumerates those unit tests again.
- [x] Make direct `GeminiVideo::new(...)` helper construction honor `GeminiConfig.http_transport`
  when no explicit transport override is passed, so video task polling uses the same custom-fetch
  wiring as the client path.
- [x] Make video helper/provider-metadata readers accept upstream `google-vertex` alias roots on
  the read path while preserving the existing public `vertex` aggregation root.
- [-] Leave providers that still require a different authenticated download runtime (for example
  current Vertex GCS-owned outputs) on the raw URL-backed path until that runtime is
  audited and shared.
