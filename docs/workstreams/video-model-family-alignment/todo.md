# Video Model Family Alignment - TODO

Last updated: 2026-04-20

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
- [x] Expose `generate_materialized(...)` as the high-level helper closest in role to AI SDK
  `experimental_generateVideo()` while keeping materialization explicit.
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
- [ ] Decide whether high-level `generate(...)` should auto-materialize URL-backed final videos by
  default instead of keeping that step explicit.
- [ ] Decide whether provider-reference-only assets should eventually expose provider-owned download
  adapters.
