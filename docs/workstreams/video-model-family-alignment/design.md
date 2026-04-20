# Video Model Family Alignment - Design

Last updated: 2026-04-20

## Problem

Compared with `repo-ref/ai/packages/provider/src/video-model/v4/video-model-v4.ts`,
`repo-ref/ai/packages/ai/src/registry/provider-registry.ts`, and
`repo-ref/ai/packages/ai/src/generate-video/generate-video.ts`, Siumai's video story had become
split across three layers:

- the shared request/data shape had already moved closer to AI SDK `VideoModelV4CallOptions`
- provider-owned runtimes for xAI, Gemini, Google Vertex, and MiniMaxi already existed on real
  task-based routes
- but the stable family-model / registry / facade boundary still treated video mostly as an
  extension capability hanging off generic clients or `LanguageModelHandle`

That meant the main structural gap was no longer provider runtime coverage. It was the missing
family-centered architecture around video:

- no dedicated core `VideoModel*` family comparable to AI SDK `videoModel()`
- no dedicated registry `video_model("provider:model")` handle
- no stable public facade module comparable in role to `siumai::{text,image,speech,...}`
- build-context propagation on registry video construction still depended on legacy generic-client
  fallbacks

## Goals

- Promote video to a real family-model surface in `siumai-core`.
- Add a dedicated registry/factory path for task-oriented video models.
- Add a stable public facade module for task creation, task querying, and high-level polling.
- Preserve the existing provider-owned task runtimes and compatibility paths while making the
  family-model lane the architectural center.
- Keep the contract honest about the remaining gap against AI SDK's higher-level
  `experimental_generateVideo()` helper.

## Non-goals

- Do not fabricate a TypeScript-style callable provider object with `doGenerate(...)` if the Rust
  runtime is still task-oriented.
- Do not pretend all providers already share a stable final-file/result lifecycle equivalent to AI
  SDK `VideoModelV4Result`.
- Do not remove the legacy `VideoGenerationCapability` path immediately; keep it as a compatibility
  bridge while the new family handle becomes the preferred route.

## Chosen design

### 1. Add a task-oriented core video family

`siumai-core` now exposes:

- `VideoModelV3`
- `VideoModelV4`
- `VideoModel`

The naming stays audit-friendly against AI SDK's `VideoModelV4`, but the Rust contract remains
explicitly task-oriented:

- `create_task(...)`
- `query_task(...)`

This is deliberate. The execution lifecycle is not the same as AI SDK `doGenerate(...)`, so the
Rust surface should not pretend otherwise.

### 2. Bridge the legacy capability instead of forking the runtime

Any existing `VideoGenerationCapability` now automatically implements the new task-oriented family
through a blanket adapter.

That keeps current provider crates and compatibility paths working while making the family trait
the stable architectural center rather than the old capability trait.

### 3. Add a dedicated registry/factory video lane

`siumai-registry` now exposes:

- `ProviderFactory::video_model(...)`
- `ProviderFactory::video_model_with_ctx(...)`
- `ProviderFactory::video_model_family(...)`
- `ProviderFactory::video_model_family_with_ctx(...)`
- `ProviderRegistryHandle::video_model("provider:model")`
- `VideoModelHandle`

`VideoModelHandle` is the dedicated registry-side video family object:

- it caches native family models with the same LRU/TTL pattern as other family handles
- it preserves `BuildContext` propagation for `base_url`, `api_key`, custom transport, headers,
  and related construction state
- it backfills the default model id on task creation when the request omits `model`

### 4. Add a stable facade module instead of leaving video on extension-only paths

`siumai::video` is now the public task-oriented helper lane:

- `CreateTaskOptions`
- `QueryTaskOptions`
- `WaitForTaskOptions`
- `GenerateOptions`
- `create_task(...)`
- `query_task(...)`
- `wait_for_task(...)`
- `generate(...)`
- `generate_materialized(...)`

The module also re-exports the core `VideoModel*` family traits plus the underlying request/result
types, so public consumers can write against the stable family interface instead of provider-owned
escape hatches.

### 5. Add a Rust-first high-level polling helper above the task family

The facade now also exposes a higher-level helper layer above the task-oriented contract:

- `wait_for_task(...)` polls a submitted task to completion
- `generate(...)` submits and polls one or more tasks, returning final generated video assets plus
  the underlying completed task responses, warnings, response envelopes, and aggregated provider
  metadata

This stays intentionally Rust-first rather than cloning AI SDK's callable `doGenerate(...)` model
shape:

- the helper still builds on explicit task submission and task querying
- larger `count` values are batched using stable `max_videos_per_call` metadata when available,
  matching the AI SDK `maxVideosPerCall` split more closely
- final assets are extracted from provider-owned `videos[]` metadata when present, with truthful
  fallback to task-level `video_url` / `file_id`
- the helper still does not pretend the current stable result surface already owns automatic
  provider-agnostic binary materialization for URL-backed videos

### 6. Close more of the AI SDK result gap without hiding the remaining differences

This workstream now closes more than just model construction. The stable video lane also carries:

- object-safe `max_videos_per_call` metadata on the capability/core/registry path
- high-level `generate(...)` batching that honors explicit or model-default per-call limits
- a higher-level `generate_materialized(...)` helper that composes generation plus explicit final
  asset materialization
- a stable `GenerateVideoResult` that separates final `videos` from underlying `tasks`
- explicit `GeneratedVideo::materialize(...)` / `GenerateVideoResult::materialize_*` helpers for
  byte/base64 materialization of final assets
- a specialized `LlmError::NoVideoGenerated` result when completed tasks expose no final assets,
  carrying best-effort final response metadata instead of collapsing that case into a generic parse
  failure
- per-call `responses[*].provider_metadata` plus aggregated `provider_metadata`, including
  provider-root fields beyond `videos[]` / `tasks[]` when providers return additional metadata on
  the audited task-query path
- provider-side multi-video metadata preservation on the audited Gemini and Vertex paths
- a shared `VideoGenerationRequest.prompt: Option<String>` shape that can represent AI SDK-style
  image-only video prompts on the stable Rust request surface
- a stable `VideoGenerationPrompt` shape plus `GenerateVideoPrompt` alias so Rust can express the
  same text-only or image-plus-optional-text prompt union as AI SDK `GenerateVideoPrompt`

That said, Siumai still does not claim full parity with AI SDK
`experimental_generateVideo()` yet.

The main remaining gaps are now narrower and explicit:

- Rust keeps the core execution contract task-oriented instead of faking AI SDK
  `Experimental_VideoModelV4.doGenerate(...)`
- URL-backed final videos are not auto-downloaded/materialized by default inside `generate(...)`
  the way AI SDK returns `GeneratedFile`; Rust currently makes that step explicit through
  materialization helpers
- provider references still remain intentionally provider-owned on the stable Rust surface because
  there is no audited provider-agnostic download contract for them yet

Those now belong in the remaining result-materialization layer above the task-oriented runtime, not
as fake behavior on the core family contract.

## Validation

This slice is locked by:

- `siumai-registry/src/registry/entry/video_tests.rs`
- `siumai-registry/src/registry/factories/contract_tests.rs`
- `siumai/tests/public_surface_imports_test.rs`
- `siumai/tests/provider_public_path_parity_test.rs`
- `cargo check -p siumai-registry --features "google-vertex xai minimaxi"`
- `cargo check -p siumai --features "google-vertex xai minimaxi"`
- focused `cargo nextest` registry/facade parity runs on the same feature set
- focused `cargo nextest run -p siumai --lib video::tests::`

## Remaining follow-up

- Decide whether high-level `generate(...)` should auto-materialize URL-backed videos by default or
  keep explicit `materialize(...)` helpers as the stable Rust contract.
- Decide whether provider-reference-only results should eventually grow provider-owned download
  adapters without pretending they are generically portable.
