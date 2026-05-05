# Video Model Family Alignment - Milestones

Last updated: 2026-04-21

## Completed

- Audited the upstream AI SDK video model, registry, and generate-video references.
- Added the core task-oriented `VideoModel` family in `siumai-core`, with `VideoModelV4` retained
  as the AI SDK-facing marker.
- Bridged legacy `VideoGenerationCapability` implementations into that family instead of forking a
  second video runtime.
- Added dedicated registry factory hooks for video family-model construction.
- Added `ProviderRegistryHandle::video_model("provider:model")` and `VideoModelHandle`.
- Preserved registry `BuildContext` propagation and default-model backfill on the dedicated video
  lane.
- Added the public `siumai::video::{create_task, query_task, wait_for_task, generate}` facade
  surface.
- Added stable `max_videos_per_call` metadata on the capability/core/registry path with audited
  defaults for the current native providers.
- Refactored `siumai::video::generate(...)` so it batches by `max_videos_per_call` and returns
  final generated-video assets separately from completed task responses.
- Tightened the high-level helper result boundary against AI SDK
  `experimental_generateVideo()`:
  `siumai::video::generate(...)` now auto-materializes URL-backed final videos by default,
  `GenerateVideoResult` / `GenerateMaterializedVideoResult` now expose an AI SDK-style first
  `video` plus `videos`, and `GeneratedVideo` now has direct `bytes()` / `base64()` accessors for
  already-inline assets.
- Added a shared provider-owned materialization hook for provider-reference-only final video
  assets, then wired audited Gemini and MiniMaxi implementations through the same task-oriented
  video-family dispatch path instead of adding a parallel trait hierarchy.
- Kept `siumai::video::generate_materialized(...)` as the explicit `MaterializedVideo`
  normalization helper on top of the now auto-materializing `generate(...)` path.
- Added explicit `GeneratedVideo::materialize(...)` and result-level materialization helpers so
  URL-backed assets can be downloaded into byte/base64-backed files without collapsing the task
  model.
- Added a specialized `LlmError::NoVideoGenerated` path so successful-but-empty multi-task video
  runs now preserve final response metadata instead of surfacing only a generic parse failure.
- Added shared AI SDK-style `VideoModelProviderMetadata` / `VideoModelResponseMetadata` on the
  stable facade and best-effort projection helpers from task-oriented video helper responses onto
  that shared metadata view.
- Promoted task-status final-asset references onto a canonical `providerReference` lane while
  keeping legacy `fileId` as a compatibility field, and made stable video task serde prefer the
  public camelCase payload shape with snake_case input aliases.
- Locked top-level query-task parity around the audited provider split: Gemini now regresses on
  canonical `providerReference`, while Vertex regresses on the intentionally raw `videoUrl` path.
- Restored the feature-gated Gemini provider-local video regression lane so the `video.rs` unit
  tests compile again when the real `google` feature is enabled.
- Tightened direct Gemini video helper construction so config-owned custom transports are reused on
  task polling even when callers do not pass a second explicit transport override.
- Made video-family metadata readers accept the upstream `google-vertex` alias on the read path
  while keeping the stable Rust aggregation root under `vertex`.
- Tightened aggregated `provider_metadata` merging so provider-root fields beyond `videos[]` /
  `tasks[]` are preserved across create/query task metadata on the audited video lanes.
- Updated the shared `VideoGenerationRequest` shape so `prompt` is optional, matching AI SDK's
  image-only generate-video prompt model more closely on the stable Rust request surface.
- Added a stable `VideoGenerationPrompt` shape plus `GenerateVideoPrompt` alias and request
  helpers, so the Rust surface can express the same text-or-image prompt union as the upstream AI
  SDK helper layer.
- Preserved provider-owned multi-video metadata on the audited Gemini/Vertex polling paths so the
  facade can recover final assets without inventing a fake provider-agnostic runtime.
- Added compile, public-path, and lower-contract coverage for the new video family story.

## Next

- Revisit the smaller remaining provider-owned download gaps that still need a separate
  authenticated runtime, including current Vertex GCS-owned URL outputs.
