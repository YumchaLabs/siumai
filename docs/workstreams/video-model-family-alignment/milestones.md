# Video Model Family Alignment - Milestones

Last updated: 2026-04-20

## Completed

- Audited the upstream AI SDK video model, registry, and generate-video references.
- Added the core task-oriented `VideoModelV3` / `VideoModelV4` / `VideoModel` family in
  `siumai-core`.
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
- Added `siumai::video::generate_materialized(...)` as the high-level helper closest in role to
  AI SDK `experimental_generateVideo()` without hiding the explicit materialization step.
- Added explicit `GeneratedVideo::materialize(...)` and result-level materialization helpers so
  URL-backed assets can be downloaded into byte/base64-backed files without collapsing the task
  model.
- Added a specialized `LlmError::NoVideoGenerated` path so successful-but-empty multi-task video
  runs now preserve final response metadata instead of surfacing only a generic parse failure.
- Added shared AI SDK-style `VideoModelProviderMetadata` / `VideoModelResponseMetadata` on the
  stable facade and best-effort projection helpers from task-oriented video helper responses onto
  that shared metadata view.
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

- Decide whether high-level `generate(...)` should auto-materialize URL-backed videos by default.
- Decide whether provider-reference-only assets should eventually grow provider-owned download
  adapters.
