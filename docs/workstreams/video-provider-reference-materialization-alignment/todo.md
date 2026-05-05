# Video Provider Reference Materialization Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared contract

- [x] Identify the smallest shared type needed for provider-owned final-video downloads.
- [x] Add `MaterializedVideoAsset` on the shared video type surface.
- [x] Extend `VideoGenerationCapability` and `VideoModel` with a default unsupported
  `materialize_video_reference(...)` hook.

## Track B - Dispatch alignment

- [x] Forward the new capability through registry-backed `VideoModelHandle`.
- [x] Forward the same hook through `Siumai` and `ClientBackedVideoModel`.
- [x] Keep the video-family architecture on one dispatch chain instead of adding a second trait
  hierarchy just for provider-owned downloads.

## Track C - Provider implementations

- [x] Implement provider-owned reference materialization for Gemini on top of existing shared file
  management.
- [x] Implement provider-owned reference materialization for MiniMaxi on top of existing shared
  file management.
- [-] Leave current Vertex GCS-style outputs on the raw URL-backed `gs://...` path until a shared
  authenticated download story exists.

## Track D - Facade behavior

- [x] Let `siumai::video::generate(...)` best-effort materialize provider-reference-backed final
  assets after the URL materialization pass.
- [x] Keep unsupported providers on the raw `ProviderReference` path instead of fabricating fake
  generic downloads.
- [x] Preserve provider-reference metadata on the helper/provider-metadata path when a reference is
  materialized eagerly.
- [x] Keep standalone `GeneratedVideo::materialize(...)` intentionally unsupported for raw provider
  references, because the isolated asset does not own a provider client.

## Track E - Verification and docs

- [x] Add focused regression coverage for supported provider-reference materialization.
- [x] Add focused regression coverage for the unsupported fallback path.
- [x] Add top-level public-path parity coverage so Gemini canonical `providerReference` task-query
  responses and Vertex raw-URL task-query responses cannot silently drift apart again.
- [x] Update the older video workstreams so the remaining-gap notes are no longer stale.
- [x] Update the structural alignment matrix/todo/milestones and `CHANGELOG.md`.
