# Video Generate Result Materialization Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Result-boundary audit

- [x] Verify the audited AI SDK helper materializes URL-backed final videos by default.
- [x] Verify the upstream result shape includes a mandatory first `video` plus `videos`.
- [x] Confirm the remaining Rust gap is at the helper/result boundary, not the provider task
  runtime.

## Track B - Helper/result alignment

- [x] Make `GenerateOptions` own default URL materialization behavior for `generate(...)`.
- [x] Allow helper-level download HTTP config during default URL materialization.
- [x] Add mandatory first `video` fields to `GenerateVideoResult` and
  `GenerateMaterializedVideoResult`.
- [x] Add direct `bytes()` / `base64()` accessors on `GeneratedVideo` for already-materialized
  assets.
- [x] Preserve source URL visibility through metadata fallback after default materialization.
- [x] Keep non-downloadable URL schemes such as `gs://...` raw on the helper path with an explicit
  warning instead of failing default URL materialization.

## Track C - Compatibility and verification

- [x] Keep an explicit opt-out path for callers that still want raw URL-backed `GeneratedVideo`.
- [x] Keep `generate_materialized(...)` as the explicit `MaterializedVideo` normalization helper.
- [x] Add focused regression coverage for default URL materialization.
- [x] Run focused public-surface compile coverage for the video family.

## Track D - Docs and changelog

- [x] Add a dedicated workstream for this follow-up slice.
- [x] Update the older `video-model-family-alignment` notes so the follow-up state is no longer
  stale.
- [x] Update the structural-alignment matrix/todo notes for the video helper row.
- [x] Update `CHANGELOG.md` `Unreleased`.
