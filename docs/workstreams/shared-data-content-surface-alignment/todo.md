# Shared Data Content Surface Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared carrier audit

- [x] Audit how upstream `repo-ref/ai` reuses `DataContent` beyond prompt-only APIs.
- [x] Identify the existing Rust public surfaces that still require parallel binary/base64 wrapper
  enums.

## Track B - Public interoperability

- [x] Add shared `DataContent` conversion bridges for audio/image/video/upload payload carriers.
- [x] Add direct `from_data_content(...)` constructors on the main public helper/request types
  that currently require per-family wrappers.
- [x] Add unit/public coverage that locks the cross-surface construction path.

## Track C - Docs and changelog

- [x] Record the new shared-data interoperability lane in a dedicated workstream folder.
- [x] Update `CHANGELOG.md` `Unreleased`.

## Track D - Intentional boundaries

- [-] Do not remove family-specific payload types in the same slice.
- [-] Do not silently accept URL payloads through `DataContent` on file uploads.
