# Upload File Call Boundary Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Boundary audit

- [x] Compare the local `files::upload(...)` parameter boundary against upstream `uploadFile(...)`.
- [x] Verify whether `UploadFileData` still carries value beyond direct helper input acceptance.

## Track B - Public helper alignment

- [x] Allow `files::upload(...)` to accept shared `DataContent` directly.
- [x] Preserve ergonomic direct bytes input without forcing callers through `DataContent`.
- [x] Keep existing `UploadFileData` call sites working as a compatibility lane.
- [x] Add helper coverage that locks direct shared-data-content uploads.

## Track C - Docs and changelog

- [x] Record the slice in a dedicated workstream folder.
- [x] Update `CHANGELOG.md` `Unreleased`.

## Track D - Intentional boundaries

- [-] Do not remove `UploadFileData` in the same slice.
- [-] Do not widen uploads to support remote URLs.
