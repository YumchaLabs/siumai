# Upload File Wrapper Removal Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Wrapper audit

- [x] Confirm `UploadFileData` no longer carries a meaningful boundary after direct `DataContent`
  upload support landed.
- [x] Verify repo-internal usage is limited enough to remove the wrapper safely.

## Track B - Public surface simplification

- [x] Remove `UploadFileData` from the public helper/module surface.
- [x] Route upload helper inputs through shared `DataContent` plus raw byte/string conversions.
- [x] Update tests to use direct shared carrier / raw bytes instead of the removed wrapper.

## Track C - Docs and changelog

- [x] Record the wrapper-removal slice in a dedicated workstream folder.
- [x] Update `CHANGELOG.md` `Unreleased`.

## Track D - Intentional boundaries

- [-] Do not widen uploads to support remote URLs.
- [-] Do not reintroduce an upload-only wrapper unless a real semantic boundary appears.
