# Upload File Input Shape Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Surface audit

- [x] Compare Rust `UploadFileData` against upstream AI SDK `uploadFile` input shape.
- [x] Confirm whether the explicit `Url` variant has meaningful in-repo usage.

## Track B - Public shape tightening

- [x] Remove the explicit upload-only `Url` variant from the stable helper surface.
- [x] Preserve runtime URL-string rejection semantics on upload inputs.
- [x] Update helper coverage to assert that URL-like string inputs are still rejected.

## Track C - Docs and changelog

- [x] Record the slice in a dedicated workstream folder.
- [x] Update `CHANGELOG.md` `Unreleased`.

## Track D - Intentional boundaries

- [-] Do not silently accept remote URL uploads.
- [-] Do not collapse `UploadFileData` into a hard alias of `DataContent` in this slice.
