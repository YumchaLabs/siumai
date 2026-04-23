# Data Content Error Surface Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Error boundary audit

- [x] Audit all stable shared payload accessors that still expose `base64::DecodeError`.
- [x] Compare that boundary against upstream AI SDK `InvalidDataContentError`.

## Track B - Public surface alignment

- [x] Add a stable shared `InvalidDataContentError` to the spec/facade surface.
- [x] Route `DataContent` / audio / image / video shared-payload decoding through that error.
- [x] Reuse the same semantic error message on the file-upload helper path.
- [x] Add unit/facade coverage for invalid-base64 behavior.

## Track C - Docs and changelog

- [x] Record the slice in a dedicated workstream folder.
- [x] Update `CHANGELOG.md` `Unreleased`.

## Track D - Intentional boundaries

- [-] Do not keep raw `DecodeError` as a second stable error lane on the same shared accessors.
- [-] Do not widen `DataContent` itself in this slice.
