# Upload File Call Boundary Alignment - Design

Last updated: 2026-04-21

## Context

Earlier upload-file alignment slices already closed two important gaps:

- the helper no longer pretends URL uploads are a first-class supported input form
- shared `DataContent` decoding now uses a stable semantic error lane

But one public ergonomics boundary was still lagging behind the upstream AI SDK intent:

- upstream `uploadFile(...)` accepts shared `DataContent` directly
- local Rust `files::upload(...)` still required callers to first wrap inputs in `UploadFileData`

That wrapper had become mostly a compatibility shell around the same two payload carriers:

- raw bytes
- base64 string content

Forcing callers through that extra wrapper made the stable helper surface less direct than it
needed to be, especially after shared `DataContent` had already been promoted across prompt,
audio, image, and video entrypoints.

## Goal

- Let `files::upload(...)` accept shared `DataContent` directly.
- Preserve current Rust ergonomics for `Vec<u8>`, `&[u8]`, and `UploadFileData`.
- Keep `UploadFileData` as a compatibility type for now instead of forcing an immediate breaking
  removal.

## Non-goals

- Do not remove `UploadFileData` in the same slice.
- Do not widen uploads to support remote URLs.
- Do not change the provider-facing upload payload/result contracts.

## Chosen design

### 1. Make the helper generic over input carriers

`files::upload(...)` now accepts `D: Into<UploadFileData>` instead of only an already-materialized
`UploadFileData`.

That means callers can pass:

- `DataContent`
- `Vec<u8>`
- `&[u8]`
- the existing `UploadFileData`

without first constructing the compatibility wrapper by hand.

### 2. Keep `UploadFileData` as a compatibility shell, not the required call shape

The wrapper still exists because it continues to provide:

- explicit bytes/base64 constructors
- a compatibility import path already used in the repo

But it is no longer the mandatory function-parameter boundary, which brings the stable Rust helper
closer in spirit to the upstream AI SDK helper.

## Follow-up

That later follow-up was completed in
`docs/workstreams/upload-file-wrapper-removal-alignment/`, where the compatibility wrapper was
removed entirely after the direct shared-carrier call boundary had already been proven out.
