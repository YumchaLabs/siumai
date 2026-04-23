# Upload File Result Surface Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Contract audit

- [x] Compare the local high-level upload result behavior against AI SDK `uploadFile()` and
  `FilesV4UploadFileResult`.
- [x] Confirm that helper-owned `blob` filename fallback and result-field backfills are not part of
  the upstream contract.

## Track B - Shared file type cleanup

- [x] Make `FileUploadRequest.filename` optional.
- [x] Make `FileObject.filename` optional.
- [x] Remove helper-owned `blob` filename defaulting from the stable upload path.

## Track C - Result shaping alignment

- [x] Stop backfilling `UploadFileResult.filename` from request-time fallbacks.
- [x] Stop backfilling `UploadFileResult.media_type` from request-time media-type detection.
- [x] Keep `providerMetadata` limited to provider-owned extra fields instead of injecting generic
  file bookkeeping.

## Track D - Provider/runtime hardening

- [x] Update provider upload adapters so missing provider response fields stay missing instead of
  being reconstructed from the request.
- [x] Keep explicit filename warnings for provider families that still reject or ignore filenames.

## Track E - Docs and verification

- [x] Add a dedicated workstream for the result-side alignment slice.
- [x] Update `CHANGELOG.md` `Unreleased`.
- [x] Update the structural alignment matrix entry for the upload helper.
- [x] Run focused integration tests for `siumai::files`.
- [x] Run focused package/unit tests for the changed file-upload crates.
