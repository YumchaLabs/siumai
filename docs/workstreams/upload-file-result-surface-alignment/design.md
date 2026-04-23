# Upload File Result Surface Alignment - Design

Last updated: 2026-04-21

## Context

The earlier upload-file slices already aligned the Rust helper much more closely with the AI SDK on
the input side:

- URL-shaped strings are rejected at runtime instead of being modeled as a first-class upload type.
- `files::upload(...)` accepts shared `DataContent` directly.
- the local `UploadFileData` compatibility wrapper has been removed.

However, one result-side mismatch remained:

- the AI SDK helper forwards `filename` and `mediaType` only when the provider returns them
- the Rust helper still synthesized result values from request-time fallbacks and generic
  `FileObject` fields
- missing filenames were still normalized to `"blob"` on the Rust path
- provider metadata on the helper path also mixed provider-owned extra fields with generic file
  bookkeeping synthesized by the helper layer

That made the public Rust surface look more certain than the upstream helper contract actually is.

## Goal

- Align the stable Rust upload helper with the AI SDK `uploadFile()` result contract more closely.
- Treat upload filenames as optional request/result data instead of a helper-owned default.
- Keep provider metadata provider-owned instead of injecting generic helper bookkeeping.

## Non-goals

- Do not widen uploads to support remote URLs.
- Do not remove provider-specific purpose handling from low-level file APIs.
- Do not pretend every provider will always return `filename` or `mediaType`.

## Chosen design

### 1. Make shared file filenames truly optional

`FileUploadRequest.filename` and `FileObject.filename` now use `Option<String>`.

This matches the upstream AI SDK direction more honestly:

- upload calls may omit a filename entirely
- provider responses may omit a filename entirely
- the helper should not invent `"blob"` as a stable result value

### 2. Stop synthesizing helper result fields from request fallbacks

`UploadFileResult` now keeps:

- `providerReference` from the provider file id
- `filename` only when the provider response exposes one
- `mediaType` only when the provider response exposes one
- `providerMetadata` only when the provider response exposes provider-owned extra fields

The helper still detects a media type for the outgoing upload request when needed, but that
request-time decision is no longer echoed back as a fake provider result field.

### 3. Keep provider metadata scoped to provider-owned extras

The helper no longer injects generic file bookkeeping such as:

- `filename`
- `purpose`
- `bytes`
- `createdAt`
- `status`

into `providerMetadata`.

Those values already belong to the low-level `FileObject` contract for list/retrieve paths. The
AI SDK-style high-level helper should expose only the narrower stable result surface plus
provider-owned metadata extras.

### 4. Remove local filename backfills from provider adapters

Provider-side upload adapters now avoid filling response filename/media-type fields from the
request when the provider response did not return them.

This keeps the high-level helper aligned with the upstream "provider pass-through" result
semantics instead of letting request-time assumptions leak back into public result values.

## Follow-up

If future parity work exposes a need for richer file-upload result inspection, that should happen
through a provider-owned extension surface or a separate low-level file-object API, not by
silently widening the AI SDK-style helper result contract again.
