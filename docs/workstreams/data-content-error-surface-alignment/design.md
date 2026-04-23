# Data Content Error Surface Alignment - Design

Last updated: 2026-04-21

## Context

The previous shared-data-content slice made `DataContent` reusable across prompt, audio, image,
video, and file-upload surfaces. However, the public Rust accessors that actually decode that
carrier still exposed the low-level `base64::DecodeError` directly:

- `DataContent::as_bytes()`
- `AudioInputData::as_bytes()`
- `SttRequest::audio_bytes()`
- `AudioTranslationRequest::audio_bytes()`
- `ImageEditFileData::as_bytes()`
- `VideoGenerationFileData::as_bytes()`

That is a leaky abstraction relative to the upstream AI SDK, which surfaces this category through
the higher-level `InvalidDataContentError` contract instead of forcing callers to reason about the
particular base64 codec implementation.

## Goal

- Expose a stable shared `InvalidDataContentError` on the Rust facade.
- Route shared data-content decoding failures through that error instead of
  `base64::DecodeError`.
- Reuse the same semantic error message across prompt/audio/image/video/file-upload entrypoints.

## Non-goals

- Do not widen the accepted shape of `DataContent`.
- Do not introduce a parallel compatibility lane that keeps returning raw `DecodeError` from the
  stable shared payload accessors.
- Do not refactor generated-file/materialization paths that are not modeled as shared input
  `DataContent`.

## Chosen design

### 1. Add a stable shared `InvalidDataContentError`

The shared prompt/types layer now owns a dedicated `InvalidDataContentError` struct that stores:

- a stable human-readable message
- the original `DataContent`
- the optional underlying `base64::DecodeError` as a source

This mirrors the upstream responsibility split more honestly than exposing the codec error
directly on stable public helpers.

### 2. Move shared payload accessors onto the new error type

The shared data-content decoding entrypoints now return `InvalidDataContentError`:

- `DataContent::as_bytes()`
- `AudioInputData::as_bytes()`
- `SttRequest::audio_bytes()`
- `AudioTranslationRequest::audio_bytes()`
- `ImageEditFileData::as_bytes()`
- `VideoGenerationFileData::as_bytes()`

This keeps the public surface at the semantic layer that users actually care about: “the supplied
shared payload is invalid”, not “the specific base64 backend failed”.

### 3. Reuse the same semantic lane in file-upload validation

`files::upload(...)` still returns `LlmError`, because that helper already owns a broader runtime
error contract. But invalid base64 file payloads now flow through `DataContent::as_bytes()` first,
so the public error message comes from the same shared `InvalidDataContentError` semantics instead
of a one-off upload-specific string.

## Follow-up

If more public helper families start accepting `DataContent` directly, they should route decoding
through this same shared error instead of reintroducing raw codec errors or helper-specific
messages.
