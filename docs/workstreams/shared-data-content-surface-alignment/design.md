# Shared Data Content Surface Alignment - Design

Last updated: 2026-04-21

## Context

Upstream `repo-ref/ai` does not keep binary payload handling trapped inside the prompt package.
The shared `DataContent` contract is reused across multiple higher-level AI SDK entrypoints, for
example:

- prompt/message content parts
- `transcribe(...)`
- `generateImage(...)`
- `uploadFile(...)`

Siumai had already exposed a stable prompt-side `DataContent`, but adjacent public Rust surfaces
still required family-specific payload enums:

- `AudioInputData`
- `ImageEditFileData`
- `VideoGenerationFileData`
- `files::UploadFileData`

That shape was not wrong from a Rust ergonomics perspective, but it made cross-family parity
audits harder and forced callers to manually re-wrap the same binary/base64 payload depending on
which helper they were about to call.

## Goal

- Keep one shared public `DataContent` carrier meaningful beyond prompt-only APIs.
- Let audio/image/video/file-upload helpers accept shared `DataContent` without direct enum-field
  mutation or manual re-wrapping.
- Preserve existing Rust-first request/input types where they still encode family-specific
  semantics such as URL support or media-type attachment.

## Non-goals

- Do not immediately collapse every family-specific payload enum into a hard type alias of
  `DataContent`.
- Do not silently widen `uploadFile` into accepting URL payloads through `DataContent`; that
  remains an intentional boundary.
- Do not remove existing `from_audio(...)`, `file(...)`, or `base64(...)` constructors in this
  slice.

## Chosen design

### 1. Treat `DataContent` as the canonical shared binary/base64 payload carrier

The stable Rust facade now uses `DataContent` as the interoperability pivot between prompt-owned
message content and adjacent helper families.

This mirrors the upstream AI SDK intent more closely than leaving `DataContent` isolated in prompt
types while nearby APIs each require their own bespoke payload conversion step.

### 2. Keep family-specific wrapper types, but add first-class conversions

Instead of forcing a large breaking collapse of public enums, the stable surface now adds
conversion bridges:

- `From<DataContent> for AudioInputData`
- `From<DataContent> for ImageEditFileData`
- `From<DataContent> for VideoGenerationFileData`
- `From<DataContent> for files::UploadFileData`

Reverse conversions are also exposed for the audio/image/video file-data wrappers so downstream
code can move back into the shared carrier when composing prompt/history payloads.

### 3. Add explicit shared-data constructors on high-level request/input helpers

The most user-visible gap was not the absence of enum conversions by itself, but the lack of a
direct ergonomic lane on stable public helper types.

That is now addressed with focused constructors:

- `SttRequest::from_data_content(...)`
- `AudioTranslationRequest::from_data_content(...)`
- `ImageEditInput::from_data_content(...)`
- `VideoGenerationInput::from_data_content(...)`
- `UploadFileData::from_data_content(...)`

These helpers keep the existing Rust-first family types while making the shared `DataContent`
story auditable and practical.

## Follow-up

If future parity work shows that more public families are effectively using the same binary/base64
carrier with no meaningful semantic distinction, we can consider a deeper refactor that centralizes
those wrapper enums onto a single owned type. This slice intentionally stops short of that larger
breaking merge.
