# Upload File Input Shape Alignment - Design

Last updated: 2026-04-21

## Context

Siumai already had a stable high-level `files::upload(...)` helper aligned in broad behavior with
AI SDK `uploadFile()`, but one public type detail still overstated the supported shape:

- local Rust surface: `UploadFileData::{Bytes, Base64, Url}`
- upstream AI SDK surface: `data: DataContent`

Upstream does not expose a dedicated URL input branch for file uploads. Instead, string content is
normalized first, URL-like strings are detected internally, and upload then rejects them at
runtime with a URL-not-supported error.

Keeping an explicit `UploadFileData::Url` variant made the Rust surface look wider and more
compile-time sanctioned than the upstream contract it is supposed to mirror.

## Goal

- Remove the explicit upload-only URL variant from the stable Rust helper surface.
- Preserve upstream runtime behavior: URL-like string inputs are still rejected with the same
  high-level message.
- Keep bytes/base64 upload ergonomics intact.

## Non-goals

- Do not silently accept remote URLs for uploads.
- Do not collapse `UploadFileData` all the way into a hard alias of `DataContent` in this slice.
- Do not change the rest of the upload helper result/options contract.

## Chosen design

### 1. Remove the explicit `UploadFileData::Url` variant

The stable Rust upload input shape now only models actual upload payload carriers:

- raw bytes
- base64 string content

This is closer to the upstream AI SDK contract than a dedicated `Url` enum branch.

### 2. Keep URL rejection as a runtime semantic check on string content

For `UploadFileData::Base64(String)`, the helper now first checks whether the string parses as a
URL. If it does, upload fails with the same URL-not-supported message used before.

That mirrors the upstream behavior much more closely:

- callers can still accidentally pass a URL-like string
- the helper detects that and rejects it at runtime
- the public type shape no longer implies that URL uploads are a supported input form

## Follow-up

If later parity work decides the dedicated Rust `UploadFileData` wrapper is no longer pulling its
weight, the next step would be to evaluate whether `files::upload(...)` should move closer still to
the shared `DataContent` carrier plus runtime string-URL detection.
