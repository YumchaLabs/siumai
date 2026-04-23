# Upload File Wrapper Removal Alignment - Design

Last updated: 2026-04-21

## Context

After the earlier call-boundary slice, `files::upload(...)` already accepted shared `DataContent`
directly and no longer required callers to first materialize `UploadFileData`.

At that point the remaining wrapper had become mostly dead compatibility surface:

- repo-internal usage was limited to tests and façade exports
- the helper already accepted `DataContent` and raw bytes directly
- the wrapper no longer carried a distinct semantic branch such as explicit URL uploads

Keeping it around would mainly preserve an extra public name that did not exist upstream and no
longer represented a necessary Rust-specific boundary.

## Goal

- Remove `UploadFileData` from the stable public Rust surface.
- Keep `files::upload(...)` ergonomic for shared `DataContent`, raw bytes, and string carriers.
- Preserve runtime URL-string rejection semantics.

## Non-goals

- Do not widen uploads to support remote URLs.
- Do not regress direct raw-bytes ergonomics.
- Do not change upload result/options/provider payload contracts.

## Chosen design

### 1. Delete the compatibility wrapper

`UploadFileData` is removed entirely from the helper boundary and façade exports.

This makes the public Rust upload helper more honest relative to the upstream AI SDK:

- shared binary/base64 carrier: `DataContent`
- direct byte convenience: `Vec<u8>` / `&[u8]` via `Into<DataContent>`
- string carrier: `String` / `&str` via `Into<DataContent>`

### 2. Add ergonomic `DataContent` conversions instead of a dedicated upload-only type

To keep the call surface practical after removing the wrapper, `DataContent` now supports focused
Rust conversions for the carriers that matter here:

- `Vec<u8> -> DataContent`
- `&[u8] -> DataContent`
- `String -> DataContent`
- `&str -> DataContent`

That keeps the upload helper direct without introducing a second upload-specific input abstraction.

### 3. Keep URL-string rejection as runtime behavior

URL-like strings are still detected and rejected at runtime on the upload helper path, matching the
upstream AI SDK behavior. Removing `UploadFileData` does not mean URLs became valid upload inputs.

## Follow-up

If future parity work shows that other helper families still own compatibility wrappers that no
longer encode a meaningful boundary beyond shared `DataContent`, the same removal pattern can be
applied there as well.
