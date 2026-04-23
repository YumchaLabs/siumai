# Anthropic Files Shared Contract Alignment - Design

Last updated: 2026-04-21

## Context

Siumai already had the main AI SDK-style high-level helper for file uploads:

- `siumai::files::upload(...)`
- shared `FileUploadRequest` / `FileObject`
- a generic `FileManagementCapability` bridge used by the current file-capable providers

However, Anthropic still drifted underneath that helper:

- `AnthropicFiles` kept provider-local wrapper/result types instead of reusing the shared
  file-management structs directly
- the public provider resource still centered older method names rather than the shared
  `upload_file(...)` / `list_files(...)` style surface
- `siumai::files` still needed an Anthropic-only upload special case instead of using the same
  generic `FileManagementCapability` adapter as the other aligned providers

That left Anthropic files structurally farther from the audited AI SDK references than necessary:

- `repo-ref/ai/packages/provider/src/files/v4/files-v4.ts`
- `repo-ref/ai/packages/anthropic/src/anthropic-files.ts`

## Goal

- Make Anthropic file upload/management reuse the shared file-management contract end-to-end.
- Remove redundant provider-local wrapper types that no longer encode a meaningful boundary.
- Keep provider-owned Anthropic list/retrieve/delete/content semantics intact.

## Non-goals

- Do not pretend AI SDK exposes provider-agnostic list/retrieve/delete/content helpers; those stay
  provider-owned on `AnthropicFiles`.
- Do not remove Anthropic beta-header handling.
- Do not remove current compatibility warnings for unsupported upload extras on the Anthropic beta
  file path.

## Chosen design

### 1. Reuse shared file-management structs directly on the provider-owned resource

`AnthropicFiles` now centers the shared contract instead of parallel Anthropic-only wrappers:

- `upload_file(FileUploadRequest) -> FileObject`
- `list_files(Option<FileListQuery>) -> FileListResponse`
- `retrieve_file(String) -> FileObject`
- `delete_file(String) -> FileDeleteResponse`
- `get_file_content(String) -> Vec<u8>`

This keeps Anthropic file management provider-owned while aligning its stable call/result shape
with the already shared helper lane.

### 2. Remove the Anthropic-only upload bridge from the high-level helper

`AnthropicFiles` and `AnthropicClient` now implement `FileManagementCapability`, so
`siumai::files::upload(...)` can reuse the same generic `upload_via_file_management(...)` adapter
path that other aligned providers already use.

The previous Anthropic-only upload special case in `siumai/src/files.rs` is removed.

### 3. Keep provider-specific semantics where they are actually real

The refactor does not flatten real Anthropic behavior:

- `anthropic-beta: files-api-2025-04-14` handling remains provider-owned
- missing upload purpose still resolves through the Anthropic-specific shared-helper rule
- the high-level helper still emits compatibility warnings when Anthropic currently ignores
  `metadata`, `providerOptions`, or non-header `httpConfig` overrides
- provider metadata still preserves Anthropic-owned fields such as `filename`, `mimeType`,
  `sizeBytes`, `createdAt`, and `downloadable`

### 4. Delete redundant provider-local wrappers instead of keeping compatibility shells

The removed provider-local wrapper layer included:

- provider-local upload/list/delete response exports
- the dedicated Anthropic-only helper bridge in `siumai::files`

The provider-owned resource remains, but the extra type shell around the shared contract is gone.

## Validation

This slice is locked by:

- `siumai/tests/anthropic_files_upload_alignment_test.rs`
- `siumai/tests/public_surface_imports_test.rs`
- `cargo check -p siumai-provider-anthropic -p siumai --features anthropic`
- `cargo check --workspace`

## Follow-up

- Re-check other provider-owned non-chat resources for the same smell: provider-specific execution
  with redundant provider-local request/result wrappers around an already shared stable contract.
