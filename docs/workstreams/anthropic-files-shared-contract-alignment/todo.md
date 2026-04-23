# Anthropic Files Shared Contract Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Contract audit

- [x] Verify the audited AI SDK Anthropic files surface is still a provider-owned `files()` helper
  backed by the shared `FilesV4` upload contract rather than a bespoke public upload result shape.
- [x] Confirm the local Anthropic file wrappers are compatibility shells, not a distinct stable
  runtime contract.

## Track B - Provider resource convergence

- [x] Make `AnthropicFiles` consume shared `FileUploadRequest` and return shared `FileObject`.
- [x] Make `AnthropicFiles` list/retrieve/delete use shared `FileListResponse`,
  `FileDeleteResponse`, and `FileObject`.
- [x] Implement shared `FileManagementCapability` for `AnthropicFiles`.
- [x] Implement shared `FileManagementCapability` for `AnthropicClient`.
- [x] Remove the Anthropic-only upload special case from `siumai::files`.

## Track C - Behavior preservation

- [x] Preserve Anthropic beta-header handling for the files API.
- [x] Preserve Anthropic compatibility warnings for ignored upload metadata/provider options.
- [x] Preserve provider-owned `providerReference` / `providerMetadata` shaping on the high-level
  helper path.

## Track D - Docs and verification

- [x] Add a dedicated workstream for the Anthropic files shared-contract slice.
- [x] Update `CHANGELOG.md` `Unreleased`.
- [x] Update the structural-alignment matrix/todo/milestones notes for the file-upload row.
- [x] Run focused Anthropic `nextest` coverage for the upload/public-surface path.
- [x] Run default-feature upload/public-surface regression coverage.
- [x] Run `cargo check -p siumai-provider-anthropic -p siumai --features anthropic`.
- [x] Run `cargo check --workspace`.
