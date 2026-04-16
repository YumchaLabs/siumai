# xAI Package Surface Alignment - TODO

Last updated: 2026-04-15

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Public type parity

- [x] Audit `repo-ref/ai/packages/xai/src/index.ts`.
- [x] Add the missing public `XaiFilesOptions` type on the provider-owned xAI surface.
- [x] Re-export `XaiFilesOptions` from `provider_ext::xai::{options::*, *}`.
- [x] Add the missing public `XaiErrorData` type on the provider-owned/public xAI surface.
- [x] Add the missing public `XaiVideoModelId` alias on the provider-owned/public xAI surface.

## Track B - Files/upload runtime parity

- [x] Audit the upstream xAI files helper shape against the current Rust upload helper.
- [x] Add a provider-owned xAI files implementation behind `XaiClient`.
- [x] Make `XaiClient` expose file management capability so `siumai::files::upload(...)` works on
  the provider-owned wrapper path.
- [x] Lower typed `XaiFilesOptions.teamId` onto the multipart `team_id` field.
- [x] Preserve provider-native `filePath -> file_path` upload lowering on the Rust xAI path.
- [x] Lock the multipart upload shape with no-network regression tests.

## Track C - Docs and changelog

- [x] Create a dedicated `docs/workstreams/xai-package-surface-alignment/` folder.
- [x] Record the public type and upload-helper parity changes in `CHANGELOG.md` `Unreleased`.
- [ ] Fold future xAI package-surface export audits into this workstream instead of scattering them
  across generic alignment notes.

## Track D - Intentional deferrals

- [-] Do not mirror TypeScript-only `createXai`, `xai`, `XaiProvider`, or `XaiProviderSettings`
  until a broader Rust facade pattern exists for those exports.
- [-] Do not widen the provider-owned xAI wrapper to expose embedding or rerank just because the
  shared OpenAI-compatible runtime supports those families for other vendors.
