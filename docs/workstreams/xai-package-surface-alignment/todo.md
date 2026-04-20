# xAI Package Surface Alignment - TODO

Last updated: 2026-04-20

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

## Track C - Video/runtime and public providerOptions parity

- [x] Audit the upstream `xai-video-options.ts` surface against the current Rust provider-owned
  video path.
- [x] Add `mode` to `XaiVideoOptions`.
- [x] Add `referenceImageUrls` to `XaiVideoOptions`.
- [x] Serialize the public xAI chat/responses/search option structs in the audited AI SDK-facing
  camelCase shape while still accepting snake_case compatibility aliases.
- [x] Serialize the public xAI video/files option structs in the audited AI SDK-facing camelCase
  shape while still accepting snake_case compatibility aliases.
- [x] Route `mode = "extend-video"` to `/videos/extensions` on the provider-owned runtime.
- [x] Route `mode = "reference-to-video"` through the generation path while lowering
  `referenceImageUrls` onto `reference_images`.
- [x] Lock the new xAI video mode behavior with provider-local and top-level no-network tests.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/xai-package-surface-alignment/` folder.
- [x] Record the public type and upload-helper parity changes in `CHANGELOG.md` `Unreleased`.
- [x] Fold future xAI package-surface export audits into this workstream instead of scattering them
  across generic alignment notes.

## Track E - Intentional deferrals

- [-] Do not mirror TypeScript-only `createXai`, `xai`, `XaiProvider`, or `XaiProviderSettings`
  until a broader Rust facade pattern exists for those exports.
- [-] Do not widen the provider-owned xAI wrapper to expose embedding or rerank just because the
  shared OpenAI-compatible runtime supports those families for other vendors.
