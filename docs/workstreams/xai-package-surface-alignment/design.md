# xAI Package Surface Alignment - Design

Last updated: 2026-04-20

## Problem

Compared with `repo-ref/ai/packages/xai/src/index.ts`, Siumai had already aligned most of the
provider-owned xAI wrapper surface:

- chat and responses typed options
- image and video typed options
- provider-owned tool factories
- provider-owned config/client wrapper construction

What was still drifting was the package boundary around xAI-specific non-text data structures, the
file-upload lane, part of the video-options surface, and the public naming contract of
`providerOptions.xai` on chat/responses/search:

- `XaiFilesOptions` was exported upstream but missing on the Rust provider/facade surface
- `XaiErrorData` and `XaiVideoModelId` were exported upstream but had no comparable Rust names
- the high-level Rust `siumai::files::upload(...)` helper routed `xai` like an OpenAI-like file
  provider, but the native `XaiClient` wrapper itself did not expose a provider-owned files lane
- `XaiVideoModelOptions` existed, but it still trailed the upstream type/runtime contract for
  `mode`, `referenceImageUrls`, and the dedicated `/videos/extensions` route
- `XaiChatOptions`, `XaiResponsesOptions`, `XaiSearchParameters`, and discriminated search-source
  structs still serialized their public Rust shape mostly as snake_case even though the audited
  AI SDK package presents those options to callers in camelCase

That left the public Rust surface in an awkward middle state:

- the shared upload helper knew about `xai`
- the provider-owned wrapper did not actually implement file management
- the audited upstream package still had a couple of missing public data structures
- the provider-owned video path still only covered generation plus edit-by-`videoUrl`, not the
  fuller upstream video-options contract

## Goals

- Audit `repo-ref/ai/packages/xai/src/index.ts` as a package boundary, not only as a runtime tool
  or image/video implementation.
- Add the missing public xAI data structures that already exist upstream:
  `XaiFilesOptions`, `XaiErrorData`, and `XaiVideoModelId`.
- Make the Rust high-level upload helper work on the provider-owned `XaiClient` path instead of
  relying on a stale shared assumption.
- Close the remaining audited `XaiVideoModelOptions` gap around `mode`,
  `referenceImageUrls`, and route selection.
- Keep the Rust public surface aligned with the upstream package without inventing unrelated xAI
  exports.

## Non-goals

- Do not widen the provider-owned xAI wrapper to expose every capability that the shared
  OpenAI-compatible runtime may happen to support internally.
- Do not invent a public `XaiFiles` facade module just for symmetry; upstream `@ai-sdk/xai`
  exports file-upload options and provider construction, not a dedicated public files resource.
- Do not mirror TypeScript-only factory/settings exports such as `createXai`, `xai`,
  `XaiProvider`, or `XaiProviderSettings` unless a broader Rust package-facade pattern is chosen
  first.

## Chosen design

### 1. Match the audited upstream package names first

The public Rust xAI facade now also carries the missing upstream names:

- `XaiFilesOptions`
- `XaiErrorData`
- `XaiVideoModelId`

This keeps the audited package index easier to compare one-to-one against
`provider_ext::xai::{options::*, *}`.

### 2. Keep xAI capability scope provider-owned, not shared-compatible

The `XaiClient` wrapper still intentionally keeps its provider-owned capability surface narrower
than the underlying generic OpenAI-compatible client:

- chat / streaming
- speech
- image generation
- video generation
- file management

It still does **not** re-open provider-owned embedding or rerank just because the inner shared
runtime has those families for other vendors. That matches the upstream `@ai-sdk/xai` package
boundary more closely.

### 3. Treat file upload as a first-class provider-owned xAI lane

Siumai now has a provider-owned xAI files implementation backing the stable Rust upload helper:

- `XaiClient` now exposes file management capability
- `siumai::files::upload(...)` now works on `XaiClient`
- the multipart body lowers typed `XaiFilesOptions` onto xAI-native form fields such as
  `team_id`

The Rust upload lane also forwards `filePath -> file_path`. This is a deliberate provider-native
compatibility choice: the field is already part of the upstream exported type surface even though
the current AI SDK upload helper mainly exercises `teamId`.

### 4. Finish the upstream video-options contract on the provider-owned xAI path

The provider-owned xAI video surface now also closes the remaining upstream package/runtime drift:

- `XaiVideoOptions` now carries `mode` plus `referenceImageUrls`
- the public typed provider-option surface now serializes the audited AI SDK-facing camelCase
  fields across chat/responses/search/video/files, including `reasoningEffort`,
  `searchParameters`, `previousResponseId`, `pollIntervalMs`, `pollTimeoutMs`, `videoUrl`,
  `referenceImageUrls`, `teamId`, and `filePath`
- the provider-owned Rust runtime now routes `mode = "extend-video"` to `/videos/extensions`
- the provider-owned Rust runtime now routes `mode = "reference-to-video"` through the normal
  generation endpoint while lowering typed `referenceImageUrls` onto `reference_images`

The runtime still preserves Rust-side compatibility:

- legacy snake_case provider-option keys continue to deserialize
- `request.video` still works as a stable Rust input and defaults to edit-video when no explicit
  xAI mode is provided

### 5. Keep non-package surfaces deferred

The package audit stays strict about deferrals:

- no Rust mirror of TypeScript-only provider factory/settings exports yet
- no top-level public `resources::files` facade added for xAI
- no attempt to widen xAI into a generic OpenAI-compatible file package beyond the provider-owned
  wrapper and stable upload helper

## Current implemented parity in this workstream

This workstream currently closes the following xAI package-surface gaps:

- `siumai-provider-xai` now exposes `XaiFilesOptions`
- `provider_ext::xai::{options::*, *}` now re-exports `XaiFilesOptions`
- the provider-owned/public xAI surface now also exposes `XaiErrorData`
- the provider-owned/public xAI surface now also exposes `XaiVideoModelId`
- the provider-owned `XaiClient` now exposes file management and backs
  `siumai::files::upload(...)`
- the xAI upload lane is locked by no-network multipart regression tests on both the provider crate
  and top-level facade/helper path
- `XaiVideoOptions` now also covers upstream `mode` and `referenceImageUrls`
- the public xAI options surface now serializes the audited AI SDK-facing camelCase fields across
  chat / responses / search / video / files while still accepting the older snake_case
  compatibility aliases
- the provider-owned xAI video runtime now covers the audited `/videos/extensions` and
  reference-to-video route split

## Validation

The current slice is locked by:

- provider-option tests in `siumai-provider-xai/src/provider_options/xai.rs`
- provider-local files tests in `siumai-provider-xai/src/providers/xai/files.rs`
- provider-local xAI video route tests in `siumai-provider-xai/src/providers/xai/video.rs`
- public surface compile guards in `siumai/tests/public_surface_imports_test.rs`
- top-level upload-helper regression tests in
  `siumai/tests/xai_files_upload_alignment_test.rs`
- top-level xAI video public-path parity tests in
  `siumai/tests/provider_public_path_parity_test.rs`

## Remaining follow-up

- Re-audit whether the upstream xAI package adds more public names that deserve direct Rust facade
  mirrors.
- Re-evaluate whether TypeScript-only factory/settings exports should ever gain a Rust package-level
  analogue across providers, instead of deciding that ad hoc on xAI alone.
- Re-check whether the upstream AI SDK starts actively lowering `filePath` in its own upload helper,
  so the Rust/provider-native behavior and audited TypeScript runtime stay fully converged.
- Re-audit whether upstream `@ai-sdk/xai` adds more public typed names or provider-owned helpers
  that deserve direct Rust mirrors without widening the provider-owned capability surface
