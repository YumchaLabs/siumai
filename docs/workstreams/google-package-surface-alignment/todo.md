# Google Package Surface Alignment - TODO

Last updated: 2026-04-20

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Public type parity

- [x] Audit `repo-ref/ai/packages/google/src/index.ts`.
- [x] Add the missing public `GoogleEmbeddingModelOptions` type on the provider-owned/public
  surface.
- [x] Add the missing public `GoogleVideoModelOptions` and `GoogleVideoModelId` aliases on the
  provider-owned/public surface.
- [x] Add the missing public `GoogleFilesUploadOptions` type on the provider-owned/public surface.
- [x] Add the missing public `GoogleProviderMetadata` and `GoogleErrorData` names on the
  provider-owned/public surface.
- [x] Preserve the upstream deprecated Google Generative AI alias names where the audited package
  still exports them.

## Track B - Runtime lowering parity

- [x] Add Google-branded request helpers for chat, embedding, image, and video requests.
- [x] Add the Google-branded upload helper lane on `UploadFileOptions`.
- [x] Lower `GoogleLanguageModelOptions.serviceTier` onto the provider-owned chat runtime.
- [x] Lower `GoogleLanguageModelOptions.streamFunctionCallArguments` onto the provider-owned chat
  runtime.
- [x] Lower `GoogleEmbeddingModelOptions.outputDimensionality` and `taskType` onto the provider-
  owned embedding runtime.
- [x] Lower positional multimodal `GoogleEmbeddingModelOptions.content[]` onto embedding requests.
- [x] Lower `GoogleVideoModelOptions.negativePrompt`, `personGeneration`, and `referenceImages`
  onto the provider-owned video runtime.
- [x] Preserve `null` entries inside Google embedding `content[]` during provider-option
  normalization.
- [x] Keep `pollIntervalMs` / `pollTimeoutMs` public but deferred honestly on the current
  task-based video helper path.

## Track C - Metadata parity

- [x] Surface `usageMetadata`, `finishMessage`, and `serviceTier` on the provider-owned/public
  Google metadata contract.
- [x] Tighten `GoogleProviderMetadata.promptFeedback` from raw JSON to typed `PromptFeedback`.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/google-package-surface-alignment/` folder.
- [x] Record the Google package-surface alignment slice in `CHANGELOG.md` `Unreleased`.
- [ ] Fold future Google package-surface audits into this workstream instead of scattering them
  across generic refactor notes.

## Track E - Intentional deferrals

- [-] Do not fabricate TypeScript-only `createGoogle`, `google`, `GoogleProvider`, or
  `GoogleProviderSettings` as callable Rust package exports.
- [-] Do not pretend `pollIntervalMs` / `pollTimeoutMs` are runtime-complete on the current
  task-based video helper before the provider-owned video story actually owns polling.
