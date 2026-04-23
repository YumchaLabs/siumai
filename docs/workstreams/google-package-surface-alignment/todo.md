# Google Package Surface Alignment - TODO

Last updated: 2026-04-22

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
- [x] Add the missing public `GoogleProviderSettings` provider-level input struct on the
  provider-owned/public surface.
- [x] Mirror the audited package-level `VERSION` constant and deprecated
  `createGoogleGenerativeAI` helper where Rust has honest equivalents.
- [x] Mirror the audited non-callable `GoogleProvider` family helper names where Rust has honest
  builder analogues.
- [x] Mirror the audited `files()` provider member where Rust already has the provider-owned
  `GeminiFiles` capability.
- [x] Expose grouped Google model-id constants on the public facade as `chat`, `embedding`,
  `image`, `video`, and `model_sets`.
- [x] Preserve the upstream deprecated Google Generative AI alias names where the audited package
  still exports them.
- [x] Replace the old `GeminiConfig`-shaped `GoogleProviderSettings` alias with builder-oriented
  settings helpers (`with_api_key`, `with_base_url`, `with_headers`, `with_fetch`,
  `with_generate_id`,
  `into_builder*`, `into_config_for_model`).
- [x] Lower the audited package-level `name` hook onto a provider-facing display-label surface
  without changing canonical provider ids.

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
- [x] Lower `GoogleProviderSettings.generateId` onto provider-owned Gemini response/streaming
  stable ids for tool calls, tool results, and sources.
- [x] Lower `GoogleProviderSettings.name` onto provider-owned display/accessor surfaces
  (`GeminiClient::provider_name()` / `GeminiFiles::provider_name()`) and provider-facing error
  text, while keeping `provider_id` / `providerReference` / `providerMetadata` canonical.
- [x] Keep `pollIntervalMs` / `pollTimeoutMs` public but deferred honestly on the current
  task-based video helper path.

## Track C - Metadata parity

- [x] Surface `usageMetadata`, `finishMessage`, and `serviceTier` on the provider-owned/public
  Google metadata contract.
- [x] Tighten `GoogleProviderMetadata.promptFeedback` from raw JSON to typed `PromptFeedback`.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/google-package-surface-alignment/` folder.
- [x] Add a dedicated root-export/data-structure matrix for the audited package boundary.
- [x] Add milestone tracking so intentional Google package-surface deferrals remain explicit.
- [x] Record the Google package-surface alignment slice in `CHANGELOG.md` `Unreleased`.
- [ ] Fold future Google package-surface audits into this workstream instead of scattering them
  across generic refactor notes.

## Track E - Intentional deferrals

- [-] Do not fabricate a TypeScript-style callable `GoogleProvider` export on the Rust facade.
- [-] Keep `google()` / `create_google()` as Rust builder helpers and keep
  `GoogleProviderSettings` as a provider-level settings struct that converts into
  `GeminiBuilder` / `GeminiConfig`, not as evidence of a callable provider object.
- [-] Do not pretend `pollIntervalMs` / `pollTimeoutMs` are runtime-complete on the current
  task-based video helper before the provider-owned video story actually owns polling.
