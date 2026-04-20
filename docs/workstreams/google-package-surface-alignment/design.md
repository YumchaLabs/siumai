# Google Package Surface Alignment - Design

Last updated: 2026-04-20

## Problem

Compared with `repo-ref/ai/packages/google/src/index.ts`, Siumai had already aligned the main
Gemini runtime story, but the package boundary was still incomplete in a few important places:

- upstream `@ai-sdk/google` exports a Google-branded typed surface for language, embedding, video,
  file upload, provider metadata, and error data
- Siumai still exposed most of that package through Gemini-native names only, which made
  package-surface diffing against `repo-ref/ai` noisy
- some newly exported Google request options were still compile-surface-only or partially lowered
  instead of being carried all the way into the provider-owned runtime

The resulting drift was not a single runtime bug. It was a package-shape and data-structure gap:

- callers could not rely on a Google-branded typed surface comparable to the upstream package
- public review against `repo-ref/ai/packages/google/src/*` required too much manual alias
  translation
- some Google-specific runtime knobs existed in typed form but were not fully exercised on the
  provider-owned path

## Goals

- Audit `repo-ref/ai/packages/google/src/index.ts` as a package boundary, not just as a Gemini
  runtime implementation.
- Expose the main audited Google typed surface on the provider-owned/public Rust facade:
  `GoogleLanguageModelOptions`, `GoogleEmbeddingModelOptions`, `GoogleVideoModelOptions`,
  `GoogleVideoModelId`, `GoogleFilesUploadOptions`, `GoogleProviderMetadata`, and
  `GoogleErrorData`, plus the upstream deprecated aliases where they still exist.
- Make the Google-branded request/helper lane work end to end instead of stopping at compile-time
  aliases.
- Keep the Rust package story honest about the parts that still differ from the TypeScript package.

## Non-goals

- Do not fabricate JavaScript-style callable provider exports such as `createGoogle`, `google`,
  `GoogleProvider`, or `GoogleProviderSettings` on the Rust facade.
- Do not widen the current task-based Rust video helper into a fake AI SDK-style auto-polling model
  surface unless the provider-owned runtime actually owns that lifecycle.
- Do not replace the existing Gemini-native names; the Google surface should layer on top of the
  provider-owned Gemini implementation rather than rename the whole crate.

## Chosen design

### 1. Expose the audited Google package names directly

The provider-owned/public facade now carries the main Google-branded typed surface expected from
the audited AI SDK package:

- `GoogleLanguageModelOptions` plus deprecated `GoogleGenerativeAIProviderOptions`
- `GoogleEmbeddingModelOptions` plus deprecated
  `GoogleGenerativeAIEmbeddingProviderOptions`
- `GoogleVideoModelOptions` plus deprecated `GoogleGenerativeAIVideoProviderOptions`
- `GoogleVideoModelId` plus deprecated `GoogleGenerativeAIVideoModelId`
- `GoogleFilesUploadOptions`
- `GoogleProviderMetadata` plus deprecated `GoogleGenerativeAIProviderMetadata`
- `GoogleErrorData`

This keeps `provider_ext::google::{options::*, metadata::*, *}` much easier to compare one-to-one
against `repo-ref/ai/packages/google/src/index.ts`.

### 2. Carry Google-branded request helpers through the real provider-owned runtime

The audited Google helper lane now exists on the stable Rust surface:

- `GoogleChatRequestExt::with_google_options(...)`
- `GoogleEmbeddingRequestExt::with_google_embedding_options(...)`
- `GoogleImageRequestExt::with_google_image_options(...)`
- `GoogleVideoRequestExt::with_google_video_options(...)`
- `UploadFileOptions::with_google_upload_options(...)`

Those helpers are no longer facade-only. They now lower onto the provider-owned Gemini runtime for
the audited field subset:

- chat: `serviceTier`, `streamFunctionCallArguments`
- embedding: `outputDimensionality`, `taskType`, multimodal `content`
- video: `negativePrompt`, `personGeneration`, `referenceImages`
- files: `displayName`, polling controls

### 3. Keep runtime behavior explicit where Rust still differs

The current Rust video helper remains task-based rather than AI SDK model-owned polling. Therefore:

- `GoogleVideoModelOptions.pollIntervalMs` / `pollTimeoutMs` are public and preserved
- those fields are not forwarded to the provider API body
- the current Rust task path reports them as warnings instead of pretending the provider-owned
  video runtime already consumes them internally

This keeps the public package surface easy to audit without overstating runtime convergence.

### 4. Tighten metadata and request normalization instead of leaving raw blobs

The Google/Gemini provider-owned metadata surface now carries the higher-value typed fields from the
audited package and runtime:

- `usageMetadata`
- `finishMessage`
- `serviceTier`
- typed `PromptFeedback` on `GoogleProviderMetadata.promptFeedback`

The lower request-normalization layer also now preserves `null` entries inside Google embedding
`content[]`, which is required for positional multimodal alignment and matches the upstream package
contract more closely.

## Current implemented parity in this workstream

This workstream currently closes the following Google package-surface gaps:

- `provider_ext::google::{options::*, metadata::*, *}` now exposes the audited Google typed option,
  metadata, video-model-id, and error-data names
- Google-branded request helpers now exist for chat, embedding, image, video, and file upload
- chat runtime lowering now consumes `serviceTier` and `streamFunctionCallArguments`
- embedding runtime lowering now consumes `outputDimensionality`, `taskType`, and positional
  multimodal `content`
- video runtime lowering now consumes `negativePrompt`, `personGeneration`, and `referenceImages`
  while keeping polling knobs deferred honestly on the current task-based path
- response metadata now carries `usageMetadata`, `finishMessage`, `serviceTier`, and typed
  `PromptFeedback`
- Google embedding option normalization now preserves `null` array entries instead of collapsing
  text-only positions

## Validation

The current slice is locked by:

- provider-option tests in `siumai-provider-gemini/src/provider_options/gemini/mod.rs`
- provider-local request-ext tests in `siumai-provider-gemini/src/providers/gemini/ext/*.rs`
- protocol-level Gemini transformer tests in
  `siumai-protocol-gemini/src/standards/gemini/transformers/tests.rs`
- public surface compile guards in `siumai/tests/public_surface_imports_test.rs`
- top-level public-path parity coverage in `siumai/tests/provider_public_path_parity_test.rs`

## Remaining follow-up

- Re-audit whether the upstream Google package adds more public names that deserve direct Rust
  mirrors.
- Re-evaluate whether the Rust facade should grow `Provider::google()` / builder aliases as a
  package-level naming convenience, without pretending to be a callable TypeScript provider.
- If Siumai later grows a provider-owned video model that internally polls to completion, revisit
  whether `pollIntervalMs` / `pollTimeoutMs` should become fully runtime-consumed rather than
  warning-only on the current task-based helper.
