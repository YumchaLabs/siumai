# Google Package Surface Alignment - Design

Last updated: 2026-04-22

## Problem

Compared with `repo-ref/ai/packages/google/src/index.ts`, Siumai had already aligned the main
Gemini runtime story, but the package boundary was still incomplete in a few important places:

- upstream `@ai-sdk/google` exports a Google-branded typed surface for language, embedding, video,
  file upload, provider metadata, provider settings, package-level helper aliases, and error data
- Siumai still exposed most of that package through Gemini-native names only, which made
  package-surface diffing against `repo-ref/ai` noisy
- some newly exported Google request options were still compile-surface-only or partially lowered
  instead of being carried all the way into the provider-owned runtime

The resulting drift was not a single runtime bug. It was a package-shape and data-structure gap:

- callers could not rely on a Google-branded typed surface comparable to the upstream package
- public review against `repo-ref/ai/packages/google/src/*` required too much manual alias
  translation
- upstream `GoogleProvider` member names such as `languageModel`, `embeddingModel`, `imageModel`,
  and `videoModel` did not have a clear Rust-side analogue on `Provider::google()`
- the public facade still lacked a Google-first grouped model-id surface that could be compared
  directly against the upstream `GoogleModelId` / `GoogleEmbeddingModelId` /
  `GoogleImageModelId` / `GoogleVideoModelId` contracts
- some Google-specific runtime knobs existed in typed form but were not fully exercised on the
  provider-owned path

## Goals

- Audit `repo-ref/ai/packages/google/src/index.ts` as a package boundary, not just as a Gemini
  runtime implementation.
- Expose the main audited Google typed surface on the provider-owned/public Rust facade:
  `GoogleLanguageModelOptions`, `GoogleEmbeddingModelOptions`, `GoogleVideoModelOptions`,
  `GoogleVideoModelId`, `GoogleFilesUploadOptions`, `GoogleProviderMetadata`,
  `GoogleProviderSettings`, and `GoogleErrorData`, plus the upstream deprecated aliases where
  they still exist.
- Mirror the honest package-level Google entry helpers and constants where Rust has direct
  equivalents, such as `google`, `create_google`, deprecated `create_google_generative_ai`, and
  `VERSION`.
- Mirror the non-callable Google provider member names where Rust already has an honest builder
  analogue.
- Expose a Google-first grouped model-id surface on `provider_ext::google::{chat, embedding, image,
  video, model_sets}` so public diffing against the audited package no longer depends on internal
  Gemini-only module names.
- Make the Google-branded request/helper lane work end to end instead of stopping at compile-time
  aliases.
- Keep the Rust package story honest about the parts that still differ from the TypeScript package.

## Non-goals

- Do not fabricate a JavaScript-style callable `GoogleProvider` interface on the Rust facade.
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
- `GoogleProviderSettings` plus deprecated `GoogleGenerativeAIProviderSettings`
- `GoogleErrorData`

This keeps `provider_ext::google::{options::*, metadata::*, *}` much easier to compare one-to-one
against `repo-ref/ai/packages/google/src/index.ts`, while still mapping provider construction onto
the existing Rust `GeminiBuilder` / `GeminiConfig` implementation instead of fabricating a callable
provider type.

The provider-settings mirror is now a dedicated Rust input struct rather than a bare config alias:

- `GoogleProviderSettings::new()`
- `with_api_key(...)`
- `with_base_url(...)`
- `with_headers(...)` / `with_header(...)`
- `with_fetch(...)`
- `with_generate_id(...)`
- `with_name(...)`
- `into_builder()`
- `into_builder_for_model(...)`
- `into_config_for_model(...)`

This is closer in spirit to the upstream `createGoogle(options)` boundary because the settings type
no longer pretends model selection is part of provider construction.

### 1b. Mirror package-level helper aliases only where Rust already has a real analogue

The audited package root also exports `google`, `createGoogle`, deprecated
`createGoogleGenerativeAI`, and `VERSION`.

On the Rust facade we mirror the parts that are structurally honest:

- `provider_ext::google::google()`
- `provider_ext::google::create_google()`
- deprecated `provider_ext::google::create_google_generative_ai()`
- `provider_ext::google::VERSION`

We still do not fabricate the TypeScript `GoogleProvider` callable object just to make the root
module look more symmetrical.

### 1c. Mirror non-callable provider members and grouped model ids

The audited `GoogleProvider` interface also exposes non-callable family helpers such as
`languageModel`, `chat`, `embedding`, `embeddingModel`, `image`, `imageModel`, `video`,
`videoModel`, and `files`.

Rust still does not expose the callable provider object itself, but it now mirrors the honest part
of that shape in two places:

- `Provider::google()` / `Provider::gemini()` now expose builder helpers:
  `language_model`, `chat`, deprecated `generative_ai`, `embedding`, `embedding_model`,
  deprecated `text_embedding`, deprecated `text_embedding_model`, `image`, `image_model`, `video`,
  `video_model`, and `files`
- `provider_ext::google` / `provider_ext::gemini` now expose grouped model ids as
  `chat`, `embedding`, `image`, `video`, and `model_sets`

This keeps the Rust facade close to the audited package without pretending a builder is the same
thing as a callable TypeScript provider object.

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

### 2b. Lower `generateId` onto the real Gemini ID-ownership points

The audited package-level `generateId` hook is now supported through the Rust provider-owned
construction and protocol layers instead of remaining a deferred facade-only difference.

- `GoogleProviderSettings::with_generate_id(...)`
- `GeminiBuilder::with_generate_id(...)`
- public `SharedIdGenerator` on the Google/Gemini facade

The configured generator is carried into `GeminiConfig` and consumed where Siumai actually owns
stable ID allocation:

- non-stream Gemini response transformation for `functionCall`, `executableCode`, and
  `codeExecutionResult`
- normalized grounding source ids on both response and streaming paths
- streaming `functionCall`, `functionResponse`, and provider-executed code-execution ids

This keeps parity honest: Rust still does not fabricate a callable TypeScript provider object, but
it now supports the upstream hook at the concrete runtime points where Siumai is responsible for
those ids.

### 2c. Lower `name` onto provider-owned display surfaces without changing canonical identity

The audited package-level `name` hook is now supported as a provider-facing display label rather
than being left deferred.

The label is carried through:

- `GoogleProviderSettings.name` / `with_name(...)`
- `GeminiBuilder::name(...)` / `with_name(...)`
- `GeminiConfig::with_provider_name(...)`
- `GeminiClient::provider_name()`
- `GeminiFiles::provider_name()`

The default behavior is intentionally split by construction entry:

- `Provider::google()` / `create_google()` default to `google.generative-ai`
- `Provider::gemini()` keeps the historical native label `gemini`

This keeps the Rust mapping honest with the upstream package intent while preserving Siumai's
canonical provider identity rules. `name` does **not** change:

- `LlmClient::provider_id()`
- `ModelMetadata::provider_id()`
- `providerReference`
- `providerMetadata` root keys
- registry/provider factory ids

The label is only used on provider-owned display/accessor surfaces and provider-facing error text.

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
  metadata, provider-settings, video-model-id, and error-data names
- `GoogleProviderSettings` is now a dedicated provider-level input struct instead of a
  `GeminiConfig` alias, and the deprecated `GoogleGenerativeAIProviderSettings` alias now points to
  that same provider-level settings surface
- `provider_ext::google` now also mirrors the audited package-level `VERSION` constant and the
  deprecated `createGoogleGenerativeAI` entry helper as `create_google_generative_ai()`
- `Provider::google()` / `Provider::gemini()` now mirror the audited non-callable family helper
  names for chat, embedding, image, and video model selection
- `Provider::google()` / `Provider::gemini()` now also mirror the audited `files()` member through
  the provider-owned `GeminiFiles` capability builder
- `provider_ext::google` / `provider_ext::gemini` now expose grouped model ids as
  `chat`, `embedding`, `image`, `video`, and `model_sets`
- `GoogleProviderSettings` / `GeminiBuilder` now support `generateId` through
  `with_generate_id(...)`, with a shared public `SharedIdGenerator`
- `GoogleProviderSettings` / `Provider::google()` now also support the audited `name` hook as a
  provider-facing display label, surfaced via `GeminiClient::provider_name()` and
  `GeminiFiles::provider_name()` without changing canonical provider ids or metadata namespaces
- Gemini response and streaming transformers now consume that generator for provider-owned
  tool-call, tool-result, and source ids instead of hard-coded local id allocation
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
- public surface compile guards in `siumai/tests/public_surface_imports_test.rs`, including the
  grouped model ids, builder family helpers, and the direct `files()` member
- top-level public-path parity coverage in `siumai/tests/provider_public_path_parity_test.rs`,
  including the direct `Provider::google().files()` list-files path

## Remaining follow-up

- Re-audit whether the upstream Google package adds more public names that deserve direct Rust
  mirrors.
- Keep the Rust package story honest: `GoogleProviderSettings` is now a provider-level settings
  struct that converts into `GeminiBuilder` / `GeminiConfig`, but `GoogleProvider` should remain
  intentionally unmirrored until there is a Rust-native concept that is structurally equivalent to
  the TypeScript callable provider.
- If Siumai later grows a provider-owned video model that internally polls to completion, revisit
  whether `pollIntervalMs` / `pollTimeoutMs` should become fully runtime-consumed rather than
  warning-only on the current task-based helper.
