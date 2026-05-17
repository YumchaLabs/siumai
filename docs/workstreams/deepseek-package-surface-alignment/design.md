# DeepSeek Package Surface Alignment - Design

Last updated: 2026-05-18

## Problem

The active AIPC provider inventory still marked DeepSeek as Amber even though the current codebase
already matches the audited `@ai-sdk/deepseek` package boundary more closely than that status
suggests.

The upstream package surface is intentionally narrow:

- `deepSeek(...)` / `createDeepSeek(...)`
- `languageModel(...)`
- `chat(...)`
- `DeepSeekProviderSettings`
- `DeepSeekLanguageModelChatOptions` and deprecated option aliases
- `DeepSeekErrorData`
- `VERSION`

It explicitly rejects embedding and image model creation. Siumai's provider-owned DeepSeek wrapper
is similarly chat-only at the stable package boundary, with reasoning options, provider metadata,
and stream usage behavior owned by the DeepSeek provider package.

## Target state

DeepSeek should be treated as a Green provider package-parity row under AIPC:

- `provider_ext::deepseek::deepseek()` is the Rust analogue of upstream `deepSeek`.
- `provider_ext::deepseek::create_deepseek()` is the Rust analogue of upstream
  `createDeepSeek`.
- `DeepSeekProviderSettings`, `DeepSeekErrorData`, `VERSION`, typed options, typed metadata
  helpers, and curated chat model constants remain public and covered by facade compile guards.
- Registry/factory capability declarations expose chat/streaming/tools/vision/thinking only.
- Completion, embedding, image, speech, transcription, audio, and rerank remain unsupported on the
  provider-owned DeepSeek package surface.

## Current implemented parity

The current implementation already has the desired package shape:

- `siumai-provider-deepseek` exposes `DeepSeekProviderSettings`, `DeepSeekErrorData`, and
  `VERSION`.
- `siumai::provider_ext::deepseek` re-exports builder/config/client, package construction helpers,
  package settings/version, typed options, typed metadata helpers, and curated chat model constants.
- `DeepSeekProviderFactory` routes text-family construction through the provider-owned
  `DeepSeekBuilder`.
- Registry and provider-owned clients reject non-text family paths instead of inheriting broad
  OpenAI-compatible capabilities from the shared runtime.
- DeepSeek stream requests preserve `stream_options.include_usage = true` where the provider path
  needs it.

## Validation

The package boundary is locked by:

- `siumai-registry` DeepSeek factory and catalog contract tests
- `siumai` public-surface compile guards for `provider_ext::deepseek`
- `siumai` public-path parity tests for package settings, builder/provider/config/registry request
  equivalence, typed options, response metadata, and stream behavior

## Remaining follow-up

- Revisit DeepSeek if upstream adds another public model family to `@ai-sdk/deepseek`.
- Keep DeepSeek error-data shape aligned with the audited upstream `DeepSeekErrorData` export.
- Keep provider-owned reasoning/include-usage behavior covered by public-path tests so generic
  OpenAI-compatible runtime changes do not accidentally widen the DeepSeek package surface.
