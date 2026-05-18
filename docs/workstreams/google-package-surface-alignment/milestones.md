# Google Package Surface Alignment - Milestones

Last updated: 2026-04-22

## Milestone 1 - Audit the upstream package boundary

Status: complete

- Audited `repo-ref/ai/packages/google/src/index.ts` as the package-surface source of truth.
- Split package-shape gaps from runtime-lowering gaps so missing exports could be fixed without
  pretending Rust has the same callable provider model as TypeScript.

## Milestone 2 - Close the honest root export gaps

Status: complete

- Added Google-branded typed exports for options, metadata, upload options, error data, and video
  model ids.
- Added a dedicated provider-level `GoogleProviderSettings` input struct plus deprecated
  `GoogleGenerativeAIProviderSettings` alias, both converting into the real Rust builder/config
  carrier.
- Mirrored package-level `google`, `create_google`, deprecated
  `create_google_generative_ai()`, and `VERSION`.
- Mirrored the non-callable `GoogleProvider` family helper names on the Rust builder surface where
  that mapping is structurally honest.
- Mirrored the audited `files()` provider member on the Rust builder surface through the existing
  provider-owned `GeminiFiles` capability instead of leaving file management trapped behind only
  built-client/resource paths.
- Added grouped Google model-id exports on the public facade:
  `provider_ext::google::{chat, embedding, image, video, model_sets}`.
- Added the `google.interactions(...)` package boundary on the Rust facade without routing it
  through ordinary Gemini chat. `GeminiBuilder::interactions(...)` now returns an explicit deferred
  `GoogleInteractionsLanguageModel` handle, and the public facade exposes Interactions model ids,
  agent names, typed options, typed metadata, and request-option helpers.
- Added honest `generateId` support by threading a shared generator through
  `GoogleProviderSettings` / `GeminiBuilder` / `GeminiConfig` and consuming it for provider-owned
  tool-call, tool-result, and source ids in Gemini response/streaming transformers.
- Added honest `name` support by carrying a provider-facing display label through
  `GoogleProviderSettings` / `GeminiBuilder` / `GeminiConfig` and surfacing it on
  `GeminiClient::provider_name()` / `GeminiFiles::provider_name()` without changing canonical
  `provider_id`, `providerReference`, or `providerMetadata` roots.
- Locked the facade with `siumai/tests/public_surface_imports_test.rs`.

## Milestone 3 - Keep deferred gaps explicit

Status: in progress

- `GoogleProvider` and `GoogleGenerativeAIProvider` remain intentionally deferred because Rust does
  not expose a TypeScript-style callable provider object.
- Task-based Google video polling remains explicitly deferred from full AI SDK-style provider-owned
  model polling semantics.
- `google.interactions(...)` execution remains explicitly deferred from ordinary Gemini chat
  execution. A dedicated `/interactions` runtime lane is required before Siumai can support actual
  Interactions HTTP calls, polling, cancellation, stream transformation, signatures, and
  interaction-id state.

## Exit criteria for this workstream

- Future audits against `repo-ref/ai/packages/google/src/index.ts` should find only intentional
  differences, not accidental missing exports.
- Any newly added upstream root export should either gain a direct Rust mirror or be recorded here
  as an intentional non-goal with rationale.
