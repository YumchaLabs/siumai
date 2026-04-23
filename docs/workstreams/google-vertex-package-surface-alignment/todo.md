# Google Vertex Package Surface Alignment - TODO

Last updated: 2026-04-22

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Root export parity

- [x] Audit `repo-ref/ai/packages/google-vertex/src/index.ts`.
- [x] Expose `VERSION` on the provider-owned/public Vertex surface.
- [x] Expose a dedicated `GoogleVertexProviderSettings` input struct on the
  provider-owned/public Vertex surface.

## Track B - Provider member parity

- [x] Audit `repo-ref/ai/packages/google-vertex/src/google-vertex-provider.ts`.
- [x] Mirror `languageModel` on `GoogleVertexBuilder`.
- [x] Mirror `embeddingModel` and deprecated `textEmbeddingModel` on `GoogleVertexBuilder`.
- [x] Mirror `image` / `imageModel` on `GoogleVertexBuilder`.
- [x] Mirror `video` / `videoModel` on `GoogleVertexBuilder`.
- [x] Lock the builder helpers through `siumai/tests/public_surface_imports_test.rs`.

## Track C - Model-id parity

- [x] Re-audit the current `GoogleVertexModelId` contract.
- [x] Re-audit the current `GoogleVertexEmbeddingModelId` contract.
- [x] Re-audit the current `GoogleVertexImageModelId` contract.
- [x] Re-audit the current `GoogleVertexVideoModelId` contract.
- [x] Expand the curated grouped Vertex ids to cover the current audited package ids.
- [x] Keep the provider-owned `imagen-3.0-edit-001` runtime id explicit as a Rust-only extra.
- [x] Make `GoogleVertexClient::supported_models()` reuse the same curated model source.
- [x] Tighten registry catalog assertions so the expanded Vertex model surface stays locked.

## Track D - Docs and changelog

- [x] Add a dedicated `docs/workstreams/google-vertex-package-surface-alignment/` folder.
- [x] Add a root-export / provider-member / model-id matrix for this package slice.
- [x] Record the slice in `CHANGELOG.md` `Unreleased`.

## Track E - `generateId` parity

- [x] Audit upstream `generateId` usage in `google-vertex-provider.ts`.
- [x] Add `generate_id` to `GoogleVertexProviderSettings`.
- [x] Add `with_generate_id(...)` / `with_shared_generate_id(...)` to the builder/config surface.
- [x] Inject the custom generator into the reused Gemini chat/stream transformer runtime.
- [x] Lock non-streaming and streaming custom-ID behavior with provider tests.

## Track F - Intentional deferrals

- [-] Do not fabricate a callable `GoogleVertexProvider` export on the Rust facade.
- [-] Do not pretend Node-only `googleAuthOptions` has a direct Rust API equivalent.
- [-] Do not fabricate extra local image/video IDs until the Rust runtime has a truthful ownership
  point for them.

## Track G - Provider-option data structures

- [x] Audit the constrained enum domains used by Vertex image/video option schemas.
- [x] Expose explicit Rust enums for the audited `personGeneration` / `safetySetting` /
  `sampleImageSize` / edit-mode / mask-mode domains.
- [x] Add fluent builder helpers to `VertexEmbeddingOptions`.
- [x] Lock the typed option enums and embedding builder through serialization/public-surface tests.

## Track H - Image/video result-shape parity

- [x] Re-audit `google-vertex-image-model.ts` and `google-vertex-video-model.ts` result metadata.
- [x] Confirm Vertex Imagen `prompt -> revised_prompt` mapping already reaches the stable Rust image
  result.
- [x] Stop duplicating inline/base64 video payloads into public `provider_metadata.videos[]`.
- [x] Keep a hidden internal raw-video carrier so the task-based Rust runtime can still reconstruct
  final generated videos without widening the public metadata contract.

## Track I - Gemini image runtime parity

- [x] Re-audit the upstream `gemini-* image` path in `google-vertex-image-model.ts`.
- [x] Split Vertex Gemini image routing away from `VertexImagenStandard`.
- [x] Route `gemini-* image` generate/edit/variation through `:generateContent`.
- [x] Mirror Gemini image request semantics for `responseModalities`, `aspectRatio`, `seed`,
  prompt+files edit payloads, and variation payloads.
- [x] Keep Gemini image open provider options scoped to `providerOptions["vertex"]`, matching the
  audited `GoogleVertexImageModel` boundary instead of the broader shared Google language-model
  fallback aliases.
- [x] Reject `mask` and `n > 1` on the Vertex Gemini image path.
- [x] Make URL-backed edit/variation image inputs provider-controlled in the shared executor so
  Vertex Gemini can preserve native `fileData.fileUri` behavior.
- [x] Lock the runtime split through provider and core regression tests.
