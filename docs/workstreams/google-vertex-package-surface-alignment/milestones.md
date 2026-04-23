# Google Vertex Package Surface Alignment - Milestones

Last updated: 2026-04-22

## Milestone 1 - Audit the package boundary

Status: complete

- Audited `repo-ref/ai/packages/google-vertex/src/index.ts` as the root export source of truth.
- Audited `repo-ref/ai/packages/google-vertex/src/google-vertex-provider.ts` as the provider-member
  shape source of truth.
- Split package-root drift, provider-member drift, and model-id drift into separate follow-up
  tracks.

## Milestone 2 - Close the honest surface gaps

Status: complete

- Exposed `VERSION` and a dedicated `GoogleVertexProviderSettings` input struct on the
  provider-owned/public Vertex surface.
- Mirrored `image` / `imageModel` / `video` / `videoModel` on `GoogleVertexBuilder`.
- Kept the Rust story honest by continuing to defer the callable `GoogleVertexProvider` object.

## Milestone 3 - Unify the model-id source

Status: complete

- Expanded the grouped Vertex ids to cover the current audited chat/embedding/image/video package
  ids.
- Kept `imagen-3.0-edit-001` explicit as a provider-owned Rust runtime extra.
- Reused the same curated model source for `GoogleVertexClient::supported_models()` and registry
  catalog output.
- Locked the surface through provider, public-facade, and registry tests.

## Milestone 4 - Close the `generateId` runtime gap

Status: complete

- Added `generate_id` to the provider-settings / builder / config path.
- Extended the reused Gemini chat standard so Vertex can inject a base `GeminiConfig`.
- Made Vertex chat/stream runtime actually honor custom stable ID generation for tool calls and
  sources instead of exposing a surface-only placeholder.

## Milestone 5 - Tighten option data structures

Status: complete

- Added explicit Rust enums for the constrained Vertex image/video option domains that matter in
  the audited AI SDK schemas.
- Added fluent `VertexEmbeddingOptions` builders so the embedding option surface is no longer just a
  raw struct alias.
- Locked the typed option surface through provider-option serialization tests and the public facade
  compile test.

## Milestone 6 - Normalize image/video result metadata

Status: complete

- Re-audited the current Vertex image/video result shapes against
  `google-vertex-image-model.ts` and `google-vertex-video-model.ts`.
- Confirmed Vertex Imagen already maps `prediction.prompt` into the stable Rust
  `GeneratedImage.revised_prompt` lane.
- Split Vertex video metadata into a public lightweight provider-metadata lane plus a hidden
  Rust-only raw-video carrier, so public `provider_metadata.videos[]` no longer duplicates inline
  payloads while the task-based runtime still has enough data to reconstruct generated assets.

## Milestone 7 - Split Gemini image runtime from Imagen

Status: complete

- Re-audited the upstream `google-vertex-image-model.ts` Gemini image branch and confirmed that
  `gemini-* image` models belong on `:generateContent`, not Imagen `:predict`.
- Added a dedicated Vertex Gemini image standard so generate/edit/variation now serialize through
  Gemini multi-part `contents[].parts[]` with `responseModalities = ["IMAGE"]`.
- Kept the open Gemini image option boundary honest as well: this lane now treats
  `providerOptions["vertex"]` as the public contract instead of inheriting the broader shared
  `GoogleLanguageModel` fallback aliases.
- Restored audited Gemini image constraints on the Vertex path: `mask` is rejected, `n > 1` is
  rejected, and `size` remains a warning.
- Fixed the shared image executor so providers can opt out of forced URL materialization on
  edit/variation paths; Vertex Gemini now preserves native URL-backed inputs as `fileData.fileUri`.

## Exit criteria

- Future `repo-ref/ai/packages/google-vertex/src/*` audits should find only intentional
  differences, not missing root exports or stale grouped model ids.
- Any future upstream package-root or provider-member addition should either get a direct Rust
  mirror or be recorded here as an intentional deferral with rationale.
- Any future `generateId` claim on the Vertex path must stay runtime-backed, not just surface-only.
- Any future Vertex Gemini image claim must stay runtime-backed too: `gemini-* image` may not
  silently regress back onto the Imagen `:predict` path or the forced URL-materialization path.
