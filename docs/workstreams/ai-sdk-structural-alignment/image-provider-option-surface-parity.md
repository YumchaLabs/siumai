# Image Provider-Option Surface Parity

Last updated: 2026-04-14

## Goal

Tighten the provider-owned image option surface so the stable Rust facade matches the audited AI SDK
package exports and request-helper behavior more closely, especially on the newer unified
`GenerateImageRequest` lane.

Upstream reference:

- `repo-ref/ai/packages/google/src/google-generative-ai-image-model.ts`
- `repo-ref/ai/packages/google/src/index.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-image-model.ts`
- `repo-ref/ai/packages/xai/src/xai-image-options.ts`
- `repo-ref/ai/packages/togetherai/src/togetherai-image-model.ts`

## What Changed

- `TogetherAiImageRequestExt`, `XaiImageRequestExt`, and `VertexImagenRequestExt` now also cover
  the stable unified `GenerateImageRequest` surface instead of only the older split image request
  structs.
- TogetherAI's typed image request helper also now covers `ImageVariationRequest`, so the older
  compatibility request family is structurally complete as a carrier for provider-owned image
  options even when runtime support still varies by provider.
- xAI and Google Vertex image request ext helpers now merge typed image options into existing
  `provider_options_map["xai"|"vertex"]` objects instead of overwriting sibling raw provider
  options.
- Google Vertex's typed image option surface now also covers the main audited generation fields
  from `GoogleVertexImageModelOptions` instead of only `negativePrompt/edit/referenceImages`:
  - `personGeneration`
  - `safetySetting`
  - `addWatermark`
  - `storageUri`
  - `sampleImageSize`
- Gemini now has a dedicated image-family typed option surface:
  - `GeminiImageOptions`
  - AI SDK-style alias `GoogleImageModelOptions`
  - deprecated migration alias `GoogleGenerativeAIImageProviderOptions`
- `GeminiImageRequestExt` now exists for:
  - `ImageGenerationRequest`
  - `ImageEditRequest`
  - `ImageVariationRequest`
  - `GenerateImageRequest`
- Gemini request helpers also merge onto existing `provider_options_map["gemini"]` instead of
  replacing the whole provider-owned object.
- The public facade now exposes the new Google/Gemini image typed surface on both:
  - `provider_ext::gemini::*`
  - `provider_ext::google::*`

## Structural Notes

- The Gemini image transformer already accepted top-level `providerOptions.google|gemini` image
  keys like `aspectRatio` and `personGeneration`; this work mainly closes the public typed-surface
  gap so callers no longer have to hand-author raw JSON for those fields.
- The Vertex Imagen runtime already accepted the wider audited provider option subset through the
  request transformer allowlist; this follow-up closes the typed-struct gap so callers no longer
  need raw `providerOptions.vertex` JSON for those generation fields.
- Merge semantics are the more correct stable behavior for provider-owned typed helpers because AI
  SDK treats `providerOptions[providerId]` as one open object that may contain both typed known
  fields and provider-specific escape hatches.
- This keeps the stable shared image request family provider-agnostic while letting provider crates
  own their own typed option shapes and aliases.

## Validation

- `cargo check -p siumai-provider-gemini -p siumai-provider-xai -p siumai-provider-togetherai -p siumai-provider-google-vertex -p siumai --all-features`
- `cargo nextest run -p siumai-provider-gemini --all-features`
- `cargo nextest run -p siumai-provider-xai --all-features`
- `cargo nextest run -p siumai-provider-togetherai --all-features`
- `cargo nextest run -p siumai-provider-google-vertex --all-features`
- `cargo nextest run -p siumai --all-features gemini_imagen_`
- `cargo nextest run -p siumai --all-features public_surface_`

## Remaining Follow-up

- The audited TogetherAI/xAI/Gemini/Google Vertex image typed surface is now structurally in a
  good state, but other provider-owned image option lanes still need the same export/request-ext
  audit treatment where upstream packages expose first-class typed image options.
- The stable shared image response still exposes a single top-level `response` slot, so helper-side
  batching compatibility metadata remains under `metadata._siumai` until a broader response-shape
  refactor is worth the compatibility cost.
