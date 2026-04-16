# Image Unified Helper Parity

Last updated: 2026-04-13

## Goal

Align the public Rust image surface more closely with AI SDK `generateImage()` without forcing all
provider runtimes into one fake generic transport path.

Upstream reference:

- `repo-ref/ai/packages/provider/src/image-model/v4/image-model-v4-call-options.ts`
- `repo-ref/ai/packages/provider/src/image-model/v4/image-model-v4.ts`
- `repo-ref/ai/packages/ai/src/generate-image/generate-image.ts`

## What Changed

- `siumai-spec/src/types/image.rs` now exposes `GenerateImageRequest`, a unified request shape that
  carries `prompt`, `files`, `mask`, shared image knobs (`count`, `size`, `aspectRatio`, `seed`),
  canonical `providerOptions`, and the existing Rust generation-oriented extras.
- `siumai::image` now exposes:
  - `generate_image(...)` for the AI SDK-style unified helper lane
  - `edit(...)` and `variation(...)` as public facade helpers over `ImageExtras`
- `GenerateImageRequest` can be constructed directly or derived from the older split request
  structs (`ImageGenerationRequest`, `ImageEditRequest`, `ImageVariationRequest`).
- `ImageGenerationCapability` and `ImageModelV3` now also expose an object-safe
  `max_images_per_call()` metadata getter so helper-level batching can mirror AI SDK
  `maxImagesPerCall` without breaking `dyn ImageModel` object safety.
- `siumai::image::GenerateOptions` now also accepts `max_images_per_call`, and the public
  `generate(...)`, `edit(...)`, `variation(...)`, and `generate_image(...)` helpers split larger
  `count` requests into parallel per-call batches using:
  - explicit `GenerateOptions.max_images_per_call`
  - otherwise the model/provider default
  - otherwise the final fallback `1`

## Dispatch Model

The Rust facade intentionally keeps the AI SDK structure/runtime split explicit:

- No `files` and no `mask`: dispatch to generation
- Any `mask`: dispatch to edit
- More than one file: dispatch to edit
- One file plus a non-empty prompt: dispatch to edit
- One file with no prompt and no mask: dispatch to variation
- If variation returns `UnsupportedOperation`, the facade falls back to edit

This matches the architectural reality in `repo-ref/ai`: the stable request shape is shared, but
provider-owned image execution still varies by provider/package.

## Architectural Decision

The unified helper must remain a facade/helper concern, not a required method on
`siumai-core::image::ImageModelV4` itself.

Reasoning:

- In upstream AI SDK, the stable provider-side contract is `ImageModelV4#doGenerate(...)`, while
  `generateImage(...)` is a separate high-level helper.
- Siumai registry/model-handle code depends heavily on `Arc<dyn FamilyImageModel>`.
- Promoting unified dispatch onto the Rust `ImageModelV4` trait as a default async method with
  extra bounds breaks object safety and makes the image model family no longer `dyn` compatible.

So the stable Rust shape is now explicit:

- `ImageModelV4` stays the object-safe family naming contract.
- `siumai::image::generate_image(...)` owns unified request classification and fallback behavior.
- Future parity work such as AI SDK-style `maxImagesPerCall` should be added through object-safe
  metadata/getter surfaces plus helper-level batching, not by turning the family trait into a
  transport-owning execution facade.

## Compatibility Notes

- The older split request structs remain public and are still the transport-facing compatibility
  layer.
- Generation-only extra fields (`negative_prompt`, `quality`, `style`, `steps`,
  `guidance_scale`, `enhance_prompt`) are preserved when the unified helper lowers into edit or
  variation by copying them into `extra_params` if the target request does not already carry them
  explicitly.
- Because the stable Rust `ImageGenerationResponse` still exposes a single top-level `response`
  field, multi-call helper batching stores the full per-call response envelopes and metadata under
  `metadata._siumai.{responses,metadata}` and emits a compatibility warning instead of pretending
  the batched result was one provider call.

## Remaining Follow-up

- Provider/runtime parity still needs per-provider audit coverage beyond the current OpenAI,
  OpenAI-compatible, Google Vertex, TogetherAI, Fireworks, DeepInfra, xAI, and the now-audited
  Google/Gemini image option surface. Provider-owned option/export follow-up is now tracked
  separately in `docs/workstreams/ai-sdk-structural-alignment/image-provider-option-surface-parity.md`.
- The stable Rust image response still has a single `response` slot, so batched helper calls must
  keep the richer per-call response array under `metadata._siumai` until a broader response-shape
  refactor is worth the compatibility cost.
