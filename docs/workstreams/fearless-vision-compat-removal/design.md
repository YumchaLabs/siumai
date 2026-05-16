# Fearless Vision Compatibility Removal - Design

Last updated: 2026-05-16

## Context

The architecture already treats image understanding as multimodal chat input and image creation as
the image-generation family. A separate `VisionCapability` family does not exist in the
Vercel-aligned target model.

The compatibility audit already classifies the dedicated vision surface as removable:

- `SiumaiBuilder::vision(...)`
- `Siumai::vision_capability()`
- `VisionCapability`
- `VisionCapabilityProxy`

Most of that surface is now deprecated, returns placeholder behavior, or only forwards to another
deprecated shape. Keeping it increases the apparent number of model families and weakens the
family-first story that the rest of the refactor has converged on.

## Decision

Remove the dedicated vision compatibility surface instead of preserving another compatibility
window.

Canonical replacements:

- image understanding: `ChatMessage` / model-message multimodal content parts
- image generation: `ImageModel`, `ImageGenerationCapability`, `GenerateImageRequest`, and
  `siumai::image::*`
- provider-specific image extras: explicit provider extension modules or `siumai::extensions::*`

## Goals

- Remove `VisionCapability` from public core/registry/facade exports.
- Remove `VisionCapabilityProxy` and `Siumai::vision_capability()`.
- Remove deprecated placeholder vision request/response aliases if they have no remaining owner.
- Update migration docs with before/after guidance.
- Add a source guard so the dedicated vision family cannot reappear as a hidden compatibility path.

## Non-goals

- Do not remove multimodal chat image/file content support.
- Do not remove the stable image-generation family.
- Do not rename provider-owned image generation APIs.
- Do not collapse provider-specific image edit/variation extras into the core image family in this
  lane.

## Target Shape

Allowed:

- `ChatMessage` and prompt/model-message content parts for image understanding
- `ImageModel`, `ImageModelV4`, `ImageGenerationCapability`, `GenerateImageRequest`, and
  image helper modules for image creation
- explicit provider extension APIs for provider-specific image behavior

Removed:

- `VisionCapability`
- `VisionCapabilityProxy`
- `Siumai::vision_capability()`
- compatibility request/response aliases that only existed for `VisionCapability`

## Validation

Focused gates for this lane:

```text
cargo fmt --package siumai-spec --package siumai-core --package siumai-registry --package siumai --check
cargo check -p siumai-core --no-default-features
cargo check -p siumai-registry --tests --features openai,google --no-default-features
cargo check -p siumai --tests --features openai,google --no-default-features
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,google --no-default-features --no-fail-fast
cargo nextest run -p siumai --test public_surface_imports_test --features openai,google --no-default-features --no-fail-fast
git diff --check
```
