# Amazon Bedrock Image Alignment - Design

Last updated: 2026-04-15

## Problem

Compared with `repo-ref/ai/packages/amazon-bedrock/src/bedrock-provider.ts`,
`bedrock-image-model.ts`, and `bedrock-image-settings.ts`, Siumai's Bedrock surface had drifted in
one important place:

- the upstream provider already exposes `image(modelId)` and `imageModel(modelId)`
- the upstream image runtime is provider-owned and routes to `/model/{id}/invoke`
- the upstream package already defines audited image batching behavior through
  `modelMaxImagesPerCall`

The Rust side still behaved like image was intentionally unsupported:

- `BedrockClient` did not expose `ImageGenerationCapability`
- native provider metadata and registry factory capabilities did not advertise image generation
- `registry.image_model("bedrock:...")` stayed on a fail-fast dead-end path
- top-level public-path and lower contract tests still locked Bedrock image as unsupported

At the same time, the upstream package boundary has an important nuance:

- `packages/amazon-bedrock/src/index.ts` exports embedding, chat, and rerank option types
- but it does not export a public Bedrock image option type

That means the real parity target is not "invent a brand-new stable Rust image options type". The
real target is "match the upstream Bedrock image runtime and public model-construction surface
without over-exporting types the reference package still keeps private".

## Goals

- Audit the Bedrock image lane against the upstream AI SDK package/runtime.
- Add a real provider-owned Bedrock image runtime instead of leaving the public boundary
  intentionally unsupported.
- Make builder/config-first/registry/public paths converge on the same final `/model/{id}/invoke`
  image request shape.
- Mirror the audited Bedrock image batching behavior, especially `amazon.nova-canvas-v1:0`.
- Preserve the upstream package-boundary nuance that Bedrock image options are still runtime-only
  provider options, not a public package export.

## Non-goals

- Do not invent a public `AmazonBedrockImageModelOptions` or `BedrockImageOptions` type while the
  upstream `index.ts` does not export one.
- Do not force Bedrock image generation onto a generic OpenAI-compatible runtime.
- Do not silently materialize URL-backed edit/variation inputs when the upstream Bedrock image
  runtime expects direct image data.

## Chosen design

### 1. Add a first-class Bedrock image standard

The Bedrock image lane is implemented as a dedicated provider-owned standard instead of ad hoc JSON
construction inside the client:

- `BedrockImageStandard`
- `BedrockImageSpec`
- request routing to `/model/{id}/invoke`
- request/response transformers owned by the Bedrock provider crate

That keeps image aligned with the existing provider-owned Bedrock chat/embedding/rerank structure
instead of creating another special-case transport path.

### 2. Follow the audited upstream task split

The request transformer mirrors the upstream `bedrock-image-model.ts` task model:

- `TEXT_IMAGE`
- `INPAINTING`
- `OUTPAINTING`
- `BACKGROUND_REMOVAL`
- `IMAGE_VARIATION`

The response transformer also matches the upstream Bedrock contract:

- `images: string[]` maps onto stable `GeneratedImage { b64_json }`
- moderated responses (`status == "Request Moderated"`) fail before returning images
- Bedrock image metadata is preserved on the response metadata map

### 3. Keep image-only provider options private to the Bedrock runtime

Because the upstream package does not export a public image options type from `index.ts`, the Rust
surface intentionally does not add a stable public `BedrockImageOptions` mirror.

Instead, Bedrock image-only knobs are parsed privately inside the provider-owned image standard:

- `quality`
- `cfgScale`
- `negativeText`
- `style`
- `maskPrompt`
- `taskType`
- `outPaintingMode`
- `similarityStrength`

The same runtime also reads stable-image fallback fields from the shared request structs where that
makes sense (`negative_prompt`, `quality`, `style`, `guidance_scale`, and `extra_params`) so the
public stable image request lane stays useful without overfitting the package boundary.

### 4. Promote image to a real Bedrock capability

The Bedrock provider surface now treats image generation as a first-class capability:

- `BedrockClient` exposes `ImageGenerationCapability`
- `BedrockClient` also exposes `ImageExtras` for edit/variation
- Bedrock native metadata and registry factories advertise image generation
- Bedrock builder adds the `image_model(...)` alias
- `registry.image_model("bedrock:...")` now resolves to a real Bedrock image handle

This closes the earlier mismatch where the reference package exposed `image()` / `imageModel()`
while the Rust public surface still rejected the same family before transport.

### 5. Preserve upstream runtime constraints instead of papering over them

Some Bedrock image constraints are intentionally surfaced as warnings or fail-fast behavior:

- `aspectRatio` is warned as unsupported in favor of `size`
- `response_format != "b64_json"` is warned because Bedrock still returns base64 payloads
- `steps` and `enhance_prompt` are warned as unsupported
- URL-backed image edit/variation inputs are rejected before transport

That keeps the Rust behavior aligned with the audited Bedrock runtime instead of pretending the
shared image abstraction implies fully generic image transport semantics.

### 6. Mirror upstream batching defaults

The Bedrock image lane also mirrors the upstream `modelMaxImagesPerCall` contract:

- `amazon.nova-canvas-v1:0` -> `5`
- all other Bedrock image models currently default to `1`

This mapping is wired through both the provider runtime and the public registry image handle path.

## Validation

This workstream is locked by:

- request/response transformer tests in
  `siumai-provider-amazon-bedrock/src/standards/bedrock/image.rs`
- provider-owned client/builder tests in
  `siumai-provider-amazon-bedrock/src/providers/bedrock/{client.rs,builder.rs}`
- Bedrock factory/registry contract tests in
  `siumai-registry/src/registry/factories/contract_tests.rs`
- public surface compile guards in `siumai/tests/public_surface_imports_test.rs`
- top-level public-path parity tests in `siumai/tests/provider_public_path_parity_test.rs`

## Remaining follow-up

- Re-audit the upstream package if it eventually exports a public Bedrock image option type; only
  then should Rust consider freezing a stable provider-owned image-options facade.
- Re-check whether additional Bedrock image model ids need explicit curated constants or audited
  `maxImagesPerCall` mappings beyond `amazon.nova-canvas-v1:0`.
- Consider whether the shared image executor should eventually expose a provider-spec hook for
  "reject URL materialization" instead of keeping that rule on the Bedrock client surface.
