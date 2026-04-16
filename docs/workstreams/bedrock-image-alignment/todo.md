# Amazon Bedrock Image Alignment - TODO

Last updated: 2026-04-15

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Upstream audit

- [x] Audit `repo-ref/ai/packages/amazon-bedrock/src/bedrock-provider.ts`.
- [x] Audit `repo-ref/ai/packages/amazon-bedrock/src/bedrock-image-model.ts`.
- [x] Audit `repo-ref/ai/packages/amazon-bedrock/src/bedrock-image-settings.ts`.
- [x] Confirm that upstream exposes `image()` / `imageModel()` but does not export a public image
  options type from `index.ts`.

## Track B - Provider-owned image runtime

- [x] Add a first-class Bedrock image standard and spec.
- [x] Route image requests to `/model/{id}/invoke`.
- [x] Implement `TEXT_IMAGE` request shaping.
- [x] Implement `INPAINTING` request shaping.
- [x] Implement `OUTPAINTING` request shaping.
- [x] Implement `BACKGROUND_REMOVAL` request shaping.
- [x] Implement `IMAGE_VARIATION` request shaping.
- [x] Parse base64 image responses and preserve metadata.
- [x] Raise a provider error for moderated Bedrock image responses.
- [x] Mirror the audited `maxImagesPerCall` defaults on the provider runtime path.

## Track C - Public/runtime capability parity

- [x] Make `BedrockClient` expose `ImageGenerationCapability`.
- [x] Expose provider-owned edit/variation through `ImageExtras`.
- [x] Add the `image_model(...)` builder alias on the provider-owned Bedrock builder.
- [x] Advertise Bedrock image generation in native provider metadata and factory capabilities.
- [x] Add `image_model_with_ctx(...)` on the Bedrock registry factory.
- [x] Replace the old fail-fast Bedrock image contract tests with request-shape parity tests.
- [x] Replace the old fail-fast public-path Bedrock image tests with positive parity tests.

## Track D - Boundary hardening

- [x] Warn for unsupported `aspectRatio`.
- [x] Warn when callers request non-`b64_json` response formats.
- [x] Warn for unsupported `steps` and `enhance_prompt`.
- [x] Reject URL-backed image edit/variation inputs before transport.
- [x] Keep image-only Bedrock provider options private to the provider runtime instead of
  exporting a fake stable public type.

## Track E - Docs and changelog

- [x] Create a dedicated `docs/workstreams/bedrock-image-alignment/` folder.
- [x] Update `CHANGELOG.md` `Unreleased` with the Bedrock image alignment changes.
- [x] Update AI SDK structural-alignment notes to reflect the new Bedrock image runtime reality.
- [x] Update older Bedrock workstreams/milestones that still described image as unsupported.

## Track F - Intentional deferrals

- [-] Do not add a public stable Bedrock image-options type until the upstream package exports one
  from `packages/amazon-bedrock/src/index.ts`.
- [-] Do not widen Bedrock image support into provider-agnostic URL materialization semantics in
  this workstream.
