# Google Vertex Package Surface Alignment - Design

Last updated: 2026-05-18

## Problem

Compared with `repo-ref/ai/packages/google-vertex/src/index.ts` and
`repo-ref/ai/packages/google-vertex/src/google-vertex-provider.ts`, Siumai still had a few
package-surface and data-source drifts on the Vertex path:

- `provider_ext::google_vertex` did not expose the audited package-root `VERSION` constant or the
  upstream primary `googleVertex` / `createGoogleVertex` aliases
- the public/provider-owned Vertex surface did not expose a dedicated
  `GoogleVertexProviderSettings` input struct
- `GoogleVertexBuilder` mirrored `languageModel` / `embeddingModel` only partially, but still
  missed the upstream non-callable `image` / `imageModel` / `video` / `videoModel` family helpers
- curated Vertex model ids had drifted behind the current AI SDK package contracts, especially for
  embedding and image ids
- `GoogleVertexClient::supported_models()` used the single configured model id instead of the same
  curated model source already used by the registry catalog
- upstream `generateId` existed on `createGoogleVertex(options)` / `createVertex(options)`, but the
  Rust Vertex path did not have a truthful equivalent wired into the Gemini chat/stream transformer
  runtime
- `gemini-*` Vertex image models still routed through the Imagen `:predict` runtime instead of the
  upstream `:generateContent` Gemini image path
- the Vertex video path still duplicated inline/base64 video payloads into public metadata, while
  the upstream AI SDK only exposes lightweight `providerMetadata['google-vertex'].videos[]`
  entries there

This made `repo-ref/ai` diffing noisy and let the same provider advertise different model surfaces
through different Rust entry points.

## Goals

- Mirror the honest `@ai-sdk/google-vertex` package-root names that already have direct Rust
  analogues.
- Mirror the upstream non-callable provider family helper names where the Rust builder has a real
  equivalent.
- Bring the grouped Vertex model-id surface closer to the current audited AI SDK package ids.
- Reuse one Vertex model source across provider-owned introspection and registry catalog output.
- Mirror the honest `generateId` provider setting and make it actually affect Vertex-owned stable
  IDs on the chat/stream runtime path.
- Route Vertex Gemini image models through the same `generateContent` request family as the audited
  AI SDK package, including edit/variation semantics.
- Keep Vertex image/video result metadata honest relative to the audited AI SDK package shape,
  without breaking the Rust-only task-based video runtime.

## Non-goals

- Do not fabricate a TypeScript-style callable `GoogleVertexProvider` object on the Rust facade.
- Do not pretend the Node-only `googleAuthOptions` story has a direct Rust equivalent; the Rust
  runtime still uses explicit token providers / ADC.
- Do not remove the provider-owned `imagen-3.0-edit-001` runtime constant just because the current
  AI SDK `GoogleVertexImageModelId` type does not expose it.
- Do not fabricate extra local image/video IDs where the current Rust Vertex runtime only forwards
  service-owned operation/task identifiers.

## Chosen design

### 1. Mirror the honest package-root names

The provider-owned/public Vertex facade now exposes:

- `VERSION`
- `google_vertex()` / `create_google_vertex()`
- `GoogleVertexProviderSettings`

`GoogleVertexProviderSettings` is now a dedicated package-level input struct again, instead of a
`GoogleVertexConfig` alias. That matters because upstream settings do not carry a model id, while
the Rust config-first carrier does.

The current Rust settings struct exposes the honest provider-construction subset directly:

- `api_key`
- `project`
- `location`
- `headers`
- `fetch`
- `generate_id`
- `base_url`

It also exposes a Rust-native auth analogue:

- `token_provider`

and converts into the provider-owned runtime through:

- `into_builder()`
- `into_builder_for_model(model)`
- `into_config_for_model(model)`

### 2. Mirror the upstream non-callable family helpers on the builder

`GoogleVertexBuilder` now mirrors the audited provider member names that have direct Rust builder
equivalents, and the public facade keeps both the primary and deprecated package aliases:

- `google_vertex`
- `create_google_vertex`
- `language_model`
- `embedding_model`
- deprecated `text_embedding_model`
- `image`
- `image_model`
- `video`
- `video_model`
- deprecated `vertex`
- deprecated `create_vertex`

This still does not imply a callable provider object. It only keeps the builder surface close to
the AI SDK provider-member shape where that mapping is structurally honest.

### 3. Expand the grouped Vertex model ids from the audited package contracts

The curated grouped Vertex ids now cover the currently audited AI SDK package ids for:

- chat
- embedding
- image
- video

This includes newer ids such as:

- `text-embedding-005`
- `gemini-embedding-2-preview`
- `imagen-3.0-generate-001`
- `imagen-4.0-ultra-generate-001`
- `gemini-2.5-flash-image`
- `gemini-3-pro-image-preview`
- `gemini-3.1-flash-image-preview`

Siumai also keeps the provider-owned `imagen-3.0-edit-001` constant as an explicit Rust runtime
extra.

### 4. Reuse the same model source for introspection and catalog output

`GoogleVertexClient::supported_models()` now reuses the same curated `models.rs` source that the
registry catalog already consumes.

That removes an internal semantic split where:

- provider-owned client introspection only returned the configured model id
- registry/catalog output returned a larger curated subset

### 5. Close the `generateId` runtime gap honestly

`@ai-sdk/google-vertex` accepts `generateId` at provider-construction time and forwards it into the
language/image/video model config objects.

On the Rust side, simply adding a `generate_id` field to `GoogleVertexProviderSettings`,
`GoogleVertexBuilder`, or `GoogleVertexConfig` would still have been incomplete, because the Vertex
chat path reused Gemini transformers by constructing a fresh default `GeminiConfig` internally.

The honest fix is now:

- `GoogleVertexProviderSettings`, `GoogleVertexBuilder`, and `GoogleVertexConfig` all preserve a
  shared `generate_id`
- `VertexGenerativeAiStandard` accepts an injected `GeminiConfig`
- `GeminiChatStandard` now supports a caller-provided base config for request/response/stream
  transformers
- the Vertex chat and stream runtime pass that config through, so provider-owned stable IDs now
  actually use the custom generator

This currently matters most for:

- tool call ids produced from Gemini `functionCall`
- normalized grounding/source ids
- streaming source/tool-call ids

It does **not** currently invent extra local IDs for Vertex image/video runtime paths. Those paths
still primarily expose provider/service-owned operation or asset identifiers, so the Rust mapping
stays explicit about that difference instead of pretending everything shares one local-ID contract.

### 6. Tighten the provider-option data structures

The upstream Vertex package also carries important schema semantics in its option objects, even when
the TypeScript surface exposes them as plain object properties validated by zod.

On the Rust side, the honest improvement is:

- keep the wire-facing fields string-compatible where that avoids unnecessary breakage
- add explicit Rust enums for the constrained option domains that matter operationally
- add fluent builders where the stable option surface was still too raw

This slice now exposes:

- `VertexPersonGeneration`
- `VertexImagenSafetySetting`
- `VertexImagenSampleImageSize`
- `VertexImagenEditMode`
- `VertexImagenMaskMode`
- `VertexEmbeddingOptions::new()` plus field-level builder helpers

That keeps the public Rust surface closer to the audited AI SDK option contracts without pretending
that TypeScript's object-schema validation maps 1:1 onto Rust type aliases.

### 7. Separate public video metadata from internal raw video payloads

The upstream `@ai-sdk/google-vertex` video model returns two distinct lanes:

- `videos`: the actual generated assets
- `providerMetadata['google-vertex'].videos`: lightweight provider-owned metadata (`gcsUri`,
  `mimeType`)

The earlier Rust task-based path blurred those lanes together by storing the raw Vertex
`bytesBase64Encoded` objects directly under `VideoTaskStatusResponse.metadata["vertex"]["videos"]`.
That kept the runtime working, but it also meant public provider metadata duplicated the actual
video payloads and drifted from the audited AI SDK result shape.

The honest fix is now split in two:

- the public provider-root metadata stays lightweight and AI SDK-like
- the Rust-only task runtime keeps an internal raw-video carrier under
  `metadata["_siumai"]["generatedVideos"]`

The shared high-level `video::generate(...)` path now prefers that internal raw lane when it needs
to reconstruct final generated videos, while the public aggregated
`GenerateVideoResult.provider_metadata["vertex"].videos[]` path strips duplicated inline payload
carriers back out.

That means:

- public provider metadata no longer repeats base64/byte payloads that already live on the actual
  generated-video objects
- the task-based Rust runtime still keeps enough information to materialize or preserve inline video
  results correctly
- `google-vertex` alias reads remain compatible on the high-level video path, but the canonical
  public provider root stays `vertex`

### 8. Split Vertex Gemini image runtime away from Imagen predict

The audited `repo-ref/ai/packages/google-vertex/src/google-vertex-image-model.ts` does two
different things on the image path:

- `imagen-*` models use `:predict`
- `gemini-* image` models use `:generateContent`

The earlier Rust Vertex client always selected `VertexImagenStandard` for image generation, edit,
and variation. That meant `gemini-2.5-flash-image` and related model ids were effectively treated
as Imagen models:

- wrong endpoint family (`:predict` instead of `:generateContent`)
- wrong request body shape (`instances/parameters` instead of `contents/generationConfig`)
- wrong capability semantics for `mask`, `n > 1`, and URL-backed input images

The honest fix is now:

- keep `vertex_imagen` focused on audited Imagen-only `:predict` behavior
- add a separate `vertex_gemini_image` standard that reuses the Gemini protocol transformers but
  keeps Vertex-specific auth/header/query semantics
- make `GoogleVertexClient` choose between those two standards from the normalized model id at
  runtime

That new Vertex Gemini image standard now mirrors the audited AI SDK behavior more closely:

- `gemini-* image` routes through `.../models/{model}:generateContent`
- `generationConfig.responseModalities` is forced to `["IMAGE"]`
- top-level `aspectRatio` maps into `generationConfig.imageConfig.aspectRatio`
- top-level `seed` maps into `generationConfig.seed`
- edit requests serialize prompt + files as Gemini multi-part `contents[].parts[]`
- variation requests reuse the same multi-part path with a single file input
- `mask` is rejected on the Gemini image path
- `n > 1` is rejected on the Gemini image path
- `size` remains a warning, matching the audited upstream contract

There is also an important contract-boundary detail on the open option surface:

- the audited exported `GoogleVertexImageModelOptions` type is still Imagen-centric
- Gemini image-only knobs such as `mediaResolution` or `imageConfig.imageSize` are therefore
  runtime-only raw `providerOptions["vertex"]` inputs, not a stronger dedicated exported Rust type
- unlike the shared `GoogleLanguageModel` path, the audited `GoogleVertexImageModel` Gemini branch
  only forwards `providerOptions.vertex`, so the Rust image-model runtime now keeps the same
  boundary instead of advertising `providerOptions["google"]` / `providerOptions["gemini"]` as a
  public Vertex image-model contract

This also exposed a more general executor bug: the shared image executor always materialized
URL-backed edit/variation inputs into downloaded inline bytes before request transformation. That
behavior is still correct for providers that only accept direct file payloads, but it is incorrect
for Gemini/Vertex image paths that can honestly forward URL inputs as `fileData.fileUri`.

The core executor now makes that behavior provider-controlled:

- URL materialization remains the default for backward compatibility
- providers can opt out per edit/variation path
- Vertex Gemini image opts out so URL-backed edit/variation inputs now stay URL-backed and reach
  the Gemini request transformer unchanged

## Validation

Locked by:

- `cargo nextest run -p siumai-provider-google-vertex --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --no-default-features --features google-vertex --no-fail-fast`
- `cargo nextest run -p siumai-registry provider_catalog_uses_native_metadata_for_vertex --features google-vertex --no-fail-fast`
- `cargo test -p siumai-provider-google-vertex generate_id --no-default-features --features google-vertex`
- `cargo test -p siumai-protocol-gemini base_config_generate_id_flows_into_response_transformer --no-default-features --features google`
- `cargo nextest run -p siumai-provider-google-vertex query_video_task_maps_vertex_operation_status_into_task_response --no-default-features --features google-vertex --no-fail-fast`
- `cargo nextest run -p siumai provider_metadata_video_value_strips_inline_payloads extract_generated_videos_prefers_internal_generated_video_payloads generate_accepts_google_vertex_provider_metadata_alias_on_video_path --no-default-features --features google-vertex --no-fail-fast`
- `cargo nextest run -p siumai-core image_edit_executor_can_preserve_url_backed_inputs_when_provider_opt_outs image_variation_executor_can_preserve_url_backed_inputs_when_provider_opt_outs image_edit_executor_materializes_url_backed_inputs_before_transformer image_variation_executor_materializes_url_backed_input_before_transformer --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-provider-google-vertex google_vertex_generate_images_routes_gemini_image_models_through_generate_content google_vertex_edit_image_routes_gemini_image_models_through_generate_content google_vertex_edit_image_for_gemini_models_preserves_url_inputs_as_file_data google_vertex_create_variation_routes_gemini_image_models_through_generate_content google_vertex_gemini_image_requests_reject_mask_and_n_greater_than_one --no-default-features --features google-vertex --no-fail-fast`
- `cargo check -p siumai --no-default-features --features google-vertex`

## Remaining follow-up

- Track the newer `@ai-sdk/google-vertex/xai` sub-entry under a separate package-boundary slice.
  It uses an OpenAI-compatible Vertex partner endpoint and should not be folded silently into native
  `xai` or generic Vertex MaaS.
- Re-audit `repo-ref/ai/packages/google-vertex/src/*` when the upstream package adds more
  root exports or provider members.
- Decide later whether the public Rust Vertex image surface should grow a more explicit typed
  provider-option contract for Gemini image models, or continue to rely on the open
  `providerOptions["vertex"]` map for Google-language-model image settings.
- Decide later whether the current Vertex image/video runtime should ever materialize extra local
  provider-owned IDs, or whether service-owned task/asset identifiers remain the only honest story.
- Keep future AI SDK model-id additions synchronized across `models.rs`,
  `provider_catalog.rs`, and public compile guards.
