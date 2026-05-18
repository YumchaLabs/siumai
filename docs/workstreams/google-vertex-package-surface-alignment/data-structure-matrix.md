# Google Vertex Package Surface Alignment - Data Structure Matrix

Last updated: 2026-05-18

## Audited upstream sources

- `repo-ref/ai/packages/google-vertex/src/index.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-options.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-embedding-options.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-image-settings.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-video-settings.ts`

## Root export matrix

| Upstream export | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| `createGoogleVertex` | `provider_ext::google_vertex::create_google_vertex()` | aligned | Honest snake_case builder entry. |
| `googleVertex` | `provider_ext::google_vertex::google_vertex()` | aligned | Honest builder entry alias. |
| `createVertex` | `provider_ext::google_vertex::create_vertex()` | aligned | Deprecated builder entry alias preserved. |
| `vertex` | `provider_ext::google_vertex::vertex()` | aligned | Deprecated builder entry alias preserved. |
| `GoogleVertexProviderSettings` | `provider_ext::google_vertex::GoogleVertexProviderSettings` | aligned | Dedicated provider-settings struct with `into_builder()` / `into_builder_for_model(...)`; `generateId` now also has an honest Rust analogue, while Node-only `googleAuthOptions` still does not. |
| `VERSION` | `provider_ext::google_vertex::VERSION` | aligned | Provider crate package version. |
| `GoogleVertexProvider` | none | intentionally deferred | Rust does not expose a callable provider object. |

## Provider member matrix

| Upstream member | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| call signature `vertex(modelId)` | none | intentionally deferred | No callable provider object in Rust. |
| `languageModel(modelId)` | `Provider::google_vertex().language_model(modelId)` | aligned | Honest builder analogue. |
| `embeddingModel(modelId)` | `Provider::google_vertex().embedding_model(modelId)` | aligned | Honest builder analogue. |
| `textEmbeddingModel(modelId)` | `Provider::google_vertex().text_embedding_model(modelId)` | aligned | Deprecated builder alias preserved. |
| `image(modelId)` | `Provider::google_vertex().image(modelId)` | aligned | Builder mirrors model selection; request settings remain request-owned. |
| `imageModel(modelId)` | `Provider::google_vertex().image_model(modelId)` | aligned | Same as `image(...)`. |
| `video(modelId)` | `Provider::google_vertex().video(modelId)` | aligned | Builder selects the default task-based video model id. |
| `videoModel(modelId)` | `Provider::google_vertex().video_model(modelId)` | aligned | Honest builder analogue. |
| `tools` | `provider_ext::google_vertex::tools::*` | aligned | Provider-defined tool factories are already exposed separately. |

## Grouped model-id matrix

| Upstream contract | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| `GoogleVertexModelId` | `provider_ext::google_vertex::chat::*` | aligned | Curated constants now cover the current audited package ids. |
| `GoogleVertexEmbeddingModelId` | `provider_ext::google_vertex::embedding::*` | aligned | Includes `text-embedding-005` and `gemini-embedding-2-preview`. |
| `GoogleVertexImageModelId` | `provider_ext::google_vertex::image::*` | aligned | Includes newer Imagen/Gemini image ids. |
| `GoogleVertexVideoModelId` | `provider_ext::google_vertex::video::*` | aligned | Covers the current audited Veo ids. |
| grouped sets | `provider_ext::google_vertex::model_sets::*` | aligned | Conventional Rust `ALL_*` family sets. |

## `GoogleVertexProviderSettings` field matrix

| Upstream field | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| `apiKey?: string` | `GoogleVertexProviderSettings.api_key` / `with_api_key(...)` | aligned | Rust keeps the same express-mode API-key concept and the builder path still honors `GOOGLE_VERTEX_API_KEY`. |
| `location?: string` | `GoogleVertexProviderSettings.location` / `with_location(...)` | aligned | Builder/config conversion preserves the enterprise-mode location field and environment fallback. |
| `project?: string` | `GoogleVertexProviderSettings.project` / `with_project(...)` | aligned | Builder/config conversion preserves the enterprise-mode project field and environment fallback. |
| `headers?: Resolvable<Record<string, string \| undefined>>` | `GoogleVertexProviderSettings.headers` / `with_headers(...)` / `with_header(...)` | partial | Rust currently supports static header maps only; async/lazy resolvable headers are not modeled yet. |
| `fetch?: FetchFunction` | `GoogleVertexProviderSettings.fetch` / `with_fetch(...)` | aligned | Rust-side analogue is `Arc<dyn HttpTransport>`. |
| `generateId?: () => string` | `GoogleVertexProviderSettings.generate_id` / `with_generate_id(...)` | aligned | Preserved through settings -> builder -> config and injected into the Vertex chat/stream Gemini transformer runtime, where it now owns stable tool/source IDs. |
| `baseURL?: string` | `GoogleVertexProviderSettings.base_url` / `with_base_url(...)` | aligned | Preserved as a provider-level base URL override. |
| Node `googleAuthOptions?: GoogleAuthOptions` | `GoogleVertexProviderSettings.token_provider` | partial | Rust does not mirror `google-auth-library`; it exposes a lower-level token-provider analogue instead. |

## Provider-option structure notes

| Upstream schema concept | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| embedding options object | `VertexEmbeddingOptions` / `GoogleVertexEmbeddingModelOptions` | aligned | Now has `new()` plus fluent builders for `outputDimensionality`, `taskType`, `title`, and `autoTruncate`. |
| public `GoogleVertexImageModelOptions` type | `GoogleVertexImageModelOptions = VertexImagenOptions` | partial | This stays intentionally Imagen-centric because that is the audited exported image-options type shape. Gemini image-only knobs such as `mediaResolution` / `imageConfig.imageSize` remain runtime-only raw `providerOptions["vertex"]` inputs instead of a stronger exported Rust struct. |
| `personGeneration` enum domain | `VertexPersonGeneration` | aligned | Shared Rust enum for image/video provider options; builders still remain string-compatible. |
| Imagen `safetySetting` enum domain | `VertexImagenSafetySetting` | aligned | Explicit Rust enum for the audited Imagen safety values. |
| Imagen `sampleImageSize` enum domain | `VertexImagenSampleImageSize` | aligned | Explicit Rust enum for `1K` / `2K`. |
| Imagen edit `mode` enum domain | `VertexImagenEditMode` | aligned | Explicit Rust enum for the audited edit-mode set. |
| Imagen edit `maskMode` enum domain | `VertexImagenMaskMode` | aligned | Explicit Rust enum for the audited mask-mode set. |
| Gemini image `responseModalities/imageConfig/seed/files` request lane | `vertex_gemini_image` + open `providerOptions["vertex"]` map | aligned | Vertex Gemini image now routes through `generateContent`; top-level `aspectRatio`/`seed` map into Gemini `generationConfig`, edit/variation use Gemini multi-part `contents[].parts[]`, `mask` / `n > 1` now follow the audited upstream rejection semantics, and the image-model lane intentionally keeps Gemini-specific open options scoped to `providerOptions["vertex"]` instead of inheriting the broader shared Google-language-model fallback aliases. |
| URL-backed edit/variation image inputs on Gemini/Vertex | provider-controlled executor opt-out | aligned | Shared image executor no longer forces URL materialization for providers that honestly support URL references; Vertex Gemini image now preserves URL inputs as `fileData.fileUri`. |

## Result-shape notes

| Upstream result detail | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| Imagen `providerMetadata.vertex.images[].revisedPrompt` | `GeneratedImage.revised_prompt` | aligned | Vertex Imagen response parsing already lifts `prediction.prompt` into the stable generated-image field. |
| Gemini image runtime route | `VertexGeminiImageStandard` | aligned | `gemini-* image` model ids now use `:generateContent` instead of the Imagen `:predict` path. |
| Video `providerMetadata['google-vertex'].videos[]` | `GenerateVideoResult.provider_metadata["vertex"].videos[]` | aligned | Public provider metadata now strips duplicated inline payload carriers (`bytesBase64Encoded` / raw bytes) and keeps only provider-owned metadata such as `gcsUri` / `mimeType`, matching the audited AI SDK result shape more honestly. |
| Task-query raw video payload carrier | `VideoTaskStatusResponse.metadata["_siumai"].generatedVideos` | Rust-only internal | The task-based Rust runtime still needs a raw per-video carrier to reconstruct final generated assets before building the stable `GenerateVideoResult`; this hidden lane is intentionally internal and removed from the public provider-metadata view. |

## Rust-only extras

| Rust-only item | Rationale |
| --- | --- |
| `image::IMAGEN_3_0_EDIT_001` | Provider-owned runtime extra kept for the dedicated Vertex edit path, even though the current AI SDK `GoogleVertexImageModelId` type does not expose it. |

## Structural notes

- `GoogleVertexProviderSettings` is now a dedicated builder-input struct again, which is a closer
  semantic match for upstream `createGoogleVertex(options)` than the earlier `GoogleVertexConfig`
  alias.
- The struct intentionally does not convert to `GoogleVertexConfig` without a model id. Instead it
  exposes `into_builder_for_model(...)` / `into_config_for_model(...)` so model selection stays
  explicit where Rust still differs from the callable TypeScript provider object.
- `generateId` is now an honest chat/stream runtime setting rather than a surface-only placeholder:
  Vertex injects a base `GeminiConfig` into the reused Gemini transformers, so custom stable IDs
  actually affect tool calls and grounding/source normalization.
- `GoogleVertexClient::supported_models()` now reuses the same curated model source consumed by the
  registry catalog, so provider-owned introspection and catalog output no longer drift.
- Node-only `googleAuthOptions` remain intentionally unmapped; the Rust runtime already exposes the
  explicit token-provider / ADC story instead of mirroring a TypeScript auth-library wrapper.
- Vertex image/video do not currently mint extra local IDs from `generateId`; those paths still
  primarily expose provider/service-owned operation or asset identifiers.
- The newer `@ai-sdk/google-vertex/xai` sub-entry is not part of this root Vertex package slice.
  It needs its own package-boundary audit because it is an OpenAI-compatible Vertex partner surface,
  not the same contract as native `xai` or generic Vertex MaaS.
