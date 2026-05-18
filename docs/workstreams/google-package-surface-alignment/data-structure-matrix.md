# Google Package Surface Alignment - Data Structure Matrix

Last updated: 2026-04-22

## Audited upstream root exports

Source of truth:

- `repo-ref/ai/packages/google/src/index.ts`
- `repo-ref/ai/packages/google/src/google-provider.ts`

## Export matrix

| Upstream export | Kind | Rust mapping | Status | Notes |
| --- | --- | --- | --- | --- |
| `GoogleLanguageModelOptions` | type | `provider_ext::google::GoogleLanguageModelOptions` | aligned | Provider-owned typed options. |
| `GoogleGenerativeAIProviderOptions` | deprecated type alias | `provider_ext::google::GoogleGenerativeAIProviderOptions` | aligned | Deprecated alias preserved. |
| `GoogleProviderMetadata` | type | `provider_ext::google::GoogleProviderMetadata` | aligned | Typed metadata helper surface. |
| `GoogleGenerativeAIProviderMetadata` | deprecated type alias | `provider_ext::google::GoogleGenerativeAIProviderMetadata` | aligned | Deprecated alias preserved. |
| `GoogleImageModelOptions` | type | `provider_ext::google::GoogleImageModelOptions` | aligned | Provider-owned typed options. |
| `GoogleGenerativeAIImageProviderOptions` | deprecated type alias | `provider_ext::google::GoogleGenerativeAIImageProviderOptions` | aligned | Deprecated alias preserved. |
| `GoogleEmbeddingModelOptions` | type | `provider_ext::google::GoogleEmbeddingModelOptions` | aligned | Provider-owned typed options. |
| `GoogleGenerativeAIEmbeddingProviderOptions` | deprecated type alias | `provider_ext::google::GoogleGenerativeAIEmbeddingProviderOptions` | aligned | Deprecated alias preserved. |
| `GoogleVideoModelOptions` | type | `provider_ext::google::GoogleVideoModelOptions` | aligned | Provider-owned typed options. |
| `GoogleGenerativeAIVideoProviderOptions` | deprecated type alias | `provider_ext::google::GoogleGenerativeAIVideoProviderOptions` | aligned | Deprecated alias preserved. |
| `GoogleVideoModelId` | type alias | `provider_ext::google::GoogleVideoModelId` | aligned | String alias on the Rust facade. |
| `GoogleGenerativeAIVideoModelId` | deprecated type alias | `provider_ext::google::GoogleGenerativeAIVideoModelId` | aligned | Deprecated alias preserved. |
| `GoogleFilesUploadOptions` | type | `provider_ext::google::GoogleFilesUploadOptions` | aligned | Shared upload helper lane. |
| `GoogleLanguageModelInteractionsOptions` | type | `provider_ext::google::GoogleLanguageModelInteractionsOptions` | aligned | Package-shape typed options for `google.interactions(...)`; runtime is a separate deferred lane. |
| `GoogleInteractionsModelId` | type alias | `provider_ext::google::GoogleInteractionsModelId` | aligned | String alias on the Rust facade, with grouped constants under `provider_ext::google::interactions`. |
| `GoogleInteractionsProviderMetadata` | type | `provider_ext::google::GoogleInteractionsProviderMetadata` | aligned | Typed Interactions metadata helper for interaction id, service tier, and signatures. |
| `GoogleInteractionsAgentName` | type | `provider_ext::google::GoogleInteractionsAgentName` | aligned | String alias on the Rust facade, with grouped constants under `provider_ext::google::agents`. |
| `createGoogle` | function | `provider_ext::google::create_google()` | aligned | Rust snake_case analogue. |
| `google` | function/default entry | `provider_ext::google::google()` | aligned | Rust builder entry analogue. |
| `createGoogleGenerativeAI` | deprecated function alias | `provider_ext::google::create_google_generative_ai()` | aligned | Deprecated builder alias preserved. |
| `GoogleProviderSettings` | type | `provider_ext::google::GoogleProviderSettings` | aligned | Dedicated provider-level settings struct with `api_key`, `base_url`, `headers`, `fetch`, and `into_builder*` conversions. |
| `GoogleGenerativeAIProviderSettings` | deprecated type alias | `provider_ext::google::GoogleGenerativeAIProviderSettings` | aligned | Deprecated alias preserved on top of `GoogleProviderSettings`. |
| `GoogleErrorData` | type | `provider_ext::google::GoogleErrorData` | aligned | Stable error envelope. |
| `VERSION` | constant | `provider_ext::google::VERSION` | aligned | Exposes provider crate package version. |
| `GoogleProvider` | callable interface | none | intentionally deferred | Rust does not have a structurally equivalent callable provider object. |
| `GoogleGenerativeAIProvider` | deprecated callable interface alias | none | intentionally deferred | Deferred with `GoogleProvider`. |

## Audited `GoogleProvider` member matrix

| Upstream member | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| call signature `google(modelId)` | none | intentionally deferred | Rust does not expose a callable provider object. |
| `languageModel(modelId)` | `Provider::google().language_model(modelId)` | aligned | Honest builder analogue. |
| `chat(modelId)` | `Provider::google().chat(modelId)` | aligned | Honest builder analogue. |
| `generativeAI(modelId)` | `Provider::google().generative_ai(modelId)` | aligned | Deprecated builder alias preserved. |
| `embedding(modelId)` | `Provider::google().embedding(modelId)` | aligned | Honest builder analogue. |
| `embeddingModel(modelId)` | `Provider::google().embedding_model(modelId)` | aligned | Honest builder analogue. |
| `textEmbedding(modelId)` | `Provider::google().text_embedding(modelId)` | aligned | Deprecated builder alias preserved. |
| `textEmbeddingModel(modelId)` | `Provider::google().text_embedding_model(modelId)` | aligned | Deprecated builder alias preserved. |
| `image(modelId, settings?)` | `Provider::google().image(modelId)` | partial | Builder mirrors model selection; request-level `GoogleImageModelOptions` still own runtime image settings. |
| `imageModel(modelId, settings?)` | `Provider::google().image_model(modelId)` | partial | Same as `image(...)`: model id is mirrored, settings stay request-owned. |
| `video(modelId)` | `Provider::google().video(modelId)` | aligned | Honest builder analogue for default video model selection. |
| `videoModel(modelId)` | `Provider::google().video_model(modelId)` | aligned | Honest builder analogue. |
| `files()` | `Provider::google().files()` | aligned | Returns the provider-owned `GeminiFiles` capability directly from the builder/config surface. |
| `interactions(modelIdOrAgent)` | `Provider::google().interactions(GoogleInteractionsModelInput::...)` | partial | Builder mirrors the package boundary and returns an explicit deferred handle. Real `/interactions` execution is a follow-on runtime lane. |
| `tools` | `provider_ext::google::tools::*` | aligned | Provider-defined tool factories are mirrored on the public facade. |

## Grouped model-id surface

| Upstream contract | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| `GoogleModelId` | `provider_ext::google::chat::*` | aligned | Public constants grouped for package-level diffing. |
| `GoogleEmbeddingModelId` | `provider_ext::google::embedding::*` | aligned | Public constants grouped for package-level diffing. |
| `GoogleImageModelId` | `provider_ext::google::image::*` | aligned | Includes Imagen and Gemini image-model ids. |
| `GoogleVideoModelId` | `provider_ext::google::video::*` | aligned | Includes current audited Veo ids. |
| `GoogleInteractionsModelId` | `provider_ext::google::interactions::*` | aligned | Includes current audited Interactions model ids. |
| `GoogleInteractionsAgentName` | `provider_ext::google::agents::*` | aligned | Includes current audited Interactions agent presets. |
| grouped sets | `provider_ext::google::model_sets::*` | aligned | Conventional Rust-first `ALL_*` and family aliases. |

## `GoogleProviderSettings` field matrix

| Upstream field | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| `baseURL?: string` | `GoogleProviderSettings.base_url` / `with_base_url(...)` | aligned | Provider-level base URL override is preserved on the Rust settings surface. |
| `apiKey?: string` | `GoogleProviderSettings.api_key` / `with_api_key(...)` | aligned | Rust now honors `GOOGLE_GENERATIVE_AI_API_KEY` first and keeps legacy `GEMINI_API_KEY` compatibility. |
| `headers?: Record<string, string \| undefined>` | `GoogleProviderSettings.headers` / `with_headers(...)` / `with_header(...)` | aligned | Rust uses a `HashMap<String, String>`; omitted keys cover the upstream `undefined` case without inventing nullable header values. |
| `fetch?: FetchFunction` | `GoogleProviderSettings.fetch` / `with_fetch(...)` | aligned | Rust-side analogue is `Arc<dyn HttpTransport>`. |
| `generateId?: () => string` | `GoogleProviderSettings.generate_id` / `with_generate_id(...)` | aligned | Public carrier uses `SharedIdGenerator`; Gemini response and streaming transformers consume it for provider-owned tool-call, tool-result, and source ids. |
| `name?: string` | `GoogleProviderSettings.name` / `with_name(...)` | aligned | Lowers to a provider-facing display label exposed by `GeminiClient::provider_name()` / `GeminiFiles::provider_name()`. It does not change canonical `provider_id`, `providerReference`, or `providerMetadata` keys. |

## Interactions provider-options matrix

| Upstream field | Rust mapping | Status | Notes |
| --- | --- | --- | --- |
| `previousInteractionId` | `GoogleLanguageModelInteractionsOptions.previous_interaction_id` / `with_previous_interaction_id(...)` | aligned | Preserved in the Google provider-options namespace. |
| `store` | `GoogleLanguageModelInteractionsOptions.store` / `with_store(...)` | aligned | Package-shape carrier only until runtime support lands. |
| `agent` | `GoogleLanguageModelInteractionsOptions.agent` / `with_agent(...)` | aligned | Pairs with `GoogleInteractionsAgentName` constants. |
| `agentConfig` | `GoogleLanguageModelInteractionsOptions.agent_config` / `GoogleInteractionsAgentConfig` | aligned | Supports dynamic and deep-research config shapes. |
| `thinkingLevel` / `thinkingSummaries` | `thinking_level` / `thinking_summaries` setters | aligned | String-backed to avoid baking upstream enum churn into the Rust ABI. |
| `responseFormat` | `GoogleInteractionsResponseFormatEntry` | aligned | Supports text, image, and audio entries in order. |
| `imageConfig` | `GoogleInteractionsImageConfig` | aligned | Deprecated upstream helper retained as a package-shape carrier. |
| `mediaResolution` | `media_resolution` / `with_media_resolution(...)` | aligned | Preserved in provider options. |
| `responseModalities` | `response_modalities` / `with_response_modalities(...)` | aligned | Preserved in provider options. |
| `serviceTier` | `service_tier` / `with_service_tier(...)` | aligned | Preserved in provider options and metadata. |
| `systemInstruction` | `system_instruction` / `with_system_instruction(...)` | aligned | Package-shape carrier only until runtime support lands. |
| `signature` | `signature` / `with_signature(...)` | aligned | Preserved for future round-tripping. |
| `interactionId` | `interaction_id` / `with_interaction_id(...)` | aligned | Preserved for future output/input compaction behavior. |
| `pollingTimeoutMs` | `polling_timeout_ms` / `with_polling_timeout_ms(...)` | aligned | Package-shape carrier only until dedicated polling runtime lands. |

## Structural notes

- `GoogleProviderSettings` is a provider-construction struct rather than a `GeminiConfig` alias
  because the audited upstream settings contract does not include model selection. Rust now keeps
  that same separation by converting settings into `GeminiBuilder` / `GeminiConfig` only when a
  model is selected.
- `generateId` is now aligned by threading a `SharedIdGenerator` through
  `GoogleProviderSettings` -> `GeminiBuilder` -> `GeminiConfig` and consuming it only at the
  provider-owned ID-allocation points in the Gemini response/streaming transformers.
- `name` is now aligned as a provider-facing display label only. The Google package path defaults
  that label to `google.generative-ai`, while the native Gemini path keeps `gemini` unless callers
  override it explicitly.
- `GoogleProvider` is intentionally not mirrored because a TypeScript callable provider object does
  not have a clean Rust analogue. Mirroring the name without equivalent behavior would create a
  misleading surface.
- Deprecated upstream aliases are mirrored where Rust already has a direct structural equivalent:
  type alias, constant alias, or builder helper alias.
- The grouped `chat` / `embedding` / `image` / `video` constants are a Rust-first audit aid layered
  on top of the provider-owned Gemini implementation; they do not imply that Rust now exposes the
  TypeScript callable provider object itself.
- The grouped `interactions` / `agents` constants are also audit aids. They lock the package-visible
  Interactions boundary while keeping execution separate from the existing Gemini
  `:generateContent` runtime.
