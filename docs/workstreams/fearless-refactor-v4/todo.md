# Fearless Refactor V4 - TODO







Last updated: 2026-03-19







Status legend:







- `[ ]` not started



- `[~]` in progress



- `[x]` done



- `[-]` intentionally deferred







## Workstream goals







- Finish the transition from generic-client-centric design to family-model-centric design.



- Keep builder ergonomics while removing builder-only architecture.



- Preserve Rust-first naming and public API shape.



- Isolate provider-specific complexity in provider and protocol crates.







## Track A - Architecture decisions







- [x] Decide that builder APIs stay, but only as ergonomic wrappers.



- [x] Decide that family model traits become the architectural center.



- [x] Decide that `LlmClient` becomes an internal/compatibility abstraction.



- [x] Write an ADR describing the builder retention policy.



- [x] Write an ADR describing the family-model-first trait policy.



- [x] Write an ADR describing the `LlmClient` demotion policy.







## Track B - Family model contracts







- [x] Add a shared model metadata trait spike (`ModelMetadata` + version marker).



- [~] Finalize shared model metadata trait.



- [x] Add a minimal `LanguageModel` trait spike on top of the text family.



- [~] Finalize `LanguageModel` trait.



- [~] Finalize `EmbeddingModel` trait.



- [~] Finalize `ImageModel` trait.



- [~] Finalize `RerankingModel` trait.



- [~] Finalize `SpeechModel` trait.
  - `SpeechModel` now exists as a metadata-bearing family trait over `SpeechModelV3`, with no-network adapter coverage in `siumai-core`.



- [~] Finalize `TranscriptionModel` trait.
  - `TranscriptionModel` now exists as a metadata-bearing family trait over `TranscriptionModelV3`, with no-network adapter coverage in `siumai-core`.



- [x] Add a migration adapter path from current chat capability to `LanguageModel`.



- [~] Add no-network unit tests for all trait adapters.

  - `RerankingModel` now exists as a metadata-bearing family trait over `RerankModelV3`, with no-network adapter coverage in `siumai-core`.
  - `SpeechModel` and `TranscriptionModel` now also have metadata-bearing family traits plus no-network adapter coverage in `siumai-core`.
  - `siumai-core` now also covers injected-transport execution for JSON, JSON-bytes, multipart JSON, and multipart-bytes request helpers, including single-401 retry parity for OpenAI-style upload flows.







## Track C - Registry redesign







- [x] Add a parallel `ProviderFactory` text-family-returning interface for incremental migration.



- [~] Redesign `ProviderFactory` to return family model objects instead of generic clients.



- [~] Update build context flow to remain shared across all family model construction.

  - `BuildContext` now carries a neutral `google_token_provider` alongside the legacy `gemini_token_provider` compatibility alias, and registry factories resolve through `resolved_google_token_provider()` so Google-family auth no longer depends on Gemini-specific naming.
  - Gemini typed metadata now also models `thoughtSignature` directly, and Google/Vertex fixture + surface-guard tests use `GeminiChatResponseExt` to validate namespace aliasing instead of only raw `provider_metadata["google"|"vertex"]` traversal.
  - Gemini now also exposes `GeminiContentPartExt` for part-level metadata, so Google fixture assertions for text/reasoning/tool-call `thoughtSignature` no longer reopen raw nested provider maps by hand; `google-vertex` still keeps its own raw namespace-key assertions in the dedicated Vertex stream fixture tests, but the plain Vertex wrapper now also exposes a provider-owned typed metadata surface for final response/content-part reads under `provider_ext::google_vertex::metadata`.



- [x] Add `LanguageModelHandle` metadata conformance so the handle can act like a family model object.



- [~] Redesign `LanguageModelHandle` to implement the final language model trait directly.



- [~] Redesign `EmbeddingModelHandle` to implement the final embedding model trait directly.



- [~] Redesign `ImageModelHandle` to implement the final image model trait directly.



- [~] Redesign `SpeechModelHandle` to implement the final speech model trait directly.
  - Registry speech handles now carry provider/model metadata and delegate through `speech_model_family_with_ctx`, so native family factories can bypass legacy generic-client downcasts.
  - [x] Add shared audio-family default-model propagation so speech requests can inherit provider-config / registry model ids without requiring request-local `TtsRequest.model` duplication.
    - Built-in provider-owned audio clients now backfill missing `TtsRequest.model` from config-level defaults, and registry speech handles inject the same fallback before calling native family models or provider extras.
    - Public-path no-network parity now locks that behavior on the native OpenAI, Groq, and MiniMaxi speech routes instead of only package-local tests.
    - OpenAI provider-specific speech SSE now also has no-network builder/provider/config/registry-construction parity over the final `/audio/speech` request body, including `stream_format: "sse"` plus default-model backfill when the request omits `model`.
    - OpenAI provider-owned generic `SpeechExtras::tts_stream(...)` delegation is now wired end-to-end as well, so `as_speech_extras()` and registry speech handles no longer advertise a stream extra that immediately falls back to `UnsupportedOperation`.
    - Groq now intentionally keeps generic `SpeechExtras` out of the provider-owned public surface until it ships a real provider-owned streaming/voice-extras implementation, so wrapper/config clients stop advertising a speech-extra handle they cannot actually satisfy.
    - xAI and MiniMaxi now also keep generic `SpeechExtras` out of their provider-owned public surface until they ship real provider-owned speech extras, so wrapper/config clients stop advertising a TTS-extra handle when only the non-stream speech family is implemented.



- [~] Redesign `TranscriptionModelHandle` to implement the final transcription model trait directly.
  - Registry transcription handles now carry provider/model metadata and delegate through `transcription_model_family_with_ctx`, so native family factories can bypass legacy generic-client downcasts.
  - [x] Align the same default-model propagation story on transcription-family requests instead of leaving model fallback provider-specific.
    - Built-in provider-owned audio clients and registry transcription handles now backfill missing `SttRequest.model` / translation request models from the configured family model id before execution.
    - Public-path no-network parity now locks that behavior on the native OpenAI and Groq transcription routes, plus OpenAI translation extras.
    - OpenAI provider-specific transcription SSE now also has no-network builder/provider/config/registry-construction parity over the final `/audio/transcriptions` multipart body, including forced `stream=true` and default-model backfill when the request omits `model`.
    - Core raw-response streaming helpers now also honor injected custom transports on both JSON and multipart paths, so provider-owned SSE routes no longer need local mock servers just to lock final request shaping.
    - OpenAI provider-owned generic `TranscriptionExtras::stt_stream(...)` delegation now also resolves through the same SSE path, so registry transcription handles and `as_transcription_extras()` no longer expose a stream extra that immediately devolves to the legacy default-unsupported branch.
    - Groq now intentionally keeps generic `TranscriptionExtras` out of the provider-owned public surface as well, so callers do not get a false-positive extras handle for unavailable stream/translation/language-listing paths; registry transcription handles still fail fast with `UnsupportedOperation` and no network when those extras are requested.



- [~] Redesign `RerankingModelHandle` to implement the final reranking model trait directly.

  - Registry rerank handles now carry provider/model metadata and delegate through `reranking_model_family_with_ctx`, so native family factories can bypass legacy generic-client downcasts.
  - Registry handles now also inherit registry-level provider build overrides (`api_key`, `base_url`, `http_client`, `http_transport`, `http_config`) instead of only interceptors / retry / HTTP config, and that build-override layer now also supports per-provider precedence through `ProviderBuildOverrides`, with provider-scoped `http_config` merging over registry-global defaults instead of replacing them wholesale, so handle-driven provider construction no longer needs ad hoc manual `BuildContext` wiring just to reach real provider-owned no-network contract coverage.
  - That `http_config` merge lane is now also pinned at the entry layer with a field-level `merged_with()` unit test, so header/timeout/user-agent/proxy precedence does not rely only on Anthropic-on-Vertex end-to-end coverage.



- [x] Preserve caching, TTL, middleware, and provider-id/model-id override behavior.

  - No-network handle tests now lock cache-key convergence on middleware-overridden `provider_id` / `model_id`, so override rewrites no longer risk fragmenting the shared LRU cache or silently bypassing the intended provider route.
  - TTL regression coverage now also runs through the middleware-override path directly, confirming that overridden model routing still reuses the cached client within TTL and rebuilds exactly once after expiration.



- [x] Add no-network tests for handle caching and middleware invariants.

  - `middleware_override_test` now covers overridden-model cache-key reuse, overridden-provider routing/cache convergence, and TTL expiration on the same no-network handle path instead of relying only on generic registry-entry unit tests.







## Track D - Builder convergence







- [x] Define provider config structs as the canonical construction inputs.



- [~] Make provider builders emit those config structs before construction.

  - OpenAI, Azure, Anthropic, Gemini, OpenAI-compatible, xAI, Groq, DeepSeek, Ollama, Google Vertex, Anthropic-on-Vertex, Bedrock, Cohere, TogetherAI, and MiniMaxi builder paths now emit provider-owned config before final client construction; the remaining convergence work is now limited to secondary compat-only packages rather than missing provider-owned builders on the current native/focused provider set.



- [~] Ensure no feature can be configured only through builder APIs.

  - Unified builder Google-family auth aliases now exist: `with_google_token_provider(...)`, `with_google_adc()`, `with_vertex_token_provider(...)`, and `with_vertex_adc()` are the preferred names, while Gemini-branded methods remain as compatibility shims.



- [x] Add parity tests between builder-based and config-based construction.



- [~] Audit builder-only fluent options and add config-first equivalents for major providers.

  - OpenAI-compatible config-first ergonomics are now converged on their major/canonical lanes; remaining audit is narrower and now mostly about leftover provider-specific builder-only helpers rather than routine HTTP setup.
  - OpenAI-compatible config-first ergonomics now also mirror the builder-owned HTTP convenience lanes directly on `OpenAiCompatibleConfig`: timeout / connect-timeout, SSE compression toggle, user-agent / proxy, HTTP headers, and single-interceptor composition all no longer require dropping to a raw `HttpConfig` mutation path, while `OpenAiCompatibleClient::with_http_client(...)` remains the config-first equivalent of builder-side client injection.
  - OpenAI and Anthropic config-first ergonomics now also mirror their provider builders’ common HTTP convenience lanes directly on `OpenAiConfig` / `AnthropicConfig`: timeout / connect-timeout, SSE compression toggle, and single-interceptor composition no longer require routing through a bulk `HttpConfig` mutation path for those routine cases, and package-local config tests now pin those helpers.
  - Azure config-first ergonomics now also mirror its provider builder’s common HTTP convenience lanes directly on `AzureOpenAiConfig`: timeout / connect-timeout and single-interceptor composition now have canonical config-first helpers instead of forcing direct `HttpConfig` mutation for those routine cases, and package-local config tests now pin that surface.
  - Google Vertex config-first ergonomics now also mirror their provider builders’ common HTTP convenience lanes directly on both `GoogleVertexConfig` and `VertexAnthropicConfig`: timeout / connect-timeout, SSE compression toggle, and single-interceptor composition now have canonical config-first helpers instead of forcing raw `HttpConfig` edits for those routine cases, and package-local config tests now pin both configs independently.
  - MiniMaxi config-first ergonomics now also mirror the provider builder’s common HTTP convenience lanes directly on `MinimaxiConfig`: timeout / connect-timeout, SSE compression toggle, and single-interceptor composition now have canonical config-first helpers instead of forcing direct `HttpConfig` mutation, and package-local config tests now pin that surface.
  - Gemini’s canonical config-first HTTP surface is now tighter too: protocol-owned `GeminiConfig` keeps the legacy `with_timeout(u64)` seconds helper for backward compatibility, but new config-first code can now use `with_http_timeout(...)`, `with_connect_timeout(...)`, `with_http_stream_disable_compression(...)`, and `with_http_interceptor(...)` without dropping to raw `HttpConfig`, and protocol-level config tests pin that compatibility bridge.
  - Groq config-first ergonomics now also mirror the wrapper builder’s common HTTP lanes directly on `GroqConfig`: timeout / connect-timeout, SSE compression toggle, and single-interceptor composition no longer require routing through a bulk `HttpConfig` mutation path on the canonical config-first surface, and a feature-scoped integration test now pins builder/config convergence on that HTTP lane.
  - xAI config-first ergonomics now also mirror the wrapper builder’s common HTTP lanes directly on `XaiConfig`: timeout / connect-timeout, SSE compression toggle, and single-interceptor composition no longer require routing through a bulk `HttpConfig` mutation path on the canonical config-first surface, and a feature-scoped integration test now pins builder/config convergence on that HTTP lane.
  - DeepSeek config-first ergonomics now also mirror the wrapper builder’s common HTTP lanes directly on `DeepSeekConfig`: timeout / connect-timeout, SSE compression toggle, and single-interceptor composition no longer require routing through a bulk `HttpConfig` mutation path on the canonical config-first surface, and a feature-scoped integration test now pins builder/config convergence on that HTTP lane.
  - Ollama config-first ergonomics now also mirror the provider builder’s common HTTP lanes on `OllamaConfigBuilder`: timeout / connect-timeout, SSE compression toggle, and single-interceptor composition no longer require dropping straight to a bulk `HttpConfig` snapshot, and the package-local provider tests now pin builder/config convergence on that lane.
  - Cohere and TogetherAI config-first ergonomics now also mirror their rerank builders’ common HTTP convenience lanes: `CohereConfig` / `TogetherAiConfig` each own `with_timeout(...)`, `with_connect_timeout(...)`, and `with_http_interceptor(...)`, and package-local parity tests now lock builder `into_config()` against those canonical config-first surfaces.



  - Ollama config builder now exposes common-param convenience and reasoning alias; xAI now owns `XaiConfig` / `XaiClient` entry types while still reusing the shared compat runtime, and DeepSeek now owns `DeepSeekConfig` / `DeepSeekClient` entry types on top of the shared runtime.
  - Groq now also owns provider-level default request helpers on both config-first and builder surfaces: `GroqConfig` / `GroqBuilder` can carry `logprobs`, `top_logprobs`, `service_tier`, `reasoning_effort`, and `reasoning_format` defaults without forcing callers back to per-request `provider_options`.
  - MiniMaxi now also owns provider-level default chat option helpers on both config-first and builder surfaces: `MinimaxiConfig` / `MinimaxiBuilder` can persist thinking-mode and structured-output defaults through `provider_options_map`, the shared `Siumai::builder().minimaxi()` facade now also exposes matching namespaced default-option helpers (`with_minimaxi_options`, `with_minimaxi_reasoning_budget`, `with_minimaxi_json_object`, ...), and request-local `with_minimaxi_options(...)` still overrides those defaults at execution time.
  - MiniMaxi’s default typed chat-option lane now also has focused no-network public-path parity on both `chat_request` and `chat_stream_request`: `Siumai::builder().minimaxi()`, `Provider::minimaxi()`, and config-first `MinimaxiClient::from_config(...)` now pin the same final `/v1/messages` body for builder/config defaults (`thinking_mode`, `response_format`) instead of relying only on lower provider-local merge tests.
  - xAI now also owns provider-level default request helpers on both config-first and builder surfaces: `XaiConfig` / `XaiBuilder` can persist `reasoning_effort` and default web-search parameters through the same provider-owned adapter path as request-level `XaiOptions`, while request-local provider options still override those defaults at execution time.
  - Anthropic builder/config parity is now tighter too: `AnthropicBuilder` now mirrors the provider-owned `AnthropicConfig` helper surface for bulk params, cache/thinking/system metadata aliases, `stream`, and beta-feature defaults, so builder-first Anthropic construction no longer trails config-first on provider-specific request shaping.
  - Gemini builder/config parity is now tighter too: `GeminiBuilder` now mirrors the provider-owned `GeminiConfig` helper surface for `with_top_k`, `with_candidate_count`, JSON-schema / reasoning aliases, and base `GenerationConfig` injection, so builder-first Gemini construction no longer trails config-first on generation-config shaping.
  - Google Vertex config-first ergonomics are now less lossy too: `GoogleVertexConfig` now owns `common_params`, and `GoogleVertexBuilder` now forwards temperature / max_tokens / top_p / stop-sequence defaults into that config instead of only preserving them on the post-build client mutation path.
  - Azure config-first ergonomics are now less builder-only too: `AzureOpenAiConfig` now mirrors the provider-owned builder shorthands for family model aliases, API-version / deployment-URL toggles, and chat-mode aliases, so config-first Azure setup no longer has to crack open `AzureUrlConfig` or manually map builder-only convenience methods just to reach the canonical provider surface.
  - OpenAI builder/config parity is now tighter too: `OpenAiBuilder` now mirrors the provider-owned `OpenAiConfig` naming surface for common params, Responses-API defaults, and provider-options defaults, so builder-first OpenAI setup can use the same `with_*` vocabulary as config-first code without changing semantics.
  - OpenAI now also exposes a typed default-options lane on the provider-owned and shared builder surfaces: `OpenAiBuilder`, `OpenAiConfig`, and the shared `Siumai::builder().openai()` path all accept `with_openai_options(OpenAiOptions)`, so provider-owned typed defaults no longer require raw `serde_json` for the text family.
  - That OpenAI default-options lane is now hardened too: `OpenAiOptions` / `ResponsesApiConfig` serialize sparsely, `OpenAiSpec` merges duplicate canonical `responsesApi` / `responses_api` keys instead of letting later aliases erase nested defaults, and the provider-owned chat executor now defers runtime transformer selection until after middleware transforms so shared-builder default `responses_api` settings (for example `previous_response_id` / `reasoning_summary`) reach the final `/responses` body instead of being lost behind pre-middleware route selection.
  - Ollama builder/config parity is now tighter too: `OllamaBuilder` now mirrors the canonical `OllamaConfig` surface for bulk `common_params`, bulk `OllamaParams`, full `HttpConfig`, provider-native `think`, and extra model middlewares, so builder-first Ollama setup no longer trails config-first on provider defaults or middleware composition.
  - The top-level `siumai` facade now also re-exports `provider_ext::ollama::OllamaParams`, and `Provider::ollama()` has focused parity coverage for `with_common_params(...)`, `with_ollama_params(...)`, `with_http_config(...)`, and extra model middleware composition, so the new Ollama config-first helper surface is reachable without dropping to the provider crate directly.
  - Ollama public-path parity now also reaches the embedding family on the real registry story: request-level `OllamaEmbeddingOptions` are locked across `Siumai::builder().ollama()`, `Provider::ollama()`, config-first `OllamaClient`, and `registry.embedding_model("ollama:...")`, while provider-specific registry build overrides are now pinned on the embedding handle too instead of only the chat handle.
  - Azure public-path parity now also reaches the native non-text families that were already implemented under the provider crate: embedding and image-generation requests are now locked across `Siumai::builder().azure()`, `Provider::azure()`, config-first `AzureOpenAiClient`, and the public registry facade, while provider-specific registry build overrides are now pinned on the Azure image handle instead of leaving non-text Azure coverage implied by text-only `responses` tests.
  - Azure public-path parity now also reaches the native audio families: TTS and STT requests are now locked across `Siumai::builder().azure()`, `Provider::azure()`, config-first `AzureOpenAiClient`, and the public registry facade, provider-specific registry build overrides are now pinned on both Azure speech/transcription handles, and the provider-owned audio client now backfills missing request models from the configured Azure deployment id instead of silently falling back to generic `tts-1` / `whisper-1`.
  - Unified Azure builder routing no longer drops Azure-specific construction defaults on the registry-backed public path either: `Siumai::builder().azure()` / `Provider::azure()` now forward `api_version(...)`, `deployment_based_urls(...)`, full `url_config(...)`, and `provider_metadata_key(...)` into `AzureOpenAiProviderFactory`, so deployment-based URLs and Responses metadata namespaces stay aligned across builder/provider/config-first/registry construction instead of reverting to the factory defaults on the unified facade.
  - OpenAI public-path parity now also locks provider-specific registry build overrides on the speech and transcription families themselves, not only chat/text handles: `registry.speech_model("openai:...")` and `registry.transcription_model("openai:...")` are now pinned to the provider override transport/base URL/API key path on the real family handles.
  - OpenAI public-path parity now also reaches the image family on the native provider story itself: image-generation requests are locked across `Siumai::builder().openai()`, `Provider::openai()`, config-first `OpenAiClient`, and `registry.image_model("openai:...")`, while provider-specific registry build overrides are now pinned on the image handle instead of relying only on OpenAI-compatible vendors for image-family coverage, and the builder-default `responsesApi` flag no longer leaks into non-chat families such as image/audio/embedding/rerank.
  - Google Vertex public-path parity now also pins provider-specific registry build overrides on the native Imagen family itself: both `registry.image_model("vertex:imagen-4.0-generate-001")` image generation and `registry.image_model("vertex:imagen-3.0-edit-001")` image edit now route through the provider override API key / base URL / transport lane instead of relying on chat-only mixed-registry coverage to imply Imagen handle precedence.



  - Google Vertex config-first ergonomics now also expose explicit constructor paths (`new` / `express` / `enterprise`), while `GoogleVertexBuilder::into_config()` converges builder resolution onto the same provider-owned config surface.



  - Builder availability now follows provider-owned feature gates; `SiumaiBuilder::new().deepseek()` is compiled only with the `deepseek` feature instead of leaking through the generic OpenAI-compatible builder surface.



- [x] Update docs to rank construction modes clearly:



  - registry-first



  - config-first



  - builder convenience







## Track E - Audio cleanup







- [x] Make `SpeechModel` the explicit TTS contract.
  - `siumai::speech` and `siumai::prelude::unified` now expose `SpeechModel` alongside `SpeechModelV3` on the public surface.
  - The public `speech::synthesize(...)` helper now requires the metadata-bearing `SpeechModel` family trait instead of the legacy `SpeechModelV3` compatibility bound.



- [x] Make `TranscriptionModel` the explicit STT contract.
  - `siumai::transcription` and `siumai::prelude::unified` now expose `TranscriptionModel` alongside `TranscriptionModelV3` on the public surface.
  - The public `transcription::transcribe(...)` helper now requires the metadata-bearing `TranscriptionModel` family trait instead of the legacy `TranscriptionModelV3` compatibility bound.



- [x] Wire the OpenAI-compatible runtime to an audio-capable base.
  - The shared compat adapter/spec/client stack now understands `audio` / `speech` / `transcription` capability enrollment, exposes an `AudioCapability`-backed speech/transcription family foundation, and has no-network wiring coverage.
  - Built-in runtime provider capability declarations for speech/transcription are now enabled for `together` and `siliconflow`; remaining vendors stay intentionally pending until each surface is verified independently.
  - The generic OpenAI-compatible factory now also guards advertised non-text family paths before transport instead of relying on late client downcasts: `embedding` and `image_generation` are checked at factory entry, Together now has a native image-family contract anchor, and OpenRouter remains an explicit no-image negative path.
  - `openrouter` remains pending because its current audio story is routed through `chat/completions` multimodal request/response semantics instead of OpenAI `/audio/*` endpoints.
  - That deferred compat audio boundary is now explicit on the real public surface too: OpenRouter builder/provider/config-first `text_to_speech` / `speech_to_text` plus registry speech/transcription handles now fail fast with `UnsupportedOperation`, so the missing `/audio/*` enrollment is a documented contract boundary rather than an implicit gap.
  - Perplexity's vendor-view scope is now explicit as well: it stays text-only on the public surface, and factory/public-path tests now pin `embedding` / `image_generation` / `rerank` / speech / transcription as unsupported before transport instead of leaving those family paths to generic compat fallbacks.
  - The remaining compat preset catalog now also has an explicit audio-capability boundary: preset guards and the shared registry audio-boundary test lock `audio == speech || transcription` across all built-in compat providers, while every preset without declared speech/transcription support now rejects `speech_model(...)` / `transcription_model(...)` before any transport call.
  - `xai` remains outside the shared compat audio family because its public docs currently expose `/v1/tts` plus Voice Agent realtime transcription events, but still do not publish a standalone REST transcription endpoint; xAI now has a provider-owned speech-family path and explicit no-network rejection coverage for transcription so this scope boundary is intentional rather than accidental.
  - `fireworks` now joins the shared compat audio story in a transcription-only mode: Siumai uses the documented dedicated audio base (`https://audio.fireworks.ai/v1`) for `/audio/transcriptions`, while speech/TTS remains intentionally unsupported.
  - Core multipart execution now also honors injected `HttpTransport`, so OpenAI-compatible STT request-body parity no longer depends on reqwest/mock servers alone; Fireworks now has both explicit-base override coverage and direct no-network transport-boundary capture coverage.
  - `together` and `siliconflow` now both have direct no-network request-body coverage on the shared compat audio surface: JSON speech requests are locked on `/audio/speech`, and multipart transcription uploads are locked on `/audio/transcriptions`.
  - Top-level public-path parity now also covers the full compat audio split for `together`, `siliconflow`, and the transcription-only `fireworks` preset: Together and SiliconFlow lock TTS + STT across builder/provider/config/registry, while Fireworks now locks STT on the registry transcription handle and keeps TTS rejected across builder/provider/config/registry.
  - Compat non-text public parity is now less fragmented: OpenRouter already anchors registry embedding request options on the same `/embeddings` boundary as config-first construction, Together/SiliconFlow registry image handles now match config-first `/images/generations` request shaping, and Jina/VoyageAI registry rerank handles now match config-first `/rerank` request bodies instead of leaving rerank aligned only on builder/provider/config paths.
  - The outward-facing compat rerank story is now runnable too: `siliconflow-rerank`, `jina-rerank`, and `voyageai-rerank` cover the currently advertised shared-runtime rerank presets instead of leaving them discoverable only through parity tests and the registry-first example.
  - Typed vendor-view response metadata is now less implicit too: OpenRouter public-path parity now locks `provider_metadata["openrouter"]` on both chat responses and stream-end responses across builder/provider/config/registry, including `logprobs` extraction under the vendor namespace instead of leaking back to a generic `openai_compatible` key; `provider_ext::openrouter::OpenRouterChatResponseExt` plus `OpenRouterSourceExt` / `OpenRouterContentPartExt` now expose that same metadata through vendor-owned typed helpers instead of forcing callers back to raw provider maps, and Perplexity now also locks registry stream-end metadata against the config-first path.
  - Compile-surface guards now pin that OpenRouter helper set too: `public_surface_imports_test` locks `OpenRouterMetadata`, `OpenRouterSource`, `OpenRouterSourceMetadata`, `OpenRouterContentPartMetadata`, and the response/source/content-part extension traits on the final `siumai::provider_ext::openrouter` export path.
  - The compat rerank split is now explicit at both factory and public layers: the generic OpenAI-compatible factory guards `rerank` before transport, Jina and VoyageAI keep native rerank family paths, and OpenRouter is now pinned as a no-rerank negative path on contract tests, preset guards, and the top-level registry handle.
  - Focused compat presets no longer over-advertise the text surface: Jina and VoyageAI now drop default `chat` / `streaming` capability declarations at adapter/factory/public-client layers, reject the generic factory `language_model_with_ctx(...)` text path, and fail before transport on chat requests, while mixed presets such as Infini still retain chat/streaming because their capability set includes real chat signals (`tools` / `vision`) and now also lock both `registry.language_model("infini:...")` chat and chat-stream parity on the real `/chat/completions` boundary.
  - The registry text-handle boundary is now fully closed for focused non-chat compat presets: `registry.language_model("jina:...")` and `registry.language_model("voyageai:...")` now reject construction itself with `UnsupportedOperation`, so these presets no longer masquerade as text/chat entries on the public registry path at all; callers must use `embedding_model(...)` or `reranking_model(...)`.
  - Preferred public entry points are now covered too: `Siumai::builder()`, `Provider::openai()`, and config-first construction all preserve Together TTS, SiliconFlow STT, Fireworks transcription-only routing, and xAI provider-owned TTS request routing without falling back to the wrong family path.
  - That intentional Fireworks boundary is now pinned on the real public path too: builder/provider/config-first `text_to_speech` calls fail with `UnsupportedOperation`, expose no speech capability handle, and emit no transport request, so the transcription-only split is guarded instead of living only in preset capability tables.
  - Provider-owned audio wrappers are now tighter too: Groq has top-level no-network parity for both TTS and STT across `Siumai::builder()`, `Provider::groq()`, and config-first `GroqClient`, while xAI now also has an explicit public-path no-network rejection guard for standalone transcription plus capability-surface assertions (`as_audio_capability()` / `as_speech_capability()` present, `as_transcription_capability()` absent) so the speech-only boundary is tested at the final facade layer.



- [x] Move streaming and provider-specific extras into extension traits where needed.
  - `LlmClient` now exposes `as_speech_extras()` / `as_transcription_extras()` alongside `as_image_extras()`, and both `ClientWrapper` plus registry-backed `Siumai` now delegate those non-unified audio extras instead of forcing callers back through the broad `AudioCapability` bucket.
  - No-network public-surface tests now lock the accessor boundary too: native OpenAI exposes both extras, xAI exposes only speech extras, and the `fireworks` compat preset exposes only transcription extras.



- [x] Reduce reliance on broad `AudioCapability` in recommended code paths.
  - Narrow speech/transcription extension accessors now exist on the public client surface, fixture/mock tests now call `SpeechCapability` / `TranscriptionCapability` plus `TranscriptionExtras` directly where appropriate, and the legacy `audio_capability_test` itself now documents the narrow family surface while leaving `AudioCapability` explicitly compatibility-only.
  - Core trait/module docs and compile-surface guards now reflect the same recommendation: `siumai-core` documentation lists `SpeechCapability` / `TranscriptionCapability` as the primary audio entry points, while `public_surface_imports_test` locks `SpeechExtras` / `TranscriptionExtras` exports and the `LlmClient` accessor signatures.
  - Registry audio handles now also behave more like the text-family path internally: `SpeechModelHandle` / `TranscriptionModelHandle` cache family-model objects with the same registry TTL/LRU policy instead of rebuilding provider family models on every call.
  - Registry transcription extras now have a real public-path parity anchor too: OpenAI `TranscriptionExtras::audio_translate` matches builder/provider/config/registry multipart `/audio/translations` requests, and the registry handle delegates translation/streaming/language-listing back through the provider-owned transcription client instead of growing a second request-shaping path inside the handle.
  - The first non-OpenAI extras boundary is now pinned too: Groq still exposes the speech/transcription family surface, but `TranscriptionExtras::audio_translate` remains intentionally unsupported across builder/provider/config/registry and fails before transport, so we are no longer implicitly promising translation just because the provider has STT.
  - Registry speech extras now use the same delegation pattern too: `SpeechModelHandle` routes streaming TTS / voice-listing back through the provider-owned speech client instead of freezing today’s default-unsupported behavior into the handle, and xAI now has a public-path negative guard that keeps `SpeechExtras::tts_stream` unsupported across builder/provider/config/registry without emitting any request.
  - Provider-owned/native audio factory overrides are now guarded explicitly too: OpenAI, Azure, OpenAI-compatible, and Groq lock native speech/transcription family overrides in source-level contract tests, while xAI and MiniMaxi lock their supported speech-family override path the same way.
  - Real public-path audio parity now also covers registry handles for representative provider-owned and compat-audio vendors: Groq TTS/STT plus Together and SiliconFlow TTS/STT all match the existing builder/provider/config request shapes when called through `registry.speech_model()` / `registry.transcription_model()`.
  - Registry negative-path audio boundaries are now executable too: `fireworks` registry speech requests fail before transport, and `xai` registry transcription requests fail before transport, matching the existing builder/provider/config-only boundary tests.
- [~] Normalize provider package tiers and promotion rules using `provider-package-alignment.md`, so native packages, focused packages, and OpenAI-compatible vendor presets stop drifting toward one-size-fits-all symmetry.
  - Vertex audit result: promote a narrow provider-owned metadata facade on `provider_ext::google_vertex`, but keep it strictly bound to `provider_metadata["vertex"]`; do not widen it into shared Google/Gemini aliasing, and keep stream custom events / resource surfaces raw or deferred until Vertex-specific contracts stabilize further.
  - That boundary is now re-validated by compile/test work too: Vertex fixture and public-surface checks now cover both the new typed metadata facade (`VertexMetadata`, `VertexChatResponseExt`, `VertexContentPartExt`) and the intentionally raw stream-event namespace assertions, while Google/Gemini typed metadata keeps carrying the shared Google response-story escape hatch.
  - Bedrock audit result: keep `provider_ext::bedrock` request-helper-focused for now; do not add typed `metadata` / `resources` until Bedrock-specific response metadata grows beyond the current narrow internal fields.
  - Compat preset audit result: keep `provider_ext::{openrouter,perplexity}` as typed vendor views over the shared compat runtime, and keep `siliconflow` / `together` / `fireworks` on the compat preset path until they accumulate clearly provider-owned metadata/resources/auth semantics worth promoting.
  - Compat audio extras are now part of that same promotion boundary too: `siliconflow`, `together`, and `fireworks` may expose transcription-family accessors through the shared compat runtime, but `audio_translate` is now explicitly locked as unsupported across builder/provider/config-first/registry paths, so “has STT” no longer gets conflated with “owns a translation extras package surface”.
  - Compat vendor-view policy is now written down in `compat-vendor-view-contract.md`, defining runtime ownership, capability-source, metadata, docs, and test invariants for `openrouter` / `perplexity`-style surfaces.
  - Compat vendor-view metadata boundary is now explicit too: `OpenRouterChatResponseExt` / `PerplexityChatResponseExt` only bind to `provider_metadata["openrouter"]` / `provider_metadata["perplexity"]`, while generic compat presets remain raw-map-only until stronger provider evidence exists.
  - Public docs/examples should now reflect the same tiering: provider-owned packages use config-first stories, focused packages stay narrow, and OpenAI-compatible vendor views/presets stay under the compat narrative rather than being described as standalone packages.
  - `examples/04-provider-specific/README.md` now acts as the directory-level package-tier map so users can distinguish provider-owned packages, focused packages, and compat vendor views before drilling into provider examples.
  - Every current subdirectory under `examples/04-provider-specific/` now also has a local `README.md`, so provider-specific examples no longer rely on filename guessing alone to communicate whether they are provider-owned, focused, or compat-preset stories.
  - Representative example headers now also carry package-tier labels (`openrouter`, `perplexity`, `moonshot`, `deepseek`, `groq`) so readers can distinguish compat vendor views, provider-owned wrapper paths, and registry-first Stable examples directly from file headers.
  - `examples/04-provider-specific/openai-compatible/README.md` now also orders examples by contract instead of provider popularity: typed compat vendor views (`openrouter`, `perplexity`) come before compat preset stories (`moonshot`, `siliconflow`), and the lone builder file stays explicitly last as a compatibility demo.
  - The remaining Moonshot compat examples now also carry the same package-tier note style as the newer vendor-view examples, so `moonshot-basic.rs`, `moonshot-long-context.rs`, `moonshot-tools.rs`, and `moonshot-siumai-builder.rs` no longer rely on directory context alone to communicate “compat preset” vs “builder convenience demo”.
  - The compat example headers now also use a more uniform run/credential pattern: `openrouter`, `perplexity`, and `siliconflow` examples explicitly name their environment variables in the file header, while the Moonshot preset examples now use the same “set API key, then run” layout instead of mixing styles across files.
  - The compat example bodies are now converging too: Moonshot and SiliconFlow examples use the same imported `OpenAiCompatibleClient` construction style, and their console output now follows the same plain `Example` / `Answer` / `Notes` voice instead of mixing decorative emoji-heavy output with the leaner OpenRouter / Perplexity examples.
  - A second pass of provider-owned/focused example headers now follows the same pattern too: `deepseek`, `groq`, and `xai` files explicitly name required credentials in-file, while `ollama` examples now distinguish package tier from local-runtime prerequisites instead of collapsing both into one opaque run block.
  - High-traffic example READMEs have now been cleaned up for readability too: `examples/README.md` and `examples/04-provider-specific/minimaxi/README.md` no longer contain visible mojibake headings or duplicated navigation text.
  - The same readability cleanup now also covers `examples/04-provider-specific/google/README.md` and `examples/04-provider-specific/openai-compatible/README.md`, so the most frequently discovered provider README entry points share the same package-tier story and no longer expose visible encoding artifacts.
  - Public-surface compile guards now also cover the shared compat entry `provider_ext::openai_compatible` plus previously uncovered provider extensions `provider_ext::azure`, `provider_ext::google` (Gemini alias), `provider_ext::anthropic_vertex`, and `provider_ext::deepseek`, reducing the chance that export drift survives documentation cleanup.
  - Top-level no-network compat preset guards now also lock that `Provider::openai().{siliconflow,together,fireworks}()` still resolves through the shared compat defaults, including aligned default `base_url` + `model` resolution, and that their documented audio capability split remains stable (`siliconflow` and `together` support speech/transcription/audio, while `fireworks` remains transcription-only for audio). The same public guard set now also includes a `mistral` sample for the config-only fallback lane plus `jina` / `voyageai` samples for non-chat preset primary defaults, so builder shortcuts fill the documented provider default model without depending on whether the preset is chat-first or embedding/rerank-first. The underlying source-of-truth split is now tighter too: compat primary defaults and family defaults now resolve from static config-owned maps first, while `default_models.rs` is only the compatibility read facade, so preset default-model lookup no longer duplicates the whole compat catalog or rebuilds the provider map on every lookup.
  - The compat runtime now also consumes that same config-owned family-default table instead of treating the primary/chat model as a universal fallback: when request-level `model` is missing, `OpenAiCompatibleClient` now backfills embedding/image/rerank/speech/transcription requests from the corresponding family default first, while still preserving an explicit non-default `config.model` override. The provider-local no-network regressions now lock both branches on representative vendors (`together`, `jina`, `fireworks`), and a focused top-level public boundary test now also locks the same behavior on real `Siumai::builder()` / `Provider::openai()` / config-first compat paths for `together` embedding + image + TTS, `jina` rerank, and `fireworks` STT.
  - The outward-facing compat audio story now also has runnable examples on the shared runtime: `siliconflow-speech` / `siliconflow-transcription` and `together-speech` / `together-transcription` cover the two full-audio preset lanes, while `fireworks-transcription` covers the dedicated transcription-only lane.
  - OpenAI-compatible image generation now also has top-level public-path parity anchors for both `siliconflow` and `together`, locking that their real builder/provider/config-first construction stories still converge onto the same `/images/generations` request shape.
  - The compat image preset story is now runnable on both enrolled vendors too: `siliconflow-image` is now joined by `together-image`, so Together no longer relies on parity tests alone for its outward-facing config-first example path.
  - The remaining shared compat presets now also have an explicit no-image boundary: preset guards lock `Fireworks` / `OpenRouter` / `Perplexity` / `Infini` / `Jina` / `VoyageAI` to `as_image_generation_capability() == None`, and the public default registry now rejects `image_model(...)` construction for those presets before any transport is touched.
  - OpenAI embedding now also has a top-level public-path parity anchor for request-level options, and the underlying facade bug is fixed: `Siumai::embed_with_config` no longer drops `EmbeddingRequest.provider_options_map` on wrapper paths, so `dimensions` / `user` now survive builder, provider, and config-first construction consistently.
  - Gemini embedding now also has a top-level public-path parity anchor for stable request fields, locking that retrieval-query task shaping, `title`, and `dimensions` survive builder, provider, and config-first construction without drifting across the facade layers.
  - Gemini embedding now also has a top-level batch parity anchor, locking that multi-input requests still converge on the same `:batchEmbedContents` routing and per-item body shape across builder, provider, and config-first construction.
  - OpenAI-compatible `mistral` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths on the real compat shortcut story, locking that `Siumai::builder().openai().mistral()`, `Provider::openai().mistral()`, config-first `OpenAiCompatibleClient`, and `registry.embedding_model("mistral:...")` all converge on the same `/embeddings` request shape without inventing a dedicated provider package.
  - The shared OpenAI-compatible facade embedding path is now also fixed at the root: `Siumai::embed_with_config` no longer collapses compat vendors back to `embed(request.input)` when the underlying client is `OpenAiCompatibleClient`, so full request fields now survive wrapper execution instead of only working on provider/config-first paths.
  - `fireworks` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for request-level options, locking that `dimensions`, `encoding_format`, and `user` survive `Siumai::builder().openai().fireworks()`, `Provider::openai().fireworks()`, config-first `OpenAiCompatibleClient`, and `registry.embedding_model("fireworks:...")` construction consistently.
  - `siliconflow` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, locking that `dimensions`, `encoding_format`, and `user` survive `Siumai::builder().openai().siliconflow()`, `Provider::openai().siliconflow()`, config-first `OpenAiCompatibleClient`, and `registry.embedding_model("siliconflow:...")` construction consistently.
  - `openrouter` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, locking that `dimensions`, `encoding_format`, and `user` survive `Siumai::builder().openai().openrouter()`, `Provider::openai().openrouter()`, config-first `OpenAiCompatibleClient`, and `registry.embedding_model("openrouter:...")` construction consistently.
  - `openrouter` now also has public-path parity anchors for typed chat request options and compat reasoning defaults across both request modes, locking that `transforms`, custom vendor params, `enable_reasoning`, `reasoning_budget`, and Stable `tool_choice` / `response_format` survive `Siumai::builder().openai().openrouter()`, `Provider::openai().openrouter()`, config-first `OpenAiCompatibleClient`, and registry `language_model("openrouter:...")` construction consistently on both `chat_request` and `chat_stream_request`.
  - The shared compat typed-client helper now also consumes `BuildContext.reasoning_enabled` / `reasoning_budget`, and registry-scoped `ProviderBuildOverrides` now forward the same defaults into that context, so neither the `Siumai::builder()` compat path nor the registry compat path drops OpenRouter reasoning defaults that were already preserved by provider/config-first construction.
  - `RegistryBuilder` / `RegistryOptions` now also expose global reasoning defaults for language-model construction, and OpenRouter parity now locks both the direct `RegistryBuilder::new(...).with_reasoning(...).with_reasoning_budget(...)` entry and the lower-level `RegistryOptions` lane against config-first defaults while still preserving provider-scoped override precedence.
  - OpenRouter reasoning public parity now also covers response-side extraction on the real compat vendor-view story, so builder / provider / config-first / registry all expose the same non-stream `ChatResponse::reasoning()` semantics plus streaming `ThinkingDelta` accumulation while preserving `transforms` and reasoning-default request shaping.
  - Reasoning alignment now also has a dedicated workstream note in `reasoning-alignment.md`, so the Stable `enable + budget` contract, provider-owned hint knobs, streaming `ThinkingDelta` rule, and the current audited-provider boundary no longer have to stay implicit inside the feature matrix alone.
  - `perplexity` now also has public-path parity anchors for typed hosted-search request options across both request modes, locking that `search_mode`, `search_recency_filter`, `return_images`, nested `web_search_options`, and Stable `tool_choice` / `response_format` survive `Siumai::builder().openai().perplexity()`, `Provider::openai().perplexity()`, config-first `OpenAiCompatibleClient`, and registry `language_model("perplexity:...")` construction consistently on both `chat_request` and `chat_stream_request`.
  - `perplexity` hosted-search metadata now also has registry parity anchors on the real compat vendor-view story, locking that typed `PerplexityMetadata` extraction plus normalized `provider_metadata["perplexity"]` roots survive both 200-response and `StreamEnd` when requests flow through `registry.language_model("perplexity:...")` instead of only builder/provider/config-first construction.
  - Known OpenAI-compatible vendors now also route registry text-family construction through `OpenAiCompatibleBuilder` / `OpenAiCompatibleConfig` instead of the older helper-only assembly path, `OpenAiCompatibleBuilder` now accepts appended model middlewares in addition to auto-middlewares, and no-network deepseek compat contracts now lock builder/config/registry request convergence plus source-level factory routing for that generic path.
  - `deepseek` lower-contract coverage now also pins Stable `response_format` precedence alongside the existing typed reasoning options and `tool_choice` shaping, so raw `providerOptions.deepseek.response_format` overrides no longer rely only on public-path structured-output tests to prove convergence.
  - xAI now also has top-level public-path parity anchors for its provider-owned web-search request options across both request modes, locking that `reasoning_effort`, `search_parameters.mode`, date filters, and nested source allow/deny lists survive `Siumai::builder().xai()`, `Provider::xai()`, and config-first `XaiClient` construction consistently on both `chat_request` and `chat_stream_request`.
  - xAI hosted-search metadata now also has registry parity anchors on the native wrapper story, locking that typed `XaiMetadata.sources` / `logprobs` extraction survives both 200-response and `StreamEnd` when requests flow through `registry.language_model("xai:...")` instead of only builder/provider/config-first construction.
  - xAI registry text handles now also have matching typed request-options parity anchors for both request modes, locking that the same `XaiOptions` web-search payload survives `registry.language_model("xai:...")` and converges on the same final transport body as config-first `XaiClient` construction for both `chat_request` and `chat_stream_request`.
  - `xai` lower-contract coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `XaiOptions` web-search request helpers are no longer proven only by top-level public-path parity when requests flow through the native `XAIProviderFactory`.
  - `xai` lower-contract coverage now also pins Stable `tool_choice` / `response_format` precedence on the provider-owned builder/config/registry layer, so raw `providerOptions.xai.*` overrides no longer rely only on public-path structured-output and tool-routing tests to prove convergence.
  - xAI now also has a no-network tool-loop parity anchor on its real provider-owned public story, locking that non-stream `ChatResponse::tool_calls()` and streaming `ToolCallDelta` accumulation plus `StreamEnd.finish_reason` converge on the same tool-call payload across `Siumai::builder().xai()`, `Provider::xai()`, config-first `XaiClient`, and `registry.language_model("xai:...")`.
  - `XaiBuilder::build()` now also preserves explicit `with_http_client(...)`, `with_retry(...)`, and registry-supplied model middlewares at the final provider-owned construction boundary, and `XAIProviderFactory` text-family routing now reuses that same builder/config path instead of hand-assembling compat config/client pairs for HTTP/retry override preservation.
  - xAI shared-builder default-option parity is now also closed on that same provider-owned wrapper story: chained `Siumai::builder().xai().with_xai_options(...)` defaults now converge on the same final `chat_request` / `chat_stream_request` body as `Provider::xai()` and `XaiConfig::with_xai_options(...)`, and the unified builder now also exposes namespaced `with_xai_reasoning_effort(...)`, `with_xai_search_parameters(...)`, and `with_xai_default_search()` helpers instead of forcing callers back to request-local provider maps for common defaults.
  - Groq now also has a top-level public-path parity anchor for provider-owned typed chat request options, locking that `logprobs`, `top_logprobs`, `service_tier`, `reasoning_effort`, and `reasoning_format` survive both `chat_request` and `chat_stream_request` across `Siumai::builder().groq()`, `Provider::groq()`, and config-first `GroqClient` construction consistently instead of being covered only by provider-local spec tests.
  - Groq shared-builder default-option parity is now also closed on that same provider-owned wrapper story: chained `Siumai::builder().groq().with_groq_options(...)` defaults now converge on the same final `chat_request` / `chat_stream_request` body as `Provider::groq()` and `GroqConfig::with_groq_options(...)`, and the shared OpenAI-compatible `with_http_client(...)` path now preserves configured model middlewares / HTTP interceptors so unified-builder or registry-supplied HTTP clients do not silently drop middleware-backed defaults.
  - Provider-owned `xai`, `groq`, and `deepseek` wrappers now also have focused provider-local stream transport-boundary regressions, locking `Accept: text/event-stream`, `stream=true`, Stable `response_format` / `tool_choice`, and each provider's typed or normalized request-option family directly on `chat_stream_request` instead of relying only on public-path parity plus non-stream spec/client tests.
  - Groq reasoning public parity now also covers response-side extraction on the provider-owned wrapper story, so builder / provider / config-first / registry all expose the same non-stream `ChatResponse::reasoning()` semantics plus streaming `ThinkingDelta` accumulation while keeping `reasoning_effort` / `reasoning_format` / `service_tier` request shaping aligned.
  - Groq registry text handles now also have typed request-options parity anchors for both request modes, locking that the same `GroqOptions` payload survives `registry.language_model("groq:...")` and converges on the same final transport body as config-first `GroqClient` construction for both `chat_request` and `chat_stream_request`.
  - `groq` lower-contract coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `GroqOptions` request helpers are no longer proven only by top-level public-path parity plus provider-local spec tests.
  - `groq` lower-contract coverage now also pins Stable `tool_choice` / `response_format` precedence on the provider-owned builder/config/registry layer, so raw `providerOptions.groq.*` overrides no longer rely only on public-path structured-output and tool-routing tests to prove convergence.
  - Groq now also has no-network public-path structured-output parity on its real provider-owned wrapper story, locking both Stable `response_format` precedence over raw `providerOptions.groq.response_format` and shared-SSE JSON extraction across builder / provider / config-first / registry construction.
  - Groq structured-output public parity now also covers interrupted synthetic-unknown-stream-end failure semantics on that same wrapper story, so builder / provider / config-first / registry all surface the same incomplete-stream `ParseError` contract instead of relying only on global facade tests.
  - Groq now also has a no-network tool-loop parity anchor on its real provider-owned public story, locking that non-stream `ChatResponse::tool_calls()` and streaming `ToolCallDelta` accumulation plus `StreamEnd.finish_reason` converge on the same tool-call payload across `Siumai::builder().groq()`, `Provider::groq()`, config-first `GroqClient`, and `registry.language_model("groq:...")`.
  - `GroqBuilder::build()` now also preserves explicit `with_http_client(...)`, `with_retry(...)`, and registry-supplied model middlewares at the final provider-owned construction boundary, and `GroqProviderFactory` text/speech/transcription family routing now reuses that same builder/config path instead of hand-assembling compat config/client pairs.
  - DeepSeek now also has a top-level public-path parity anchor for provider-owned typed chat request options, locking that `enable_reasoning`, `reasoning_budget`, and custom vendor params survive both `chat_request` and `chat_stream_request` across `Siumai::builder().deepseek()`, `Provider::deepseek()`, and config-first `DeepSeekClient` construction consistently instead of being exercised only through raw provider maps.
  - DeepSeek registry text handles now also have matching typed request-options parity anchors for both request modes, locking that the same `DeepSeekOptions` payload survives `registry.language_model("deepseek:...")` and converges on the same final transport body as config-first `DeepSeekClient` construction.
  - DeepSeek reasoning public parity now also covers response-side extraction on that same wrapper story, so builder / provider / config-first / registry all expose the same non-stream `ChatResponse::reasoning()` semantics plus streaming `ThinkingDelta` accumulation while preserving `enable_reasoning` / `reasoning_budget` request shaping.
  - DeepSeek now also has no-network public-path parity for Stable `tool_choice` precedence over raw `providerOptions.deepseek.tool_choice`, covering builder / provider / config-first / registry on the real provider-owned wrapper story instead of leaving that rule only in provider-local spec tests.
  - DeepSeek now also has no-network public-path structured-output parity on its real provider-owned wrapper story, locking both Stable `response_format` precedence over raw `providerOptions.deepseek.response_format` and shared-SSE JSON extraction across builder / provider / config-first / registry construction.
  - DeepSeek structured-output public parity now also covers interrupted synthetic-unknown-stream-end failure semantics on that same wrapper story, so builder / provider / config-first / registry all surface the same incomplete-stream `ParseError` contract while preserving Stable `response_format` precedence and `reasoningBudget` normalization.
  - DeepSeek now also has a no-network tool-loop parity anchor on its real provider-owned public story, locking that non-stream `ChatResponse::tool_calls()` and streaming `ToolCallDelta` accumulation plus `StreamEnd.finish_reason` converge on the same tool-call payload across `Siumai::builder().deepseek()`, `Provider::deepseek()`, config-first `DeepSeekClient`, and `registry.language_model("deepseek:...")`.
  - `DeepSeekBuilder::build()` now also preserves explicit `with_http_client(...)`, `with_retry(...)`, and registry-supplied model middlewares at the final provider-owned construction boundary, and `DeepSeekProviderFactory` text-family routing now reuses that same builder/config path instead of hand-assembling provider config/client pairs.
  - DeepSeek shared-builder default-option parity is now also closed on that same provider-owned wrapper story: chained `Siumai::builder().deepseek().with_deepseek_options(...)` defaults now converge on the same final `chat_request` / `chat_stream_request` body as `Provider::deepseek()` and `DeepSeekConfig::with_deepseek_options(...)`, and the unified builder now also exposes namespaced `with_deepseek_reasoning(...)` / `with_deepseek_reasoning_budget(...)` helpers for common defaults.
  - Shared-builder provider default-options middleware now also has deterministic override semantics across OpenAI / Anthropic / Vertex Anthropic / MiniMaxi / Groq / xAI / DeepSeek: later `with_*_options(...)` calls are inserted ahead of earlier default middlewares, so chained builder defaults and custom middlewares consistently override older defaults instead of being silently treated as request-local overrides.
  - Focused DeepSeek regressions now pin that override rule on the real shared-builder request path too: a later `with_deepseek_reasoning_budget(...)` overrides an earlier `with_deepseek_reasoning(false)`, and appended custom model middleware still wins over builder-level default options on the final request body.
  - Ollama now also has a top-level public-path parity anchor for provider-owned typed chat request options, locking that `keep_alive`, `raw`, `format`, and merged model options such as `think` / `num_ctx` survive both `chat_request` and `chat_stream_request` across `Siumai::builder().ollama()`, `Provider::ollama()`, and config-first `OllamaClient` construction consistently instead of being exercised only through raw provider maps.
  - Ollama shared-builder typed default-options are now also closed on that same wrapper story: chained `Siumai::builder().ollama().with_ollama_options(...)` defaults now converge on the same final `chat_request` / `chat_stream_request` body as `Provider::ollama().with_ollama_options(...)` and `OllamaConfig::builder().with_ollama_options(...)`, with sparse `OllamaOptions` serialization preventing null-placeholder overwrites during default-option merges.
  - The adjacent provider-local Ollama coverage is runnable again too: the chat-executor test helper now unwraps the optional request transformer explicitly after the executor API moved to `Option`, so the new sparse/default-option unit tests are no longer blocked by an unrelated local compile failure.
  - Ollama registry text handles now also have matching typed request-options parity anchors for both request modes, locking that the same `OllamaOptions` payload survives `registry.language_model("ollama:...")` and converges on the same final transport body as config-first `OllamaClient` construction.
  - `ollama` lower-contract coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `OllamaOptions` request helpers — including the previously dropped `raw` flag — are no longer proven only by top-level public-path parity on the native chat path.
  - MiniMaxi now also has a top-level public-path parity anchor for provider-owned typed chat request options, locking that provider-owned thinking mode and structured output helpers survive both `chat_request` and `chat_stream_request` across `Siumai::builder().minimaxi()`, `Provider::minimaxi()`, and config-first `MinimaxiClient` construction consistently instead of being guarded only by provider-local normalization tests.
  - MiniMaxi registry text handles now also have matching typed request-options parity anchors for both request modes, locking that the same `MinimaxiOptions` payload survives `registry.language_model("minimaxi:...")` and converges on the same final transport body as config-first `MinimaxiClient` construction.
  - `minimaxi` lower-contract coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `MinimaxiOptions` request helpers are no longer proven only by top-level public-path parity on the native text path.
  - Provider-owned `minimaxi` now also has a focused provider-local stream transport-boundary regression, locking `Accept: text/event-stream`, `stream=true`, typed thinking-mode shaping, Stable `response_format` precedence over provider-owned structured-output helpers, and the resulting tool stripping directly on `MinimaxiClient::chat_stream_request`.
  - Anthropic typed chat request helpers now also have a complete public-path parity story: provider-owned `thinking_mode`, structured output, and stable tool-choice shaping survive both `chat_request` and `chat_stream_request` across `Siumai::builder().anthropic()`, `Provider::anthropic()`, config-first `AnthropicClient`, and `registry.language_model("anthropic:...")`.
  - `anthropic` lower-contract coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `AnthropicOptions` request helpers pin `thinking_mode`, `context_management`, `container`, provider-owned structured output, and `output_config.effort` on both request modes instead of relying only on top-level public-path parity.
  - Provider-owned `AnthropicClient` now also has a focused provider-local stream transport-boundary regression, locking `Accept: text/event-stream`, `stream=true`, typed `thinking_mode`, `context_management`, `container`, provider-owned `output_format` / `output_config`, and the opt-out of `fine-grained-tool-streaming` directly on `chat_stream_request`.
  - Anthropic default typed-option middleware merging is now hardened too: `AnthropicOptions` serialize sparsely instead of emitting `null` placeholders, `HttpChatExecutor::execute_stream(...)` now resolves runtime transformer bundles only after model middlewares run, and the public builder/provider/config parity lane now locks default `thinking_mode`, `structured_output_mode`, `context_management`, `tool_streaming`, and `effort` on the final streaming `/v1/messages` body without null-overwrite drift.
  - Google Vertex registry public paths now also lock provider-owned non-text typed request helpers directly: `VertexEmbeddingOptions` survive `registry.embedding_model("vertex:...")`, while `VertexImagenOptions` survive both `registry.image_model("vertex:...").generate_images(...)` and `edit_image(...)`, each converging on the same final transport body as config-first `GoogleVertexClient`.
  - Google Vertex stream lane now also has matching public-path plus lower-contract anchors for Stable `response_format` / `tool_choice` together with provider-owned `thinkingConfig` / `structuredOutputs`, so `chat_stream_request` no longer relies on non-stream `generateContent` coverage to prove those fields survive onto `:streamGenerateContent`.
  - Provider-owned `OpenAiClient` now also exposes the same resource-helper tier as the stronger native wrapper packages: `resource_config()` snapshots runtime auth/base URL / organization / project / provider-options / HTTP state, and `files()` / `models()` / `moderation()` / `rerank()` now reuse that state directly instead of forcing provider-package callers back through ad hoc side construction for provider-owned extras.
  - `provider_ext::gemini::resources` now also re-exports `GeminiCachedContents` and `GeminiTokens` alongside the already-exposed file/model helpers, so the top-level provider package matches the real `GeminiClient` resource-accessor surface instead of hiding two provider-owned extras behind a lower package-only import path.
  - Public package compile guards now also pin provider-owned resource modules for `provider_ext::openai`, `provider_ext::anthropic`, and the stable `provider_ext::google` alias, so `files` / `models` / `moderation` / `rerank`, Anthropic token/file/batch helpers, and Gemini cached-content/token helpers cannot silently fall off the top-level package surface.
  - The same public-surface guard layer now also pins non-unified `provider_ext::*::ext` modules for OpenAI, Anthropic, Gemini, and the stable Google alias, covering OpenAI moderation/responses/audio-stream helpers, Anthropic structured-output/thinking/tool-event helpers, and Gemini code-execution/file-search/tool-event helpers so these escape hatches cannot drift out of the top-level package facade unnoticed.
  - Focused wrapper packages now also participate in that `ext` facade guard story where they expose real provider-owned escape hatches: `provider_ext::groq::ext::audio_options` now has a public compile anchor for typed TTS/STT options, and `provider_ext::minimaxi::ext` now locks video/music builders plus structured-output/thinking helper entry points on the top-level package facade instead of leaving them proven only by lower crates.
  - `provider_ext::xai::provider_tools` now also has a direct public compile anchor for `web_search`, `x_search`, and `code_execution`, so the provider-owned tool-factory facade is no longer implicitly trusted just because the lower shared `tools::xai` module exists.
  - Compatibility aliases now have explicit public-surface anchors too: `provider_ext::openai::provider_tools`, `provider_ext::anthropic::provider_tools`, `provider_ext::gemini::provider_tools`, and `provider_ext::google::provider_tools` are now compile-guarded alongside their preferred `tools` modules, so old imports can stay stable without drifting silently.
  - `provider_ext::google_vertex` public compile guards now also pin the Vertex-owned `vertex_rag_store` tool entry point on both `tools` and `hosted_tools`, so the package facade no longer relies on shared Google tool coverage alone to prove its unique RAG helper surface stays reachable.
  - Focused rerank / split-runtime wrapper helper tiers are now also pinned on the public facade: `provider_ext::cohere::CohereClient`, `provider_ext::togetherai::TogetherAiClient`, and `provider_ext::bedrock::BedrockClient` compile-guard their exposed inspection helpers (`provider_context` / `base_url` / HTTP/retry accessors, plus Bedrock's split runtime helpers), so these top-level provider packages no longer rely on lower crate tests alone to keep their debug/inspection surface stable.
  - The remaining wrapper helper tiers now also have direct public compile anchors: `provider_ext::{xai,groq,deepseek,ollama}` guard their `provider_context` / `base_url` / HTTP/retry accessor set on the top-level facade, while `provider_ext::{openai,anthropic,gemini,google,google_vertex}` now also pin their currently exposed base-url / retry helpers, closing the last obvious “helper exists but only lower crates prove it” gap in the provider package surface.
  - Shared Google tool facades now also have broader top-level compile anchors: `provider_ext::gemini`, the stable `provider_ext::google` alias, and `provider_ext::google_vertex` now pin representative `code_execution` / `google_maps` / `url_context` / `enterprise_web_search` and `file_search` builder entry points in addition to `google_search`, so alias modules stop depending on a single shared tool symbol as their only public guard.
  - `together` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, locking that `dimensions`, `encoding_format`, and `user` survive `Siumai::builder().openai().together()`, `Provider::openai().together()`, config-first `OpenAiCompatibleClient`, and `registry.embedding_model("together:...")` construction consistently.
  - `jina`, `voyageai`, and `infini` embedding now also have public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, and `Provider::openai()` now exposes matching vendor shortcuts so their compat embedding stories no longer depend on stringly `compatible(\"...\")` calls alone.
  - Embedding gap audit result is now closed for the currently enrolled OpenAI-compatible catalog: every advertised embedding vendor is either covered by a parity anchor or an explicit boundary decision, with Cohere intentionally staying compat-only for embeddings while its native package remains rerank-led.
  - Cohere boundary decision is now explicit and guard-tested: `provider_ext::cohere` stays rerank-led on the native `/v2` package story, while Cohere embeddings stay on the shared OpenAI-compatible story (`Provider::openai().compatible("cohere")` and config-first `OpenAiCompatibleClient` against `https://api.cohere.ai/v1/embeddings`) instead of forcing a fake three-way top-level parity anchor.
  - Ollama embedding now also has a provider-owned top-level public-path parity anchor, and the underlying facade/client gaps are fixed: `OllamaClient` now honors `EmbeddingExtensions::embed_with_config`, while `Siumai::embed_with_config` now preserves full `EmbeddingRequest` payloads for Ollama instead of collapsing them to bare input text.
  - xAI and Groq embedding boundaries are now explicit and guard-tested too: their provider-owned native stories remain text/audio-led, registry factories now reject native embedding-family construction with `UnsupportedOperation`, top-level public paths now reject `embed_with_config(...)` without emitting HTTP traffic, and wrapper clients no longer leak shared-compat embedding capability through `as_embedding_capability()`.
  - xAI and Groq image/rerank boundaries are now explicit and guard-tested too: their provider-owned native stories continue to stop at text/audio, registry factories now reject native image-family construction, top-level public paths now reject image/rerank calls without emitting HTTP traffic, and wrapper clients no longer leak shared-compat image/rerank capability through `as_image_generation_capability()` / `as_image_extras()` / `as_rerank_capability()`.
  - Groq audio family routing now also has registry-level speech/transcription coverage plus top-level parity on the real provider-owned wrapper story, and the underlying root-base normalization drift is fixed at the config layer so audio requests no longer disagree with chat requests about when `/openai/v1` must be appended.
  - DeepSeek now follows the same focused-wrapper capability contract end-to-end: the provider-owned `DeepSeekClient` fail-fast rejects embedding/image/rerank directly instead of delegating to the shared compat runtime, `capabilities()` stays aligned with the native metadata surface (`chat + streaming + tools + vision + thinking` only), and both contract/public-path tests now lock that the typed wrapper does not leak deferred capabilities through `as_embedding_capability()` / `as_image_generation_capability()` / `as_rerank_capability()`.
  - The focused-wrapper contract is now encoded through shared helper assertions in both `contract_tests.rs` and `provider_public_path_parity_test.rs`, so future wrappers can reuse one pattern for “deferred capability absent” instead of re-encoding ad hoc negative checks per provider.
  - Top-level no-network compat vendor-view guards now also lock that `Provider::openai().{openrouter,perplexity}()` still resolves through the same shared compat registry defaults, and that their public role stays narrow: `openrouter` remains a tools/embedding/reasoning-capable typed vendor view with no audio surface, while `perplexity` remains a tools-led typed vendor view without separate audio or embedding promotion. Built public clients now also preserve that same split through `capabilities()` and `as_*_capability()` checks, so compat preset metadata no longer drifts away from the final client surface.
  - Top-level focused-provider guards now also lock the intended request-helper boundary directly on request objects: `provider_ext::google_vertex` continues to write typed request options under the stable `providerOptions.vertex` key, while `provider_ext::bedrock` continues to expose Bedrock-specific request helpers through `providerOptions.bedrock` instead of pretending to have a wider metadata/resources public story.



- [x] Add migration shims for legacy audio capability users.
  - The legacy `AudioCapability -> SpeechCapability / TranscriptionCapability` adapter layer is now also covered by direct core regression tests, so migration users keep a tested bridge while recommended public paths continue moving toward the narrower speech/transcription families.







## Track F - Provider migration







### Core providers first







- [x] Migrate OpenAI provider to a native text-family path spike.



- [x] Migrate Anthropic provider to a native text-family path spike.



- [x] Migrate Gemini provider to a native text-family path spike.
  - Gemini now also has a direct builder/config/registry contract anchor for provider-owned batch embedding request shaping, so the provider factory path itself now locks `:batchEmbedContents` endpoint selection and final request-body convergence instead of relying only on top-level public-path parity.
  - Gemini now also fail-fast rejects provider-factory `speech` / `transcription` family construction instead of reusing the embedding-family bridge internally, and the existing public registry audio-family guard remains the outer boundary so unsupported audio handles cannot drift back in through factory fallback.



- [x] Migrate OpenAI-compatible base implementation to a native text-family path spike.



- [~] Extend the OpenAI-compatible base beyond text/embedding into speech/transcription family foundations.

  - The shared OpenAI-compatible runtime now has native speech/transcription family factory entry points plus spec/client audio wiring.
  - Built-in vendor capability enrollment and contract validation are now in place for `together` and `siliconflow`; the rest of the compat catalog remains pending provider-by-provider verification.







### Secondary providers







- [x] Migrate Ollama.



  - Provider-owned `OllamaConfig` / `OllamaClient` entry points, registry-native text-family routing, builder/config/registry/public-path parity coverage, provider-owned typed response metadata (`OllamaMetadata` / `OllamaChatResponseExt`), and injected-transport JSON-stream coverage are now in place, so Ollama no longer depends on the generic client bridge as its primary migration story.
  - Top-level public-path coverage for Ollama is now deeper too: builder/provider/config-first construction now also locks the `reasoning(true) -> think` default path, and a no-network 200-response guard now verifies that typed timing metadata survives the full public wrapper story instead of only provider-local response-conversion tests.
  - Ollama now also has a native registry embedding-family factory override plus direct builder/config/factory contract coverage on `/api/embed`, and the top-level embedding parity test now includes the public `registry.embedding_model("ollama:...")` handle too, so its declared `embedding` capability no longer relies on the generic `embedding_model_with_ctx -> language_model_with_ctx` bridge alone.



- [x] Migrate Groq to a native text-family path spike.



- [x] Migrate xAI to a native text-family path spike.



- [~] Migrate Google Vertex.



  - Provider-owned `GoogleVertexConfig` now has explicit config-first constructors (`new` / `express` / `enterprise`), `GoogleVertexBuilder` now emits canonical config through `into_config()`, registry factory construction now routes through the provider-owned config/client path without dropping interceptors or model middlewares, native text/image/embedding family construction now materializes the provider-owned `GoogleVertexClient` directly, the generic embedding factory path no longer ignores `BuildContext`, and no-network public-path parity now locks `Siumai::builder().vertex()` / `Provider::vertex()` / config-first equivalence for chat, chat streaming, embedding, image generation, and extension-only image editing at the final transport boundary.
  - Google Vertex now also has an explicit builder/config/registry contract anchor for provider-owned image generation request shaping, so the provider factory path is no longer relying only on top-level public-path tests to keep Imagen request convergence honest.
  - Google Vertex embedding execution now also preserves full `EmbeddingRequest` payloads on both the provider-owned client path and the native registry embedding handle path: `EmbeddingModelHandle::embed_with_config` now dispatches into provider-owned typed embedding clients before falling back to raw `EmbeddingCapability`, and both top-level public-path parity plus builder/config/registry contract coverage now lock provider-owned embedding request shaping (`task_type`, `title`, `outputDimensionality`, `autoTruncate`) at the final transport boundary.
  - Google Vertex now also has direct builder/config/registry contract anchors for provider-owned chat and chat-stream request shaping, so the provider factory path no longer relies only on top-level public-path parity to keep `generateContent` / `streamGenerateContent` transport-boundary convergence honest.
  - Anthropic-on-Vertex now also has a complete provider-owned builder story: `VertexAnthropicBuilder` / `VertexAnthropicConfig` / `VertexAnthropicClient` converge through config-first construction, `VertexAnthropicConfig` now exposes auth/header helpers plus `with_http_transport`, `Provider::anthropic_vertex()` and `Siumai::builder().anthropic_vertex()` are now available, and both top-level public-path parity plus builder/config/registry contract coverage now lock `:rawPredict` / `:streamRawPredict` request shaping at the final transport boundary, including explicit full-request `model` overrides instead of silently collapsing back to the handle/config default model.
  - Anthropic-on-Vertex top-level public registry parity now also locks `registry.language_model("anthropic-vertex:...")` directly onto `:rawPredict` / `:streamRawPredict`, so the public registry lane no longer trails the already-landed lower builder/config/registry contract coverage.
  - Anthropic-on-Vertex native structured-output depth now also reaches both the provider-owned client path and the real public wrapper/registry story, so the supported native `output_format` lane on both `:rawPredict` and `:streamRawPredict` now locks non-stream JSON extraction plus stream complete accumulated JSON / final `StreamEnd` `id` / `model` / `finish_reason`, together with the matching invalid-response and truncated-stream `ParseError` paths across builder / provider / config-first / `language_model("anthropic-vertex:...")` instead of relying on generic Anthropic fallback coverage or request-shape-only regressions; the same path now also has a runnable config-first example (`examples/04-provider-specific/google/vertex_anthropic_structured_output.rs`) so this capability is no longer documented only through tests.
  - Anthropic-on-Vertex native reasoning depth now also reaches both the provider-owned client path and the top-level public wrapper story, so raw `providerOptions.anthropic.{thinking_mode,thinkingMode,thinking}` now converge to the native `thinking` body on both `:rawPredict` and `:streamRawPredict`, while non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` and final `providerMetadata.anthropic.{thinking_signature,redacted_thinking_data}` are locked on that same provider-owned/public path instead of relying on generic Anthropic reasoning coverage or request-shape-only assertions.
  - Anthropic-on-Vertex now also exposes provider-owned typed request helpers on the public facade, so `siumai::provider_ext::anthropic_vertex::{VertexAnthropicOptions, VertexAnthropicChatRequestExt}` can drive native `thinking_mode`, `structured_output_mode`, `disable_parallel_tool_use`, and `send_reasoning` shaping without dropping back to raw `with_provider_option("anthropic", ..)` JSON; provider-local serialization tests, public surface compile guards, and the existing reasoning parity path now all execute through that typed route.
  - Anthropic-on-Vertex now also exposes the protocol-owned Anthropic typed metadata facade on its own wrapper path, so `siumai::provider_ext::anthropic_vertex::{AnthropicChatResponseExt, AnthropicMetadata}` can read `thinking_signature` / `redacted_thinking_data` from the native Vertex wrapper directly under the `google-vertex` feature; provider-local reasoning tests, top-level public-path parity, and public-surface compile guards now all execute through that typed metadata path instead of hand-walking raw `provider_metadata["anthropic"]` maps.
  - Anthropic-on-Vertex config-first and builder-first construction now also persist provider-owned default request options before request-local overrides, so `VertexAnthropicConfig` / `VertexAnthropicBuilder` can carry default `thinking_mode`, `structured_output_mode`, `disable_parallel_tool_use`, and `send_reasoning` through the same `provider_options_map["anthropic"]` adapter path used by request-level typed options; provider-local regression coverage now pins both config merge precedence and builder `into_config()` preservation.
  - The shared `Siumai::builder().anthropic_vertex()` path now also carries those same Anthropic-on-Vertex typed defaults through namespaced fluent helpers (`with_anthropic_vertex_options`, `with_anthropic_vertex_thinking_mode`, `with_anthropic_vertex_structured_output_mode`, `with_anthropic_vertex_disable_parallel_tool_use`, `with_anthropic_vertex_send_reasoning`) implemented as model middleware defaults on the unified builder, so the top-level public story closes without consuming globally generic method names that the shared builder may need for native Anthropic later; public-surface compile guards and focused no-network parity now lock builder/provider/config convergence on the final `:rawPredict` body, including thinking-budget token expansion, reserved-`json` tool fallback, `disable_parallel_tool_use`, and `send_reasoning = false` replay stripping.
  - Anthropic-on-Vertex default typed options now also have an explicit streaming parity anchor: provider-owned `VertexAnthropicOptions` serialization is pinned to omit unset fields, and the top-level builder/provider/config public path now locks the same default thinking/json-tool/disable-parallel/send-reasoning shaping on `:streamRawPredict` that was already covered on `:rawPredict`, so this path cannot regress into the null-overwrite class of bug fixed on native Anthropic.
  - `ProviderBuildOverrides` can now also carry provider-scoped `http_config`, and Anthropic-on-Vertex now pins that lane on the real registry chat handle with merge-over-global semantics, so mixed registries can override Bearer auth headers, base URLs, and transports for header-auth providers without dropping registry-level headers, timeout defaults, or user-agent wiring.
  - Anthropic-on-Vertex now also has a runnable provider-owned example plus builder-alignment coverage, and its enterprise auth story is no longer header-only: `VertexAnthropicConfig` / `VertexAnthropicBuilder` now accept a token provider, runtime context construction lazily injects `Authorization: Bearer ...`, provider-owned builder auto-enables ADC under `gcp` when no explicit auth is supplied, registry construction reuses `BuildContext.google_token_provider` (with backward-compatible fallback to `gemini_token_provider`) plus the same ADC fallback, and the unified builder now exposes neutral aliases (`with_google_token_provider`, `with_google_adc`, `with_vertex_token_provider`, `with_vertex_adc`) instead of forcing Gemini-specific naming onto Vertex-family auth flows.
  - Anthropic-on-Vertex capability-split closure now also reaches the lower boundary: deferred `embedding` / `image` / `rerank` / audio-family builders reject explicitly with `UnsupportedOperation`, and both registry-contract coverage plus public `registry.*_model("anthropic-vertex:...")` guards now pin the same no-request failure path instead of inheriting a generic text fallback.



- [~] Migrate Cohere.

  - Provider-owned `CohereConfig` / `CohereClient` / `CohereBuilder` now exist for the rerank-only surface, registry factory construction now materializes the provider-owned typed client instead of an ad-hoc local factory client, `Siumai::builder().cohere()` / `Provider::cohere()` / config-first construction are aligned, and typed rerank request options now live under `provider_ext::cohere::options`.
  - Cohere now also has a direct builder/config/registry contract anchor for provider-owned rerank request shaping, so the provider factory path itself now locks `/rerank` URL selection, authorization propagation, and Cohere-specific rerank fields (`top_n`, `max_tokens_per_doc`, `priority`) instead of relying only on top-level public-path parity.
  - `CohereClient` now also exposes the same wrapper helper tier used by other provider-owned clients (`provider_context`, `base_url`, `http_client`, `retry_options`, `http_interceptors`, `http_transport`, `set_retry_options`), so focused rerank packages no longer lag behind chat-heavy wrappers in observability/debug ergonomics.
  - `siumai-registry` now also keeps the shared provider-inference helper available under standalone `cohere` / `togetherai` builds, fixing the same cfg boundary class that previously affected `bedrock` when rerank-only provider features compiled without unrelated text-provider features.
  - Cohere’s focused-provider boundary is now explicit as well: factory/catalog tests lock it to rerank-only capabilities, and no-network public-path tests now assert chat requests fail with `UnsupportedOperation` before any HTTP request is emitted across siumai/provider/config construction.
  - The registry factory `language_model_with_ctx` entry is now pinned to that same boundary too: Cohere no longer materializes a compatibility rerank client on the generic language-model path, and contract tests now lock that the factory rejects unsupported text-family construction instead of silently widening into a non-chat client.
  - Registry builder / handle construction now also has a provider-scoped Cohere rerank anchor: `ProviderBuildOverrides` precedence is locked on the real `reranking_model(...)` handle path, so mixed registries can route Cohere-specific API keys, base URLs, and transports to `/rerank` without leaking global overrides into the focused rerank package.
  - Cohere top-level public registry parity now also has a real rerank-handle anchor: `registry.reranking_model("cohere:...")` matches config-first `/rerank` request shaping, and the same public facade now also pins `ProviderBuildOverrides` precedence on the final rerank request instead of leaving both guarantees only at the lower contract layer.



- [~] Migrate TogetherAI.

  - Provider-owned `TogetherAiConfig` / `TogetherAiClient` / `TogetherAiBuilder` now exist for the rerank-only surface, registry factory construction now materializes the provider-owned typed client instead of an ad-hoc local factory client, `Siumai::builder().togetherai()` / `Provider::togetherai()` / config-first construction are aligned, and typed rerank request options now live under `provider_ext::togetherai::options`.
  - TogetherAI now also has a direct builder/config/registry contract anchor for provider-owned rerank request shaping, so the provider factory path itself now locks `/rerank` URL selection, authorization propagation, and TogetherAI-specific rerank fields (`return_documents`, `rank_fields`) instead of relying only on top-level public-path parity.
  - `TogetherAiClient` now also exposes the same wrapper helper tier used by other provider-owned clients (`provider_context`, `base_url`, `http_client`, `retry_options`, `http_interceptors`, `http_transport`, `set_retry_options`), keeping the focused rerank wrapper story aligned with the rest of the provider-owned surface.
  - TogetherAI’s focused-provider boundary is now explicit too: factory/catalog tests lock it to rerank-only capabilities, and no-network public-path tests now assert chat requests fail with `UnsupportedOperation` before any HTTP request is emitted across siumai/provider/config construction.
  - The registry factory `language_model_with_ctx` entry is now pinned to that same boundary too: TogetherAI no longer reuses the rerank client on the generic language-model path, and contract tests now lock that the factory rejects unsupported text-family construction instead of silently widening into a non-chat client.
  - Registry builder / handle construction now also has a provider-scoped TogetherAI rerank anchor: `ProviderBuildOverrides` precedence is locked on the real `reranking_model(...)` handle path, so mixed registries can route TogetherAI-specific API keys, base URLs, and transports to `/rerank` without contaminating adjacent providers.
  - TogetherAI top-level public registry parity now also has a real rerank-handle anchor: `registry.reranking_model("togetherai:...")` matches config-first `/rerank` request shaping, and the same public facade now also pins `ProviderBuildOverrides` precedence on the final rerank request instead of leaving both guarantees only at the lower contract layer.



- [~] Migrate Bedrock.

  - Provider-owned `BedrockConfig` / `BedrockClient` / `BedrockBuilder` now exist on the split runtime vs agent-runtime story, registry construction now materializes the provider-owned typed client directly, and no-network public-path parity already locks chat, chat-stream, and rerank request shaping across `Siumai::builder().bedrock()` / `Provider::bedrock()` / config-first construction.
  - Amazon Bedrock now also has direct builder/config/registry contract anchors for chat, chat-stream, and rerank request shaping, so the provider factory path itself now locks runtime vs agent-runtime URL routing, authorization headers, and Bedrock-specific request-body passthrough instead of depending only on top-level public-path parity.
  - `BedrockClient` now also exposes split runtime helper accessors (`chat_provider_context`, `rerank_provider_context`, `runtime_base_url`, `agent_runtime_base_url`) plus the common wrapper helpers (`http_client`, `retry_options`, `http_interceptors`, `http_transport`, `set_retry_options`), making the split-endpoint provider-owned client easier to inspect without raw field access.
  - Amazon Bedrock now also exposes provider-owned typed response metadata through `provider_ext::bedrock::metadata::{BedrockMetadata, BedrockChatResponseExt}`, so provider-specific chat metadata (`isJsonResponseFromTool`, `stopSequence`) no longer requires raw `provider_metadata["bedrock"]` traversal.
  - Bedrock alignment tests and focused surface guards now also consume that typed metadata helper directly, so the new escape hatch is covered as part of real fixture-alignment and provider-boundary checks instead of existing only as a public re-export.
  - Top-level builder/provider/config-first Bedrock parity now also locks that typed metadata on both 200-response and JSON-stream `StreamEnd`, so `BedrockChatResponseExt` is exercised on the real public construction surface instead of only package-local or focused provider tests.
  - Top-level public registry parity now also locks that same typed Bedrock metadata on both 200-response and JSON-stream `StreamEnd`, so `registry.language_model("bedrock:...")` no longer trails config-first metadata extraction on the real public path.
  - Bedrock provider-local standard tests now also use `BedrockChatResponseExt` for the reserved-`json` tool response path, so the last response-side raw `provider_metadata["bedrock"]` assertion in the Bedrock crate has been removed.
  - `siumai-registry` now also keeps the provider-inference helper available under the standalone `bedrock` feature, fixing the single-feature compile boundary where `SiumaiBuilder` routing could call a cfg-elided resolver symbol even though Bedrock itself does not depend on model-prefix inference.
  - `create_registry_with_defaults()` now also registers the native `bedrock` factory when that feature is enabled, closing the gap between Bedrock’s provider-owned migration story and the built-in registry surface while also removing the feature-minimal `unused import ids` warning that came from the missing registration branch.
  - `siumai/build.rs` now also counts `cohere`, `togetherai`, and `bedrock` as valid provider features, so rerank-only / Bedrock-only builds no longer fail the top-level “at least one provider enabled” guard before public-path tests even start.
  - `siumai-registry` single-feature compile sweep is now green for the current native provider set (`openai`, `azure`, `anthropic`, `google`, `google-vertex`, `ollama`, `xai`, `groq`, `deepseek`, `minimaxi`, `cohere`, `togetherai`, `bedrock`), and the built-in registry registration checks for the rerank-only packages (`cohere`, `togetherai`, `bedrock`) are now all explicitly validated in feature-minimal mode.
  - Bedrock’s current native boundary is now explicit too: factory/catalog tests lock it to chat + streaming + tools + rerank, while no-network public-path tests now assert `embedding` and `image_generation` fail with `UnsupportedOperation` before any HTTP request is emitted across siumai/provider/config construction.
  - Amazon Bedrock now also has matching top-level public registry anchors for that same split boundary: `registry.reranking_model("bedrock:...")` matches config-first `/rerank` request shaping, while `registry.embedding_model("bedrock:...")` and `registry.image_model("bedrock:...")` now fail before transport on the real public registry path.
  - Bedrock public registry chat parity now also covers Stable request options directly: `registry.language_model("bedrock:...")` matches config-first `tool_choice` / structured-output / `additionalModelRequestFields` shaping on the final `/converse` request body.
  - Bedrock public registry chat parity now also has explicit request-shape anchors for both `/converse` and `/converse-stream`, so the native registry language-model handle no longer relies on metadata or stable-options coverage to prove final request routing and provider-specific passthrough.
  - `RegistryBuilder::with_provider_build_overrides("bedrock", ...)` now also has top-level text-handle anchors on `language_model("bedrock:...")` for both `chat_request` and `chat_stream_request`, so provider-scoped auth / base-url / transport precedence is pinned on the real `/converse` and `/converse-stream` public paths instead of only on the rerank handle.
  - The same Bedrock text-handle override lane is now also pinned at the `siumai-registry` lower contract layer for both non-streaming and streaming chat, so `RegistryBuilder` provider overrides for `language_model("bedrock:...")` are enforced both on the public facade and on the factory/handle contract boundary.
  - Bedrock now also has the matching lower contract for stable request options, so provider-factory builder/config/registry construction directly guards `tool_choice`, structured-output `json` tool injection, and `additionalModelRequestFields` convergence on `/converse` instead of leaving that guarantee only on the public facade.
  - Registry builder / handle construction can now also inject Bedrock-specific build overrides without contaminating other providers in the same registry: a dedicated rerank-handle contract now proves `ProviderBuildOverrides` precedence on the split runtime vs agent-runtime route, including provider-specific Bearer auth and final `bedrock-agent-runtime` URL selection.

  - Rerank audit result is now closed for the currently advertised OpenAI-compatible catalog: focused provider packages (Cohere / TogetherAI / Bedrock) already had provider-owned parity anchors, and compat `siliconflow`, `jina`, and `voyageai` now also have top-level siumai/provider/config-first rerank parity coverage.
  - Shared OpenAI-compatible rerank execution now also preserves injected custom transports, fixing the public-path regression where compat rerank requests could bypass capture/no-network transport wiring even though chat, embedding, image, and audio paths already honored it.



- [~] Migrate MiniMaxi.

  - Provider-owned `MinimaxiConfig` / `MinimaxiClient` construction now converges through `MinimaxiBuilder::into_config()` and `MinimaxiClient::with_http_client()`, builder env API-key fallback is restored, `ModelMetadata` is now implemented on the typed client, registry factory construction now preserves common params, HTTP config, interceptors, and transport through the provider-owned config surface, `Provider::minimaxi()` is now exposed on the facade, no-network public-path parity tests now lock siumai/provider/config equivalence for non-streaming chat, streaming chat, image generation, TTS, video task creation/query, music generation, and MiniMaxi file-management upload/list/retrieve/content/delete paths, and registry-native text/image/speech family paths now materialize the provider-owned `MinimaxiClient` directly; full native family migration is still pending.

  - MiniMaxi now also owns provider-specific typed chat metadata via `provider_ext::minimaxi::metadata`, and its Anthropic-derived chat runtime locally rekeys response / stream-end / finish-event metadata from `provider_metadata["anthropic"]` to `provider_metadata["minimaxi"]` so provider escape hatches align with the actual public provider identity instead of the borrowed wire protocol.
  - MiniMaxi top-level builder/provider/config-first parity now also locks that provider-owned typed metadata on both 200-response and `StreamEnd`, and the same public-path guards assert the normalized root key directly (`provider_metadata["minimaxi"]` present, borrowed `provider_metadata["anthropic"]` absent) so facade construction cannot silently fall back to the wire-protocol namespace.
  - MiniMaxi registry `language_model("minimaxi:...")` now also locks the same typed metadata boundary on both 200-response and `StreamEnd`, including the normalized `provider_metadata["minimaxi"]` root, so registry-native text handles cannot regress back to the borrowed Anthropic namespace.
  - MiniMaxi `chat_stream_request` now also preserves `stream=true` on the provider-owned client path, fixing the override drift where the public streaming entry point had been issuing a non-streaming request body even though the transport/executor path was streaming.
  - MiniMaxi top-level public-path parity now also covers the provider-owned image and speech families directly: `Siumai::builder().minimaxi()`, `Provider::minimaxi()`, and config-first `MinimaxiClient` converge on the same `/v1/image_generation` and `/v1/t2a_v2` request shapes, and typed `MinimaxiTtsOptions` survive all three construction paths consistently.
  - MiniMaxi top-level public-path parity now also includes the public `registry.image_model("minimaxi:...")` handle on the image family, so builder/provider/config-first image generation and the registry-native image handle all converge on the same `/v1/image_generation` request body.
  - MiniMaxi typed TTS request helpers now also have a native registry speech-handle anchor: `registry.speech_model("minimaxi:...")` preserves `MinimaxiTtsOptions` against config-first `MinimaxiClient` on the final `/v1/t2a_v2` transport body instead of leaving typed speech options guarded only on the pre-registry public paths.
  - MiniMaxi top-level public-path parity now also includes the public `registry.language_model("minimaxi:...")` handle on video task creation, so builder/provider/config-first video generation and the registry-native extension handle all converge on the same `/v1/video_generation` request body.
  - MiniMaxi top-level public-path parity now also includes the public `registry.language_model("minimaxi:...")` handle on video task query and music generation, so builder/provider/config-first extension flows and the registry-native extension handle all converge on `/v1/query/video_generation` and `/v1/music_generation` too.
  - MiniMaxi top-level public-path parity now also covers the provider-owned video and music escape hatches directly: builder/provider/config-first construction converges on the same `/v1/video_generation`, `/v1/query/video_generation`, and `/v1/music_generation` public paths, and the typed `MinimaxiVideoRequestBuilder` / `MinimaxiMusicRequestBuilder` helpers now have transport-boundary anchors on the real provider-owned story.
  - MiniMaxi file-management public paths now also lock builder/provider/config-first equivalence for `/v1/files/upload`, `/v1/files/list`, `/v1/files/retrieve`, `/v1/files/retrieve_content`, and `/v1/files/delete`, including the provider-specific base-URL normalization from `.../anthropic/v1` back to the API root for file endpoints.
  - MiniMaxi top-level public-path parity now also includes the public `registry.language_model("minimaxi:...")` handle on file-management operations, so upload/list/retrieve/content/delete all converge across builder/provider/config-first and registry-native extension paths on the same `/v1/files/*` endpoints.
  - MiniMaxi OpenAI-style secondary endpoints now also preserve the caller-supplied host during base-URL normalization: `image`, `speech`, `video`, and `music` no longer jump to the hard-coded production domain when public construction starts from a custom `.../anthropic/v1` base URL, so local proxies, staging gateways, and no-network parity tests all observe the same `/v1/...` routing rule.
  - Registry-native MiniMaxi extension capability routing is now tighter as well: `LanguageModelHandle` forwards video/music/file-management capability calls instead of dropping them at the generic handle layer, generic registry-entry tests now lock that delegation boundary directly, and MiniMaxi’s provider-owned file surface is no longer limited to builder/provider/config-first clients.
  - MiniMaxi now also has direct builder/config/registry contract anchors for provider-owned chat, chat-stream, image, speech, and file-management request shaping, so the provider factory path itself now locks auth propagation, SSE headers, secondary-endpoint routing, typed TTS / image request bodies, and `/v1/files/*` normalization instead of depending only on top-level public-path parity.
  - Registry builder / handle construction can now inject MiniMaxi provider-owned build overrides directly as well: `RegistryBuilder::with_api_key(...)`, `with_base_url(...)`, `with_http_client(...)`, and `fetch(...)` all flow into the final factory `BuildContext`, while `ProviderBuildOverrides` now gives MiniMaxi a provider-scoped precedence lane inside mixed registries, and the no-network MiniMaxi registry-handle contract now proves that the public registry path can reach the same provider-owned chat request boundary without hand-assembling `BuildContext`.
  - MiniMaxi now also has a top-level public-path anchor for that same provider-scoped override lane on file management: `RegistryBuilder::with_provider_build_overrides("minimaxi", ...)` drives `language_model("minimaxi:...").upload_file(...)` to the provider-specific host/auth pair, so mixed registries cannot silently fall back to global overrides on `/v1/files/upload`.
  - The same MiniMaxi file-management override lane is now also pinned at the lower `siumai-registry` contract layer, so `language_model("minimaxi:...")` file operations keep provider-scoped host/auth precedence both on the public facade and on the factory/handle boundary.
  - MiniMaxi file-management override coverage now also extends across `list_files`, `retrieve_file`, `get_file_content`, and `delete_file`, so mixed registries cannot silently fall back to global defaults on the rest of the `/v1/files/*` surface after `upload_file` was already pinned.
  - MiniMaxi now also has matching public-path and lower-contract override anchors for video generation on `language_model("minimaxi:...")`, so mixed registries cannot silently fall back to global defaults when extension calls normalize `.../anthropic/v1` onto `/v1/video_generation`.
  - MiniMaxi now also has matching public-path and lower-contract override anchors for query-video generation on `language_model("minimaxi:...")`, so mixed registries cannot silently fall back to global defaults when extension calls normalize `.../anthropic/v1` onto `/v1/query/video_generation`.
  - MiniMaxi now also has matching public-path and lower-contract override anchors for music generation on `language_model("minimaxi:...")`, so mixed registries cannot silently fall back to global defaults when extension calls normalize `.../anthropic/v1` onto `/v1/music_generation`.
  - MiniMaxi now also has matching public-path and lower-contract override anchors for image generation on `registry.image_model("minimaxi:...")`, so mixed registries cannot silently fall back to global defaults when the image family normalizes onto `/v1/image_generation`.
  - MiniMaxi now also has matching public-path and lower-contract override anchors for speech generation on `registry.speech_model("minimaxi:...")`, so mixed registries cannot silently fall back to global defaults when the speech family normalizes onto `/v1/t2a_v2`.



- [~] Migrate DeepSeek.

  - Provider-owned `DeepSeekConfig` / `DeepSeekClient` entry points, registry-native text-family routing, deepseek-only feature wiring, unified builder routing, provider-owned typed request options (`DeepSeekOptions` / `DeepSeekChatRequestExt`), provider-owned typed response metadata (`DeepSeekMetadata` / `DeepSeekChatResponseExt`), a provider-owned `DeepSeekSpec` normalization layer for chat request options, and a first-class `ProviderType::DeepSeek` are now in place; registry and unified builder paths now also materialize the provider-owned `DeepSeekClient` instead of leaking the generic OpenAI-compatible runtime, while full native provider migration is still pending.
  - DeepSeek provider-local spec tests now also consume `DeepSeekChatResponseExt` for `logprobs`, removing the last response-side raw `provider_metadata["deepseek"]` assertion from the native package’s request/response normalization coverage.
  - DeepSeek non-text routes are now explicitly locked as deferred instead of ambiguously half-exposed: factory/catalog capability metadata stays text-family-only, and no-network builder/provider/config tests now assert `embedding`, `rerank`, and `image_generation` fail with `UnsupportedOperation` before any HTTP request is emitted.
  - DeepSeek's lower-level factory boundary is now explicit too: deferred `embedding` / `image` / `speech` / `transcription` / `rerank` family builders return `UnsupportedOperation` directly instead of delegating through the generic text client, and both registry-handle contract tests plus public `registry.*_model("deepseek:...")` negative-path tests now pin the same no-request failure path.
  - Registry builder / handle construction now also has a provider-scoped DeepSeek chat anchor: `ProviderBuildOverrides` precedence is locked on the real `language_model("deepseek:deepseek-chat")` path, so mixed registries can route DeepSeek-specific API keys, base URLs, and transports to the final `/chat/completions` boundary without leaking global overrides into adjacent providers.
  - DeepSeek top-level public registry parity now also has explicit request-shape anchors for both non-streaming and streaming `chat_request` on `registry.language_model("deepseek:...")`, so the registry path no longer relies only on stable-options or metadata tests to prove final `/chat/completions` routing and normalized reasoning fields.
  - DeepSeek top-level public registry parity now also covers provider-option `tool_choice` shaping directly on `registry.language_model("deepseek:...")`, so the public facade no longer leaves that normalization guarantee solely to the lower builder/config/registry contract tests.
  - DeepSeek now also has the matching top-level public override anchor on `registry.language_model("deepseek:...")`, so `RegistryBuilder::with_provider_build_overrides("deepseek", ...)` is pinned on the real `/chat/completions` path instead of existing only at the lower `siumai-registry` contract layer.
  - DeepSeek now also has direct builder/config/registry contract anchors for typed stable request options and provider-option tool-choice shaping, so the provider factory path itself now locks `DeepSeekOptions` reasoning normalization plus tool-choice/body convergence instead of leaving those guarantees only on the public facade.







## Track G - Public API cleanup







- [x] Ensure `siumai::text`, `embedding`, `image`, `rerank`, `speech`, and `transcription` are the primary documented entry points.
  - Root README, `siumai/examples/README.md`, and the public API story now all name the six family modules explicitly as the default public entry points for new code, while provider-specific and compat surfaces are described as secondary by intent.



- [x] Keep `provider_ext::<provider>` for long-term provider-specific features.
  - Public API guidance now keeps provider-specific request/response escape hatches under `provider_ext::<provider>` and does not redirect new provider features into `compat`.



- [x] Mark `compat` APIs as time-bounded.
  - `siumai::compat` module docs, top-level crate docs, and the public API story now all state the compatibility-only role explicitly and pin a documented removal target of no earlier than `0.12.0`.



- [x] Prevent new features from landing only in legacy compatibility surfaces.
  - Public API guidance now states that new capabilities must land on family APIs, config-first provider clients, or `provider_ext` first; `compat` may mirror them later but is no longer the only allowed public home.



- [x] Review `prelude::unified` exports for clarity and minimality.
  - `prelude::unified` docs now describe the stable-family-centered contract accurately, compatibility aliases inside that module are doc-hidden instead of looking like first-class defaults, and `prelude::compat` now provides an explicit migration-oriented import path with compile coverage.







## Track H - Tests and safety rails







- [x] Keep fixture/alignment tests as migration safety nets.
  - Provider-local regression guards now explicitly pin root-cause request-shape invariants too: `siumai-provider-openai` now has focused no-network tests proving `chat_stream_request(request)` forces `stream=true` plus `stream_options.include_usage=true` on both Chat Completions and Responses request-based stream paths, and a direct Responses reasoning-stream regression now also locks stable `ThinkingDelta` plus final `ContentPart::Reasoning` metadata (`itemId` / `reasoningEncryptedContent`) on the provider-owned client path; native OpenAI full-request chat now also routes both `chat_request(...)` and `chat_stream_request(...)` through a provider-local `prepare_chat_request(...)` merge step so sparse request-level `CommonParams` inherit config/builder defaults (`model`, `temperature`, `max_tokens`, and other shared fields) instead of only merging provider options, focused unit tests now pin both fallback and explicit-override behavior on that direct path, and a targeted proxy-backed real smoke against `gpt-5.2` confirms both explicit-request-model and builder-default-model `chat_request(...)` flows return `OK`; `siumai-provider-openai-compatible` now has focused unit coverage proving shared-runtime `prepare_chat_request(..., stream)` flips the stream lane, backfills config-level model/http defaults, and preserves explicit request models on the non-stream lane before executor selection; `siumai-provider-amazon-bedrock` plus `siumai-provider-azure` now have focused unit coverage proving `prepare_chat_request(..., stream)` flips the stream lane, backfills the configured default model/deployment, and preserves explicit request models on the non-stream lane before executor selection; `siumai-provider-gemini` has focused unit coverage proving `prepare_chat_request(..., stream)` flips the stream lane and fills client defaults before executor selection, and now also has focused no-network client regressions on the native `:streamGenerateContent?alt=sse` lane proving request-based `chat_stream_request(...)` preserves Stable `response_format -> responseMimeType/responseSchema` shaping while splitting complete accumulated JSON plus final `StreamEnd` metadata from interrupted incomplete JSON on the provider-owned structured-output path; `siumai-provider-google-vertex` now also has matching no-network client regressions on that same native stream lane, locking both the request-side `Accept: text/event-stream` plus Stable `response_format` / `tool_choice` / `thinkingConfig` / `structuredOutputs` contract and the response-side split between complete accumulated JSON with final `provider_metadata["vertex"]` `StreamEnd` metadata versus truncated interrupted JSON on the provider-owned structured-output path; `siumai-provider-google-vertex`'s `anthropic-vertex` wrapper now also has provider-local regressions proving request-based `chat_request/chat_stream_request` no longer fall back to the message/tool-only trait default, and that explicit request models reach the real `:rawPredict` / `:streamRawPredict` URL boundary; and `siumai-provider-ollama` now has focused no-network client tests proving request-based `chat_stream_request(...)` forces the native NDJSON `/api/chat` lane to `stream=true` while preserving Stable `response_format -> format` precedence, plus fail-fast coverage for unsupported `ToolChoice::Required` before any transport call, so registry/public parity fixes cannot silently regress below the provider-owned client layer. The matching registry-side root cause is now locked too: `LanguageModelHandle` only backfills `request.model` when it is absent, and focused public/lower-contract parity now anchors explicit full-request model overrides across `anthropic-vertex`, `azure`, `deepseek`, `groq`, and `xai` config-first plus registry text-handle paths; `groq` / `xai` now close the OpenAI-compatible representative lane instead of leaving that guard anchored on a single compat sample. Compat vendor views now also lock the same boundary directly: OpenRouter and Perplexity preserve explicit full-request `model` overrides while still merging vendor options on both public-path and builder/config/registry contract tests, provider-local stream regressions now pin `Accept: text/event-stream`, Stable `response_format` / `tool_choice`, plus vendor-specific OpenRouter reasoning defaults or Perplexity typed search options on the final shared-runtime transport body without adding an extra adapter rewrite branch, and representative non-text runtime regressions now also pin Fireworks embedding on the inference host, Infini embedding on `/embeddings`, Together embedding/image on `/embeddings` plus `/images/generations`, and SiliconFlow/Jina/VoyageAI rerank plus SiliconFlow image on `/rerank` or `/images/generations`, so shared compat embedding/image/rerank executors all have direct provider-local transport-boundary anchors too.



- [x] Add focused contract tests for the initial `LanguageModel` bridge/native paths.



- [x] Add focused registry tests for model handle behavior.



- [x] Add no-network unit tests for `ToolChoice` wire-format mapping (OpenAI, Anthropic, Gemini).



- [x] Add no-network tests for tool loop invariants (tool-call/tool-result parts across core providers).



- [x] Add no-network Anthropic structured-output parity tests for reserved `json` tool fallback (response + streaming).
- [x] Move Anthropic container-metadata fixture assertions onto `AnthropicChatResponseExt` where the response surface is already typed.
  - Anthropic message + streaming fixture checks for container metadata now use typed `AnthropicMetadata.container` on `ChatResponse` / `StreamEnd`, reducing raw `provider_metadata["anthropic"]` traversal in the regression suite while keeping event-level custom payload assertions unchanged.
  - Anthropic thinking replay metadata is now also part of the typed response surface: `AnthropicMetadata` models `thinking_signature` and `redacted_thinking_data`, protocol response/stream-end tests read those fields through `AnthropicChatResponseExt`, and `assistant_message_with_thinking_metadata(...)` now consumes typed metadata instead of manually traversing the raw provider map.
  - Anthropic tool-call metadata now also has a typed response-side escape hatch: `AnthropicContentPartExt::anthropic_tool_call_metadata()` exposes `caller` on `ContentPart::ToolCall`, the programmatic tool-calling fixture uses it instead of manual nested-map traversal, and protocol streaming tests now read `contextManagement` / `sources` from `AnthropicMetadata` on `StreamEnd` rather than reopening raw provider maps.
  - Anthropic now also has top-level public-path parity coverage for provider-owned request helpers: builder/provider/config-first clients converge on the same final request body for typed thinking, Stable `response_format`, and Stable `tool_choice`, public-surface imports now lock `AnthropicChatRequestExt` plus the typed option structs, and the duplicated `chat_before_send` application bug in Anthropic-style HTTP chat clients has been removed so additive request rewrites no longer run twice at runtime. A follow-up openai-compatible regression guard now also pins the corrected executor contract: runtime provider request rewrites stay reachable through `provider_spec.chat_before_send(...)` instead of being duplicated into `exec.policy.before_send`.
  - Anthropic top-level builder/provider/config-first parity now also locks `AnthropicMetadata.container` on both 200-response and `StreamEnd`, so provider-owned typed metadata is exercised on the real public construction surface rather than only in protocol fixture tests; the same guard also keeps `service_tier` on the top-level `ChatResponse` field instead of pretending it is vendor metadata.
  - Anthropic registry `language_model("anthropic:...")` now also locks the same `AnthropicMetadata.container` boundary on both 200-response and `StreamEnd`, so registry-native text handles stay aligned with config-first `AnthropicClient` instead of relying only on top-level parity.
  - That registry stream-end parity also closed a real config-path gap: `AnthropicClient::chat_stream_request(...)` now forces `stream=true` before executor dispatch, so config-first and registry-native streaming metadata tests no longer diverge when callers start from a plain non-streaming `ChatRequest`.
  - Anthropic's public capability split is now explicit too: the top-level `Siumai` facade fail-fast rejects `embedding`, `image_generation`, and `reranking` without emitting transport traffic, provider/config clients expose no deferred non-text capability handles, and registry-native `embedding_model("anthropic:...")` / `image_model("anthropic:...")` / `reranking_model("anthropic:...")` now match that same unsupported contract.
  - Anthropic's lower-level factory boundary is now explicit too: deferred `embedding` / `image` / `speech` / `transcription` / `rerank` family builders return `UnsupportedOperation` directly instead of delegating through the generic text client, and registry-handle contract tests now pin the same no-request failure path.
  - OpenAI Responses streaming now also documents the same boundary explicitly: finish / reasoning custom events keep raw `providerMetadata` assertions because they are event-level payload contracts, while stable `StreamEnd` metadata uses typed helpers only where fields are actually promoted today; top-level builder/provider/config-first parity now verifies typed `OpenAiSourceExt` metadata on final Responses `StreamEnd` payloads and typed `OpenAiChatResponseExt.logprobs` on Chat Completions `StreamEnd`, and Azure stream alignment tests likewise remain raw because they validate namespace-key rewriting on custom parts rather than typed response metadata.
  - OpenAI non-stream response parts now also have a narrow typed escape hatch: `OpenAiContentPartExt` exposes `itemId` and `reasoningEncryptedContent` on `ContentPart`, and protocol transformer regression coverage now validates the helper against a real Responses reasoning item instead of metadata-only fixture construction.
  - OpenAI source annotations now also have the same narrow typed escape hatch: `OpenAiSourceExt` exposes `fileId` / `containerId` / `index` from `OpenAiSource.provider_metadata`, protocol transformer regressions cover real `file_citation`, `container_file_citation`, and `file_path` annotations instead of raw nested map reads, and top-level builder/provider/config-first parity now verifies the same typed source extraction on real Responses 200-response payloads while the shared compat helper now also carries Chat Completions `logprobs` into final `OpenAiChatResponseExt` metadata on both 200-response and `StreamEnd`.
  - OpenAI Responses SSE source annotation mapping is now also pinned on both directions: `response.output_text.annotation.added` for `container_file_citation` / `file_path` is covered in converter tests, and `openai:source` serialization back to those annotation shapes is covered without widening the event-level raw metadata boundary.
  - OpenAI Responses SSE finish aggregation now also preserves message-level source annotations into the final `response.completed` payload: serializer state retains `openai:source` annotations until finish, and a round-trip regression validates that `containerId` / `index` survive through the completed response transformer into typed `OpenAiSourceExt` metadata.
  - The V4 typed metadata boundary is now also documented in one place: `typed-metadata-boundary-matrix.md` tracks which providers currently own typed `ChatResponse`, `ContentPart`, and `Source` escape hatches, and which paths intentionally remain raw event contracts.

- [ ] Use `typed-metadata-boundary-matrix.md` as the canonical backlog driver for the next response-side typing pass.
  - Priority A: Perplexity hosted-search metadata audit is now closed on the current typed boundary: public-surface imports expose `PerplexityChatResponseExt` / `PerplexityMetadata`, provider-local runtime coverage already locks `citations`, `images`, and `usage.{citation_tokens,num_search_queries,reasoning_tokens}`, and top-level builder/provider/config-first parity now also verifies the same typed extraction plus normalized `provider_metadata["perplexity"]` roots on both 200-response and `StreamEnd` without forcing speculative extra fields into the stable surface.
  - Priority A: Anthropic / MiniMaxi source-boundary audit is now closed on the current typed surface: `AnthropicMetadata.sources` / `MinimaxiMetadata.sources` already expose typed `AnthropicSource`-shaped entries directly on the stable facade, so no extra `SourceExt` helper is needed unless a future provider-specific source payload grows beyond the shared struct.
  - Priority B audit is now closed for xAI / Groq / DeepSeek: xAI hosted-search metadata is already closed on the current typed boundary through `XaiMetadata.sources` plus `XaiSourceExt`, while streaming search details remain event-level custom parts; Groq's in-repo fixtures still justify only `logprobs` / `sources` typed metadata, and DeepSeek already normalizes `reasoning_content` into unified `ContentPart::Reasoning`, so no new provider-owned `ContentPart` helper was added.
  - OpenAI / Azure public-path parity has now been tightened too: `Provider` exposes explicit `openai_responses`, `openai_chat`, `azure`, and `azure_chat` entry points, compile guards cover those top-level constructors, and no-network parity tests now lock builder/provider/config convergence for both Responses and Chat Completions request routes instead of relying on builder-only routing smoke tests.
  - Azure request-side capability parity is now also closed for this pass: `AzureOpenAiSpec` now owns `chat_before_send` request shaping for reasoning knobs plus Stable `response_format` / `tool_choice` semantics on both Responses and Chat Completions routes, and the duplicate client-side before-send injection has been removed so those additive rewrites run exactly once at the final transport boundary.
  - Google / Gemini request-side capability parity is now also closed for this pass: no-network public-path tests now lock builder/provider/config convergence for Gemini chat plus embedding routes, public-surface import guards now pin `GeminiChatRequestExt` together with the typed thinking options, and Stable `response_format` / `tool_choice` now converge with provider-owned `thinkingConfig` on the final `generateContent` request body.
  - Google / Gemini top-level public-path parity now also locks the mixed typed/raw metadata boundary: builder/provider/config-first guards verify typed `GeminiContentPartExt` / `GeminiChatResponseExt` extraction on 200-response and `StreamEnd`, while streaming `reasoning-start` custom payloads still keep raw `providerMetadata.google.thoughtSignature` plus normalized `provider_metadata["google"]` roots; the root-cause stream gap is fixed because `GeminiClient` now normalizes `stream=true` before choosing its executor bundle.
  - Bedrock request-side capability parity is now also closed for this pass: no-network public-path tests now lock builder/provider/config convergence for Stable `response_format` / `tool_choice` on the provider-owned Converse route, including the reserved `json` tool injection and final `toolConfig` / `additionalModelRequestFields` body shape at the transport boundary.
  - Google Vertex request-side capability parity is now also closed for this pass: no-network public-path tests now lock builder/provider/config convergence for Stable `response_format` / `tool_choice` plus provider-owned `thinkingConfig` on the final Vertex `generateContent` request body, and the root-cause wrapper gap is fixed because `GoogleVertexClient` now overrides the default `chat_request` / `chat_stream_request` fallback that previously dropped request-level fields by collapsing back to `messages + tools`.
  - Ollama public-path parity is now also closed for this pass: top-level no-network tests already lock builder/provider/config convergence for chat, chat-stream, and embedding request shaping, while the public-surface compile guard now pins `OllamaChatRequestExt`, `OllamaEmbeddingRequestExt`, and `OllamaChatResponseExt` on the stable facade.
  - Ollama registry `language_model("ollama:...")` now also locks provider-owned timing metadata on both 200-response and JSON-stream `StreamEnd`, so registry-native text handles carry the same `OllamaChatResponseExt` contract as config-first `OllamaClient`.
  - Ollama's focused capability split is now explicit as well: embeddings remain the only public non-text family, top-level wrapper paths reject `image_generation` and `reranking` before transport dispatch, and registry-native `image_model("ollama:...")` / `reranking_model("ollama:...")` now lock the same unsupported-family contract instead of leaving those paths implicitly deferred.
  - Ollama's lower-level factory boundary is now explicit too: deferred `image` / `speech` / `transcription` / `rerank` family builders return `UnsupportedOperation` directly instead of falling back through the text client, and registry-handle contract tests now pin the same no-request failure path.
  - MiniMaxi request-side capability parity is now also closed for Stable `tool_choice`: dedicated MiniMaxi fixture coverage now locks Anthropic-shaped `required -> {\"type\":\"any\"}` request mapping on the provider-owned route, `MinimaxiClient` now overrides the default full-request fallback so `chat_request` / `chat_stream_request` no longer collapse back to `messages + tools`, and no-network public-path parity now confirms builder/provider/config convergence for that final `tool_choice` body shape.
  - MiniMaxi's capability split is now explicit too: image + speech remain the public non-text families, while `embedding`, `reranking`, and `transcription` now fail fast across top-level siumai/provider/config paths and the registry-native `embedding_model("minimaxi:...")` / `reranking_model("minimaxi:...")` / `transcription_model("minimaxi:...")` handles all match that same unsupported boundary.
  - MiniMaxi's lower-level factory boundary is now explicit too: deferred `embedding` / `transcription` / `rerank` family builders return `UnsupportedOperation` directly instead of delegating through the text client, and registry-handle contract tests now pin the same no-request failure path below the public facade.
  - Priority B: keep OpenRouter's broader hosted-search story deferred, but treat the current alias-based vendor metadata surface as closed: public-path parity already locks `OpenRouterChatResponseExt` / `OpenRouterSourceExt` / `OpenRouterContentPartExt` over the shared compat payload on both 200-response and `StreamEnd`, so only widen beyond `logprobs` / `sources` if stronger provider-specific response evidence appears.
  - Priority C: keep Azure response/event metadata and `google-vertex` explicitly raw/options-focused until there is a stronger stable contract than today's namespace/event assertions.
  - Priority C: Azure hosted-search metadata should now be treated as `Deferred`, not `Partial`: the current public contract is the raw `provider_metadata["azure"]` stream/event boundary, and that boundary already has top-level builder/provider/config-first parity coverage.
  - Priority C: Azure top-level builder/provider/config-first parity now also locks the Responses streaming raw/event boundary on the real public path: `text-start` / `finish` custom parts and final `provider_metadata["azure"]` roots are verified to stay namespace-scoped raw metadata rather than a widened typed response surface.
  - Priority C: Google Vertex top-level builder/provider/config-first parity now also locks the Gemini-stream raw/event boundary on the real public path: `reasoning-start` / `reasoning-delta` custom parts must keep `providerMetadata.vertex.thoughtSignature`, and final `provider_metadata["vertex"]` roots must stay namespace-led raw metadata rather than a provider-specific typed facade.
- [ ] Use `provider-capability-alignment-matrix.md` as the canonical backlog driver for the next provider capability alignment pass.
  - Current release-gate snapshot: no provider-wide `Priority A` capability gap is currently queued on the advertised public facade; the remaining open cells in the matrix are now intentional `Deferred` / focused-boundary decisions unless new provider evidence appears.
  - Priority A: Perplexity hosted-search parity is now considered closed at the current typed boundary; top-level builder/provider/config-first guards now also anchor the stable response-side contract on both 200-response and `StreamEnd`, so only widen beyond `citations`, `images`, and usage-side search/reasoning counters if new fixtures or runtime captures show a stronger stable provider contract.
  - Priority A: Anthropic hosted-search/source typing is now considered closed on the current facade boundary; MiniMaxi request-side parity is also closed for Stable `tool_choice`, so only revisit it if the provider grows a genuinely divergent source or tool-routing contract beyond the current Anthropic-shaped path.
  - Priority B: keep xAI / Groq / DeepSeek focused on parity maintenance unless a concrete new stable capability gap appears.
  - Priority B: keep OpenRouter hosted-search unification deferred, but do not regress the current typed response metadata story: the vendor-owned alias surface for `logprobs` / `sources` is already part of the public compat contract and should now be maintained rather than treated as speculative.
  - Priority B: treat Ollama public-path parity as closed unless a new top-level family route or wider stable response contract is intentionally introduced.
  - Priority B: Ollama tool-choice semantics are now explicit on the current native boundary: `ToolChoice::None` still omits tools, while `Required` and specific-tool forcing now fail fast with `UnsupportedOperation` across builder/provider/config/registry paths instead of silently degrading to `Auto`.
  - Priority C: treat Cohere / TogetherAI as rerank-led native packages with public-path parity already closed unless product scope changes.



- [x] Add OpenAI-compatible runtime-provider tool loop parity coverage for xAI streaming/non-streaming equivalence.



- [x] Add DeepSeek runtime-provider smoke coverage for OpenAI-compatible `response_format`, `tool_choice`, and streaming/non-streaming tool-call equivalence.



- [x] Add no-network coverage for OpenAI-compatible full-request provider-options preservation and DeepSeek request-option normalization.



- [x] Add Ollama tool loop parity coverage for tool-result request shaping plus non-streaming/streaming tool-call equivalence.

- [x] Map Ollama `ToolChoice::None` to tool omission in the final request body.
  - No-network public-path parity now also covers builder / provider / config-first / registry construction for all three native Ollama tool-choice branches: `None` omits tools, while `Required` and specific-tool forcing fail fast with `UnsupportedOperation` before any request is emitted.



- [x] Add provider parity tests for old path vs new path during migration.



- [x] Add smoke tests for builder/config/registry equivalence on major providers.



  - xAI, Groq, and DeepSeek now have no-network parity coverage that compares final `chat_request` / `chat_stream_request` transport payloads across provider builder, provider config-first client, and registry factory construction paths.



- [x] Add top-level public-path parity coverage for `Siumai::builder()` / `Provider::*()` / provider config-first clients on xAI, Groq, and DeepSeek.



  - The unified `Siumai` wrapper now also delegates full `chat_request` / `chat_stream_request` objects directly to provider chat capabilities instead of falling back to message/tool-only defaults, so request-level provider options and model overrides survive the top-level compatibility surface.
  - DeepSeek now also has a dedicated top-level parity anchor for builder-level reasoning defaults, and the underlying registry factory now consumes `BuildContext.reasoning_enabled` / `reasoning_budget`, closing the drift where `Siumai::builder().deepseek()` had silently dropped default reasoning knobs that provider/config-first construction already preserved.
  - DeepSeek and xAI now also lock the new registry-global reasoning lane on their provider-owned wrapper stories: no-network public-path parity compares config-first defaults with both `RegistryOptions`-driven and direct `RegistryBuilder`-driven `language_model(...)` construction, so the shared global-default path is now exercised on both compat vendor views and native wrapper providers.







## Track I - Documentation







- [x] Update architecture docs to show family-model-first design.



- [x] Update migration docs with construction guidance.



- [x] Update README examples to prefer registry-first or config-first construction.



- [x] Keep one explicit builder example for convenience and comparison.



- [x] Document which surfaces are stable, extension, or compatibility-only.



- [x] Update runnable examples to avoid builder-default construction unless explicitly labeled as compatibility demos.

- [x] Add rerank family examples for registry-first plus config-first Cohere / TogetherAI / Bedrock paths.



- [x] Add a provider feature matrix for cross-provider alignment work.
  - The matrix legend now treats audited fail-fast Stable-family boundaries as aligned work, so `Embedding`, `Image generation`, and the audited `Rerank` negative-path columns no longer under-report native provider coverage when the real decision is “intentionally unsupported before transport”.
  - `Audio split` now follows the same rule on audited native providers too: Anthropic, Gemini, xAI, and Ollama now have explicit public-path and/or registry audio-family rejection coverage, so the matrix no longer conflates “provider-owned audio semantics” with “must expose Stable speech/transcription families”.
  - `Rerank` is now closed under that same rule too: OpenAI native rerank is pinned to the canonical `api.openai.com` negative path, `registry.reranking_model("openai:...")` no longer constructs a dead-end handle, compat rerank stays on enrolled OpenAI-compatible vendors, and Gemini now also has explicit public-path plus registry-handle rejection coverage instead of an ambiguous blank cell.
  - Registry family handles now follow that same fail-fast contract beyond rerank as well: unsupported native `embedding_model`, `image_model`, `speech_model`, and `transcription_model` entries reject construction up front instead of leaking dead-end handles that only fail on first use.
  - Focused OpenAI-compatible presets now also lock the same rule on the generic registry text entry: `registry.language_model("jina:...")` and `registry.language_model("voyageai:...")` reject construction up front, while mixed-surface `infini` keeps the positive text handle and remains the reference case for “chat + embedding can coexist on one preset”.
  - The generic registry text entry now uses the same rule for all focused non-chat providers, not just OpenAI-compatible presets: rerank-only `cohere` / `togetherai` entries now reject `registry.language_model("provider:model")` construction up front, while chat-capable mixed surfaces like `infini` still keep the positive path.

- [x] Add a validation matrix and wire CI gates to package boundaries.
  - `validation-matrix.md` now defines Tier 0 / Tier 1 / Tier 2 / Tier 3 validation lanes so the post-refactor phase is governed by explicit package-boundary gates instead of only ad hoc full-workspace runs.
  - PR CI now compiles the `siumai` facade under every first-class provider feature (`openai`, `azure`, `anthropic`, `google`, `google-vertex`, `ollama`, `xai`, `groq`, `minimaxi`, `deepseek`, `cohere`, `togetherai`, `bedrock`) instead of only the earlier narrow subset.
  - PR CI now also runs provider-scoped no-network nextest bundles for those same facade lanes, including a separate `openai-compat` vendor/preset lane so compile-only feature smoke is backed by executable request/response contract coverage.
  - PR CI now also runs cross-feature no-network nextest bundles for the current multi-feature coupling set: `openai,openai-websocket`, `google,gcp`, and `openai,json-repair`.
  - The `openai,json-repair` bundle now also pins the refusal/content-filter contract: structured-output helpers must keep returning the dedicated parse error when no valid JSON was actually produced, even if JSON repair is enabled.
  - CI now also compiles each provider package directly under its own feature gate, including the compatibility packages, so provider-owned public surfaces can no longer regress silently behind the top-level facade build.
  - Local smoke/test scripts now follow that same matrix more closely by including the OpenAI-compatible package plus the focused provider packages in the `openai-compatible` / `all-providers` presets.

- [x] Add a public API story doc that assigns registry-first, config-first, `provider_ext`, and `compat` surfaces to clear roles.



- [x] Add a structured output parity note (`structured-output-parity.md`) and keep it in sync with request transformer behavior.
  - Stable structured-output failure semantics are now explicit too: helpers still do best-effort JSON repair for explicit responses / explicit `StreamEnd`, but interrupted streams without `StreamEnd` now require a complete JSON value and refusal/content-filter endings now return a dedicated parse error instead of silently falling back to generic "no JSON" behavior.
  - Refusal/content-filter endings now also outrank best-effort JSON repair on non-streaming responses unless a complete strict JSON value or explicit reserved `json` tool payload already exists, which closes the `openai,json-repair` regression where plain refusal text could be misclassified as a repaired JSON string.
  - Interrupted reserved-`json` tool streams now also converge on that same stable error wording: if tool arguments are truncated before `StreamEnd`, the public facade returns the dedicated incomplete-stream parse error instead of leaking a lower-level generic parse failure.
  - Typed structured-output helpers now also distinguish the second failure stage in a stable way: once JSON extraction succeeds, deserialization mismatch is reported as a target-type failure instead of being conflated with missing/invalid JSON extraction.
  - Public `siumai::structured_output` integration coverage now pins direct delegation to `siumai-core`, so the facade cannot drift on typed mismatch or interrupted-stream failure semantics.
  - Synthetic interrupted `StreamEnd` paths are now pinned too: if a provider/runtime closes streaming with `FinishReason::Unknown`, accumulated deltas / reserved-tool payloads are reused so complete JSON still parses while truncated output returns the same dedicated incomplete-stream error; Ollama public-path parity is the reference coverage for that rule.
  - The same synthetic-end rule is now locked on the shared OpenAI-compatible SSE route as well: xAI public-path parity covers builder/provider/config/registry streaming extraction when the SSE stream terminates without a provider-native finish frame, proving the contract is not limited to native NDJSON providers.
  - Vendor-view compat providers now also carry that same rule on the real public shortcuts: Perplexity parity covers `Siumai::builder().openai().perplexity()`, `Provider::openai().perplexity()`, config-first `OpenAiCompatibleClient`, and registry streaming extraction for the same synthetic-end success/failure split.
  - OpenRouter now also satisfies that same vendor-view guard on its real public shortcuts: builder/provider/config/registry streaming extraction pins the synthetic-end complete-vs-truncated structured-output split while preserving OpenRouter-specific request params such as `transforms`.
  - Provider-owned reserved-tool streaming now also has a public-path anchor: Bedrock parity covers builder/provider/config streaming extraction on the Converse JSON stream when reserved `json` tool deltas complete successfully versus terminate in a truncated state before `messageStop`.
  - Anthropic now also has the SSE fallback variant of that public-path anchor: forcing `providerOptions.anthropic.structuredOutputMode = "jsonTool"` pins builder/provider/config/registry extraction when reserved `json` tool deltas are already complete versus truncated before `message_stop`.
  - Anthropic now also has the matching provider-local regression on the real provider-owned client path: `AnthropicClient::chat_stream_request(...)` is exercised against interrupted SSE `jsonTool` payloads, locking both the successful complete-JSON extraction branch and the dedicated incomplete-stream parse error branch while keeping the final request body on the reserved-tool fallback shape (`tool_choice.any`, reserved `json` tool, no native `output_format`, no structured-outputs beta header).
  - Anthropic now also exposes that fallback knob as a typed provider option (`AnthropicStructuredOutputMode` via `AnthropicOptions::with_structured_output_mode(...)`), so public examples and future tests can stop reaching for raw provider-option JSON.
  - Anthropic builder-first/config-first/shared-builder construction now also carries provider-owned typed default request options through namespaced fluent helpers (`with_anthropic_options`, `with_anthropic_thinking_mode`, `with_anthropic_structured_output_mode`, `with_anthropic_context_management`, `with_anthropic_tool_streaming`, `with_anthropic_effort`, `with_anthropic_container`), so defaults for `thinking_mode`, reserved-`json` tool fallback, context management, effort, and streaming beta opt-out no longer have to be restated on every request; focused no-network public-path parity now locks the final `/v1/messages` SSE request across `Siumai::builder()`, `Provider::anthropic()`, and `AnthropicClient::from_config(...)`, including thinking-budget token expansion, reserved `json` tool injection, context-management/output-config mapping, and `tool_streaming = false` beta suppression.
  - xAI speech now also has the same typed-helper cleanup on its provider-owned audio path: `XaiTtsOptions` plus `XaiTtsRequestExt` cover `sample_rate` / `bit_rate`, so the `xai-tts` example and public TTS parity no longer rely on raw `with_provider_option(...)`.



- [x] Map Ollama Stable `response_format` to `/api/chat` `format` with request-level override tests.



- [x] Add Groq provider-level smoke tests for OpenAI-style `response_format` passthrough and tool-call response mapping.



- [x] Add xAI runtime-provider smoke tests for OpenAI-compatible `response_format` + `tool_choice` semantics.



- [x] Add Groq wrapper no-network coverage for full-request provider-options preservation.



- [x] Document provider-owned typed response metadata escape hatches for xAI and Groq.



- [x] Add registry contract coverage for provider-owned xAI and Groq wrapper materialization plus public-surface compile checks for their typed metadata exports.







## Deferred items







- [-] Do not rename existing spec types just to mirror AI SDK naming.



- [-] Do not split tools into many additional crates.



- [-] Do not remove builders immediately.



- [-] Do not attempt a single-PR migration of every provider.







## Current spike status







- [x] `ModelMetadata` and `ModelSpecVersion` added.



- [x] Minimal `LanguageModel` trait added as a family-model bridge.



- [x] `LanguageModelHandle` now satisfies model metadata semantics.



- [x] `ProviderFactory` now has a parallel text-family-returning interface.



- [x] Default bridge path validated with no-network tests.



- [x] OpenAI native text-family path validated.



- [x] Anthropic native text-family path validated.



- [x] OpenAI-compatible native text-family path validated.



- [x] Gemini native text-family path validated.

- [x] OpenAI registry handle now also locks provider-scoped build-override precedence on the real chat path.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("openai:gpt-4o")`, so mixed registries can route OpenAI-specific auth, base URLs, and transports all the way to the final `/responses` request boundary without leaking global overrides across providers.

- [x] Anthropic registry handle now also locks provider-scoped build-override precedence on the real chat path.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("anthropic:claude-3-5-sonnet-20241022")`, so mixed registries can route Anthropic-specific `x-api-key`, base URLs, and transports all the way to `/v1/messages` without leaking global overrides across providers.

- [x] Gemini registry handle now also locks provider-scoped build-override precedence on the real chat path.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("gemini:gemini-2.5-flash")`, so mixed registries can route Gemini-specific `x-goog-api-key`, base URLs, and transports to the final `/models/{model}:generateContent` request boundary without leaking global overrides.

- [x] Google Vertex registry handle now also locks provider-scoped build-override precedence on the real chat path.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("vertex:gemini-2.5-flash")`, so mixed registries can route Vertex-specific express `?key=...` auth, base URLs, and transports to the final `generateContent` request boundary without leaking global overrides.

- [x] Ollama registry handle now also locks provider-scoped build-override precedence on the real chat path.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("ollama:llama3.2")`, so mixed registries can route Ollama-specific base URLs and transports to `/api/chat` while keeping Ollama’s no-api-key contract intact.



- [x] Groq native text-family path validated.

- [x] Groq registry handle now also locks provider-scoped build-override precedence on the real chat path.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("groq:llama-3.1-70b-versatile")`, so mixed registries can route Groq-specific auth, base URLs, and transports to the final `/chat/completions` boundary while still preserving Groq's root-base `.../openai/v1` normalization.



- [x] xAI native text-family path validated.

- [x] xAI registry handle now also locks provider-scoped build-override precedence on the real chat path.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("xai:grok-beta")`, so mixed registries can route xAI-specific auth, base URLs, and transports to `/chat/completions` without leaking global overrides across providers.

- [x] MiniMaxi registry-native text-family path validated.



- [x] xAI registry-backed and provider extension surfaces now materialize the provider-owned `XaiClient` / `XaiConfig` pair instead of leaking the generic OpenAI-compatible client alias.



- [x] xAI and Groq now also expose provider-owned typed response metadata (`XaiMetadata` / `XaiChatResponseExt`, `GroqMetadata` / `GroqChatResponseExt`) so apps no longer need raw nested `provider_metadata` traversal.



- [x] xAI now also has no-network request-path coverage for typed `XaiOptions` normalization, and Stable `response_format` now wins over raw `providerOptions.xai.response_format` while xAI-specific post-merge stripping still removes unsupported `stop` / `stream_options` fields.



- [x] xAI now also locks Stable `tool_choice` precedence over raw `providerOptions.xai.tool_choice`, so runtime-provider option merging cannot silently break the Stable tool loop contract.
  - No-network public-path parity now also covers builder / provider / config-first / registry construction on the real provider-owned `XaiClient` story, instead of relying only on package-local smoke coverage.

- [x] xAI now also has a provider-owned typed source metadata escape hatch: `XaiSourceExt` / `XaiSourceMetadata` keep `xai` naming on the public surface while reusing the OpenAI-compatible source shape underneath, and the helper intentionally accepts both `provider_metadata["xai"]` and legacy compatible `provider_metadata["openai"]` envelopes.

- [x] Groq now also has a provider-owned typed source metadata escape hatch: `GroqSourceExt` / `GroqSourceMetadata` keep `groq` naming on the public surface while reusing the compatible source shape underneath, and the helper intentionally accepts both `provider_metadata["groq"]` and legacy compatible `provider_metadata["openai"]` envelopes.

- [x] Perplexity now also promotes hosted-search `citations` onto the typed metadata surface: `PerplexityMetadata.citations` carries returned citation URLs directly, so common search-answer consumers no longer need to read `meta.extra["citations"]` for the stable case.
- [x] Perplexity now also promotes `usage.reasoning_tokens` onto the typed metadata surface: `PerplexityUsage.reasoning_tokens` carries provider-side reasoning cost directly, so the usage escape hatch keeps only genuinely unknown vendor fields in `usage.extra`.
- [x] Perplexity hosted-search metadata has been re-checked against the current in-repo runtime fixtures and examples: no stronger stable response-side field is presently evidenced beyond `citations`, `images`, `usage.citation_tokens`, `usage.num_search_queries`, and `usage.reasoning_tokens`, so the typed surface intentionally stops there until new provider evidence appears.

- [x] DeepSeek now also has a provider-owned typed source metadata escape hatch: `DeepSeekSourceExt` / `DeepSeekSourceMetadata` keep `deepseek` naming on the public surface while reusing the compatible source shape underneath, and the helper intentionally accepts both `provider_metadata["deepseek"]` and legacy compatible `provider_metadata["openai"]` envelopes.

- [x] MiniMaxi now also has a provider-owned typed tool-call content-part escape hatch: `MinimaxiContentPartExt::minimaxi_tool_call_metadata()` reuses the Anthropic-compatible `caller` payload under `minimaxi` naming, accepts both `provider_metadata["minimaxi"]` and legacy `provider_metadata["anthropic"]`, and avoids inventing a source-level typed surface that Anthropic itself does not yet define.
- [x] MiniMaxi now also owns provider-named chat request helpers for the stable request-side capability story: `MinimaxiOptions` / `MinimaxiChatRequestExt` expose thinking-budget and structured-output configuration under `provider_options["minimaxi"]`, Stable `response_format` now maps onto MiniMaxi's native `output_format` path without leaking Anthropic's reserved `json` tool fallback, legacy `provider_options["anthropic"]` thinking payloads remain accepted for compatibility, and the new MiniMaxi chat-request fixtures plus public-surface guards lock that contract.



- [x] The generic `OpenAiCompatibleClient` path now also has no-network transport-boundary capture coverage for runtime xAI requests, locking final non-streaming and streaming request bodies after shared compat merging.



- [x] The shared OpenAI-compatible runtime now also normalizes `enableReasoning` / `reasoningBudget` for runtime DeepSeek requests, and the generic `OpenAiCompatibleClient` path has a no-network transport-boundary guard for that final request body.



- [x] OpenRouter now also has public alignment coverage plus a direct `OpenAiCompatibleClient` transport-boundary guard confirming Stable `response_format` / `tool_choice` precedence while vendor params like `transforms` still merge.

- [x] OpenRouter registry handle now also locks provider-scoped build-override precedence on the real chat and chat-stream paths.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("openrouter:openai/gpt-4o")` for both `chat_request` and `chat_stream_request`, and the public-path anchors now run through typed `OpenRouterOptions`, so mixed registries can route OpenRouter-specific auth, base URLs, and transports to `/chat/completions` while preserving vendor params such as `transforms` plus SSE request headers on the final request boundary.

- [x] OpenRouter registry embedding handle now also locks provider-scoped build-override precedence on the real embedding path.
  - `ProviderBuildOverrides` precedence is now covered on `embedding_model("openrouter:openai/text-embedding-3-small")`, so mixed registries can route OpenRouter-specific auth, base URLs, and transports to `/embeddings` while preserving request-level fields such as `dimensions`, `encoding_format`, and `user` on the final request body.

- [x] OpenRouter / Perplexity registry metadata extraction now also survives provider-scoped override routing.
  - OpenRouter now pins typed `OpenRouterMetadata` extraction on both the override-backed registry chat-response path and override-backed registry `StreamEnd` path, and Perplexity now does the same on both override-backed 200-response and `StreamEnd` paths, so mixed registries do not lose vendor metadata roots when auth/base-url/transport are routed through `ProviderBuildOverrides`.

- [x] xAI / Groq registry metadata extraction now also survives provider-scoped override routing.
  - Both provider-owned wrapper stories now pin override-backed registry chat-response plus override-backed registry `StreamEnd` metadata extraction on the real `language_model("{provider}:...")` path, so mixed registries do not lose `provider_metadata["xai"]` / `provider_metadata["groq"]` or break typed `XaiChatResponseExt` / `GroqChatResponseExt` accessors when auth/base-url/transport are routed through `ProviderBuildOverrides`.

- [x] DeepSeek registry metadata extraction now also survives provider-scoped override routing.
  - The provider-owned DeepSeek wrapper story now also pins override-backed registry chat-response plus override-backed registry `StreamEnd` metadata extraction on the real `language_model("deepseek:...")` path, so mixed registries do not lose `provider_metadata["deepseek"]` or break typed `DeepSeekChatResponseExt` access when auth/base-url/transport are routed through `ProviderBuildOverrides`.

- [x] xAI / Groq / DeepSeek provider-owned wrapper request override anchors now cover both request modes across public facade and lower contract.
  - Groq and xAI public-path tests now explicitly lock provider-scoped override routing on both `chat_request` and `chat_stream_request`, DeepSeek now adds the missing public stream anchor, and lower-contract tests now also pin the stream path for all three wrappers, so mixed registries preserve provider-specific auth/base-url/transport plus typed request fields instead of only covering the non-stream lane or contract-only construction.

- [x] Bedrock rerank mixed-registry override parity now also exists on the public facade.
  - Public-path no-network coverage now pins `registry.reranking_model("bedrock:...")` under `ProviderBuildOverrides`, matching the existing lower-contract anchor so rerank requests keep Bedrock-specific auth/base-url routing on the real public registry path instead of only being guarded below the facade.

- [x] Fireworks transcription mixed-registry override parity now also covers the dedicated audio host path.
  - Public-path and lower-contract no-network coverage now both pin `registry.transcription_model("fireworks:whisper-v3")` under `ProviderBuildOverrides`, so the compat audio story preserves provider-scoped auth/base-url/transport on the real multipart `/audio/transcriptions` path instead of drifting back to the shared inference host.

- [x] SiliconFlow speech/transcription mixed-registry override parity now also covers the compat audio routes end to end.
  - Public-path and lower-contract no-network coverage now both pin `registry.speech_model("siliconflow:...")` and `registry.transcription_model("siliconflow:...")` under `ProviderBuildOverrides`, so compat audio routing preserves provider-scoped auth/base-url/transport on the real `/audio/speech` plus multipart `/audio/transcriptions` paths instead of silently falling back to the shared global transport lane.

- [x] Together speech/transcription mixed-registry override parity now also covers the compat audio routes end to end.
  - Public-path and lower-contract no-network coverage now both pin `registry.speech_model("together:...")` and `registry.transcription_model("together:...")` under `ProviderBuildOverrides`, so compat audio routing preserves provider-scoped auth/base-url/transport on the real `/audio/speech` plus multipart `/audio/transcriptions` paths instead of silently falling back to the shared global transport lane.

- [x] Together/SiliconFlow compat image mixed-registry override parity now also covers the shared image route end to end.
  - Public-path and lower-contract no-network coverage now both pin `registry.image_model("together:...")` and `registry.image_model("siliconflow:...")` under `ProviderBuildOverrides`, so compat image routing preserves provider-scoped auth/base-url/transport on the real `/images/generations` path instead of silently falling back to registry-global defaults.

- [x] SiliconFlow/Jina/VoyageAI compat rerank mixed-registry override parity now also covers the shared rerank route end to end.
  - Public-path and lower-contract no-network coverage now both pin `registry.reranking_model("siliconflow:...")`, `registry.reranking_model("jina:...")`, and `registry.reranking_model("voyageai:...")` under `ProviderBuildOverrides`, so compat rerank routing preserves provider-scoped auth/base-url/transport on the real `/rerank` path instead of silently falling back to registry-global defaults.

- [x] Compat embedding mixed-registry override parity now also covers the shared embedding route end to end for the current preset set.
  - Public-path and lower-contract no-network coverage now pin `registry.embedding_model("mistral:...")`, `registry.embedding_model("fireworks:...")`, `registry.embedding_model("siliconflow:...")`, `registry.embedding_model("together:...")`, `registry.embedding_model("openrouter:...")`, `registry.embedding_model("jina:...")`, `registry.embedding_model("voyageai:...")`, and `registry.embedding_model("infini:...")` under `ProviderBuildOverrides`, so compat embedding routing preserves provider-scoped auth/base-url/transport on the real `/embeddings` path instead of silently falling back to registry-global defaults.

- [x] Anthropic mixed-registry override parity now also reaches the public registry facade on both request modes.
  - Public-path and lower-contract no-network coverage now both pin `registry.language_model("anthropic:...")` under `ProviderBuildOverrides` for both `chat_request` and `chat_stream_request`, so provider-scoped `x-api-key` / base-url / transport routing stays explicit on the real `/messages` path instead of only being guarded below the facade or on the non-stream lane.

- [x] OpenAI / Gemini / Vertex / Ollama mixed-registry override parity now also reaches the public registry facade on their real native request lanes.
  - Public-path no-network coverage now pins `registry.language_model("openai:...")`, `registry.language_model("gemini:...")`, `registry.language_model("vertex:...")`, and `registry.language_model("ollama:...")` under `ProviderBuildOverrides`, matching the existing lower-contract anchors so each native provider keeps its own auth/base-url/transport routing on the real `responses` / `generateContent` / `api/chat` path instead of relying on contract-only coverage.

- [x] Vertex / Ollama mixed-registry override parity now also pins the native stream lanes end to end.
  - Public-path and lower-contract no-network coverage now both pin `registry.language_model("vertex:...").chat_stream_request(...)` and `registry.language_model("ollama:...").chat_stream_request(...)` under `ProviderBuildOverrides`, so provider-scoped auth/base-url/transport routing stays explicit on the real `streamGenerateContent` and `api/chat` stream boundaries instead of only being covered on the non-stream lane.

- [x] OpenAI / Gemini native stream request parity and mixed-registry override coverage now also close the remaining stream-lane gaps.
  - OpenAI now pins `registry.language_model("openai:...").chat_stream_request(...)` under `ProviderBuildOverrides` on the real `/responses` SSE lane, and Gemini now adds both direct stream request-shape parity plus public/lower-contract `ProviderBuildOverrides` anchors on `:streamGenerateContent?alt=sse`, so those native stream paths no longer depend only on response-metadata tests or non-stream-only override coverage.

- [x] Gemini / Vertex lower-contract request-option coverage now also closes the remaining native `generateContent` package-alignment gaps.
  - `siumai-provider-gemini` builder/config/registry construction now converges on Stable `tool_choice`, `response_format`, `thinkingConfig`, and `structuredOutputs` on the final `:generateContent` request body, while `siumai-provider-google-vertex` now has matching lower-contract anchors for Stable `response_format`, provider-owned `thinkingConfig` / `structuredOutputs`, and `tool_choice -> toolConfig.functionCallingConfig` mapping, and its provider-owned client path now also locks the corresponding response-side structured-output split (`StreamEnd` metadata vs interrupted incomplete JSON) instead of relying only on public-path parity.

- [x] Gemini / Vertex public registry handles now also lock Stable request-option convergence directly on the final `generateContent` body.
  - Public no-network coverage now pins `registry.language_model("gemini:...")` against config-first `GeminiClient` for Stable `tool_choice`, `response_format`, `thinkingConfig`, and `structuredOutputs`, and does the same for `registry.language_model("vertex:...")` on both Stable structured-output shaping and `tool_choice -> toolConfig.functionCallingConfig`, so registry text handles no longer rely only on builder/provider/config parity or lower-contract coverage.

- [x] Vertex structured-output public-path parity now also closes the remaining response-side stream gap.
  - Public no-network coverage now pins `Siumai::builder().vertex()`, `Provider::vertex()`, config-first `GoogleVertexClient`, and `registry.language_model("vertex:...")` on the same native `streamGenerateContent` structured-output story, locking complete accumulated JSON plus final raw `provider_metadata["vertex"]` `StreamEnd` metadata and the matching interrupted incomplete-stream `ParseError` contract instead of leaving that response-side split only to provider-local regressions.

- [x] Azure raw metadata boundary now also reaches the registry response and stream lanes end to end.
  - Public-path and lower-contract no-network coverage now pin both `registry.language_model("azure:...").chat_request(...)` and `registry.language_model("azure:...").chat_stream_request(...)` against config-first construction on the real `/v1/responses?api-version=v1` path, so 200-response roots plus `text-start` / `finish` custom events and final `StreamEnd` metadata all stay explicitly namespace-scoped `provider_metadata["azure"]` and never widen into `provider_metadata["openai"]`.

- [x] OpenAI typed metadata parity now also reaches the real registry facade on both native text lanes.
  - Public-path no-network coverage now pins `registry.language_model("openai-chat:...")` against config-first Chat Completions for typed `OpenAiChatResponseExt.logprobs` on both 200-response and `StreamEnd`, and separately pins `registry.language_model("openai:...")` against config-first Responses for typed `OpenAiSourceExt` extraction on both 200-response and `StreamEnd`, so the remaining OpenAI builder/config/registry metadata gap is closed on the actual registry path rather than only on builder/provider/config-first construction.

- [x] Perplexity registry handle now also locks provider-scoped build-override precedence on the real chat and chat-stream paths.
  - `ProviderBuildOverrides` precedence is now covered on `language_model("perplexity:sonar")` for both `chat_request` and `chat_stream_request`, and the public-path anchors now run through typed `PerplexityOptions`, so mixed registries can route Perplexity-specific auth, base URLs, and transports to `/chat/completions` while preserving hosted-search vendor params such as `search_mode` plus SSE request headers on the final request boundary.

- [x] OpenAI-only public-path compat tests no longer drag MiniMaxi-only registry helpers into scope.
  - The stray MiniMaxi helper in the `openai` public-path module is gone, so `cargo nextest ... --features "openai"` can compile the OpenRouter / Perplexity public override anchors without depending on unrelated provider factories.



- [x] Perplexity now also has public alignment coverage plus a direct `OpenAiCompatibleClient` transport-boundary guard confirming Stable `response_format` / `tool_choice` precedence while generic vendor params still merge.



- [x] OpenRouter now also exposes provider-owned typed request helpers through `OpenRouterOptions` / `OpenRouterChatRequestExt`, and `siumai::provider_ext::openrouter` is now the stable typed escape hatch for common vendor params such as `transforms`.



- [x] Perplexity now also exposes provider-owned typed request helpers through `PerplexityOptions` / `PerplexityChatRequestExt`, covering common hosted-search knobs such as `search_mode`, `search_recency_filter`, `return_images`, and `web_search_options`.



- [x] Perplexity now also exposes provider-owned typed response metadata (`PerplexityMetadata` / `PerplexityChatResponseExt`) for hosted-search usage/images, so apps no longer need raw `provider_metadata["perplexity"]` traversal for common cases.



- [x] Perplexity streaming now also preserves the same hosted-search metadata on `StreamEnd`, and the shared OpenAI-compatible SSE path has no-network coverage for that typed extraction.



- [x] Perplexity hosted-search metadata extraction is now centralized in the shared OpenAI-compatible compat helper, so non-streaming response shaping and streaming `StreamEnd` fallback no longer maintain separate extraction rules.



- [x] Decide hosted-search surface scope for V4: OpenRouter / Perplexity remain provider-owned typed extensions for this cycle, and a new Stable unified hosted-search request surface is explicitly deferred; see `hosted-search-surface.md`.



- [ ] Re-evaluate a Stable hosted-search surface only after at least three providers converge on both request and response semantics.



- [x] OpenRouter now also has explicit config-first and builder-to-final-request coverage for shared compat reasoning helpers (`with_reasoning` / `with_reasoning_budget`), so unified reasoning ergonomics are no longer guarded only through DeepSeek paths.



- [x] Groq now also exposes typed request-option builders for logprobs, service tier, and reasoning hints, and `groq-logprobs` demonstrates the request+metadata escape hatch on the config-first wrapper path.



- [x] Groq now also locks Stable `response_format` precedence over raw `providerOptions.groq.response_format`, while still merging typed logprobs/service-tier/reasoning knobs through the provider-owned wrapper path.



- [x] Groq now also locks Stable `tool_choice` precedence over raw `providerOptions.groq.tool_choice` on the provider-owned before-send path.
  - No-network public-path parity now also covers builder / provider / config-first / registry construction on the real provider-owned `GroqClient` story, instead of relying only on provider-spec smoke coverage.



- [x] Groq now also aligns its provider-owned wrapper with family-model metadata semantics (`ModelMetadata`) and registry contract coverage confirms both xAI and Groq materialize provider-owned wrapper clients on the public registry path.



- [x] Groq now also normalizes trailing slashes for custom path-style `base_url` inputs across builder/config/registry entry points, avoiding wrapper-observation drift and request-path double-slash risks.



- [x] Groq config-first wrapper parity is now closer to xAI/DeepSeek through `with_http_client(...)` plus public provider-context / retry / transport helper accessors on `GroqClient`.



- [x] DeepSeek registry-native text-family path validated.



- [x] DeepSeek deepseek-only registry and builder wiring validated in `--no-default-features --features deepseek` builds.



- [x] DeepSeek is now exposed as a first-class `ProviderType` across spec/core/registry/unified metadata instead of `Custom("deepseek")`.



- [x] OpenAI-compatible full-request chat paths now preserve request-level provider options while merging client defaults instead of falling back to message/tool-only trait defaults.



- [x] DeepSeek now owns provider-specific chat request option normalization (`enableReasoning` / `reasoningBudget` -> snake_case) with no-network request-capture coverage.



- [x] DeepSeek provider-owned typed response metadata now also has no-network top-level plus registry `language_model("deepseek:...")` 200-response guards across `Siumai::builder()`, `Provider::deepseek()`, config-first `DeepSeekClient`, and registry-native construction, so `DeepSeekMetadata` logprobs no longer rely on metadata-only unit fixtures or provider-local wrapper tests.



- [x] xAI, Groq, and DeepSeek typed response metadata now converge on the shared OpenAI-compatible compat helper for `sources` / `logprobs`, allowing DeepSeek to drop its wrapper-specific response transformer.



- [x] xAI and Groq now also have no-network top-level plus registry `language_model("{provider}:...")` 200-response guards across `Siumai::builder()`, `Provider::xai()` / `Provider::groq()`, config-first clients, and registry-native construction, so shared compat typed metadata is exercised on the real public construction surfaces instead of only wrapper-local tests.



- [x] xAI, Groq, and DeepSeek provider-owned wrapper clients now also have no-network `chat_stream_request` StreamEnd guards, and all three now additionally lock that metadata on both the top-level builder/provider/config-first public path and registry `language_model("{provider}:...")` construction where available, so typed metadata parity is exercised across wrapper-local, public, and registry-native surfaces.



- [x] DeepSeek now also locks Stable `response_format` precedence over raw `providerOptions.deepseek.response_format` on the provider-owned spec path, while preserving `reasoningBudget` camelCase normalization and the final snake_case wire shape.



- [x] DeepSeek now also locks Stable `tool_choice` precedence over raw `providerOptions.deepseek.tool_choice` on the provider-owned spec path.



- [x] Runnable provider-specific examples now exist for xAI web search, Groq structured output, DeepSeek reasoning, OpenRouter typed transforms, and Perplexity typed search + metadata on preferred config-first, registry-first, or built-in generic-client construction paths.



- [x] Real integration-test sources used by feature validation now remain valid UTF-8, so `cargo fmt --all` and feature-gated builds no longer fail while parsing DeepSeek/OpenAI test targets.



- [x] Minimal `EmbeddingModel` trait added as a family-model bridge.



- [x] `EmbeddingModelHandle` now satisfies model metadata semantics.



- [x] Default embedding-family bridge path validated with no-network tests.



- [x] OpenAI-compatible native embedding-family path validated.



- [x] Gemini native embedding-family path validated.

  - Gemini now also has a provider-owned `embed_batch(...)` path for the public batch helper surface: homogeneous `BatchEmbeddingRequest` values are coalesced into one native `:batchEmbedContents` request when that mapping is lossless, while mixed-shape batches still fall back to per-request execution instead of forcing an unsafe merge.

  - Top-level no-network parity now also locks `siumai::embedding::embed_many(...)` across `Siumai::builder().gemini()`, `Provider::gemini()`, config-first `GeminiClient`, and `registry.embedding_model("gemini:...")`, so the public batch helper no longer trails the provider-owned Gemini embedding route.



- [x] Minimal `ImageModel` trait added as a family-model bridge.



- [x] `ImageModelHandle` now satisfies model metadata semantics.



- [x] Default image-family bridge path validated with no-network tests.



- [x] OpenAI native image-family path validated.



- [x] OpenAI native speech-family path validated.



- [x] OpenAI native transcription-family path validated.



- [x] Gemini native image-family path validated.

- [x] MiniMaxi registry-native image-family path validated.

- [x] MiniMaxi registry-native speech-family path validated.

- [x] MiniMaxi registry `language_model` handle now delegates provider-specific video/music capability paths.

- [x] Azure native text-family path validated.



- [x] Azure native embedding-family path validated.



- [x] Azure native image-family path validated.



- [x] Azure native speech-family path validated.



- [x] Azure native transcription-family path validated.



- [x] xAI registry-native speech-family path validated through the provider-owned `/v1/tts` execution path and contract tests.



- [x] Fireworks shared OpenAI-compatible transcription-family path validated against its dedicated audio base and registry contract coverage.



- [x] Fireworks speech-family path remains intentionally unsupported because the current public docs only expose transcription on the dedicated audio host.
  - Top-level no-network parity now also locks that boundary across `Siumai::builder().openai().fireworks()`, `Provider::openai().fireworks()`, and config-first `OpenAiCompatibleClient`: `text_to_speech` returns `UnsupportedOperation`, `as_speech_capability()` stays `None`, and no request reaches the transport layer.



- [x] xAI native transcription-family remains intentionally unsupported until a standalone STT contract is validated.
  - Top-level no-network parity now also locks that speech-only boundary across `Siumai::builder().xai()`, `Provider::xai()`, and config-first `XaiClient`: `speech_to_text` returns `UnsupportedOperation`, `as_transcription_capability()` stays `None` while speech/audio handles remain available, and no request reaches the transport layer.



- [x] `LanguageModelHandle` execution now routes through the language-family path.



- [x] `EmbeddingModelHandle` execution now routes through the embedding-family path.

  - `EmbeddingModelV3` now prefers request-aware `EmbeddingExtensions` when the underlying capability exposes them, so family helpers and registry/bridge paths no longer downgrade `EmbeddingRequest` into a bare `Vec<String>` and silently drop `model`, `dimensions`, `user`, `provider_options_map`, or `http_config`.

  - Focused no-network coverage now also locks the public helper lane: `siumai::embedding::embed(&registry.embedding_model("openai:..."), ...)` preserves request-level embedding fields plus helper-injected headers on the final OpenAI `/embeddings` request.

  - The public `Siumai` wrapper no longer hardcodes a provider-specific downcast ladder for request-aware embeddings; it now delegates through `LlmClient::as_embedding_extensions()`, so Azure and future embedding providers inherit request-aware `embed_with_config(...)` behavior without wrapper-local branching.

  - `EmbeddingModelHandle::embed_many(...)` now also routes through the family-model batch path instead of defaulting to repeated single-request execution, so native batch embedding models keep their provider-owned batch semantics on the public registry surface.



- [x] `ImageModelHandle` execution now routes through the image-family path.



- [x] `ImageExtras` remains on the generic client path for V4 because image extras are still extension-only.



- [x] Handle-level no-network tests validate native family execution for text, embedding, and image.



- [x] OpenAI, Anthropic, and Gemini builders now emit canonical provider configs via `into_config()` before client construction.



- [x] OpenAI-compatible, xAI, Groq, and DeepSeek builders now also expose config-first convergence via `into_config()`, and xAI / DeepSeek now wrap that path with provider-owned config/client entry types.



- [x] Google Vertex builder now also emits canonical config via `into_config()`, and the provider-owned config surface now has explicit `new` / `express` / `enterprise` constructor paths instead of relying on struct literals for config-first setup.
- [x] Google Vertex registry-native text-family path validated.
- [x] Google Vertex registry-native image-family path validated.
- [x] Google Vertex registry-native embedding-family path validated.
- [x] Google Vertex top-level public-path parity now covers chat / chat-stream / embedding / image generation / image editing across `Siumai::builder().vertex()` / `Provider::vertex()` / config-first clients.



- [x] Ollama builder now also emits canonical config via `into_config()`, and registry factory construction is aligned with the expanded config shape.



- [x] Ollama now also exposes provider-owned typed response metadata (`OllamaMetadata` / `OllamaChatResponseExt`), a runnable metadata example, and no-network builder/config/registry plus top-level public-path parity coverage for full `chat_request` / `chat_stream_request` semantics.
- [x] Ollama top-level public-path parity now also locks `ChatStreamEvent::StreamEnd` timing metadata through `Siumai::builder()`, `Provider::ollama()`, and config-first `OllamaClient`, so `OllamaChatResponseExt` is guarded on both 200-response and JSON-stream end paths.
- [x] Amazon Bedrock now also exposes a provider-owned config/client/builder surface, typed request helpers for chat/rerank, registry-native typed client construction, and no-network public-path parity across `chat_request`, `chat_stream_request`, and rerank with split runtime vs agent-runtime endpoint ownership locked in config-first form.
- [x] Amazon Bedrock now also has a runnable config-first example plus provider-local auth guidance, so the split-endpoint design is documented for both SigV4-header injection and Bearer/proxy compatibility flows.



- [x] JSON streaming execution now also honors injected `HttpTransport` on stream paths, so Ollama no-network parity covers both request shaping and streaming transport execution without accidental real HTTP fallback.



- [x] JSON bytes execution now also honors injected `HttpTransport` on non-streaming byte-response paths, closing the hidden parity gap that previously caused provider-owned JSON TTS tests such as xAI to fall back to live HTTP.



- [x] `OllamaClient` now also exposes provider-context / retry / transport wrapper accessors plus `ModelMetadata`, bringing its config-first wrapper semantics closer to xAI / Groq / DeepSeek and reducing special-case handling on public construction paths.



- [x] JSON streaming now also mirrors SSE end-of-stream semantics through `handle_stream_end_events()`, with coverage for reqwest JSON streams, custom-transport JSON streams, and middleware-expanded end events.



- [x] Gemini typed request options now have no-network contract coverage across provider-owned serialization and final request shaping for `responseLogprobs` / `logprobs`, `responseJsonSchema`, `retrievalConfig`, `mediaResolution`, and `imageConfig`, and Gemini structured-output tests now also lock `responseFormat` vs `structuredOutputs` vs legacy `responseJsonSchema` precedence, so common Google-specific knobs are backed by executable mapping guards instead of example-only behavior.







## Exit checklist







- [x] Family model traits are the default internal and public execution contracts.

  - Public compile guards now cover the full stable family surface instead of only text/embedding: all six registry handles (`LanguageModelHandle`, `EmbeddingModelHandle`, `ImageModelHandle`, `RerankingModelHandle`, `SpeechModelHandle`, `TranscriptionModelHandle`) compile as their final metadata-bearing family model traits, and the top-level helper modules (`text`, `embedding`, `image`, `rerank`, `speech`, `transcription`) are pinned as callable against those stable family traits rather than requiring users to reason about `LlmClient`.



- [x] Registry handles are family model objects.
  - `LanguageModelHandle` / `EmbeddingModelHandle` now also have explicit public compile guards proving they satisfy the final family-model trait bounds plus `ModelMetadata`, while the earlier execution-path work already locked that real calls route through the family APIs instead of a legacy generic client path.



- [x] Builders are thin wrappers over config-first construction.
  - Cross-provider public-surface consistency tests now lock that major provider builders preserve common inherited HTTP settings on `into_config()` instead of re-deriving them later, while provider-local builder tests for focused/native packages already guard `build()` ↔ `into_config()` ↔ `from_config()` convergence.



- [x] Major providers are migrated.

  - The major provider-owned package story is now explicit and guarded instead of implied: OpenAI, Anthropic, Gemini, Vertex, and Anthropic-on-Vertex all have public compile/parity coverage for their top-level `Provider::*` entry points plus provider-owned config/client surfaces, while the package-alignment note already records that remaining work on these providers is depth completion rather than another surface redesign.



- [x] Public docs tell a single coherent story.

  - README, example navigation, and workstream docs now share the same surface ladder and public-surface map.
  - Release-readiness wording is now aligned too: README explicitly frames OpenAI-compatible vendor views as typed layers over the shared compat runtime and keeps builder-based compat entry points labeled as non-default migration/convenience paths outside the main guides as well.







- [~] Secondary provider cleanup notes:



  - Groq config-first construction now also avoids direct field mutation for HTTP config in parity tests.

  - OpenAI public-path parity now also covers typed reasoning options on both Responses and Chat Completions routes, so Stable `response_format` / `tool_choice` plus provider-owned `OpenAiOptions` reasoning settings are guarded across siumai/provider/config-first construction.

  - OpenAI native Responses reasoning public depth is now stronger too: the top-level no-network public-path parity suite now locks non-stream `ChatResponse::reasoning()` plus typed reasoning-part metadata (`providerMetadata.openai.{itemId, reasoningEncryptedContent}` on `ContentPart::Reasoning`) across builder/provider/config-first/registry, and the streaming lane now pins `ThinkingDelta` accumulation plus the same `StreamEnd` reasoning metadata across those four construction routes while keeping request-side `reasoning.{effort,summary}` aligned on the final `/responses` payload.

  - That OpenAI Responses stream pass also closed a protocol-level gap: inbound `response.reasoning_summary_text.delta` chunks now emit stable `ThinkingDelta` alongside the existing OpenAI-specific reasoning stream-part events, so top-level public streaming paths no longer rely on provider-specific custom events to surface reasoning text.

  - Gemini reasoning public depth is now stronger too: the top-level no-network public-path parity suite now locks non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` accumulation across builder/provider/config-first/registry on the provider-owned Gemini path, while the parallel Google Vertex stream lane now also pins the same `ThinkingDelta` accumulation across builder/provider/config-first/registry without collapsing the raw `providerMetadata.vertex.thoughtSignature` event namespace into a fake typed surface.

  - Anthropic reasoning public depth is now stronger too: the top-level no-network public-path parity suite now locks non-stream `ChatResponse::reasoning()` plus typed `providerMetadata.anthropic.{thinking_signature, redacted_thinking_data}` across builder/provider/config-first/registry on the real provider-owned path, and the streaming lane now pins `ThinkingDelta` accumulation plus the same end-of-stream metadata across those four construction routes.

  - That Anthropic stream pass also exposed a real implementation gap rather than just a missing test: `AnthropicEventConverter` had been emitting `StreamEnd` with an empty content shell, so the protocol layer now backfills accumulated text/reasoning content into `StreamEnd` when normal text/thinking SSE blocks were seen, bringing the direct provider stream path in line with the Stable extraction contract instead of relying on an external stream processor to reconstruct content.



  - Provider-specific example coverage now exists for xAI web search plus provider-owned xAI TTS, Groq structured output, DeepSeek reasoning, OpenRouter typed transforms, Perplexity typed search + metadata, Ollama metadata, and compat SiliconFlow image generation; the examples indexes now also point readers to those files in the same provider-owned config-first -> compat vendor-view -> builder-demo order, and the top-level compat preset guard now locks both the built-client audio split for Together / SiliconFlow / Fireworks and the built-client image split for SiliconFlow / Together / Fireworks, so the remaining secondary-provider cleanup is mostly feature-depth work rather than navigation or basic capability labeling.

  - Observability example placement is now also explicit instead of implied: `custom-middleware` and `basic-telemetry` are documented as registry-first application examples, `http-interceptor` is documented as a config-first provider-client example, and `middleware_builder.rs` is labeled as a middleware-composition utility rather than a default `Siumai` construction path.

  - xAI structured output depth is now stronger too: the provider-owned wrapper path now has a runnable config-first `xai-structured-output` example, and the top-level public-path parity suite now locks Stable `response_format` plus xAI-specific request normalization (`reasoningEffort` passthrough, unsupported `stop` stripping, Stable-over-raw `response_format` precedence) across builder/provider/config-first construction.

  - Ollama structured output is now also easier to discover from the real public story: the provider-owned local-runtime path now has a runnable config-first `ollama-structured-output` example layered on top of the already-existing no-network parity coverage for schema-shaped `format` request mapping.

  - The canonical advanced structured-output example is now back on the actual Stable story too: `03-advanced-features/provider-params/structured-output.rs` now demonstrates `ChatRequest::with_response_format(...)` plus `extract_json_from_response::<T>(...)` directly instead of routing readers through OpenAI-specific request options.
  - Public `siumai::structured_output` facade coverage is now a bit tighter as well: the top-level test file now explicitly locks both the dedicated `content filtering/refusal` parse error and the best-effort repair path when JSON arrives on an explicit `StreamEnd` response, so those stable semantics are no longer guarded only inside `siumai-core`.
  - xAI reasoning public depth is now also less implicit: the provider-owned wrapper path now has a runnable config-first `xai-reasoning` example, and the top-level public-path parity suite now locks non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` accumulation across builder/provider/config-first/registry in addition to the existing `enable_reasoning` / `reasoning_budget` defaults and request-level `reasoningEffort` normalization coverage.
  - Compat embedding public depth is now a bit stronger too: `openrouter-embedding` gives the shared OpenAI-compatible runtime a runnable vendor-view example for Stable `embedding::embed` plus request-level options (`dimensions` / `encoding_format` / `user`) on the real OpenRouter public story instead of relying only on no-network parity coverage.
  - The `provider_ext` typed-surface audit is tighter again: `azure` and plain `google_vertex` have both moved out of the remaining-gap bucket. The Vertex public facade now also exposes a provider-owned `metadata` module (`VertexMetadata`, `VertexChatResponseExt`, `VertexContentPartExt`, plus typed alias re-exports such as `VertexSource` / `VertexGroundingMetadata` / `VertexUrlContextMetadata` / `VertexUsageMetadata` / `VertexSafetyRating`), bound only to the `vertex` namespace instead of falling back to `google`; focused no-network tests now lock response-level `usageMetadata` / `safetyRatings`, `groundingMetadata` / `urlContextMetadata`, normalized `sources`, and part-level `thoughtSignature` extraction across the public builder/provider/config-first/registry story, while stream custom events intentionally stay raw because they validate namespace-scoped payloads rather than a stable response-side contract. Lower-priority/likely-boundary cases remain `cohere` and `togetherai`, which still look more like explicit rerank-only metadata boundaries than urgent typed-surface omissions.























