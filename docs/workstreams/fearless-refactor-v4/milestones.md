# Fearless Refactor V4 - Milestones







Last updated: 2026-03-19







This workstream defines the next architectural refactor after the current V3 line.



Its focus is not package splitting. Its focus is finishing the model-family-centered design.







## V4-M0 - Design locked







Acceptance criteria:







- Builder retention policy is documented.



- Family-model-first trait policy is documented.



- `LlmClient` demotion policy is documented.



- Public naming strategy is explicitly Rust-first.







Status: done







## V4-M1 - Final family traits exist







Acceptance criteria:







- Shared model metadata trait exists.



- Final family traits exist for language, embedding, image, reranking, speech, and transcription.



- Adapter shims from legacy capability traits exist.
- Speech and transcription now also expose metadata-bearing family traits over the legacy audio capability adapters.



- No-network unit tests cover the adapters.
- Speech/transcription adapter coverage now also includes metadata assertions in `siumai-core` plus registry bridge/native-family-path tests.
- No-network executor coverage now also includes injected-transport multipart JSON/bytes execution and single-401 retry parity, so OpenAI-style upload flows are protected without real HTTP.







Notes:







- `ModelMetadata` and `ModelSpecVersion` spike landed.



- Minimal `LanguageModel` trait spike landed.

- Minimal `RerankingModel` trait spike landed, including metadata conformance and adapter coverage.



- Minimal `EmbeddingModel` trait spike landed.



- Minimal `ImageModel` trait spike landed.
- The public `speech::synthesize(...)` / `transcription::transcribe(...)` helpers now also bind to the final metadata-bearing family traits instead of the legacy `*ModelV3` compatibility bounds.



- OpenAI, Anthropic, OpenAI-compatible, Gemini, Groq, xAI, MiniMaxi, and Google Vertex now validate the initial text-family path.



- OpenAI-compatible, Gemini, and Google Vertex now validate the initial embedding-family path.



- OpenAI, Gemini, MiniMaxi, and Google Vertex now validate the initial image-family path.



- Handle-level no-network tests now validate native family execution for text, embedding, and image.







Status: in progress







## V4-M2 - Registry returns family model objects







Acceptance criteria:







- `ProviderFactory` returns family model objects rather than generic client objects.



- `LanguageModelHandle` implements the final language model trait directly.



- `EmbeddingModelHandle` implements the final embedding model trait directly.



- `ImageModelHandle` implements the final image model trait directly.



- `SpeechModelHandle` implements the final speech model trait directly.
- Speech-family requests now inherit the configured family model id on registry and built-in provider-client paths when `TtsRequest.model` is omitted, so native family models no longer need per-request duplication just to reach the same routing target.
- No-network public-path parity now locks that convergence on native OpenAI, Groq, and MiniMaxi speech-family routes across builder/provider/config/registry entry points.
- OpenAI provider-specific speech SSE now also has no-network public-path parity across builder/provider/config and registry-factory construction, locking `stream_format: "sse"` plus missing-model backfill on the final `/audio/speech` body.
- OpenAI generic `SpeechExtras` streaming delegation now lands on that same provider-owned SSE path, so both wrapper clients and registry speech handles expose a working stream extra instead of a false-positive capability surface.
- Groq now also stops leaking generic `SpeechExtras` on its provider-owned public surface until a real provider-owned stream/voice-extras implementation exists, so wrapper/config clients only advertise the non-stream speech family they actually implement.
- xAI and MiniMaxi now also stop leaking generic `SpeechExtras` on their provider-owned public surface until a real provider-owned speech-extras implementation exists, so wrapper/config clients keep exposing the working non-stream speech family without advertising a dead-end extra handle.



- `TranscriptionModelHandle` implements the final transcription model trait directly.
- Speech/transcription handles now carry provider/model metadata and prefer family-factory paths over legacy generic-client downcasts.
- The same convergence rule now also covers built-in transcription-family request preparation: missing `SttRequest.model` / translation models inherit the configured family model id before execution on registry and provider-client paths.
- No-network public-path parity now also locks that convergence on native OpenAI and Groq transcription-family routes, plus OpenAI translation extras.
- OpenAI provider-specific transcription SSE now also has no-network public-path parity across builder/provider/config and registry-factory construction, locking forced multipart `stream=true` plus missing-model backfill on the final `/audio/transcriptions` body.
- Core raw-response streaming helpers now also honor injected custom transports on both JSON and multipart paths, which closes the last transport-level blocker for provider-owned SSE parity tests.
- OpenAI generic `TranscriptionExtras` streaming delegation now also resolves through that provider-owned SSE path, so `as_transcription_extras()` and registry transcription handles stop leaking a non-functional stream-extra surface.
- Groq now also stops leaking generic `TranscriptionExtras` on its provider-owned public surface, and the registry transcription handle now fails fast for stream/translation extras without issuing any HTTP request.



- `RerankingModelHandle` implements the final reranking model trait directly.



- Cache, TTL, and middleware invariants are preserved, and no-network handle tests now explicitly lock middleware-overridden `provider_id` / `model_id` cache-key reuse plus TTL expiration on the public registry path.
- DeepSeek’s current native boundary is now explicit instead of accidental: registry/catalog capability metadata stays text-family-led, and no-network builder/provider/config tests lock `embedding`, `rerank`, and `image_generation` as pre-HTTP `UnsupportedOperation` paths until the provider contract is intentionally widened.
- Cohere and TogetherAI now also have explicit focused-provider boundaries: registry/catalog tests lock both packages to rerank-only capability metadata, and no-network public-path tests confirm chat requests fail before transport instead of silently implying a wider text-model surface.
- The same focused-provider boundary is now pinned on the registry factory’s generic `language_model_with_ctx` entry as well: Cohere and TogetherAI now reject that unsupported text-family entry explicitly, while the unified `Siumai::builder()` path still materializes the correct default rerank family by selecting from declared capabilities instead of depending on a fake text fallback.
- OpenAI-compatible focused non-chat presets now enforce the same boundary at registry construction time too: Jina and VoyageAI no longer expose `registry.language_model("provider:model")` as a valid public text/chat entry, and public-path tests lock that rejection before any provider client or transport is touched.
- OpenAI-compatible mixed/split presets now also lock the positive public registry paths they are supposed to keep: Infini registry text handles match config-first `/chat/completions` request shaping for both chat and chat streaming, and Fireworks registry transcription handles match config-first multipart `/audio/transcriptions` routing while registry speech/TTS remains intentionally unsupported.
- Bedrock’s native surface boundary is now explicit at the same level: registry/catalog tests lock it to chat/streaming/tools/rerank, and no-network public-path tests confirm `embedding` plus `image_generation` fail before transport rather than implying unsupported extra families are available.







Notes:







- A parallel text-family-returning `ProviderFactory` interface now exists.



- `LanguageModelHandle` already carries family-model metadata.



- `EmbeddingModelHandle` now also carries family-model metadata.

- The embedding-family bridge is now request-aware end to end: when a provider/client exposes `EmbeddingExtensions`, family helpers and registry handles preserve the full `EmbeddingRequest` instead of collapsing to raw inputs, so request-scoped `model`, `dimensions`, `user`, provider options, and HTTP overrides survive both native and client-bridge paths.

- That request-aware closure now also reaches the top-level wrapper and batch handle lanes: `Siumai` delegates embedding extensions generically through `as_embedding_extensions()` instead of a provider-specific downcast ladder, and `EmbeddingModelHandle::embed_many(...)` now resolves the family model once and forwards to `embed_many(...)` directly instead of silently flattening native batch execution into repeated single-call fallbacks.

- `RerankingModelHandle` now also carries family-model metadata and prefers the parallel reranking-family factory path when available.



- `ImageModelHandle` now also carries family-model metadata.



- Language, embedding, and image handle execution now routes through family-model-centered delegation.



- `ImageExtras` intentionally still uses the generic client path because image editing and variation remain extension-only for V4.



- Remaining handle cleanup still applies to other families and broader cache/middleware invariants.







Status: in progress







## V4-M3 - Builder and config paths converge







Acceptance criteria:







- Provider config structs are canonical.



- Builders compile down to the same config-first constructor path.



- There are no builder-only features.



- Parity tests verify builder/config equivalence on major providers.







Notes:







- OpenAI, Anthropic, Gemini, OpenAI-compatible, xAI, Groq, DeepSeek, and Ollama builders now emit canonical config structs before client construction or expose `into_config()` for config-first flows.



- Builder/config parity tests now cover OpenAI, Anthropic, Gemini, OpenAI-compatible, Groq, xAI, DeepSeek, Ollama, Google Vertex, and MiniMaxi.
- OpenAI-compatible config-first ergonomics now also mirror the builder-owned HTTP convenience surface directly on `OpenAiCompatibleConfig`: timeout / connect-timeout, streaming-compression toggle, user-agent / proxy, HTTP headers, and single-interceptor composition now have canonical config-first shorthands, while `OpenAiCompatibleClient::with_http_client(...)` remains the explicit config-first equivalent of builder-side client injection.
- Groq config-first ergonomics now also mirror the wrapper builder’s common HTTP surface directly on `GroqConfig`: timeout / connect-timeout, streaming-compression toggle, and single-interceptor composition now have canonical config-first helpers instead of forcing bulk `HttpConfig` mutation for those common lanes, and a feature-scoped integration test now pins builder/config convergence on that HTTP lane.
- xAI config-first ergonomics now also mirror the wrapper builder’s common HTTP surface directly on `XaiConfig`: timeout / connect-timeout, streaming-compression toggle, and single-interceptor composition now have canonical config-first helpers instead of forcing bulk `HttpConfig` mutation for those common lanes, and a feature-scoped integration test now pins builder/config convergence on that HTTP lane.
- DeepSeek config-first ergonomics now also mirror the wrapper builder’s common HTTP surface directly on `DeepSeekConfig`: timeout / connect-timeout, streaming-compression toggle, and single-interceptor composition now have canonical config-first helpers instead of forcing bulk `HttpConfig` mutation for those common lanes, and a feature-scoped integration test now pins builder/config convergence on that HTTP lane.
- Ollama config-first ergonomics now also mirror the provider builder’s common HTTP surface on `OllamaConfigBuilder`: timeout / connect-timeout, streaming-compression toggle, and single-interceptor composition now have canonical config-first helpers instead of forcing a bulk `HttpConfig` snapshot for those common lanes, and the package-local provider tests now pin builder/config convergence on that HTTP lane.
- Cohere and TogetherAI config-first ergonomics now also mirror their rerank builders’ common HTTP surface directly on `CohereConfig` / `TogetherAiConfig`: timeout / connect-timeout and single-interceptor composition now have canonical config-first helpers, and package-local parity tests pin `CohereBuilder::into_config()` / `TogetherAiBuilder::into_config()` against those same rerank-only config surfaces.
- OpenAI and Anthropic config-first ergonomics now also mirror their provider builders’ common HTTP surface directly on `OpenAiConfig` / `AnthropicConfig`: timeout / connect-timeout, streaming-compression toggle, and single-interceptor composition now have canonical config-first helpers instead of forcing raw `HttpConfig` mutation for those routine cases, and package-local config tests pin those helpers directly.
- Azure, Google Vertex, and MiniMaxi config-first ergonomics now also mirror the provider builders’ common HTTP surface directly on `AzureOpenAiConfig`, `GoogleVertexConfig`, `VertexAnthropicConfig`, and `MinimaxiConfig`: timeout / connect-timeout plus single-interceptor composition are now explicit config-first helpers everywhere, and the two Vertex configs plus MiniMaxi also expose the streaming-compression toggle on the same canonical config-first surface; package-local config tests pin each provider-owned config independently.
- Gemini’s canonical config-first HTTP story is now tightened without breaking legacy code: protocol-owned `GeminiConfig` keeps the historic `with_timeout(u64)` path for backward compatibility, but new config-first code can now use `with_http_timeout(...)`, `with_connect_timeout(...)`, `with_http_stream_disable_compression(...)`, and `with_http_interceptor(...)` directly on the config surface instead of dropping to raw `HttpConfig`, and protocol-level config tests pin that compatibility bridge.
- Azure config-first helper parity is now explicit too: `AzureOpenAiConfig` mirrors builder-owned family aliases plus API-version / deployment-based URL and chat-mode shorthands, so config-first Azure setup no longer needs direct `AzureUrlConfig` mutation just to match the provider-owned builder surface.
- Azure non-text public-path parity is now explicit too: embedding and image-generation requests now match across builder/provider/config-first plus the public registry facade, and `registry.image_model("azure:...")` now has a direct mixed-registry override guard so Azure-specific `api-key` / base-url / transport precedence is no longer implied only by text-family `responses` coverage.
- Azure audio public-path parity is now explicit too: TTS and STT requests now match across builder/provider/config-first plus the public registry facade, `registry.speech_model("azure:...")` / `registry.transcription_model("azure:...")` now have direct mixed-registry override guards, and the provider-owned Azure audio client now backfills missing request models from the configured deployment id instead of leaking generic OpenAI defaults into Azure routes.
- The unified Azure facade now preserves provider-owned URL-routing and metadata-namespace semantics too: `Siumai::builder().azure()` / `Provider::azure()` forward `api_version(...)`, `deployment_based_urls(...)`, full `url_config(...)`, and `provider_metadata_key(...)` into the registry factory path, so deployment-based URL routing plus Responses metadata namespace selection stay aligned with config-first/provider-first Azure construction on embedding, image, speech, transcription, and chat instead of silently falling back to the default factory settings.
- xAI and Groq parity now also cover both top-level and registry-native 200-response / SSE `StreamEnd` typed metadata extraction, so builder/provider/config-first plus `language_model("{provider}:...")` paths stay aligned on `sources` / `logprobs` instead of only request shaping.
- xAI and Groq mixed-registry override convergence is now explicit too: provider-scoped `ProviderBuildOverrides` on the real registry chat-response plus `StreamEnd` paths preserve `provider_metadata["xai"]` / `provider_metadata["groq"]`, so typed `XaiChatResponseExt` / `GroqChatResponseExt` extraction no longer depends on the non-override lane.
- DeepSeek mixed-registry override convergence is now explicit too: provider-scoped `ProviderBuildOverrides` on the real registry chat-response plus `StreamEnd` paths preserve `provider_metadata["deepseek"]`, so typed `DeepSeekChatResponseExt` extraction no longer depends on the non-override lane either.
- The provider-owned wrapper request override story is now explicit on both public and contract surfaces for xAI, Groq, and DeepSeek: registry `language_model("{provider}:...")` now has direct mixed-registry anchors for both `chat_request` and `chat_stream_request`, so provider-scoped auth/base-url/transport precedence is no longer only implied by config parity or non-stream-only tests.
- Bedrock rerank mixed-registry override coverage now also reaches the public registry facade, so `registry.reranking_model("bedrock:...")` no longer relies only on lower-contract override tests to pin Bedrock-specific auth/base-url routing.
- Fireworks transcription mixed-registry override coverage now also reaches both the public registry facade and lower contract on the dedicated audio host path, so compat audio routing preserves provider-scoped auth/base-url/transport on `/audio/transcriptions` instead of implicitly depending on the default `audio.fireworks.ai` preset only.
- SiliconFlow speech/transcription mixed-registry override coverage now also reaches both the public registry facade and lower contract, so compat audio routing preserves provider-scoped auth/base-url/transport on `/audio/speech` and `/audio/transcriptions` instead of implicitly falling back to the shared global transport lane.
- Together speech/transcription mixed-registry override coverage now also reaches both the public registry facade and lower contract, so compat audio routing preserves provider-scoped auth/base-url/transport on `/audio/speech` and `/audio/transcriptions` instead of implicitly falling back to the shared global transport lane.
- Together/SiliconFlow compat image mixed-registry override coverage now also reaches both the public registry facade and lower contract, so shared compat image routing preserves provider-scoped auth/base-url/transport on `/images/generations` instead of implicitly falling back to registry-global defaults.
- SiliconFlow/Jina/VoyageAI compat rerank mixed-registry override coverage now also reaches both the public registry facade and lower contract, so shared compat rerank routing preserves provider-scoped auth/base-url/transport on `/rerank` instead of implicitly falling back to registry-global defaults.
- Compat embedding mixed-registry override coverage now also reaches both the public registry facade and lower contract for the current preset set (`mistral`, `fireworks`, `siliconflow`, `together`, `openrouter`, `jina`, `voyageai`, `infini`), so shared compat embedding routing preserves provider-scoped auth/base-url/transport on `/embeddings` instead of implicitly falling back to registry-global defaults.
- Anthropic mixed-registry override coverage now also reaches the public registry facade on both request modes, and the lower contract now pins the stream lane too, so provider-scoped `x-api-key` / base-url / transport routing stays explicit on the real `/messages` path instead of relying on non-stream-only or contract-only coverage.
- OpenAI / Gemini / Vertex / Ollama mixed-registry override coverage now also reaches the public registry facade on their real native request lanes, so those provider-owned paths no longer rely on lower-contract-only anchors to prove provider-scoped auth/base-url/transport precedence on `responses`, `generateContent`, or `api/chat`.
- Vertex / Ollama mixed-registry override coverage now also reaches both public and lower-contract stream lanes, so those provider-owned stream paths no longer rely on config parity alone to prove provider-scoped auth/base-url/transport precedence on `streamGenerateContent` or streaming `api/chat`.
- Vertex mixed-registry override coverage now also reaches the native Imagen family on the public registry facade, so both `registry.image_model("vertex:imagen-4.0-generate-001")` generation and `registry.image_model("vertex:imagen-3.0-edit-001")` edit preserve provider-scoped API key / base-url / transport precedence on the real `:predict` path instead of leaving Imagen handle routing implied by chat-only coverage.
- OpenAI / Gemini native stream request coverage is now explicit too: OpenAI now has direct mixed-registry override anchors on the `responses` SSE lane, and Gemini now has both request-side stream parity plus mixed-registry override anchors on `streamGenerateContent`, so those native stream paths no longer depend on response-metadata-only checks or non-stream-only coverage.
- Gemini / Vertex lower-contract request-option coverage now also reaches the provider-owned builder/config/registry layer, so native `generateContent` packages no longer rely only on top-level public parity to prove Stable `response_format` / `tool_choice` plus provider-owned `thinkingConfig` / `structuredOutputs` convergence.
- Gemini / Vertex public registry handles now also pin that same Stable request-option contract directly on the facade layer, so `registry.language_model("{provider}:...")` no longer trails config-first clients on structured-output or tool-routing semantics for native `generateContent`.
- Anthropic-on-Vertex native structured-output coverage now also spans both lower-contract and top-level public paths, so the provider-owned `output_format` lanes on `:rawPredict` and `:streamRawPredict` are locked across builder/provider/config-first/registry with the same non-stream success/failure plus stream complete-vs-truncated JSON extraction contract instead of trailing either native Anthropic fallback tests or provider-local request-shape regressions; a matching runnable config-first example now exists as `vertex_anthropic_structured_output`.
- Anthropic-on-Vertex native reasoning coverage now also spans both lower-contract and top-level public paths, so raw `providerOptions.anthropic.{thinking_mode,thinkingMode,thinking}` converge to the native `thinking` body on both `:rawPredict` and `:streamRawPredict`, while non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` and final `providerMetadata.anthropic.{thinking_signature,redacted_thinking_data}` stay aligned on that provider-owned/public wrapper story instead of trailing generic Anthropic reasoning coverage.
- Anthropic-on-Vertex now also has provider-owned typed request helpers on its public facade, so `provider_ext::anthropic_vertex::{VertexAnthropicOptions, VertexAnthropicChatRequestExt}` can express native thinking/structured-output/tool-routing hints without raw JSON escape hatches while the same typed serialization is pinned both in provider-local unit coverage and the existing no-network reasoning parity path.
- Anthropic-on-Vertex now also exposes the protocol-owned Anthropic typed metadata facade on its own wrapper path, so `provider_ext::anthropic_vertex::{AnthropicChatResponseExt, AnthropicMetadata}` can read native Vertex-wrapper reasoning metadata under the `google-vertex` feature without enabling the separate Anthropic provider package, and both provider-local plus top-level public-path reasoning checks now assert that typed facade instead of raw nested metadata maps.
- Anthropic-on-Vertex config-first and builder-first construction now also persist provider-owned default request options before request-local overrides, so `VertexAnthropicConfig` / `VertexAnthropicBuilder` can carry default `thinking_mode`, `structured_output_mode`, `disable_parallel_tool_use`, and `send_reasoning` through the same `provider_options_map["anthropic"]` adapter path as request-level typed options; provider-local regression coverage now locks both config-merge precedence and builder `into_config()` preservation.
- The shared `Siumai::builder().anthropic_vertex()` path now also carries those same Anthropic-on-Vertex typed defaults through namespaced fluent helpers on the unified builder, so the public builder/provider/config story is closed without burning globally generic method names that would conflict with future shared-builder Anthropic affordances; the top-level compile guard and focused no-network parity test now pin convergence on the final `:rawPredict` body across `Siumai::builder()`, `Provider::anthropic_vertex()`, and `VertexAnthropicClient::from_config(...)`, including thinking-budget expansion, reserved-`json` tool fallback, `disable_parallel_tool_use`, and `send_reasoning = false` reasoning-input stripping.
- Groq top-level parity now also covers provider-owned typed request builders on both `chat_request` and `chat_stream_request`, so `GroqOptions` is locked on the real builder/provider/config-first transport boundary instead of only through provider-local spec tests.
- Groq registry language-model handles now also participate in that request-side contract for both request modes, so provider-owned typed request helpers are no longer builder/provider/config-only coverage on the text-family path.
- Groq lower-contract request-option coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `GroqOptions` helpers are pinned on both request modes even before the public-path facade enters the picture.
- Groq lower-contract coverage now also pins Stable `tool_choice` and `response_format` precedence directly on the provider-owned builder/config/registry layer, so structured-output and tool-routing merge rules are executable before the top-level facade path is involved.
- Groq config-first convergence now also covers provider-owned default request shaping, not only per-request escape hatches: `GroqConfig` and `GroqBuilder` can persist `logprobs`, `top_logprobs`, `service_tier`, `reasoning_effort`, and `reasoning_format` defaults through the same provider-owned adapter path that already powers request-level `GroqOptions`.
- Groq public-path parity now also locks response-side reasoning extraction on that same provider-owned wrapper story, so non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` accumulation stay aligned across builder/provider/config-first/registry while provider-specific `reasoning_effort` / `reasoning_format` hints keep converging on the same final request body.
- Groq public-path parity now also locks Stable `response_format` precedence plus shared-SSE structured-output extraction across builder/provider/config-first plus `language_model("groq:...")`, so the provider-owned wrapper story now carries executable structured-output coverage instead of stopping at spec-local normalization tests.
- Groq public-path parity now also locks interrupted synthetic-unknown-stream-end structured-output failure semantics on that same wrapper story, so incomplete streamed JSON now fails with the same public `ParseError` contract across builder/provider/config-first/registry instead of relying only on facade-level failure tests.
- Groq public-path parity now also locks tool-loop invariants on the real provider-owned wrapper story, so non-stream `tool_calls()` and streaming `ToolCallDelta` accumulation plus `StreamEnd.finish_reason` stay aligned across builder/provider/config-first plus `language_model("groq:...")` construction instead of relying only on compat adapter smoke tests.
- DeepSeek top-level parity now also covers provider-owned typed request builders on both `chat_request` and `chat_stream_request`, so `DeepSeekOptions` is locked on the real builder/provider/config-first transport boundary instead of only through raw provider-map coverage.
- DeepSeek registry language-model handles now also participate in that request-side contract for both request modes, so provider-owned typed request helpers are no longer builder/provider/config-only coverage on the text-family path there either.
- Provider-owned `xai`, `groq`, and `deepseek` wrappers now also pin the final `chat_stream_request` transport body directly, so no-network provider-local regressions lock `Accept: text/event-stream`, `stream=true`, Stable `response_format` / `tool_choice`, and each wrapper's typed or normalized request-option family without depending solely on facade-level parity tests.
- DeepSeek public-path parity now also locks response-side reasoning extraction on that same wrapper story, so non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` accumulation stay aligned across builder/provider/config-first/registry instead of being verified only on config-first examples and provider-local fixture tests.
- DeepSeek public-path parity now also locks Stable `tool_choice` precedence over raw `providerOptions.deepseek.tool_choice` across builder/provider/config-first plus `language_model("deepseek:...")`, so that merge rule no longer lives only in provider-local spec tests.
- DeepSeek public-path parity now also locks Stable `response_format` precedence plus shared-SSE structured-output extraction across builder/provider/config-first plus `language_model("deepseek:...")`, so the provider-owned wrapper story now carries executable structured-output coverage instead of stopping at spec-local normalization tests.
- DeepSeek public-path parity now also locks interrupted synthetic-unknown-stream-end structured-output failure semantics on that same wrapper story, so incomplete streamed JSON fails with the same public `ParseError` contract across builder/provider/config-first/registry while preserving Stable-over-raw schema precedence and `reasoningBudget` normalization.
- DeepSeek public-path parity now also locks tool-loop invariants on the real provider-owned wrapper story, so non-stream `tool_calls()` and streaming `ToolCallDelta` accumulation plus `StreamEnd.finish_reason` stay aligned across builder/provider/config-first plus `language_model("deepseek:...")` construction instead of relying only on core runtime-provider tests.
- Ollama top-level parity now also covers provider-owned typed request builders on both `chat_request` and `chat_stream_request`, so `OllamaOptions` is locked on the real builder/provider/config-first transport boundary instead of only through raw provider-map coverage; the native chat request builder now also forwards the previously dropped `raw` flag into the final `/api/chat` body.
- Ollama registry language-model handles now also participate in that request-side contract for both request modes, so provider-owned typed request helpers are no longer builder/provider/config-only coverage on the text-family path there either.
- Ollama lower-contract request-option coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `OllamaOptions` helpers — including `raw` — are pinned on both request modes even before the public-path facade enters the picture.
- MiniMaxi top-level parity now also covers provider-owned typed request builders on both `chat_request` and `chat_stream_request`, so `MinimaxiOptions` is locked on the real builder/provider/config-first transport boundary instead of only through provider-local normalization coverage.
- MiniMaxi registry language-model handles now also participate in that request-side contract for both request modes, so provider-owned typed request helpers are no longer builder/provider/config-only coverage on the text-family path there either.
- MiniMaxi lower-contract request-option coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `MinimaxiOptions` helpers are pinned on both request modes even before the public-path facade enters the picture.
- MiniMaxi config-first convergence now also covers provider-owned default chat shaping, not only request-local escape hatches: `MinimaxiConfig` and `MinimaxiBuilder` can persist thinking-mode and structured-output defaults in `provider_options_map`, the shared `Siumai::builder().minimaxi()` facade now exposes the matching namespaced default-option helpers, and `MinimaxiClient` merges those defaults before request-local overrides so config-first, builder-first, and top-level facade construction stop trailing the per-request extension path.
- MiniMaxi top-level default-option parity now also closes on both request modes: no-network public-path tests pin builder/provider/config-first default `thinking_mode` + `response_format` shaping on the final `/v1/messages` request body, so facade-level construction can no longer drift away from the already-landed provider-local default-merge contract.
- Provider-owned `MinimaxiClient` now also pins the final `chat_stream_request` transport body directly, so no-network provider-local regression coverage locks `Accept: text/event-stream`, `stream=true`, typed thinking-mode shaping, Stable `response_format` precedence, and the associated tool stripping without depending solely on facade-level parity tests.
- Anthropic registry language-model handles now also participate in the same request-side contract for both request modes, so provider-owned typed request helpers are now locked end-to-end across builder/provider/config/registry instead of stopping at the pre-registry public paths.
- Anthropic lower-contract request-option coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `AnthropicOptions` helpers now pin `thinking_mode`, `context_management`, `container`, provider-owned structured output, and `output_config.effort` on both request modes below the public-path facade.
- Anthropic builder-first/config-first/shared-builder construction now also persists provider-owned typed default request options through namespaced fluent helpers, so default `thinking_mode`, reserved-`json` tool fallback, `context_management`, `effort`, and `tool_streaming` behavior can be configured once on `Provider::anthropic()`, `AnthropicConfig`, and `Siumai::builder().anthropic()` instead of being repeated per request; focused no-network public-path parity now pins the final streaming `/v1/messages` body across those three construction paths, including thinking-budget expansion, reserved `json` tool injection, context-management/output-config mapping, and `tool_streaming = false` beta suppression.
- Provider-owned `AnthropicClient` now also pins the final `chat_stream_request` transport body directly, so no-network provider-local regression coverage now locks `Accept: text/event-stream`, `stream=true`, typed `thinking_mode`, `context_management`, `container`, provider-owned `output_format` / `output_config`, and the opt-out of `fine-grained-tool-streaming` without relying solely on facade-level parity tests.
- Gemini native structured-output stream depth now also reaches the provider-owned client path below the facade layer, so `GeminiClient::chat_stream_request(...)` directly locks Stable `response_format -> responseMimeType/responseSchema` shaping, complete accumulated JSON plus final `StreamEnd` metadata, and the matching truncated interrupted-stream `ParseError` split instead of leaving that guarantee only to public-path parity plus request-shape tests.
- Google Vertex stream Stable request-option coverage now also reaches both public-path and lower-contract layers, so `streamGenerateContent` no longer relies on non-stream-only anchors to prove Stable `response_format` / `tool_choice` plus provider-owned `thinkingConfig` / `structuredOutputs` convergence.
- Provider-owned `GoogleVertexClient` now also pins the final `chat_stream_request` transport body directly, so no-network provider-local regression coverage now locks `Accept: text/event-stream`, the real `:streamGenerateContent?alt=sse` URL boundary, Stable `response_format` / `tool_choice`, and provider-owned `thinkingConfig` / `structuredOutputs` below the facade layer too.
- Google Vertex native structured-output stream depth now also reaches that provider-owned client path below the facade layer, so `GoogleVertexClient::chat_stream_request(...)` directly locks complete accumulated JSON plus final raw `provider_metadata["vertex"]` `StreamEnd` metadata and the matching truncated interrupted-stream `ParseError` split instead of leaving that response-side guarantee only to public-path parity plus request-shape tests.
- Google Vertex public-path structured-output depth now also reaches the real wrapper/registry story, so builder / provider / config-first / `language_model("vertex:...")` now all lock the same complete-vs-truncated stream split plus raw `provider_metadata["vertex"]` `StreamEnd` metadata on the native `streamGenerateContent` lane instead of relying on provider-local regressions alone.
- Provider-owned `OpenAiClient` now also exposes first-class resource accessors for `files`, `models`, `moderation`, and `rerank`, all rooted in a shared `resource_config()` snapshot so config-first runtime auth/base-url/organization/project/provider-options/HTTP state stays aligned when package callers step into provider-owned extras instead of rebuilding side clients manually.
- `provider_ext::gemini::resources` now also exposes `GeminiCachedContents` and `GeminiTokens` at the top-level package surface, so the public provider package no longer trails the real `GeminiClient` resource-accessor story when users need provider-owned cached-content or token-count helpers.
- Public provider-package compile guards now also pin the top-level resource modules for OpenAI, Anthropic, and the stable Google alias, so native provider-owned extras exports cannot drift out of `provider_ext::*::resources` without an immediate public-surface regression.
- Public provider-package compile guards now also pin the top-level non-unified `ext` modules for OpenAI, Anthropic, Gemini, and the stable Google alias, so provider-owned moderation/responses/audio-stream helpers, structured-output/thinking/tool-event helpers, and code-execution/file-search/tool-event escape hatches stay locked on the actual package facade instead of only existing in lower crates.
- Focused wrapper packages now also have matching `ext` facade compile anchors where they own real escape hatches: Groq's typed audio option helpers and MiniMaxi's video/music builders plus structured-output/thinking helper entry points are now locked on `provider_ext::*` instead of being exercised only through lower package imports.
- `provider_ext::xai::provider_tools` now also has a direct public compile anchor for `web_search`, `x_search`, and `code_execution`, so xAI's provider-owned tool-factory facade is locked on the real top-level package surface instead of being covered only indirectly through the shared `tools::xai` implementation.
- Compatibility aliases for provider tool factories are now also pinned on the public facade: `provider_ext::{openai,anthropic,gemini,google}::provider_tools` compile alongside the preferred `tools` modules, so older imports remain intentionally supported instead of passively inherited from internal re-exports.
- `provider_ext::google_vertex` now also has direct public compile anchors for the Vertex-specific `vertex_rag_store` tool on both the raw `tools` facade and the typed `hosted_tools` facade, so the package's unique RAG helper surface is guarded independently from the shared Google tool set.
- Focused wrapper helper tiers are now also pinned on the top-level package facade: `CohereClient`, `TogetherAiClient`, and `BedrockClient` compile with their public inspection/debug accessors (`provider_context` / base-url / retry / transport helpers, plus Bedrock's split runtime contexts/base URLs), so those provider-owned helper surfaces cannot silently drift behind lower package-only coverage.
- The remaining wrapper helper tiers are now also compile-guarded on the public facade: xAI, Groq, DeepSeek, and Ollama lock their provider-context/base-url/HTTP/retry accessor set directly on `provider_ext::*`, while OpenAI, Anthropic, Gemini, Google, and Google Vertex now also pin the smaller base-url / retry helper set they currently expose there.
- Shared Google tool alias facades now also have broader public compile coverage: Gemini, the stable Google alias, and Google Vertex pin representative `code_execution` / `google_maps` / `url_context` / `enterprise_web_search` plus `file_search` builder entry points in addition to `google_search`, reducing the chance that a single surviving alias masks drift in the rest of the tool module.
- xAI registry language-model handles now also participate in the provider-owned typed request-options contract for both request modes, so `XaiOptions` web-search fields are no longer builder/provider/config-only coverage on the text-family facade.
- xAI lower-contract request-option coverage now also reaches the provider-owned builder/config/registry layer directly, so typed `XaiOptions` web-search helpers are pinned on both request modes even before the public-path facade enters the picture.
- xAI lower-contract coverage now also pins Stable `tool_choice` and `response_format` precedence directly on the provider-owned builder/config/registry layer, so structured-output and tool-routing merge rules are executable before the top-level facade path is involved there too.
- xAI config-first convergence now also covers provider-owned default request shaping, not only per-request escape hatches: `XaiConfig` and `XaiBuilder` can persist `reasoning_effort` plus default web-search parameters through the provider-owned adapter path, and no-network client coverage now locks that those defaults still land in the final request body before request-local overrides are applied.
- Anthropic builder/config convergence now also covers the rest of the provider-owned parameter helper surface, not only the previously mirrored thinking subset: builder-first construction can now replace full `AnthropicParams`, bulk metadata, `stream`, and beta-feature defaults through the same provider-owned config shape as `AnthropicConfig`, so builder-first Anthropic setup no longer trails config-first request shaping on those fields.
- Gemini builder/config convergence now also covers the rest of the provider-owned generation-config helper surface, not only the earlier shorthand subset: builder-first construction can now use `with_top_k`, `with_candidate_count`, JSON-schema / reasoning aliases, and a base `with_generation_config(...)` that is overridden by later builder-local helpers, so Gemini builder-first setup no longer trails config-first generation shaping.
- Google Vertex builder/config convergence now also covers shared request defaults instead of only endpoint/auth wiring: `GoogleVertexConfig` now persists `common_params`, and `GoogleVertexBuilder::into_config()` forwards temperature / max_tokens / top_p / stop-sequence defaults into that canonical config so config-first Vertex clients stop trailing builder-built clients on chat and embedding defaults.
- OpenAI builder/config convergence now also covers the config-first naming surface itself, not only final config equivalence: `OpenAiBuilder` now exposes `with_*` aliases for common params, Responses API defaults, and provider-options defaults, and alias-based builder coverage now locks that those entry points converge on the same canonical `OpenAiConfig` as the original builder vocabulary.
- Ollama builder/config convergence now also covers the provider-owned default and middleware surface, not only final metadata extraction: `OllamaBuilder` can now bulk seed `common_params` and `OllamaParams`, replace the full `HttpConfig`, append extra model middlewares, and use the provider-native `think` alias, while focused builder/config contract coverage locks those helpers against the canonical `OllamaConfig`.
- The top-level `siumai` facade now also closes the last Ollama bulk-param reachability gap: `provider_ext::ollama::OllamaParams` is exported on the public facade, and `Provider::ollama()` has direct compile-and-config coverage for the bulk-param / HTTP-config / middleware parity helpers instead of relying on provider-local tests alone.
- Ollama public-path parity now also covers registry-native embedding handles on the real provider-owned wrapper story: request-level `OllamaEmbeddingOptions` stay aligned across builder/provider/config-first plus `registry.embedding_model("ollama:...")`, and provider-specific registry build overrides are now locked on the embedding family instead of only the chat family.
- OpenAI public-path parity now also locks provider-specific registry build overrides on the speech and transcription family handles themselves, so `registry.speech_model("openai:...")` and `registry.transcription_model("openai:...")` are verified against the provider override transport/base URL/API key path instead of relying on chat-handle override coverage by analogy.
- OpenAI public-path parity now also covers the native image family on that same provider-owned story: image-generation requests are locked across builder/provider/config-first plus `registry.image_model("openai:...")`, the registry image handle now has explicit provider-override coverage instead of inheriting confidence only from compat-image vendors, and the builder-default `responsesApi` setting is now stripped from non-chat families instead of leaking into image/audio/embedding/rerank request bodies.
- xAI public-path parity now also locks tool-loop invariants on the real provider-owned wrapper story, so non-stream `tool_calls()` and streaming `ToolCallDelta` accumulation plus `StreamEnd.finish_reason` stay aligned across builder/provider/config-first plus `language_model("xai:...")` construction instead of relying only on core compat tests.
- Google Vertex registry-native non-text family handles now also participate in request-helper parity directly: `VertexEmbeddingOptions` and `VertexImagenOptions` are locked against config-first `GoogleVertexClient` on the final embedding / image-generation / image-edit transport boundary instead of relying only on factory-level contract tests, and the embedding side is now rooted in a concrete public-handle fix where `EmbeddingModelHandle::embed_with_config` dispatches to provider-owned typed embedding clients before falling back to raw capability adapters.
- DeepSeek top-level parity now also covers both 200-response and SSE `StreamEnd` typed metadata extraction, so builder/provider/config-first paths stay aligned on `DeepSeekMetadata.logprobs` instead of only request shaping and reasoning defaults.
- Ollama top-level parity now also covers JSON-stream `StreamEnd` typed metadata extraction, so builder/provider/config-first paths stay aligned on timing metadata instead of only request shaping.



- OpenAI and Anthropic config-first surfaces now also expose fluent setters for builder-era provider defaults, reducing builder-only configuration drift.



- OpenAI-compatible config-first construction now also exposes fluent reasoning/thinking defaults, so adapter-based provider extras are no longer builder-only.
- Compat vendor-view registry convergence is now explicit too: OpenRouter and Perplexity public-path parity now covers the registry handle path for both non-streaming and streaming chat, OpenRouter’s mixed-registry override lane now also reaches the registry embedding handle, the request-side anchors are pinned directly on typed `OpenRouterOptions` / `PerplexityOptions` plus OpenRouter embedding request fields and SSE `Accept` header preservation, provider-scoped OpenRouter reasoning defaults flow through `ProviderBuildOverrides -> BuildContext -> OpenAiCompatibleClient` instead of being lost on registry construction, and `RegistryBuilder` / `RegistryOptions` now expose the same reasoning defaults globally with provider overrides taking precedence.
- The compat public-path test surface is now cleaner in feature-minimal mode too: the `openai`-only test build no longer drags a stray MiniMaxi registry helper into scope, so OpenRouter / Perplexity override anchors remain runnable under `--no-default-features --features "openai"` instead of requiring unrelated provider features just to compile.
- Compat vendor-view metadata namespace convergence is now explicit too: OpenRouter chat responses and stream-end responses now preserve `provider_metadata["openrouter"]` across builder/provider/config/registry paths, the extracted `logprobs` payload stays attached to the vendor namespace instead of drifting to a generic compat root, the mixed-registry override lane now preserves that same metadata root on both the real registry chat-response path and the real registry `StreamEnd` path, and the public `provider_ext::openrouter::OpenRouterChatResponseExt` plus `OpenRouterSourceExt` / `OpenRouterContentPartExt` surfaces now expose that metadata as typed accessors; Perplexity registry chat-response and stream-end metadata now also match the config-first typed metadata path, including when the request is routed through provider-scoped overrides.
- That registry-global reasoning lane is now also anchored on the direct builder API instead of only the struct-literal path: OpenRouter has no-network parity for both `RegistryOptions` and `RegistryBuilder` global defaults, while DeepSeek and xAI now do too on their provider-owned wrapper stories.



- Gemini config-first construction now also exposes fluent generation/thinking helpers so `candidate_count`, JSON schema, and reasoning defaults do not require builder-only ergonomics.



- xAI now owns `XaiConfig` / `XaiClient` entry types over the shared compat runtime, its wrapper builder still mirrors OpenAI-compatible reasoning/thinking convenience, DeepSeek now owns `DeepSeekConfig` / `DeepSeekClient` entry types over the shared compat runtime, and Ollama config-first construction now has direct common-parameter convenience instead of requiring full `CommonParams` assembly.



- Groq config-first construction now has explicit HTTP config/header convenience, reducing the need for direct field mutation in config-based paths.



- Google Vertex config-first construction now also exposes `new` / `express` / `enterprise` convenience constructors plus builder `into_config()` convergence, so config-first setup no longer depends on public struct literals.
- Google Vertex registry construction now also has explicit native text/image/embedding family paths backed by the provider-owned `GoogleVertexClient`, and its generic embedding factory path no longer drops `BuildContext` by falling back to the default no-context registry bridge.
- Google Vertex facade/public-path parity now also covers chat, chat streaming, embedding, image generation, and extension-only image editing across `Siumai::builder().vertex()`, `Provider::vertex()`, and config-first clients, so transport-boundary request shaping stays aligned even outside the stable image family surface.
- Anthropic-on-Vertex now also has a complete provider-owned builder/config/client path with explicit `Provider::anthropic_vertex()` and `Siumai::builder().anthropic_vertex()` entry points, and both top-level public-path parity plus builder/config/registry contract coverage now lock `:rawPredict` / `:streamRawPredict` request convergence at the final transport boundary, including explicit full-request model overrides across config-first, public wrapper, and registry text-handle paths.
- Anthropic-on-Vertex now also has a runnable provider-owned example and builder-alignment coverage, and its enterprise auth story now matches the main Vertex client more closely: provider-owned config/builder surfaces accept a token provider, runtime context injects Bearer auth lazily, provider-owned builder auto-enables ADC under `gcp`, registry construction also consumes `BuildContext.google_token_provider` (with backward-compatible fallback to `gemini_token_provider`) with the same fallback behavior, and the unified builder now exposes neutral Google/Vertex auth aliases instead of only Gemini-branded method names.
- The shared registry/auth naming cleanup is now also executable rather than aspirational: `SiumaiBuilder` stores Google-family auth under a neutral `google_token_provider`, emits both new/legacy fields into `BuildContext`, and focused tests lock the alias behavior for Gemini and Anthropic-on-Vertex public paths.
- Anthropic-on-Vertex now also closes the same capability-split lower boundary as the other focused wrappers: deferred `embedding` / `image` / `rerank` / audio-family builders reject explicitly with `UnsupportedOperation`, and both registry-contract coverage plus public registry-handle tests pin the same no-request failure path below the chat-family surface.
- Gemini typed metadata now also exposes `thoughtSignature` as a first-class field on `GeminiMetadata`, and the Google/Vertex alignment tests now consume `GeminiChatResponseExt` for response-level alias parsing so the `google`/`vertex` metadata split is no longer guarded only through raw map access.
- Gemini now also exposes `GeminiContentPartExt` for part-level metadata reads, so Google thought-signature fixture checks for text/reasoning/tool-call parts no longer depend on raw nested provider-map traversal; the dedicated `google-vertex` fixture still keeps explicit raw namespace-key assertions on streaming custom events, but the plain Vertex wrapper now also has its own typed metadata facade for response/content-part reads under `provider_ext::google_vertex::metadata`.
- The complementary package-tier boundary is now narrower rather than absent: `provider_ext::google_vertex` is no longer options/tools-only, but its new typed metadata surface stays intentionally strict and provider-owned — `VertexMetadata`, `VertexChatResponseExt`, and `VertexContentPartExt` bind only to `provider_metadata["vertex"]` without falling back to `google`, while streaming `reasoning-start` / `reasoning-delta` payloads remain raw and namespace-scoped.
- Gemini now also has a matching builder/config/registry contract anchor for provider-owned batch embedding request shaping, locking `:batchEmbedContents` endpoint selection plus per-item request-body convergence on the provider-factory path itself.
- Gemini now also closes its unsupported audio-family boundary at the lower factory layer: provider-factory `speech` / `transcription` construction fails fast instead of aliasing the embedding path, matching the existing public registry/audio negative guards.

- Cohere rerank migration now also has a provider-owned config/client/builder path, registry construction now materializes the provider-owned typed client directly, and top-level public-path parity now locks rerank request shaping across siumai/provider/config-first entry points.
- Cohere now also has a matching builder/config/registry contract anchor for provider-owned rerank request shaping, locking final `/rerank` URL selection, Bearer auth propagation, and the native `max_tokens_per_doc` / `priority` body fields on the provider-factory path itself.
- `siumai-registry` now also compiles cleanly under standalone `cohere` / `togetherai` rerank-only builds with the shared provider-inference helper kept in scope, removing another feature-boundary regression in `SiumaiBuilder`.
- TogetherAI rerank migration now also has a provider-owned config/client/builder path, registry construction now materializes the provider-owned typed client directly, and top-level public-path parity now locks rerank request shaping across siumai/provider/config-first entry points.
- Amazon Bedrock migration now also has a provider-owned config/client/builder path with split runtime vs agent-runtime endpoint ownership, registry construction now materializes the provider-owned typed client directly, and top-level public-path parity now locks chat, chat-stream, and rerank request shaping across siumai/provider/config-first entry points.
- Amazon Bedrock now also has matching builder/config/registry contract anchors for chat, chat-stream, and rerank request shaping, locking runtime vs agent-runtime endpoint routing, Bearer auth propagation, and Bedrock-specific request-body passthrough on the provider-factory path itself.
- Amazon Bedrock now also has the matching top-level public registry boundary: `registry.reranking_model("bedrock:...")` matches config-first `/rerank` request shaping, and public `registry.embedding_model("bedrock:...")` / `registry.image_model("bedrock:...")` now fail before transport instead of trailing the lower contract coverage.
- Amazon Bedrock top-level registry parity now also has explicit request-shape coverage for both `chat_request` and `chat_stream_request`, so `registry.language_model("bedrock:...")` is pinned directly on `/converse` and `/converse-stream` routing instead of only indirectly through metadata and stable-option assertions.
- Amazon Bedrock registry override precedence is now also pinned on the public text handle for both `chat_request` and `chat_stream_request`: `RegistryBuilder::with_provider_build_overrides("bedrock", ...)` drives `language_model("bedrock:...")` to the provider-scoped auth/base-url/transport lane on the final `/converse` and `/converse-stream` requests instead of relying only on the rerank-handle contract.
- Amazon Bedrock now also has the matching lower contract for that text-handle override lane inside `siumai-registry`, so provider-scoped `language_model("bedrock:...")` override precedence is guarded at both the public facade and factory/handle boundary for both non-streaming and streaming chat instead of existing only as a top-level regression test.
- Amazon Bedrock lower-contract coverage now also includes stable request options, so the provider-factory path directly guards `tool_choice`, structured-output `json` tool injection, and `additionalModelRequestFields` convergence on `/converse` in addition to the public-path stable-option regression.
- Focused provider-owned wrappers now also expose the same inspection/helper tier as the broader wrapper family: `CohereClient` and `TogetherAiClient` provide `provider_context` / `base_url` / retry / transport accessors, while `BedrockClient` additionally exposes split runtime vs agent-runtime helper contexts, reducing the amount of provider-specific test/debug code that still needs private-field reach-through.
- Amazon Bedrock now also has a provider-owned typed metadata escape hatch on the public surface: `provider_ext::bedrock::metadata::{BedrockMetadata, BedrockChatResponseExt}` parses the existing `provider_metadata["bedrock"]` payload, so Bedrock-specific chat metadata is no longer trapped behind raw map traversal.
- That Bedrock metadata escape hatch is now exercised by public alignment tests and focused provider-surface guards too, turning it into an enforced boundary rather than a passive export.
- Bedrock top-level parity now also covers both 200-response and JSON-stream `StreamEnd` typed metadata extraction, so builder/provider/config-first paths stay aligned on `BedrockMetadata` instead of only request shaping.
- Bedrock top-level registry parity now also covers both 200-response and JSON-stream `StreamEnd` typed metadata extraction, so `registry.language_model("bedrock:...")` stays aligned with config-first `BedrockMetadata` parsing instead of trailing the non-registry public paths.
- The Bedrock crate’s own reserved-`json` protocol test now also reads `isJsonResponseFromTool` through `BedrockChatResponseExt`, so the remaining response-side raw `provider_metadata["bedrock"]` assertion inside the native package is gone as well.
- `siumai-registry` now also compiles cleanly under the standalone `bedrock` feature with the shared provider-inference helper kept in scope, so Bedrock-focused validation no longer needs an unrelated extra feature just to satisfy a cfg boundary in `SiumaiBuilder`.
- The built-in registry surface now also registers the native `bedrock` factory when that feature is enabled, bringing `ProviderRegistry::with_builtin_providers()` back in line with the Bedrock migration story and removing the feature-minimal `unused import ids` warning caused by the previously missing branch.
- `siumai/build.rs` now also recognizes `cohere`, `togetherai`, and `bedrock` as first-class provider features, so rerank-only / Bedrock-only public/test builds no longer trip the crate-level provider guard before the actual provider-boundary checks run.
- `siumai-registry` now also passes a full native single-feature compile sweep across the current provider set, and the built-in registry registration checks for `cohere`, `togetherai`, and `bedrock` are all green in feature-minimal mode, so the registry abstraction layer is no longer quietly depending on multi-feature builds just to stay compilable.
- Rerank audit is now closed for the currently advertised catalog: focused provider-owned packages are already in good shape (Cohere / TogetherAI / Bedrock), and OpenAI-compatible `siliconflow`, `jina`, and `voyageai` now also have top-level public-path parity anchors across siumai/provider/config-first construction.
- TogetherAI now also has a matching builder/config/registry contract anchor for provider-owned rerank request shaping, locking final `/rerank` URL selection, Bearer auth propagation, and native `return_documents` / `rank_fields` body semantics on the provider-factory path itself.
- Cohere and TogetherAI now also have provider-scoped registry-handle rerank contracts on top of those factory anchors: `ProviderBuildOverrides` precedence is locked on the real `reranking_model(...)` path for both packages, so mixed registries can carry provider-specific auth / base-URL / transport overrides without cross-provider leakage.
- Cohere and TogetherAI rerank migration now also closes the top-level public registry lane: `registry.reranking_model("cohere:...")` and `registry.reranking_model("togetherai:...")` both match config-first `/rerank` request shaping, and the same public facade now also pins provider-scoped override precedence on the final rerank handle instead of relying only on lower-contract coverage.
- The shared OpenAI-compatible rerank runtime now preserves injected custom transports, fixing the parity gap where rerank could bypass capture/no-network transport wiring even though the other compat executors already honored it.
- Amazon Bedrock now also has a runnable config-first example and provider-local auth notes, so the public usage story matches the new provider-owned construction surface instead of stopping at tests and internal docs.

- MiniMaxi config-first construction now also converges through builder `into_config()`, provider-owned `with_http_client()`, and registry factory routing, so common params, HTTP config, interceptors, and transport no longer drift between builder/config/registry paths.
- MiniMaxi now also has matching builder/config/registry contract anchors for provider-owned chat, chat-stream, image, speech, and file-management request shaping, locking auth propagation, SSE headers, secondary-endpoint routing, typed TTS / image request bodies, and `/v1/files/*` normalization on the provider-factory path itself in addition to the already-landed top-level public-path parity tests.
- MiniMaxi top-level registry override precedence is now also pinned on the file-management public path: `RegistryBuilder::with_provider_build_overrides("minimaxi", ...)` drives `language_model("minimaxi:...").upload_file(...)` onto the provider-scoped host/auth lane instead of falling back to the registry-global defaults.
- MiniMaxi now also has the matching lower contract for that file-management override lane inside `siumai-registry`, so provider-scoped `language_model("minimaxi:...")` file operations are guarded at both the public facade and the factory/handle boundary.
- MiniMaxi file-management override precedence now also extends across `list_files`, `retrieve_file`, `get_file_content`, and `delete_file`, so the rest of `/v1/files/*` now keeps the same provider-scoped host/auth lane at both the public facade and lower contract layers instead of only `upload_file`.
- MiniMaxi video-generation override precedence is now also pinned at both the public facade and lower contract layers, so `RegistryBuilder::with_provider_build_overrides("minimaxi", ...)` keeps `language_model("minimaxi:...").create_video_task(...)` on the provider-scoped host/auth lane through `/v1/video_generation` normalization instead of falling back to registry-global defaults.
- MiniMaxi query-video override precedence is now also pinned at both the public facade and lower contract layers, so `RegistryBuilder::with_provider_build_overrides("minimaxi", ...)` keeps `language_model("minimaxi:...").query_video_task(...)` on the provider-scoped host/auth lane through `/v1/query/video_generation` normalization instead of falling back to registry-global defaults.
- MiniMaxi music-generation override precedence is now also pinned at both the public facade and lower contract layers, so `RegistryBuilder::with_provider_build_overrides("minimaxi", ...)` keeps `language_model("minimaxi:...").generate_music(...)` on the provider-scoped host/auth lane through `/v1/music_generation` normalization instead of falling back to registry-global defaults.
- MiniMaxi image-generation override precedence is now also pinned at both the public facade and lower contract layers, so `RegistryBuilder::with_provider_build_overrides("minimaxi", ...)` keeps `registry.image_model("minimaxi:...")` on the provider-scoped host/auth lane through `/v1/image_generation` normalization instead of falling back to registry-global defaults.
- MiniMaxi speech-generation override precedence is now also pinned at both the public facade and lower contract layers, so `RegistryBuilder::with_provider_build_overrides("minimaxi", ...)` keeps `registry.speech_model("minimaxi:...")` on the provider-scoped host/auth lane through `/v1/t2a_v2` normalization instead of falling back to registry-global defaults.
- Anthropic-on-Vertex top-level public registry parity now also closes the text handle lane: `registry.language_model("anthropic-vertex:...")` matches config-first `:rawPredict` / `:streamRawPredict` request shaping, so the public registry path no longer trails the lower builder/config/registry contract coverage.
- `ProviderBuildOverrides` now also carries provider-scoped `http_config`, and Anthropic-on-Vertex pins that lane at both the public facade and lower contract layers with merge-over-global semantics, so mixed registries can override Bearer auth headers, base URLs, and transports for header-auth providers on the final registry text handle path without dropping shared registry headers or timeout defaults.
- DeepSeek top-level public registry parity now also has explicit request-shape coverage for both non-streaming and streaming chat requests, so `registry.language_model("deepseek:...")` is pinned directly on `/chat/completions` routing plus normalized reasoning-field serialization instead of only indirectly through stable-option and metadata assertions.
- DeepSeek top-level public registry parity now also covers provider-option `tool_choice` shaping directly on `registry.language_model("deepseek:...")`, so public no-network coverage now guards that body convergence in addition to the lower builder/config/registry contract tests.
- DeepSeek top-level registry override precedence is now also pinned on the public chat handle: `RegistryBuilder::with_provider_build_overrides("deepseek", ...)` drives `language_model("deepseek:...")` onto the provider-scoped auth/base-url/transport lane on the final `/chat/completions` request instead of relying only on the lower contract.
- DeepSeek lower-contract coverage now also includes typed stable request options plus provider-option `tool_choice` / `response_format` shaping, so the provider-factory path directly guards reasoning normalization and stable-field/body convergence in addition to the public-path request regressions.
- MiniMaxi image generation parity is now also closed at the top-level registry boundary: `registry.image_model("minimaxi:...")` joins builder/provider/config-first construction on the same final `/v1/image_generation` request body, so the public image family no longer trails the provider-factory registry contract.
- MiniMaxi video generation parity is now also closed at the top-level registry boundary: `registry.language_model("minimaxi:...")` joins builder/provider/config-first construction on the same final `/v1/video_generation` request body, so the public extension capability no longer trails the provider-factory registry contract.
- MiniMaxi query-video and music parity are now also closed at the top-level registry boundary: `registry.language_model("minimaxi:...")` now joins builder/provider/config-first construction on both `/v1/query/video_generation` and `/v1/music_generation`, so the public extension capabilities no longer trail the provider-factory registry contract.
- MiniMaxi file-management parity is now also closed at the top-level registry boundary: `registry.language_model("minimaxi:...")` now joins builder/provider/config-first construction across `/v1/files/upload`, `/v1/files/list`, `/v1/files/retrieve`, `/v1/files/retrieve_content`, and `/v1/files/delete`, so the public file-management extension path no longer trails the provider-factory registry contract.

- MiniMaxi facade/public-path parity now also includes `Provider::minimaxi()` plus no-network siumai/provider/config request-capture tests for non-streaming chat, streaming chat, image generation, TTS, and file-management upload/list/retrieve/content/delete paths, chat-header construction is now locked to Bearer auth instead of leaking Anthropic default headers, and registry-native text/image/speech family construction now materializes the provider-owned `MinimaxiClient` directly instead of falling back to the generic client-backed family bridge.
- MiniMaxi now also exposes provider-owned typed response metadata under `provider_ext::minimaxi::metadata`, with local chat/stream metadata-key normalization so Anthropic-compatible runtime payloads surface under `provider_metadata["minimaxi"]` instead of leaking the borrowed protocol namespace.
- MiniMaxi parity now also covers both top-level and registry-native 200-response / `StreamEnd` typed metadata extraction on the real public construction path, and those no-network guards additionally pin the normalized provider root key so builder/provider/config-first and `language_model("minimaxi:...")` clients do not regress back to `provider_metadata["anthropic"]`.
- MiniMaxi `chat_stream_request` now also preserves `stream=true` on the provider-owned override path, closing the bug where the public streaming facade could emit a non-streaming request body even though execution continued through the SSE transport lane.
- MiniMaxi capability-split closure is now explicit too: image + speech remain the only public non-text families, while builder/provider/config-first and registry-native `embedding` / `rerank` / `transcription` entry points now all fail with the same no-transport unsupported contract.
- MiniMaxi capability-split closure now also reaches the provider-factory layer: deferred `embedding` / `rerank` / `transcription` family builders reject explicitly with `UnsupportedOperation`, and registry-handle contract tests pin the same no-request failure below the public facade.
- MiniMaxi typed TTS options now also have top-level public-path parity coverage: `MinimaxiTtsOptions` survive builder/provider/config-first construction consistently onto the final `/v1/t2a_v2` request body instead of being a config-first-only escape hatch.
- MiniMaxi registry speech handles now also participate in that typed TTS contract directly: `registry.speech_model("minimaxi:...")` matches config-first `MinimaxiClient` on the final `/v1/t2a_v2` request body, so provider-owned speech request helpers are no longer only guarded on the pre-registry public paths.
- MiniMaxi file-management public paths now also lock the provider-specific base-URL normalization rule: even when public construction starts from `.../anthropic/v1`, file endpoints converge back to `/v1/files/*` on the API root consistently across builder/provider/config-first clients, including metadata retrieval and raw-content download paths.
- MiniMaxi top-level public-path parity now also covers provider-owned video and music escape hatches: `create_video_task`, `query_video_task`, and `generate_music` converge across builder/provider/config-first construction, with typed `MinimaxiVideoRequestBuilder` / `MinimaxiMusicRequestBuilder` request shaping locked at the final transport boundary.
- MiniMaxi OpenAI-style non-chat endpoint normalization is now fixed at the root: when callers start from a custom `.../anthropic/v1` base URL, image/speech/video/music paths preserve the caller host and normalize only the path to `/v1/...` instead of silently hardcoding the production `api.minimaxi.com` domain.
- Registry `LanguageModelHandle` now also forwards provider-specific extension capability calls for surfaces without a stable family handle, and the handle-level regression set now covers `file_management` in addition to MiniMaxi’s already-anchored video-task creation, video-task query, and music generation routes, so provider-owned extension surfaces no longer get silently truncated at the generic registry boundary.
- Registry build overrides now also flow all the way into handle-driven provider construction: `api_key`, `base_url`, `http_client`, `http_transport`, and `http_config` are no longer stranded on builder/config-first paths, that layer now supports per-provider precedence through `ProviderBuildOverrides`, provider-scoped `http_config` now merges over registry-global defaults instead of replacing them wholesale, and the handle-level regression set plus new MiniMaxi / Bedrock / Anthropic-Vertex no-network registry contracts now lock that provider-owned construction surface directly.
- The same `http_config` merge rule is now also covered below the transport boundary: `registry::entry::ProviderBuildOverrides::merged_with()` has a dedicated field-level unit test for timeout/connect-timeout/header/proxy/user-agent/compression precedence, so the root helper cannot regress silently even if provider-specific end-to-end anchors still pass.
- The same provider-scoped override lane is now anchored on text-family public handles too: OpenAI, Anthropic, Gemini, Google Vertex, Ollama, DeepSeek, xAI, Groq, OpenRouter, and Perplexity now all have no-network `language_model(...)` chat contracts proving provider-specific auth, base-URL, and transport precedence at the final request boundary instead of relying only on factory-level `BuildContext` tests, with OpenAI `/responses`, Anthropic `/v1/messages`, Groq's root-base `.../openai/v1` normalization, Vertex express `?key=...`, and OpenRouter/Perplexity vendor-parameter merges all locked on the public handle path as well.



- Performance guardrail for V4-M3: config-first convergence should not introduce repeated adapter wrapping per request; adapter assembly must remain a one-time construction concern.



- DeepSeek config/client convergence now follows that guardrail by recovering provider-specific compat defaults during config/build conversion rather than re-deriving them per request.



- Documentation now consistently ranks construction modes as registry-first, config-first, then builder convenience across README and migration workstream docs.



- Example navigation docs now follow the same rule, and compatibility demos are explicitly labeled instead of being mixed into the default learning path.



- Runnable examples now avoid `Provider::*` builder construction by default; builder usage is confined to explicitly labeled compatibility demos.

- Runnable rerank examples now also exist for the stable registry-first path plus config-first Cohere / TogetherAI / Bedrock provider paths, so the rerank family story is no longer doc-only.



- Provider feature alignment work is tracked in `provider-feature-matrix.md`.
  - The matrix now also distinguishes “not started” from “intentionally unsupported but already contract-locked”: audited fail-fast Stable-family boundaries count as aligned, which closes the earlier under-reporting on embedding / image / rerank rows for text-led native providers.
  - The same matrix cleanup now also reaches `Audio split`: native providers that intentionally keep audio inside text/chat or speech-only provider-owned surfaces now have explicit contract coverage instead of showing up as ambiguous `[ ]` gaps.
  - `Rerank` is now fully closed in the matrix as well: OpenAI native rerank is treated as an explicit negative boundary on the canonical OpenAI base URL, `registry.reranking_model("openai:...")` now rejects construction instead of returning a dead-end handle, Gemini now has matching native rejection coverage, and the positive rerank story remains owned by focused provider packages plus the enrolled OpenAI-compatible presets.
  - The same root-cause cleanup now also applies to unsupported native embedding / image / speech / transcription family entries: registry handle construction fails immediately, so text-led providers no longer expose dead-end family handles outside the capability set they actually advertise.
  - Focused OpenAI-compatible non-chat presets now also pin the generic registry text boundary explicitly: Jina and VoyageAI reject `registry.language_model("provider:model")` construction instead of widening into dead-end chat handles, while Infini remains the positive mixed-surface reference that still supports registry text handles alongside embeddings.
  - The generic registry text guard is now provider-agnostic instead of compat-specific: rerank-only focused packages like Cohere and TogetherAI also reject `registry.language_model("provider:model")` construction, so the public registry no longer widens non-chat providers into dead-end text handles just because their factory still carries a compatibility `language_model_with_ctx` path internally.



- Tool calling groundwork: `ToolChoice` wire-format mapping is now covered by no-network tests across OpenAI, Anthropic, and Gemini protocol crates.



- Tool calling design notes and acceptance criteria are tracked in `tools-parity.md`.



- Structured output mapping notes and acceptance criteria are tracked in `structured-output-parity.md`.
- Stable structured-output extraction failure semantics are now explicit in that note and in code: explicit responses / `StreamEnd` remain best-effort JSON repair paths, interrupted streams without `StreamEnd` now require a complete JSON payload, and `FinishReason::ContentFilter` now yields a dedicated parse error when no valid JSON was produced.
- Interrupted reserved-`json` tool streams now follow that same stable contract too: if the stream ends before tool arguments form a complete JSON value, extraction returns the dedicated incomplete-stream error instead of a generic parser-specific failure.
- Typed structured-output extraction now also separates "JSON missing/invalid" from "JSON present but target type mismatch" at the stable error-message boundary, and the public `siumai::structured_output` helpers now delegate that logic directly to `siumai-core` instead of maintaining a second deserialization path.
- Public `siumai` integration coverage now also locks that facade delegation: typed response extraction preserves the target-type mismatch wording, and interrupted streams without `StreamEnd` keep the strict "complete JSON required" contract.
- Public API cleanup now also has an explicit compatibility boundary in code comments, crate docs, compile guards, and the written API story: `siumai::compat` is documented as a time-bounded migration surface (no earlier than `0.12.0`), while new capabilities must land on family APIs, config-first clients, or `provider_ext` before any compatibility mirroring.
- README and examples index guidance now also lock the primary entry-point story more concretely: the six family modules `text`, `embedding`, `image`, `rerank`, `speech`, and `transcription` are named explicitly as the default public surface for new code, instead of leaving rerank/speech/transcription implied behind provider-specific docs.
- Prelude cleanup now also reflects that same boundary in code: `prelude::unified` is documented as the stable-family-centered surface, `Siumai` / `Provider` compatibility aliases there are doc-hidden, and a dedicated `prelude::compat` import path exists for migration-oriented code without advertising compat builders as part of the default unified story.
- OpenAI request-side parity is now also closed for this pass: top-level siumai/provider/config-first construction now has no-network guards for typed reasoning options on both Responses and Chat Completions routes, locking Stable `response_format` / `tool_choice` together with provider-owned `OpenAiOptions` reasoning settings on the final request body.
- OpenAI now also closes the shared-builder default-options gap on the Responses text path: `OpenAiBuilder`, `OpenAiConfig`, and `Siumai::builder().openai()` all accept typed `with_openai_options(...)` defaults, sparse serialization keeps chained defaults from writing null/empty placeholders, duplicate `responsesApi` / `responses_api` aliases are merged canonically inside `OpenAiSpec`, and focused no-network public-path parity now pins both request and stream `/responses` bodies for default `reasoning_effort`, `previous_response_id`, and `reasoning_summary` shaping across those three construction paths.
- Builder/config convergence is now also explicit at the public test layer: major provider builders preserve common inherited HTTP settings on `into_config()`, and the focused/native packages that do not participate in the cross-provider guard already have provider-local `build()` ↔ `into_config()` ↔ `from_config()` convergence tests, so builder surfaces are now treated as thin wrappers over config-first construction rather than parallel architecture.
- Registry handles now also have an explicit public trait-bound guard on top of the earlier execution-path work: `LanguageModelHandle` and `EmbeddingModelHandle` compile as final family model objects plus `ModelMetadata`, so the registry surface no longer depends on documentation alone to communicate that these handles belong on the stable family APIs.
- Family-model execution is now also locked across the full stable public surface: compile-only guards cover all six registry handle families, and the top-level `text` / `embedding` / `image` / `rerank` / `speech` / `transcription` helper modules are pinned as callable against the final metadata-bearing family traits instead of a legacy generic-client mental model.
- Major provider package migration is now also treated as closed for this pass: OpenAI, Anthropic, Gemini, Vertex, and Anthropic-on-Vertex have explicit public compile/parity coverage for both top-level `Provider::*` entry points and provider-owned config/client stories, so the remaining work on those providers is depth completion under the chosen package taxonomy rather than another public-surface reshuffle.
- Public-story cleanup is now also closed for this pass: README, examples guidance, and workstream notes use the same `registry-first -> config-first -> builder convenience` ladder, OpenAI-compatible vendor views are described as typed layers over the shared compat runtime, and builder-based compat paths are labeled consistently as migration/convenience surfaces rather than the architectural default.
- Secondary-provider example navigation is now tighter too: the provider-specific indexes explicitly route readers through provider-owned config-first examples (`deepseek/reasoning.rs`, `groq/structured-output.rs`, `xai/web-search.rs`, `ollama/metadata.rs`) before compat vendor views (`openrouter-transforms.rs`, `perplexity-search.rs`) and the lone builder demo, so this part of the cleanup no longer depends on readers inferring package tiers from directory names alone.
- Secondary-provider audio depth is now a bit stronger as well: xAI's provider-owned TTS path now has a runnable config-first `xai-tts` example, the public-surface compile guard also pins `XaiConfig`, `Provider::xai()`, and the provider-owned TTS request shape, so the remaining audio-split cleanup is less about surfacing xAI speech and more about finishing provider-by-provider coverage elsewhere.
- Compat audio split is now also guarded at the top-level public client boundary: Together and SiliconFlow built through the real `Provider::openai().<preset>()` paths expose both speech and transcription capabilities, while Fireworks exposes transcription without speech, matching the documented preset policy instead of relying only on lower-level compat-runtime tests.
- Compat audio now also has runnable shared-runtime examples: `siliconflow-speech` / `siliconflow-transcription` and `together-speech` / `together-transcription` anchor the two full-audio preset lanes, while `fireworks-transcription` anchors the transcription-only lane on its dedicated audio host.
- Compat image generation is now also less implicit: the top-level preset guard locks `image_generation` on real SiliconFlow/Together built clients and locks its absence on Fireworks, while `siliconflow-image` and `together-image` now give the shared compat image story runnable config-first examples instead of leaving Together covered only by request-capture parity tests.
- Compat non-text registry parity is now deeper too: OpenRouter embedding already matches config-first construction on the final registry `/embeddings` boundary, Together/SiliconFlow image handles now match config-first `/images/generations` request bodies, and Jina/VoyageAI rerank handles now match config-first `/rerank` request bodies as well, so the public registry surface no longer trails the lower compat factory contract on the currently advertised non-text presets.
- Compat rerank is now also easier to discover from the example surface itself: `siliconflow-rerank`, `jina-rerank`, and `voyageai-rerank` now give the shared compat runtime three runnable config-first rerank examples instead of leaving that story implicit behind tests plus the generic registry-first rerank sample.
- Compat rerank capability boundaries are now explicit too: the generic OpenAI-compatible factory fail-fast guards `rerank`, Jina and VoyageAI keep native rerank family paths, and OpenRouter is locked to an unsupported rerank path instead of inheriting that family accidentally from the generic text entry.
- Compat focused-preset text boundaries are now explicit as well: Jina and VoyageAI no longer advertise default chat/streaming support just because they share the OpenAI-compatible runtime, adapter/factory/client paths now also reject the generic `language_model_with_ctx(...)` text fallback for those presets, and Infini remains a positive mixed-surface reference because its declared `tools` / `vision` capabilities still imply a real chat path.
- Registry text handles now align with that same contract: the compat `language_model("jina:...")` compatibility entry no longer masquerades as a chat capability, and direct handle-level chat calls fail fast before the underlying compat client is even constructed.
- Observability example placement is now also closed for this pass: runnable middleware and telemetry examples are explicitly documented on registry-first or config-first paths (`custom-middleware`, `basic-telemetry`, `http-interceptor`), while `middleware_builder.rs` is clearly scoped as a middleware-composition utility instead of a default `Siumai` construction pattern.
- xAI structured-output public depth is now one step stronger: `xai-structured-output` gives the provider-owned wrapper path a runnable config-first schema example, and the top-level parity suite now confirms that builder/provider/config-first xAI construction all converge on the same Stable `response_format` body while still applying xAI-specific request normalization rules.
- Ollama structured-output public depth is now one step stronger too: the provider-owned local-runtime path now has a runnable config-first `ollama-structured-output` example, matching the earlier top-level request-shaping parity that already locked Stable schema output onto Ollama's `format` field across builder/provider/config-first construction.
- The canonical advanced structured-output example now also matches the Stable API story again: `03-advanced-features/provider-params/structured-output.rs` uses `ChatRequest::with_response_format(...)` plus typed extraction helpers directly, so the public learning path no longer teaches provider-specific OpenAI request options as the default schema-output pattern.
- Public structured-output facade coverage is now also less implicit: top-level integration tests now pin both the dedicated `content filtering/refusal` parse error and the best-effort repair path when structured JSON arrives on `StreamEnd`, so those stable failure semantics are no longer guarded only at the `siumai-core` unit-test layer.
- xAI reasoning public depth is now one step stronger too: `xai-reasoning` gives the provider-owned wrapper path a runnable config-first reasoning example, and the top-level parity suite now confirms that builder/provider/config-first/registry xAI construction all converge on the same non-stream `ChatResponse::reasoning()` semantics plus streaming `ThinkingDelta` accumulation while keeping `enable_reasoning` / `reasoning_budget` defaults aligned and leaving request-level `reasoningEffort` as a typed xAI extension knob.
- Compat embedding public depth is now one step stronger too: `openrouter-embedding` gives the shared OpenAI-compatible runtime a runnable vendor-view example for Stable `embedding::embed` with request-level options, so OpenRouter embedding usage is no longer discoverable only from public-path parity tests and matrix notes.
- The provider-owned typed-surface audit is now tighter too: `azure` and plain `google_vertex` have both moved out of the remaining-gap bucket. The Vertex public facade now also exposes provider-owned typed metadata under `provider_ext::google_vertex::metadata`, focused no-network guards now lock response-level `usageMetadata` / `safetyRatings`, `groundingMetadata` / `urlContextMetadata`, normalized `sources`, and part-level `thoughtSignature` extraction across the real public builder/provider/config-first/registry story, and the namespace contract stays deliberately strict (`vertex` only, no `google` fallback) while streaming reasoning custom events remain raw. `cohere` and `togetherai` still look more like explicit low-priority rerank-only boundaries than urgent typed-surface omissions.
- OpenAI-compatible non-text factory closure is now deeper too: the generic compat factory explicitly advertises and guards `embedding` / `image_generation` family paths, source-level contracts now pin the native image-family override, Together builder/config/registry image requests converge on the same `/images/generations` transport boundary, and OpenRouter is locked to a fail-fast no-image path instead of inheriting image support accidentally.
- Compat vendor-view capability boundaries are now explicit too: OpenRouter keeps `embedding` as its only public non-text family while top-level builder/provider/config-first plus registry speech/transcription paths now fail fast before transport, and Perplexity is now pinned as a text-only vendor view with `embedding` / `image_generation` / `rerank` / speech / transcription all rejected at factory and public layers instead of leaking through the generic compat runtime.



- Anthropic reserved `json` tool fallback now has no-network response/streaming parity coverage, locking the Stable JSON-text behavior across both paths.
- Anthropic fixture coverage now also leans on the provider-owned typed metadata escape hatch where that surface already exists: container-metadata checks for message responses and streaming `StreamEnd` now read `AnthropicMetadata.container` instead of duplicating raw `provider_metadata["anthropic"]` traversal.
- Anthropic thinking replay metadata now also follows the same rule: `thinking_signature` and `redacted_thinking_data` are first-class fields on `AnthropicMetadata`, protocol tests assert them through `AnthropicChatResponseExt`, and the provider-owned replay helper no longer needs to dig through raw provider maps by hand.
- Anthropic response-side metadata typing now also reaches one level deeper into unified content parts: `AnthropicContentPartExt::anthropic_tool_call_metadata()` exposes programmatic tool-call `caller` data on `ContentPart::ToolCall`, the corresponding fixture regression uses the typed helper, and protocol streaming tests now read `contextManagement` / `sources` from `AnthropicMetadata` on `StreamEnd` instead of reopening raw provider maps.
- Anthropic now also has top-level public-path parity coverage for provider-owned request helpers: builder/provider/config-first clients converge on the same final request body for typed thinking, Stable `response_format`, and Stable `tool_choice`, public-surface import guards now pin `AnthropicChatRequestExt` together with the typed option structs, and the duplicated `chat_before_send` application bug in Anthropic-style HTTP chat clients has been removed so additive request rewrites no longer run twice. The matching openai-compatible regression guard now also locks the corrected executor boundary: runtime provider rewrites remain on `provider_spec.chat_before_send(...)` rather than being duplicated into executor policy state.
- Anthropic default-option parity now also closes the remaining builder/provider/config gap on the streaming path: provider-owned default `AnthropicOptions` no longer overwrite each other via serialized `null` placeholders, runtime transformer selection on `HttpChatExecutor::execute_stream(...)` now happens after model middleware transforms, and no-network parity now locks default `context_management`, `output_config.effort`, and `tool_streaming = false` together with the already-covered thinking/json-tool fallback story on the final `/v1/messages` request body.
- Anthropic parity now also covers both top-level and registry-native provider-owned typed container metadata on 200-response and `StreamEnd`, so builder/provider/config-first plus `language_model("anthropic:...")` paths stay aligned on `AnthropicMetadata.container` while `service_tier` remains explicitly modeled on the top-level `ChatResponse` instead of being widened into vendor metadata.
- Anthropic stream metadata parity also caught and fixed a real config-first execution gap: `AnthropicClient::chat_stream_request(...)` now injects `stream=true` before dispatch, so plain `ChatRequest` values no longer produce divergent request bodies between config-first and registry-native streaming paths.
- Anthropic and MiniMaxi source-boundary audit is now also closed on the current typed facade: `AnthropicMetadata.sources` / `MinimaxiMetadata.sources` already expose typed source entries directly, public-surface guards now exercise those typed `source_type` / `filename` fields, and there is no need to invent an extra provider-owned `SourceExt` helper while the shared Anthropic-shaped struct still covers the stable contract.
- The xAI / Groq / DeepSeek response-boundary audit is now also closed for this pass: xAI non-stream provider-hosted tool calls already use generic `ContentPart::ToolCall` with `provider_executed`, xAI hosted-search metadata is already typed through `XaiMetadata.sources` plus `XaiSourceExt` while streaming search details remain explicit custom events, Groq's stable typed response surface remains `logprobs` plus `sources`, and DeepSeek already promotes `reasoning_content` into unified `ContentPart::Reasoning`, so no new provider-owned `ContentPart` helper was introduced.
- OpenAI and Azure top-level public paths are now tighter too: `Provider` now exposes explicit `openai_responses`, `openai_chat`, `azure`, and `azure_chat` constructors, compile guards pin those entry points, and no-network parity coverage now confirms builder/provider/config convergence for both Responses and Chat Completions routes on each provider family.
- Azure request-side capability parity is now also closed for this pass: `AzureOpenAiSpec` now owns `chat_before_send` request shaping for reasoning knobs plus Stable `response_format` / `tool_choice` semantics on both Responses and Chat Completions routes, and the duplicate client-side before-send injection has been removed so those additive rewrites now execute exactly once at the final transport boundary.
- Google / Gemini request-side capability parity is now also closed for this pass: no-network public-path tests now lock builder/provider/config convergence for Gemini chat plus embedding routes, public-surface import guards pin `GeminiChatRequestExt` alongside the typed thinking options, and Stable `response_format` / `tool_choice` now converge with provider-owned `thinkingConfig` on the final `generateContent` request body.
- Bedrock request-side capability parity is now also closed for this pass: no-network public-path tests now lock builder/provider/config convergence for Stable `response_format` / `tool_choice` on the provider-owned Converse route, including reserved `json` tool injection plus final `toolConfig` / `additionalModelRequestFields` request-body convergence at the transport boundary.
- Google Vertex request-side capability parity is now also closed for this pass: no-network public-path tests now lock builder/provider/config convergence for Stable `response_format` / `tool_choice` plus provider-owned `thinkingConfig` on the final Vertex `generateContent` request body, and the root-cause wrapper gap is fixed because `GoogleVertexClient` now overrides the default `chat_request` / `chat_stream_request` fallback that previously discarded request-level fields by collapsing back to `messages + tools`.
- Anthropic-on-Vertex default-options coverage is now tighter too: provider-owned `VertexAnthropicOptions` explicitly serialize sparsely, and the same default thinking / reserved-`json` fallback / `disable_parallel_tool_use` / `send_reasoning = false` shaping is now pinned across builder/provider/config on both `:rawPredict` and `:streamRawPredict` instead of only the non-stream request path.
- Cohere and TogetherAI public-path parity are now also explicitly closed in the capability matrix: both native packages remain rerank-led by design, and their no-network builder/provider/config-first rerank request anchors plus public-surface compile guards confirm there is no hidden top-level parity gap left to chase in this pass.
- Perplexity hosted-search metadata is now also explicitly closed in the capability matrix: the stable typed surface already covers `citations`, `images`, and `usage.{citation_tokens,num_search_queries,reasoning_tokens}`, public-surface imports expose that boundary directly, provider-local runtime tests pin the shared compat extraction, and both top-level builder/provider/config-first plus registry `language_model("perplexity:...")` guards now verify 200-response and `StreamEnd` typed extraction plus normalized `provider_metadata["perplexity"]` roots without inventing a wider vendor response contract.
- xAI hosted-search metadata is now also explicitly closed in the capability matrix: the stable facade already exposes typed `sources` through `XaiMetadata`, source-level provider fields through `XaiSourceExt`, and both top-level plus registry non-streaming / `StreamEnd` runtime tests pin that boundary without widening event-level search payloads into a fake stable contract.
- Ollama public-path parity is now also explicitly closed in the capability matrix: top-level and registry-native no-network tests already anchor builder/provider/config plus `language_model("ollama:...")` convergence for chat/chat-stream metadata and embedding request shaping, and the public-surface compile guard now pins `OllamaChatRequestExt`, `OllamaEmbeddingRequestExt`, and `OllamaChatResponseExt` on the stable facade while keeping the response-side contract limited to provider-owned timing metadata.
- Ollama capability-split closure is now explicit in the same matrix: embeddings stay public, while top-level wrapper paths and registry-native `image` / `rerank` handles now all share the same fail-fast unsupported-family contract without emitting requests.
- Anthropic capability-split closure is now explicit too: typed text metadata/request parity remains public, but non-text `embedding` / `image_generation` / `reranking` entry points are now locked to absent capabilities plus fail-fast unsupported requests across both wrapper and registry-native construction paths.
- Anthropic / Ollama / MiniMaxi capability-split closure now also reaches the lower factory/handle contract: deferred family builders reject explicitly with `UnsupportedOperation`, and registry-level family handles now pin the same no-request failure path so generic factory fallbacks cannot silently reintroduce those capabilities.
- DeepSeek now also reaches that same lower factory/handle closure: deferred `embedding` / `image` / `rerank` / audio-family builders reject explicitly with `UnsupportedOperation`, and both registry-contract plus public registry-handle tests pin the same no-request failure path below the text-family wrapper story.
- MiniMaxi now also closes Stable `tool_choice` on its provider-owned Anthropic-style chat path: dedicated request fixtures lock the MiniMaxi-local `required -> { type: "any" }` mapping, `MinimaxiClient` now overrides the default full-request fallback so `chat_request` / `chat_stream_request` preserve request-level fields instead of collapsing back to `messages + tools`, and no-network public-path parity now confirms builder/provider/config convergence on that same final `tool_choice` body shape instead of relying only on inherited Anthropic protocol coverage.
- OpenAI/Azure streaming alignment now also has an explicit stable-vs-event boundary: finish/reasoning/text-start custom events keep raw `providerMetadata` assertions because they validate event payload shape and provider-key rewriting, while stable `StreamEnd` checks use typed response metadata only where that surface is actually promoted today; OpenAI top-level builder/provider/config-first parity now verifies typed `OpenAiSourceExt` extraction on final Responses `response.completed` payloads and typed `OpenAiChatResponseExt.logprobs` on Chat Completions `StreamEnd` rather than stopping at protocol-only source-annotation tests.
- OpenAI non-stream response parts now also follow the same narrow-typed pattern: `OpenAiContentPartExt` exposes `itemId` and `reasoningEncryptedContent` on unified `ContentPart`s, with regression coverage against a real Responses reasoning item, while event-level SSE payload assertions remain raw and namespace-focused.
- OpenAI source annotations now follow the same rule as other stable response surfaces: `OpenAiSourceExt`/`OpenAiSourceMetadata` expose `fileId` / `containerId` / `index` for unified `OpenAiSource` values, transformer regressions pin real `file_citation`, `container_file_citation`, and `file_path` annotation mapping without reopening raw provider metadata maps, top-level builder/provider/config-first parity now locks that typed source extraction on real Responses 200-response payloads, and the shared compat metadata helper now also carries Chat Completions `logprobs` onto the same top-level public path for both 200-response and `StreamEnd`.
- OpenAI now also closes the remaining registry-side metadata gap on those same native text lanes: public-path no-network coverage pins `registry.language_model("openai-chat:...")` against config-first Chat Completions for typed `OpenAiChatResponseExt.logprobs` on both 200-response and `StreamEnd`, and pins `registry.language_model("openai:...")` against config-first Responses for typed `OpenAiSourceExt` extraction on both 200-response and `StreamEnd`, so metadata parity no longer stops at builder/provider/config-first construction.
- The migration safety-net story is tighter at the provider-owned layer too: `siumai-provider-openai` now has focused no-network regressions proving request-based `chat_stream_request(...)` always forces `stream=true` plus `stream_options.include_usage=true` on both Chat Completions and Responses routes, and it now also pins the Responses reasoning stream itself so provider-owned streaming clients emit stable `ThinkingDelta` while preserving final `ContentPart::Reasoning` metadata (`itemId` / `reasoningEncryptedContent`) on `StreamEnd`; `siumai-provider-openai-compatible` now has focused unit regressions proving shared-runtime `prepare_chat_request(..., stream)` flips the stream lane, backfills config-level model/http defaults, and preserves explicit request models on the non-stream lane before executor selection; `siumai-provider-amazon-bedrock` plus `siumai-provider-azure` now have focused unit regressions proving `prepare_chat_request(..., stream)` flips the stream lane, backfills the configured default model/deployment, and preserves explicit request models on the non-stream lane before executor selection; `siumai-provider-gemini` has focused unit regressions proving `prepare_chat_request(..., stream)` flips the stream lane and fills client defaults before executor selection, while `siumai-provider-google-vertex` now also has a focused no-network client regression proving request-based `chat_stream_request(...)` reaches the real `:streamGenerateContent?alt=sse` boundary with `Accept: text/event-stream`, Stable `response_format` / `tool_choice`, and provider-owned `thinkingConfig` / `structuredOutputs`; `siumai-provider-google-vertex`'s `anthropic-vertex` wrapper now also has focused request-based regressions proving `chat_request/chat_stream_request` no longer collapse back to the trait default message/tool path and that explicit request models reach the real `:rawPredict` / `:streamRawPredict` URL boundary; and `siumai-provider-ollama` now has focused no-network client regressions proving request-based `chat_stream_request(...)` always forces `stream=true` on its native NDJSON `/api/chat` route while keeping Stable `response_format -> format` precedence intact, plus fail-fast coverage for unsupported `ToolChoice::Required` before transport, closing the root-cause gaps that previously let direct config-first stream calls drift from registry/public stream request shaping. The matching registry-side root cause is now pinned too: `LanguageModelHandle` only backfills `request.model` when it is absent, and explicit full-request model overrides are now covered end-to-end on config-first plus registry text-handle paths for `anthropic-vertex`, `azure`, `deepseek`, `groq`, and `xai`; `groq` / `xai` now serve as the OpenAI-compatible representative parity anchors instead of leaving that lane covered by a single sample. OpenRouter and Perplexity now also anchor that compat-vendor-view request boundary directly, proving explicit request-model overrides survive while vendor options still merge onto the final body on both public-path and lower-contract coverage, provider-local stream regressions now pin `Accept: text/event-stream`, Stable `response_format` / `tool_choice`, plus OpenRouter reasoning defaults or typed Perplexity search options on the shared-runtime transport boundary itself, and representative non-text runtime regressions now also pin Fireworks embedding on its inference host, Infini embedding on `/embeddings`, Together embedding/image on `/embeddings` plus `/images/generations`, and SiliconFlow/Jina/VoyageAI rerank plus SiliconFlow image on `/rerank` or `/images/generations`, so shared compat embedding/image/rerank executors are covered below the public-path layer too.
- OpenAI native full-request chat defaulting is now closed at the root as well: provider-local `prepare_chat_request(...)` now merges sparse request `CommonParams` with config/builder defaults before executor selection on both non-stream and stream direct request paths, focused unit tests pin missing-vs-explicit `model` / `temperature` / `max_tokens` behavior on that path, and a targeted proxy-backed real smoke against `gpt-5.2` confirms both explicit-request-model and builder-default-model `chat_request(...)` flows now return `OK` instead of sending an empty model to the transport layer.
- OpenAI Responses SSE source annotations now have matching regression coverage in both directions: converter tests pin `container_file_citation` / `file_path` into `openai:source` custom events, and serializer tests pin the inverse mapping back to annotation payloads while keeping those event-level contracts raw and namespace-scoped.
- OpenAI Responses SSE finish aggregation now keeps source annotations all the way into the final `response.completed` response body: serializer state carries message annotations until completion, and round-trip regression coverage confirms `container_file_citation` / `file_path` metadata survives the completed-response transformer as typed `OpenAiSourceMetadata`.
- V4 now also has an explicit typed metadata boundary matrix in `typed-metadata-boundary-matrix.md`, making the current provider-by-provider contract visible in one place: which providers own typed `ChatResponse`, `ContentPart`, and `Source` escape hatches, and which streaming/event paths intentionally remain raw.
- The next response-side typing backlog is now also tied directly to that matrix instead of ad hoc follow-up decisions: compat-family content-part helpers plus the explicit non-goals for Azure/`google-vertex` remain documented boundary work, while OpenRouter's current alias-based vendor metadata surface and the Perplexity / Anthropic / MiniMaxi source audits are now considered closed on the current stable surface.
- V4 now also has an explicit provider capability alignment matrix in `provider-capability-alignment-matrix.md`, so the next phase can move from metadata cleanup into provider capability parity with a documented priority order instead of opportunistic gap filling.
- Google / Gemini now also has a no-network top-level public-path guard for its mixed typed/raw metadata boundary: builder/provider/config-first parity verifies typed `GeminiContentPartExt` / `GeminiChatResponseExt` extraction on 200-response and `StreamEnd`, while streaming `reasoning-start` payloads keep raw `providerMetadata.google.thoughtSignature` plus normalized `provider_metadata["google"]` roots; the root-cause stream path gap is fixed because `GeminiClient` now normalizes `request.stream` before executor construction.
- Azure now also has a no-network top-level public-path guard for the Responses streaming raw/event boundary: builder/provider/config-first `chat_stream_request` parity verifies `text-start` / `finish` custom events plus final `provider_metadata["azure"]` roots without promoting any typed Azure response metadata surface.
- That Azure boundary decision is now also reflected in the capability matrix itself: Azure hosted-search metadata is treated as `Deferred` rather than `Partial`, because the current audited contract is the raw namespace-scoped stream/event metadata path, not an unfinished typed metadata promotion.
- Azure registry handles now also lock that same raw metadata boundary on the real Responses request and SSE lanes: public-path and lower-contract no-network coverage both pin `registry.language_model("azure:...").chat_request(...)` and `registry.language_model("azure:...").chat_stream_request(...)` against config-first construction, so override-scoped routing preserves `provider_metadata["azure"]` on 200-response roots, custom events, and final `StreamEnd` roots without widening Azure into typed OpenAI metadata semantics.
- Google Vertex now also has a no-network top-level public-path guard for its Gemini-style streaming raw/event boundary: builder/provider/config-first `chat_stream_request` parity verifies `reasoning-start` / `reasoning-delta` custom events plus final `provider_metadata["vertex"]` roots, while intentionally keeping that namespace on the raw/event side instead of creating a provider-owned typed metadata facade.



- Ollama now maps Stable `response_format` to `/api/chat` `format`, and request-level schema hints override `providerOptions["ollama"].format`.



- Ollama now also maps Stable `ToolChoice::None` by omitting tools from the final request body, and its tool loop invariants are covered by non-streaming/streaming parity tests.



- Groq now has provider-level smoke tests confirming OpenAI-style `response_format` passthrough and tool-call response mapping under its adapter path.



- xAI now has runtime-provider smoke tests confirming OpenAI-compatible `response_format` and `tool_choice` semantics under provider-id keyed routing.



- DeepSeek now also has runtime-provider smoke tests confirming OpenAI-compatible `response_format`, `tool_choice`, and tool-call response semantics under provider-id keyed routing.



- OpenAI-compatible tool-loop invariants now also have explicit streaming/non-streaming parity coverage for the xAI and DeepSeek runtime-provider paths; Groq inherits the same compat invariant while preserving tool-call response mapping in adapter smoke tests.



- xAI, Groq, DeepSeek, OpenRouter, and Perplexity now also have runnable provider-specific examples on the preferred public paths: `xai-web-search`, `groq-structured-output`, `groq-logprobs`, `deepseek-reasoning`, `openrouter-transforms`, and `perplexity-search`.
- Ollama tool-choice behavior is now less ambiguous too: `ToolChoice::None` continues to omit tools on the native chat route, while `Required` and specific-tool forcing now fail fast with `UnsupportedOperation` across builder/provider/config-first and registry construction instead of silently collapsing back to `Auto`.
- Structured-output streaming now also closes the synthetic-end gap on public paths: the shared extractor reuses accumulated deltas / reserved-tool payloads when providers emit `StreamEnd` with `FinishReason::Unknown`, and Ollama top-level builder/provider/config/registry parity now pins both the successful complete-JSON case and the interrupted incomplete-stream failure case on its NDJSON chat route.
- That synthetic-end closure now also reaches the shared OpenAI-compatible SSE path: xAI top-level builder/provider/config/registry parity pins the same complete-vs-truncated structured-output split when the stream ends without a terminal provider finish frame, so the contract is shared across both native NDJSON and compat SSE streaming providers.
- The same compat-SSE closure now also reaches vendor-view public shortcuts: Perplexity top-level builder/provider/config/registry parity pins the same complete-vs-truncated structured-output split on the shared OpenAI-compatible SSE path, proving the contract holds even when the public story stays vendor-view rather than provider-owned.
- OpenRouter now also joins that same vendor-view compat-SSE closure: top-level builder/provider/config/registry parity pins the same complete-vs-truncated structured-output split on the shared OpenAI-compatible SSE path while still preserving OpenRouter-specific request shaping.
- Bedrock now also closes the reserved-tool streaming half of that story on a provider-owned path: top-level builder/provider/config parity pins the same complete-vs-truncated structured-output split when the Converse JSON stream ends after reserved `json` tool deltas but before a terminal `messageStop`.
- Anthropic now also closes the provider-owned SSE fallback half of that story: forcing `providerOptions.anthropic.structuredOutputMode = "jsonTool"` pins the same complete-vs-truncated split across builder/provider/config/registry when reserved `json` tool deltas arrive before transport close but the stream never reaches `message_stop`.
- Anthropic now also has the matching provider-local regression below that facade layer: `AnthropicClient::chat_stream_request(...)` is exercised directly against interrupted SSE `jsonTool` payloads, locking both complete reserved-tool JSON extraction and the dedicated incomplete-stream parse error on the real provider-owned client path while the captured request body stays on the fallback reserved-tool shape instead of drifting back to native `output_format`.
- Anthropic now also has a provider-owned typed escape hatch for that route: `AnthropicStructuredOutputMode` is exported through `provider_ext::anthropic` and wired into `AnthropicOptions::with_structured_output_mode(...)`, removing the need for raw JSON provider-option payloads when callers want to force `jsonTool`.
- xAI speech now also has a provider-owned typed escape hatch on `/v1/tts`: `XaiTtsOptions` is exported through `provider_ext::xai` and applied via `XaiTtsRequestExt`, so `sample_rate` / `bit_rate` no longer need raw JSON provider-option payloads on examples or public-path parity tests.







Status: in progress







## V4-M4 - OpenAI path migrated







Acceptance criteria:







- OpenAI implements the new family model contracts directly.



- OpenAI registry construction works through the new factory path.



- Existing fixture/alignment behavior remains equivalent.



- Public OpenAI builder and config constructors both use the same internal path.







Notes:







- OpenAI native text-family factory path exists and is covered by a contract test.



- OpenAI native image-family factory path now also exists and is covered by a contract test.
- OpenAI native speech/transcription family factory paths now also exist and are covered by contract tests.
- Azure OpenAI registry construction now also has native text/embedding/image/speech/transcription family paths backed by the provider-owned `AzureOpenAiClient`, with contract coverage for metadata-bearing family construction.
- Azure OpenAI now also has a provider-owned builder/config/client story instead of config-first only construction: `AzureOpenAiBuilder` emits `AzureOpenAiConfig`, `provider_ext::azure` exports that builder on the public surface, package-local `into_config()` tests guard the new path, the registry factory now routes through that same builder/config boundary instead of hand-assembling config/client pairs, and focused registry contract tests now lock builder/config/registry request convergence for both Responses and Chat Completions routing.



- This milestone is only partially complete because the migration has not yet expanded across all remaining OpenAI family surfaces.







Status: in progress







## V4-M5 - Anthropic and Gemini migrated







Acceptance criteria:







- Anthropic implements the new family model contracts directly.



- Gemini implements the new family model contracts directly.



- Major streaming/tool/multimodal flows still pass alignment tests.







Notes:







- Anthropic native text-family factory path exists and is covered by a contract test.



- Gemini native text-family factory path now exists and is covered by a contract test.



- Gemini native embedding-family and image-family paths now also exist and are covered by contract tests.







Status: in progress







## V4-M6 - OpenAI-compatible and secondary providers migrated







Acceptance criteria:







- OpenAI-compatible base is migrated.



- Secondary providers are ported or adapted onto the new family model contracts.



- No new provider relies on `LlmClient` as its primary public execution abstraction.







Notes:







- OpenAI-compatible base now has a native text-family factory path and contract coverage.



- OpenAI-compatible base now also has a native embedding-family factory path and contract coverage.



- OpenAI-compatible base now also has an audio-capable client/spec foundation for speech/transcription family wiring, plus no-network coverage for capability exposure and executor wiring.



- Built-in OpenAI-compatible vendor presets now claim speech/transcription support for `Together` and `SiliconFlow`, where the OpenAI-style `/audio/speech` and `/audio/transcriptions` surfaces have been verified and registry contract coverage now exists.



- The rest of the OpenAI-compatible vendor catalog still does not claim speech/transcription support until each endpoint contract is validated independently.



- `OpenRouter` remains outside the shared compat audio family for now because its documented audio path is still tied to `chat/completions` multimodal semantics rather than standalone OpenAI `/audio/*` endpoints.



- `xAI` remains outside the shared compat audio family for now because its documented TTS path uses `/v1/tts`, and its current voice docs do not expose a matching standalone STT surface for the shared compat adapter.



- xAI now also has a provider-owned speech-family path over `/v1/tts`, registry-native factory routing, and contract coverage; only shared-compat enrollment and transcription remain intentionally deferred.



- `Fireworks` now participates in the shared compat audio family as a transcription-only provider: its documented `/audio/transcriptions` route is wired through the dedicated `https://audio.fireworks.ai/v1` base, while speech/TTS remains intentionally unsupported.



- Groq and xAI now also have native text-family factory paths with contract coverage.



- Ollama now also has a registry-native text-family factory path with contract coverage, while keeping its provider-owned JSON-streaming runtime and config-first wrapper surface.
- Ollama now also has a matching registry-native embedding-family factory path with contract coverage, builder/config/factory embedding requests now converge on the same final `/api/embed` body at the provider-factory boundary, and the top-level public parity suite now includes the `registry.embedding_model("ollama:...")` handle on that same endpoint instead of relying only on the generic embedding bridge.



- DeepSeek now also has a registry-native text-family factory path, dedicated default-registry routing, and contract coverage while still reusing the shared OpenAI-compatible runtime.



- The DeepSeek feature graph no longer depends on the OpenAI registry feature, and `SiumaiBuilder::new().deepseek()` now works in deepseek-only builds.



- DeepSeek builder availability now follows provider-owned feature gates rather than piggybacking on the generic OpenAI-compatible builder surface.



- DeepSeek now also has a first-class `ProviderType::DeepSeek`, so registry catalogs, retry policy routing, and unified metadata no longer degrade it into `Custom("deepseek")`.



- xAI runtime-provider routing now also has explicit streaming/non-streaming tool-call equivalence coverage, provider-owned `XaiConfig` / `XaiClient` entry types, registry-backed wrapper materialization, and request-option normalization smoke coverage; Groq relies on the same OpenAI-compatible loop invariant with adapter-level response-mapping smoke coverage and now also has wrapper-level no-network coverage for full-request provider-options preservation.



- xAI now also locks typed `XaiOptions` normalization on the request path, and Stable `response_format` now survives runtime-provider option merging even when raw `providerOptions.xai.response_format` is present; xAI-specific post-merge stripping still removes unsupported `stop` / `stream_options` fields.



- xAI now also locks Stable `tool_choice` precedence over raw `providerOptions.xai.tool_choice`, so runtime-provider option merging cannot silently degrade the Stable tool loop contract.
- xAI top-level public-path parity now also covers that `tool_choice` precedence rule across builder / provider / config-first / registry construction on the provider-owned client path, so the Stable-over-raw contract is no longer guarded only inside provider-local smoke tests.

- xAI now also has a provider-owned typed source metadata surface: `XaiSourceExt` / `XaiSourceMetadata` sit on top of the reused compatible `XaiSource` shape but preserve `xai` naming on the public API, and the helper accepts both provider-owned and legacy compatible metadata envelopes so refactors do not leak `openai` naming back into xAI call sites.

- xAI registry text/speech-family construction now also converges through the provider-owned `XaiBuilder` / `XaiConfig` boundary instead of hand-assembled compat config/client wiring, `XaiBuilder::build()` now preserves explicit HTTP client / retry overrides plus appended model middlewares, and focused xAI contract tests now lock both request-shape convergence and source-level factory routing so that path cannot silently fall back to manual assembly.

- Groq now also has a provider-owned typed source metadata surface: `GroqSourceExt` / `GroqSourceMetadata` sit on top of the reused compatible `GroqSource` shape but preserve `groq` naming on the public API, and the helper accepts both provider-owned and legacy compatible metadata envelopes so refactors do not leak `openai` naming back into Groq call sites.
- Groq top-level public-path parity now also covers Stable `tool_choice` precedence over raw `providerOptions.groq.tool_choice` across builder / provider / config-first / registry construction on the provider-owned client path, so the Stable-over-raw contract is no longer guarded only inside provider-spec smoke tests.

- Perplexity now also promotes hosted-search `citations` onto its typed response metadata surface: `PerplexityMetadata.citations` carries returned citation URLs directly, while images and usage stay on the existing typed path and `extra` remains reserved for genuinely unknown vendor fields.
- Perplexity now also promotes `usage.reasoning_tokens` onto its typed metadata surface, so the hosted-search usage story now has first-class fields for `citation_tokens`, `num_search_queries`, and provider-side reasoning cost instead of leaving that stable field behind in `usage.extra`.
- Perplexity hosted-search metadata has now also been re-audited against the current repo fixtures, runtime tests, and provider example paths: no stronger stable response-side field is evidenced beyond `citations`, `images`, `usage.citation_tokens`, `usage.num_search_queries`, and `usage.reasoning_tokens`, so the typed boundary remains intentionally narrow until new provider evidence appears.

- DeepSeek now also has a provider-owned typed source metadata surface: `DeepSeekSourceExt` / `DeepSeekSourceMetadata` sit on top of the reused compatible `DeepSeekSource` shape but preserve `deepseek` naming on the public API, and the helper accepts both provider-owned and legacy compatible metadata envelopes so refactors do not leak `openai` naming back into DeepSeek call sites.

- MiniMaxi now also has a provider-owned typed tool-call content-part surface: `MinimaxiContentPartExt::minimaxi_tool_call_metadata()` reuses the Anthropic-compatible `caller` payload while preserving `minimaxi` naming on the public API, accepts both normalized and legacy provider keys, and intentionally stops short of inventing source-level typed metadata before the underlying Anthropic-shaped surface is actually stabilized.
- MiniMaxi now also owns a provider-named request-side capability surface for chat: `MinimaxiOptions` / `MinimaxiChatRequestExt` expose typed thinking-budget and structured-output helpers under `provider_options["minimaxi"]`, Stable `response_format` now converges on the native `output_format` request body instead of leaking Anthropic's reserved `json` tool fallback, legacy `provider_options["anthropic"]` thinking payloads are still accepted for compatibility, and dedicated MiniMaxi chat-request fixtures plus public-surface import guards lock that contract.



- The generic `OpenAiCompatibleClient` path now also has transport-boundary capture coverage for runtime xAI requests, so final non-streaming and streaming request bodies are guarded after shared compat merge hooks run.



- Core JSON-bytes execution now also honors injected `HttpTransport`, closing the hidden parity gap for provider-owned TTS integrations and preventing accidental live HTTP on no-network speech tests.



- The shared OpenAI-compatible runtime now also normalizes `enableReasoning` / `reasoningBudget` for runtime DeepSeek requests, and the generic `OpenAiCompatibleClient` path has transport-boundary capture coverage for that final request body as well.



- OpenRouter now also has public alignment coverage plus direct `OpenAiCompatibleClient` transport-boundary capture coverage, locking Stable `response_format` / `tool_choice` precedence while OpenRouter-specific vendor params such as `transforms` still merge into final request bodies.



- Perplexity now also has public alignment coverage plus direct `OpenAiCompatibleClient` transport-boundary capture coverage, locking Stable `response_format` / `tool_choice` precedence while generic vendor params still merge into final request bodies on the shared OpenAI-compatible path.



- OpenRouter now also exposes provider-owned typed request options through `OpenRouterOptions` / `OpenRouterChatRequestExt`, so common vendor params such as `transforms` no longer need raw `with_provider_option("openrouter", ..)` calls on preferred public paths.



- Perplexity now also exposes provider-owned typed request options through `PerplexityOptions` / `PerplexityChatRequestExt`, so common hosted-search knobs no longer need raw `with_provider_option("perplexity", ..)` calls.



- Perplexity now also exposes provider-owned typed response metadata through `PerplexityMetadata` / `PerplexityChatResponseExt`, and `siumai::provider_ext::perplexity::metadata` is now the stable typed escape hatch for hosted-search usage/images instead of raw `provider_metadata["perplexity"]` map traversal.



- Perplexity streaming on the generic OpenAI-compatible path now also carries the same typed hosted-search metadata on `StreamEnd`, and no-network SSE coverage locks that parity across request shaping plus end-of-stream extraction.



- V4 now also has an explicit hosted-search boundary decision: OpenRouter / Perplexity search-like semantics remain in `provider_ext::<provider>` for this cycle, and a new Stable unified hosted-search request surface is deferred until more providers converge.



- OpenRouter now also has explicit config-first and builder-to-final-request coverage for shared compat reasoning helpers (`with_reasoning` / `with_reasoning_budget`), confirming those unified ergonomics survive all the way to the final request body on the generic OpenAI-compatible path.
- OpenRouter public-path parity now also locks response-side reasoning extraction on that same compat vendor-view story, so non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` accumulation stay aligned across builder/provider/config-first/registry while `transforms` and reasoning defaults continue to converge on the same final request body.
- OpenAI native Responses public-path reasoning parity now also reaches the real provider-owned story: non-stream `ChatResponse::reasoning()` plus typed reasoning-part metadata (`providerMetadata.openai.{itemId, reasoningEncryptedContent}` on `ContentPart::Reasoning`) are locked across builder/provider/config-first/registry, and the streaming lane now pins `ThinkingDelta` accumulation plus the same `StreamEnd` reasoning metadata across those four construction paths while request-side `reasoning.{effort,summary}` continues to converge on the final `/responses` payload.
- That OpenAI Responses reasoning pass also closed a protocol-level gap: inbound `response.reasoning_summary_text.delta` chunks now emit stable `ThinkingDelta` alongside the existing OpenAI-specific reasoning stream-part events, so direct provider-owned streaming paths surface Stable reasoning text without forcing callers to parse custom events.
- Anthropic public-path reasoning parity now also reaches the real provider-owned story: non-stream `ChatResponse::reasoning()` plus typed `providerMetadata.anthropic.{thinking_signature, redacted_thinking_data}` are locked across builder/provider/config-first/registry, and the streaming lane now pins `ThinkingDelta` accumulation plus the same end-of-stream metadata across those four construction paths.
- That Anthropic reasoning pass also closed a protocol-level gap: `AnthropicEventConverter` now backfills accumulated text/reasoning content into `StreamEnd` instead of returning an empty content shell after normal text/thinking SSE traffic, so direct Anthropic streaming paths no longer depend on a separate stream processor to reconstruct final response content.
- Gemini public-path reasoning depth is now stronger too: the top-level no-network parity suite now locks non-stream `ChatResponse::reasoning()` plus streaming `ThinkingDelta` accumulation across builder/provider/config-first/registry on the real provider-owned Gemini path, while still keeping Google-specific `thoughtSignature` metadata on the explicit typed/raw boundary.
- Google Vertex now also has matching stream-side reasoning parity on its provider-owned path: builder/provider/config-first/registry all accumulate the same `ThinkingDelta` text while the custom reasoning events intentionally remain namespace-led raw `providerMetadata.vertex.thoughtSignature` payloads instead of growing a separate typed reasoning facade.
- Reasoning alignment now also has a dedicated workstream note in `reasoning-alignment.md`, which records the current Stable/request-hint boundary explicitly: DeepSeek, xAI, OpenRouter, and Groq now all have audited public-path response-side semantics there, while Perplexity remains metadata-only on the reasoning axis.

- The generic OpenAI-compatible registry text-family path now also converges through `OpenAiCompatibleBuilder` / `OpenAiCompatibleConfig` for known built-in vendors instead of relying only on helper-side config/client assembly, `OpenAiCompatibleBuilder` now accepts appended model middlewares beyond the auto layer, and focused deepseek-based compat contracts now lock builder/config/registry request convergence plus source-level factory routing on that shared path.



- xAI and Groq now also expose provider-owned typed response metadata through `XaiMetadata` / `XaiChatResponseExt` and `GroqMetadata` / `GroqChatResponseExt`, and `siumai::provider_ext::{xai,groq}::metadata` is now the stable typed escape hatch instead of raw `provider_metadata[provider]` map traversal.



- Groq now also exposes typed request-option builders for `logprobs`, `top_logprobs`, `service_tier`, and reasoning hints, so request-level Groq knobs no longer need raw `with_param(...)` calls for common cases.



- Groq now also locks that Stable `response_format` survives provider-owned before-send merging even when raw `providerOptions.groq.response_format` is present, while typed logprobs and reasoning hints continue to merge around it.



- Groq now also locks Stable `tool_choice` precedence over raw `providerOptions.groq.tool_choice` on the same provider-owned before-send path.



- Groq now also exposes family-model metadata through its provider-owned wrapper client, and registry contract tests now explicitly lock that both `groq` and `xai` public registry paths materialize provider-owned wrapper clients rather than generic OpenAI-compatible aliases.



- Groq custom path-style `base_url` inputs now also normalize trailing slashes consistently across builder, config-first, and registry construction paths.

- Groq registry text/speech/transcription-family construction now also converges through the provider-owned `GroqBuilder` / `GroqConfig` boundary instead of hand-assembled compat config/client wiring, `GroqBuilder::build()` now preserves explicit HTTP client / retry overrides plus appended model middlewares, and focused Groq contract tests now lock both request-shape convergence and source-level factory routing so that path cannot silently fall back to manual compat assembly.

- DeepSeek registry text-family construction now also converges through the provider-owned `DeepSeekBuilder` / `DeepSeekConfig` boundary instead of hand-assembled provider config/client wiring, `DeepSeekBuilder::build()` now preserves explicit HTTP client / retry overrides plus appended model middlewares, and focused DeepSeek contract tests now lock both request-shape convergence and source-level factory routing so that path cannot silently fall back to manual assembly.



- Groq wrapper ergonomics now also converge further on xAI/DeepSeek by exposing `with_http_client(...)` and public wrapper helper accessors (`provider_context`, `base_url`, retry/http helpers) on `GroqClient`.
- Groq shared-builder default options now also converge on the same provider-owned request body as provider/config-first construction: `Siumai::builder().groq().with_groq_options(...)` matches `Provider::groq()` and `GroqConfig::with_groq_options(...)` on both request modes, and the underlying OpenAI-compatible `with_http_client(...)` path now preserves configured model middlewares / HTTP interceptors so factory-built wrapper paths no longer drop middleware-backed defaults.

- xAI shared-builder default options now also converge on the same provider-owned request body as provider/config-first construction: `Siumai::builder().xai().with_xai_options(...)` matches `Provider::xai()` and `XaiConfig::with_xai_options(...)` on both request modes, and the unified builder now also exposes namespaced `with_xai_reasoning_effort(...)`, `with_xai_search_parameters(...)`, and `with_xai_default_search()` helpers so common hosted-search defaults no longer depend on request-local provider maps.

- DeepSeek shared-builder default options now also converge on the same provider-owned request body as provider/config-first construction: `Siumai::builder().deepseek().with_deepseek_options(...)` matches `Provider::deepseek()` and `DeepSeekConfig::with_deepseek_options(...)` on both request modes, and the unified builder now also exposes namespaced `with_deepseek_reasoning(...)` / `with_deepseek_reasoning_budget(...)` helpers so common reasoning defaults no longer depend on request-local provider maps.

- Shared-builder provider default-options middleware now also has deterministic override semantics across the enrolled typed-default providers: later `with_*_options(...)` calls are inserted ahead of earlier default middlewares, so chained builder defaults and custom middlewares consistently override older defaults instead of silently preserving stale provider values.

- DeepSeek now also has focused shared-builder regressions that pin those override semantics directly on the final request path: later builder defaults beat earlier defaults, and appended custom model middleware still overrides builder-level default options.

- Ollama shared-builder typed default options now also converge on the same provider-owned request body as provider/config-first construction: chained `Siumai::builder().ollama().with_ollama_options(...)` defaults match `Provider::ollama().with_ollama_options(...)` and `OllamaConfig::builder().with_ollama_options(...)` on both request modes, and sparse `OllamaOptions` serialization prevents null-placeholder drift when multiple default-option layers merge.

- The adjacent Ollama provider-local lower-contract coverage is runnable again too: the test helper now unwraps the executor's optional request transformer explicitly after the shared chat-executor API moved to `Option`, so the new sparse/default-option unit tests no longer sit behind a local compile failure.



- xAI, Groq, DeepSeek, and Ollama migration safety rails now also include no-network builder/config/registry parity tests for both `chat_request` and `chat_stream_request`, locking request URL/header/body equivalence while old and new construction paths coexist.



- Top-level public construction safety rails now also cover `Siumai::builder()` / `Provider::*()` / provider config-first clients for xAI, Groq, DeepSeek, and Ollama, and the unified `Siumai` wrapper now preserves full-request semantics instead of degrading `chat_request` / `chat_stream_request` into message/tool-only fallbacks.
- DeepSeek public-path parity now also locks builder-level reasoning defaults on the real wrapper story, and the registry factory now consumes `BuildContext.reasoning_enabled` / `reasoning_budget` so `Siumai::builder().deepseek()` no longer drops default reasoning knobs that provider/config-first construction already preserved.



- DeepSeek wrapper parity is now stronger through provider-owned config/client entry types, config-first convergence, deepseek-only feature wiring, unified builder routing, registry-native text-family routing, and runtime-provider smoke coverage, but its full native provider migration is still pending.



- Registry-backed and unified-builder DeepSeek construction now also surface the provider-owned `DeepSeekClient`, and `siumai::provider_ext::deepseek` is available as the stable typed escape hatch.



- DeepSeek now also exposes provider-owned typed request options through `DeepSeekOptions` and `DeepSeekChatRequestExt`, so request-level reasoning knobs no longer need generic `with_provider_option("deepseek", ..)` calls.



- DeepSeek now also exposes provider-owned typed response metadata through `DeepSeekMetadata` and `DeepSeekChatResponseExt`, so typed metadata access no longer needs raw `provider_metadata["deepseek"]` map traversal.



- DeepSeek provider-owned typed response metadata now also has no-network wrapper-path coverage for `logprobs`, and the wrapper no longer needs a custom response transformer because extraction now reuses the shared OpenAI-compatible compat helper.



- The native DeepSeek spec tests now also read `logprobs` through `DeepSeekChatResponseExt`, so provider-owned request/response normalization coverage no longer keeps a leftover raw `provider_metadata["deepseek"]` assertion around.



- xAI, Groq, and DeepSeek now also have no-network wrapper-path plus registry-native 200-response coverage for typed metadata where registry text handles are public, and shared compat `sources` / `logprobs` extraction now converges across the xAI/Groq/DeepSeek wrapper family.



- xAI, Groq, and DeepSeek wrapper clients now also have no-network `chat_stream_request` StreamEnd coverage for typed metadata, and the same boundary is now pinned on registry-native `language_model("{provider}:...")` construction for the enrolled text-family providers, so provider-owned public paths no longer rely only on shared-runtime streaming guards.



- Google Vertex secondary migration now also routes registry factory construction through the provider-owned config/client path, builder/config tests lock `into_config()` convergence plus express/enterprise constructor behavior, registry-native text/image/embedding family construction now materializes the provider-owned `GoogleVertexClient` directly, and no-network public-path parity now also covers `Siumai::builder().vertex()` / `Provider::vertex()` / config-first chat + chat-stream request shaping.
- Google Vertex now also has a direct builder/config/registry contract anchor for provider-owned Imagen generation request shaping, closing the most obvious remaining gap between its provider factory path and the already-locked top-level public-path parity tests.
- Google Vertex now also preserves full provider-owned embedding requests on the direct client path (`EmbeddingExtensions::embed_with_config`) and has matching top-level public-path plus builder/config/registry contract anchors for embedding request shaping, so `task_type` / `title` plus Vertex-specific `outputDimensionality` / `autoTruncate` no longer risk being dropped on any supported construction path.
- Google Vertex now also has matching builder/config/registry contract anchors for provider-owned chat and chat-stream request shaping, locking `generateContent` / `streamGenerateContent` URL selection, SSE headers, and final `contents` body convergence in addition to the already-landed top-level public-path parity tests.



- OpenAI-compatible full-request chat execution now preserves request-level provider options and merges client-level defaults (`CommonParams` / `HttpConfig`) instead of silently degrading into the trait fallback path.



- DeepSeek now also owns a provider-specific chat spec normalization layer, so request-level reasoning options such as `enableReasoning` and `reasoningBudget` are normalized before shared OpenAI-compatible request mapping.



- DeepSeek now also locks Stable `response_format` precedence over raw `providerOptions.deepseek.response_format` on that provider-owned normalization path, preventing raw escape hatches from overriding the Stable structured-output contract while request-level reasoning knobs still normalize correctly.



- DeepSeek now also locks Stable `tool_choice` precedence over raw `providerOptions.deepseek.tool_choice` on that provider-owned normalization path, so tool-loop semantics no longer depend on merge order.



- xAI, Groq, and DeepSeek migration guidance now also includes runnable provider-specific examples that demonstrate typed provider options, typed metadata, Stable `response_format`, and preferred config-first or registry-first construction.



- Integration-test source hygiene now also keeps `cargo fmt --all` green by requiring valid UTF-8 in real-LLM and streaming tool-call test sources.



- Ollama now also locks tool loop invariants through request-shaping and response/streaming parity coverage, even though stricter `tool_choice` modes remain protocol-limited.
- Ollama top-level public-path parity now also covers all native `tool_choice` branches: `ToolChoice::None` omits tools on the final request body, while `Required` and specific-tool forcing fail fast with `UnsupportedOperation` across builder / provider / config-first / registry construction.



- Ollama now also exposes provider-owned typed timing metadata through `OllamaMetadata` / `OllamaChatResponseExt`, and `ollama-metadata` demonstrates the stable typed escape hatch on the public registry path.
- Ollama now also has top-level no-network 200-response coverage for that typed timing metadata plus a public-path parity guard for `reasoning(true) -> think`, so the documented metadata/reasoning story is now locked across `Siumai::builder()`, `Provider::ollama()`, and config-first `OllamaClient`.



- JSON-line streaming execution now also respects injected `HttpTransport` on stream paths, eliminating accidental reqwest fallback and making no-network parity tests authoritative for Ollama builder/config/registry/public construction.
- Non-streaming multipart execution now also respects injected `HttpTransport` on request/bytes paths, eliminating accidental reqwest fallback for OpenAI-style STT uploads and making Fireworks transcription transport-boundary tests authoritative.
- Shared compat audio providers now also have provider-boundary capture coverage across both major transport shapes: Together and SiliconFlow each lock `/audio/speech` JSON-bytes request shaping plus `/audio/transcriptions` multipart request shaping on their preset identities.
- Shared compat audio providers now also have top-level public-path parity across the full audio split where that support is advertised: Together and SiliconFlow each lock builder/provider/config-first equivalence for both TTS and STT request shaping, while Fireworks remains intentionally transcription-only.



- Ollama config-first wrapper semantics are now closer to xAI / Groq / DeepSeek through public `provider_context()` / `retry_options()` / `http_transport()` accessors and `ModelMetadata` conformance on `OllamaClient`, so public-path parity no longer depends on Ollama-only wrapper introspection.



- JSON streaming converters now also support multi-event end-of-stream emission like SSE converters, so provider-defined finish markers and final `StreamEnd` events can coexist without executor-specific hacks.



- Gemini now also locks common typed request options through both provider-owned serialization coverage and protocol-level request-shaping tests, including `responseLogprobs` / `logprobs`, `responseJsonSchema`, `retrievalConfig`, `mediaResolution`, and `imageConfig`; its structured-output path now also has explicit precedence coverage for `responseFormat`, `structuredOutputs`, and legacy `responseJsonSchema`, turning the Google-specific escape hatch story into a documented contract rather than example-only behavior.



- Secondary provider migration remains pending beyond those providers.







Status: in progress







## V4-M7 - Audio surface cleaned up







Acceptance criteria:







- `SpeechModel` is the preferred TTS surface.



- `TranscriptionModel` is the preferred STT surface.
- The unified public prelude now exports both `SpeechModel` and `TranscriptionModel` alongside the existing family modules.



- Broad `AudioCapability` is no longer the preferred public path.
- The non-unified audio escape hatches are now surfaced explicitly instead of only via the legacy bucket: `LlmClient` exposes `as_speech_extras()` / `as_transcription_extras()` alongside `as_image_extras()`, wrapper delegation is wired through `ClientWrapper` and registry-backed `Siumai`, and no-network public tests lock the expected split across native OpenAI (both), xAI (speech-only), and `fireworks` (transcription-only).
- Preferred public-path audio regressions now also exist for Together TTS, SiliconFlow STT, Fireworks transcription-only STT, and xAI provider-owned TTS, so top-level builders and config-first construction are locked to the same family routing as the underlying provider clients.
- Recommended audio call sites now also use the narrower families rather than the compatibility bucket: OpenAI fixture tests and MiniMaxi mock TTS coverage now call `SpeechCapability` / `TranscriptionCapability` plus `TranscriptionExtras`, and the legacy capability smoke test now documents the speech/transcription split instead of encouraging new `AudioCapability` call sites.
- Core narrative and public compile guards now match that guidance too: `siumai-core` module docs describe speech/transcription as the primary audio family split, while `public_surface_imports_test` locks `SpeechExtras` / `TranscriptionExtras` exports plus the `LlmClient` accessor signatures that expose them.
- Registry audio handles now also reuse cached family-model objects under the same TTL/LRU policy as the text-family handle, so repeated speech/transcription calls no longer rebuild provider family models on every invocation.
- Registry transcription extras now also have one locked end-to-end parity story: OpenAI `audio_translate` produces the same multipart `/audio/translations` request from builder, provider, config-first, and `registry.transcription_model()` entry points, and the registry handle reaches that path by delegating back into the provider-owned transcription client rather than forking request shaping locally.
- Extras boundary auditing is now executable for a second provider too: Groq keeps exposing speech/transcription-family accessors, but translation is explicitly locked as unsupported on builder/provider/config/registry entry points and short-circuits before transport instead of failing only after a request is shaped.
- Registry speech extras are now aligned with that same architecture: `SpeechModelHandle` delegates extras back into the provider-owned speech client, and xAI locks the current negative path by proving `SpeechExtras::tts_stream` stays unsupported across builder/provider/config/registry while remaining no-network.
- Provider-owned/native audio factory overrides are now guarded as a set: OpenAI, Azure, OpenAI-compatible, and Groq all have source-level contract tests for their speech/transcription family overrides, while xAI and MiniMaxi keep matching guards for their supported speech-family override path.
- Real public-path parity now includes registry audio handles too: Groq, Together, and SiliconFlow TTS/STT requests match the existing builder/provider/config-first request shapes, so `registry.speech_model()` / `registry.transcription_model()` are now tested on the same final transport boundary rather than assumed from lower-layer factory coverage.
- Registry negative-path audio boundaries now match that same story: `fireworks` transcription-only and `xai` speech-only constraints are both enforced on registry handles before transport, rather than being tested only on builder/provider/config construction paths.
- Fireworks transcription-only audio is now also guarded on the negative path: top-level builder/provider/config-first `text_to_speech` calls fail before transport, keep `as_speech_capability()` absent, and preserve the documented “STT-only on the dedicated audio host” boundary instead of leaving it as a capability-table convention.
- Provider-owned audio parity is now stronger as well: Groq locks `/audio/speech` and `/audio/transcriptions` request equivalence across `Siumai::builder()`, `Provider::groq()`, and config-first `GroqClient`, while xAI now also has an explicit top-level no-network rejection guard for standalone transcription on its speech-only wrapper story, including capability-surface assertions that keep speech/audio handles present while `as_transcription_capability()` remains absent.
- Provider package alignment now also has an explicit written policy in `provider-package-alignment.md`, separating full provider-owned packages, focused packages, and OpenAI-compatible vendor presets so future refactors do not promote every compat preset into a fake first-class package.
- Focused gap audit now also locks the narrower package boundary directly: `provider_ext::google_vertex` stays a focused wrapper with request/tool helpers plus a strict typed metadata facade under the `vertex` namespace, while `provider_ext::bedrock` stays request-helper-led until Bedrock accumulates clearly provider-owned public metadata/resources worth stabilizing.
- Compat preset promotion audit now also locks that `provider_ext::{openrouter,perplexity}` remain typed vendor views over the shared compat runtime, while `siliconflow` / `together` / `fireworks` remain preset-only until they earn promotion through stronger provider-owned public semantics.
- The compat vendor-view contract is now explicit in `compat-vendor-view-contract.md`, so `openrouter` / `perplexity` follow a written checklist for runtime ownership, capability split, metadata scope, and test layering instead of relying on ad hoc notes.
- Public API story, migration guidance, and example navigation now also describe those package tiers explicitly, so outward-facing docs no longer imply that every OpenAI-compatible vendor should become a dedicated provider package.
- `examples/04-provider-specific/README.md` now also provides a directory-level package-tier map, giving provider-owned packages, focused packages, and compat vendor views a single entry point in the examples tree.
- Every current provider-specific example subdirectory now also has its own `README.md`, so example discovery no longer depends on implicit naming or prior knowledge of which directories are provider-owned packages versus focused or compat-preset stories.
- Representative provider-specific example files now also carry package-tier header notes, so readers can see whether a file is a compat vendor view, provider-owned wrapper example, or registry-first Stable showcase without opening the surrounding README first.
- `examples/04-provider-specific/openai-compatible/README.md` now also ranks compat examples in the same narrative order used by the contract docs: typed vendor views (`openrouter`, `perplexity`) first, compat presets (`moonshot`, `siliconflow`) second, and builder-only compatibility demos last.
- The Moonshot compat examples now also match that labeling rule directly in file headers, so the preset story and the builder-demo exception are visible even when the file is opened outside the directory README context.
- The compat example headers now also share one clearer run/credential pattern: `openrouter`, `perplexity`, and `siliconflow` name their expected env vars directly in-file, and the Moonshot preset examples now use the same “set API key, then run” layout rather than mixing multiple header styles.
- The compat example bodies now also read more like one family: Moonshot and SiliconFlow use the same imported `OpenAiCompatibleClient` construction style, and their console output now follows the same plain `Example` / `Answer` / `Notes` pattern instead of mixing presentation-heavy output with the leaner vendor-view examples.
- A second pass of provider-owned/focused example headers now also follows that rule: `deepseek`, `groq`, and `xai` examples name their required credentials directly in-file, while `ollama` examples now separate package-tier guidance from local runtime prerequisites.
- High-traffic example READMEs such as `examples/README.md` and `examples/04-provider-specific/minimaxi/README.md` have now also been cleaned up to remove mojibake and duplicated navigation text, improving outward-facing docs quality without changing example behavior.
- `examples/04-provider-specific/google/README.md` and `examples/04-provider-specific/openai-compatible/README.md` now also follow the same cleaned package-tier narrative, so the most visible provider README entry points present a consistent story without encoding artifacts.
- Public-surface compile guards now also cover `provider_ext::openai_compatible`, `provider_ext::azure`, `provider_ext::google`, `provider_ext::anthropic_vertex`, and `provider_ext::deepseek`, closing the most obvious remaining export-surface holes before deeper provider feature alignment work resumes.
- Top-level compat preset guards now also lock the shared-registry construction path for `Provider::openai().{siliconflow,together,fireworks}()`, including aligned default `base_url` + `model` resolution, together with the intended audio capability split: `siliconflow` and `together` remain speech+transcription capable, while `fireworks` remains transcription-only on audio. A matching `mistral` public guard now also pins the config-only fallback lane, and `jina` / `voyageai` now pin the non-chat preset primary-default lane, so compat shortcuts resolve the documented provider default model even when the preset is embedding/rerank-led instead of chat-led. The underlying source-of-truth split is now tighter too: compat primary defaults plus family defaults now resolve from static config-owned maps first, while `default_models.rs` is only the compatibility read facade, so preset default-model lookup no longer duplicates the whole compat catalog or rebuilds the provider map on every lookup.
- The compat runtime behavior now closes the remaining model-default drift under that same source-of-truth: missing request-level `model` fields on embedding/image/rerank/speech/transcription calls now resolve from config-owned family defaults rather than blindly reusing the primary/chat default, while explicit non-default `config.model` overrides still win. Provider-local transport-boundary regressions now pin both the family-default branch and the explicit-override branch on representative vendors (`together`, `jina`, `fireworks`), and a focused public-path boundary test now carries that guarantee outward onto the real `Siumai::builder()` / `Provider::openai()` / config-first compat construction story for `together` embedding/image/TTS, `jina` rerank, and `fireworks` STT.
- OpenAI-compatible image generation now also has top-level public-path parity anchors for both `siliconflow` and `together`, locking that builder, provider, and config-first construction still converge on the same `/images/generations` request shape instead of drifting between convenience layers.
- OpenAI embedding now also has a top-level public-path parity anchor for request-level options, and the `Siumai` facade no longer drops full `EmbeddingRequest` payloads on wrapper paths: `dimensions` / `user` now survive builder, provider, and config-first construction consistently.
- Gemini embedding now also has a top-level public-path parity anchor for stable request fields, locking that retrieval-query task shaping, `title`, and `dimensions` survive builder, provider, and config-first construction consistently.
- Gemini embedding now also has a top-level batch parity anchor, locking that multi-input requests still converge on the same `:batchEmbedContents` routing and per-item body shape across builder, provider, and config-first construction.
- OpenAI-compatible `mistral` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths on the real compat shortcut story, locking that `Siumai::builder().openai().mistral()`, `Provider::openai().mistral()`, config-first `OpenAiCompatibleClient`, and `registry.embedding_model("mistral:...")` still converge on the same `/embeddings` request shape without promoting `mistral` into a standalone provider package.
- The shared OpenAI-compatible facade embedding path now also preserves full `EmbeddingRequest` payloads when the wrapper is holding an `OpenAiCompatibleClient`, so compat vendors no longer silently lose request-level options by falling back to the bare `embed(input)` path inside `Siumai::embed_with_config`.
- `fireworks` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for request-level options, locking that `dimensions`, `encoding_format`, and `user` survive builder, provider, config-first, and `registry.embedding_model("fireworks:...")` construction consistently on the real compat preset story.
- `siliconflow` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, locking that `dimensions`, `encoding_format`, and `user` survive builder, provider, config-first, and `registry.embedding_model("siliconflow:...")` construction consistently on the real compat preset story.
- `openrouter` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, locking that `dimensions`, `encoding_format`, and `user` survive builder, provider, config-first, and `registry.embedding_model("openrouter:...")` construction consistently on the real compat vendor-view story.
- `openrouter` now also has public-path parity anchors for typed chat request options and compat reasoning defaults across both request modes, locking that `transforms`, custom vendor params, `enable_reasoning`, `reasoning_budget`, and Stable `tool_choice` / `response_format` survive builder, provider, config-first, and registry construction consistently on the real compat vendor-view story for both `chat_request` and `chat_stream_request`.
- The shared compat typed-client helper now also consumes `BuildContext.reasoning_enabled` / `reasoning_budget`, closing the wrapper-path drift where `Siumai::builder()` had dropped OpenRouter reasoning defaults even though provider/config-first construction already preserved them.
- `perplexity` now also has public-path parity anchors for typed hosted-search request options across both request modes, locking that `search_mode`, `search_recency_filter`, `return_images`, nested `web_search_options`, and Stable `tool_choice` / `response_format` survive builder, provider, config-first, and registry construction consistently on the real compat vendor-view story for both `chat_request` and `chat_stream_request`.
- xAI now also has a top-level public-path parity anchor for its provider-owned web-search request options, locking that `reasoning_effort`, `search_parameters.mode`, date filters, and nested source allow/deny lists survive builder, provider, and config-first construction consistently on the real native wrapper story.
- `together` embedding now also has public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, locking that `dimensions`, `encoding_format`, and `user` survive builder, provider, config-first, and `registry.embedding_model("together:...")` construction consistently on the real compat preset story.
- `jina`, `voyageai`, and `infini` embedding now also have public-path parity anchors on both the top-level and registry embedding-handle paths for the same OpenAI-compatible request-level options, and `Provider::openai()` now exposes matching vendor shortcuts for those compat embedding stories.
- Embedding gap audit is now closed for the currently enrolled OpenAI-compatible catalog: all advertised embedding vendors are either covered by parity anchors or an explicit boundary decision, with Cohere remaining the intentional compat-only outlier because its native package is still rerank-led.
- Cohere embedding boundary is now settled and guard-tested: the native `provider_ext::cohere` package remains rerank-led on `/v2`, while embedding stays on the shared OpenAI-compatible construction story (`Provider::openai().compatible("cohere")` plus config-first `OpenAiCompatibleClient`) targeting `https://api.cohere.ai/v1/embeddings`.
- Ollama embedding now also has a provider-owned top-level public-path parity anchor, and the underlying facade/client gaps are fixed: `OllamaClient` now honors `EmbeddingExtensions::embed_with_config`, while `Siumai::embed_with_config` preserves full `EmbeddingRequest` payloads for Ollama instead of silently collapsing them to the bare input list.
- xAI and Groq embedding boundaries are now also settled and guard-tested on the provider-owned native story: registry factories reject native embedding-family construction with `UnsupportedOperation`, top-level public paths fail fast on `embed_with_config(...)` before any HTTP request is emitted, and the wrapper clients no longer surface compat embedding capability through `as_embedding_capability()`.
- xAI and Groq image/rerank boundaries are now also settled on that same native story: registry factories reject image-family construction with `UnsupportedOperation`, top-level public paths fail fast on image/rerank calls before any HTTP request is emitted, and the wrapper clients no longer surface compat image/rerank capability through `as_image_generation_capability()` / `as_image_extras()` / `as_rerank_capability()`.
- Groq audio routing now also closes a config-first drift on the same provider-owned story: root custom `base_url` values append `/openai/v1` consistently for chat and audio alike, and registry contract tests now lock native Groq speech/transcription family paths in addition to the top-level public-path parity coverage.
- DeepSeek now also conforms to that same focused-wrapper capability contract instead of being the odd one out: the provider-owned `DeepSeekClient` fail-fast rejects embedding/image/rerank locally, its wrapper-level `capabilities()` now stays aligned with the native metadata surface (`chat + streaming + tools + vision + thinking`), and both registry contract plus top-level public-path tests now lock that deferred capabilities do not leak back through `as_embedding_capability()` / `as_image_generation_capability()` / `as_rerank_capability()`.
- The focused-wrapper negative contract is now also less ad hoc in the test suite itself: shared helper assertions now encode the common “deferred capability absent / no request emitted” rules for DeepSeek, Groq, and xAI, so adding the next focused wrapper should no longer require copying three separate negative-check patterns by hand.
- Top-level compat vendor-view guards now also lock the shared-registry construction path for `Provider::openai().{openrouter,perplexity}()`, together with the intended capability split: `openrouter` remains a tools/embedding/reasoning typed vendor view without audio, while `perplexity` remains a tools-led typed vendor view without separate embedding or audio promotion. The built public-client surface now also carries that same split through `capabilities()` / `as_*_capability()` checks, closing the drift where preset metadata had advertised `reasoning` but the final OpenRouter compat client had not surfaced it.
- Top-level focused-provider guards now also lock that `provider_ext::google_vertex` remains options-led on the stable `providerOptions.vertex` key, while `provider_ext::bedrock` remains request-helper-led on `providerOptions.bedrock`; this gives the documented “stay focused, do not invent metadata/resources symmetry” rule a direct no-network test anchor.



- Legacy audio migration shims are now also locked by direct core regression tests: the compatibility-only `AudioCapability` bridge continues to adapt into the narrower `SpeechCapability` and `TranscriptionCapability` families while recommended public code paths move away from the broad audio trait.
- Provider-specific extras remain available through extension traits.







Status: in progress







## V4-M8 - Public API story unified







Acceptance criteria:







- README and architecture docs tell one coherent story.



- Recommended app-level construction is registry-first.



- Recommended provider-level construction is config-first.



- Builder usage is documented as convenience, not architecture.



- `compat` surfaces are clearly marked as temporary or compatibility-only.







Notes:

- README now includes an explicit public-surface map that separates registry-first, config-first,
  provider extension, builder convenience, and raw provider-option usage.
- Example navigation docs now use the same surface policy and keep builder examples explicitly labeled
  as convenience or compatibility demos.
- Workstream docs now point to `public-api-story.md` as the single place that defines surface intent and
  future-change policy.

Status: in progress







## V4-M9 - Stabilization and release readiness







Acceptance criteria:







- Regression coverage exists for the new contracts.



- Release notes clearly document architectural migration guidance.



- All major examples compile on the new recommended path.



- Follow-up cleanup candidates are identified for the next cycle.







Status: in progress

Notes:

- The post-refactor validation phase now has an explicit `validation-matrix.md` document that maps local smoke loops, PR gates, merge-time heavy lanes, and release readiness checks.
- CI now validates the split package topology more directly: PRs compile `siumai` under every first-class provider feature, and a dedicated provider-package build matrix compiles each `siumai-provider-*` crate under its own feature gate.
- PR CI now also runs provider-scoped no-network facade contract bundles, so each first-class provider lane has executable request/response coverage instead of compile-only smoke.
- PR CI now also runs cross-feature contract bundles for the current multi-feature coupling set (`openai,openai-websocket`, `google,gcp`, `openai,json-repair`), so non-default feature interactions are release-gated by execution instead of compile-only coverage.
- The `openai,json-repair` cross-feature lane now also guards a concrete semantic invariant: refusal/content-filter structured-output failures still surface the dedicated parse error unless a complete strict JSON value or reserved `json` tool payload already exists, so JSON repair cannot silently convert plain refusal text into a successful result.
- `pr-facade-guardrails` continues to compile `siumai` examples under `all-providers` and run `public_surface_imports_test`, so the example tree and public export paths remain part of the release gate rather than informal follow-up work.







## Release recommendation







The release should not wait for every provider to be perfectly reshaped internally.



It should wait for the following minimum bar:







- V4-M0 completed



- V4-M1 completed



- V4-M2 completed



- V4-M3 completed



- V4-M4 completed



- V4-M5 completed







That gives the project a real architectural pivot with the most important providers covered,



while allowing secondary providers to finish migration incrementally.



























