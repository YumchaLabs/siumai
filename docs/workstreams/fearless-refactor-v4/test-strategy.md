# Fearless Refactor V4 - Test Strategy







Last updated: 2026-03-15







## Purpose







The V4 refactor is only safe if test coverage is used intentionally.



This document defines what each test category is meant to protect.







## Testing principles







### T1 - Keep existing alignment tests as safety rails







The current fixture and alignment tests are a strategic advantage.



They should be preserved and reused rather than replaced casually.







### T2 - Add narrow contract tests around the new architecture







Broad fixture tests are good for behavior preservation.



They are not enough to validate new trait boundaries, registry contracts, and builder/config convergence.







### T3 - Prefer no-network verification for refactor correctness







The migration should rely primarily on:







- unit tests



- fixture/alignment tests



- mock API tests



- provider parity tests







Live integration tests remain valuable, but should not be the main signal for architectural correctness.

### T4 - Feature-minimal builds are part of the contract

The V4 package split makes feature-minimal builds a first-class correctness signal.

It is no longer enough that `--all-features` happens to pass.

We must prove that:

- each provider package builds under its own feature gate
- the `siumai` facade builds under each first-class provider feature
- `all-providers` examples still compile on the public recommended paths

### T5 - CI lanes should mirror package boundaries

Validation should follow the public package story:

- provider packages get direct package-level build coverage
- the facade gets feature-matrix coverage
- examples and public-surface imports get their own guardrail lane

See `validation-matrix.md` for the operational mapping.







## Test pyramid for V4







## Level 1 - Unit tests







Purpose:







- verify family trait adapters



- verify request/response transform helpers



- verify cache key logic



- verify middleware composition rules







Examples:







- `LanguageModel` adapter from legacy chat capability



- `SpeechModel` adapter from compatibility audio capability



- handle cache TTL expiration logic



- provider-id/model-id override functions







Expected properties:







- fast



- deterministic



- no network







## Level 2 - Contract tests







Purpose:







- validate the final family model interfaces independent of specific providers



- ensure registry handles behave as family model objects







Examples:







- language model handle satisfies generate/stream invariants



- speech model handle uses speech-only semantics
- non-unified extras should be surfaced through family-specific accessors on the public client too: `as_speech_extras()` / `as_transcription_extras()` should mirror the documented capability split instead of forcing call sites back to `AudioCapability`.
- speech-only providers should also have a top-level negative-path guard: calling the transcription family must fail before transport and keep the transcription handle absent while speech/audio handles remain available.
- transcription-only providers should also have a top-level negative-path guard: calling the speech family must fail before transport and keep the speech handle absent, so documented capability splits stay executable rather than example-only.
- once the narrow speech/transcription families exist, fixture and mock tests should exercise those traits directly; keep `AudioCapability` coverage only as a compatibility shim/regression layer rather than the recommended public example.
- compile-surface guards should also pin the public extension story: `SpeechExtras` / `TranscriptionExtras` stay exported from `siumai::extensions`, and `LlmClient` keeps `as_speech_extras()` / `as_transcription_extras()` visible on the public client trait.
- registry audio-handle tests should also pin cache reuse semantics: repeated `SpeechModelHandle` / `TranscriptionModelHandle` calls should reuse the cached family model until TTL expiry, matching the existing text-handle LRU/TTL story.
- provider-owned audio factories should also have explicit override guards where runtime behavior alone cannot distinguish a native family return from the default `ClientBacked*` bridge; source-level contract tests now cover the native audio-family override inventory directly (OpenAI/Azure/OpenAI-compatible/Groq for speech+transcription, xAI/MiniMaxi for speech-only).
- top-level public-path parity should exercise registry audio handles on the same no-network transport boundary as builder/provider/config-first construction, at least for one provider-owned audio wrapper (`groq`) and representative compat-audio presets (`together`, `siliconflow`, and the transcription-only `fireworks` split).
- once a provider-owned speech request helper becomes public, extend that same rule to `registry.speech_model(...)` as well; MiniMaxi is now the reference case, where `MinimaxiTtsOptions` are verified on both config-first `MinimaxiClient` and `registry.speech_model("minimaxi:...")` at the final `/v1/t2a_v2` request boundary.
- registry public-path negative tests should also cover split-capability vendors: transcription-only `fireworks` must reject registry speech requests before transport, and speech-only `xai` must reject registry transcription requests before transport.
- focused non-chat compat presets should also be tested at registry handle construction time, not only at invocation time: `registry.language_model("jina:...")` / `registry.language_model("voyageai:...")` must fail with `UnsupportedOperation`, while the family-specific registry entries remain the only valid public construction path.
- the same generic registry text guard should apply to focused non-chat native packages too: rerank-only `cohere` / `togetherai` must also reject `registry.language_model("provider:model")` construction up front instead of widening into dead-end chat handles.
- mixed compat presets should also keep at least one positive registry text-path parity test once they intentionally retain chat: `infini` is the reference case, where `registry.language_model("infini:...")` must stay aligned with config-first `/chat/completions` request shaping for both chat and chat streaming even though the same preset also exposes `embedding_model(...)`.
- registry transcription-handle tests should also pin one real extras path rather than only the unified STT call: OpenAI `TranscriptionExtras::audio_translate` should match builder/provider/config/registry multipart `/v1/audio/translations` requests, so handle-level delegation stays aligned with the provider-owned client surface.
- extras boundary tests should also pin at least one negative-path provider: Groq is the current reference case, where `TranscriptionExtras::audio_translate` is intentionally unsupported even though STT exists, and all builder/provider/config/registry entry points must fail before transport.
- speech extras should get the same registry-level treatment: `SpeechModelHandle` must delegate extras through the provider-owned speech client, and xAI is the current negative-path reference where wrapper/config clients intentionally hide `as_speech_extras()`, while the registry speech handle still fail-fast locks `SpeechExtras::tts_stream` with no transport activity.



- transcription model handle uses transcription-only semantics







Expected properties:







- no network



- stable and focused on architecture contracts







## CI validation lanes

The V4 test pyramid now maps into explicit CI lanes:

- **PR safety net**: fast core tests plus minimal facade checks
- **Facade feature smoke**: one feature-minimal build per first-class provider
- **Provider package build matrix**: direct compile coverage for each `siumai-provider-*` crate
- **Public guardrails**: `public_surface_imports_test` plus example compilation
- **Mainline heavy lane**: `cargo nextest run --profile ci --all-features --workspace`

This is intentional.

The architectural risk after the refactor is mostly cross-boundary drift, not lack of raw test count.

## Level 3 - Provider parity tests







Purpose:







- compare legacy path vs new path during migration



- guarantee behavior equivalence while internals change







Examples:







- builder path vs config-first path



- legacy capability path vs new family trait path



- old registry path vs new registry path







Expected properties:







- mostly no network



- removed gradually after migration completes and confidence is high







## Level 4 - Fixture/alignment tests







Purpose:







- preserve real provider protocol behavior



- catch regressions in request encoding and response decoding



- protect streaming/tool/multimodal edge cases







Examples:







- OpenAI response fixtures



- Anthropic message stream fixtures



- Gemini request/response fixture alignment







Expected properties:







- fixture-driven



- high value for protocol stability



- source files and fixtures stay valid UTF-8 so rustfmt and feature-gated compilation can parse the full tree reliably







## Level 5 - Mock API tests







Purpose:







- validate provider integration behavior against simulated APIs



- confirm HTTP flow, status handling, and response parsing remain intact







Expected properties:







- no external network



- broader than pure unit tests







## Level 6 - Live integration tests







Purpose:







- final confidence against real providers



- validate environment-dependent flows and credentials







Expected properties:







- optional for local iteration



- not the primary architecture gate







## Test matrix by concern







### Family traits







- unit tests for adapters



- contract tests for trait semantics



- parity tests during migration







### Registry







- unit tests for split/resolve/cache behavior



- contract tests for model handle execution



- parity tests for middleware and override behavior







### Builders and config







- parity tests on major providers



- smoke tests for equivalent HTTP/retry/transport settings







### Providers







- fixture/alignment tests



- mock API tests



- selected live integration tests



- feature-gated protocol crates must be exercised with the relevant feature set; for example, `siumai-protocol-openai` shared compat tests only run when `--features openai-standard` is enabled, so plain `cargo test -p siumai-protocol-openai --lib` is not a meaningful architecture gate.







### Streaming







- unit tests for chunk assembly helpers



- fixture tests for event sequences



- mock API tests for stream transport behavior







### Tools







- unit tests for tool conversion helpers



- fixture tests for tool call/result mapping



- parity tests for hosted/provider-defined tool behavior







## Required tests per milestone







### V4-M1







- unit tests for family trait adapters



- contract tests for family trait semantics







### V4-M2







- registry contract tests



- cache and middleware invariant tests







### V4-M3







- builder/config parity tests for major providers



- builder/config/registry parity tests for xAI, Groq, DeepSeek, and Ollama now compare captured no-network `chat_request` and `chat_stream_request` payloads to lock migration equivalence at the final transport boundary.

- rerank-only providers should follow the same no-network parity rule once they gain provider-owned construction surfaces; Cohere and TogetherAI now satisfy that guardrail through provider-owned config/client/builder coverage, registry typed-client construction tests, and captured rerank-request parity across siumai/provider/config-first entry points.
- mixed-capability providers with split service endpoints should lock the endpoint split at the provider-owned config layer before broad public rollout; Amazon Bedrock is now the reference pattern, with runtime vs agent-runtime ownership captured by config/client tests, registry contract tests, and no-network public-path parity for chat, chat-stream, and rerank.



- partially migrated secondary providers should still lock builder->config convergence before broader parity lands; Google Vertex and MiniMaxi now satisfy that guardrail through provider-local `into_config()` tests, config/client constructor coverage, registry factory smoke tests that verify provider-owned config routing plus preserved HTTP/common-parameter state, and MiniMaxi additionally now has siumai/provider/config public-path parity captures for both chat and chat streaming, a dedicated Bearer-header regression test on its chat spec, contract coverage for registry-native text/image family model construction, and direct no-network chat/stream metadata normalization tests that lock `provider_metadata["minimaxi"]` instead of the borrowed `anthropic` key. Google Vertex now also has registry-native text/image/embedding family contract coverage, a regression test that locks `embedding_model_with_ctx(...)` onto the provider-owned context-aware construction path instead of the default no-context fallback, and no-network public-path parity coverage across `Siumai::builder().vertex()` / `Provider::vertex()` / config-first chat, chat-stream, embedding, image-generation, and extension-only image-edit execution.
- config-first-only native packages should graduate through the same guardrail before broader parity work resumes; Azure is now the reference case where a new provider-owned `AzureOpenAiBuilder`, package-local `into_config()` tests, public export guards, focused builder/config/registry capture tests, and a source-level factory anchor were added before treating the package as builder-aligned.



- shared runtime-provider clients should also have at least one direct transport-boundary capture test for high-risk providers such as xAI and DeepSeek, so merge-hook invariants are verified after final request shaping rather than only at spec/unit level.



- pure generic runtime providers should also keep at least one public alignment test plus one direct transport-boundary capture test; OpenRouter now serves as that guardrail for the non-special-cased OpenAI-compatible path.

- pure generic runtime providers should also lock response metadata namespaces once provider-specific extras are surfaced; OpenRouter is now the reference case, where builder/provider/config/registry chat responses plus stream-end responses must preserve `provider_metadata["openrouter"]`, keep extracted `logprobs` out of a borrowed `openai_compatible` root, and expose the same payload through vendor-owned typed accessors at the response/source/content-part layers instead of only raw provider maps.
- compile-surface guards should pin those vendor-owned helpers too once they are exported; OpenRouter is now the reference case, where `siumai::provider_ext::openrouter` must keep the response/source/content-part metadata types and extension traits importable instead of letting the typed path live only in lower provider crates.



- Perplexity now follows the same generic-provider guardrail as OpenRouter, so the non-special-cased OpenAI-compatible path is protected by more than one runtime-provider exemplar.



- once a pure generic runtime provider gains provider-owned typed request helpers, add both a public alignment test and a direct transport-boundary capture test for those typed helpers; OpenRouter and Perplexity now satisfy that guardrail through `OpenRouterOptions` / `PerplexityOptions`, and both now also carry that guard across `chat_request` plus `chat_stream_request` on the real builder/provider/config-first/registry paths.



- native provider-owned typed request helpers need the same two-layer guard once they exist: keep provider-local spec tests for transformer normalization, but add at least one top-level public-path parity test that proves the helper survives both request modes and builder/provider/config-first construction; Anthropic, Groq, DeepSeek, Ollama, and MiniMaxi now satisfy that guardrail through `AnthropicOptions` / `AnthropicChatRequestExt`, `GroqOptions` / `GroqChatRequestExt`, `DeepSeekOptions` / `DeepSeekChatRequestExt`, `OllamaOptions` / `OllamaChatRequestExt`, and `MinimaxiOptions` / `MinimaxiChatRequestExt`.



- once the registry path is native for that provider family, extend the same typed-helper guard to `registry.*_model(...)` as well; Anthropic, Groq, DeepSeek, Ollama, and MiniMaxi are now the reference cases for text-family request helpers, where `AnthropicOptions` / `GroqOptions` / `DeepSeekOptions` / `OllamaOptions` / `MinimaxiOptions` are verified on both config-first clients and `registry.language_model("...")` at the final transport boundary for both `chat_request` and `chat_stream_request`.

- xAI now also demonstrates the full text-family version of that rule for provider-owned web-search helpers: `XaiOptions` fields are verified on both config-first `XaiClient` and `registry.language_model("xai:...")` at the final transport boundary for both `chat_request` and `chat_stream_request`.

- provider-owned builder convergence tests also need to pin stateful override lanes, not just final request normalization; xAI, Groq, and DeepSeek are the provider-owned reference cases, and the generic OpenAI-compatible path now joins them for known built-in vendors: package-local tests assert `with_http_client(...)` / `with_retry(...)` survive `XaiBuilder::build()` / `GroqBuilder::build()` / `DeepSeekBuilder::build()` / `OpenAiCompatibleBuilder::build()`, registry parity tests lock request equivalence for both request modes, and source-level contracts guard that `XAIProviderFactory` / `GroqProviderFactory` / `DeepSeekProviderFactory` / `OpenAICompatibleProviderFactory` still route through `XaiBuilder::new(...).with_http_config(...).with_model_middlewares(...)` / `GroqBuilder::new(...).with_http_config(...).with_model_middlewares(...)` / `DeepSeekBuilder::new(...).with_http_config(...).with_model_middlewares(...)` / `OpenAiCompatibleBuilder::new(...).with_http_config(...).with_model_middlewares(...)` instead of drifting back to manual assembly.

- the same registry-level rule also applies to native non-text typed helpers once those handle surfaces are public; Google Vertex is now the reference case, where `registry.embedding_model("vertex:...")` and `registry.image_model("vertex:...")` must preserve `VertexEmbeddingOptions` / `VertexImagenOptions` against config-first `GoogleVertexClient` on the final embedding / image-generation / image-edit transport boundary instead of relying only on factory-level contract tests.

- shared OpenAI-compatible non-text request surfaces should get the same registry-level guard once a vendor view or compat preset publicly exposes them; OpenRouter is the typed vendor-view reference, and `fireworks` / `siliconflow` / `together` / `jina` / `voyageai` / `infini` now satisfy the same rule on the preset side, where `registry.embedding_model("...")` must preserve Stable embedding request fields (`dimensions`, `encoding_format`, `user`) against config-first `OpenAiCompatibleClient` on the final `/embeddings` transport boundary.
- shared OpenAI-compatible preset coverage should also lock capability-gated image family routing, not just request shaping; Together is now the reference image-generation preset where builder/config/registry must converge on `/images/generations`, while OpenRouter is the negative-path reference and must reject `image_model_family_with_ctx(...)` before any transport is touched.
- once a shared OpenAI-compatible preset already has config-first/public parity for a non-text path, extend the same guard to registry handles instead of stopping at factory contracts; OpenRouter embedding is the reference `/embeddings` case, and Together/SiliconFlow now serve as the image-generation `/images/generations` reference presets.
- apply the same rule to shared compat rerank presets once they are advertised publicly; Jina and VoyageAI are now the reference `/rerank` presets where config-first and `registry.reranking_model("...")` must stay aligned, while OpenRouter is the negative-path reference and must reject rerank before transport because it only advertises embedding on the current compat non-text surface.
- focused compat presets should also have an explicit no-chat guard when their config only advertises non-text families; Jina and VoyageAI are now the reference cases where adapter capabilities, factory capabilities, preset guards, and top-level builder/provider/config/registry chat calls must all agree on `UnsupportedOperation` before transport, while Infini remains the mixed-surface reference that still advertises chat/streaming.
- if the registry still exposes a compatibility `language_model("provider:model")` entry for such focused presets, add a handle-level guard too; the handle must not surface `as_chat_capability()` and direct `chat_request(...)` / streaming calls must fail before client construction so the type name does not silently widen the provider contract.



- once a shared OpenAI-compatible runtime provider gains provider-owned typed response metadata, add at least one no-network 200-response test that asserts the typed extractor after shared compat response shaping; OpenRouter, Perplexity, xAI, Groq, and DeepSeek now satisfy that guardrail for their current typed fields, and OpenRouter additionally carries the same guard across `StreamEnd`, source metadata, and content-part metadata on the public path.



- if a wrapper provider previously needed response post-processing for typed metadata, prefer moving extraction into the shared compat helper instead of maintaining a provider-specific transformer; DeepSeek now follows that guardrail for `logprobs`, converging on the same helper path as xAI and Groq.



- if that provider also streams through the shared OpenAI-compatible SSE path, add a matching no-network stream-end test so typed metadata parity is locked across both non-streaming and streaming execution; Perplexity now satisfies that guardrail on the generic runtime path, and xAI / Groq / DeepSeek now satisfy it on their provider-owned wrapper paths for current typed metadata.



- when typed provider metadata is extracted from the shared OpenAI-compatible runtime, keep the extraction rules in one compat helper consumed by non-streaming transformers and any matching SSE end-event conversion; Perplexity uses that helper across 200-response and `StreamEnd`, and xAI / Groq / DeepSeek now verify the same helper through non-streaming response guards plus wrapper-path `StreamEnd` tests for `sources` / `logprobs`.



- shared builder/config helper semantics should also have at least one no-network end-to-end request-body guard; OpenRouter now covers that requirement for `with_reasoning` / `with_reasoning_budget` on the generic OpenAI-compatible path.
- compat vendor-view defaults that must survive registry construction need a second guard at the provider-scoped override boundary: OpenRouter now locks `ProviderBuildOverrides::{with_reasoning,with_reasoning_budget}` plus final registry request capture, while Perplexity locks the same registry public path for request-level vendor options without inventing a fake provider-owned config layer.
- when the registry also exposes a global default lane, lock that lane separately at the final transport boundary instead of assuming provider-scoped override tests are enough; OpenRouter now has a dedicated no-network guard for `RegistryOptions` / `RegistryBuilder` global reasoning defaults.
- apply the same rule to provider-owned wrappers that consume shared reasoning defaults through `BuildContext`; DeepSeek and xAI now anchor `RegistryOptions` global reasoning defaults against config-first request capture, so the global lane is not only guarded on compat vendor views.
- if `RegistryBuilder` is part of the public construction story, give it its own transport-boundary anchor instead of treating it as a thin alias of `create_provider_registry(...)`; OpenRouter, DeepSeek, and xAI now each lock direct builder-based global reasoning defaults against config-first request capture.



- top-level public-path parity tests now also compare `Siumai::builder()` / `Provider::*()` / config-first provider clients for xAI, Groq, DeepSeek, and Ollama, ensuring the compatibility wrapper does not strip request-level fields while old and new entry points coexist.
- provider-owned wrapper defaults that travel through `BuildContext` need their own public-path parity anchors too: DeepSeek now locks builder/provider/config-first reasoning defaults at the final request boundary, catching the real drift where `Siumai::builder().deepseek()` had dropped `reasoning` / `reasoning_budget` even though provider/config-first construction already preserved them.
- provider-owned typed metadata surfaces should also gain a top-level public-path anchor once the request-side wrapper surface is already public: xAI, Groq, and DeepSeek now have no-network builder/provider/config-first plus registry `language_model("{provider}:...")` guards for both 200-response and `StreamEnd` typed metadata on their enrolled text-family paths, and Perplexity now does too on the generic compat vendor-view path plus registry `language_model("perplexity:...")`, including normalized `provider_metadata["perplexity"]` roots, so shared compat extraction is exercised beyond provider-local wrapper tests.
- the same rule also applies when the typed boundary lives in protocol-owned helpers exposed through a native provider facade: OpenAI now has no-network builder/provider/config-first guards for typed `OpenAiSourceExt` extraction on Responses 200-response / `StreamEnd` plus typed `OpenAiChatResponseExt.logprobs` on Chat Completions 200-response / `StreamEnd`, so source-annotation and logprobs coverage no longer stop at protocol transformer / serializer tests.
- the same mixed-boundary rule also needs an explicit top-level guard when a provider owns typed response metadata but still leaves streaming event payloads raw: Google / Gemini now has no-network builder/provider/config-first checks for typed `GeminiContentPartExt` / `GeminiChatResponseExt` extraction on 200-response and `StreamEnd`, while `reasoning-start` custom payloads still assert raw `providerMetadata.google.thoughtSignature` plus normalized `provider_metadata["google"]` roots on the event boundary.
- the inverse rule must be pinned just as explicitly when a provider intentionally stays raw: Azure now has a no-network builder/provider/config-first guard on the Responses streaming public path that verifies `text-start` / `finish` custom event payloads and final `provider_metadata["azure"]` roots remain namespace-scoped raw metadata, preventing wrapper refactors from silently widening the typed boundary.
- the same inverse rule now also applies to borrowed Gemini protocol wrappers that intentionally remain namespace-led at the provider layer: Google Vertex now has a no-network builder/provider/config-first guard on the streaming public path that verifies `reasoning-start` / `reasoning-delta` custom payloads keep `providerMetadata.vertex.thoughtSignature` and final `provider_metadata["vertex"]` roots preserve the Vertex namespace without introducing a dedicated Vertex metadata extractor surface.
- the same rule also applies to native non-compat providers whose typed metadata already exists below the public facade: Anthropic now has no-network builder/provider/config-first plus registry `language_model("anthropic:...")` guards for `AnthropicMetadata.container` on both 200-response and `StreamEnd`, while the same tests intentionally keep `service_tier` on `ChatResponse.service_tier` because that field is part of the unified response contract rather than provider-owned metadata.
- the same Anthropic guard should start from a plain `ChatRequest` rather than a pre-marked streaming request when possible; that variant already caught a real config-first gap where `AnthropicClient::chat_stream_request(...)` had failed to force `stream=true` before executor dispatch.
- when a provider-owned metadata surface intentionally renames a borrowed wire-protocol namespace, the top-level guard should pin both the typed extractor and the normalized raw root key: MiniMaxi now has no-network builder/provider/config-first plus registry `language_model("minimaxi:...")` checks for both 200-response and `StreamEnd` metadata, and those same checks also caught/fixed the missing `stream=true` bit on its overridden `chat_stream_request` path.
- provider-owned typed metadata surfaces should not stop at package-local extractor tests once they are documented publicly: Ollama now has no-network top-level plus registry `language_model("ollama:...")` 200-response guards that verify `OllamaChatResponseExt` timing metadata through `Siumai::builder()`, `Provider::ollama()`, config-first `OllamaClient`, and registry-native construction, not only through provider-local conversion tests.
- the same rule should hold for provider-owned JSON streaming when typed metadata is finalized on `StreamEnd`: Ollama now also has no-network top-level plus registry `chat_stream_request` guards that verify `OllamaChatResponseExt` timing metadata on `ChatStreamEvent::StreamEnd`, using captured NDJSON transport payloads so JSON-line execution is exercised instead of accidentally testing SSE fallback behavior.
- the same JSON-stream rule now also has a focused-provider reference pattern: Bedrock top-level parity verifies `BedrockChatResponseExt` on both 200-response and JSON-stream `StreamEnd`, using captured Converse NDJSON chunks so typed metadata coverage does not silently stop at request shaping or package-local reserved-`json` tests.
- provider-owned request-helper surfaces that are part of the documented public story also need a final wrapper-path capture test: xAI web search now locks typed `search_parameters` plus `reasoning_effort` across `Siumai::builder()`, `Provider::xai()`, and config-first `XaiClient`, so nested source filters no longer rely only on provider-local request-normalization tests.
- when a provider is a focused wrapper package, use the checklist in `focused-wrapper-contract.md` and prefer shared helper assertions for “deferred capability absent / no request emitted” instead of re-encoding negative checks per provider.
- once a focused capability-split provider already has request/metadata parity on its public family, add unsupported-family guards on both top-level wrapper paths and registry-native handles so absent capabilities stay explicit instead of drifting back in through generic factories.
- once those public-path guards are green, add lower-level provider-factory rejection tests for the same unsupported families and a registry-handle no-request anchor so generic builder bridges cannot silently reintroduce the capability through `language_model_with_ctx(...)`; Cohere and TogetherAI are now the reference cases for that completed pattern.
- apply the same lower-boundary rule to non-text family leaks too: Gemini is now the reference case where public `speech` / `transcription` unsupported guards are paired with provider-factory rejection tests, preventing an internal embedding/text fallback from silently reconstructing unsupported audio-family clients.
- focused wrappers that borrow another wire protocol still need the full negative-path ladder under their own provider identity: Anthropic-on-Vertex now serves as the reference case where builder/provider/config-first unsupported-family checks, provider-factory rejection tests, and public registry-handle guards all assert the `anthropic-vertex` boundary instead of assuming Anthropic coverage is sufficient.
- when a provider is a compat vendor view, use the checklist in `compat-vendor-view-contract.md` and prefer preset guards + provider-override capture tests before adding broader parity coverage.



- JSON-streaming providers such as Ollama must exercise captured custom-transport stream payloads in parity coverage, preventing accidental fallback to real HTTP from hiding at the executor boundary.
- Structured-output parity on those JSON-streaming paths now needs a second behavioral guard beyond request shaping: if the converter synthesizes `StreamEnd { finish_reason: Unknown }` on transport close, the public extractor must reuse accumulated deltas / reserved-tool payloads, succeeding only for already-complete JSON and otherwise returning the dedicated incomplete-stream parse error. Ollama is the reference no-network builder/provider/config/registry case for that rule.
- Apply the same guard to shared OpenAI-compatible SSE routes whenever a provider-owned wrapper publicly exposes that path: xAI is now the reference case where builder/provider/config/registry all stream through shared compat SSE, and public-path parity pins the synthetic-unknown-end structured-output success/failure split instead of relying only on request-body capture coverage.
- The same guard should also be applied to vendor-view public paths that stay on the shared compat client rather than promoting a provider-owned config type; Perplexity and OpenRouter are now the reference builder/provider/config/registry cases for that vendor-view branch.
- Apply the same idea to provider-owned reserved-tool streaming paths too: Bedrock is now the JSON-stream reference builder/provider/config case where reserved `json` tool deltas on the Converse JSON stream must still distinguish complete accumulated tool input from truncated interrupted input on the public extractor path.
- Anthropic is now the SSE fallback reference for the same rule: forcing `providerOptions.anthropic.structuredOutputMode = "jsonTool"` on a supported model pins builder/provider/config/registry extraction when reserved `json` tool deltas complete before transport close versus terminate in a truncated state before `message_stop`.



- JSON-streaming core tests now explicitly cover reqwest execution, injected transport execution, and middleware-expanded end events so end-of-stream semantics stay aligned with SSE during refactors.




- Multipart providers that rely on OpenAI-style file uploads must also exercise captured custom-transport payloads at the executor boundary, including boundary-bearing `Content-Type`, raw body bytes, and single-401 retry parity; Fireworks transcription now satisfies that guardrail through `siumai-core` multipart transport tests plus a direct `OpenAiCompatibleClient` capture test.
- Shared compat audio providers should also cover both body-shape families on real preset identities: Together and SiliconFlow now each lock JSON TTS request shaping on `/audio/speech` plus multipart STT request shaping on `/audio/transcriptions`.
- If a provider has only a partial audio contract in public docs, add explicit no-network rejection tests for the missing half instead of silently routing to a shared fallback; xAI now follows that guardrail by locking provider-owned speech support while explicitly rejecting standalone transcription.
- Once provider-level audio mapping is green, add at least one preferred-public-path regression per provider family so wrapper/builders cannot silently drop audio semantics; this now exists for Together TTS, SiliconFlow STT, Fireworks STT, and xAI TTS across `Siumai::builder()`, `Provider::*`, and config-first construction.
- Provider-owned audio wrappers with OpenAI-shaped endpoints need the same root-base normalization checks as chat wrappers: Groq now locks builder/provider/config-first parity for both `/audio/speech` and `/audio/transcriptions`, and the no-network regression caught a real config-first drift where root `base_url` had not appended `/openai/v1` on audio requests.







### V4-M4 to V4-M6







- provider fixture/alignment tests



- mock API tests



- parity tests while old/new paths coexist







### V4-M7







- audio-family split tests



- extension-trait coverage for provider-specific extras







## Removal policy for temporary tests







Provider parity tests created only for migration may be removed after:







1. the new architecture is the only production path for that provider



2. fixture and contract coverage are sufficient



3. builders and config paths are already converged







## Definition of done for test coverage







A refactor milestone is not done just because code compiles.



It is done when:







- the new contract has dedicated tests



- the high-risk behaviors still have fixture coverage



- migration-only equivalence has been verified where needed



- docs point contributors to the intended test layer



