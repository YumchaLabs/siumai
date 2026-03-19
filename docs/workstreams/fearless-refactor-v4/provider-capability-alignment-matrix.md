# Provider Capability Alignment Matrix

This document tracks the next-stage provider alignment work after the V4 metadata refactor pass.

The goal is not to force every provider into the same shape. The goal is to make the current
contract explicit:

- which capabilities are already aligned and guarded,
- which are intentionally provider-owned,
- which still need parity work across builder/config/registry/runtime paths.

## Legend

- `Done`: already aligned and guarded by tests/docs.
- `Partial`: supported, but still missing parity or one public-path/runtime anchor.
- `Deferred`: intentionally not unified in V4 yet.

## Capability Matrix

| Provider | `response_format` | `tool_choice` | Reasoning knobs | Hosted search metadata | Public-path parity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | Done | Done | Done | Done | Done | Native OpenAI already owns the richest stable response-side contract, and top-level provider/config/builder parity now explicitly covers both Responses and Chat Completions entry points plus typed reasoning + Stable `response_format` / `tool_choice` request shaping; mixed-registry `ProviderBuildOverrides` now explicitly cover both non-stream and stream `responses` lanes, Responses 200-response / `StreamEnd` now verify typed `OpenAiSourceExt` extraction on final source annotations, Chat Completions 200-response / `StreamEnd` now verify typed `OpenAiChatResponseExt.logprobs` on the real public path, and the same typed metadata contract is now pinned on the actual registry facade for both `openai` and `openai-chat` handles against config-first construction. |
| Azure | Done | Done | Done | Deferred | Done | Audit complete on the current boundary: top-level builder/provider/config parity now covers both `azure` (Responses) and `azure_chat` (Chat Completions), Azure-owned `chat_before_send` closes reasoning plus Stable `response_format` / `tool_choice` request shaping on both routes, and Responses public-path plus registry/contract parity now lock 200-response roots, `text-start` / `finish` custom events, and final `provider_metadata["azure"]` `StreamEnd` roots as intentionally raw namespace-rewrite checks rather than a typed Azure metadata contract. |
| Anthropic | Done | Done | Done | Done | Done | Audit complete: response-side metadata remains Anthropic-shaped rather than OpenAI-shaped, hosted-search/source payloads are already typed through `AnthropicMetadata.sources` / citations on the stable facade, top-level builder/provider/config-first guards now also verify `AnthropicMetadata.container` on both 200-response and `StreamEnd` without widening unified fields like `service_tier` into vendor metadata, lower-contract builder/config/registry anchors now also pin typed `AnthropicOptions` request helpers on both request modes, provider-local `AnthropicClient::chat_stream_request(...)` regressions now lock transport headers plus typed thinking/container/context-management/output-config shaping, and deferred non-text family builders/handles now reject explicitly at the factory layer instead of falling back to the generic language-model client. |
| Google / Gemini | Done | Done | Done | Deferred | Done | Top-level builder/provider/config parity now covers Gemini chat, chat-stream, and embedding routes, lower builder/config/registry contract now also locks Stable `response_format` / `tool_choice` plus provider-owned `thinkingConfig` / `structuredOutputs` on the final `generateContent` request body, public registry handles now also pin that same structured-output/tool-routing contract directly against config-first construction, mixed-registry `ProviderBuildOverrides` now explicitly cover both `generateContent` and `streamGenerateContent`, and public-path metadata parity now also verifies typed `GeminiContentPartExt` / `GeminiChatResponseExt` extraction on 200-response and `StreamEnd` while keeping `reasoning-start` `providerMetadata.google.thoughtSignature` payloads plus normalized `provider_metadata["google"]` roots on the raw event boundary. |
| Google Vertex | Done | Done | Done | Done | Done | Top-level builder/provider/config parity now also locks Stable `response_format` / `tool_choice` plus provider-owned `thinkingConfig` on the final Vertex `generateContent` request body, lower builder/config/registry contract now also pins that same `response_format` / `tool_choice` plus `thinkingConfig` / `structuredOutputs` convergence directly, and the plain Vertex wrapper now also exposes a strict provider-owned typed metadata facade (`VertexMetadata`, `VertexChatResponseExt`, `VertexContentPartExt`) bound only to `provider_metadata["vertex"]`; focused public-path plus registry guards now verify typed `usageMetadata` / `safetyRatings`, response-level `groundingMetadata` / `urlContextMetadata` / normalized `sources`, and per-part `thoughtSignature`, while stream public-path plus lower-contract parity continue to lock raw `reasoning-start` / `reasoning-delta` payloads and final namespace-root assertions. |
| Bedrock | Done | Done | Deferred | Deferred | Done | Provider-owned public-path parity now locks chat, chat-stream, rerank, Stable `response_format` / `tool_choice`, and the current narrow `BedrockMetadata` surface across builder/provider/config-first construction; response metadata remains intentionally Bedrock-shaped rather than widened into a broader cross-provider contract. |
| Ollama | Done | Done | Deferred | Deferred | Done | Audit complete: top-level builder/provider/config parity already locks chat, chat-stream, and embedding request shaping, lower-contract builder/config/registry anchors now also pin typed `OllamaOptions` request helpers on both chat request modes, the native chat request builder now also forwards `raw` into the final `/api/chat` body, mixed-registry `ProviderBuildOverrides` now explicitly cover both non-stream and stream `api/chat` lanes, the public-surface compile guard now also pins `OllamaChatRequestExt`, `OllamaEmbeddingRequestExt`, and `OllamaChatResponseExt` without widening the response-side contract past provider-owned timing metadata, and deferred `image` / `rerank` / audio-family builders now reject explicitly at the provider-factory layer instead of falling back through the text client. |
| xAI | Done | Done | Done | Done | Done | Audit complete: hosted-search metadata is already closed on the stable facade through typed `sources` on `XaiMetadata`, provider-owned `XaiSourceExt` metadata for source-level file/container fields, top-level builder/provider/config-first guards now also lock both 200-response and `StreamEnd` extraction, lower-contract builder/config/registry anchors now also pin typed `XaiOptions` request helpers plus Stable `tool_choice` / `response_format` precedence on the native wrapper story, and provider-local stream regressions now also pin `Accept: text/event-stream`, `stream=true`, Stable `response_format` / `tool_choice`, and typed `XaiOptions` search fields on the final transport body; streaming search-event details remain intentionally raw custom payloads. |
| Groq | Done | Done | Done | Deferred | Done | Audit complete: current stable response contract is still `logprobs`/`sources`, top-level builder/provider/config-first guards now also lock both 200-response and `StreamEnd` extraction, lower-contract builder/config/registry anchors now also pin typed `GroqOptions` request helpers plus Stable `tool_choice` / `response_format` precedence on the native wrapper story, and provider-local stream regressions now also pin `Accept: text/event-stream`, `stream=true`, Stable `response_format` / `tool_choice`, plus typed `GroqOptions` fields on the final transport body; no extra `ContentPart` helper is justified by in-repo fixtures. |
| DeepSeek | Done | Done | Done | Deferred | Done | Audit complete: `reasoning_content` already lands on unified `ContentPart::Reasoning`, top-level builder/provider/config-first guards now also lock both 200-response and `StreamEnd` `DeepSeekMetadata.logprobs`, lower-contract builder/config/registry anchors now also pin typed reasoning options plus Stable `tool_choice` / `response_format` precedence, provider-local stream regressions now also pin `Accept: text/event-stream`, `stream=true`, Stable `response_format` / `tool_choice`, and DeepSeek request-option normalization on the final transport body, and deferred `embedding` / `image` / `rerank` / audio-family builders now reject explicitly at the provider-factory layer instead of falling back through the text client. |
| OpenRouter | Done | Done | Done | Deferred | Done | Hosted-search widening remains deferred, but the current provider-owned typed response story is now closed for the OpenAI-shaped compat payload: builder/provider/config/registry parity locks `provider_metadata["openrouter"]` plus typed `OpenRouterChatResponseExt` / `OpenRouterSourceExt` / `OpenRouterContentPartExt` on both 200-response and `StreamEnd` without inventing a broader hosted-search contract, capability-boundary tests keep `embedding` as the only public non-text family while pinning `image` / `rerank` / audio-family paths to fail-fast unsupported behavior, and provider-local stream regressions now also pin `Accept: text/event-stream`, Stable `response_format` / `tool_choice`, builder reasoning defaults, and `OpenRouterOptions` on the final shared compat transport body without requiring a custom adapter rewrite branch. |
| Perplexity | Done | Done | Deferred | Done | Done | Audit complete: the current typed surface already owns the stable hosted-search fields observed in-repo (`citations`, `images`, `usage.citation_tokens`, `usage.num_search_queries`, `usage.reasoning_tokens`) across public-surface imports, provider-local non-stream/stream runtime mapping, and top-level builder/provider/config-first guards for both 200-response and `StreamEnd`; non-text capability-boundary tests now also pin Perplexity as a text-only vendor view with `embedding` / `image` / `rerank` / audio-family paths all rejected before transport, and provider-local stream regressions now also pin `Accept: text/event-stream`, Stable `response_format` / `tool_choice`, and typed `PerplexityOptions` on the final shared compat transport body without introducing a provider-specific adapter rewrite. |
| MiniMaxi | Done | Done | Done | Deferred | Done | Anthropic-shaped compatibility still matters, but MiniMaxi now also locks Stable `tool_choice`, preserves `stream=true` on `chat_stream_request`, lower-contract builder/config/registry anchors now also pin typed `MinimaxiOptions` request helpers on both chat request modes, provider-local stream regressions now also pin `Accept: text/event-stream`, typed thinking-mode shaping, Stable `response_format` precedence, and the resulting tool stripping on the final transport body, verifies provider-owned typed metadata plus normalized `provider_metadata["minimaxi"]` keys on both 200-response and `StreamEnd`, and explicitly rejects deferred `embedding` / `rerank` / `transcription` family builders before any generic text-client fallback can reintroduce them. |
| Cohere | Deferred | Deferred | Deferred | Deferred | Done | Native package remains rerank-led by design, and top-level builder/provider/config-first public-path parity for native rerank request shaping is already locked. |
| TogetherAI | Deferred | Deferred | Deferred | Deferred | Done | Native package remains rerank-led by design, and top-level builder/provider/config-first public-path parity for native rerank request shaping is already locked. |

## Current Release Gate

- No provider-wide `Priority A` parity gap is currently identified on the advertised public facade.
- Rich native packages (`openai`, `anthropic`, `gemini` / `google`, `google_vertex`) are now in
  maintenance mode on the current public contract, while `azure` remains intentionally raw/deferred
  on hosted metadata typing.
- Focused wrappers (`xai`, `groq`, `deepseek`, `ollama`, `minimaxi`, `cohere`, `togetherai`,
  `bedrock`) are now considered closed on the current package boundary: helper tiers, request
  helpers, focused capability splits, and public-path parity all have explicit guards.
- Remaining `Deferred` cells should be treated as release-approved scope boundaries, not hidden work:
  widen them only when a provider ships a stronger stable contract than the one already locked here.

## Next Pass Priorities

### Priority A

- No cross-provider capability promotion is queued right now.
- Start a new `Priority A` item only when at least one of the following appears:
  - a new provider-owned public API lane,
  - new stable response metadata or resource semantics,
  - a documented matrix/package-policy claim that no longer matches the guarded public facade.

### Priority B

- xAI / Groq / DeepSeek audit is closed for now: keep xAI hosted-search custom event payloads raw
  where they validate streaming event shape, keep Groq on `logprobs`/`sources`, and keep
  DeepSeek on unified `ContentPart::Reasoning` unless stronger provider-specific response payloads
  appear in real fixtures/runtime evidence.
- Keep OpenAI request/metadata parity in maintenance mode unless a new native lane or wider stable
  response surface is intentionally introduced; Azure, Google/Gemini, Google Vertex, and Ollama
  request/public-path parity are now considered closed for this pass, with Azure now treated as an
  explicit deferred/raw hosted-metadata boundary rather than a partial typed-metadata gap, and
  Azure, Google/Gemini, and Google Vertex stream metadata boundaries now pinned explicitly on both
  the public path and the provider-owned registry lane where applicable.

### Priority C

- Keep OpenRouter's broader hosted-search surface deferred even though the current alias-based
  vendor metadata story is now closed for `logprobs` / `sources`; only widen beyond that
  provider-owned compat view if stronger response-side evidence appears.
- Perplexity hosted-search metadata audit is closed on the current typed boundary; only widen past
  `citations`, `images`, and usage-side search/reasoning counters if stronger stable response
  fixtures or runtime captures appear.
- Keep Azure custom streaming payloads and `google-vertex` metadata raw/options-focused until the
  provider contract becomes more stable than today's namespace/event assertions; current top-level
  parity now pins those namespace/event boundaries, and Azure registry parity now pins the same raw
  namespace contract on both 200-response and `StreamEnd`, but that is still not a reason to widen
  Azure or Google Vertex response-side typing without stronger non-event evidence.
- Treat Cohere / TogetherAI as rerank-led native packages with public-path parity already closed
  unless product scope broadens beyond rerank.

## Usage Guidance

- Use this matrix to choose the next provider alignment task.
- If a capability is marked `Deferred`, treat that as an explicit boundary, not missing work.
- When a capability moves from `Partial` to `Done`, update this matrix in the same change as the
  code/tests/docs.

## Compat Audio Boundary Notes

- `siliconflow` and `together` remain the reference OpenAI-compatible audio presets for shared
  TTS/STT request-shape parity.
- The shared compat-audio mixed-registry path is now explicit too: `siliconflow` and `together`
  both have public-path plus lower-contract `ProviderBuildOverrides` anchors on the real
  `/audio/speech` and `/audio/transcriptions` routes, while `fireworks` carries the same override
  coverage on its transcription-only `/audio/transcriptions` host path.
- `fireworks` remains transcription-only on the compat audio path.
- None of the current compat audio presets (`siliconflow`, `together`, `fireworks`) should be
  treated as owning a stable translation extras contract yet: `TranscriptionExtras::audio_translate`
  is intentionally unsupported on builder/provider/config-first/registry public paths, and that
  boundary is now locked by no-network tests rather than implied by missing examples.

## Compat Image Boundary Notes

- `siliconflow` and `together` remain the reference OpenAI-compatible image presets on the shared
  `/images/generations` surface.
- The shared compat-image mixed-registry path is now explicit too: `siliconflow` and `together`
  both have public-path plus lower-contract `ProviderBuildOverrides` anchors on the real
  `/images/generations` route, so registry-native image handles preserve provider-scoped
  auth/base-url/transport instead of falling back to registry-global defaults.
- `siliconflow` now also has a provider-local no-network runtime anchor on that same
  `/images/generations` boundary, so the shared compat image executor is pinned below the
  top-level/provider/registry parity layer as well.
- `together` now also has a provider-local no-network runtime anchor on `/images/generations`,
  so both currently enrolled compat image presets are pinned below the public-path layer instead of
  relying on a single representative provider-local guard.
- `openrouter` remains an explicit no-image compat preset unless the provider advertises a stable
  provider-owned image contract later.

## Compat Rerank Boundary Notes

- `siliconflow`, `jina`, and `voyageai` remain the reference OpenAI-compatible rerank presets on
  the shared `/rerank` surface.
- The shared compat-rerank mixed-registry path is now explicit too: `siliconflow`, `jina`, and
  `voyageai` all have public-path plus lower-contract `ProviderBuildOverrides` anchors on the real
  `/rerank` route, so registry-native rerank handles preserve provider-scoped
  auth/base-url/transport instead of falling back to registry-global defaults.
- `siliconflow` now also has a provider-local no-network runtime anchor on `/rerank`, giving the
  shared compat rerank executor a direct provider-local transport-boundary guard in addition to the
  public/registry parity layers.
- `jina` and `voyageai` now also have provider-local no-network runtime anchors on `/rerank`, so
  the shared compat rerank executor is pinned on all three currently advertised compat rerank
  presets instead of relying on a single representative provider-local guard.
- Focused native rerank packages (`cohere`, `togetherai`, `bedrock`) remain provider-owned paths
  and should not be folded back into the compat preset backlog.

## Compat Embedding Boundary Notes

- The current OpenAI-compatible embedding preset set (`mistral`, `fireworks`, `siliconflow`,
  `together`, `openrouter`, `jina`, `voyageai`, `infini`) now all has explicit shared
  `/embeddings` request-shape parity plus mixed-registry override anchors.
- The shared compat-embedding mixed-registry path is now explicit too: those preset lanes all have
  public-path plus lower-contract `ProviderBuildOverrides` anchors on the real `/embeddings`
  route, so registry-native embedding handles preserve provider-scoped auth/base-url/transport
  instead of falling back to registry-global defaults.
- `together` now also has a provider-local no-network runtime anchor on `/embeddings`, joining the
  earlier Fireworks provider-local embedding anchor so the shared compat embedding executor is
  covered on more than one preset family below the public-path layer.
- `infini` now also has a provider-local no-network runtime anchor on `/embeddings`, so the mixed
  chat-plus-embedding compat preset is pinned below the public-path layer as well rather than
  relying only on registry/config parity.
- `fireworks` now also has a provider-local no-network runtime anchor on the inference-host
  `/embeddings` boundary, so the shared compat embedding executor is pinned directly against the
  split-host preset that also owns the separate audio transcription base.
- `openrouter` remains the vendor-owned embedding view in this set; the others stay compat presets
  unless they accumulate enough provider-owned surface area to justify promotion.
