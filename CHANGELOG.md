# Changelog

## [0.11.0] - 2025-10-19

### Highlights (concise additions)
- BREAKING: Unify Chat entry points on ChatCapability (remove legacy extension methods)
  - New unified methods on ChatCapability: `chat_request(ChatRequest)` and `chat_stream_request(ChatRequest)`; prefer these going forward.
  - Removed the similarly named methods from ChatExtensions (`chat_request/chat_stream_request/chat_stream_request_with_cancel`). Call the trait methods directly on ChatCapability.
  - OpenAI client implements these unified methods using the Transformers + Executors pipeline and preserves enhanced fields such as `provider_params/http_config/web_search/telemetry`.
  - Added: Gemini / Anthropic / Groq / xAI / Ollama now also override `chat_request/chat_stream_request`, routing via their Transformers + Executors with full preservation of enhanced fields.
- Middleware pipeline: wired `wrap_generate_async`/`wrap_stream_async` with `transform → pre_* → wrap_* → HTTP → post/on_event`, plus tests and docs.
- **Middleware Override**: `LanguageModelMiddleware` now supports `override_provider_id()` and `override_model_id()` for dynamic routing (aligned with Vercel AI SDK's `wrapLanguageModel`).
- **Registry LRU Cache**: `ProviderRegistryHandle` now features LRU cache with configurable capacity (`max_cache_entries`) and optional TTL (`client_ttl`) to prevent unbounded growth; includes concurrent access de-duplication.
- **Orchestrator Advanced Tests**: Added comprehensive tests for usage aggregation precision, error injection after tool execution, mixed approval decisions (Approve/Deny), and concurrent tool call ordering.
- **Server Adapters Enhancement**: Improved with Axum integration (`to_sse_response`, `to_text_stream`), enhanced `SseOptions` with presets (`development()`/`production()`/`minimal()`), configurable error masking, and comprehensive documentation.
- **Telemetry & Observability**: New `telemetry` module with structured event tracking (SpanEvent, GenerationEvent, OrchestratorEvent, ToolExecutionEvent); Langfuse and Helicone exporter support; optional telemetry in Orchestrator and HttpChatExecutor via `TelemetryConfig`; automatic span tracking for all LLM requests; complete stream telemetry with automatic GenerationEvent emission on stream completion.
- Orchestrator (streaming): added `generate_stream_owned` (tokio::spawn + mpsc) for multi‑step streaming with `on_chunk/on_step_finish/on_finish`; re‑enabled stream test.
- HTTP 401 retry: non‑stream chat path restores one‑shot 401 header rebuild + retry; `build_headers` changed to `Arc<..>` internally and providers updated.
- Server adapters: feature renamed to `server-adapters`; examples declare `required-features` for cleaner builds.
- Retry module layout: moved to `src/retry/{policy.rs, backoff.rs}` with `retry::policy` and `retry::backoff`; keep `retry_api` as the stable facade.

### Changed
- Core refactor introducing a clear Transformers + Executors architecture:
  - Transformers: request/response/stream/audio/files traits (`src/transformers/*`).
  - Executors: HTTP orchestration for chat/embedding/image/audio/files (`src/executors/*`).
- Unified streaming pipeline based on `eventsource-stream` with multi‑event emission (StreamStart, deltas, Usage, StreamEnd) via `StreamFactory`.
- OpenAI client now routes non‑streaming and streaming chat through the new `HttpChatExecutor` with `OpenAi*Transformer`s.
- OpenAI‑compatible providers now share centralized adapter + transformer + streaming event conversion logic (`providers/openai_compatible/*`).
- Introduced `ProviderRegistryV2` + factory helpers to reduce builder branching and enable config‑driven provider wiring.
- **Provider Registry Architecture** (aligned with Vercel AI SDK):
  - New `ProviderFactory` trait with async capability-specific methods (language_model, embedding_model, etc.).
  - `ProviderRegistryHandle` delegates client creation to factory instances; capability handles implement corresponding traits.
  - String-driven model resolution via `"provider:model"` format; middleware applied at handle level.
  - Internal: trait objects now use `Arc<dyn LlmClient>` instead of `Box<dyn LlmClient>` for better cloning performance.
- Request headers unified across providers via `utils::http_headers::ProviderHeaders` and `inject_tracing_headers`; custom headers merged from `http_config.headers`.
- OpenAI-Compatible header unification preserves adapter custom headers + `http_config.headers` + config `custom_headers` (compat client).
- OpenAI Rerank now uses `ProviderHeaders::openai`; accepts `HttpConfig` for custom headers/tracing.

### Added
- Registry LRU Cache with TTL
  - `ProviderRegistryHandle` caches language model clients with LRU eviction (default: 100 entries).
  - Configurable via `RegistryOptions::max_cache_entries` and `RegistryOptions::client_ttl`.
  - TTL-based expiration; concurrent access de‑duplication via async Mutex.
  - Cache key includes provider and model ID to support middleware overrides correctly.
- OpenAI native transformers: `OpenAiRequestTransformer`, `OpenAiResponseTransformer`, plus Responses API transformers.
- Streaming utilities: `SseEventConverter` and helpers for unified `ChatStreamEvent` mapping.
- Structured Output parity across providers:
  - OpenAI: `response_format` (json_object/json_schema + strict) for Chat and Responses.
  - Gemini: maps `provider_params.structured_output` to `generationConfig.responseMimeType/responseSchema`.
  - Anthropic: maps `provider_params.structured_output` to `response_format` (json_object/json_schema + strict).
  - Groq/xAI (OpenAI‑compatible style): maps `provider_params.structured_output` to `response_format` for JSON mode.
  - Ollama: maps `provider_params.structured_output` to `format` (schema object or "json").
- Files and Audio transformers/executors for consistent upload/STT/TTS flows.
- Public typed options scaffold under `siumai::public::options` (converts to `ProviderParams`).
- Tests: end-to-end header flow tests (OpenAI Files, Anthropic chat with beta, Gemini Files, OpenAI-Compatible chat, Groq/xAI chat); multipart negative checks (Groq STT, OpenAI Files upload).
- HTTP Interceptors (experimental but stable API):
  - New `utils::http_interceptor::{HttpInterceptor, HttpRequestContext}` with hooks:
    `on_before_send`, `on_response`, `on_error`, `on_sse_event`.
  - Unified `LlmBuilder::with_http_interceptor(...)` and `http_debug(true)` to install
    custom interceptors and enable a built‑in `LoggingInterceptor` (no sensitive data).
  - Provider builders inherit interceptors from `LlmBuilder`; provider clients expose
    `with_http_interceptors(...)` (e.g. OpenAI/Groq) for direct installation.
  - Interceptors run for both non‑streaming and streaming chat; SSE events are surfaced
    via a wrapper converter so `on_sse_event` observes provider chunks.
  - Headers visible to interceptors reflect merged runtime headers (tracing + custom + http_config).

### Removed
- Legacy `retry_strategy` (use the new `retry_api` facade).
- Legacy `request_factory` and scattered per‑provider request builders in favor of transformers.
- Duplicated per‑provider streaming parsers replaced by unified transformers/converters.
- Legacy `SiumaiBuilder::build_legacy` (modern build path is `provider/build.rs`).
- Unused legacy provider registry code from `src/provider.rs` (replaced by new registry system in `src/registry/entry.rs`).

### Migration Guide (short)
- Retries: replace any usage of `retry_strategy` with `retry_api::{retry, retry_for_provider, retry_with(RetryOptions)}`. Builders support `with_retry(...)`.
- Custom/third‑party providers: implement `RequestTransformer`, `ResponseTransformer`, and (if streaming) `StreamChunkTransformer`; wire them through the generic `Http*Executor`s. For SSE, prefer `StreamFactory::create_eventsource_stream` and return multiple `ChatStreamEvent`s as needed.
- OpenAI‑compatible integrations: implement a `ProviderAdapter` with `FieldMappings` for content/thinking/tool fields; register via `ProviderRegistryV2` or continue using the compatible builder which now uses the centralized transformers.
- OpenAI native: if you built on internal mapping code, move to `OpenAiRequestTransformer` (Chat/Embedding/Image) and Responses API transformers. ParameterMapper is no longer needed.
- Streaming behavior: ensure your stream transformer emits `StreamStart` and `StreamEnd` and can produce multiple events per provider chunk (thinking/tool call/usage updates).
- If you instantiate `OpenAiRerank::new(...)` directly, add a final `HttpConfig` argument; usage via `OpenAiClient` is unchanged.
- **Custom ProviderFactory implementations**: If you implemented the old `ProviderFactory` trait, migrate to the new async trait in `src/registry/entry.rs` (change `Box::new(client)` to `Arc::new(client)`, implement async capability methods). Most users unaffected.

## [0.10.3] - 2025-10-10

### Added
- Unified retry API `retry_api` (`retry`, `retry_for_provider`, `retry_with`).
- Builder-level retry options: `with_retry(...)` for `Siumai` and provider builders (OpenAI, Gemini, Anthropic, Groq, xAI, Ollama, OpenAI-compatible).
- Convenience methods: `chat_with_retry`, `ask_with_retry` on `ChatExtensions`.
- Stream processor: overflow handler now accepts closures.

### Deprecated
- `retry_strategy` (planned removal in 0.11).

### Changed
- SiliconFlow and OpenRouter now use the OpenAI-compatible adapter path.
- Simplified tracing guard type and provider identification; removed an unused `Siumai` field.

### Fixed
- Responses API `web_search` now returns `UnsupportedOperation` when not implemented.

### Migration
- Replace `retry_strategy` usage with the unified `retry_api` facade:
  - Use `retry`, `retry_for_provider`, or `retry_with(RetryOptions::...)`.
  - Prefer builder-level `with_retry(...)` for chat operations (applies to Siumai and provider builders).
- `retry_strategy` is deprecated and will be removed in `0.11`.

## [0.10.2] - 2025-10-04

- Unified HTTP client across providers, exposed fine-grained HTTP options on SiumaiBuilder, added with_http_client for Gemini/Custom, and updated docs/examples.

## [0.10.1] - 2025-09-14

### Fixed

- **OpenAI StreamDelta Thinking Field Support** - Fixed #7: Added unified thinking field priority handling (reasoning_content > thinking > reasoning) to OpenAI StreamDelta, matching OpenAI-compatible adapter behavior for consistent thinking content processing across all providers
- **OpenAiCompatibleBuilder Base URL Configuration** - Fixed #7: Added base_url() method to OpenAiCompatibleBuilder enabling custom base URLs for self-deployed OpenAI-compatible servers, alternative endpoints, and local development scenarios

## [0.10.0] - 2025-08-29

### Added

- **Provider-Specific Embedding Configurations** - Added type-safe embedding configuration options for each provider (GeminiEmbeddingOptions with task types, OpenAiEmbeddingOptions with custom dimensions, OllamaEmbeddingOptions with model parameters) through extension traits, enabling optimized embeddings while maintaining unified interface
- **Enhanced ChatMessage System** - Improved ChatMessage with better serialization/deserialization support
- **OpenAI-Compatible Adapter System** - Completely refactored OpenAI-compatible provider system with centralized configuration through unified registry system, supporting 36 providers: DeepSeek, OpenRouter, Together AI, Fireworks, Perplexity, Mistral, Cohere, Zhipu, Moonshot, Doubao, Qwen, 01.AI, Baichuan, SiliconFlow (with comprehensive chat, embeddings, image generation, and document reranking capabilities), Groq, xAI, GitHub Copilot, GitHub Models, Nvidia, Hyperbolic, Jina AI, VoyageAI, StepFun, MiniMax, Infini AI, ModelScope, Hunyuan, Baidu Cloud, Tencent Cloud TI, Xirang, 302.AI, AiHubMix, PPIO, OcoolAI, Poe, and enhanced OpenAI-compatible builder with comprehensive HTTP configuration support including timeout, proxy, and custom headers.
- **Secure Debug Trait Implementation** - Implemented custom Debug trait for all client types with complete sensitive information hiding (API keys, tokens) using clean `has_*` flags instead of masked values, providing production-safe debugging output.

### Fixed

- **StreamStart Event Generation** - Fixed missing StreamStart events in streaming responses across all providers, now properly emitting metadata (id, model, created, provider, request_id) at stream beginning. Implemented multi-event emission architecture that preserves all content while ensuring StreamStart events.

### Changed

- **Streaming Architecture** - Refactored streaming traits to support multi-event emission (breaking change for internal APIs only, user-facing APIs unchanged)
- **Provider Implementations** - All providers now use optimized multi-event conversion logic for better content preservation and consistency

## [0.9.1] - 2025-08-28

### Added

- **Comprehensive Clone Support** - All client types, builders, and configuration structs now implement `Clone` for seamless concurrent usage and multi-threading scenarios

## [0.9.0] - 2025-08-27

### Added

- **Provider Feature Flags** - Added optional feature flags for selective provider inclusion (`openai`, `anthropic`, `google`, `ollama`, `xai`, `groq`) with build-time validation

### Fixed

- **Ollama API Key Requirement** - Fixed SiumaiBuilder to allow Ollama provider creation without API key, as Ollama doesn't require authentication

## [0.8.1] - 2025-08-25

### Added

- **RequestBuilder Send+Sync Support** - Added Send+Sync constraints to RequestBuilder trait for better multi-threading support

### Fixed

- **Type Downcasting Anti-pattern** - Replaced runtime type downcasting with capability methods in `LlmClient` trait
- **Memory Limits in Stream Processing** - Added configurable buffer limits (10MB content, 5MB thinking, 100 tool calls) with overflow handlers
- **Inconsistent Macro Return Types** - All message macros now consistently return `ChatMessage` instead of mixed types
- **Send+Sync Static Assertions** - Added compile-time verification for error type thread safety

### Added

- **Application-Level Timeout Support** - New `TimeoutCapability` trait provides timeout control for complete operations including retries, complementing existing HTTP-level timeouts

## [0.8.0] - 2025-08-13

### Breaking Changes

- **Security and Reliability Improvements** - Introduced `secrecy` crate for secure API key handling, `backoff` crate for professional retry mechanisms.
- **Streaming Infrastructure Overhaul** - Replaced custom streaming implementations with `eventsource-stream` for professional SSE parsing and UTF-8 handling across all providers.

### Added

- OpenAI Responses API support (sync, streaming, background, tools, chaining)
- **Simplified Model Constants** - Introduced simplified namespace for model constants (`siumai::models`) with direct access to model names. Replaced complex categorization system with intuitive model selection: `models::openai::GPT_4O`, `models::anthropic::CLAUDE_OPUS_4_1`, `models::gemini::GEMINI_2_5_FLASH`. Provides better IDE auto-completion and faster model discovery without abstract groupings.

### Fixed

- **URL Compatibility** - Fixed URL construction across all providers to handle base URLs with and without trailing slashes correctly, preventing double slash issues in API endpoints.
- **Anthropic API Compatibility** - Fixed Anthropic API max_tokens requirement by automatically setting default value (4096) when not provided, resolving "max_tokens: Field required" errors.
- **xAI Grok Streaming** - Implemented complete streaming support for xAI Grok models with reasoning capabilities. Added `XaiEventConverter` and `XaiStreaming` components that handle real-time content streaming, reasoning content processing (`reasoning_content` field), tool calling, and usage statistics including reasoning tokens. The implementation follows the same reliable eventsource-stream architecture used by other providers.
- **Common Parameters Not Applied** - Fixed issue where `common_params` (temperature, max_tokens, top_p, etc.) were not being applied in certain scenarios when using ChatCapability trait methods. The parameter passing mechanism has been corrected to ensure both common parameters and provider-specific parameters are properly merged and sent to API endpoints in both streaming and non-streaming modes across all providers.

### Internal

- **Request Builder Refactoring** - Internally refactored parameter construction and request builder implementation for improved maintainability and consistency across all providers.
- **Enhanced Test Coverage** - Added comprehensive test suites including mock framework, concurrency safety tests, network error handling tests, resource management tests, and configuration validation tests to ensure production-ready reliability.
- **Architecture Cleanup** - Removed redundant `UnifiedLlmClient` struct that was a thin wrapper around `ClientWrapper`, simplifying the architecture and reducing API confusion. Removed unused `ClientFactory` that duplicated functionality already provided by `SiumaiBuilder`. Fixed misleading error messages in embedding capability that incorrectly stated OpenAI and Gemini don't support embeddings. Corrected inconsistent documentation examples and parameter type formats across README and examples. All Clippy warnings have been resolved and code consistency has been improved throughout the codebase.

## [0.7.0] - 2025-08-02

### Fixed

- **Code Quality and Documentation** - Fixed all clippy warnings, documentation URL formatting, memory leaks in string interner and optimized re-exports to reduce namespace pollution
- **Tool Call Streaming** - Fixed incomplete tool call arguments in streaming responses. Previously, only the first SSE event from each HTTP chunk was processed, causing tool call parameters to be truncated. Now all SSE events are properly parsed and queued, ensuring complete tool call arguments in streaming mode. [#1](https://github.com/YumchaLabs/siumai/issues/1)
- **StreamEnd Events** - Fixed missing or incorrect StreamEnd events in streaming responses. StreamEnd events are now properly sent when `finish_reason` is received, with correct finish reason values (Stop, ToolCalls, Length, ContentFilter).
- **Send + Sync Markers** - Added Send + Sync bounds to all capability traits and stream types for proper multi-threading support. [#2](https://github.com/YumchaLabs/siumai/issues/2)

## [0.6.0] - 2025-08-01

### Added

- **Unified Embedding API** - Unified embedding API through `Siumai` client with builder patterns and provider-specific optimizations for OpenAI, Gemini, and Ollama

## [0.5.1] - 2025-07-27

### Added

- **Unified Tracing API** - All provider builders (Anthropic, Gemini, Ollama, Groq, xAI) now support tracing methods (`debug_tracing()`, `json_tracing()`, `minimal_tracing()`, `pretty_json()`, `mask_sensitive_values()`)

## [0.5.0] - 2025-07-27

### Added

- **Enhanced Tracing and Monitoring System** - Complete HTTP request/response tracing with security features
  - **Pretty JSON Formatting** - `.pretty_json(true)` enables human-readable JSON bodies and headers

    ```rust
    .debug_tracing().pretty_json(true)  // Multi-line indented JSON
    ```

  - **Sensitive Value Masking** - `.mask_sensitive_values(true)` automatically masks API keys and tokens (enabled by default)

    ```rust
    // Default: "Bearer sk-1...cdef" (secure)
    .mask_sensitive_values(false)  // Shows full keys (not recommended)
    ```

  - **Comprehensive HTTP Tracing** - Automatic logging of request/response headers, bodies, timing, and status codes
  - **Multiple Tracing Modes** - `.debug_tracing()`, `.json_tracing()`, `.minimal_tracing()` for different use cases
  - **UTF-8 Stream Handling** - Proper handling of multi-byte characters in streaming responses
  - **Security by Default** - API keys automatically masked as `sk-1...cdef` to prevent accidental exposure

## [0.4.0] - 2025-06-22

### Added

- **Groq Provider Support** - Added high-performance Groq provider with ultra-fast inference for Llama, Mixtral, Gemma, and Whisper models
- **xAI Provider Support** - Added dedicated xAI provider with Grok models support, reasoning capabilities, and thinking content processing
- **OpenAI Responses API Support** - Complete implementation of OpenAI's Responses API
  - Stateful conversations with automatic context management
  - Background processing for long-running tasks (`create_response_background`)
  - Built-in tools support (Web Search, File Search, Computer Use)
  - Response lifecycle management (`get_response`, `cancel_response`, `list_responses`)
  - Response chaining with `continue_conversation` method
  - New types: `ResponseStatus`, `ResponseMetadata`, `ListResponsesQuery`
  - New trait: `ResponsesApiCapability` for Responses API specific functionality
- Configuration enhancements for Responses API
  - `with_responses_api()` - Enable Responses API mode
  - `with_built_in_tool()` - Add built-in tools (WebSearch, FileSearch, ComputerUse)
  - `with_previous_response_id()` - Chain responses together
- Comprehensive documentation and examples for Responses API usage

### Changed

- **BREAKING**: Simplified `ChatStreamEvent` enum for better consistency
  - Unified `ThinkingDelta` and `ReasoningDelta` into single `ThinkingDelta` event
  - Removed duplicate `Usage` event (kept `UsageUpdate`)
  - Removed duplicate `Done` event (kept `StreamEnd`)
  - Reduced from 10 to 7 stream event types while maintaining full functionality
- Enhanced `OpenAiConfig` with Responses API specific fields
- Updated examples to demonstrate Responses API capabilities

### Fixed

- Updated all examples to use new `StreamEnd` event instead of deprecated `Done` event
  - Fixed `simple_chatbot.rs`, `streaming_chat.rs`, and `capability_detection.rs` examples
  - Ensured all streaming examples work with the simplified event structure

## [0.3.0] - 2025-06-21

### Added

- `ChatExtensions` trait with convenience methods (ask, translate, explain)
- Capability proxies: `AudioCapabilityProxy`, `EmbeddingCapabilityProxy`, `VisionCapabilityProxy`
- Static string methods (`user_static`, `system_static`) for zero-copy literals
- LRU response cache with configurable capacity
- `as_any()` method for type-safe client casting

### Fixed

- Streaming output JSON parsing errors caused by network packet truncation
- UTF-8 character handling in streaming responses across all providers
- Inconsistent streaming architecture between providers

### Changed

- **BREAKING**: Capability access returns proxies directly (no `Result<Proxy, Error>`)
- **BREAKING**: Capability checks are advisory only, never block operations
- Split `ChatCapability` into core functionality and extensions
- Improved error handling with better retry logic and HTTP status handling
- Optimized parameter validation and string processing performance
- Refactored streaming implementations with dedicated modules for better maintainability
- Added line/JSON buffering mechanisms to handle incomplete data chunks
- Unified streaming architecture across OpenAI, Anthropic, Ollama, and Gemini providers

### Removed

- `register_capability()` and `get_capability()` methods
- `with_capability()`, `with_audio()`, `with_embedding()` deprecated methods
- `FinishReason::FunctionCall` (use `ToolCalls` instead)
- Automatic capability warnings

## [0.2.0] - 2025-06-21

### Added

- Ollama provider support (chat, streaming, embeddings, model management)
- Multimodal support for vision-capable models
- `PartialEq` support for `MessageContent` and `ContentPart`

## [0.1.0] - 2025-06-20

### Added

- Initial release with unified LLM interface
- Providers: OpenAI, Anthropic Claude, Google Gemini, xAI, OpenRouter, DeepSeek
- Capabilities: Chat, Audio, Vision, Tools, Embeddings
- Streaming support and multimodal content
- Retry mechanisms and parameter validation
- Macros: `user!()`, `system!()`, `assistant!()`, `tool!()`
