# Changelog

## [0.9.2] - 2025-08-29

### Added

- **SiliconFlow Provider with Rerank Support** - Added comprehensive SiliconFlow provider integration including chat, embeddings, image generation, and document reranking capabilities for improved RAG systems

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
