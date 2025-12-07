# Changelog

This file lists noteworthy changes. Sections are grouped by version to make upgrades clearer.

## [0.11.0-beta.4] - 2025-12-03

### Added

- Provider factories for Anthropic on Vertex AI and MiniMaxi
  - New factory types: `AnthropicVertexProviderFactory`, `MiniMaxiProviderFactory`
  - Both fully support the shared `BuildContext` (HTTP client/config, tracing, middlewares, retry)
- Default registry factory wiring
  - `registry::helpers::create_registry_with_defaults()` now pre-registers factories for:
    - Native providers: `openai`, `anthropic`, `anthropic-vertex`, `gemini`, `groq`, `xai`, `ollama`, `minimaxi`
    - All built-in OpenAI-compatible providers (DeepSeek, SiliconFlow, OpenRouter, Together, Fireworks, etc.)
- `siumai-extras` workflow and memory abstractions
  - New `WorkflowBuilder<M>` + `Workflow<M>` on top of `Orchestrator<M>` and `ToolLoopAgent<M>`
  - Semantic worker role helpers and constants: `WORKER_PLANNER`, `WORKER_CODER`, `WORKER_RESEARCHER`
  - Pluggable `WorkflowMemory` trait and in-process `InMemoryWorkflowMemory` implementation
  - Example `workflow_planner_coder` showing planner + coder + in-memory memory using OpenAI
- Unified structured output decoding helpers in `siumai-extras`
  - `structured_output::OutputDecodeConfig` used by high-level object helpers, agents, orchestrator, and workflows
  - Shared JSON repair, shape hints, and optional JSON Schema validation (via `schema` feature)

### Changed

- Base URL override semantics for native providers
  - Custom `base_url` values for OpenAI, Gemini, Anthropic, Ollama, xAI, and MiniMaxi are now treated as full API prefixes
  - When a custom `base_url` is set, Siumai no longer appends provider default paths such as `/v1` or `/v1beta`; callers must include any required path segments explicitly
  - Default base URLs (e.g. `https://api.openai.com/v1`, `https://generativelanguage.googleapis.com/v1beta`) are still used when no override is provided
- Unified construction path for `SiumaiBuilder` and Registry
  - `SiumaiBuilder::build()` no longer calls provider helpers directly
  - Instead, it builds a `BuildContext` and delegates to the corresponding `ProviderFactory::language_model_with_ctx()`
  - Ensures that HTTP config, custom `reqwest::Client`, API keys, base URLs, tracing, interceptors, middlewares, and retry options behave identically across:
    - `Siumai::builder()...build()`
    - `registry::global().language_model("provider:model")`
- Anthropic / Gemini / Groq / xAI / Ollama / MiniMaxi registry factories
  - All providers now construct clients via the shared helper functions in `registry::factory` (or their own config/client types) using `BuildContext`
  - Registry-level HTTP interceptors and model middlewares are consistently installed across all clients
- Retry option propagation (builder + registry)
  - `BuildContext.retry_options` is now applied uniformly to all supported providers (OpenAI, Anthropic, Anthropic Vertex, Gemini, Groq, xAI, Ollama, MiniMaxi, and all OpenAI-compatible adapters)
  - `Siumai::builder().with_retry(...)` and `RegistryOptions.retry_options` configure the underlying provider clients via their unified `set_retry_options` / `with_retry` APIs, rather than adding separate ad hoc layers
- Siumai outer retry wrapper semantics
  - `SiumaiBuilder::build()` no longer automatically wraps the resulting `Siumai` instance in an additional retry layer
  - Recommended usage: configure retry via the builder or registry; `Siumai::with_retry_options(...)` remains available as an explicit, opt-in wrapper for advanced scenarios
- Orchestrator and high-level object helpers moved to `siumai_extras`
  - `siumai::orchestrator::*` and `siumai::highlevel::object::*` are now provided by `siumai_extras::orchestrator` and `siumai_extras::highlevel::object`
  - Core `siumai` focuses on low-level provider/client APIs; application-level workflows (agents, structured objects with schema validation) live in `siumai_extras`
- `siumai-extras` structured output API clean-up
  - Renamed extras-side decode config from `StructuredOutputConfig` to `OutputDecodeConfig` to clarify separation from provider-native structured output configs (e.g., OpenAI)
  - High-level `generate_object` / `stream_object`, `ToolLoopAgent` structured output, `Orchestrator::run_typed`, and `Workflow::run_typed` are all backed by the same decode pipeline
- Orchestrator / workflow ergonomics
  - Added `tool_choice(...)` and `active_tools(...)` builders on `OrchestratorBuilder`, `Orchestrator`, and `WorkflowBuilder`
  - These are thin sugar over `prepare_step`, mirroring Vercel AI SDK's `toolChoice` / `activeTools` for common cases
- Provider registry metadata
  - Added a native `anthropic-vertex` entry (with alias `google-vertex-anthropic` and `claude` model prefix) to align routing between builder and registry

### Removed

- Deprecated top-level helper modules from the core crate
  - Removed `siumai::benchmarks`; benchmarking and diagnostics helpers now live in `siumai-extras` or in user code
  - Removed the `siumai::telemetry` shim; telemetry is now wired via `siumai::observability::telemetry` in the core crate and `siumai-extras::telemetry` for subscriber setup

## [0.11.0-beta.3] - 2025-11-09

### Added

- Unified model-level middleware on `Siumai::builder()`
  - New APIs: `add_model_middleware(...)`, `with_model_middlewares(...)`
  - Auto middlewares now also apply to the unified builder path
- OpenTelemetry 0.31 compatibility
  - Switch to `SdkTracerProvider`, use `Resource::builder_*`, update `PeriodicReader::builder(...)`
  - Updated example under `siumai-extras/examples/opentelemetry_tracing.rs`

### Changed

- MiniMaxi moved to factory flow for consistency
  - Middlewares and interceptors are installed uniformly across all providers
- Consolidated builder helpers and advanced HTTP options
  - Shared utilities for API key/base URL/model normalization
  - Parity of advanced HTTP options between `Siumai::builder()` and `LlmBuilder`

### Fixed

- Applied gzip/brotli/cookie_store flags when building HTTP client
- Correct model propagation for OpenAI‑compatible in unified builder
- Env var loading for OpenAI‑compatible (`{PROVIDER_ID}_API_KEY`)
- Default/alias model handling across providers

## [0.11.0-beta.2] - 2025-11-08

### Added

- MiniMaxi provider support with multi-modal capabilities (text, speech, image generation).
- **Gemini File Search (RAG) support** - Provider-specific implementation for Gemini's File Search API
  - File Search Store management (create, list, get, delete)
  - Example: `siumai/examples/04-provider-specific/google/file_search.rs`

## [0.11.0-beta.1] - 2025-10-28

This beta delivers a major refactor of module layout, execution/streaming, and provider integration. Design inspired by Cherry Studio’s transformer design and the Vercel AI SDK’s adapter architecture.

### Added
- Provider Registry and model handles (`siumai/src/registry/*`)
  - Unified string-based `provider:model` resolution with LRU caching and optional TTL
  - Customizable registry options (middlewares, interceptors, retry)
- HTTP Interceptors (`execution::http::interceptor`)
  - Request/response hooks and SSE event observation
  - Built-in `LoggingInterceptor`
- Execution layer and middleware system (`execution::{executors,transformers,middleware}`)
  - Auto middlewares based on provider/model (defaults/clamping/reasoning extraction)
- Orchestrator rework (`siumai-extras/src/orchestrator/*`)
  - Multi-step tool calling, agent pattern, tool approval, streaming tool execution
  - See examples under `siumai/examples/03-advanced-features/orchestrator/`
- High-level object APIs (`siumai_extras::highlevel::object`)
  - `generate_object` / `stream_object` for provider-agnostic typed JSON outputs
  - Optional JSON repair and schema validation; partial object streaming
- `siumai-extras` crate
  - Optional features: `schema`, `telemetry`, `opentelemetry`, `server`, `mcp`
- Example rework (`siumai/examples/`)

### Changed
- Workspace split into `siumai` and `siumai-extras`.
- Unified streaming events (start/delta/usage/end); improved UTF‑8-safe chunking and tag extraction.
- Unified retry facade (`retry_api`) with idempotency and 401 token refresh retry.
- OpenAI‑compatible providers consolidated via adapter; consistent transformers/executors paths.
- Clippy cleanups; boxed large enum variants internally (minor internal breaking).

### Removed
- Top-level `examples/` moved to `siumai/examples/`.
- Removed obsolete `docs/openapi.documented.yml`.

### Fixed
- Ensure `before_send_hook` is correctly applied across providers.
- UTF‑8 safety: tag extraction, string slicing, streaming chunk boundaries, and token masking.
- Reliability fixes in streaming, headers, and parameter mapping; expanded fixture-based tests.

### Known Issues
- OpenAI Responses API `web_search` is not implemented; calling returns `UnsupportedOperation`.

### Stability
- This is a beta pre-release; minor API adjustments may follow.

### Roadmap
- Starting with `0.11.0-beta.5`, the workspace will be split into multiple crates (core / providers / extras) to mirror the architectural separation already present in the code. The `0.11.0-beta.4` release focuses on closing the feature loop and stabilizing the unified crate API before this split.

### API Keys and Environment Variables

- OpenAI: `.api_key(..)` or `OPENAI_API_KEY` (env fallback)
- Anthropic: `.api_key(..)` or `ANTHROPIC_API_KEY` (env fallback)
- Groq: `.api_key(..)` or `GROQ_API_KEY` (env fallback)
- Gemini: `.api_key(..)` or `GEMINI_API_KEY` (env fallback)
- xAI: `.api_key(..)` or `XAI_API_KEY` (env fallback)
- Ollama: no API key (local service, default `http://localhost:11434`)
- OpenAI‑compatible via Builder: `.api_key(..)` or `{PROVIDER_ID}_API_KEY`
- OpenAI‑compatible via Registry: reads `{PROVIDER_ID}_API_KEY` (e.g., `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`)

### Migration Guide

#### Tracing Subscriber Initialization

**Before (v0.10.3 and earlier):**
```rust
use siumai::tracing::{init_default_tracing, init_debug_tracing, TracingConfig, OutputFormat};

// Initialize with default configuration
init_default_tracing()?;

// Or with custom configuration
let config = TracingConfig::builder()
    .log_level_str("debug")?
    .output_format(OutputFormat::Json)
    .build();
init_tracing(config)?;
```

**After (v0.11.0):**

Option 1: Use `siumai-extras::telemetry` for advanced configuration:
```rust
use siumai_extras::telemetry;

// Add to Cargo.toml:
// For the beta release:
// siumai-extras = { version = "0.11.0-beta.3", features = ["telemetry"] }

// Initialize with default configuration
telemetry::init_default()?;

// Or with custom configuration
let config = telemetry::SubscriberConfig::builder()
    .log_level_str("debug")?
    .output_format(telemetry::OutputFormat::Json)
    .build();
telemetry::init_subscriber(config)?;
```

Option 2: Use `tracing-subscriber` directly for simple cases:
```rust
// Add to Cargo.toml:
// tracing-subscriber = "0.3"

// Simple console logging
tracing_subscriber::fmt::init();
```

#### JSON Schema Validation

**Before:**
```rust
// Schema validation was not available in core siumai
```

**After:**
```rust
use siumai_extras::schema;

// Add to Cargo.toml:
// For the beta release:
// siumai-extras = { version = "0.11.0-beta.3", features = ["schema"] }

// Validate JSON against schema
schema::validate_json(&instance, &schema)?;

// Or use the validator for multiple validations
let validator = schema::SchemaValidator::new(&schema)?;
validator.validate(&instance)?;
```

#### MCP Integration (NEW)

MCP integration is now available as an optional feature in `siumai-extras`:

```toml
[dependencies]
siumai = { version = "0.11", features = ["openai"] }
siumai-extras = { version = "0.11", features = ["mcp"] }
```

**Quick Start:**
```rust
use siumai::prelude::*;
use siumai_extras::mcp::mcp_tools_from_stdio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Connect to MCP server
    let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;

    // 2. Create model
    let model = Siumai::builder().openai().build().await?;

    // 3. Use with orchestrator
    let (response, _) = siumai_extras::orchestrator::generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        vec![siumai_extras::orchestrator::step_count_is(10)],
        Default::default(),
    ).await?;

    Ok(())
}
```

**Supported Transports:**
- **Stdio**: `mcp_tools_from_stdio("node server.js")` - Local development
- **SSE**: `mcp_tools_from_sse("http://localhost:8080/sse")` - Remote servers
- **HTTP**: `mcp_tools_from_http("http://localhost:3000/mcp")` - Stateless

**Documentation:**
- Integration guide: `siumai/docs/guides/MCP_INTEGRATION.md`
- API reference: `siumai-extras/docs/MCP_FEATURE.md`
- Examples: `siumai/examples/05-integrations/mcp/`

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
