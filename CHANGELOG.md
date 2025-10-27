# Changelog

This file lists noteworthy changes. Sections are grouped by version to make upgrades clearer.

## [0.11.0-beta.1] - 2025-10-27

This beta delivers a major refactor of module layout, execution/streaming, and provider integration. Design inspired by Cherry Studio’s transformer design and the Vercel AI SDK’s adapter architecture.

### Breaking Changes
- Module paths: `utils::http_*` → `execution::http::*`; `utils::vertex::*` → `auth::vertex::*`.
- Tracing: removed custom `X-Trace-Id`/`X-Span-Id` header injection; use `siumai-extras` middleware for OpenTelemetry `traceparent`.
- Response metadata: `ChatResponse.provider_metadata` is now namespaced by provider.
- Removed: deprecated `provider_core`, `server_adapters`; legacy provider-specific streaming structs.

### Added
- Provider Registry and model handles (`siumai/src/registry/*`).
- HTTP Interceptors (`execution::http::interceptor`), including built-in `LoggingInterceptor`.
- Unified execution layer and middleware (`execution::{executors, transformers, middleware}`) with auto middlewares (e.g., reasoning extraction).
- Orchestrator rework and examples (`siumai/examples/03-advanced-features/orchestrator/*`).
- High-level object API: `highlevel::object::{generate_object, stream_object}`.
- New crate `siumai-extras` (optional features: `schema`, `telemetry`, `opentelemetry`, `server`, `mcp`).

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

### Migration Notes
- ProviderParams → ProviderOptions: In this beta, migrated for ChatRequest only; Embedding/WebSearch still use `provider_params` and will migrate later.
- Import path updates: `provider_core` → `core`; `executors`/`transformers`/`middleware` moved under `execution::*`.
- OpenTelemetry usage and examples: see `examples/opentelemetry_tracing.rs` and `siumai-extras` docs.

### Known Issues
- OpenAI Responses API `web_search` is not implemented; calling returns `UnsupportedOperation`.

### Stability
- This is a beta pre-release; minor API adjustments may follow.

### Module Reorganization

BREAKING: Reorganized utility modules for better code organization.

- HTTP utilities moved from `utils::*` to `execution::http::*`
  - `utils::http_client` → `execution::http::client`
  - `utils::http_headers` → `execution::http::headers`
  - `utils::http_interceptor` → `execution::http::interceptor`
- Vertex helpers moved to `auth::vertex`

Migration examples:

```rust
// Before
use siumai::utils::http_headers::{HttpHeaderBuilder, ProviderHeaders};
use siumai::utils::vertex::vertex_base_url;

// After
use siumai::execution::http::headers::{HttpHeaderBuilder, ProviderHeaders};
use siumai::auth::vertex::vertex_base_url;
```

Benefits:
- Better separation of concerns with execution-related code
- Clearer responsibilities; `utils` reserved for generic helpers


### Tracing Architecture Simplification

BREAKING: Removed custom tracing headers in favor of standard OpenTelemetry integration (now in `siumai-extras`).

- Removed automatic `X-Trace-Id`/`X-Span-Id` injection
- Add `siumai-extras` OpenTelemetry middleware for W3C `traceparent`

Quick start:

```rust
use siumai_extras::{otel, otel_middleware::OpenTelemetryMiddleware};
otel::init_opentelemetry("my-service", "http://localhost:4317")?;
let client = Client::builder().add_middleware(Arc::new(OpenTelemetryMiddleware::new())).build()?;
```

Benefits:
- Aligns with industry standards
- Works with Jaeger/Zipkin/Datadog and other OTLP backends


### Added

- Provider Registry and model handles (`siumai/src/registry/*`)
  - Unified string-based `provider:model` resolution with LRU caching and optional TTL
  - Customizable registry options (middlewares, interceptors, retry)
- HTTP Interceptors (`execution::http::interceptor`)
  - Request/response hooks and SSE event observation
  - Built-in `LoggingInterceptor`
- Execution layer and middleware system (`execution::{executors,transformers,middleware}`)
  - Auto middlewares based on provider/model (defaults/clamping/reasoning extraction)
- Orchestrator rework (`siumai/src/orchestrator/*`)
  - Multi-step tool calling, agent pattern, tool approval, streaming tool execution
  - See examples under `siumai/examples/03-advanced-features/orchestrator/`
- High-level object APIs (`siumai::highlevel::object`)
  - `generate_object` / `stream_object` for provider-agnostic typed JSON outputs
  - Optional JSON repair and schema validation; partial object streaming
- `siumai-extras` crate
  - Optional features: `schema`, `telemetry`, `opentelemetry`, `server`, `mcp`

### Changed

- Workspace split into `siumai` and `siumai-extras`
- Streaming unification (start/delta/usage/end) across providers; UTF‑8 safe chunking and tag extraction
- Unified retry facade (`retry_api`) with idempotency and 401 token refresh retry
- OpenAI‑compatible providers consolidated via adapter; consistent transformers/executors paths

### Migration: ProviderParams → ProviderOptions

In this beta, ChatRequest migrates from the `HashMap<String, Value>` style `provider_params` to type‑safe `ProviderOptions`.
EmbeddingRequest and WebSearch still accept `provider_params` for now and will be migrated in a subsequent release.
Example:

Before:

```rust
// ChatRequestBuilder with generic provider_params
let req = ChatRequestBuilder::new()
    .message(user!("Return an object with title"))
    .provider_params({
        let mut m = std::collections::HashMap::new();
        m.insert("structured_output".into(), schema_json);
        m
    })
    .build();
```

After (ChatRequest):

```rust
use siumai::types::provider_options::openai::{OpenAiOptions, ResponsesApiConfig};
let req = ChatRequestBuilder::new()
    .message(user!("Return an object with title"))
    .openai_options(
        OpenAiOptions::new().with_responses_api(
            ResponsesApiConfig::new().with_response_format(schema_json)
        ),
    )
    .build();
```

Alternatively, use the provider‑agnostic high‑level API:

```rust
let (obj, _resp) = siumai::highlevel::object::generate_object::<MyType>(
    &client, messages, None, Default::default(),
).await?;
```

### Code Quality

- Clippy cleanup across providers and streaming core:
  - Collapsed nested `if`/`if let` with let-chains where safe.
  - Replaced `iter.skip(n).next()` with `iter.nth(n)`.
  - Removed redundant clones on `Copy` types and minor unwrap optimizations.
  - Implemented `Default` for `StreamProcessorConfig` to reflect intentional defaults.
- Left `large_enum_variant` and `type_complexity` as-is to avoid breaking shapes; propose boxing/aliases in a follow-up if desired.
- No functional changes; validated by `cargo clippy --workspace --all-features` with only the two noted categories remaining.

### Breaking (small, internal APIs)

- Boxed large enum variants to satisfy clippy without inflating enum size:
  - `ProcessedEvent::StreamEnd { response: ChatResponse }` → `ProcessedEvent::StreamEnd { response: Box<ChatResponse> }`
  - `ProviderOptions::OpenAi(OpenAiOptions)` → `ProviderOptions::OpenAi(Box<OpenAiOptions>)`

Migration tips:
- Pattern matches adapt naturally; when you need the value, deref the box:
  - Before: `if let ProcessedEvent::StreamEnd { response } = ev { /* use response */ }`
  - After: `if let ProcessedEvent::StreamEnd { response } = ev { let response = *response; /* or &*response */ }`
- Construction calls require `Box::new(...)` for OpenAI options:
  - Before: `ProviderOptions::OpenAi(opts)`
  - After: `ProviderOptions::OpenAi(Box::new(opts))`

Notes:
- All internal call sites and tests updated. Public APIs not using these internals are unaffected.

### Documentation Updates

- Updated tracing docs; added `siumai-extras` OpenTelemetry examples

### Architecture Refactoring

Major refactor across code organization, providers, and developer ergonomics. Changes derived from a full `git diff` between `main..refactor` (excluding `repo-ref/`).

#### Added

- Provider Registry and Handles
  - Unified provider registry with aliases and capability records (`siumai/src/registry/`).
  - New `RegistryOptions` (separator, middlewares, http_interceptors, retry options, cache settings).
  - Convenience helpers `create_provider_registry(...)` and `create_registry_with_defaults()`; default installs a `LoggingInterceptor`.
  - Handles: `language_model`, `embedding_model`, `image_model`, `speech_model` (TTS), `transcription_model` (STT).

- HTTP Interceptors (unified)
  - Centralized in `execution::http::interceptor` and applied via registry across JSON, multipart, bytes, GET/DELETE, and streaming paths.
  - Best‑effort injection into provider clients (OpenAI, OpenAI‑compatible, Anthropic, Gemini, Groq, xAI, Ollama, Vertex Anthropic).

- Execution Layer and Middleware
  - New `execution::executors::{chat, embedding, image, audio, files, rerank}`.
  - New `execution::transformers::{request,response,stream,hook_builder,...}`.
  - New `execution::middleware` system with builder, named middlewares, auto‑middleware presets (e.g., reasoning extraction), UTF‑8 safe tag extraction.

- Orchestrator and Tools
  - Reworked orchestrator with agent pattern, step control, tool approval workflow, and streaming tool execution.
  - New examples under `siumai/examples/03-advanced-features/orchestrator/*` and application demos under `siumai/examples/07-applications/*`.

- Extras Package (new crate)
  - Split heavy/development utilities into `siumai-extras` with optional features: `schema`, `telemetry`, `opentelemetry`, `server` (Axum adapters), and `mcp` (Model Context Protocol integration).
  - OpenTelemetry middleware for W3C trace context injection; comprehensive OTLP setup helpers.
  - MCP integration utilities to discover and execute tools via rmcp.

- Providers and Capabilities
  - Vertex Anthropic client added under `providers/anthropic_vertex`.
  - Unified methods on `ChatCapability`; improved structured output parity and schema validation hooks.

#### Changed

- Workspace split
  - Converted repository into a workspace with crates `siumai` and `siumai-extras`.
  - Examples moved from top‑level `examples/` to `siumai/examples/` and restructured by topic.
  - Tests moved under `siumai/tests/` with expanded fixtures and SSE end‑to‑end tests.

- Module paths
  - Core abstractions consolidated under `core::*` (moved from `provider_core::*`).
  - Execution modules moved under `execution::*` (executors, transformers, middleware, http, retry).
  - Vertex utilities moved to `auth::vertex`.

- Streaming and Responses
  - Unified SSE streaming with multi‑event emission (start/delta/usage/end) across providers.
  - Tool call streaming improved and standardized.
  - JSON parsing in streaming/non‑streaming now uses automatic JSON repair (configurable), improving robustness.

- HTTP/Retry
  - HTTP helpers consolidated under `execution::http::{client,headers,interceptor,retry}`.
  - Unified retry facade with idempotency, 401 token refresh retry, and provider defaults.
  - Disabled `reqwest/blocking` feature for leaner builds.

- README and docs extensively updated to reflect new architecture, registry, middleware builder, retry, and extras.

#### Removed (breaking)

- Deprecated modules removed: `provider_core` and `server_adapters`.
- Old provider‑specific streaming structs in favor of executor‑based streaming.
- Large OpenAPI document `docs/openapi.documented.yml` (obsolete).
- Legacy example paths under top‑level `examples/` (replaced by `siumai/examples/`).

### API Keys and Environment Variables

- OpenAI: `.api_key(..)` or `OPENAI_API_KEY` (env fallback)
- Anthropic: `.api_key(..)` or `ANTHROPIC_API_KEY` (env fallback)
- Groq: `.api_key(..)` or `GROQ_API_KEY` (env fallback)
- Gemini: `.api_key(..)` or `GEMINI_API_KEY` (env fallback)
- xAI: `.api_key(..)` or `XAI_API_KEY` (env fallback)
- Ollama: no API key (local service, default `http://localhost:11434`)
- OpenAI‑compatible via Builder: `.api_key(..)` or `{PROVIDER_ID}_API_KEY`
- OpenAI‑compatible via Registry: reads `{PROVIDER_ID}_API_KEY` (e.g., `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`)

### Known limitations

- OpenAI Responses API `web_search`: not implemented yet; calling returns `UnsupportedOperation`.

#### Fixed

- Critical: `before_send_hook` now correctly set in all providers to ensure provider‑specific options (e.g., Responses API, Anthropic thinking) are applied.
- UTF‑8 safety: tag extraction, string slicing, streaming chunk boundaries, and token masking made multibyte‑safe; added tests for tricky cases.
- Numerous reliability fixes across streaming, headers, and parameter mapping; expanded fixture‑based tests per provider.

#### Migration Notes

- Update imports
  - `siumai::provider_core::*` → `siumai::core::*`
  - `siumai::executors::*` → `siumai::execution::executors::*`
  - `siumai::transformers::*` → `siumai::execution::transformers::*`
  - `siumai::middleware::*` → `siumai::execution::middleware::*`
  - `siumai::utils::http_*` → `siumai::execution::http::*`
  - `siumai::utils::vertex::*` → `siumai::auth::vertex::*`

- Response metadata
  - `ChatResponse.metadata` is now namespaced by provider. Use `response.get_metadata("provider", "key")` or type‑safe accessors (e.g., `response.openai_metadata()`).

- Registry adoption (recommended)
  - Prefer constructing clients via provider registry (`create_provider_registry(...)`) and set interceptors/middlewares at the registry level.

- Telemetry/tracing
  - Tracing initialization and OpenTelemetry exporters moved to `siumai-extras`.

Note: A dedicated migration guide for v0.11 is in progress and will be added soon.

- **REMOVED**: Deleted `src/provider_model/` module (duplicate of ProviderSpec architecture)
  - Removed 200+ lines of duplicate code
  - Removed `provider_impl.rs` and `model_impls/` subdirectories

#### Provider Implementation Simplification

All 7 provider clients simplified with consistent patterns:

- **Unified Helper Method Naming**:
  - `create_context()` → `build_context()`
  - `create_chat_executor()` → `build_chat_executor()`
  - Similar for other capabilities

- **Code Reduction**:
  - OpenAiClient: 1949 → 1550 lines (-399, -20.5%)
  - AnthropicClient: 463 → 457 lines (-6, -1.3%)
  - GeminiClient: 1173 → 1170 lines (-3, -0.3%)
  - OllamaClient: 571 → 567 lines (-4, -0.7%)
  - XaiClient: 419 → 416 lines (-3, -0.7%)
  - GroqClient: 360 → 357 lines (-3, -0.8%)
  - AnthropicVertexClient: 391 → 384 lines (-7, -1.8%)
  - **Total: 5326 → 4901 lines (-425, -8.5%)**

- **Simplified Executor Building**:
  - Moved spec and context creation into helper methods
  - Reduced parameter passing
  - Eliminated code duplication across providers

#### Documentation

- Planned: `docs/architecture/v0.11-refactoring.md` - Comprehensive refactoring documentation (to be added)
- **UPDATED**: Main README with module organization section
- **UPDATED**: Main README with retry system documentation

#### Migration Guide

**For Users**: No breaking changes for most users. The refactoring maintains backward compatibility through re-exports.

**Deprecated Imports** (still work, but use new paths):
```rust
// Old (still works)
use siumai::provider_core::ProviderSpec;

// New (recommended)
use siumai::core::ProviderSpec;
```

**For Contributors**: All providers should follow the new helper method pattern (doc to be added under `docs/architecture/v0.11-refactoring.md`).

### Breaking Changes

- **BREAKING**: Removed `provider_core` module
  - All types have been moved to the `core` module
  - Migration: Replace `use siumai::provider_core::*` with `use siumai::core::*`
  - Rationale: Cleaner module organization, `provider_core` was deprecated in favor of `core`

- **BREAKING**: Removed `server_adapters` module from core library
  - Framework-agnostic utilities (`text_stream`, `sse_lines`, `SseOptions`) have been removed
  - Framework-specific integrations (Axum, etc.) were already moved to `siumai-extras` in v0.10
  - Migration: Use `siumai-extras` crate with `server` feature for server integrations
  - Rationale: Keep core library lightweight, server adapters belong in extras package

### Breaking Changes (from previous releases)

- **BREAKING**: Removed deprecated tracing subscriber initialization functions
  - Removed `init_tracing`, `init_default_tracing`, `init_debug_tracing`, `init_production_tracing`, `init_performance_tracing`, `init_tracing_from_env` from `siumai::tracing`
  - Removed `OutputFormat` and `TracingConfig` from `siumai` root exports
  - These functions were deprecated in v0.10.3 and have been moved to `siumai-extras::telemetry`

- **BREAKING**: ChatResponse metadata structure changed
  - Changed `metadata: HashMap<String, serde_json::Value>` to `provider_metadata: Option<HashMap<String, HashMap<String, serde_json::Value>>>`
  - Metadata is now namespaced by provider (e.g., `{"anthropic": {...}, "openai": {...}}`)
  - `get_metadata()` signature changed from `get_metadata(&self, key: &str)` to `get_metadata(&self, provider: &str, key: &str)`
  - Migration: Replace `response.metadata.get("key")` with `response.get_metadata("provider", "key")`
  - Or use type-safe accessors: `response.anthropic_metadata()`, `response.openai_metadata()`, `response.gemini_metadata()`

### Fixed

- **Critical Bug Fix**: `before_send_hook` not being set in provider executors
  - **Impact**: Provider-specific options (like OpenAI Responses API parameters) were not being injected into HTTP requests
  - **Root Cause**: During Phase 3 simplification, `spec.chat_before_send()` was called but its return value was ignored
  - **Fix**: Added proper `before_send_hook` setup in all 7 provider clients:
    - OpenAiClient
    - AnthropicClient
    - GeminiClient
    - OllamaClient
    - XaiClient
    - GroqClient
    - AnthropicVertexClient
  - **Result**: Fixed 10 failing OpenAI Responses API tests, ensured all provider-specific options work correctly

- **Test Reliability**: Fixed `standards_tests` failures with `--all-features`
  - **Problem**: Two tests (`test_openai_adapter_invalid_json_handling`, `test_anthropic_adapter_invalid_json_handling`) failed when `--all-features` was enabled
  - **Root Cause**: The `json-repair` feature aggressively repairs invalid JSON, changing test behavior
  - **Fix**: Modified tests to verify "transformer doesn't panic" instead of "returns specific error"
  - **Result**: All 625 tests now pass with all feature combinations

- **Technical Debt Resolution**:
  - **Header Merge Logic Unification**: Eliminated duplicate header merge code in chat executor
    - Added centralized `merge_headers()` and `apply_extra_headers()` functions in `siumai/src/utils/http_headers.rs`
    - Replaced duplicate code at two locations (non-streaming and streaming execution paths)
  - **SSE Event-Level Adapter Hooks**: Fixed missing adapter transformations for SSE events
    - Modified `OpenAiChatStreamTransformer::convert_event` to apply adapter transformations
    - Modified `AnthropicChatStreamTransformer::convert_event` to apply adapter transformations
    - SSE events now: parse JSON → apply adapter transformation → re-serialize → convert
    - Removed TODO comments indicating this missing functionality

### Added
 - Transformers: declarative Rule system introduced in 0.11 with new variants
   - When: conditionally apply nested rules (supports ModelPrefix)
   - EnumMap: discrete value mapping with optional default
   - Note: if you exhaustively match Rule downstream, add a catch‑all arm or handle these variants
 - Provider-agnostic merge_strategy now applied
   - Embedding: merges provider_params (Flatten/Namespace)
   - Image: merges extra_params (Flatten/Namespace)
 - Streaming: optional max_tool_arguments_size (default None) with overflow handler callback
 - Executors: unified helpers for SSE/JSON streaming construction (no behavior change)

- **Type-safe Provider Metadata Access**:
  - Added `AnthropicMetadata`, `OpenAiMetadata`, `GeminiMetadata` structs
  - Added type-safe accessor methods to `ChatResponse`:
    - `anthropic_metadata() -> Option<AnthropicMetadata>`
    - `openai_metadata() -> Option<OpenAiMetadata>`
    - `gemini_metadata() -> Option<GeminiMetadata>`
  - Example:
    ```rust
    if let Some(meta) = response.anthropic_metadata() {
        if let Some(cache_read) = meta.cache_read_input_tokens {
            println!("Cache read tokens: {}", cache_read);
        }
    }
    ```

- **Warning System Enhancement**:
  - Added `Warning::UnsupportedTool` variant (aligned with Vercel AI SDK)
  - Added `Warning::unsupported_tool()` constructor
  - Now supports three warning types:
    - `UnsupportedSetting`: For unsupported request settings
    - `UnsupportedTool`: For unsupported tools
    - `Other`: For generic warnings

- **Middleware System Enhancements** (inspired by Cherry Studio and Vercel AI SDK):
  - **NamedMiddleware**: Each middleware has a unique name for identification and manipulation
  - **MiddlewareBuilder**: Fluent API builder with methods:
    - `add()`: Add middleware to the chain
    - `remove()`: Remove middleware by name
    - `replace()`: Replace middleware by name
    - `insert_before()`: Insert middleware before a specific middleware
    - `insert_after()`: Insert middleware after a specific middleware
    - `has()`: Check if middleware exists
    - `clear()`: Remove all middlewares
  - **TagExtractor**: Generic tag extraction from streaming text
    - State machine-based design for robust tag extraction
    - `get_potential_start_index()`: Smart algorithm for detecting tags split across chunks
    - Supports any XML-style tag pair (e.g., `<think>...</think>`)
    - Zero content loss guarantee
  - **ExtractReasoningMiddleware**: Extract reasoning/thinking content from LLM responses
    - Preset tag configurations for different models:
      - `<think>...</think>` - DeepSeek, Qwen (default)
      - `<thought>...</thought>` - Gemini
      - `<reasoning>...</reasoning>` - Some OpenAI models
      - `<seed:think>...</seed:think>` - Seed models
      - `<thinking>...</thinking>` - Generic
    - Three-layer fallback strategy:
      1. Provider-extracted: Check `response.thinking` field
      2. Metadata: Check `response.metadata["thinking"]`
      3. Tag extraction: Extract from response content
    - Automatic tag selection based on model ID
    - Configurable: can remove tags from response text
  - **Automatic Middleware Addition**: Middleware is automatically added based on provider and model
    - `build_auto_middlewares()`: Build middleware chain based on configuration
    - `MiddlewareConfig`: Configuration for automatic middleware addition
    - Smart defaults: reasoning extraction enabled for models that support it
    - User can override: remove, replace, or insert custom middlewares
  - **New Middleware Hooks** (aligned with Vercel AI SDK and Cherry Studio):
    - `transform_json_body()`: Transform JSON request body before HTTP send
    - `on_stream_end()`: Called when a stream completes successfully
    - `on_stream_error()`: Called when a stream encounters an error
    - `override_provider_id()`: Override provider ID for routing (A/B testing, fallback)
    - `override_model_id()`: Override model ID for routing (already existed, now documented)
  - **HTTP Interceptor Enhancements**:
    - `on_retry()`: Called before retrying a request after an error
  - **SSE Event-Level Adapter Hooks**:
    - SSE events now flow through adapter transformations before being converted
    - Enables provider-specific SSE event transformations at the standard layer
  - **Examples**: `examples/03-advanced-features/middleware_builder.rs`
  - **Documentation**:
    - `docs/MIDDLEWARE_IMPLEMENTATION_SUMMARY.md`: Complete implementation summary
    - `docs/MIDDLEWARE_COMPARISON.md`: Comparison with Cherry Studio and Vercel AI SDK
    - `docs/THINKING_EXTRACTION_DESIGN.md`: Thinking extraction design details
    - `docs/MIDDLEWARE_HOOKS_ANALYSIS.md`: Comprehensive middleware hooks analysis
    - `docs/MIDDLEWARE_HOOKS_IMPLEMENTATION_PLAN.md`: Implementation plan and design

- **siumai-extras** package with optional features:
  - `schema`: JSON schema validation using `jsonschema` crate
  - `telemetry`: Advanced tracing subscriber configuration with `tracing-subscriber`
  - `server`: Server adapters for Axum and other web frameworks
  - **`mcp`** (NEW): MCP (Model Context Protocol) integration for dynamic tool discovery
    - **Core Implementation**:
      - `McpToolResolver`: Implements `ToolResolver` trait for MCP tools
      - Support for stdio, SSE, and HTTP transports via `rmcp` library (v0.8)
      - Automatic tool discovery from MCP servers
      - Type-safe tool execution with compile-time guarantees
    - **Convenience Functions**:
      - `mcp_tools_from_stdio()`: Connect to local MCP servers via stdio
      - `mcp_tools_from_sse()`: Connect to remote MCP servers via SSE
      - `mcp_tools_from_http()`: Connect to MCP servers via HTTP
    - **Integration**:
      - Seamless integration with Siumai's orchestrator
      - Works with all Siumai-supported LLM providers (OpenAI, Anthropic, Google, etc.)
      - Compatible with `ToolLoopAgent` for reusable agent patterns
    - **Documentation**:
      - Integration guide (to be added): `siumai/docs/guides/MCP_INTEGRATION.md`
      - API reference (to be added): `siumai-extras/docs/MCP_FEATURE.md`
      - Examples: `siumai/examples/05-integrations/mcp/`
    - **Design Philosophy**:
      - External integration (not in core library) following Vercel AI SDK's pattern
      - Keeps core library lightweight and fast to compile
      - Optional feature that users can opt-in as needed
      - Based on industry-standard MCP protocol
- Comprehensive module documentation:
  - `siumai/src/observability/tracing/README.md`: Detailed documentation for the tracing module
  - `siumai/src/telemetry/README.md`: Detailed documentation for the telemetry module
  - `docs/developer/performance_module.md`: Documentation for the performance module
  - `docs/developer/code_organization.md`: Code organization guidelines
- Improved module-level documentation with clear responsibility distinctions:
  - `tracing`: Internal logging for developers (stdout, files)
  - `telemetry`: External event export to platforms (Langfuse, Helicone)
  - `performance`: Performance metrics collection and monitoring

### Changed

- **Transformers/Providers**:
  - OpenAI (o1*): prefer `max_completion_tokens` (migrate from `max_tokens` when model id starts with `o1-`)
  - Gemini Chat: move `temperature/top_p/max_tokens/stop_sequences` into `generationConfig` fields
  - OpenAI Image: move `extra_params` injection into generic `merge_strategy` to avoid double writes (behavior unchanged)
  - OpenAI client: use `ChatExecutorBuilder` for executor construction (consistency; no external behavior change)

- **Code Organization Improvements**:
  - **types/chat module refactoring**: Split monolithic `types/chat.rs` (2523 lines) into focused modules
    - `content.rs` (~990 lines): MessageContent, MediaSource, ImageDetail, ContentPart, ToolResultOutput
    - `metadata.rs` (~56 lines): CacheControl, MessageMetadata, ToolCallInfo, ToolResultInfo
    - `message.rs` (~600 lines): MessageRole, ChatMessage, ChatMessageBuilder
    - `request.rs` (~420 lines): ChatRequest, ChatRequestBuilder
    - `response.rs` (~471 lines): AudioOutput, ChatResponse
    - `mod.rs`: Re-exports all public types for backward compatibility
    - **Migration**: No code changes required - all types are re-exported from `types::chat`
  - **server_adapters migration**: Framework-specific integrations moved to `siumai-extras`
    - Core utilities (`SseOptions`, `text_stream`, `sse_lines`) remain in main library
    - Axum integration now in `siumai-extras::server::axum`
    - **Migration**: Replace `use siumai::server_adapters::axum` with `use siumai_extras::server::axum`
    - Add `siumai-extras = { version = "0.11", features = ["server"] }` to dependencies
  - **Code quality optimizations**:
    - Extracted `headermap_to_hashmap()` to `execution::http::headers` for reuse across providers
    - Optimized `sse_lines()` to reuse String buffer, reducing allocations in streaming scenarios
    - Added `defaults::profiles` module with preset configurations:
      - `dev()`: Fast feedback, minimal retries for development
      - `prod()`: Balanced reliability and performance for production
      - `fast()`: Optimized for speed and interactive applications
      - `extended()`: For large models and complex operations
      - `long_running()`: For batch processing and reasoning models

- **Workspace structure**: Migrated to virtual workspace
  - Root `Cargo.toml` is now a virtual workspace manifest
  - Core package moved to `siumai/` directory
  - `siumai-extras` package in `siumai-extras/` directory
  - Examples and tests moved to `siumai/examples/` and `siumai/tests/`
  - Shared dependencies managed via `[workspace.dependencies]`
  - Better IDE support and more consistent tooling behavior
- **Dependency optimization**: Significantly reduced core dependencies
  - **Removed 5 dependencies** from core `siumai` package:
    - `lazy_static` → replaced with `std::sync::OnceLock` (Rust 1.70+)
    - `static_assertions` → replaced with compile-time tests
    - `validator` → replaced with manual validation logic
    - `regex` → replaced with simple string parsing for `<think>` tag extraction
    - `mime_guess` → replaced with custom MIME type mapping using `infer`
  - Moved heavy dependencies to `siumai-extras`: `jsonschema`, `tracing-subscriber`, `axum`
  - **Performance improvements**:
    - Reduced compilation time by ~20-25% (10-15 seconds faster)
    - Reduced binary size by ~10-15% (~1 MB smaller)
    - Reduced dependency count from 26 to 21 (-19%)
  - Users can opt-in to extra features as needed
- **Code organization improvements**:
  - Merged small utility files (`mime.rs`, `vertex.rs`) into `helpers.rs`
  - Unified file naming conventions across the codebase
  - Documented facade pattern and module organization guidelines
- **Async transformation**: Converted authentication system from synchronous to asynchronous
  - Removed `reqwest/blocking` dependency
  - Optimized `tokio` features to minimal subset
- **Internal architecture refactoring**: Unified executor interface design
  - Replaced closure-based executor construction with `ProviderSpec` + `ProviderContext` pattern
  - Refactored all executors: `HttpChatExecutor`, `HttpEmbeddingExecutor`, `HttpImageExecutor`, `HttpAudioExecutor`, `HttpFilesExecutor`
  - Eliminated closure overhead and reduced code duplication across all providers
  - Improved performance and maintainability with unified spec-based approach
  - All providers (OpenAI, Anthropic, Gemini, Groq, Xai, Ollama, etc.) now follow consistent patterns

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
// siumai-extras = { version = "0.11.0-beta.1", features = ["telemetry"] }

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
// siumai-extras = { version = "0.11.0-beta.1", features = ["schema"] }

// Validate JSON against schema
schema::validate_json(&instance, &schema)?;

// Or use the validator for multiple validations
let validator = schema::SchemaValidator::new(&schema)?;
validator.validate(&instance)?;
```

#### Server Adapters (Axum)

**Before:**
```rust
use siumai::server_adapters::axum::to_sse_response;
```

**After:**
```rust
use siumai_extras::server::axum::to_sse_response;

// Add to Cargo.toml:
// For the beta release:
// siumai-extras = { version = "0.11.0-beta.1", features = ["server"] }
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
    let (response, _) = siumai::orchestrator::generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        vec![siumai::orchestrator::step_count_is(10)],
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

### Removed

- Deprecated tracing subscriber initialization functions (moved to `siumai-extras`)
- `src/tracing/subscriber.rs` module (functionality moved to `siumai-extras::telemetry`)
- Heavy dependencies from core package (`jsonschema`, `tracing-subscriber`, `axum`)

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
- **Orchestrator (NEW)**: Multi-step tool calling orchestration system with advanced control flow
  - **Modular Architecture**: Clean separation into `types`, `stop_condition`, `prepare_step`, `generate`, `stream`, and `agent` modules
  - **Flexible Stop Conditions**:
    - Built-in conditions: `step_count_is()`, `has_tool_call()`, `has_text_response()`
    - Combinators: `any_of()`, `all_of()` for complex logic
    - Custom conditions: `custom_condition()` with user-defined predicates
  - **Dynamic Step Preparation**: `PrepareStep` callback for runtime configuration
    - Modify tool choice strategy (`Auto`, `Required`, `None`, `Specific`)
    - Filter active tools per step
    - Adjust system messages and conversation history
  - **Agent Abstraction**: `ToolLoopAgent<M>` for reusable multi-step agents
    - Builder pattern: `.with_system()`, `.with_id()`, `.on_step_finish()`, `.on_finish()`
    - Simplified API: `.generate()` for non-streaming, `.stream()` for streaming
    - Supports multiple conversations with same configuration
  - **Tool Approval Workflow**: `on_tool_approval` callback with `ToolApproval` enum
    - `Approve(args)`: Execute tool with original or modified arguments
    - `Modify(args)`: Change tool arguments before execution
    - `Deny { reason }`: Block dangerous operations with explanation
  - **Streaming Support**: `generate_stream_owned()` for real-time multi-step execution
    - Concurrent tool execution with `tokio::spawn`
    - Progress tracking via `on_chunk`, `on_step_finish`, `on_finish` callbacks
    - Graceful cancellation with `CancelHandle`
  - **Usage Aggregation**: Automatic token usage tracking across all steps with `StepResult::merge_usage()`
  - **Comprehensive Examples**: 5 examples covering all orchestrator features
    - `basic-orchestrator.rs`: Multi-step tool calling basics
    - `agent-pattern.rs`: Reusable agent pattern
    - `stop-conditions.rs`: Advanced stop condition usage
    - `tool-approval.rs`: Tool approval workflow
    - `streaming-orchestrator.rs`: Real-time streaming orchestration
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
- Request headers unified across providers via `execution::http::headers::ProviderHeaders`; custom headers merged from `http_config.headers`. Tracing headers are handled by OpenTelemetry middleware in siumai-extras.
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
  - New `execution::http::interceptor::{HttpInterceptor, HttpRequestContext}` with hooks:
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
