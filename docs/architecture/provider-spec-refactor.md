# Siumai Architecture & Future Roadmap

This document outlines the current architecture status and future work for the Siumai unified LLM client library.

## ✅ Completed Work

### Phase 1: ProviderSpec Architecture (v0.10-0.11)
**Status**: ✅ **Completed** - Unified HTTP routing, headers, and transformers across all providers via ProviderSpec trait.

### Phase 2: Type-Safe Provider Options (v0.11-0.12)
**Status**: ✅ **Completed** - Replaced `HashMap<String, Value>` with strongly-typed `ProviderOptions` enum, automatic feature injection, and user extensibility via `CustomProviderOptions` trait.

## Current Architecture Overview

```text
User Code → SiumaiBuilder → Provider Client → ProviderSpec
                                      ↘ Executors (HTTP)
                                       ↘ Transformers (Req/Resp/Stream)
                                        ↘ Utilities (Headers/Interceptors/Retry)
```

**Key Components**:
- **ProviderSpec**: Declares HTTP routing, headers, and transformers per provider
- **ProviderOptions**: Type-safe provider-specific configuration (OpenAI, Anthropic, xAI, Gemini, etc.)
- **Executors**: Provider-agnostic HTTP executors for Chat/Embedding/Image
- **Transformers**: Provider-specific request/response/stream mapping

---

## 🚀 Future Work

### Phase 3: OpenAI Responses API Session Management

**Goal**: Implement full session lifecycle management for OpenAI Responses API.

**Current Status**:
- ✅ Basic Responses API routing and configuration via `ResponsesApiConfig`
- ✅ `previous_response_id` and `response_format` injection
- ❌ Session management methods not implemented

**Planned Features**:
```rust
impl OpenAiClient {
    /// Create a new response session
    pub async fn create_response_session(
        &self,
        config: ResponseSessionConfig,
    ) -> Result<ResponseSession, LlmError>;

    /// Chat within an existing session
    pub async fn chat_with_session(
        &self,
        session_id: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<ChatResponse, LlmError>;

    /// Delete a response session
    pub async fn delete_response_session(
        &self,
        session_id: &str,
    ) -> Result<(), LlmError>;

    /// List all active sessions
    pub async fn list_response_sessions(&self) -> Result<Vec<ResponseSession>, LlmError>;
}
```

**Benefits**:
- Automatic session lifecycle management
- Simplified multi-turn conversations with Responses API
- Built-in session state tracking

**Estimated Effort**: 2-3 days

---

### Phase 4: Code Structure Reorganization & Interface Cohesion Review

**Goal**: Reorganize codebase for better maintainability and ensure interface cohesion.

**Current Issues**:
- 📁 Too many scattered files (代码文件有点乱和多)
- 🔍 Need to review if interfaces are properly cohesive (功能是否内聚)
- 🧩 Some modules may have overlapping responsibilities

**Planned Actions**:

#### 4.1 File Structure Audit
- [ ] Map all current modules and their responsibilities
- [ ] Identify duplicate or overlapping functionality
- [ ] Consolidate related utilities into cohesive modules

#### 4.2 Interface Cohesion Review
- [ ] Review all public APIs for single responsibility principle
- [ ] Identify interfaces that do too much or too little
- [ ] Refactor to ensure each module has clear, focused purpose

#### 4.3 Proposed Reorganization
```text
siumai/src/
├── core/                    # Core abstractions (NEW)
│   ├── provider_spec.rs     # ProviderSpec trait
│   ├── provider_context.rs  # ProviderContext
│   ├── executors/           # HTTP executors
│   └── transformers/        # Transformer traits
├── providers/               # Provider implementations
│   ├── openai/
│   ├── anthropic/
│   ├── gemini/
│   └── ...
├── types/                   # Unified types
│   ├── chat.rs
│   ├── embedding.rs
│   ├── provider_options.rs
│   └── ...
├── utils/                   # Utilities (CONSOLIDATE)
│   ├── http/                # HTTP-related utilities
│   │   ├── interceptor.rs
│   │   ├── retry.rs
│   │   └── client.rs
│   ├── middleware/          # Middleware system
│   └── telemetry/           # Telemetry utilities
└── registry/                # Provider registry
```

#### 4.4 Interface Cohesion Principles
1. **Single Responsibility**: Each module should have one clear purpose
2. **High Cohesion**: Related functionality should be grouped together
3. **Low Coupling**: Minimize dependencies between modules
4. **Clear Boundaries**: Well-defined public APIs with minimal surface area

**Estimated Effort**: 1-2 weeks

---

### Phase 5: Additional Provider Capabilities

**Goal**: Extend ProviderSpec to support more capabilities uniformly.

**Planned Extensions**:
- [ ] Files API (OpenAI, Anthropic)
- [ ] Audio API (OpenAI Whisper, TTS)
- [ ] Moderation API (OpenAI)
- [ ] Fine-tuning API (OpenAI, Anthropic)
- [ ] Batch API (OpenAI)

**Estimated Effort**: 3-4 weeks

---

### Phase 6: Provider Feature Validation Matrix

**Goal**: Formalize which features are supported by which providers.

**Planned Features**:
```rust
pub struct ProviderCapabilities {
    pub chat: bool,
    pub streaming: bool,
    pub function_calling: bool,
    pub vision: bool,
    pub embedding: bool,
    pub image_generation: bool,
    pub web_search: bool,
    pub reasoning: bool,
    pub prompt_caching: bool,
}

impl ProviderCapabilities {
    pub fn for_provider(provider_id: &str) -> Self;
    pub fn validate_request(&self, req: &ChatRequest) -> Result<(), LlmError>;
}
```

**Benefits**:
- Compile-time or early runtime validation
- Clear documentation of provider capabilities
- Better error messages when using unsupported features

**Estimated Effort**: 1 week

---

## 📝 Migration Notes

### Deprecated APIs (v0.11 → v0.12)

The following APIs were **completely removed** in v0.11 during the major refactor:

- ❌ `ProviderParams` struct and all related methods
- ❌ `ChatRequest.provider_params` field (replaced by `provider_options`)
- ❌ `ChatRequest.web_search` field (moved to provider-specific options)

**Migration Path**:
```rust
// Old (v0.10)
let req = ChatRequest::new(messages)
    .with_provider_params(
        ProviderParams::new()
            .with_param("search_parameters", json!({...}))
    );

// New (v0.11+)
let req = ChatRequest::new(messages)
    .with_xai_options(
        XaiOptions::new()
            .with_search(XaiSearchParameters { ... })
    );
```

---

## 🎯 Design Principles

### 1. Type Safety First
- Prefer strongly-typed enums over `HashMap<String, Value>`
- Use builder patterns for complex configurations
- Leverage Rust's type system for compile-time validation

### 2. Provider Agnostic Core
- Keep executors and core logic provider-independent
- Isolate provider-specific logic in ProviderSpec implementations
- Use transformers for provider-specific mapping

### 3. Automatic Feature Injection
- Provider features should "just work" when configured
- Minimize manual setup and boilerplate
- Use `ProviderSpec::chat_before_send()` for automatic injection

### 4. User Extensibility
- Support custom provider features via `CustomProviderOptions` trait
- Allow users to extend without library updates
- Provide clear extension points and documentation

### 5. Backward Compatibility (When Possible)
- During major versions (0.x), breaking changes are acceptable
- Deprecated APIs are removed directly in v0.11 (no `#[deprecated]` markers)
- Clear migration guides for breaking changes

---

## 📚 Related Documentation

- [Provider Options API Reference](../api/provider-options.md) (TODO)
- [Custom Provider Extensions Guide](../guides/custom-providers.md) (TODO)
- [Migration Guide v0.10 → v0.11](../migration/v0.10-to-v0.11.md) (TODO)
- [Archived Phase 2 Design](./archived/phase2-provider-options.md) (Completed in v0.11)
