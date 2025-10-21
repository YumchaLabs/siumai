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

### Phase 3: OpenAI Responses API - Multi-turn Conversation Support

**Status**: ✅ **COMPLETED** - Core functionality implemented

**Implementation Summary**:
- ✅ Full Responses API parameter support (all 12+ parameters)
- ✅ `previous_response_id` for multi-turn conversations
- ✅ `store` parameter for server-side context management
- ✅ `include` parameter for encrypted reasoning content
- ✅ Helper methods: `ChatResponse::response_id()`
- ✅ Example code: `responses-multi-turn.rs`
- ✅ Comprehensive documentation

**Design Philosophy**:

OpenAI Responses API is **stateless by design**, unlike the Assistants API:

| Feature | Assistants API | Responses API |
|---------|---------------|---------------|
| State Management | Stateful (server-side) | Stateless (client-side) |
| Session Storage | Server stores threads | Client manages response IDs |
| Multi-turn | `thread_id` | `previous_response_id` |
| List API | ✅ `GET /threads` | ❌ Not available |
| Delete API | ✅ `DELETE /threads/{id}` | ❌ Not available |
| Retrieve API | ✅ `GET /threads/{id}` | ❌ Not available |

**Usage Pattern**:
```rust
// Turn 1: Initial request
let response1 = client.chat_request(request1).await?;
let response_id = response1.response_id().expect("Response ID not found");

// Turn 2: Follow-up (automatically loads context from Turn 1)
let request2 = ChatRequest::new(messages)
    .with_openai_options(
        OpenAiOptions::new().with_responses_api(
            ResponsesApiConfig::new()
                .with_previous_response(response_id)
                .with_store(true)  // Enable server-side storage
        )
    );
```

**Why No Session Management APIs?**

The following methods are **NOT implemented** (and not needed):
- ❌ `create_response_session()` - No such OpenAI API exists
- ❌ `list_response_sessions()` - No such OpenAI API exists
- ❌ `delete_response_session()` - No such OpenAI API exists

**Rationale**:
1. OpenAI Responses API doesn't provide server-side session management endpoints
2. Implementing these would be misleading and suggest functionality that doesn't exist
3. Session management should be handled at the application layer if needed

**Recommended Approach for Applications**:

If your application needs session management:
1. Store response IDs in your database
2. Associate them with user sessions
3. Pass appropriate `previous_response_id` when needed
4. Implement session lifecycle at the application layer

**Examples**:
- `examples/04-provider-specific/openai/responses-api.rs` - Basic usage
- `examples/04-provider-specific/openai/responses-multi-turn.rs` - Multi-turn conversations

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
