# Siumai Examples

Welcome to the Siumai examples! This directory contains examples organized by complexity and use case.

## 📚 Directory Structure

```
examples/
├── 01-quickstart/          # Start here! (3 examples)
├── 02-core-api/            # Core API methods (10 examples)
├── 03-advanced-features/   # Advanced features (12 examples)
├── 04-provider-specific/   # Provider-unique features (9 examples)
├── 05-integrations/        # Registry, MCP, telemetry (7 examples)
├── 06-extensibility/       # Custom providers (6 examples)
└── 07-applications/        # Complete applications (3 examples)
```

**Total: ~50 focused examples**

---

## 🚀 Quick Start

### 1. Basic Chat (15 lines)
```bash
cargo run --example basic-chat --features openai
```

### 2. Streaming (30 lines)
```bash
cargo run --example streaming --features openai
```

### 3. Provider Switching (50 lines)
```bash
cargo run --example provider-switching --features "openai,anthropic,google"
```

---

## 📖 Learning Paths

### 🎓 Beginner Path
Start with the basics and build up:

1. **01-quickstart/basic-chat.rs** - Simplest possible usage
2. **01-quickstart/streaming.rs** - Real-time responses
3. **02-core-api/chat/simple-chat.rs** - Using `client.chat()`
4. **02-core-api/chat/chat-request.rs** - Using `chat_request()` (recommended ⭐)
5. **02-core-api/tools/function-calling.rs** - Adding tools

### 💼 Developer Path
For production applications:

1. **02-core-api/chat/chat-request.rs** - Recommended API
2. **03-advanced-features/provider-params/structured-output.rs** - JSON schemas
3. **03-advanced-features/request-building/complex-request.rs** - Full control
4. **03-advanced-features/retry/retry-config.rs** - Resilience
5. **05-integrations/registry/basic-registry.rs** - String-driven models
6. **07-applications/chatbot.rs** - A practical app example

### 🔬 Advanced Path
For complex integrations:

1. **siumai-extras/examples/basic-orchestrator.rs** - Multi-step tool calling
2. **siumai-extras/examples/agent-pattern.rs** - Reusable agents
3. **05-integrations/registry/custom-middleware.rs** - Transform requests
4. **03-advanced-features/middleware/http-interceptor.rs** - Observe traffic
5. **04-provider-specific/anthropic/prompt-caching.rs** - Cost optimization
6. **05-integrations/mcp/stdio-client.rs** - MCP protocol

---

## 📂 Directory Guide

### 01-quickstart/ (Start Here!)
The fastest way to get started. Each example is 15-50 lines.

- **basic-chat.rs** - Simplest chat usage
- **streaming.rs** - Real-time streaming
- **provider-switching.rs** - Unified interface demo

### 02-core-api/ (Core Methods)
Learn the core API methods. Organized by functionality.

**chat/**
- **simple-chat.rs** - `client.chat()`
- **chat-with-tools.rs** - `client.chat_with_tools()`
- **chat-request.rs** - `client.chat_request()` ⭐ Recommended
- **usage-builder-demo.rs** - Usage statistics builder API

**streaming/**
- **basic-stream.rs** - `client.chat_stream()`
- **stream-request.rs** - `client.chat_stream_request()` ⭐ Recommended
- **stream-with-cancel.rs** - Cancellable streams

**tools/**
- **function-calling.rs** - Define and use tools
- **tool-loop.rs** - Complete tool execution cycle
- **tool-choice-demo.rs** - Control tool usage strategies

**multimodal/**
- **vision.rs** - Image understanding

### 03-advanced-features/ (Power Features)
Advanced features for production use.

**provider-params/**
- **structured-output.rs** - Cross-provider JSON schemas
- **reasoning-effort.rs** - Control thinking depth

**request-building/**
- **complex-request.rs** - ChatRequestBuilder with all features

**middleware/**
- **http-interceptor.rs** - Observe requests/responses

**retry/**
- **retry-config.rs** - Handle transient failures

**error-handling/**
- **error-types.rs** - Understanding LlmError

**orchestrator/** (moved to `siumai-extras`)
- Runnable examples live under `siumai-extras/examples/*` (the `siumai` crate keeps stub entry points).

### 04-provider-specific/ (Unique Features)
Provider-specific features that don't have cross-provider equivalents.

**openai/**
- **responses-api.rs** - Stateful conversations
- **responses-multi-turn.rs** - Multi-turn conversations

**anthropic/**
- **extended-thinking.rs** - Deep reasoning
- **prompt-caching.rs** - Cost optimization

**google/**
- **grounding.rs** - Web search integration

**ollama/**
- **local-models.rs** - Run models locally

**xai/**
- **grok.rs** - Using Grok models

**minimaxi/**
- **minimaxi_basic.rs** - Basic chat with MiniMaxi
- **music-generation.md** - Music generation guide
- **video-generation.md** - Video generation guide

### 05-integrations/ (Ecosystem)
Integration with the broader ecosystem.

**registry/**
- **quickstart.rs** - Quick start with registry
- **basic-registry.rs** - String-driven model resolution
- **registry-with-cache.rs** - LRU cache for efficiency
- **custom-middleware.rs** - Transform requests globally

**mcp/**
- **stdio-client.rs** - Model Context Protocol

**server-adapters/** (moved to `siumai-extras`)
- See `siumai-extras` crate for server integration examples

**telemetry/** (moved to `siumai-extras`)
- See `siumai-extras` crate for telemetry examples

### 06-extensibility/ (Custom Providers)
Build your own providers and extend Siumai.

- **complete-custom-provider.rs** - Full custom provider implementation
- **custom-provider-spec.rs** - Custom ProviderSpec example
- **executor-testing.rs** - Testing executors with mocks
- **custom-provider-options.rs** - Custom provider options
- **custom_provider_implementation.rs** - Provider implementation patterns
- **custom_provider_parameters.rs** - Custom parameter handling

### 07-applications/ (Complete Apps)
Full applications demonstrating real-world usage.

- **chatbot.rs** - Interactive conversational AI
- **code-assistant.rs** - AI-powered coding helper
- **api-server.rs** - HTTP API for LLM access

---

## 🎯 Key Concepts

### Unified Interface
Siumai's biggest advantage is the unified interface. The same code works across all providers:

```rust
// Works with OpenAI, Anthropic, Google, Ollama, xAI, Groq
let client = Siumai::builder()
    .openai()  // or .anthropic() / .google() / .ollama()
    .build()
    .await?;

let response = client.chat(vec![user!("Hello!")]).await?;
```

### Recommended API (0.11.0+)
Use `chat_request()` and `chat_stream_request()` for full control:

```rust
use siumai::types::{ProviderOptions, OpenAiOptions};
let request = ChatRequest::builder()
    .message(user!("Hello!"))
    .temperature(0.7)
    .provider_options(ProviderOptions::OpenAi(OpenAiOptions::new()))
    .build();

let response = client.chat_request(request).await?;
```

### Provider Registry
String-driven model selection for dynamic applications:

```rust
let registry = create_registry_with_defaults();
let lm = registry.language_model("openai:gpt-4o-mini")?;
let response = lm.chat(vec![user!("Hello!")]).await?;
```

---

## 🔧 Running Examples

### Prerequisites
Set up API keys:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

Or use Ollama (no API key needed):
```bash
ollama serve
ollama pull llama3.2
```

### Run an Example
```bash
# Basic example
cargo run --example basic-chat --features openai

# With multiple providers
cargo run --example provider-switching --features "openai,anthropic,google"
```

---

## 📊 Example Comparison

| Category | Old Structure | New Structure |
|----------|--------------|---------------|
| **Total Examples** | 54 files | ~33 files |
| **Average Lines** | ~200 lines | ~40 lines |
| **Organization** | By provider + feature | By API functionality |
| **Redundancy** | High (provider × feature) | Low (unified interface) |
| **Learning Curve** | Steep | Gradual |
| **0.11.0 Features** | Not highlighted | Emphasized |

---

## 💡 Tips

1. **Start Simple**: Begin with `01-quickstart/` examples
2. **Use Recommended APIs**: Prefer `chat_request()` over `chat()`
3. **Leverage Unified Interface**: Write provider-agnostic code
4. **Check Provider-Specific**: Only when you need unique features
5. **Explore Complete Apps**: See real-world patterns in `06-applications/`

---

## 🔗 Additional Resources

- **MCP Integration**: See `siumai-extras` crate with `mcp` feature for complete MCP examples
- **Server Adapters**: See `siumai-extras` crate with `server` feature for Axum integration
- **Telemetry**: See `siumai-extras` crate with `telemetry` and `opentelemetry` features for observability

---

## 📝 Contributing

When adding new examples:
1. Keep them focused (one concept per example)
2. Use the recommended APIs (`chat_request`, `chat_stream_request`)
3. Add clear comments and documentation
4. Place in the appropriate directory
5. Update this README

---

**Happy coding with Siumai! 🚀**
