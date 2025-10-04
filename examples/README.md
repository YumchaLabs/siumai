# Siumai Examples

Comprehensive examples for the Siumai Rust library, organized by difficulty and use case to help you learn progressively.

## 🚀 Quick Navigation

| I want to... | Example | Difficulty |
|--------------|---------|------------|
| **Get started quickly** | [quick_start.rs](01_getting_started/quick_start.rs) | Beginner |
| **Learn basic chat** | [chat_basics.rs](02_core_features/chat_basics.rs) | Beginner |
| **Compare providers** | [provider_comparison.rs](01_getting_started/provider_comparison.rs) | Beginner |
| **Use streaming** | [streaming_chat.rs](02_core_features/streaming_chat.rs) | Intermediate |
| **Handle errors** | [error_handling.rs](02_core_features/error_handling.rs) | Intermediate |
| **Build a chatbot** | [simple_chatbot.rs](05_use_cases/simple_chatbot.rs) | Intermediate |
| **Use advanced features** | [batch_processing.rs](03_advanced_features/batch_processing.rs) | Advanced |
| **Build production apps** | [api_integration.rs](05_use_cases/api_integration.rs) | Advanced |

## 📁 Directory Structure

### 🌱 [01_getting_started](01_getting_started/) - *First Steps*
Perfect for newcomers to Siumai and LLM development.

- **[quick_start.rs](01_getting_started/quick_start.rs)** - 5-minute introduction with multiple providers
- **[provider_comparison.rs](01_getting_started/provider_comparison.rs)** - Compare OpenAI, Anthropic, and Ollama
- **[embedding.rs](02_core_features/embedding.rs)** - Generate text embeddings
- **[convenience_methods.rs](01_getting_started/convenience_methods.rs)** - Simplified APIs and helpers

### ⚙️ [02_core_features](02_core_features/) - *Essential Skills*
Master the fundamental capabilities every developer needs.

- **[chat_basics.rs](02_core_features/chat_basics.rs)** - Foundation of AI interactions
- **[streaming_chat.rs](02_core_features/streaming_chat.rs)** - Real-time response streaming
- **[unified_interface.rs](02_core_features/unified_interface.rs)** - Provider-agnostic programming
- **[error_handling.rs](02_core_features/error_handling.rs)** - Production-ready error management
- **[capability_detection.rs](02_core_features/capability_detection.rs)** - Feature detection and fallbacks

### 🚀 [03_advanced_features](03_advanced_features/) - *Specialized Capabilities*
Advanced patterns for sophisticated applications.

- **[custom_provider.rs](03_advanced_features/custom_provider.rs)** - Add a custom provider
- **[multimodal_processing.rs](03_advanced_features/multimodal_processing.rs)** - Multimodal inputs and outputs
- **[batch_processing.rs](03_advanced_features/batch_processing.rs)** - High-volume concurrent processing

### 🔌 [04_providers](04_providers/) - *Provider-Specific Features*
Leverage unique capabilities of each AI provider.

| Provider | Strengths | Examples |
|----------|-----------|----------|
| **[OpenAI](04_providers/openai/)** | GPT models, vision, audio | Chat, vision processing, enhanced features |
| **[Anthropic](04_providers/anthropic/)** | Claude models, reasoning | Basic chat, thinking showcase |
| **[Google](04_providers/google/)** | Gemini models | Basic usage |
| **[Ollama](04_providers/ollama/)** | Local models, privacy | Basic setup |
| **[OpenAI Compatible](04_providers/openai_compatible/)** | DeepSeek, Groq, etc. | Models showcase |

### 🎯 [05_use_cases](05_use_cases/) - *Real-World Applications*
Complete applications demonstrating production patterns.

- **[simple_chatbot.rs](05_use_cases/simple_chatbot.rs)** - Interactive chatbot with conversation memory
- **[code_assistant.rs](05_use_cases/code_assistant.rs)** - AI-powered programming helper
- **[content_generator.rs](05_use_cases/content_generator.rs)** - Multi-format content creation tool
- **[api_integration.rs](05_use_cases/api_integration.rs)** - REST API service with AI capabilities

### 🔗 [06_mcp_integration](06_mcp_integration/) - *Model Context Protocol*
Integration with MCP servers for tool calling and external capabilities.

- **[http_mcp_server.rs](06_mcp_integration/http_mcp_server.rs)** - HTTP MCP server implementation
- **[http_mcp_client.rs](06_mcp_integration/http_mcp_client.rs)** - MCP client with LLM integration

## ⚡ Quick Setup

### API Keys
Set environment variables for the providers you want to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
```

### Local Setup (Ollama)

```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull llama3.2
```

### Running Examples

```bash
# Start here - 5 minute introduction
cargo run --example quick_start

# Core functionality
cargo run --example chat_basics
cargo run --example streaming_chat

# Real-world applications
cargo run --example simple_chatbot
cargo run --example api_integration
```

## 🎓 Learning Paths

Choose your path based on your experience and goals:

### 🌱 **Beginner Path** (New to LLMs)

1. [quick_start.rs](01_getting_started/quick_start.rs) - Get your first AI response
2. [provider_comparison.rs](01_getting_started/provider_comparison.rs) - Understand different providers
3. [chat_basics.rs](02_core_features/chat_basics.rs) - Learn conversation fundamentals
4. [simple_chatbot.rs](05_use_cases/simple_chatbot.rs) - Build your first application

### ⚙️ **Developer Path** (Building Applications)

1. [unified_interface.rs](02_core_features/unified_interface.rs) - Provider-agnostic programming
2. [streaming_chat.rs](02_core_features/streaming_chat.rs) - Real-time user experience
3. [error_handling.rs](02_core_features/error_handling.rs) - Production-ready error management
4. [api_integration.rs](05_use_cases/api_integration.rs) - REST API with AI capabilities

### 🚀 **Advanced Path** (Specialized Features)

1. [multimodal_processing.rs](03_advanced_features/multimodal_processing.rs) - Multimodal processing
2. [batch_processing.rs](03_advanced_features/batch_processing.rs) - High-volume processing
3. [Provider-specific examples](04_providers/) - Leverage unique capabilities
4. [MCP integration](06_mcp_integration/) - External tool integration

## 💡 Key Concepts

### Core Components

- **Messages**: `system!()`, `user!()`, `assistant!()` macros for conversation
- **Providers**: OpenAI, Anthropic, Ollama, and OpenAI-compatible services
- **Streaming**: Real-time response processing for better UX
- **Error Handling**: Robust error management for production applications

### Essential Patterns

- **Unified Interface**: Write once, run with any provider
- **Builder Pattern**: Flexible client configuration
- **Async/Await**: Non-blocking operations throughout
- **Environment Variables**: Secure API key management

## 🛡️ Best Practices

1. **Security**: Store API keys in environment variables, never in code
2. **Error Handling**: Always handle errors gracefully with proper fallbacks
3. **Performance**: Use streaming for long responses, implement caching
4. **Cost Management**: Monitor token usage, choose appropriate models
5. **Reliability**: Test with multiple providers, implement retry logic
6. **User Experience**: Provide real-time feedback with streaming responses

## 📚 Additional Resources

- **[Main Documentation](../README.md)** - Library overview and installation
- **[API Reference](https://docs.rs/siumai/)** - Complete API documentation
- **[GitHub Repository](https://github.com/YumchaLabs/siumai)** - Source code and issues
- **[Crates.io](https://crates.io/crates/siumai)** - Package information and versions

---

**Ready to start? Begin with [quick_start.rs](01_getting_started/quick_start.rs) for a 5-minute introduction! 🚀**
