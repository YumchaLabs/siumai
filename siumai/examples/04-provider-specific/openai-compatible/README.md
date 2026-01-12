# OpenAI-Compatible Providers Examples

This directory contains examples for OpenAI-compatible providers that use the unified OpenAI client interface with custom base URLs.

## 🌙 Moonshot AI (Kimi)

Moonshot AI, developed by Dark Side of the Moon, specializes in long-context understanding and Chinese language processing.

### Features

- **Exceptional Long Context**: Up to 256K tokens (Kimi K2) or 128K tokens (V1 models)
- **Bilingual Excellence**: Outstanding performance in both Chinese and English
- **Function Calling**: Full support for OpenAI-compatible tool/function calling
- **Vision Models**: Support for image understanding (V1 Vision models)

### Examples

#### 1. Basic Chat (`moonshot-basic.rs`)

Demonstrates basic usage of Moonshot AI models with different context windows.

```bash
export MOONSHOT_API_KEY="your-api-key-here"
cargo run --example moonshot-basic --features openai
```

**What you'll learn:**
- How to use Kimi K2 (latest model)
- Choosing the right context window model (8K, 32K, 128K)
- Bilingual conversation (Chinese and English)
- Cost-effective model selection

#### 2. Function Calling (`moonshot-tools.rs`)

Shows how to use Moonshot's function calling capabilities.

```bash
export MOONSHOT_API_KEY="your-api-key-here"
cargo run --example moonshot-tools --features openai
```

**What you'll learn:**
- Defining tools/functions for Kimi
- Single and multiple tool calls
- Multi-turn conversations with tool execution
- Tool result handling

**Reference:** https://platform.moonshot.cn/docs/guide/use-kimi-api-to-complete-tool-calls

#### 3. Long Context Processing (`moonshot-long-context.rs`)

Demonstrates Moonshot's exceptional long-context capabilities.

```bash
export MOONSHOT_API_KEY="your-api-key-here"
cargo run --example moonshot-long-context --features openai
```

**What you'll learn:**
- Processing long documents (up to 256K tokens)
- Document summarization
- Multi-turn conversations with long context
- Document Q&A
- Context window comparison

### Getting Started

1. **Get API Key**: Visit [Moonshot Platform](https://platform.moonshot.cn/) to register and get your API key

2. **Set Environment Variable**:
   ```bash
   export MOONSHOT_API_KEY="your-api-key-here"
   ```

3. **Run Examples**:
   ```bash
   cargo run --example moonshot-basic --features openai
   ```

### Model Selection Guide

| Model | Context Window | Best For | Cost |
|-------|---------------|----------|------|
| `kimi-k2-0905-preview` | 256K tokens | Latest features, Agentic Coding | Higher |
| `moonshot-v1-128k` | 128K tokens | Long documents, research papers | Medium |
| `moonshot-v1-32k` | 32K tokens | Long articles, conversations | Medium |
| `moonshot-v1-8k` | 8K tokens | Short chats, quick queries | Lower |

### Model Constants

Use type-safe model constants from the library:

```rust
use siumai::models;

// Latest Kimi K2 model
models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW

// Auto-updated models
models::openai_compatible::moonshot::KIMI_LATEST
models::openai_compatible::moonshot::MOONSHOT_V1_AUTO

// Specific context windows
models::openai_compatible::moonshot::MOONSHOT_V1_8K
models::openai_compatible::moonshot::MOONSHOT_V1_32K
models::openai_compatible::moonshot::MOONSHOT_V1_128K

// Vision models
models::openai_compatible::moonshot::MOONSHOT_V1_128K_VISION
```

### API Compatibility

Moonshot AI is fully compatible with the OpenAI API specification:

- ✅ Chat Completions
- ✅ Streaming
- ✅ Function Calling (Tools)
- ✅ Vision (with Vision models)
- ✅ System Messages
- ✅ Multi-turn Conversations

### Use Cases

**Perfect for:**
- 📚 Long document analysis (research papers, books, legal documents)
- 🇨🇳 Chinese language tasks (translation, writing, understanding)
- 💬 Extended conversations with long context retention
- 🔍 Multi-document comparison and analysis
- 📊 Data extraction from lengthy reports
- 🎓 Academic research and literature review

**Not ideal for:**
- Real-time applications requiring ultra-low latency
- Tasks requiring specialized domain knowledge (use fine-tuned models)
- Image generation (Moonshot focuses on understanding, not generation)

### Tips & Best Practices

1. **Choose the Right Model**:
   - Use `KIMI_K2_0905_PREVIEW` for latest features and best performance
   - Use `MOONSHOT_V1_8K` for cost-effective short conversations
   - Use `MOONSHOT_V1_128K` for long document processing

2. **Optimize Context Usage**:
   - Only include necessary context in your prompts
   - Use summarization for very long documents when possible
   - Consider chunking extremely long documents

3. **Leverage Long Context**:
   - Moonshot excels at maintaining coherence across long conversations
   - Perfect for analyzing entire books or research papers
   - Great for multi-document comparison

4. **Chinese Language**:
   - Moonshot has exceptional Chinese language understanding
   - Ideal for Chinese-English translation and bilingual tasks
   - Excellent for Chinese document analysis

5. **Function Calling**:
   - Moonshot supports OpenAI-compatible function calling
   - Works well with complex tool interactions
   - Good at understanding tool requirements in Chinese

### Resources

- **Official Documentation**: https://platform.moonshot.cn/docs
- **API Reference**: https://platform.moonshot.cn/docs/api-reference
- **Model List**: https://platform.moonshot.cn/docs/intro
- **Pricing**: https://platform.moonshot.cn/docs/pricing
- **Migration Guide**: https://platform.moonshot.cn/docs/guide/migrating-from-openai-to-kimi

### Troubleshooting

**API Key Issues:**
```bash
# Make sure your API key is set correctly
echo $MOONSHOT_API_KEY

# Or set it in your code
let client = Siumai::builder()
    .moonshot()
    .api_key("your-api-key-here")
    .build()
    .await?;
```

**Context Length Errors:**
- Check your input length doesn't exceed the model's context window
- Use a model with larger context window (e.g., V1 128K or Kimi K2)
- Consider summarizing or chunking your input

**Rate Limits:**
- Moonshot has rate limits based on your account tier
- Implement retry logic with exponential backoff
- Contact Moonshot support for higher limits

---

## Adding More Providers

This directory is designed to hold examples for all OpenAI-compatible providers. To add examples for other providers (DeepSeek, OpenRouter, SiliconFlow, etc.), follow the same pattern:

1. Create provider-specific example files
2. Use model constants from `siumai::models::openai_compatible::{provider}`
3. Document provider-specific features and capabilities
4. Include setup instructions and API key information

## Contributing

When adding new examples:
- Follow the existing example structure
- Include clear documentation and comments
- Add setup instructions and prerequisites
- Demonstrate provider-specific features
- Include error handling and best practices
