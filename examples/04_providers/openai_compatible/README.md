# 🔌 OpenAI-Compatible Providers

This directory contains examples demonstrating OpenAI-compatible providers through the Siumai library. These providers offer OpenAI-compatible APIs with different models, pricing, and capabilities.

## 📋 Available Examples

### 🚀 [Models Showcase](./models_showcase.rs)
**Exploring different OpenAI-compatible providers and models**

Features demonstrated:
- DeepSeek models and capabilities
- Groq high-speed inference
- Other OpenAI-compatible providers
- Performance and cost comparisons
- Provider-specific optimizations

```bash
# Run the models showcase example
cargo run --example openai_compatible_models_showcase
```

**Key Learning Points:**
- How to configure different compatible providers
- Understanding each provider's strengths
- Cost and performance trade-offs
- When to use which provider

### 🎯 [SiliconFlow Comprehensive](./siliconflow.rs)
**Complete showcase of SiliconFlow's AI capabilities**

Features demonstrated:
- Chat with DeepSeek and Qwen models
- Text embeddings with BGE models
- Document reranking for improved search relevance
- Complete RAG (Retrieval-Augmented Generation) workflow
- Multi-capability integration

```bash
# Run the SiliconFlow example
cargo run --example siliconflow
```

**Key Learning Points:**
- How to use SiliconFlow's reranking capabilities
- Building complete RAG systems
- Combining chat, embeddings, and reranking
- Advanced document relevance scoring

### 🧠 [Adapter System Showcase](./adapter_system_showcase.rs)
**Comprehensive OpenAI-compatible adapter system demonstration**

Features demonstrated:
- Provider adapter architecture and capabilities
- Parameter transformation (thinking_budget → reasoning_effort)
- Field mapping (reasoning_content → thinking)
- Model-specific configurations and optimizations
- Real API integration with DeepSeek thinking
- Complete workflow from architecture to production usage

```bash
# Run the adapter system showcase
cargo run --example adapter_system_showcase

# With API key for real integration testing
SILICONFLOW_API_KEY="your-key" cargo run --example adapter_system_showcase
```

**Key Learning Points:**
- Understanding the new adapter architecture
- How parameter transformation works automatically
- Field mapping for different provider formats
- Model-specific optimizations and configurations
- Real-world integration patterns with thinking capabilities
- Production-ready usage examples

## 🚀 Getting Started

### Prerequisites

Set up API keys for the providers you want to use:

```bash
# DeepSeek (cost-effective, good performance)
export DEEPSEEK_API_KEY="your-deepseek-key"

# SiliconFlow (comprehensive AI platform with reranking)
export SILICONFLOW_API_KEY="your-siliconflow-key"

# Groq (ultra-fast inference)
export GROQ_API_KEY="your-groq-key"

# Together AI (diverse model selection)
export TOGETHER_API_KEY="your-together-key"

# Perplexity (search-augmented responses)
export PERPLEXITY_API_KEY="your-perplexity-key"
```

### Quick Start

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // DeepSeek - cost-effective
    let deepseek = LlmBuilder::new()
        .deepseek()
        .api_key(&std::env::var("DEEPSEEK_API_KEY")?)
        .model("deepseek-chat")
        .build()
        .await?;

    // OpenRouter - access to multiple models
    let openrouter = LlmBuilder::new()
        .openrouter()
        .api_key(&std::env::var("OPENROUTER_API_KEY")?)
        .model("openai/gpt-4")
        .build()
        .await?;

    let messages = vec![user!("Explain the benefits of AI")];

    let response = deepseek.chat(messages.clone()).await?;
    println!("DeepSeek: {}", response.content_text().unwrap_or_default());

    Ok(())
}
```

## 🎯 Provider Comparison

### DeepSeek
**Strengths:** Very cost-effective, good performance, reasoning capabilities
**Best for:** Cost-sensitive applications, general tasks, coding assistance
**Models:** deepseek-chat, deepseek-coder

```rust
let deepseek = LlmBuilder::new()
    .deepseek()
    .api_key(&deepseek_key)
    .model("deepseek-chat")
    .build().await?;
```

### OpenRouter
**Strengths:** Access to multiple AI models, unified API, model routing
**Best for:** Accessing various models through one API, fallback strategies
**Models:** openai/gpt-4, anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-70b

```rust
let openrouter = LlmBuilder::new()
    .openrouter()
    .api_key(&openrouter_key)
    .model("openai/gpt-4")
    .build().await?;
```

### Other OpenAI-Compatible Providers
For other providers, you can use the OpenAI client with custom base URLs:

```rust
// Groq - ultra-fast inference
let groq = LlmBuilder::new()
    .openai()
    .base_url("https://api.groq.com/openai/v1")
    .api_key(&groq_key)
    .model("llama-3.1-70b-versatile")
    .build().await?;

// Together AI - diverse model selection
let together = LlmBuilder::new()
    .openai()
    .base_url("https://api.together.xyz/v1")
    .api_key(&together_key)
    .model("meta-llama/Llama-3-70b-chat-hf")
    .build().await?;

// Perplexity - search-augmented responses
let perplexity = LlmBuilder::new()
    .openai()
    .base_url("https://api.perplexity.ai")
    .api_key(&perplexity_key)
    .model("llama-3.1-sonar-small-128k-online")
    .build().await?;
```

## 🌟 Key Benefits

### Cost Effectiveness
- Often significantly cheaper than OpenAI
- Competitive pricing models
- Good performance per dollar

### Speed & Performance
- Some providers offer ultra-fast inference (Groq)
- Optimized infrastructure
- Low latency options

### Model Diversity
- Access to various open-source models
- Different model architectures
- Specialized models for specific tasks

### Innovation
- Cutting-edge features and capabilities
- Rapid deployment of new models
- Experimental features

## ⚙️ Configuration Best Practices

### Provider Selection Strategy

```rust
// For cost-sensitive applications
let cost_effective = LlmBuilder::new()
    .deepseek()
    .model("deepseek-chat")
    .temperature(0.7)
    .build().await?;

// For speed-critical applications
let ultra_fast = LlmBuilder::new()
    .openai()
    .base_url("https://api.groq.com/openai/v1")
    .model("llama-3.1-70b-versatile")
    .temperature(0.3)
    .build().await?;

// For research and current information
let research_focused = LlmBuilder::new()
    .openai()
    .base_url("https://api.perplexity.ai")
    .model("llama-3.1-sonar-small-128k-online")
    .temperature(0.1)
    .build().await?;
```

### Fallback Strategy

```rust
async fn chat_with_fallback(message: &str) -> Result<String, Box<dyn std::error::Error>> {
    let providers = vec![
        ("Groq", groq_client),
        ("DeepSeek", deepseek_client),
        ("Together", together_client),
    ];
    
    for (name, client) in providers {
        match client.chat(vec![user!(message)]).await {
            Ok(response) => {
                println!("Success with {}", name);
                return Ok(response.content_text().unwrap_or_default().to_string());
            }
            Err(e) => {
                println!("Failed with {}: {}", name, e);
                continue;
            }
        }
    }
    
    Err("All providers failed".into())
}
```

## 💰 Cost Optimization

### Tips for Cost Management

1. **Choose the right provider for your use case**
2. **Monitor token usage across providers**
3. **Implement response caching**
4. **Use smaller models when possible**
5. **Set appropriate token limits**

### Performance vs. Cost Trade-offs

- **DeepSeek**: Best cost/performance ratio
- **Groq**: Premium for speed
- **Together**: Good balance with model variety
- **Perplexity**: Premium for search capabilities

## 🔧 Troubleshooting

### Common Issues

**API Key Issues**
- Verify API key is correct and active
- Check if you have sufficient credits
- Ensure proper environment variable setup

**Model Availability**
- Some models may not be available in all regions
- Check provider documentation for model lists
- Verify model name spelling

**Rate Limiting**
- Different providers have different rate limits
- Implement proper retry logic
- Consider upgrading plans for higher limits

## 🔗 Related Resources

- [DeepSeek Documentation](https://platform.deepseek.com/api-docs/)
- [Groq Documentation](https://console.groq.com/docs/)
- [Together AI Documentation](https://docs.together.ai/)
- [Perplexity Documentation](https://docs.perplexity.ai/)

## 🎯 Next Steps

1. **Explore Providers**
   - Sign up for different provider accounts
   - Get API keys and test basic functionality
   - Compare performance and costs

2. **Model Experimentation**
   - Try different models from each provider
   - Compare quality and speed
   - Find the best fit for your use case

3. **Production Integration**
   - Implement fallback strategies
   - Set up monitoring and alerting
   - Optimize for cost and performance

---

**Ready to explore OpenAI-compatible providers? Start with [models_showcase.rs](models_showcase.rs)! 🔌**
