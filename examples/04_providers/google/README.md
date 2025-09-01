# 🧠 Google Gemini Provider

This directory contains examples demonstrating Google Gemini's capabilities through the Siumai library. Gemini offers powerful multimodal capabilities and competitive performance.

## 📋 Available Examples

### 🚀 [Basic Usage](./basic_usage.rs)
**Introduction to Google Gemini capabilities**

Features demonstrated:
- Gemini model selection and configuration
- Basic chat functionality with Gemini
- Parameter optimization for different use cases
- Understanding Gemini's strengths and limitations

```bash
# Run the basic usage example
cargo run --example google_basic_usage
```

**Key Learning Points:**
- How to configure Gemini models
- Understanding Gemini's capabilities
- Cost and performance characteristics
- Best practices for Gemini usage

### 🔍 [Embedding with Task Types](./embedding_with_task_types.rs)
**Advanced embedding capabilities with task type optimization**

Features demonstrated:
- Gemini's 8 different task type optimizations
- Provider-specific embedding configurations
- Type-safe configuration methods
- Batch processing with task types
- Performance comparison between task types

```bash
# Run the embedding example
cargo run --example google_embedding_task_types
```

**Key Learning Points:**
- How to optimize embeddings for specific use cases
- Understanding task type benefits
- Provider-specific configuration patterns
- Best practices for embedding quality

## 🚀 Getting Started

### Prerequisites

1. **Get your Google AI API key:**
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create an API key
   - Set the environment variable:
   ```bash
   export GEMINI_API_KEY="your-google-api-key"
   ```

2. **Install dependencies:**
   ```bash
   cargo build --examples
   ```

### Quick Start

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = LlmBuilder::new()
        .google()
        .api_key(&std::env::var("GEMINI_API_KEY")?)
        .model("gemini-1.5-flash")
        .temperature(0.7)
        .build()
        .await?;

    let messages = vec![
        user!("Explain quantum computing in simple terms")
    ];

    let response = ai.chat(messages).await?;
    println!("Gemini: {}", response.content_text().unwrap_or_default());
    
    Ok(())
}
```

## 🎯 Gemini Model Guide

### Available Models

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| **gemini-1.5-flash** | Fast responses, general tasks | ⭐⭐⭐ | ⭐⭐⭐ |
| **gemini-1.5-pro** | Complex reasoning, high quality | ⭐⭐ | ⭐⭐ |

### When to Use Gemini

**Gemini 1.5 Flash:**
- Quick responses and general tasks
- Cost-effective applications
- High-volume processing
- Real-time applications

**Gemini 1.5 Pro:**
- Complex reasoning tasks
- High-quality content generation
- Advanced analysis and research
- Multimodal processing

## 🌟 Gemini's Unique Strengths

### Multimodal Capabilities
- Native support for text, images, and documents
- Excellent vision understanding
- Document analysis and processing
- Code understanding across images

### Performance
- Fast response times
- Competitive pricing
- Good reasoning capabilities
- Large context windows

### Integration
- Google ecosystem integration
- Easy setup and configuration
- Reliable API performance
- Good documentation

## 💰 Cost Optimization

### Model Selection Strategy
```rust
// Use Flash for most tasks
let fast_ai = LlmBuilder::new()
    .google()
    .model("gemini-1.5-flash")
    .max_tokens(500)
    .build().await?;

// Use Pro for complex tasks
let advanced_ai = LlmBuilder::new()
    .google()
    .model("gemini-1.5-pro")
    .max_tokens(1000)
    .build().await?;
```

### Best Practices
- Choose Flash for general tasks
- Use Pro only for complex reasoning
- Monitor token usage
- Implement response caching
- Set appropriate token limits

## 🔗 Related Resources

- [Google AI Documentation](https://ai.google.dev/)
- [Gemini API Reference](https://ai.google.dev/api)
- [Google AI Studio](https://aistudio.google.com/)
- [Pricing Information](https://ai.google.dev/pricing)

## 🎯 Next Steps

1. **Start with Basic Usage**
   - Run the basic usage example
   - Try different models and parameters
   - Experiment with various prompts

2. **Explore Multimodal Features**
   - Try image analysis capabilities
   - Test document processing
   - Experiment with vision tasks

3. **Production Integration**
   - Implement proper error handling
   - Add usage monitoring
   - Set up cost tracking
   - Create fallback mechanisms

---

**Ready to explore Gemini? Start with [basic_usage.rs](basic_usage.rs)! 🚀**
