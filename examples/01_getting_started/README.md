# Getting Started

Welcome to Siumai! These examples will help you get up and running quickly with the unified LLM interface.

## Examples

### [quick_start.rs](quick_start.rs)
**5-minute introduction to Siumai**

The fastest way to get started. Shows basic usage with OpenAI, Anthropic, and Ollama providers.

```bash
cargo run --example quick_start
```

**What you'll learn:**
- How to create LLM clients
- Basic message creation
- Simple chat interactions
- Environment variable setup

### [provider_comparison.rs](provider_comparison.rs)
**Understanding different AI providers**

Compare performance, costs, and capabilities across different providers.

```bash
cargo run --example provider_comparison
```

**What you'll learn:**
- Provider strengths and weaknesses
- Performance characteristics
- Cost considerations
- Use case recommendations

### [chat_basics.rs](../02_core_features/chat_basics.rs)
**Core concepts and message types**

Deep dive into the fundamental concepts of the library.

```bash
cargo run --example chat_basics
```

**What you'll learn:**
- Message types (system, user, assistant)
- Multimodal messages
- Parameter configuration
- Error handling basics

### [convenience_methods.rs](convenience_methods.rs)
**Simplified APIs for common tasks**

Learn about helper methods that make common tasks easier.

```bash
cargo run --example convenience_methods
```

**What you'll learn:**
- Quick client creation
- Preset configurations
- Message macros
- Common patterns

## Setup

Before running these examples, you'll need to set up API keys:

```bash
# Choose one or more providers
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"

# For Ollama (local)
ollama serve
ollama pull llama3.2
```

## Learning Path

1. **Start here**: [quick_start.rs](quick_start.rs)
   - Get your first AI response in minutes
   - Test different providers

2. **Understand providers**: [provider_comparison.rs](provider_comparison.rs)
   - Learn which provider to choose
   - Understand cost implications

3. **Learn the basics**: [chat_basics.rs](../02_core_features/chat_basics.rs)
   - Master message types
   - Understand configuration options

4. **Explore convenience**: [convenience_methods.rs](convenience_methods.rs)
   - Discover shortcuts and helpers
   - Learn common patterns

## Common Issues

### "Authentication Error"
- Check that your API key is set correctly
- Verify the API key is valid and has credits
- Make sure you're using the right environment variable name

### "Model not found"
- Verify the model name is correct
- Check if you have access to the model
- Try a different model (e.g., gpt-4o-mini instead of gpt-4)

### "Connection refused" (Ollama)
- Make sure Ollama is running: `ollama serve`
- Check if the model is installed: `ollama list`
- Pull the model if needed: `ollama pull llama3.2`

## Key Concepts

### Message Types
```rust
use siumai::prelude::*;

// System message - sets AI behavior
let system_msg = system!("You are a helpful assistant");

// User message - human input
let user_msg = user!("Hello!");

// Assistant message - AI response (for conversation history)
let assistant_msg = assistant!("Hi there!");
```

### Provider Creation
```rust
// OpenAI
let client = LlmBuilder::new()
    .openai()
    .api_key("your-key")
    .model("gpt-4o-mini")
    .temperature(0.7)
    .build()
    .await?;
```

### Basic Chat
```rust
let messages = vec![user!("What is Rust?")];
let response = client.chat(messages).await?;

if let Some(text) = response.content_text() {
    println!("AI: {}", text);
}
```

## Next Steps

Once you're comfortable with these basics, explore:

- **[Core Features](../02_core_features/)**: Streaming, error handling, advanced patterns
- **[Provider Features](../04_providers/)**: Provider-specific capabilities
- **[Use Cases](../05_use_cases/)**: Complete application examples

## Tips for Success

1. **Start simple**: Begin with `quick_start.rs` and a single provider
2. **Use environment variables**: Keep API keys secure
3. **Handle errors**: Always wrap API calls in proper error handling
4. **Experiment**: Try different models and parameters
5. **Read the docs**: Each example has detailed comments explaining concepts

## Getting Help

- Check the [main README](../../README.md) for library documentation
- Look at other examples for more advanced patterns
- Review error messages carefully - they often contain helpful hints
- Try different providers if one isn't working
