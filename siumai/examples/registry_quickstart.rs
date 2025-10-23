//! Registry Quick Start Example
//!
//! This example demonstrates the recommended way to use Siumai: the Registry pattern.
//! The Registry provides unified access to all providers with automatic caching and middleware.
//!
//! Run with:
//! ```bash
//! # Set your API keys
//! export OPENAI_API_KEY=your-key-here
//! export ANTHROPIC_API_KEY=your-key-here
//! export GEMINI_API_KEY=your-key-here
//!
//! # Run the example
//! cargo run --example registry_quickstart --features all-providers
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Siumai Registry Quick Start\n");

    // Get the global registry
    let reg = registry::global();

    // Example 1: OpenAI
    println!("📝 Example 1: OpenAI GPT-4o-mini");
    #[cfg(feature = "openai")]
    {
        let model = reg.language_model("openai:gpt-4o-mini")?;
        let resp = model
            .chat(vec![user!("Say hello in one sentence")], None)
            .await?;
        println!("Response: {}\n", resp.content_text().unwrap_or_default());
    }

    // Example 2: Anthropic Claude
    println!("📝 Example 2: Anthropic Claude");
    #[cfg(feature = "anthropic")]
    {
        let model = reg.language_model("anthropic:claude-3-5-sonnet-20240620")?;
        let resp = model
            .chat(vec![user!("What is Rust in one sentence?")], None)
            .await?;
        println!("Response: {}\n", resp.content_text().unwrap_or_default());
    }

    // Example 3: Google Gemini
    println!("📝 Example 3: Google Gemini");
    #[cfg(feature = "google")]
    {
        // Note: You can use either "gemini:" or "google:" prefix
        let model = reg.language_model("gemini:gemini-2.0-flash-exp")?;
        let resp = model
            .chat(vec![user!("Explain AI in one sentence")], None)
            .await?;
        println!("Response: {}\n", resp.content_text().unwrap_or_default());
    }

    // Example 4: Streaming
    println!("📝 Example 4: Streaming with OpenAI");
    #[cfg(feature = "openai")]
    {
        use futures::StreamExt;

        let model = reg.language_model("openai:gpt-4o-mini")?;
        let mut stream = model
            .chat_stream(vec![user!("Count from 1 to 5")], None)
            .await?;

        print!("Streaming: ");
        while let Some(event) = stream.next().await {
            if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event {
                print!("{}", delta);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
        }
        println!("\n");
    }

    // Example 5: OpenAI-Compatible Provider (DeepSeek)
    println!("📝 Example 5: OpenAI-Compatible Provider (DeepSeek)");
    #[cfg(feature = "openai")]
    {
        // Note: Requires DEEPSEEK_API_KEY environment variable
        if std::env::var("DEEPSEEK_API_KEY").is_ok() {
            let model = reg.language_model("deepseek:deepseek-chat")?;
            let resp = model.chat(vec![user!("What is your name?")], None).await?;
            println!("Response: {}\n", resp.content_text().unwrap_or_default());
        } else {
            println!("Skipped: DEEPSEEK_API_KEY not set\n");
        }
    }

    // Example 6: Embedding Model
    println!("📝 Example 6: Embedding Model");
    #[cfg(feature = "openai")]
    {
        let model = reg.embedding_model("openai:text-embedding-3-small")?;
        let resp = model.embed(vec!["Hello, world!".to_string()]).await?;
        println!("Embedding dimension: {}\n", resp.embeddings[0].len());
    }

    println!("✅ All examples completed!");
    println!("\n💡 Key Benefits of Registry Pattern:");
    println!("  - Unified access: Same API for all providers");
    println!("  - Auto caching: LRU cache with TTL for better performance");
    println!("  - Auto middleware: Automatic middleware injection based on model");
    println!("  - Easy switching: Change providers by changing the string");

    Ok(())
}
