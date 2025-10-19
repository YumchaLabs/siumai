// Basic example for using the experimental Provider Registry
// Requires the `openai` feature to run the OpenAI path.
// For OpenAI-Compatible (e.g., OpenRouter), set corresponding env var.

// To run (with OpenAI):
//   OPENAI_API_KEY=sk-... cargo run --example registry_basic --features openai
// To run (with OpenRouter):
//   OPENROUTER_API_KEY=sk-... cargo run --example registry_basic --features openai

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a registry with reasonable defaults:
    // - separator ':'
    // - language model middlewares to fill default temperature and clamp top_p
    let reg = siumai::registry::helpers::create_registry_with_defaults();

    // Example A: OpenAI (requires OPENAI_API_KEY)
    #[cfg(feature = "openai")]
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let lm = reg.language_model("openai:gpt-4o")?;
        let resp = lm
            .chat(vec![user!("Hello from registry (OpenAI)!")], None)
            .await?;
        println!("OpenAI> {}", resp.content_text().unwrap_or("<no text>"));
    } else {
        eprintln!("[note] set OPENAI_API_KEY to run the OpenAI example");
    }

    // Example B: OpenAI-Compatible via OpenRouter (requires OPENROUTER_API_KEY)
    // You can change the model id to another OpenRouter-supported model.
    #[cfg(feature = "openai")]
    if std::env::var("OPENROUTER_API_KEY").is_ok() {
        let lm = reg.language_model("openrouter:openai/gpt-4o-mini")?;
        let resp = lm
            .chat(vec![user!("Hello from registry (OpenRouter)!")], None)
            .await?;
        println!("OpenRouter> {}", resp.content_text().unwrap_or("<no text>"));
    } else {
        eprintln!("[note] set OPENROUTER_API_KEY to run the OpenRouter example");
    }

    Ok(())
}
