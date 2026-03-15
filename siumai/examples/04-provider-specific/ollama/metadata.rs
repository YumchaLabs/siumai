//! Ollama typed timing metadata on the registry-first path.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this example: registry-first, because the example focuses on
//!   Stable generation plus typed Ollama response metadata
//!
//! Local runtime:
//! - start Ollama locally with `ollama serve`
//! - ensure the model exists with `ollama pull llama3.2`
//!
//! Run:
//! ```bash
//! ollama serve
//! ollama pull llama3.2
//! cargo run --example ollama-metadata --features ollama
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::ollama::OllamaChatResponseExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = registry::global().language_model("ollama:llama3.2")?;

    let request = ChatRequest::new(vec![user!(
        "Explain Rust ownership in three concise bullet points."
    )]);

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    if let Some(metadata) = response.ollama_metadata() {
        println!("Ollama metrics:");
        if let Some(value) = metadata.tokens_per_second {
            println!("- tokens_per_second: {value:.2}");
        }
        if let Some(value) = metadata.total_duration_ms {
            println!("- total_duration_ms: {value}");
        }
        if let Some(value) = metadata.load_duration_ms {
            println!("- load_duration_ms: {value}");
        }
        if let Some(value) = metadata.prompt_eval_duration_ms {
            println!("- prompt_eval_duration_ms: {value}");
        }
        if let Some(value) = metadata.eval_duration_ms {
            println!("- eval_duration_ms: {value}");
        }
    }

    Ok(())
}
