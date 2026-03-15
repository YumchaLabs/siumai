//! Ollama structured output via Stable `response_format`.
//!
//! Package tier:
//! - provider-owned package
//! - preferred path in this example: config-first, because local runtime setup
//!   should stay explicit on `OllamaConfig` / `OllamaClient`
//!
//! Local runtime:
//! - start Ollama locally with `ollama serve`
//! - ensure the model exists with `ollama pull llama3.2`
//!
//! Run:
//! ```bash
//! ollama serve
//! ollama pull llama3.2
//! cargo run --example ollama-structured-output --features ollama
//! ```

use serde::Deserialize;
use serde_json::json;
use siumai::prelude::unified::*;
use siumai::provider_ext::ollama::{OllamaClient, OllamaConfig};
use siumai::structured_output::extract_json_from_response;

#[derive(Debug, Deserialize)]
struct RustSummary {
    topic: String,
    highlights: Vec<String>,
    difficulty: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::from_config(
        OllamaConfig::builder()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .build()?,
    )?;

    let schema = json!({
        "type": "object",
        "properties": {
            "topic": { "type": "string" },
            "highlights": {
                "type": "array",
                "items": { "type": "string" }
            },
            "difficulty": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced"]
            }
        },
        "required": ["topic", "highlights", "difficulty"],
        "additionalProperties": false
    });

    let request = ChatRequest::new(vec![user!(
        "Explain Rust ownership and return only structured data."
    )])
    .with_response_format(
        ResponseFormat::json_schema(schema)
            .with_name("rust_summary")
            .with_strict(true),
    );

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;
    let typed: RustSummary = extract_json_from_response(&response)?;

    println!(
        "Raw text:\n{}\n",
        response.content_text().unwrap_or_default()
    );
    println!("Typed structured output:");
    println!("  topic: {}", typed.topic);
    println!("  difficulty: {}", typed.difficulty);
    println!("  highlights:");
    for highlight in typed.highlights {
        println!("  - {highlight}");
    }

    Ok(())
}
