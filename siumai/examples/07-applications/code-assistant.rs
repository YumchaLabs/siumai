//! Code Assistant - AI-powered coding helper
//!
//! This example demonstrates building a code assistant with:
//! - Code explanation and documentation
//! - Bug detection and fixing
//! - Code review and optimization
//! - Multi-language support
//!
//! ## Run
//! ```bash
//! cargo run --example code-assistant --features openai
//! ```
//!
//! ## Learn More
//! See `siumai/examples/05_use_cases/code_assistant.rs` for the complete
//! implementation with all features.

use siumai::prelude::unified::*;
use siumai::text::TextModelV3;
use std::fs;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💻 Code Assistant\n");

    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let client = registry::global().language_model("openai:gpt-4o-mini")?;

    println!("Commands:");
    println!("  explain <file>  - Explain code");
    println!("  review <file>   - Review code");
    println!("  quit            - Exit\n");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "quit" => break,
            "explain" => {
                if parts.len() < 2 {
                    println!("Usage: explain <file>");
                    continue;
                }
                explain_code(&client, parts[1]).await?;
            }
            "review" => {
                if parts.len() < 2 {
                    println!("Usage: review <file>");
                    continue;
                }
                review_code(&client, parts[1]).await?;
            }
            _ => println!("Unknown command: {}", parts[0]),
        }
    }

    Ok(())
}

async fn explain_code(
    client: &impl TextModelV3,
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let code = fs::read_to_string(file_path)?;

    let prompt = format!("Explain this code in simple terms:\n\n```\n{}\n```", code);

    let response = text::generate(
        client,
        ChatRequest::new(vec![user!(&prompt)]),
        text::GenerateOptions::default(),
    )
    .await?;
    println!("\n{}\n", response.content_text().unwrap_or_default());

    Ok(())
}

async fn review_code(
    client: &impl TextModelV3,
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let code = fs::read_to_string(file_path)?;

    let prompt = format!(
        "Review this code for bugs, performance issues, and best practices:\n\n```\n{}\n```",
        code
    );

    let response = text::generate(
        client,
        ChatRequest::new(vec![user!(&prompt)]),
        text::GenerateOptions::default(),
    )
    .await?;
    println!("\n{}\n", response.content_text().unwrap_or_default());

    Ok(())
}
