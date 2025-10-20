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

use siumai::prelude::*;
use std::fs;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💻 Code Assistant\n");

    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

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
    client: &impl ChatCapability,
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let code = fs::read_to_string(file_path)?;

    let prompt = format!("Explain this code in simple terms:\n\n```\n{}\n```", code);

    let response = client.chat(vec![user!(&prompt)]).await?;
    println!("\n{}\n", response.content_text().unwrap());

    Ok(())
}

async fn review_code(
    client: &impl ChatCapability,
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let code = fs::read_to_string(file_path)?;

    let prompt = format!(
        "Review this code for bugs, performance issues, and best practices:\n\n```\n{}\n```",
        code
    );

    let response = client.chat(vec![user!(&prompt)]).await?;
    println!("\n{}\n", response.content_text().unwrap());

    Ok(())
}
