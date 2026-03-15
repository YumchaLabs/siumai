//! Groq typed logprobs request options and metadata helpers.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this directory: config-first `GroqConfig` / `GroqClient`
//!
//! Run:
//! ```bash
//! cargo run --example groq-logprobs --features groq
//! ```

use siumai::models;
use siumai::prelude::unified::*;
use siumai::provider_ext::groq::{
    GroqChatRequestExt, GroqChatResponseExt, GroqClient, GroqConfig, GroqOptions,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GROQ_API_KEY")?;

    let client = GroqClient::from_config(
        GroqConfig::new(api_key).with_model(models::groq::popular::FLAGSHIP),
    )
    .await?;

    let request = ChatRequest::new(vec![user!(
        "Reply in one short sentence: why are typed provider options useful in SDK design?"
    )])
    .with_groq_options(GroqOptions::new().with_logprobs(true).with_top_logprobs(3));

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    if let Some(metadata) = response.groq_metadata() {
        if let Some(logprobs) = metadata.logprobs {
            println!("Logprobs metadata:");
            println!("{}", serde_json::to_string_pretty(&logprobs)?);
        } else {
            println!("Groq response did not include logprobs metadata.");
            println!("This can happen if the selected model or endpoint ignores logprobs.");
        }
    } else {
        println!("Groq response did not include Groq metadata.");
    }

    Ok(())
}
