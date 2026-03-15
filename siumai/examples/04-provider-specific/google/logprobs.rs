//! Gemini typed logprobs request options and metadata helpers.
//!
//! Run:
//! ```bash
//! cargo run --example google-logprobs --features google
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::gemini::{
    GeminiChatRequestExt, GeminiChatResponseExt, GeminiClient, GeminiConfig, GeminiOptions,
};

fn read_api_key() -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(k) = std::env::var("GOOGLE_API_KEY") {
        return Ok(k);
    }
    if let Ok(k) = std::env::var("GEMINI_API_KEY") {
        return Ok(k);
    }
    Err("Missing GOOGLE_API_KEY or GEMINI_API_KEY".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = read_api_key()?;

    let client = GeminiClient::from_config(
        GeminiConfig::new(api_key).with_model("gemini-2.0-flash-exp".to_string()),
    )?;

    let request = ChatRequest::new(vec![user!(
        "Answer in one sentence: why are typed provider options useful in SDK design?"
    )])
    .with_gemini_options(
        GeminiOptions::new()
            .with_response_logprobs(true)
            .with_logprobs(3),
    );

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    if let Some(metadata) = response.gemini_metadata() {
        if let Some(avg) = metadata.avg_logprobs {
            println!("Average logprobs: {avg}");
        }

        if let Some(result) = metadata.logprobs_result {
            println!("Logprobs result:");
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("Gemini response did not include logprobs_result metadata.");
            println!("This can happen if the selected model or endpoint ignores logprobs.");
        }
    } else {
        println!("Gemini response did not include Gemini metadata.");
    }

    Ok(())
}
