//! OpenAI Responses API (extension API)
//!
//! This example demonstrates the provider extension entry point:
//! `siumai::provider_ext::openai::responses::chat_via_responses_api`.
//!
//! ## Run
//! ```bash
//! cargo run --example responses-ext --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};
use siumai::user;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-5-mini")
        .build()
        .await?;

    let req = ChatRequest::new(vec![user!("用一句话总结 Rust 的优势。")])
        .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    // Option A: explicit extension helper (makes intent obvious)
    let resp = siumai::provider_ext::openai::responses::chat_via_responses_api(
        &client,
        req,
        ResponsesApiConfig::new(),
    )
    .await?;
    println!("resp = {}", resp.content_text().unwrap_or_default());

    Ok(())
}
