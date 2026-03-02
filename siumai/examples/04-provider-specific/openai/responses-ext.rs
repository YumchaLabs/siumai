//! OpenAI Responses API (extension API)
//!
//! This example demonstrates the provider extension entry point:
//! `siumai::provider_ext::openai::ext::responses::chat_via_responses_api`.
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
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-5-mini")?;

    let req = ChatRequest::new(vec![user!("用一句话总结 Rust 的优势。")])
        .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    // Option A: explicit extension helper (makes intent obvious)
    let resp = siumai::provider_ext::openai::ext::responses::chat_via_responses_api(
        &model,
        req,
        ResponsesApiConfig::new(),
    )
    .await?;
    println!("resp = {}", resp.content_text().unwrap_or_default());

    Ok(())
}
