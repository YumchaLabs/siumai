//! OpenRouter typed request options and typed response metadata
//! on the generic OpenAI-compatible client.
//!
//! Package tier:
//! - compat vendor view
//! - preferred public story: typed vendor helpers + typed metadata on `OpenAiCompatibleClient`
//! - not a dedicated provider-owned client package
//!
//! Credentials:
//! - reads `OPENROUTER_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export OPENROUTER_API_KEY="your-api-key-here"
//! cargo run --example openrouter-transforms --features openai
//! ```

use siumai::models;
use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;
use siumai::provider_ext::openrouter::{
    OpenRouterChatRequestExt, OpenRouterChatResponseExt, OpenRouterOptions, OpenRouterSourceExt,
    OpenRouterTransform,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAiCompatibleClient::from_builtin_env(
        "openrouter",
        Some(models::openai_compatible::openrouter::openai::GPT_4O),
    )
    .await?;

    let request = ChatRequest::new(vec![user!(
        "Explain in three short bullet points why typed provider extensions are useful in an SDK."
    )])
    .with_openrouter_options(
        OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
    );

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    println!("Answer:\n{}", response.content_text().unwrap_or_default());
    println!(
        "\nRequest-side OpenRouter transforms were configured through typed provider options."
    );

    if let Some(metadata) = response.openrouter_metadata() {
        println!("\nOpenRouter metadata:");
        println!("- logprobs present: {}", metadata.logprobs.is_some());
        println!(
            "- sources returned: {}",
            metadata.sources.as_ref().map(Vec::len).unwrap_or(0)
        );

        if let Some(source_metadata) = metadata
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .and_then(|source| source.openrouter_metadata())
        {
            println!("- first source file_id: {:?}", source_metadata.file_id);
            println!(
                "- first source container_id: {:?}",
                source_metadata.container_id
            );
        }
    }

    Ok(())
}
