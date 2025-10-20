//! Reasoning Effort - Control thinking depth
//!
//! This example shows how to use ProviderParams to control reasoning effort.
//! Supported by: OpenAI o1/o3 models, Anthropic extended thinking.
//!
//! ## Run
//! ```bash
//! cargo run --example reasoning-effort --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("o1-mini") // Reasoning model
        .build()
        .await?;

    // Set reasoning effort
    let provider_params =
        ProviderParams::new().with_param("reasoning_effort", serde_json::json!("high"));

    let request = ChatRequest::builder()
        .message(user!("Solve this logic puzzle: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?"))
        .provider_params(provider_params)
        .build();

    let response = client.chat_request(request).await?;

    println!("AI (with high reasoning effort):");
    println!("{}", response.content_text().unwrap());

    Ok(())
}
