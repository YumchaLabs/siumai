//! Vertex AI (Gemini) - minimal chat via ADC (Bearer token)
//!
//! This example shows the recommended Vertex setup:
//! - Use a Vertex base URL derived from (project, location, publisher)
//! - Authenticate with ADC (`Authorization: Bearer <token>`)
//!
//! Run:
//! ```bash
//! cargo run --example vertex_chat --features "google gcp"
//! ```

#[cfg(all(feature = "google", feature = "gcp"))]
use siumai::prelude::unified::*;
#[cfg(all(feature = "google", feature = "gcp"))]
use std::sync::Arc;

#[cfg(all(feature = "google", feature = "gcp"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project = std::env::var("GOOGLE_CLOUD_PROJECT")
        .or_else(|_| std::env::var("GCP_PROJECT"))
        .expect("Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT)");
    let location = std::env::var("GOOGLE_CLOUD_LOCATION")
        .or_else(|_| std::env::var("GCP_LOCATION"))
        .unwrap_or_else(|_| "us-central1".to_string());

    // Recommended construction: provider config-first (no unified builder required).
    // Vertex enterprise auth uses ADC (`Authorization: Bearer <token>`).
    let base_url =
        siumai::experimental::auth::vertex::vertex_base_url(&project, &location, "google");
    let adc = siumai::experimental::auth::adc::AdcTokenProvider::default_client();
    let client = siumai::providers::gemini::GeminiClient::from_config(
        siumai::providers::gemini::GeminiConfig::new("")
            .with_base_url(base_url)
            .with_model("gemini-2.0-flash".to_string())
            .with_token_provider(Arc::new(adc))
            .with_provider_metadata_key("vertex"),
    )?;

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!("Say hello from Vertex AI!")]),
        text::GenerateOptions::default(),
    )
    .await?;
    println!("{}", response.content_text().unwrap_or_default());
    Ok(())
}

#[cfg(not(all(feature = "google", feature = "gcp")))]
fn main() {
    eprintln!(
        "This example requires features: google + gcp.\\n\
Run: cargo run --example vertex_chat --features \"google gcp\""
    );
}
