//! Anthropic on Vertex AI - minimal chat via ADC (Bearer token).
//!
//! This example shows the current recommended `Anthropic on Vertex` setup:
//! - use the provider-owned config-first surface
//! - build the Vertex publisher base URL for `anthropic`
//! - pass an ADC-backed token provider and let runtime auth injection handle Bearer headers
//!
//! Run:
//! ```bash
//! cargo run --example vertex_anthropic_chat --features "google-vertex gcp"
//! ```
//!
//! Environment:
//! - `GOOGLE_CLOUD_PROJECT` (or `GCP_PROJECT`)
//! - `GOOGLE_CLOUD_LOCATION` (or `GCP_LOCATION`, default: `us-central1`)
//! - `ANTHROPIC_VERTEX_MODEL` (optional, default: `claude-3-5-sonnet-20241022`)

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::experimental::auth::adc::AdcTokenProvider;
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::prelude::unified::*;
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::provider_ext::anthropic_vertex::{VertexAnthropicClient, VertexAnthropicConfig};
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use std::sync::Arc;

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project = read_non_empty_env("GOOGLE_CLOUD_PROJECT")
        .or_else(|| read_non_empty_env("GCP_PROJECT"))
        .expect("Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT)");
    let location = read_non_empty_env("GOOGLE_CLOUD_LOCATION")
        .or_else(|| read_non_empty_env("GCP_LOCATION"))
        .unwrap_or_else(|| "us-central1".to_string());
    let model = read_non_empty_env("ANTHROPIC_VERTEX_MODEL")
        .unwrap_or_else(|| "claude-3-5-sonnet-20241022".to_string());

    let base_url =
        siumai::experimental::auth::vertex::vertex_base_url(&project, &location, "anthropic");

    let client = VertexAnthropicClient::from_config(
        VertexAnthropicConfig::new(base_url, model)
            .with_token_provider(Arc::new(AdcTokenProvider::default_client())),
    )?;

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!(
            "Explain in three concise bullet points why provider-owned config-first clients are useful in a Rust SDK."
        )]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("{}", response.content_text().unwrap_or_default());
    Ok(())
}

#[cfg(not(all(feature = "google-vertex", feature = "gcp")))]
fn main() {
    eprintln!(
        "This example requires features: google-vertex + gcp.\n\
Run: cargo run --example vertex_anthropic_chat --features \"google-vertex gcp\""
    );
}
