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
use siumai::prelude::*;

#[cfg(all(feature = "google", feature = "gcp"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project = std::env::var("GOOGLE_CLOUD_PROJECT")
        .or_else(|_| std::env::var("GCP_PROJECT"))
        .expect("Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT)");
    let location = std::env::var("GOOGLE_CLOUD_LOCATION")
        .or_else(|_| std::env::var("GCP_LOCATION"))
        .unwrap_or_else(|_| "us-central1".to_string());

    let client = Siumai::builder()
        .gemini()
        .base_url_for_vertex(&project, &location, "google")
        .with_gemini_adc()
        .model("gemini-2.0-flash")
        .build()
        .await?;

    let response = client
        .chat(vec![user!("Say hello from Vertex AI!")])
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
