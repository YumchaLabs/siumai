//! Anthropic on Vertex AI - native structured output via `output_format`.
//!
//! This example shows the current recommended `Anthropic on Vertex` structured-output setup:
//! - use the provider-owned config-first surface
//! - use a 4.5 Anthropic model that supports native `output_format`
//! - keep Stable `response_format` at the request layer
//! - extract typed JSON from the final unified response
//!
//! Run:
//! ```bash
//! cargo run --example vertex_anthropic_structured_output --features "google-vertex gcp"
//! ```
//!
//! Environment:
//! - `GOOGLE_CLOUD_PROJECT` (or `GCP_PROJECT`)
//! - `GOOGLE_CLOUD_LOCATION` (or `GCP_LOCATION`, default: `us-central1`)
//! - `ANTHROPIC_VERTEX_MODEL` (optional, default: `claude-sonnet-4-5-latest`)

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use serde::Deserialize;
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use serde_json::json;
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::experimental::auth::adc::AdcTokenProvider;
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::prelude::unified::*;
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::provider_ext::anthropic_vertex::{VertexAnthropicClient, VertexAnthropicConfig};
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::structured_output::extract_json_from_response;
#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use std::sync::Arc;

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
#[derive(Debug, Deserialize)]
struct ReleaseSummary {
    service: String,
    rollout_risk: String,
    actions: Vec<String>,
}

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
        .unwrap_or_else(|| "claude-sonnet-4-5-latest".to_string());

    let base_url =
        siumai::experimental::auth::vertex::vertex_base_url(&project, &location, "anthropic");

    let client = VertexAnthropicClient::from_config(
        VertexAnthropicConfig::new(base_url, model)
            .with_token_provider(Arc::new(AdcTokenProvider::default_client())),
    )?;

    let schema = json!({
        "type": "object",
        "properties": {
            "service": { "type": "string" },
            "rollout_risk": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            },
            "actions": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["service", "rollout_risk", "actions"],
        "additionalProperties": false
    });

    let request = ChatRequest::new(vec![user!(
        "Return a fictional release note summary for the service siumai-anthropic-vertex-runtime. Respond only with structured data."
    )])
    .with_response_format(
        ResponseFormat::json_schema(schema)
            .with_name("release_summary")
            .with_strict(true),
    );

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;
    let typed: ReleaseSummary = extract_json_from_response(&response)?;

    println!(
        "Raw text:\n{}\n",
        response.content_text().unwrap_or_default()
    );
    println!("Typed structured output:");
    println!("  service: {}", typed.service);
    println!("  rollout_risk: {}", typed.rollout_risk);
    println!("  actions:");
    for action in typed.actions {
        println!("  - {action}");
    }

    Ok(())
}

#[cfg(not(all(feature = "google-vertex", feature = "gcp")))]
fn main() {
    eprintln!(
        "This example requires features: google-vertex + gcp.\n\
Run: cargo run --example vertex_anthropic_structured_output --features \"google-vertex gcp\""
    );
}
