//! xAI structured output via Stable `response_format`.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this example: config-first, because xAI-specific setup
//!   should still flow through `XaiConfig` / `XaiClient`
//!
//! Credentials:
//! - set `XAI_API_KEY` before running this example
//!
//! Run:
//! ```bash
//! cargo run --example xai-structured-output --features xai
//! ```

use serde::Deserialize;
use serde_json::json;
use siumai::prelude::unified::*;
use siumai::provider_ext::xai::{XaiClient, XaiConfig};
use siumai::structured_output::extract_json_from_response;

#[derive(Debug, Deserialize)]
struct ReleaseSummary {
    package_name: String,
    highlights: Vec<String>,
    risk_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("XAI_API_KEY")?;
    let client = XaiClient::from_config(
        XaiConfig::new(api_key).with_model(siumai::models::xai::grok_4::GROK_4_LATEST),
    )
    .await?;

    let schema = json!({
        "type": "object",
        "properties": {
            "package_name": { "type": "string" },
            "highlights": {
                "type": "array",
                "items": { "type": "string" }
            },
            "risk_level": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            }
        },
        "required": ["package_name", "highlights", "risk_level"],
        "additionalProperties": false
    });

    let request = ChatRequest::new(vec![user!(
        "Summarize a fictional Rust crate release for siumai-xai-runtime and return only structured data."
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
    println!("  package_name: {}", typed.package_name);
    println!("  risk_level: {}", typed.risk_level);
    println!("  highlights:");
    for highlight in typed.highlights {
        println!("  - {highlight}");
    }

    Ok(())
}
