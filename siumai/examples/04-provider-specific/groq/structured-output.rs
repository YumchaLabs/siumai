//! Groq structured output via Stable `response_format`.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this example: registry-first, because the showcased feature is already part of the Stable family story
//!
//! Credentials:
//! - set `GROQ_API_KEY` before running this example
//!
//! Run:
//! ```bash
//! cargo run --example groq-structured-output --features groq
//! ```

use serde_json::json;
use siumai::prelude::unified::*;
use siumai::structured_output::extract_json_value_from_response;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = registry::global().language_model(&format!(
        "groq:{}",
        siumai::models::groq::LLAMA_3_3_70B_VERSATILE
    ))?;

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
            },
            "recommended_action": { "type": "string" }
        },
        "required": ["package_name", "highlights", "risk_level", "recommended_action"],
        "additionalProperties": false
    });

    let request = ChatRequest::new(vec![user!(
        "Generate a release summary for a fictional Rust crate named siumai-runtime and return only structured data."
    )])
    .with_response_format(
        ResponseFormat::json_schema(schema)
            .with_name("release_summary")
            .with_strict(true),
    );

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;
    let value = extract_json_value_from_response(&response)?;

    println!(
        "Raw text:\n{}\n",
        response.content_text().unwrap_or_default()
    );
    println!(
        "Structured JSON:\n{}",
        serde_json::to_string_pretty(&value)?
    );

    Ok(())
}
