//! Structured Output - Cross-provider JSON schema support
//!
//! This example demonstrates the Stable structured-output surface:
//! `ChatRequest::with_response_format(...)` plus typed JSON extraction.
//! The same request shape is meant to work across providers even when the
//! provider-specific wire format differs internally.
//!
//! ## Run
//! ```bash
//! cargo run --example structured-output --features openai
//! ```

use serde::Deserialize;
use serde_json::json;
use siumai::prelude::unified::*;
use siumai::structured_output::extract_json_from_response;

#[derive(Debug, Deserialize)]
struct PersonProfile {
    name: String,
    age: u32,
    hobbies: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    println!("📝 Structured Output Example\n");

    // Define JSON schema
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "hobbies"],
        "additionalProperties": false
    });

    let request = ChatRequest::new(vec![user!(
        "Generate a profile for a fictional software engineer and return only structured data."
    )])
    .with_response_format(
        ResponseFormat::json_schema(schema)
            .with_name("person_info")
            .with_strict(true),
    );

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;
    let typed: PersonProfile = extract_json_from_response(&response)?;

    println!("✅ Raw Structured Output:");
    println!("{}", response.content_text().unwrap_or_default());
    println!();

    println!("✅ Typed Extraction:");
    println!("  name: {}", typed.name);
    println!("  age: {}", typed.age);
    println!("  hobbies:");
    for hobby in typed.hobbies {
        println!("  - {hobby}");
    }

    println!();
    println!("💡 Stable API:");
    println!("   Request: ChatRequest::with_response_format(ResponseFormat::json_schema(...))");
    println!("   Parse:   siumai::structured_output::extract_json_from_response::<T>(...)");
    println!("   Benefit: one cross-provider public story, provider mapping stays internal");

    Ok(())
}
