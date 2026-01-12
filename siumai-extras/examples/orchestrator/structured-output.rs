//! Orchestrator agent with structured output example.
//!
//! This example demonstrates how to use the ToolLoopAgent with structured output
//! to extract JSON data from the model's response.
//!
//! Run with:
//! ```bash
//! cargo run -p siumai-extras --example orchestrator_structured_output --features openai
//! ```

use serde_json::json;
use siumai::prelude::Siumai;
use siumai::prelude::unified::{ChatMessage, OutputSchema};
use siumai_extras::orchestrator::{ToolLoopAgent, step_count_is};

// Simple tool resolver that doesn't actually execute tools
struct DummyResolver;

#[async_trait::async_trait]
impl siumai_extras::orchestrator::ToolResolver for DummyResolver {
    async fn call_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::prelude::unified::LlmError> {
        Ok(json!({"status": "ok"}))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the model via the unified Siumai builder
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Define the output schema
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's full name"
            },
            "age": {
                "type": "number",
                "description": "The person's age in years"
            },
            "occupation": {
                "type": "string",
                "description": "The person's occupation"
            },
            "hobbies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of hobbies"
            }
        },
        "required": ["name", "age"]
    });

    // Create an agent with structured output
    let agent = ToolLoopAgent::new(client.clone(), vec![], vec![step_count_is(5)])
        .with_system(
            "You are a helpful assistant that extracts person information. \
             Always respond with valid JSON matching the requested schema.",
        )
        .with_output_schema(
            OutputSchema::new(schema.clone())
                .with_name("person_info")
                .with_description("Information about a person"),
        )
        .with_temperature(0.7);

    println!("🤖 Agent with Structured Output Example\n");
    println!("📋 Expected Schema:");
    println!("{}\n", serde_json::to_string_pretty(&schema)?);

    // Example 1: Extract person info from text
    println!("{}", "=".repeat(80));
    println!("Example 1: Extract person information");
    println!("{}", "=".repeat(80));

    let messages = vec![
        ChatMessage::user(
            "Extract information about this person: \
         John Smith is a 35-year-old software engineer who enjoys hiking, \
         photography, and playing guitar in his free time.",
        )
        .build(),
    ];

    let resolver = DummyResolver;
    let result = agent.generate(messages, &resolver).await?;

    println!("\n📝 Response Text:");
    if let Some(text) = result.text() {
        println!("{}", text);
    }

    println!("\n✨ Extracted Structured Output:");
    if let Some(output) = &result.output {
        println!("{}", serde_json::to_string_pretty(output)?);
    } else {
        println!("No structured output extracted");
    }

    // Example 2: Handle multiple people
    println!("\n{}", "=".repeat(80));
    println!("Example 2: Extract multiple people");
    println!("{}", "=".repeat(80));

    let multi_schema = json!({
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                        "occupation": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }
        },
        "required": ["people"]
    });

    let agent2 = ToolLoopAgent::new(client.clone(), vec![], vec![step_count_is(5)])
        .with_system(
            "You are a helpful assistant that extracts information about multiple people. \
             Always respond with valid JSON matching the requested schema.",
        )
        .with_output_schema(
            OutputSchema::new(multi_schema.clone())
                .with_name("people_list")
                .with_description("List of people"),
        );

    let messages2 = vec![
        ChatMessage::user(
            "Extract information about these people: \
         Alice Johnson, 28, is a data scientist. \
         Bob Williams, 42, works as a teacher. \
         Carol Davis, 31, is a graphic designer.",
        )
        .build(),
    ];

    let result2 = agent2.generate(messages2, &resolver).await?;

    println!("\n✨ Extracted Structured Output:");
    if let Some(output) = &result2.output {
        println!("{}", serde_json::to_string_pretty(output)?);
    } else {
        println!("No structured output extracted");
    }

    // Example 3: Demonstrate validation with siumai-extras (if available)
    #[cfg(feature = "schema")]
    {
        use siumai::prelude::unified::SchemaValidator;
        use siumai_extras::schema::JsonSchemaValidator;

        println!("\n{}", "=".repeat(80));
        println!("Example 3: Schema Validation (with siumai-extras)");
        println!("{}", "=".repeat(80));

        if let Some(output) = &result.output {
            let validator = JsonSchemaValidator::new(&schema)?;

            match validator.validate(output) {
                Ok(_) => println!("✅ Output is valid according to schema!"),
                Err(e) => println!("❌ Validation failed: {}", e),
            }
        }
    }

    #[cfg(not(feature = "schema"))]
    {
        println!("\n💡 Tip: Enable the 'schema' feature in siumai-extras for validation:");
        println!(
            "   cargo run -p siumai-extras --example orchestrator_structured_output --features \"openai schema\""
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ All examples completed!");
    println!("{}", "=".repeat(80));

    Ok(())
}
