//! Advanced Agent Features Example
//!
//! This example demonstrates the new advanced features added to ToolLoopAgent:
//! - Agent-level tool_choice and active_tools
//! - New stop conditions (has_tool_result, has_no_tool_calls)
//! - Structured output with schema validation
//!
//! Run with:
//! ```bash
//! cargo run -p siumai-extras --example orchestrator_advanced_agent_features --features openai
//! ```

use serde_json::json;
use siumai::prelude::Siumai;
use siumai::prelude::unified::{ChatMessage, OutputSchema, Tool};
use siumai_extras::orchestrator::{
    ToolChoice, ToolLoopAgent, has_no_tool_calls, has_tool_result, step_count_is,
};

// Simple tool resolver
struct SimpleResolver;

#[async_trait::async_trait]
impl siumai_extras::orchestrator::ToolResolver for SimpleResolver {
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::prelude::unified::LlmError> {
        match tool_name {
            "get_weather" => {
                let location = arguments
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                Ok(json!({
                    "location": location,
                    "temperature": 72,
                    "condition": "Sunny"
                }))
            }
            "calculate" => {
                let a = arguments.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b = arguments.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let op = arguments
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("add");

                let result = match op {
                    "add" => a + b,
                    "subtract" => a - b,
                    "multiply" => a * b,
                    "divide" if b != 0.0 => a / b,
                    _ => 0.0,
                };

                Ok(json!({"result": result}))
            }
            _ => Ok(json!({"error": "Unknown tool"})),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Define tools
    let weather_tool = Tool::function(
        "get_weather",
        "Get the current weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }),
    );

    let calculator_tool = Tool::function(
        "calculate",
        "Perform a mathematical calculation",
        json!({
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                }
            },
            "required": ["a", "b", "operation"]
        }),
    );

    let tools = vec![weather_tool, calculator_tool];
    let resolver = SimpleResolver;

    println!("🚀 Advanced Agent Features Demo\n");
    println!("{}", "=".repeat(80));

    // Example 1: Agent-level tool_choice
    println!("\n📌 Example 1: Agent-level tool_choice (Required)");
    println!("{}", "-".repeat(80));

    let agent1 = ToolLoopAgent::new(client.clone(), tools.clone(), vec![step_count_is(3)])
        .with_system("You are a helpful assistant.")
        .with_tool_choice(ToolChoice::Required); // Force tool usage

    let messages1 = vec![ChatMessage::user("What's 100 + 50?").build()];
    let result1 = agent1.generate(messages1, &resolver).await?;

    println!("Response: {}", result1.text().unwrap_or(""));
    println!("Tool calls made: {}", result1.all_tool_calls().len());

    // Example 2: Agent-level active_tools
    println!("\n📌 Example 2: Agent-level active_tools (Only weather)");
    println!("{}", "-".repeat(80));

    let agent2 = ToolLoopAgent::new(client.clone(), tools.clone(), vec![step_count_is(3)])
        .with_system("You are a helpful assistant.")
        .with_active_tools(vec!["get_weather".to_string()]); // Only allow weather tool

    let messages2 =
        vec![ChatMessage::user("What's the weather in Tokyo and what's 5 * 10?").build()];
    let result2 = agent2.generate(messages2, &resolver).await?;

    println!("Response: {}", result2.text().unwrap_or(""));
    println!("Tool calls made: {}", result2.all_tool_calls().len());
    for tool_call in result2.all_tool_calls() {
        if let Some(info) = tool_call.as_tool_call() {
            println!("  - Tool: {}", info.tool_name);
        }
    }

    // Example 3: New stop condition - has_tool_result
    println!("\n📌 Example 3: Stop condition - has_tool_result");
    println!("{}", "-".repeat(80));

    let agent3 = ToolLoopAgent::new(
        client.clone(),
        tools.clone(),
        vec![has_tool_result()], // Stop after first tool result
    )
    .with_system("You are a helpful assistant.");

    let messages3 = vec![ChatMessage::user("Get the weather in Paris").build()];
    let result3 = agent3.generate(messages3, &resolver).await?;

    println!("Steps taken: {}", result3.steps.len());
    println!("Response: {}", result3.text().unwrap_or(""));

    // Example 4: New stop condition - has_no_tool_calls
    println!("\n📌 Example 4: Stop condition - has_no_tool_calls");
    println!("{}", "-".repeat(80));

    let agent4 = ToolLoopAgent::new(
        client.clone(),
        tools.clone(),
        vec![has_no_tool_calls()], // Stop when model doesn't call tools
    )
    .with_system("You are a helpful assistant.");

    let messages4 = vec![ChatMessage::user("What's the capital of France?").build()];
    let result4 = agent4.generate(messages4, &resolver).await?;

    println!("Steps taken: {}", result4.steps.len());
    println!("Response: {}", result4.text().unwrap_or(""));

    // Example 5: Combining features - structured output + tool_choice
    println!("\n📌 Example 5: Structured output + tool_choice");
    println!("{}", "-".repeat(80));

    let schema = json!({
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "temperature": {"type": "number"},
            "condition": {"type": "string"},
            "recommendation": {"type": "string"}
        },
        "required": ["location", "temperature", "condition"]
    });

    let agent5 = ToolLoopAgent::new(client.clone(), tools.clone(), vec![step_count_is(5)])
        .with_system(
            "You are a weather assistant. Always use the get_weather tool and \
         respond with JSON matching the schema.",
        )
        .with_tool_choice(ToolChoice::Required)
        .with_active_tools(vec!["get_weather".to_string()])
        .with_output_schema(
            OutputSchema::new(schema.clone())
                .with_name("weather_report")
                .with_description("Weather report with recommendation"),
        );

    let messages5 = vec![ChatMessage::user("What's the weather in London?").build()];
    let result5 = agent5.generate(messages5, &resolver).await?;

    println!("Structured output:");
    if let Some(output) = &result5.output {
        println!("{}", serde_json::to_string_pretty(output)?);
    }

    // Example 6: Validation with siumai-extras (if available)
    #[cfg(feature = "schema")]
    {
        use siumai::prelude::unified::SchemaValidator;
        use siumai_extras::schema::JsonSchemaValidator;

        println!("\n📌 Example 6: Schema validation with siumai-extras");
        println!("{}", "-".repeat(80));

        if let Some(output) = &result5.output {
            let validator = JsonSchemaValidator::new(&schema)?;

            match validator.validate(output) {
                Ok(_) => println!("✅ Output is valid according to schema!"),
                Err(e) => println!("❌ Validation failed: {}", e),
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ All examples completed!");
    println!("{}", "=".repeat(80));

    Ok(())
}
