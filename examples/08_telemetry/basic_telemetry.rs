//! Basic Telemetry Example
//!
//! This example demonstrates how to use the telemetry module to track
//! LLM operations and export them to observability platforms.

use siumai::orchestrator::{generate, OrchestratorOptions, ToolResolver};
use siumai::prelude::*;
use siumai::telemetry::{
    self,
    events::TelemetryEvent,
    exporters::{helicone::HeliconeExporter, langfuse::LangfuseExporter, TelemetryExporter},
    TelemetryConfig,
};
use std::sync::Arc;

/// A simple console exporter for demonstration
struct ConsoleExporter;

#[async_trait::async_trait]
impl TelemetryExporter for ConsoleExporter {
    async fn export(&self, event: &TelemetryEvent) -> Result<(), siumai::error::LlmError> {
        match event {
            TelemetryEvent::SpanStart(span) => {
                println!("📊 [SPAN START] {} (trace: {})", span.name, span.trace_id);
                for (key, value) in &span.attributes {
                    println!("   - {}: {}", key, value);
                }
            }
            TelemetryEvent::SpanEnd(span) => {
                println!(
                    "✅ [SPAN END] {} (duration: {:?})",
                    span.name, span.duration
                );
                if let Some(error) = &span.error {
                    println!("   ❌ Error: {}", error);
                }
            }
            TelemetryEvent::Generation(gen) => {
                println!("🤖 [GENERATION] {}/{}", gen.provider, gen.model);
                if let Some(usage) = &gen.usage {
                    println!(
                        "   - Tokens: {} prompt + {} completion = {} total",
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    );
                }
            }
            TelemetryEvent::Orchestrator(orch) => {
                println!(
                    "🎯 [ORCHESTRATOR] Step {}/{} ({:?})",
                    orch.current_step, orch.total_steps, orch.step_type
                );
                if let Some(usage) = &orch.total_usage {
                    println!(
                        "   - Total Tokens: {} prompt + {} completion = {} total",
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    );
                }
            }
            TelemetryEvent::ToolExecution(tool) => {
                println!("🔧 [TOOL] {}", tool.tool_call.function.as_ref().map(|f| f.name.as_str()).unwrap_or("unknown"));
                if let Some(duration) = tool.duration {
                    println!("   - Duration: {:?}", duration);
                }
            }
        }
        Ok(())
    }
}

/// Simple tool resolver for demonstration
struct SimpleToolResolver;

#[async_trait::async_trait]
impl ToolResolver for SimpleToolResolver {
    async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::error::LlmError> {
        match name {
            "get_weather" => {
                let location = args
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Ok(serde_json::json!({
                    "location": location,
                    "temperature": 72,
                    "condition": "sunny"
                }))
            }
            _ => Err(siumai::error::LlmError::InvalidInput(format!(
                "Unknown tool: {}",
                name
            ))),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    // Note: This function is deprecated. For production use, consider using siumai-extras::telemetry
    #[allow(deprecated)]
    siumai::tracing::init_default_tracing()?;

    println!("🚀 Telemetry Example\n");

    // Register a console exporter for demonstration
    telemetry::add_exporter(Box::new(ConsoleExporter)).await;

    // Optionally, you can also register Langfuse or Helicone exporters:
    // telemetry::add_exporter(Box::new(LangfuseExporter::new(
    //     "https://cloud.langfuse.com",
    //     "your-public-key",
    //     "your-secret-key",
    // ))).await;

    // Create an OpenAI client
    let client = LlmBuilder::new()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4")
        .temperature(0.7)
        .build()
        .await?;

    // Define a simple weather tool
    let weather_tool = Tool::new(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
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

    // Create telemetry configuration
    let telemetry_config = TelemetryConfig::builder()
        .enabled(true)
        .record_inputs(true)
        .record_outputs(true)
        .record_tools(true)
        .record_usage(true)
        .session_id("demo-session-123")
        .user_id("demo-user-456")
        .tag("example")
        .tag("weather")
        .build();

    // Create orchestrator options with telemetry
    let opts = OrchestratorOptions {
        max_steps: 3,
        telemetry: Some(telemetry_config),
        ..Default::default()
    };

    // Run orchestrator with telemetry
    println!("\n📝 Sending request with telemetry enabled...\n");

    let messages = vec![user!("What's the weather like in San Francisco?")];

    let (response, steps) = generate(
        &client,
        messages,
        Some(vec![weather_tool]),
        Some(&SimpleToolResolver),
        opts,
    )
    .await?;

    println!("\n📋 Final Response:");
    if let Some(text) = response.content_text() {
        println!("{}", text);
    }

    println!("\n📊 Total Steps: {}", steps.len());

    // Demonstrate Helicone integration
    println!("\n\n🔍 Helicone Integration Example:");
    let helicone = HeliconeExporter::new("your-helicone-api-key");
    let mut headers = std::collections::HashMap::new();
    helicone.add_headers(&mut headers, Some("session-123"), Some("user-456"));

    println!("Headers to add to LLM requests:");
    for (key, value) in headers {
        println!("  {}: {}", key, value);
    }

    println!("\n✅ Example completed!");

    Ok(())
}

