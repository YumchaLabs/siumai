//! Basic Telemetry - Track LLM operations
//!
//! This example demonstrates using telemetry to track and export
//! LLM operations to observability platforms.
//!
//! Construction mode: registry-first.
//! Prefer this pattern for application-level observability so telemetry stays
//! attached to the stable family + registry path rather than builder demos.
//!
//! Supported exporters:
//! - Langfuse
//! - Helicone
//! - Custom exporters
//!
//! ## Run
//! ```bash
//! cargo run --example basic-telemetry --features openai
//! ```
//!
//! ## Learn More
//! See `siumai/examples/08_telemetry/` for complete examples:
//! - basic_telemetry.rs - Console and platform exporters
//! - stream_telemetry.rs - Streaming with telemetry

use siumai::experimental::observability::telemetry::{
    TelemetryConfig, events::TelemetryEvent, exporters::TelemetryExporter,
};
use siumai::prelude::unified::*;

// Simple console exporter for demonstration
struct ConsoleExporter;

#[async_trait::async_trait]
impl TelemetryExporter for ConsoleExporter {
    async fn export(&self, event: &TelemetryEvent) -> Result<(), LlmError> {
        match event {
            TelemetryEvent::SpanStart(span) => {
                println!("📊 [START] {} (trace: {})", span.name, span.trace_id);
            }
            TelemetryEvent::SpanEnd(span) => {
                println!("✅ [END] {} (duration: {:?})", span.name, span.duration);
            }
            TelemetryEvent::Generation(generation) => {
                println!("🤖 [GEN] {}/{}", generation.provider, generation.model);
                if let Some(usage) = &generation.usage {
                    println!("   Tokens: {}", usage.total_tokens);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Basic Telemetry Example\n");

    // Create telemetry config with console exporter
    siumai::experimental::observability::telemetry::add_exporter(Box::new(ConsoleExporter)).await;
    let telemetry = TelemetryConfig::development();

    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    // Use ChatRequest to include telemetry
    let mut request = ChatRequest::builder()
        .message(user!("What is Rust?"))
        .build();
    request.telemetry = Some(telemetry);

    println!("Sending request with telemetry...\n");
    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;

    println!("\nAI: {}", response.content_text().unwrap_or_default());

    println!("\n💡 For production use:");
    println!("  - Use Langfuse exporter for full observability");
    println!("  - Use Helicone exporter for caching and analytics");
    println!("  - See examples/08_telemetry/ for complete examples");

    Ok(())
}
