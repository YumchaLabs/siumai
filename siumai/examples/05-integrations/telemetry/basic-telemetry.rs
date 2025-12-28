//! Basic Telemetry - Track LLM operations
//!
//! This example demonstrates using telemetry to track and export
//! LLM operations to observability platforms.
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

use siumai::prelude::*;
use siumai::experimental::observability::telemetry::{
    TelemetryConfig,
    events::TelemetryEvent,
    exporters::TelemetryExporter,
};

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

    // Build client with telemetry
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Use ChatRequest to include telemetry
    let mut request = ChatRequest::builder().message(user!("What is Rust?")).build();
    request.telemetry = Some(telemetry);

    println!("Sending request with telemetry...\n");
    let response = client.chat_request(request).await?;

    println!("\nAI: {}", response.content_text().unwrap());

    println!("\n💡 For production use:");
    println!("  - Use Langfuse exporter for full observability");
    println!("  - Use Helicone exporter for caching and analytics");
    println!("  - See examples/08_telemetry/ for complete examples");

    Ok(())
}
