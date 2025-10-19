//! Stream Telemetry Example
//!
//! This example demonstrates how to use telemetry with streaming responses.
//! It shows how telemetry automatically tracks:
//! - Stream creation (span start/end)
//! - Complete stream data collection (GenerationEvent at stream end)
//! - Input messages, output content, usage statistics
//!
//! Run with:
//! ```bash
//! cargo run --example stream_telemetry --features server-adapters
//! ```

use siumai::error::Result;
use siumai::telemetry::{self, events::TelemetryEvent, TelemetryConfig};
use siumai::types::{ChatMessage, ChatRequest};
use siumai::{ProviderRegistryHandle, RegistryOptions};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🔍 Stream Telemetry Example\n");

    // Create registry
    let registry = ProviderRegistryHandle::new(RegistryOptions::default());

    // Configure telemetry
    let telemetry_config = TelemetryConfig::builder()
        .enabled(true)
        .record_inputs(true)
        .record_outputs(true)
        .record_usage(true)
        .session_id("stream-session-123")
        .user_id("user-456")
        .build();

    // Set up a custom telemetry collector to capture events
    let events = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();

    // Register a custom exporter to capture events
    telemetry::register_exporter(Box::new(move |event| {
        let events = events_clone.clone();
        Box::pin(async move {
            let mut events = events.lock().await;
            events.push(event.clone());
            println!("📊 Telemetry Event: {}", event_type_name(&event));
            Ok(())
        })
    }))
    .await;

    // Create a chat request with telemetry enabled
    let messages = vec![
        ChatMessage::user("Tell me a very short joke about programming."),
    ];

    let mut request = ChatRequest::new(messages.clone());
    request.common_params.model = "gpt-4o-mini".to_string();
    request.stream = true;
    request.telemetry = Some(telemetry_config);

    println!("📤 Sending streaming request with telemetry enabled...\n");

    // Get language model client
    let client = registry
        .language_model("openai", Some("gpt-4o-mini".to_string()))
        .await?;

    // Execute streaming request
    match client.chat_stream_with_request(request).await {
        Ok(mut stream) => {
            println!("📡 Streaming response:\n");

            // Consume the stream
            while let Some(event) = stream.next().await {
                match event {
                    Ok(event) => {
                        use siumai::stream::ChatStreamEvent;
                        match event {
                            ChatStreamEvent::ContentDelta { delta, .. } => {
                                print!("{}", delta);
                                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            }
                            ChatStreamEvent::StreamEnd { response } => {
                                println!("\n\n✅ Stream completed!");
                                if let Some(usage) = &response.usage {
                                    println!("   Usage: {} prompt + {} completion = {} total tokens",
                                        usage.prompt_tokens.unwrap_or(0),
                                        usage.completion_tokens.unwrap_or(0),
                                        usage.total_tokens.unwrap_or(0)
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        eprintln!("\n❌ Stream error: {}", e);
                        break;
                    }
                }
            }

            // Wait a bit for telemetry events to be processed
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            // Display captured telemetry events
            println!("\n📊 Captured Telemetry Events:");
            let events = events.lock().await;
            for (i, event) in events.iter().enumerate() {
                println!("   {}. {}", i + 1, event_summary(event));
            }

            println!("\n✨ Stream telemetry tracking completed!");
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    Ok(())
}

fn event_type_name(event: &TelemetryEvent) -> &'static str {
    match event {
        TelemetryEvent::SpanStart(_) => "SpanStart",
        TelemetryEvent::SpanEnd(_) => "SpanEnd",
        TelemetryEvent::Generation(_) => "Generation",
        TelemetryEvent::Orchestrator(_) => "Orchestrator",
        TelemetryEvent::ToolExecution(_) => "ToolExecution",
    }
}

fn event_summary(event: &TelemetryEvent) -> String {
    match event {
        TelemetryEvent::SpanStart(span) => {
            format!("SpanStart: {} ({})", span.name, span.span_id)
        }
        TelemetryEvent::SpanEnd(span) => {
            format!(
                "SpanEnd: {} ({}) - {:?}",
                span.name,
                span.span_id,
                span.status
            )
        }
        TelemetryEvent::Generation(gen) => {
            let has_input = gen.input.is_some();
            let has_output = gen.output.is_some();
            let has_usage = gen.usage.is_some();
            format!(
                "Generation: {} (input: {}, output: {}, usage: {})",
                gen.model, has_input, has_output, has_usage
            )
        }
        TelemetryEvent::Orchestrator(orch) => {
            format!(
                "Orchestrator: step {}/{}",
                orch.current_step, orch.total_steps
            )
        }
        TelemetryEvent::ToolExecution(tool) => {
            format!("ToolExecution: {}", tool.tool_call.function.name)
        }
    }
}

