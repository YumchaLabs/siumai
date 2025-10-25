//! OpenTelemetry Distributed Tracing Example
//!
//! This example demonstrates how to use OpenTelemetry for distributed tracing
//! with automatic W3C traceparent header injection.
//!
//! ## Prerequisites
//!
//! 1. Start Jaeger (or any OTLP-compatible backend):
//!    ```bash
//!    docker run -d --name jaeger \
//!      -e COLLECTOR_OTLP_ENABLED=true \
//!      -p 16686:16686 \
//!      -p 4317:4317 \
//!      -p 4318:4318 \
//!      jaegertracing/all-in-one:latest
//!    ```
//!
//! 2. Set your API key:
//!    ```bash
//!    export OPENAI_API_KEY=sk-...
//!    ```
//!
//! 3. Run the example:
//!    ```bash
//!    cargo run --example opentelemetry_tracing --features otel
//!    ```
//!
//! 4. View traces in Jaeger UI:
//!    Open http://localhost:16686 in your browser
//!
//! ## What This Example Shows
//!
//! - How to initialize OpenTelemetry with OTLP exporter
//! - How to create a client with OpenTelemetry middleware
//! - How to create parent spans for your operations
//! - How traceparent headers are automatically injected
//! - How to view distributed traces in Jaeger

use siumai::{Client, types::ChatRequest};
use siumai_extras::otel;
use siumai_extras::otel_middleware::OpenTelemetryMiddleware;
use opentelemetry::trace::{Tracer, TracerProvider};
use opentelemetry::global;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber for console logs
    tracing_subscriber::fmt::init();

    println!("🚀 Initializing OpenTelemetry...");
    
    // Initialize OpenTelemetry with Jaeger OTLP exporter
    let _guard = otel::init_opentelemetry(
        "siumai-example",           // Service name
        "http://localhost:4317",    // OTLP endpoint
    )?;

    println!("✅ OpenTelemetry initialized");
    println!("📊 Traces will be sent to Jaeger at http://localhost:16686\n");

    // Create client with OpenTelemetry middleware
    let client = Client::builder()
        .add_middleware(Arc::new(OpenTelemetryMiddleware::new()))
        .build()?;

    println!("🔧 Client created with OpenTelemetry middleware\n");

    // Example 1: Simple request (automatic tracing)
    println!("📝 Example 1: Simple request with automatic tracing");
    simple_request(&client).await?;

    // Example 2: Request with parent span (distributed tracing)
    println!("\n📝 Example 2: Request with parent span (distributed tracing)");
    request_with_parent_span(&client).await?;

    // Example 3: Multiple requests in a single trace
    println!("\n📝 Example 3: Multiple requests in a single trace");
    multiple_requests_in_trace(&client).await?;

    println!("\n✅ All examples completed!");
    println!("🔍 View traces in Jaeger UI: http://localhost:16686");
    println!("   - Service: siumai-example");
    println!("   - Look for operations: user_request, chat_sequence, llm.chat");

    // Shutdown OpenTelemetry to flush remaining spans
    opentelemetry::global::shutdown_tracer_provider();

    Ok(())
}

/// Example 1: Simple request with automatic tracing
async fn simple_request(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let request = ChatRequest::new(vec![
        ("user", "What is 2+2?"),
    ]);

    let response = client.chat()
        .model("gpt-3.5-turbo")
        .create(request)
        .await?;

    println!("   Response: {}", response.content_text().unwrap_or_default());
    println!("   ✅ Span created automatically by OpenTelemetryMiddleware");

    Ok(())
}

/// Example 2: Request with parent span for distributed tracing
async fn request_with_parent_span(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    // Create a parent span for the entire operation
    let tracer = global::tracer("siumai-example");
    let mut span = tracer.start("user_request");
    
    // Add custom attributes to the parent span
    span.set_attribute(opentelemetry::KeyValue::new("user_id", "12345"));
    span.set_attribute(opentelemetry::KeyValue::new("operation", "math_question"));

    // Attach the span to the current context
    let cx = opentelemetry::Context::current_with_span(span);
    let _guard = cx.attach();

    println!("   🔗 Parent span created: user_request");

    // Make LLM request - it will be a child span
    let request = ChatRequest::new(vec![
        ("user", "What is the capital of France?"),
    ]);

    let response = client.chat()
        .model("gpt-3.5-turbo")
        .create(request)
        .await?;

    println!("   Response: {}", response.content_text().unwrap_or_default());
    println!("   ✅ Child span created with traceparent header injected");
    println!("   📊 View the parent-child relationship in Jaeger");

    Ok(())
}

/// Example 3: Multiple requests in a single trace
async fn multiple_requests_in_trace(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    // Create a parent span for a sequence of operations
    let tracer = global::tracer("siumai-example");
    let mut span = tracer.start("chat_sequence");
    span.set_attribute(opentelemetry::KeyValue::new("sequence_type", "multi_turn"));

    let cx = opentelemetry::Context::current_with_span(span);
    let _guard = cx.attach();

    println!("   🔗 Parent span created: chat_sequence");

    // First request
    println!("   📤 Request 1: Asking for a topic");
    let request1 = ChatRequest::new(vec![
        ("user", "Give me a random topic in one word"),
    ]);

    let response1 = client.chat()
        .model("gpt-3.5-turbo")
        .create(request1)
        .await?;

    let topic = response1.content_text().unwrap_or_default();
    println!("   📥 Response 1: {}", topic);

    // Second request (using the response from the first)
    println!("   📤 Request 2: Asking about the topic");
    let request2 = ChatRequest::new(vec![
        ("user", &format!("Tell me one interesting fact about {}", topic)),
    ]);

    let response2 = client.chat()
        .model("gpt-3.5-turbo")
        .create(request2)
        .await?;

    println!("   📥 Response 2: {}", response2.content_text().unwrap_or_default());
    println!("   ✅ Both requests traced as children of chat_sequence");
    println!("   📊 View the complete sequence in Jaeger");

    Ok(())
}

