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
//! 3. Run the example (enable extras opentelemetry feature):
//!    ```bash
//!    cargo run -p siumai-extras --example opentelemetry_tracing --features "opentelemetry,openai"
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

use opentelemetry::global;
use opentelemetry::trace::{Span as _, TraceContextExt, Tracer};
use siumai::prelude::unified::*;
use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig};
use siumai_extras::otel;
use siumai_extras::otel_middleware::OpenTelemetryMiddleware;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber for console logs
    tracing_subscriber::fmt::init();

    println!("🚀 Initializing OpenTelemetry...");

    // Initialize OpenTelemetry with Jaeger OTLP exporter
    let config = otel::OtelConfig::builder()
        .service_name("siumai-example")
        .otlp_endpoint("http://localhost:4317")
        .build();
    let _guard = otel::init_opentelemetry(config).await?;

    println!("✅ OpenTelemetry initialized");
    println!("📊 Traces will be sent to Jaeger at http://localhost:16686\n");

    // Build an OpenAI client (config-first) and attach OpenTelemetry middleware.
    let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)
        .with_model("gpt-4o-mini")
        .with_temperature(0.0);
    let openai_client = OpenAiClient::from_config(cfg)?;

    // Keep default auto middlewares and add OpenTelemetry middleware
    let model_id = openai_client.common_params().model.clone();
    let mut middlewares = siumai::experimental::execution::middleware::build_auto_middlewares_vec(
        "openai", &model_id,
    );
    middlewares.push(Arc::new(OpenTelemetryMiddleware::new()));
    let client = openai_client.with_model_middlewares(middlewares);

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

    // Guard drop will shutdown providers and flush remaining spans

    Ok(())
}

/// Example 1: Simple request with automatic tracing
async fn simple_request(
    model: &impl siumai::text::TextModelV3,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = ChatRequest::new(vec![user!("What is 2+2?")]);

    let response = text::generate(model, request, text::GenerateOptions::default()).await?;

    println!(
        "   Response: {}",
        response.content_text().unwrap_or_default()
    );
    println!("   ✅ Span created automatically by OpenTelemetryMiddleware");

    Ok(())
}

/// Example 2: Request with parent span for distributed tracing
async fn request_with_parent_span(
    model: &impl siumai::text::TextModelV3,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let request = ChatRequest::new(vec![user!("What is the capital of France?")]);

    let response = text::generate(model, request, text::GenerateOptions::default()).await?;

    println!(
        "   Response: {}",
        response.content_text().unwrap_or_default()
    );
    println!("   ✅ Child span created with traceparent header injected");
    println!("   📊 View the parent-child relationship in Jaeger");

    Ok(())
}

/// Example 3: Multiple requests in a single trace
async fn multiple_requests_in_trace(
    model: &impl siumai::text::TextModelV3,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a parent span for a sequence of operations
    let tracer = global::tracer("siumai-example");
    let mut span = tracer.start("chat_sequence");
    span.set_attribute(opentelemetry::KeyValue::new("sequence_type", "multi_turn"));

    let cx = opentelemetry::Context::current_with_span(span);
    let _guard = cx.attach();

    println!("   🔗 Parent span created: chat_sequence");

    // First request
    println!("   📤 Request 1: Asking for a topic");
    let request1 = ChatRequest::new(vec![user!("Give me a random topic in one word")]);

    let response1 = text::generate(model, request1, text::GenerateOptions::default()).await?;

    let topic = response1.content_text().unwrap_or_default();
    println!("   📥 Response 1: {}", topic);

    // Second request (using the response from the first)
    println!("   📤 Request 2: Asking about the topic");
    let request2 = ChatRequest::new(vec![user!(format!(
        "Tell me one interesting fact about {}",
        topic
    ))]);

    let response2 = text::generate(model, request2, text::GenerateOptions::default()).await?;

    println!(
        "   📥 Response 2: {}",
        response2.content_text().unwrap_or_default()
    );
    println!("   ✅ Both requests traced as children of chat_sequence");
    println!("   📊 View the complete sequence in Jaeger");

    Ok(())
}
