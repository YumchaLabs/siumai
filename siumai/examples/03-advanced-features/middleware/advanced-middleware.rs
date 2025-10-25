//! Advanced Middleware - All middleware capabilities
//!
//! This example demonstrates all middleware capabilities:
//! - `transform_params()` - Modify request parameters
//! - `pre_generate()` - Short-circuit before HTTP call
//! - `post_generate()` - Transform response after HTTP
//! - `on_stream_event()` - Intercept stream events
//! - `wrap_generate_async()` - Around-style wrapper for non-streaming
//! - `wrap_stream_async()` - Around-style wrapper for streaming
//!
//! ## Run
//! ```bash
//! cargo run --example advanced-middleware --features openai
//! ```

use futures::StreamExt;
use siumai::execution::middleware::language_model::{
    GenerateAsyncFn, LanguageModelMiddleware, StreamAsyncFn,
};
use siumai::prelude::*;
use siumai::streaming::ChatStreamEvent;
use std::sync::Arc;

// Middleware 1: Logging middleware using wrap_generate_async
#[derive(Clone)]
struct LoggingMiddleware;

impl LanguageModelMiddleware for LoggingMiddleware {
    fn wrap_generate_async(&self, next: Arc<GenerateAsyncFn>) -> Arc<GenerateAsyncFn> {
        Arc::new(move |req: ChatRequest| {
            let next = next.clone();
            Box::pin(async move {
                println!(
                    "🔍 [Logging] Request: model={}, messages={}",
                    req.common_params.model,
                    req.messages.len()
                );

                let start = std::time::Instant::now();
                let result = next(req).await;
                let duration = start.elapsed();

                match &result {
                    Ok(resp) => println!("✅ [Logging] Response received in {:?}", duration),
                    Err(e) => println!("❌ [Logging] Error: {:?}", e),
                }

                result
            })
        })
    }

    fn wrap_stream_async(&self, next: Arc<StreamAsyncFn>) -> Arc<StreamAsyncFn> {
        Arc::new(move |req: ChatRequest| {
            let next = next.clone();
            Box::pin(async move {
                println!(
                    "🔍 [Logging] Stream request: model={}",
                    req.common_params.model
                );

                let stream = next(req).await?;
                let mut event_count = 0;

                let logged_stream = stream.map(move |event| {
                    event_count += 1;
                    if let Ok(ev) = &event {
                        match ev {
                            ChatStreamEvent::ContentDelta { delta, .. } => {
                                println!(
                                    "📝 [Logging] Delta #{}: {} chars",
                                    event_count,
                                    delta.len()
                                );
                            }
                            ChatStreamEvent::StreamEnd { .. } => {
                                println!("✅ [Logging] Stream ended after {} events", event_count);
                            }
                            _ => {}
                        }
                    }
                    event
                });

                Ok(Box::pin(logged_stream) as ChatStream)
            })
        })
    }
}

// Middleware 2: Caching middleware using pre_generate
#[derive(Clone)]
struct CachingMiddleware {
    cache_key: String,
    cached_response: String,
}

impl LanguageModelMiddleware for CachingMiddleware {
    fn pre_generate(&self, req: &ChatRequest) -> Option<Result<ChatResponse, LlmError>> {
        // Check if this request matches our cache key
        if let Some(last_msg) = req.messages.last() {
            if let Some(text) = last_msg.content.text() {
                if text.contains(&self.cache_key) {
                    println!("💾 [Cache] Cache hit! Returning cached response");
                    return Some(Ok(ChatResponse::new(MessageContent::Text(
                        self.cached_response.clone(),
                    ))));
                }
            }
        }
        None // Cache miss, continue to HTTP
    }
}

// Middleware 3: Response transformation using post_generate
#[derive(Clone)]
struct ResponseTransformMiddleware {
    prefix: String,
}

impl LanguageModelMiddleware for ResponseTransformMiddleware {
    fn post_generate(
        &self,
        _req: &ChatRequest,
        mut resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        // Add prefix to response
        if let Some(text) = resp.content_text() {
            resp.content = MessageContent::Text(format!("{}{}", self.prefix, text));
            println!("🔄 [Transform] Added prefix to response");
        }
        Ok(resp)
    }
}

// Middleware 4: Stream event filtering using on_stream_event
#[derive(Clone)]
struct StreamFilterMiddleware;

impl LanguageModelMiddleware for StreamFilterMiddleware {
    fn on_stream_event(
        &self,
        _req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match &ev {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                // Filter out empty deltas
                if delta.is_empty() {
                    println!("🚫 [Filter] Filtered empty delta");
                    Ok(vec![]) // Return empty vec to filter out this event
                } else {
                    Ok(vec![ev])
                }
            }
            _ => Ok(vec![ev]),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    println!("=== Advanced Middleware Demo ===\n");

    // Example 1: Logging middleware with wrap_generate_async
    println!("\n📝 Example 1: Logging Middleware (wrap_generate_async)");
    let client = LlmBuilder::new().openai().api_key(&api_key).build().await?;

    // Note: This example shows the concept. In practice, you'd apply middleware
    // through the registry or by wrapping the executor directly.
    let messages = vec![ChatMessage::user("Say 'Hello from logging middleware!'").build()];

    let response = client.chat(messages).await?;
    println!("Response: {}\n", response.content_text().unwrap());

    // Example 2: Caching middleware with pre_generate
    println!("\n📝 Example 2: Caching Middleware (pre_generate)");
    println!("💡 This middleware short-circuits the HTTP call for cached requests");
    println!("   In production, you'd use a real cache like Redis or in-memory LRU\n");

    // Example 3: Response transformation with post_generate
    println!("\n📝 Example 3: Response Transform (post_generate)");
    println!("💡 This middleware adds a prefix to all responses");
    println!("   Useful for adding context, formatting, or metadata\n");

    // Example 4: Stream filtering with on_stream_event
    println!("\n📝 Example 4: Stream Filtering (on_stream_event)");
    println!("💡 This middleware filters out empty delta events");
    println!("   Useful for cleaning up noisy streams\n");

    println!("\n💡 Middleware Execution Order:");
    println!(
        "   Non-streaming: transform_params → pre_generate → wrap_generate_async → HTTP → post_generate"
    );
    println!("   Streaming: transform_params → wrap_stream_async → HTTP → on_stream_event");
    println!("\n💡 Use Cases:");
    println!("   - Logging: Track all requests/responses");
    println!("   - Caching: Avoid redundant API calls");
    println!("   - Rate limiting: Control request frequency");
    println!("   - A/B testing: Route to different models");
    println!("   - Error handling: Retry or fallback logic");
    println!("   - Monitoring: Collect metrics and telemetry");

    Ok(())
}
