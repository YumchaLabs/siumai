//! HTTP Interceptor Example
//!
//! This example shows how to:
//! - Implement a custom `HttpInterceptor`
//! - Install interceptors on the unified builder (`LlmBuilder`)
//! - Observe safe request/response metadata and SSE events
//! - Add an idempotent correlation header on every request
//!
//! Notes:
//! - Do NOT log sensitive data (e.g., Authorization, API keys, full bodies).
//! - Keep hooks lightweight and non-blocking.

use std::sync::Arc;

use siumai::prelude::*;
use siumai::utils::http_interceptor::{HttpInterceptor, HttpRequestContext, LoggingInterceptor};
use uuid::Uuid;

#[derive(Clone, Default)]
struct CorrelationIdInterceptor;

impl HttpInterceptor for CorrelationIdInterceptor {
    fn on_before_send(
        &self,
        ctx: &HttpRequestContext,
        builder: reqwest::RequestBuilder,
        _body: &serde_json::Value,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        // Idempotently inject a correlation ID if not present
        let has_cid = headers
            .keys()
            .any(|k| k.as_str().eq_ignore_ascii_case("x-correlation-id"));
        let builder = if has_cid {
            builder
        } else {
            let cid = Uuid::new_v4().to_string();
            builder.header("X-Correlation-Id", cid)
        };
        tracing::debug!(
            target: "siumai::http_demo",
            provider=%ctx.provider_id,
            url=%ctx.url,
            stream=%ctx.stream,
            "on_before_send"
        );
        Ok(builder)
    }

    fn on_response(
        &self,
        ctx: &HttpRequestContext,
        response: &reqwest::Response,
    ) -> Result<(), LlmError> {
        tracing::debug!(
            target: "siumai::http_demo",
            provider=%ctx.provider_id,
            url=%ctx.url,
            status=%response.status().as_u16(),
            "on_response"
        );
        Ok(())
    }

    fn on_error(&self, ctx: &HttpRequestContext, error: &LlmError) {
        tracing::warn!(
            target: "siumai::http_demo",
            provider=%ctx.provider_id,
            url=%ctx.url,
            stream=%ctx.stream,
            err=%error,
            "on_error"
        );
    }

    fn on_sse_event(
        &self,
        ctx: &HttpRequestContext,
        event: &eventsource_stream::Event,
    ) -> Result<(), LlmError> {
        // Keep this minimal to avoid high-volume logs; event name only
        tracing::trace!(
            target: "siumai::http_demo",
            provider=%ctx.provider_id,
            url=%ctx.url,
            event_name=%event.event,
            "on_sse_event"
        );
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧩 HTTP Interceptor Example\n===========================\n");

    // Soft check for API key to avoid surprising failures in demo runs
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!("Set OPENAI_API_KEY to run this example against OpenAI.");
            return Ok(());
        }
    };

    // Install interceptors globally via LlmBuilder
    let client = LlmBuilder::new()
        .with_http_interceptor(Arc::new(CorrelationIdInterceptor))
        .with_http_interceptor(Arc::new(LoggingInterceptor))
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Non-streaming demo
    let reply = client
        .ask("Say hello in one short sentence.".to_string())
        .await?;
    println!("🤖 {}", reply);

    // Streaming demo (shows on_sse_event hook in action)
    use futures::StreamExt;
    let mut stream = client
        .chat_stream(
            vec![user!("Stream three short words, spaced.".to_string())],
            None,
        )
        .await?;
    print!("🌊 ");
    while let Some(ev) = stream.next().await {
        match ev? {
            ChatStreamEvent::ContentDelta { delta, .. } => print!("{}", delta),
            ChatStreamEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }
    println!();

    println!("\n✅ Done");
    Ok(())
}
