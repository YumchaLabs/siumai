//! HTTP Interceptor - Observe requests and responses
//!
//! This example demonstrates using HTTP interceptors for logging,
//! monitoring, or debugging.
//!
//! ## Run
//! ```bash
//! cargo run --example http-interceptor --features openai
//! ```

use siumai::experimental::execution::http::{HttpInterceptor, HttpRequestContext};
use siumai::prelude::unified::*;
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
use std::sync::Arc;

#[derive(Clone)]
struct LoggingInterceptor;

impl HttpInterceptor for LoggingInterceptor {
    fn on_before_send(
        &self,
        ctx: &HttpRequestContext,
        builder: reqwest::RequestBuilder,
        _body: &serde_json::Value,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        println!("📤 Outgoing request:");
        println!("   Provider: {}", ctx.provider_id);
        println!("   URL: {}", ctx.url);
        println!("   Stream: {}", ctx.stream);
        println!("   Headers: {} headers", headers.len());
        Ok(builder)
    }

    fn on_response(
        &self,
        ctx: &HttpRequestContext,
        response: &reqwest::Response,
    ) -> Result<(), LlmError> {
        println!("📥 Incoming response:");
        println!("   Provider: {}", ctx.provider_id);
        println!("   Status: {}", response.status());
        println!("   Headers: {} headers", response.headers().len());
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let interceptor: Arc<dyn HttpInterceptor> = Arc::new(LoggingInterceptor);

    let client = OpenAiClient::from_config(
        OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?).with_model("gpt-4o-mini"),
    )?
    .with_http_interceptors(vec![interceptor]);

    println!("Sending request with HTTP interceptor...\n");

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!("Hello!")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("\nAI: {}", response.content_text().unwrap_or_default());

    Ok(())
}
