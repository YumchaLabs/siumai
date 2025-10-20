//! HTTP Interceptor - Observe requests and responses
//!
//! This example demonstrates using HTTP interceptors for logging,
//! monitoring, or debugging.
//!
//! ## Run
//! ```bash
//! cargo run --example http-interceptor --features openai
//! ```

use siumai::prelude::*;
use siumai::utils::{HttpInterceptor, HttpRequestContext};
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
    let interceptor = Arc::new(LoggingInterceptor);

    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .with_http_interceptor(interceptor)
        .build()
        .await?;

    println!("Sending request with HTTP interceptor...\n");

    let response = client.chat(vec![user!("Hello!")]).await?;

    println!("\nAI: {}", response.content_text().unwrap());

    Ok(())
}
