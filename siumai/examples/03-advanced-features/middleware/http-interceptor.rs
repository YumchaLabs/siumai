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
use std::sync::Arc;

#[derive(Clone)]
struct LoggingInterceptor;

#[async_trait::async_trait]
impl HttpInterceptor for LoggingInterceptor {
    async fn before_request(&self, request: &HttpRequest) -> Result<(), LlmError> {
        println!("📤 Outgoing request:");
        println!("   URL: {}", request.url);
        println!("   Method: {}", request.method);
        println!("   Headers: {} headers", request.headers.len());
        Ok(())
    }

    async fn after_response(&self, response: &HttpResponse) -> Result<(), LlmError> {
        println!("📥 Incoming response:");
        println!("   Status: {}", response.status);
        println!("   Body size: {} bytes", response.body.len());
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
        .http_interceptor(interceptor)
        .build()
        .await?;

    println!("Sending request with HTTP interceptor...\n");

    let response = client.chat(vec![user!("Hello!")]).await?;

    println!("\nAI: {}", response.content_text().unwrap());

    Ok(())
}
