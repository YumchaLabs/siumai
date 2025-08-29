//! SiliconFlow HTTP Configuration Example
//!
//! This example demonstrates how to configure HTTP settings for SiliconFlow,
//! including timeouts, proxies, custom headers, and retry mechanisms.
//!
//! Before running, set your API key:
//! ```bash
//! export SILICONFLOW_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example siliconflow_http_config
//! ```

use siumai::prelude::*;
use siumai::types::{CommonParams, HttpConfig};
use std::collections::HashMap;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 SiliconFlow HTTP Configuration Example\n");

    // Get API key from environment
    let api_key = match std::env::var("SILICONFLOW_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("❌ SILICONFLOW_API_KEY environment variable not set");
            println!("   Please set it with: export SILICONFLOW_API_KEY=\"your-key\"");
            return Ok(());
        }
    };

    // Demonstrate different HTTP configurations
    demonstrate_basic_config(&api_key).await?;
    demonstrate_timeout_config(&api_key).await?;
    demonstrate_proxy_config(&api_key).await?;
    demonstrate_custom_headers(&api_key).await?;
    demonstrate_custom_http_client(&api_key).await?;

    println!("\n✅ HTTP configuration examples completed!");
    Ok(())
}

/// Demonstrate basic HTTP configuration
async fn demonstrate_basic_config(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Basic HTTP Configuration:");

    let client = LlmBuilder::new()
        .siliconflow()
        .api_key(api_key)
        .model("deepseek-chat")
        .build()
        .await?;

    let messages = vec![user!("Hello! This is a test with basic HTTP config.")];

    match client.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("   ✅ Response: {}", text.chars().take(100).collect::<String>());
            }
        }
        Err(e) => {
            println!("   ❌ Request failed: {}", e);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate timeout configuration
async fn demonstrate_timeout_config(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("⏱️  Timeout Configuration:");

    // Create HTTP config with custom timeouts
    let mut http_config = HttpConfig::default();
    http_config.timeout = Some(Duration::from_secs(60)); // 60 second timeout
    http_config.connect_timeout = Some(Duration::from_secs(10)); // 10 second connect timeout

    let client = LlmBuilder::new()
        .siliconflow()
        .api_key(api_key)
        .model("deepseek-chat")
        .with_http_config(http_config)
        .build()
        .await?;

    let messages = vec![user!("Test with custom timeout settings.")];

    match client.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("   ✅ Response with custom timeouts: {}", text.chars().take(100).collect::<String>());
            }
        }
        Err(e) => {
            println!("   ❌ Request failed: {}", e);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate proxy configuration
async fn demonstrate_proxy_config(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Proxy Configuration:");

    // Note: This example shows how to configure a proxy, but doesn't actually use one
    // Uncomment and modify the proxy URL if you have a proxy server
    let mut http_config = HttpConfig::default();
    // http_config.proxy = Some("http://proxy.example.com:8080".to_string());

    let client = LlmBuilder::new()
        .siliconflow()
        .api_key(api_key)
        .model("deepseek-chat")
        .with_http_config(http_config)
        .build()
        .await?;

    println!("   📝 Proxy configuration set (but not used in this example)");
    println!("   💡 To use a proxy, uncomment and set the proxy URL in the code");

    // Test without actually using proxy
    let messages = vec![user!("Test proxy configuration setup.")];

    match client.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("   ✅ Response: {}", text.chars().take(100).collect::<String>());
            }
        }
        Err(e) => {
            println!("   ❌ Request failed: {}", e);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate custom headers
async fn demonstrate_custom_headers(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Custom Headers Configuration:");

    // Create HTTP config with custom headers
    let mut http_config = HttpConfig::default();
    let mut headers = HashMap::new();
    headers.insert("X-Custom-Client".to_string(), "siumai-example/1.0".to_string());
    headers.insert("X-Request-ID".to_string(), "example-request-123".to_string());
    http_config.headers = headers;

    let client = LlmBuilder::new()
        .siliconflow()
        .api_key(api_key)
        .model("deepseek-chat")
        .with_http_config(http_config)
        .build()
        .await?;

    let messages = vec![user!("Test with custom headers.")];

    match client.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("   ✅ Response with custom headers: {}", text.chars().take(100).collect::<String>());
            }
        }
        Err(e) => {
            println!("   ❌ Request failed: {}", e);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate custom HTTP client
async fn demonstrate_custom_http_client(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Custom HTTP Client Configuration:");

    // Create a custom reqwest client with advanced settings
    let custom_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(45))
        .connect_timeout(Duration::from_secs(15))
        .user_agent("siumai-custom-client/1.0")
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .pool_max_idle_per_host(10)
        .pool_idle_timeout(Duration::from_secs(90))
        .build()?;

    let client = LlmBuilder::new()
        .siliconflow()
        .api_key(api_key)
        .model("deepseek-chat")
        .with_http_client(custom_client)
        .build()
        .await?;

    let messages = vec![user!("Test with custom HTTP client configuration.")];

    match client.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("   ✅ Response with custom HTTP client: {}", text.chars().take(100).collect::<String>());
            }
        }
        Err(e) => {
            println!("   ❌ Request failed: {}", e);
        }
    }

    println!();
    Ok(())
}

/*
🎯 Key HTTP Configuration Concepts:

Timeout Settings:
- Request Timeout: Maximum time for entire request/response cycle
- Connect Timeout: Maximum time to establish connection
- Read Timeout: Maximum time to read response data

Proxy Configuration:
- HTTP Proxy: http://proxy.example.com:8080
- HTTPS Proxy: https://proxy.example.com:8080
- SOCKS Proxy: socks5://proxy.example.com:1080
- Authentication: http://user:pass@proxy.example.com:8080

Custom Headers:
- User-Agent: Identify your application
- X-Request-ID: Track requests across systems
- X-Custom-*: Application-specific headers
- Authorization: Custom auth schemes (beyond Bearer tokens)

Connection Pooling:
- Keep-Alive: Reuse connections for better performance
- Pool Size: Maximum connections per host
- Idle Timeout: How long to keep unused connections

Best Practices:
1. Set reasonable timeouts (30-60 seconds for chat)
2. Use connection pooling for high-throughput applications
3. Configure retries for transient failures
4. Use custom User-Agent for identification
5. Implement proper error handling for network issues

Security Considerations:
- Validate proxy URLs to prevent SSRF attacks
- Use HTTPS proxies when possible
- Don't log sensitive headers
- Implement certificate validation for custom CAs

Next Steps:
- ../error_handling.rs: Handle HTTP errors gracefully
- ../performance.rs: Optimize for high-throughput scenarios
- ../monitoring.rs: Add metrics and observability
*/
