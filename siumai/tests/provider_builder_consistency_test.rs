//! Provider Builder API Consistency Tests
//!
//! This test suite ensures that all provider builders have consistent APIs
//! for core functionality that should be available across all providers.
//!
//! Tests verify that:
//! - All providers support timeout configuration
//! - All providers support custom HTTP clients
//! - All providers support HTTP interceptors
//! - All providers support debug mode
//! - All providers support tracing configuration

use siumai::Provider;
use siumai::execution::http::interceptor::HttpInterceptor;
use std::sync::Arc;
use std::time::Duration;

/// Mock HTTP interceptor for testing
#[derive(Clone)]
struct MockInterceptor;

impl HttpInterceptor for MockInterceptor {
    fn on_before_send(
        &self,
        _ctx: &siumai::execution::http::interceptor::HttpRequestContext,
        builder: reqwest::RequestBuilder,
        _body: &serde_json::Value,
        _headers: &reqwest::header::HeaderMap,
    ) -> Result<reqwest::RequestBuilder, siumai::LlmError> {
        Ok(builder)
    }
}

/// Test that all providers support timeout() method
#[test]
fn test_all_providers_support_timeout() {
    let timeout = Duration::from_secs(30);

    // OpenAI
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .timeout(timeout);

    // Anthropic
    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .timeout(timeout);

    // xAI
    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .timeout(timeout);

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .timeout(timeout);

    // Groq
    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .timeout(timeout);

    // Ollama
    let _ = Provider::ollama().model("llama3.2").timeout(timeout);
}

/// Test that all providers support connect_timeout() method
#[test]
fn test_all_providers_support_connect_timeout() {
    let timeout = Duration::from_secs(10);

    // OpenAI
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .connect_timeout(timeout);

    // Anthropic
    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .connect_timeout(timeout);

    // xAI
    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .connect_timeout(timeout);

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .connect_timeout(timeout);

    // Groq
    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .connect_timeout(timeout);

    // Ollama
    let _ = Provider::ollama()
        .model("llama3.2")
        .connect_timeout(timeout);
}

/// Test that all providers support with_http_client() method
#[test]
fn test_all_providers_support_custom_http_client() {
    let client = reqwest::Client::new();

    // OpenAI
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .with_http_client(client.clone());

    // Anthropic
    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .with_http_client(client.clone());

    // xAI
    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .with_http_client(client.clone());

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .with_http_client(client.clone());

    // Groq
    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .with_http_client(client.clone());

    // Ollama
    let _ = Provider::ollama()
        .model("llama3.2")
        .with_http_client(client);
}

/// Test that all providers support http_debug() method
#[test]
fn test_all_providers_support_http_debug() {
    // OpenAI
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .http_debug(true);

    // Anthropic
    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .http_debug(true);

    // xAI
    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .http_debug(true);

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .http_debug(true);

    // Groq
    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .http_debug(true);

    // Ollama
    let _ = Provider::ollama().model("llama3.2").http_debug(true);
}

/// Test that all providers support with_http_interceptor() method
#[test]
fn test_all_providers_support_http_interceptor() {
    let interceptor = Arc::new(MockInterceptor);

    // OpenAI
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .with_http_interceptor(interceptor.clone());

    // Anthropic
    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .with_http_interceptor(interceptor.clone());

    // xAI
    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .with_http_interceptor(interceptor.clone());

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .with_http_interceptor(interceptor.clone());

    // Groq
    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .with_http_interceptor(interceptor.clone());

    // Ollama
    let _ = Provider::ollama()
        .model("llama3.2")
        .with_http_interceptor(interceptor);
}

/// Test that all providers support tracing configuration methods
#[test]
fn test_all_providers_support_tracing() {
    // OpenAI
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);

    // Anthropic
    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);

    // xAI
    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);

    // Groq
    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);

    // Ollama
    let _ = Provider::ollama()
        .model("llama3.2")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);
}

/// Test that all providers support http_stream_disable_compression() method
#[test]
fn test_all_providers_support_http_stream_disable_compression() {
    // OpenAI
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .http_stream_disable_compression(true);

    // Anthropic
    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .http_stream_disable_compression(true);

    // xAI
    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .http_stream_disable_compression(true);

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .http_stream_disable_compression(true);

    // Groq
    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .http_stream_disable_compression(true);

    // Ollama
    let _ = Provider::ollama()
        .model("llama3.2")
        .http_stream_disable_compression(true);
}

/// Test that all providers support method chaining
#[test]
fn test_all_providers_support_method_chaining() {
    let timeout = Duration::from_secs(30);
    let client = reqwest::Client::new();
    let interceptor = Arc::new(MockInterceptor);

    // Test full chain on each provider
    let _ = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true)
        .debug_tracing();

    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true)
        .debug_tracing();

    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true)
        .debug_tracing();

    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true)
        .debug_tracing();

    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true)
        .debug_tracing();

    let _ = Provider::ollama()
        .model("llama3.2")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client)
        .http_debug(true)
        .with_http_interceptor(interceptor)
        .http_stream_disable_compression(true)
        .debug_tracing();
}
