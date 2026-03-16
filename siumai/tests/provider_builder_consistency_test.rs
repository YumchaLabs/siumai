#![cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
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
use siumai::experimental::execution::http::interceptor::HttpInterceptor;
use siumai::prelude::unified::LlmError;
use std::sync::Arc;
use std::time::Duration;

/// Mock HTTP interceptor for testing
#[derive(Clone)]
struct MockInterceptor;

impl HttpInterceptor for MockInterceptor {
    fn on_before_send(
        &self,
        _ctx: &siumai::experimental::execution::http::interceptor::HttpRequestContext,
        builder: reqwest::RequestBuilder,
        _body: &serde_json::Value,
        _headers: &reqwest::header::HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
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

    // MiniMaxi
    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
        .timeout(timeout);
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

    // MiniMaxi
    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
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
        .with_http_client(client.clone());

    // MiniMaxi
    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
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

    // MiniMaxi
    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
        .http_debug(true);
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
        .with_http_interceptor(interceptor.clone());

    // MiniMaxi
    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
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

    // Gemini
    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);

    // Ollama
    let _ = Provider::ollama()
        .model("llama3.2")
        .debug_tracing()
        .pretty_json(true)
        .mask_sensitive_values(true);

    // MiniMaxi
    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
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

    // MiniMaxi
    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
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
        .http_stream_disable_compression(true);

    let _ = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true);

    let _ = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true);

    let _ = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true);

    let _ = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true);

    let _ = Provider::ollama()
        .model("llama3.2")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client.clone())
        .http_debug(true)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(true);

    let _ = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
        .timeout(timeout)
        .connect_timeout(timeout)
        .with_http_client(client)
        .http_debug(true)
        .with_http_interceptor(interceptor)
        .http_stream_disable_compression(true);
}

fn assert_common_http_config_shape(
    cfg: &siumai::prelude::unified::HttpConfig,
    timeout: Duration,
    connect_timeout: Duration,
) {
    assert_eq!(cfg.timeout, Some(timeout));
    assert_eq!(cfg.connect_timeout, Some(connect_timeout));
    assert!(!cfg.stream_disable_compression);
}

/// Test that the top-level `Provider::ollama()` facade reaches the recent
/// config-first parity helpers without requiring direct provider-crate imports.
#[test]
fn test_provider_facade_ollama_builder_parity_helpers_survive_into_config() {
    use siumai::experimental::execution::middleware::language_model::LanguageModelMiddleware;
    use siumai::prelude::unified::{CommonParams, HttpConfig};
    use siumai::provider_ext::ollama::OllamaParams;

    let timeout = Duration::from_secs(17);
    let http_config = HttpConfig {
        timeout: Some(timeout),
        ..Default::default()
    };
    let model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>> = Vec::new();

    let config = Provider::ollama()
        .with_common_params(CommonParams {
            model: "llama3.2".to_string(),
            temperature: Some(0.4),
            ..Default::default()
        })
        .with_ollama_params(OllamaParams {
            raw: Some(true),
            think: Some(true),
            ..Default::default()
        })
        .stop(vec!["END".to_string()])
        .think(false)
        .with_http_config(http_config)
        .with_model_middlewares(model_middlewares)
        .into_config()
        .expect("ollama into_config");

    assert_eq!(config.model.as_deref(), Some("llama3.2"));
    assert_eq!(config.common_params.model, "llama3.2");
    assert_eq!(config.common_params.temperature, Some(0.4));
    assert_eq!(config.ollama_params.raw, Some(true));
    assert_eq!(config.ollama_params.think, Some(false));
    assert_eq!(config.ollama_params.stop, Some(vec!["END".to_string()]));
    assert_eq!(config.http_config.timeout, Some(timeout));
}

/// Test that major provider builders expose `into_config()` and preserve the common
/// inherited HTTP/builder settings there instead of re-deriving them later in `build()`.
#[test]
fn test_major_provider_builders_into_config_preserve_common_http_settings() {
    let timeout = Duration::from_secs(30);
    let connect_timeout = Duration::from_secs(10);
    let interceptor = Arc::new(MockInterceptor);

    let openai = Provider::openai()
        .api_key("test")
        .model("gpt-4")
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(false)
        .into_config()
        .expect("openai into_config");
    assert_eq!(openai.common_params.model, "gpt-4");
    assert_common_http_config_shape(&openai.http_config, timeout, connect_timeout);
    assert_eq!(openai.http_interceptors.len(), 1);

    let anthropic = Provider::anthropic()
        .api_key("test")
        .model("claude-3-5-sonnet-20241022")
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(false)
        .into_config()
        .expect("anthropic into_config");
    assert_eq!(anthropic.common_params.model, "claude-3-5-sonnet-20241022");
    assert_common_http_config_shape(&anthropic.http_config, timeout, connect_timeout);
    assert_eq!(anthropic.http_interceptors.len(), 1);

    let xai = Provider::xai()
        .api_key("test")
        .model("grok-beta")
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(false)
        .into_config()
        .expect("xai into_config");
    assert_eq!(xai.common_params.model, "grok-beta");
    assert_common_http_config_shape(&xai.http_config, timeout, connect_timeout);
    assert_eq!(xai.http_interceptors.len(), 1);

    let gemini = Provider::gemini()
        .api_key("test")
        .model("gemini-1.5-flash")
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(false)
        .into_config()
        .expect("gemini into_config");
    assert_eq!(gemini.common_params.model, "gemini-1.5-flash");
    assert_common_http_config_shape(&gemini.http_config, timeout, connect_timeout);
    assert_eq!(gemini.http_interceptors.len(), 1);

    let groq = Provider::groq()
        .api_key("test")
        .model("llama-3.3-70b-versatile")
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(false)
        .into_config()
        .expect("groq into_config");
    assert_eq!(groq.common_params.model, "llama-3.3-70b-versatile");
    assert_common_http_config_shape(&groq.http_config, timeout, connect_timeout);
    assert_eq!(groq.http_interceptors.len(), 1);

    let ollama = Provider::ollama()
        .model("llama3.2")
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .with_http_interceptor(interceptor.clone())
        .http_stream_disable_compression(false)
        .into_config()
        .expect("ollama into_config");
    assert_eq!(ollama.common_params.model, "llama3.2");
    assert_common_http_config_shape(&ollama.http_config, timeout, connect_timeout);
    assert_eq!(ollama.http_interceptors.len(), 1);

    let minimaxi = Provider::minimaxi()
        .api_key("test")
        .model("MiniMax-M2")
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .with_http_interceptor(interceptor)
        .http_stream_disable_compression(false)
        .into_config()
        .expect("minimaxi into_config");
    assert_eq!(minimaxi.common_params.model, "MiniMax-M2");
    assert_common_http_config_shape(&minimaxi.http_config, timeout, connect_timeout);
    assert_eq!(minimaxi.http_interceptors.len(), 1);
}
