//! Provider-Model Architecture Integration Tests
//!
//! Tests the new Provider-Model architecture to ensure it works correctly.

use siumai::provider_model::Provider;
use siumai::providers::openai::{OpenAiConfig, OpenAiProvider};

#[test]
fn test_openai_provider_creation() {
    let config = OpenAiConfig::new("test-key").with_model("gpt-4");
    let provider = OpenAiProvider::new(config);

    assert_eq!(provider.id(), "openai");
}

#[test]
fn test_openai_chat_model_creation() {
    let config = OpenAiConfig::new("test-key").with_model("gpt-4");
    let provider = OpenAiProvider::new(config);

    let chat_model = provider.chat("gpt-4");
    assert!(chat_model.is_ok());
}

#[test]
fn test_openai_embedding_model_creation() {
    let config = OpenAiConfig::new("test-key");
    let provider = OpenAiProvider::new(config);

    let embedding_model = provider.embedding("text-embedding-3-small");
    assert!(embedding_model.is_ok());
}

#[test]
fn test_openai_image_model_creation() {
    let config = OpenAiConfig::new("test-key");
    let provider = OpenAiProvider::new(config);

    let image_model = provider.image("dall-e-3");
    assert!(image_model.is_ok());
}

#[test]
fn test_openai_chat_executor_creation() {
    let config = OpenAiConfig::new("test-key").with_model("gpt-4");
    let provider = OpenAiProvider::new(config);

    let chat_model = provider.chat("gpt-4").expect("Failed to create chat model");
    let http_client = reqwest::Client::new();

    let executor = chat_model.create_executor(
        http_client,
        vec![], // interceptors
        vec![], // middlewares
        None,   // retry_options
    );

    assert_eq!(executor.provider_id, "openai");
    assert!(executor.interceptors.is_empty());
    assert!(executor.middlewares.is_empty());
}

#[test]
fn test_openai_executor_with_interceptors_and_middlewares() {
    use siumai::utils::http_interceptor::{HttpInterceptor, LoggingInterceptor};
    use std::sync::Arc;

    let config = OpenAiConfig::new("test-key").with_model("gpt-4");
    let provider = OpenAiProvider::new(config);

    let chat_model = provider.chat("gpt-4").expect("Failed to create chat model");
    let http_client = reqwest::Client::new();

    // Add interceptor
    let interceptor: Arc<dyn HttpInterceptor> = Arc::new(LoggingInterceptor);

    let executor = chat_model.create_executor(http_client, vec![interceptor], vec![], None);

    assert_eq!(executor.provider_id, "openai");
    assert_eq!(executor.interceptors.len(), 1);
}

#[test]
fn test_provider_unsupported_operations() {
    let config = OpenAiConfig::new("test-key");
    let provider = OpenAiProvider::new(config);

    // OpenAI doesn't support rerank (yet)
    let rerank_result = provider.rerank("some-model");
    assert!(rerank_result.is_err());

    if let Err(e) = rerank_result {
        assert!(e.to_string().contains("does not support rerank"));
    }
}
