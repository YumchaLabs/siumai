#![cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama"
))]
//! URL Compatibility Tests
//!
//! Tests to ensure all providers handle base URLs with and without trailing slashes correctly.

use siumai::prelude::*;
use siumai::utils::url::{join_url, normalize_url};

#[test]
fn test_url_join_compatibility() {
    // Test cases that would previously cause double slashes
    assert_eq!(
        join_url("https://api.openai.com/v1/", "chat/completions"),
        "https://api.openai.com/v1/chat/completions"
    );
    assert_eq!(
        join_url("https://api.anthropic.com/", "v1/messages"),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        join_url("http://localhost:11434/", "api/chat"),
        "http://localhost:11434/api/chat"
    );

    // Test cases without trailing slash
    assert_eq!(
        join_url("https://api.openai.com/v1", "chat/completions"),
        "https://api.openai.com/v1/chat/completions"
    );
    assert_eq!(
        join_url("https://api.anthropic.com", "v1/messages"),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        join_url("http://localhost:11434", "api/chat"),
        "http://localhost:11434/api/chat"
    );
}

#[test]
fn test_normalize_url_compatibility() {
    // Test normalization of URLs with double slashes
    assert_eq!(
        normalize_url("https://api.openai.com//v1//chat//completions"),
        "https://api.openai.com/v1/chat/completions"
    );
    assert_eq!(
        normalize_url("https://api.anthropic.com//v1//messages"),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        normalize_url("http://localhost:11434//api//chat"),
        "http://localhost:11434/api/chat"
    );
}

#[tokio::test]
async fn test_provider_url_building() {
    // Test that all providers can be built with different URL formats
    let base_urls = vec![
        "https://api.example.com",
        "https://api.example.com/",
        "https://api.example.com//",
    ];

    for base_url in base_urls {
        // These should all succeed in building (though may fail on actual requests)
        let _openai = LlmBuilder::new()
            .openai()
            .base_url(base_url)
            .api_key("test")
            .model("gpt-4")
            .build()
            .await;

        let _anthropic = LlmBuilder::new()
            .anthropic()
            .base_url(base_url)
            .api_key("test")
            .model("claude-3-5-sonnet-20241022")
            .build()
            .await;

        let _ollama = LlmBuilder::new()
            .ollama()
            .base_url(base_url)
            .model("llama3.2")
            .build()
            .await;

        let _gemini = LlmBuilder::new()
            .gemini()
            .base_url(base_url)
            .api_key("test")
            .model("gemini-1.5-pro")
            .build()
            .await;
    }
}

#[test]
fn test_real_world_url_cases() {
    // Test real-world scenarios

    // OpenAI
    assert_eq!(
        join_url("https://api.openai.com/v1", "chat/completions"),
        "https://api.openai.com/v1/chat/completions"
    );
    assert_eq!(
        join_url("https://api.openai.com/v1/", "chat/completions"),
        "https://api.openai.com/v1/chat/completions"
    );

    // Anthropic
    assert_eq!(
        join_url("https://api.anthropic.com", "v1/messages"),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        join_url("https://api.anthropic.com/", "v1/messages"),
        "https://api.anthropic.com/v1/messages"
    );

    // Ollama
    assert_eq!(
        join_url("http://localhost:11434", "api/chat"),
        "http://localhost:11434/api/chat"
    );
    assert_eq!(
        join_url("http://localhost:11434/", "api/chat"),
        "http://localhost:11434/api/chat"
    );

    // Custom proxy with trailing slash (like in the user's example)
    assert_eq!(
        join_url("https://api1.oaipro.com/", "v1/messages"),
        "https://api1.oaipro.com/v1/messages"
    );
    assert_eq!(
        join_url("https://api1.oaipro.com/", "chat/completions"),
        "https://api1.oaipro.com/chat/completions"
    );

    // Gemini
    assert_eq!(
        join_url(
            "https://generativelanguage.googleapis.com/v1beta/",
            "models/gemini-1.5-pro:generateContent"
        ),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    );
}

#[test]
fn test_edge_cases() {
    // Multiple slashes
    assert_eq!(
        join_url("https://api.example.com///", "///v1/chat"),
        "https://api.example.com/v1/chat"
    );

    // Empty path
    assert_eq!(
        join_url("https://api.example.com", ""),
        "https://api.example.com"
    );
    assert_eq!(
        join_url("https://api.example.com/", ""),
        "https://api.example.com"
    );

    // Only slashes
    assert_eq!(
        join_url("https://api.example.com", "/"),
        "https://api.example.com"
    );
    assert_eq!(
        join_url("https://api.example.com/", "/"),
        "https://api.example.com"
    );
}

#[tokio::test]
async fn test_provider_specific_url_patterns() {
    // Test provider-specific URL patterns work correctly

    // OpenAI patterns
    let openai_urls = vec![
        (
            "chat/completions",
            "https://api.openai.com/v1/chat/completions",
        ),
        ("models", "https://api.openai.com/v1/models"),
        ("models/gpt-4", "https://api.openai.com/v1/models/gpt-4"),
    ];

    for (path, expected) in openai_urls {
        assert_eq!(join_url("https://api.openai.com/v1", path), expected);
        assert_eq!(join_url("https://api.openai.com/v1/", path), expected);
    }

    // Anthropic patterns
    let anthropic_urls = vec![("v1/messages", "https://api.anthropic.com/v1/messages")];

    for (path, expected) in anthropic_urls {
        assert_eq!(join_url("https://api.anthropic.com", path), expected);
        assert_eq!(join_url("https://api.anthropic.com/", path), expected);
    }

    // Ollama patterns
    let ollama_urls = vec![
        ("api/chat", "http://localhost:11434/api/chat"),
        ("api/generate", "http://localhost:11434/api/generate"),
        ("api/embed", "http://localhost:11434/api/embed"),
        ("api/tags", "http://localhost:11434/api/tags"),
    ];

    for (path, expected) in ollama_urls {
        assert_eq!(join_url("http://localhost:11434", path), expected);
        assert_eq!(join_url("http://localhost:11434/", path), expected);
    }
}
