#![cfg(feature = "google-vertex")]

use siumai::Provider;
use std::sync::Arc;

#[test]
fn provider_anthropic_vertex_builder_requires_base_url() {
    let result = Provider::anthropic_vertex()
        .model("claude-3-5-sonnet-20241022")
        .build();
    assert!(result.is_err(), "missing base_url should be an error");

    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("base_url"), "unexpected error: {msg}");
}

#[test]
fn provider_anthropic_vertex_builder_requires_model_id() {
    let result = Provider::anthropic_vertex()
        .base_url("https://example.com/custom")
        .build();
    assert!(result.is_err(), "missing model should be an error");

    let msg = result.err().unwrap().to_string();
    assert!(
        msg.contains("requires a non-empty model id"),
        "unexpected error: {msg}"
    );
}

#[test]
fn provider_anthropic_vertex_builder_custom_base_url_is_preserved() {
    let client = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .model("claude-3-5-sonnet-20241022")
        .bearer_token("test-token")
        .build()
        .expect("build anthropic vertex client");

    assert_eq!(client.base_url(), "https://custom.example.com");
}

#[test]
fn provider_anthropic_vertex_builder_language_model_alias_sets_model() {
    let cfg = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .language_model("claude-3-5-sonnet-20241022")
        .bearer_token("test-token")
        .into_config()
        .expect("into_config");

    assert_eq!(cfg.model, "claude-3-5-sonnet-20241022");
}

#[test]
fn provider_anthropic_vertex_builder_bearer_token_helper_sets_authorization_header() {
    let cfg = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .model("claude-3-5-sonnet-20241022")
        .bearer_token("test-token")
        .into_config()
        .expect("into_config");

    assert_eq!(
        cfg.http_config
            .headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer test-token")
    );
}

#[test]
fn provider_anthropic_vertex_builder_authorization_helper_preserves_exact_value() {
    let cfg = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .model("claude-3-5-sonnet-20241022")
        .authorization("Bearer custom-token")
        .into_config()
        .expect("into_config");

    assert_eq!(
        cfg.http_config
            .headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer custom-token")
    );
}

#[test]
fn provider_anthropic_vertex_builder_token_provider_sets_config_field() {
    let cfg = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .model("claude-3-5-sonnet-20241022")
        .token_provider(Arc::new(
            siumai::experimental::auth::StaticTokenProvider::new("test-token"),
        ))
        .into_config()
        .expect("into_config");

    assert!(cfg.token_provider.is_some());
}
