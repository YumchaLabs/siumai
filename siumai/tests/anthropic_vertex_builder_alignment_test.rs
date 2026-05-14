#![cfg(feature = "google-vertex")]
#![allow(deprecated)]

use siumai::Provider;
use siumai::prelude::compat::Siumai;
use std::sync::Arc;

#[test]
fn provider_anthropic_vertex_builder_requires_base_url() {
    let result = Provider::anthropic_vertex()
        .model("claude-3-5-sonnet-v2@20241022")
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

#[tokio::test]
async fn siumai_builder_anthropic_vertex_requires_explicit_model_id() {
    let result = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://custom.example.com")
        .http_header("Authorization", "Bearer test-token")
        .build()
        .await;
    assert!(result.is_err(), "missing model should be an error");

    let msg = result.err().unwrap().to_string();
    assert!(
        msg.contains("Anthropic on Vertex requires an explicit model id"),
        "unexpected error: {msg}"
    );
}

#[test]
fn provider_anthropic_vertex_builder_custom_base_url_is_preserved() {
    let client = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .model("claude-3-5-sonnet-v2@20241022")
        .bearer_token("test-token")
        .build()
        .expect("build anthropic vertex client");

    assert_eq!(client.base_url(), "https://custom.example.com");
}

#[test]
fn provider_anthropic_vertex_builder_derives_base_url_from_project_and_location() {
    let cfg = Provider::anthropic_vertex()
        .project("demo-project")
        .location("global")
        .model("claude-3-5-sonnet-v2@20241022")
        .into_config()
        .expect("into_config");

    assert_eq!(
        cfg.base_url,
        siumai::experimental::auth::vertex::google_vertex_anthropic_base_url(
            "demo-project",
            "global"
        )
    );
}

#[test]
fn provider_anthropic_vertex_builder_language_model_alias_sets_model() {
    let cfg = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .language_model("claude-3-5-sonnet-v2@20241022")
        .bearer_token("test-token")
        .into_config()
        .expect("into_config");

    assert_eq!(cfg.model, "claude-3-5-sonnet-v2@20241022");
}

#[test]
fn provider_anthropic_vertex_builder_bearer_token_helper_sets_authorization_header() {
    let cfg = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .model("claude-3-5-sonnet-v2@20241022")
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
        .model("claude-3-5-sonnet-v2@20241022")
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
fn provider_vertex_anthropic_alias_matches_anthropic_vertex_builder() {
    let cfg = Provider::vertex_anthropic()
        .project("demo-project")
        .location("us-central1")
        .model("claude-3-5-sonnet-v2@20241022")
        .into_config()
        .expect("into_config");

    assert_eq!(
        cfg.base_url,
        siumai::experimental::auth::vertex::google_vertex_anthropic_base_url(
            "demo-project",
            "us-central1"
        )
    );
}

#[test]
fn provider_anthropic_vertex_builder_token_provider_sets_config_field() {
    let cfg = Provider::anthropic_vertex()
        .base_url("https://custom.example.com")
        .model("claude-3-5-sonnet-v2@20241022")
        .token_provider(Arc::new(
            siumai::experimental::auth::StaticTokenProvider::new("test-token"),
        ))
        .into_config()
        .expect("into_config");

    assert!(cfg.token_provider.is_some());
}

#[tokio::test]
async fn siumai_builder_anthropic_vertex_accepts_project_and_location_without_explicit_base_url() {
    let result = Siumai::builder()
        .vertex_anthropic()
        .project("demo-project")
        .location("global")
        .model("claude-3-5-sonnet-v2@20241022")
        .http_header("Authorization", "Bearer test-token")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "project+location should synthesize base_url"
    );
}
