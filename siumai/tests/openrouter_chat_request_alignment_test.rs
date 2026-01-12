#![cfg(feature = "openai")]

use reqwest::header::HeaderMap;
use siumai::experimental::core::ProviderContext;
use siumai::experimental::core::ProviderSpec;
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("openrouter").expect("openrouter provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let mut extra_headers = std::collections::HashMap::new();
    extra_headers.insert(
        "HTTP-Referer".to_string(),
        "https://example.com".to_string(),
    );
    extra_headers.insert("X-Title".to_string(), "siumai-test".to_string());

    let ctx = ProviderContext::new(
        "openrouter".to_string(),
        provider_config.base_url,
        Some("sk-openrouter-test".to_string()),
        extra_headers,
    );

    (ctx, adapter)
}

#[test]
fn openrouter_chat_url_is_openai_compatible() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "openai/gpt-4o".to_string();

    assert_eq!(
        spec.chat_url(false, &req, &ctx),
        "https://openrouter.ai/api/v1/chat/completions"
    );
}

#[test]
fn openrouter_headers_include_auth_and_pass_through_attribution_headers() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let headers = spec.build_headers(&ctx).expect("headers");
    let auth = headers
        .get(reqwest::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert_eq!(auth, "Bearer sk-openrouter-test");

    assert_eq!(
        headers.get("HTTP-Referer").and_then(|v| v.to_str().ok()),
        Some("https://example.com")
    );
    assert_eq!(
        headers.get("X-Title").and_then(|v| v.to_str().ok()),
        Some("siumai-test")
    );
}

#[test]
fn openrouter_provider_options_are_merged_into_request_body() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "openai/gpt-4o".to_string();
    req = req.with_provider_option(
        "openrouter",
        serde_json::json!({
            "transforms": ["middle-out"],
            "someVendorParam": true
        }),
    );

    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let body = bundle.request.transform_chat(&req).expect("transform");

    let hook = spec
        .chat_before_send(&req, &ctx)
        .expect("before_send hook for providerOptions");
    let out = hook(&body).expect("merged body");

    assert_eq!(
        out.get("model").and_then(|v| v.as_str()),
        Some("openai/gpt-4o")
    );
    assert_eq!(
        out.get("transforms")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str()),
        Some("middle-out")
    );
    assert_eq!(
        out.get("someVendorParam").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn openrouter_error_mapping_is_openai_compatible() {
    let body = r#"{"error":{"message":"bad request","type":"invalid_request_error","code":null}}"#;
    let spec = OpenAiCompatibleSpecWithAdapter::new(make_ctx().1);
    let err = spec
        .classify_http_error(400, body, &HeaderMap::new())
        .expect("classified");
    match err {
        LlmError::InvalidInput(msg) => assert_eq!(msg, "bad request"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
