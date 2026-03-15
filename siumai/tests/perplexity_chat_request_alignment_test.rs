#![cfg(feature = "openai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai::provider_ext::perplexity::{
    PerplexityChatRequestExt, PerplexityOptions, PerplexitySearchContextSize, PerplexitySearchMode,
    PerplexitySearchRecencyFilter, PerplexityUserLocation,
};
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("perplexity").expect("perplexity provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let ctx = ProviderContext::new(
        "perplexity".to_string(),
        provider_config.base_url,
        Some("pplx-test".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn perplexity_provider_options_are_merged_into_request_body() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_provider_option(
            "perplexity",
            serde_json::json!({
                "search_mode": "academic",
                "someVendorParam": true
            }),
        );

    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let body = bundle.request.transform_chat(&req).expect("transform");
    let hook = spec
        .chat_before_send(&req, &ctx)
        .expect("before_send hook for providerOptions");
    let out = hook(&body).expect("merged body");

    assert_eq!(out.get("model").and_then(|v| v.as_str()), Some("sonar"));
    assert_eq!(
        out.get("search_mode").and_then(|v| v.as_str()),
        Some("academic")
    );
    assert_eq!(
        out.get("someVendorParam").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn perplexity_typed_options_are_merged_into_request_body() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_perplexity_options(
            PerplexityOptions::new()
                .with_search_mode(PerplexitySearchMode::Academic)
                .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
                .with_return_images(true)
                .with_search_context_size(PerplexitySearchContextSize::High)
                .with_user_location(PerplexityUserLocation::new().with_country("US"))
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let body = bundle.request.transform_chat(&req).expect("transform");
    let hook = spec
        .chat_before_send(&req, &ctx)
        .expect("before_send hook for providerOptions");
    let out = hook(&body).expect("merged body");

    assert_eq!(out.get("model").and_then(|v| v.as_str()), Some("sonar"));
    assert_eq!(
        out.get("search_mode").and_then(|v| v.as_str()),
        Some("academic")
    );
    assert_eq!(
        out.get("search_recency_filter").and_then(|v| v.as_str()),
        Some("month")
    );
    assert_eq!(
        out.get("return_images").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        out.get("web_search_options")
            .and_then(|v| v.get("search_context_size"))
            .and_then(|v| v.as_str()),
        Some("high")
    );
    assert_eq!(
        out.get("web_search_options")
            .and_then(|v| v.get("user_location"))
            .and_then(|v| v.get("country"))
            .and_then(|v| v.as_str()),
        Some("US")
    );
    assert_eq!(
        out.get("someVendorParam").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn perplexity_stable_fields_win_over_raw_provider_options() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let req = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({ "type": "object", "properties": {} }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_provider_option(
            "perplexity",
            serde_json::json!({
                "search_mode": "academic",
                "response_format": { "type": "json_object" },
                "tool_choice": "auto"
            }),
        );

    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let body = bundle.request.transform_chat(&req).expect("transform");
    let hook = spec
        .chat_before_send(&req, &ctx)
        .expect("before_send hook for providerOptions");
    let out = hook(&body).expect("merged body");

    assert_eq!(
        out.get("search_mode").and_then(|v| v.as_str()),
        Some("academic")
    );
    assert_eq!(out.get("tool_choice"), Some(&serde_json::json!("none")));
    assert_eq!(
        out.get("response_format"),
        Some(&serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": true
            }
        }))
    );
}
