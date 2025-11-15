#![cfg(feature = "std-openai-external")]

//! Verify OpenAI ResponsesApiConfig ProviderOptions mapping
//! via the std-openai Responses standard.

use std::collections::HashMap;

use siumai::core::{ProviderContext, ProviderSpec};
use siumai::execution::transformers::request::RequestTransformer;
use siumai::types::{
    ChatMessage, ChatRequest, CommonParams, OpenAiOptions, ProviderOptions,
};
use siumai::types::provider_options::openai::{
    ResponsesApiConfig, TextVerbosity, Truncation,
};

/// Ensure that ResponsesApiConfig fields are correctly mapped into
/// the final Responses API JSON body when using std-openai-external.
#[test]
fn openai_responses_provider_options_are_mapped_via_std_openai() {
    // Build a minimal ChatRequest targeting Responses API.
    let mut metadata = HashMap::new();
    metadata.insert("foo".to_string(), "bar".to_string());

    let responses_cfg = ResponsesApiConfig::new()
        .with_previous_response("resp_123".to_string())
        .with_response_format(serde_json::json!({ "type": "json_schema" }))
        .with_background(true)
        .with_include(vec!["file_search_call.results".to_string()])
        .with_instructions("Follow these instructions".to_string())
        .with_max_tool_calls(8)
        .with_store(true)
        .with_truncation(Truncation::Auto)
        .with_text_verbosity(TextVerbosity::High)
        .with_metadata(metadata.clone())
        .with_parallel_tool_calls(true);

    let options = OpenAiOptions::new().with_responses_api(responses_cfg);

    let req = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .common_params(CommonParams {
            model: "gpt-4.1".to_string(),
            ..Default::default()
        })
        .provider_options(ProviderOptions::OpenAi(Box::new(options)))
        .build();

    let ctx = ProviderContext::new(
        "openai",
        "https://api.openai.com/v1",
        Some("test-key".to_string()),
        HashMap::new(),
    );

    let spec = siumai::providers::openai::spec::OpenAiSpec::new();
    let txs = spec.choose_chat_transformers(&req, &ctx);

    let body = txs
        .request
        .transform_chat(&req)
        .expect("transform via ResponsesRequestBridge");

    // Basic shape: model + input[]
    assert_eq!(body.get("model").and_then(|v| v.as_str()), Some("gpt-4.1"));
    assert!(body.get("input").and_then(|v| v.as_array()).is_some());

    // Core responses fields
    assert_eq!(body.get("stream").and_then(|v| v.as_bool()), Some(false));

    // ResponsesApiConfig → JSON body
    assert_eq!(
        body.get("previous_response_id")
            .and_then(|v| v.as_str()),
        Some("resp_123")
    );
    assert_eq!(
        body.get("response_format")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str()),
        Some("json_schema")
    );
    assert_eq!(
        body.get("background").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        body.get("include").and_then(|v| v.as_array()).map(|a| a.len()),
        Some(1)
    );
    assert_eq!(
        body.get("instructions")
            .and_then(|v| v.as_str()),
        Some("Follow these instructions")
    );
    assert_eq!(
        body.get("max_tool_calls").and_then(|v| v.as_u64()),
        Some(8)
    );
    assert_eq!(body.get("store").and_then(|v| v.as_bool()), Some(true));

    // truncation
    assert!(body.get("truncation").is_some());

    // text_verbosity nested under text.verbosity
    assert_eq!(
        body.get("text")
            .and_then(|t| t.get("verbosity"))
            .and_then(|v| v.as_str()),
        Some("high")
    );

    // metadata
    let meta = body
        .get("metadata")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    assert_eq!(meta.get("foo").and_then(|v| v.as_str()), Some("bar"));

    // parallel_tool_calls
    assert_eq!(
        body.get("parallel_tool_calls")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
}

