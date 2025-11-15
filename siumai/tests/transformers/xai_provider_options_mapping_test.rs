#![cfg(feature = "std-openai-external")]

//! Verify xAI ProviderOptions are mapped into request JSON
//! via the OpenAI standard + xAI adapters.

use siumai::core::{ProviderContext, ProviderSpec};
use siumai::execution::transformers::request::RequestTransformer;
use siumai::types::{ChatMessage, ChatRequest, CommonParams, ProviderOptions, XaiOptions};

#[test]
fn xai_search_and_reasoning_options_are_mapped_via_std_openai() {
    // Build XaiOptions with search_parameters + reasoning_effort
    let search_params = siumai::types::XaiSearchParameters::default();
    let xai_opts = XaiOptions::new()
        .with_search(search_params)
        .with_reasoning_effort("medium");

    let req = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .common_params(CommonParams {
            model: "grok-beta".to_string(),
            ..Default::default()
        })
        .provider_options(ProviderOptions::Xai(xai_opts))
        .build();

    let ctx = ProviderContext::new(
        "xai",
        "https://api.x.ai/v1",
        Some("test-key".to_string()),
        std::collections::HashMap::new(),
    );

    let spec = siumai::providers::xai::spec::XaiSpec::default();
    let transformers = spec.choose_chat_transformers(&req, &ctx);

    let body = transformers
        .request
        .transform_chat(&req)
        .expect("transform ok");

    // Verify that search_parameters and reasoning_effort were injected
    assert!(body.get("search_parameters").is_some());
    assert_eq!(
        body.get("reasoning_effort")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "medium"
    );
}

