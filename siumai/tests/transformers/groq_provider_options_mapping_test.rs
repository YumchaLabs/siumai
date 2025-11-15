#![cfg(feature = "std-openai-external")]

//! Verify Groq ProviderOptions are mapped into request JSON
//! via the OpenAI standard + Groq adapter.

use std::collections::HashMap;

use siumai::core::{ProviderContext, ProviderSpec};
use siumai::execution::transformers::request::RequestTransformer;
use siumai::types::{ChatMessage, ChatRequest, CommonParams, GroqOptions, ProviderOptions};

#[test]
fn groq_extra_params_are_mapped_via_std_openai() {
    let mut extra = HashMap::new();
    extra.insert("service_tier".to_string(), serde_json::json!("lite"));
    extra.insert("some_flag".to_string(), serde_json::json!(true));

    let groq_opts = GroqOptions { extra_params: extra };

    let req = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .common_params(CommonParams {
            model: "llama-3.1-70b".to_string(),
            ..Default::default()
        })
        .provider_options(ProviderOptions::Groq(groq_opts))
        .build();

    let ctx = ProviderContext::new(
        "groq",
        "https://api.groq.com/openai/v1",
        Some("test-key".to_string()),
        HashMap::new(),
    );

    let spec = siumai::providers::groq::spec::GroqSpec::default();
    let transformers = spec.choose_chat_transformers(&req, &ctx);

    let body = transformers
        .request
        .transform_chat(&req)
        .expect("transform ok");

    // Groq adapter should have merged extra_params into the root body.
    assert_eq!(
        body.get("service_tier")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "lite"
    );
    assert_eq!(
        body.get("some_flag").and_then(|v| v.as_bool()),
        Some(true)
    );
}

