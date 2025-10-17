//! Verify OpenAI request transformer adds `stream` and `stream_options` when streaming

use siumai::providers::openai::transformers::request::OpenAiRequestTransformer;
use siumai::transformers::request::RequestTransformer;
use siumai::types::{ChatRequest, ProviderParams, CommonParams};

#[test]
fn openai_stream_request_includes_stream_options() {
    let req = ChatRequest {
        messages: vec![],
        tools: None,
        common_params: CommonParams {
            model: "gpt-4o-mini".to_string(),
            ..Default::default()
        },
        provider_params: Some(ProviderParams::OpenAi(Default::default())),
        http_config: None,
        web_search: None,
        stream: true,
    };
    let transformer = OpenAiRequestTransformer;
    let body = transformer.transform_chat(&req).expect("transform ok");
    assert_eq!(body.get("stream").and_then(|v| v.as_bool()), Some(true));
    let include_usage = body
        .get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(|v| v.as_bool());
    assert_eq!(include_usage, Some(true));
}

