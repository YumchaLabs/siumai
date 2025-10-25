//! Verify OpenAI request transformer adds `stream` and `stream_options` when streaming

use siumai::providers::openai::transformers::request::OpenAiRequestTransformer;
use siumai::execution::transformers::request::RequestTransformer;
use siumai::types::{ChatRequest, CommonParams, OpenAiOptions};

#[test]
fn openai_stream_request_includes_stream_options() {
    let mut req = ChatRequest::default();
    req.common_params = CommonParams { model: "gpt-4o-mini".to_string(), ..Default::default() };
    req.stream = true;
    // Ensure OpenAI path by setting provider options
    req = req.with_openai_options(OpenAiOptions::new());
    let transformer = OpenAiRequestTransformer;
    let body = transformer.transform_chat(&req).expect("transform ok");
    assert_eq!(body.get("stream").and_then(|v| v.as_bool()), Some(true));
    let include_usage = body
        .get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(|v| v.as_bool());
    assert_eq!(include_usage, Some(true));
}
