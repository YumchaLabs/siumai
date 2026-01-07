#![cfg(feature = "xai")]

use siumai::experimental::execution::transformers::request::RequestTransformer;
use siumai::prelude::unified::*;

#[test]
fn xai_code_execution_tool_maps_to_code_interpreter_in_responses_request() {
    let req = ChatRequest {
        messages: vec![user!("hello")],
        tools: Some(vec![tools::xai::code_execution()]),
        common_params: CommonParams {
            model: "grok-4-fast".to_string(),
            ..Default::default()
        },
        stream: false,
        ..Default::default()
    };

    let tx =
        siumai::experimental::standards::openai::transformers::request::OpenAiResponsesRequestTransformer;
    let body = tx.transform_chat(&req).expect("transform request");

    let tools = body.get("tools").and_then(|v| v.as_array()).expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(
        tools[0].get("type").and_then(|v| v.as_str()),
        Some("code_interpreter")
    );
}
