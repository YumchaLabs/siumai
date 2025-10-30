//! Extra mapping rules tests for Groq/xAI/OpenAI Responses
use siumai::execution::transformers::request::RequestTransformer;

#[test]
fn groq_chat_stable_ranges_and_stop_mapping() {
    use siumai::providers::groq::transformers::GroqRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole};

    // temperature out of stable range should error
    let req_err = ChatRequest::builder()
        .messages(vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hi".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }])
        .common_params(CommonParams { model: "llama-3.1-70b-versatile".into(), temperature: Some(3.0), ..Default::default() })
        .build();
    let err = GroqRequestTransformer.transform_chat(&req_err).unwrap_err().to_string();
    assert!(err.contains("temperature"));

    // stop_sequences -> stop mapping; stream flag preserved
    let req_ok = ChatRequest::builder()
        .messages(vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hi".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }])
        .common_params(CommonParams { model: "llama-3.1-70b-versatile".into(), stop_sequences: Some(vec!["END".into()]), ..Default::default() })
        .stream(true)
        .build();
    let json = GroqRequestTransformer.transform_chat(&req_ok).expect("map");
    assert_eq!(json["stop"], serde_json::json!(["END"]));
    assert_eq!(json["stream"], true);
}

#[test]
fn xai_chat_stable_ranges_and_merge() {
    use siumai::providers::xai::transformers::XaiRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole};

    // top_p out of range should error
    let req_err = ChatRequest::builder()
        .messages(vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hi".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }])
        .common_params(CommonParams { model: "grok-beta".into(), top_p: Some(1.5), ..Default::default() })
        .build();
    let err = XaiRequestTransformer.transform_chat(&req_err).unwrap_err().to_string();
    assert!(err.contains("top_p"));

    // Note: provider-specific custom merges now happen via provider hooks and typed ProviderOptions.
    // Transformer tests no longer verify arbitrary extra key merging.
}

#[test]
fn openai_responses_request_basic_shape() {
    use siumai::providers::openai::transformers::OpenAiResponsesRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole, Tool};

    let messages = vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("Hello".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }];
    let tools = Some(vec![Tool::function("sum", "", serde_json::json!({"type":"object"}))]);
    let req = ChatRequest::builder()
        .messages(messages)
        .tools(tools.unwrap())
        .common_params(CommonParams { model: "gpt-5-mini".into(), temperature: Some(0.2), max_tokens: Some(128), ..Default::default() })
        .stream(true)
        .build();
    let json = OpenAiResponsesRequestTransformer.transform_chat(&req).expect("map");
    assert_eq!(json["model"], "gpt-5-mini");
    assert_eq!(json["stream"], true);
    assert_eq!(json["stream_options"], serde_json::json!({"include_usage": true}));
    assert!(json["input"].is_array());
    assert_eq!(json["temperature"], 0.2);
    assert_eq!(json["max_output_tokens"], 128);
}

#[test]
fn openai_rerank_transformer_payload() {
    use siumai::providers::openai::transformers::OpenAiRequestTransformer;
    use siumai::execution::transformers::request::RequestTransformer;
    use siumai::types::RerankRequest;

    let req = RerankRequest::new(
        "rerank-model".into(),
        "hello".into(),
        vec!["doc1".into(), "doc2".into()],
    )
    .with_top_n(3)
    .with_return_documents(true);
    let json = OpenAiRequestTransformer.transform_rerank(&req).expect("transform rerank");
    assert_eq!(json["model"], "rerank-model");
    assert_eq!(json["query"], "hello");
    assert_eq!(json["top_n"], 3);
    assert_eq!(json["return_documents"], true);
}

#[test]
fn openai_moderation_transformer_payload() {
    use siumai::providers::openai::transformers::OpenAiRequestTransformer;
    use siumai::execution::transformers::request::RequestTransformer;
    use siumai::types::ModerationRequest;

    let req = ModerationRequest { input: "some text".into(), model: None };
    let json = OpenAiRequestTransformer.transform_moderation(&req).expect("transform moderation");
    assert_eq!(json["input"], "some text");
    assert!(json["model"].is_string());
}
