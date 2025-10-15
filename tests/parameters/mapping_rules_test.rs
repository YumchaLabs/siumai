//! Mapping rules tests focusing on stable, spec-aligned validations and field placement.

use siumai::transformers::request::RequestTransformer;

#[test]
fn openai_o1_models_forbid_temperature_and_top_p() {
    use siumai::providers::openai::transformers::OpenAiRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole};

    // Build request targeting an o1-* model with temperature set
    let req = ChatRequest {
        messages: vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hi".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }],
        tools: None,
        common_params: CommonParams { model: "o1-mini".to_string(), temperature: Some(0.5), top_p: Some(0.9), max_tokens: None, stop_sequences: None, seed: None },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    let err = OpenAiRequestTransformer.transform_chat(&req).unwrap_err().to_string();
    assert!(err.contains("o1 models do not support temperature parameter"));
}

#[test]
fn openai_tools_upper_bound() {
    use siumai::providers::openai::transformers::OpenAiRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole, Tool};

    let tools: Vec<Tool> = (0..129)
        .map(|i| Tool::function(format!("f{i}"), "", serde_json::json!({})))
        .collect();

    let req = ChatRequest {
        messages: vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hi".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }],
        tools: Some(tools),
        common_params: CommonParams { model: "gpt-4o".to_string(), temperature: None, top_p: None, max_tokens: None, stop_sequences: None, seed: None },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    let err = OpenAiRequestTransformer.transform_chat(&req).unwrap_err().to_string();
    assert!(err.contains("maximum 128 tools"));
}

#[test]
fn gemini_generation_config_mapping() {
    use siumai::providers::gemini::transformers::GeminiRequestTransformer;
    use siumai::providers::gemini::types::GeminiConfig;
    use siumai::transformers::request::RequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole};

    let cfg = GeminiConfig { api_key: String::new(), base_url: String::new(), model: "gemini-1.5-pro".into(), generation_config: None, safety_settings: None, timeout: None };
    let req = ChatRequest {
        messages: vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hi".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }],
        tools: None,
        common_params: CommonParams { model: "gemini-1.5-pro".to_string(), temperature: Some(0.7), top_p: Some(0.9), max_tokens: Some(1000), stop_sequences: Some(vec!["STOP".into()]), seed: None },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    let json = GeminiRequestTransformer { config: cfg }.transform_chat(&req).expect("map");
    let gen = &json["generationConfig"];
    assert_eq!(gen["maxOutputTokens"], 1000);
    assert!((gen["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    assert!((gen["topP"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(gen["stopSequences"], serde_json::json!(["STOP"]));
}

#[test]
fn anthropic_stable_range_checks() {
    use siumai::providers::anthropic::transformers::AnthropicRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole};

    let req = ChatRequest {
        messages: vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hi".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }],
        tools: None,
        common_params: CommonParams { model: "claude-3-5-sonnet".to_string(), temperature: Some(1.5), top_p: None, max_tokens: None, stop_sequences: None, seed: None },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    let err = AnthropicRequestTransformer::new(None).transform_chat(&req).unwrap_err().to_string();
    assert!(err.contains("temperature must be between 0.0 and 1.0"));
}

