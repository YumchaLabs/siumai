#![cfg(feature = "google")]
use siumai::execution::RequestTransformer;
use siumai::types::{ChatRequest, CommonParams};

#[test]
fn gemini_transformer_moves_common_params_into_generation_config() {
    // Prepare Gemini config with default base and bare model id
    let mut cfg = siumai::providers::gemini::types::GeminiConfig::default();
    cfg = cfg.with_model("gemini-2.5-flash".to_string());

    // Build a simple chat request with unified common_params set
    let cp = CommonParams {
        model: "gemini-2.5-flash".to_string(),
        temperature: Some(0.3),
        max_tokens: Some(999),
        max_completion_tokens: None,
        top_p: Some(0.7),
        stop_sequences: Some(vec!["END".to_string()]),
        seed: None,
    };
    let req = ChatRequest::builder()
        .messages(vec![siumai::user!("Hello")])
        .common_params(cp)
        .build();

    // Transform via provider request transformer
    let tx = siumai::providers::gemini::transformers::GeminiRequestTransformer { config: cfg };
    let body = tx.transform_chat(&req).expect("transform should succeed");

    // Assert model id remains bare (path builder adds /models/{id})
    assert_eq!(body["model"], "gemini-2.5-flash");

    // Assert generationConfig fields picked from common_params
    let gen_cfg = body
        .get("generationConfig")
        .expect("generationConfig present");
    assert_eq!(gen_cfg["temperature"], 0.3);
    assert_eq!(gen_cfg["topP"], 0.7);
    assert_eq!(gen_cfg["maxOutputTokens"], 999);
    assert_eq!(gen_cfg["stopSequences"], serde_json::json!(["END"]));
}

#[test]
fn gemini_system_messages_are_mapped_to_system_instruction() {
    // 准备 Gemini 配置
    let mut cfg = siumai::providers::gemini::types::GeminiConfig::default();
    cfg = cfg.with_model("gemini-2.5-flash".to_string());

    // 构造带有 system + user + assistant 的对话
    let messages = vec![
        siumai::system!("You are a helpful assistant."),
        siumai::user!("Hello"),
        siumai::assistant!("Hi, how can I help you?"),
    ];

    let cp = CommonParams {
        model: "gemini-2.5-flash".to_string(),
        ..Default::default()
    };

    let req = ChatRequest::builder()
        .messages(messages)
        .common_params(cp)
        .build();

    let tx = siumai::providers::gemini::transformers::GeminiRequestTransformer { config: cfg };
    let body = tx.transform_chat(&req).expect("transform should succeed");

    // system 消息应聚合到 systemInstruction.parts 中
    let sys = body
        .get("systemInstruction")
        .expect("systemInstruction should be present for system messages");
    let parts = sys
        .get("parts")
        .and_then(|v| v.as_array())
        .expect("systemInstruction.parts should be an array");
    assert_eq!(parts.len(), 1);
    assert_eq!(
        parts[0].get("text").and_then(|v| v.as_str()),
        Some("You are a helpful assistant.")
    );

    // contents 中只应包含 user/model 对话历史
    let contents = body
        .get("contents")
        .and_then(|v| v.as_array())
        .expect("contents should be an array");

    assert_eq!(
        contents.len(),
        2,
        "system message should not appear in contents"
    );

    // 第一条 user
    assert_eq!(
        contents[0].get("role").and_then(|v| v.as_str()),
        Some("user")
    );
    let user_parts = contents[0]
        .get("parts")
        .and_then(|v| v.as_array())
        .expect("user parts should be array");
    assert_eq!(
        user_parts[0].get("text").and_then(|v| v.as_str()),
        Some("Hello")
    );

    // 第二条 assistant → model
    assert_eq!(
        contents[1].get("role").and_then(|v| v.as_str()),
        Some("model")
    );
    let model_parts = contents[1]
        .get("parts")
        .and_then(|v| v.as_array())
        .expect("model parts should be array");
    assert_eq!(
        model_parts[0].get("text").and_then(|v| v.as_str()),
        Some("Hi, how can I help you?")
    );
}
