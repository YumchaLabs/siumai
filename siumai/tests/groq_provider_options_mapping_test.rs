#![cfg(feature = "groq")]

//! Groq ProviderOptions → ChatInput::extra 映射测试
//!
//! 验证 `ProviderOptions::Groq` 经由
//! `groq_chat_request_to_core_input` 正确写入
//! `ChatInput::extra["groq_extra_params"]`。

use serde_json::json;
use siumai::core::provider_spec::groq_chat_request_to_core_input;
use siumai::types::{ChatMessage, ChatRequest, GroqOptions};

#[test]
fn groq_options_extra_params_are_mapped_into_chatinput_extra() {
    // 基本 ChatRequest
    let messages = vec![ChatMessage::user("Hello, Groq!").build()];

    let options = GroqOptions::new()
        .with_param("service_tier", json!("lite"))
        .with_param("some_flag", json!(true));

    let req = ChatRequest::new(messages).with_groq_options(options);

    // 聚合层：将 typed GroqOptions 映射到 ChatInput::extra。
    let core_input = groq_chat_request_to_core_input(&req);

    let extra = core_input
        .extra
        .get("groq_extra_params")
        .expect("groq_extra_params should be present in ChatInput::extra");

    let obj = extra
        .as_object()
        .expect("groq_extra_params should be a JSON object");

    assert_eq!(
        obj.get("service_tier").and_then(|v| v.as_str()),
        Some("lite")
    );
    assert_eq!(obj.get("some_flag").and_then(|v| v.as_bool()), Some(true));
}
