#![cfg(feature = "std-anthropic-external")]

//! Anthropic ProviderOptions → ChatInput::extra 映射测试
//!
//! 覆盖三类配置：
//! - Prompt Caching：经由 `anthropic_like_chat_request_to_core_input`
//!   → `ChatInput::extra["anthropic_prompt_caching"]` →
//!   `build_messages_payload` 映射为 Messages `cache_control` 字段。
//! - Thinking Mode：映射为 `ChatInput::extra["anthropic_thinking"]`。
//! - Response Format：映射为 `ChatInput::extra["anthropic_response_format"]`。

use siumai::core::provider_spec::anthropic_like_chat_request_to_core_input;
use siumai::types::{
    AnthropicCacheControl, AnthropicCacheType, AnthropicOptions, AnthropicResponseFormat,
    ChatMessage, ChatRequest, ThinkingModeConfig,
};
use siumai_std_anthropic::anthropic::utils::build_messages_payload;

#[test]
fn anthropic_prompt_caching_is_mapped_to_cache_control() {
    // 构造两条消息：system + user，期望对 user 消息应用缓存控制。
    let messages = vec![
        ChatMessage::system("You are Claude.").build(),
        ChatMessage::user("Hello").build(),
    ];

    // PromptCachingConfig：对 message_index = 1（第二条，即 user 消息）启用 ephemeral 缓存。
    let prompt_caching = siumai::types::PromptCachingConfig {
        enabled: true,
        cache_control: vec![AnthropicCacheControl {
            cache_type: AnthropicCacheType::Ephemeral,
            message_index: 1,
        }],
    };

    let options = AnthropicOptions::new().with_prompt_caching(prompt_caching);

    let req = ChatRequest::new(messages).with_anthropic_options(options);

    // 1) 聚合层：将 typed AnthropicOptions 映射到 ChatInput::extra。
    let core_input = anthropic_like_chat_request_to_core_input(&req);

    let extra = core_input
        .extra
        .get("anthropic_prompt_caching")
        .expect("ChatInput::extra should contain anthropic_prompt_caching");
    let arr = extra
        .as_array()
        .expect("anthropic_prompt_caching should be an array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["index"], 1);
    assert_eq!(arr[0]["cache_control"]["type"], "ephemeral");

    // 2) 标准层：将 ChatInput → Anthropic Messages JSON，并在目标 message 上挂载 cache_control。
    let (messages_json, system_opt) =
        build_messages_payload(&core_input).expect("build_messages_payload should succeed");

    // system message 聚合到单独的 system 字段中。
    assert_eq!(system_opt.as_deref(), Some("You are Claude."));

    // 只剩下 user 消息，且应包含 cache_control。
    assert_eq!(messages_json.len(), 1);
    let user_msg = &messages_json[0];
    assert_eq!(user_msg["role"], "user");
    assert_eq!(
        user_msg["content"][0]["text"],
        "Hello",
        "user content should be preserved"
    );
    assert_eq!(
        user_msg["cache_control"]["type"],
        "ephemeral",
        "cache_control.type should be mapped from PromptCachingConfig"
    );
}

#[test]
fn anthropic_thinking_mode_is_mapped_into_extra() {
    // 构造简单消息。
    let messages = vec![ChatMessage::user("Hello").build()];

    // ThinkingModeConfig：enabled=true + budget。
    let thinking = ThinkingModeConfig {
        enabled: true,
        thinking_budget: Some(4096),
    };

    let options = AnthropicOptions::new().with_thinking_mode(thinking);
    let req = ChatRequest::new(messages).with_anthropic_options(options);

    let core_input = anthropic_like_chat_request_to_core_input(&req);

    let thinking_val = core_input
        .extra
        .get("anthropic_thinking")
        .expect("ChatInput::extra should contain anthropic_thinking");

    let obj = thinking_val
        .as_object()
        .expect("anthropic_thinking should be a JSON object");
    assert_eq!(obj.get("type").and_then(|v| v.as_str()), Some("enabled"));
    assert_eq!(
        obj.get("budget_tokens").and_then(|v| v.as_u64()),
        Some(4096)
    );
}

#[test]
fn anthropic_response_format_is_mapped_into_extra() {
    let messages = vec![ChatMessage::user("Hello").build()];

    // 1) JsonObject 形式
    let opts_obj = AnthropicOptions::new().with_json_object();
    let req_obj = ChatRequest::new(messages.clone()).with_anthropic_options(opts_obj);
    let core_input_obj = anthropic_like_chat_request_to_core_input(&req_obj);

    let rf_obj = core_input_obj
        .extra
        .get("anthropic_response_format")
        .expect("extra should contain anthropic_response_format for JsonObject");
    let obj = rf_obj
        .as_object()
        .expect("anthropic_response_format should be a JSON object");
    assert_eq!(obj.get("type").and_then(|v| v.as_str()), Some("json_object"));

    // 2) JsonSchema 形式
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "foo": { "type": "string" } },
        "required": ["foo"]
    });

    let opts_schema = AnthropicOptions::new().with_json_schema("MySchema", schema.clone(), true);
    let req_schema = ChatRequest::new(messages).with_anthropic_options(opts_schema);
    let core_input_schema = anthropic_like_chat_request_to_core_input(&req_schema);

    let rf_schema = core_input_schema
        .extra
        .get("anthropic_response_format")
        .expect("extra should contain anthropic_response_format for JsonSchema");
    let obj = rf_schema
        .as_object()
        .expect("anthropic_response_format should be a JSON object");

    assert_eq!(
        obj.get("type").and_then(|v| v.as_str()),
        Some("json_schema")
    );

    let json_schema = obj
        .get("json_schema")
        .and_then(|v| v.as_object())
        .expect("json_schema should be an object");

    assert_eq!(
        json_schema.get("name").and_then(|v| v.as_str()),
        Some("MySchema")
    );
    assert_eq!(
        json_schema.get("strict").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(json_schema.get("schema"), Some(&schema));
}
