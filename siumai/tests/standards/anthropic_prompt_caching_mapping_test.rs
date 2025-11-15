#![cfg(feature = "std-anthropic-external")]

//! Anthropic Prompt Caching mapping tests
//!
//! 验证 ProviderOptions::Anthropic.prompt_caching 经由
//! `anthropic_like_chat_request_to_core_input` → `ChatInput::extra` →
//! `build_messages_payload` 后，正确映射为 Anthropic Messages
//! 请求中的 `cache_control` 字段。

use siumai::core::provider_spec::anthropic_like_chat_request_to_core_input;
use siumai::types::{
    AnthropicCacheControl, AnthropicCacheType, AnthropicOptions, ChatMessage, ChatRequest,
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

