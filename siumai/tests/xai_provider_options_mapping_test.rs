#![cfg(feature = "xai")]

//! xAI ProviderOptions → ChatInput::extra 映射测试
//!
//! 验证 `ProviderOptions::Xai` 经由
//! `xai_chat_request_to_core_input` 正确写入
//! `ChatInput::extra["xai_search_parameters" | "xai_reasoning_effort"]`。

use siumai::core::provider_spec::xai_chat_request_to_core_input;
use siumai::types::{ChatMessage, ChatRequest, SearchMode, XaiOptions, XaiSearchParameters};

#[test]
fn xai_options_are_mapped_into_chatinput_extra() {
    // 基本 ChatRequest
    let messages = vec![ChatMessage::user("Hello, xAI!").build()];

    let search = XaiSearchParameters {
        mode: SearchMode::On,
        return_citations: Some(true),
        max_search_results: Some(10),
        from_date: Some("2024-01-01".to_string()),
        to_date: Some("2024-02-01".to_string()),
        sources: None,
    };

    let options = XaiOptions::new()
        .with_search(search)
        .with_reasoning_effort("high");

    let req = ChatRequest::new(messages).with_xai_options(options);

    // 聚合层：将 typed XaiOptions 映射到 ChatInput::extra。
    let core_input = xai_chat_request_to_core_input(&req);

    // 检查 reasoning_effort 是否写入 extra。
    let effort = core_input
        .extra
        .get("xai_reasoning_effort")
        .expect("xai_reasoning_effort should be present in ChatInput::extra");
    assert_eq!(effort.as_str(), Some("high"));

    // 检查 search_parameters 是否写入 extra，并保留关键字段。
    let search_params = core_input
        .extra
        .get("xai_search_parameters")
        .expect("xai_search_parameters should be present in ChatInput::extra");

    let obj = search_params
        .as_object()
        .expect("xai_search_parameters should be a JSON object");

    // mode 采用 lower-case 序列化（参见 SearchMode 的 serde 配置）。
    assert_eq!(obj.get("mode").and_then(|v| v.as_str()), Some("on"));
    assert_eq!(
        obj.get("return_citations").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        obj.get("max_search_results").and_then(|v| v.as_u64()),
        Some(10)
    );
}
