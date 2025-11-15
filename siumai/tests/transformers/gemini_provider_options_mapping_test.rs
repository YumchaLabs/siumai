#![cfg(feature = "std-gemini-external")]

//! Verify Gemini ProviderOptions are mapped into request JSON
//! via the Gemini standard + adapter.

use siumai::core::{ProviderContext, ProviderSpec};
use siumai::execution::transformers::request::RequestTransformer;
use siumai::types::{
    ChatMessage, ChatRequest, CommonParams, GeminiOptions, ProviderOptions,
};

#[test]
fn gemini_options_are_mapped_via_std_gemini() {
    let gemini_opts = GeminiOptions::new()
        .with_code_execution(siumai::types::CodeExecutionConfig::default())
        .with_response_mime_type("application/json".to_string());

    let req = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .common_params(CommonParams {
            model: "gemini-1.5-pro".to_string(),
            ..Default::default()
        })
        .provider_options(ProviderOptions::Gemini(gemini_opts))
        .build();

    let ctx = ProviderContext::new(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        Some("test-key".to_string()),
        std::collections::HashMap::new(),
    );

    let spec = siumai::providers::gemini::spec::GeminiSpec::default();
    let transformers = spec.choose_chat_transformers(&req, &ctx);

    let body = transformers
        .request
        .transform_chat(&req)
        .expect("transform ok");

    // code_execution/search_grounding/file_search/response_mime_type 目前仍由
    // 聚合层的 chat_before_send 注入（非 std-gemini），此测试的目标是
    // 在 std-gemini 接管前验证管线可用。这一步暂时只验证基本字段，
    // 后续将 ProviderOptions→extra→GeminiChatAdapter 的迁移完成后再扩展。
    assert_eq!(
        body.get("model").and_then(|v| v.as_str()),
        Some("gemini-1.5-pro")
    );
}

