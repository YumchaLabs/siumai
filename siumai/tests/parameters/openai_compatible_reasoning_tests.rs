//! OpenAI-compatible reasoning parameter mapping tests.
//!
//! 这些测试验证 unified reasoning 接口在 OpenAI-Compatible provider 上的最终 JSON 行为：
//! - SiliconFlow: `with_thinking(true)` → `enable_thinking: true`
//! - DeepSeek: `reasoning(true)` → `enable_reasoning: true`
//! - Doubao: `reasoning(true)` → `thinking: { type: "enabled" }`（不再保留 `enable_thinking`）

#[path = "../support/mod.rs"]
mod support;

use mockito::Matcher;

use siumai::{ChatCapability, ChatMessage, builder::LlmBuilder};

/// SiliconFlow: unified thinking → enable_thinking
#[tokio::test]
async fn siliconflow_with_thinking_sets_enable_thinking_flag() {
    let mut server = support::mockito::start().await;

    // 匹配 SiliconFlow OpenAI-compatible chat 路由以及 enable_thinking 字段
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .match_header(
            "authorization",
            Matcher::Regex(r"^Bearer sk-test-silicon$".to_string()),
        )
        .match_header("content-type", "application/json")
        .match_body(Matcher::Regex(
            r#""enable_thinking"\s*:\s*true"#.to_string(),
        ))
        .with_status(200)
        .with_body(
            r#"{
                "id": "cmpl_1",
                "object": "chat.completion",
                "created": 1710000000,
                "model": "deepseek-ai/DeepSeek-V3",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "ok" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            }"#,
        )
        .create_async()
        .await;

    let base_url = format!("{}/v1", support::mockito::url(&server));

    let client = LlmBuilder::new()
        .siliconflow()
        .api_key("sk-test-silicon")
        .base_url(base_url)
        .model("deepseek-ai/DeepSeek-V3")
        .with_thinking(true)
        .build()
        .await
        .expect("build siliconflow client");

    let messages = vec![ChatMessage::user("hi").build()];
    let resp = client
        .chat_with_tools(messages, None)
        .await
        .expect("chat ok");
    assert!(matches!(
        resp.content,
        siumai::types::MessageContent::Text(_)
    ));
}

/// Generic OpenAI-compatible provider: text_verbosity → verbosity
#[tokio::test]
async fn openrouter_text_verbosity_sets_verbosity_field() {
    let mut server = support::mockito::start().await;

    let _m = server
        .mock("POST", "/v1/chat/completions")
        .match_header(
            "authorization",
            Matcher::Regex(r"^Bearer sk-test-openrouter$".to_string()),
        )
        .match_header("content-type", "application/json")
        .match_body(Matcher::Regex(r#""verbosity"\s*:\s*"low""#.to_string()))
        .with_status(200)
        .with_body(
            r#"{
                "id": "cmpl_1",
                "object": "chat.completion",
                "created": 1710000000,
                "model": "openai/gpt-5.1-mini",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "ok" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            }"#,
        )
        .create_async()
        .await;

    let base_url = format!("{}/v1", support::mockito::url(&server));

    let client = LlmBuilder::new()
        .openrouter()
        .api_key("sk-test-openrouter")
        .base_url(base_url)
        .model("openai/gpt-5.1-mini")
        .text_verbosity("low")
        .build()
        .await
        .expect("build openrouter client");

    let messages = vec![ChatMessage::user("hi").build()];
    let resp = client
        .chat_with_tools(messages, None)
        .await
        .expect("chat ok");
    assert!(matches!(
        resp.content,
        siumai::types::MessageContent::Text(_)
    ));
}

/// DeepSeek: unified reasoning → enable_reasoning
#[tokio::test]
async fn deepseek_reasoning_sets_enable_reasoning_flag() {
    let mut server = support::mockito::start().await;

    let _m = server
        .mock("POST", "/v1/chat/completions")
        .match_header(
            "authorization",
            Matcher::Regex(r"^Bearer sk-test-deepseek$".to_string()),
        )
        .match_header("content-type", "application/json")
        .match_body(Matcher::Regex(
            r#""enable_reasoning"\s*:\s*true"#.to_string(),
        ))
        .with_status(200)
        .with_body(
            r#"{
                "id": "cmpl_1",
                "object": "chat.completion",
                "created": 1710000000,
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "ok", "reasoning_content": "..." },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            }"#,
        )
        .create_async()
        .await;

    let base_url = format!("{}/v1", support::mockito::url(&server));

    let client = LlmBuilder::new()
        .deepseek()
        .api_key("sk-test-deepseek")
        .base_url(base_url)
        .model("deepseek-chat")
        .reasoning(true)
        .build()
        .await
        .expect("build deepseek client");

    let messages = vec![ChatMessage::user("hi").build()];
    let resp = client
        .chat_with_tools(messages, None)
        .await
        .expect("chat ok");
    assert!(matches!(
        resp.content,
        siumai::types::MessageContent::Text(_)
    ));
}

/// Doubao: unified reasoning → thinking object, not enable_thinking
#[tokio::test]
async fn doubao_reasoning_sets_thinking_object_without_enable_thinking() {
    let mut server = support::mockito::start().await;

    // Doubao OpenAI-compatible endpoint (ark.cn-beijing.volces.com /api/v3/chat/completions)
    let _m = server
        .mock("POST", "/api/v3/chat/completions")
        .match_header(
            "authorization",
            Matcher::Regex(r"^Bearer sk-test-doubao$".to_string()),
        )
        .match_header("content-type", "application/json")
        // ensure thinking is enabled（由 adapter 从 enable_thinking 映射而来）
        .match_body(Matcher::Regex(
            r#""thinking"\s*:\s*\{\s*"type"\s*:\s*"enabled"\s*\}"#.to_string(),
        ))
        .with_status(200)
        .with_body(
            r#"{
                "id": "cmpl_1",
                "object": "chat.completion",
                "created": 1710000000,
                "model": "doubao-pro-4k",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "ok" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            }"#,
        )
        .create_async()
        .await;

    let base_url = format!("{}/api/v3", support::mockito::url(&server));

    let client = LlmBuilder::new()
        .doubao()
        .api_key("sk-test-doubao")
        .base_url(base_url)
        .model("doubao-pro-4k")
        .reasoning(true)
        .build()
        .await
        .expect("build doubao client");

    let messages = vec![ChatMessage::user("hi").build()];
    let resp = client
        .chat_with_tools(messages, None)
        .await
        .expect("chat ok");
    assert!(matches!(
        resp.content,
        siumai::types::MessageContent::Text(_)
    ));
}
