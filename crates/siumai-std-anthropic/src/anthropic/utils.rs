//! Anthropic 标准工具函数（provider 无关）
//!
//! 基于 `siumai-core` 的 ChatInput/ChatResult 抽象，提供：
//! - ChatInput → Anthropic Messages JSON 结构
//! - Anthropic usage/finish_reason 等字段解析

use siumai_core::error::LlmError;
use siumai_core::execution::chat::{ChatInput, ChatResult, ChatRole, ChatUsage};

/// 将 ChatInput 转换为 Anthropic Messages payload 与可选 system 提示
pub fn build_messages_payload(
    input: &ChatInput,
) -> Result<(Vec<serde_json::Value>, Option<String>), LlmError> {
    let mut system: Option<String> = None;
    let mut messages = Vec::new();

    for m in &input.messages {
        match m.role {
            ChatRole::System => {
                // 多条 system message 简单拼接
                match &mut system {
                    Some(acc) => {
                        if !acc.is_empty() {
                            acc.push_str("\n");
                        }
                        acc.push_str(&m.content);
                    }
                    None => system = Some(m.content.clone()),
                }
            }
            ChatRole::User => {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{ "type": "text", "text": m.content }],
                }));
            }
            ChatRole::Assistant => {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": [{ "type": "text", "text": m.content }],
                }));
            }
        }
    }

    Ok((messages, system))
}

/// 从 Anthropic usage 对象构建 ChatUsage
pub fn parse_usage(usage: Option<&serde_json::Value>) -> Option<ChatUsage> {
    let u = usage?;
    let prompt_tokens = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let completion_tokens = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let total_tokens = prompt_tokens.saturating_add(completion_tokens);
    Some(ChatUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

/// 从 Anthropic 响应 JSON 中提取最小 ChatResult（仅 content + usage）
pub fn parse_minimal_chat_result(resp: &serde_json::Value) -> ChatResult {
    // Anthropic 返回 content blocks 数组，这里仅提取 text 字段，按行拼接
    let mut text_acc = String::new();
    if let Some(arr) = resp.get("content").and_then(|c| c.as_array()) {
        for block in arr {
            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                if !text_acc.is_empty() {
                    text_acc.push_str("\n");
                }
                text_acc.push_str(t);
            }
        }
    }

    let usage = parse_usage(resp.get("usage"));

    ChatResult {
        content: text_acc,
        finish_reason: None,
        usage,
        metadata: Default::default(),
    }
}
