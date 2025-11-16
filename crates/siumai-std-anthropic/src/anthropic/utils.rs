//! Anthropic standard helper functions (provider-agnostic).
//!
//! Built on top of `siumai-core` ChatInput/ChatResult abstractions, this module provides:
//! - ChatInput → Anthropic Messages JSON mapping
//! - Anthropic usage/finish_reason parsing utilities

use siumai_core::error::LlmError;
use siumai_core::execution::chat::{ChatInput, ChatResult, ChatRole, ChatUsage};
use std::collections::HashMap;

/// Convert ChatInput into Anthropic Messages payload and optional system prompt.
pub fn build_messages_payload(
    input: &ChatInput,
) -> Result<(Vec<serde_json::Value>, Option<String>), LlmError> {
    let mut system: Option<String> = None;
    let mut messages = Vec::new();

    // Build a cache-control map based on message index:
    //
    // Shape of `ChatInput::extra["anthropic_prompt_caching"]`:
    // [
    //   { "index": 0, "cache_control": { "type": "ephemeral" } },
    //   ...
    // ]
    let mut cache_by_index: HashMap<usize, serde_json::Value> = HashMap::new();
    if let Some(cfg) = input.extra.get("anthropic_prompt_caching") {
        if let Some(items) = cfg.as_array() {
            for item in items {
                if let (Some(idx), Some(cc)) = (
                    item.get("index").and_then(|v| v.as_u64()),
                    item.get("cache_control"),
                ) {
                    cache_by_index.insert(idx as usize, cc.clone());
                }
            }
        }
    }

    for (idx, m) in input.messages.iter().enumerate() {
        match m.role {
            ChatRole::System => {
                // Concatenate multiple system messages with newline separators.
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
                let mut msg = serde_json::json!({
                    "role": "user",
                    "content": [{ "type": "text", "text": m.content }],
                });

                if let Some(cc) = cache_by_index.get(&idx) {
                    msg["cache_control"] = cc.clone();
                }
                messages.push(msg);
            }
            ChatRole::Assistant => {
                let mut msg = serde_json::json!({
                    "role": "assistant",
                    "content": [{ "type": "text", "text": m.content }],
                });

                if let Some(cc) = cache_by_index.get(&idx) {
                    msg["cache_control"] = cc.clone();
                }
                messages.push(msg);
            }
        }
    }

    Ok((messages, system))
}

/// Build ChatUsage from Anthropic `usage` object.
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

/// Extract a minimal ChatResult (only content + usage) from Anthropic response JSON.
pub fn parse_minimal_chat_result(resp: &serde_json::Value) -> ChatResult {
    // Anthropic returns an array of content blocks; here we only collect text fields and join by newline.
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
