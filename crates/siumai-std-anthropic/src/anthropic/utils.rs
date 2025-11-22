//! Anthropic standard helper functions (provider-agnostic).
//!
//! Built on top of `siumai-core` ChatInput/ChatResult abstractions, this module provides:
//! - ChatInput → Anthropic Messages JSON mapping
//! - Anthropic usage/finish_reason parsing utilities

use siumai_core::error::LlmError;
use siumai_core::execution::chat::{
    ChatInput, ChatParsedContentCore, ChatParsedToolCallCore, ChatResult, ChatRole, ChatUsage,
};
use siumai_core::types::FinishReasonCore;
use std::collections::HashMap;

/// Core representation of a single Anthropic tool call.
#[derive(Debug, Clone)]
pub struct AnthropicToolCallCore {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Core representation of Anthropic message content.
///
/// This mirrors the unified content model used by higher-level layers:
/// - `text`: aggregated assistant text content
/// - `tool_calls`: tool_use blocks with ids and JSON arguments
/// - `thinking`: optional thinking/reasoning content
#[derive(Debug, Clone, Default)]
pub struct AnthropicParsedContentCore {
    pub text: String,
    pub tool_calls: Vec<AnthropicToolCallCore>,
    pub thinking: Option<String>,
}

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
    if let Some(cfg) = input.extra.get("anthropic_prompt_caching")
        && let Some(items) = cfg.as_array()
    {
        for item in items {
            if let (Some(idx), Some(cc)) = (
                item.get("index").and_then(|v| v.as_u64()),
                item.get("cache_control"),
            ) {
                cache_by_index.insert(idx as usize, cc.clone());
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
                            acc.push('\n');
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

/// Extract a core-level parsed content structure from an Anthropic
/// response JSON. This is used by both non-streaming response
/// handling and can be used by higher layers to construct richer
/// content models (text + tool calls + thinking).
pub fn parse_content_blocks_core(resp: &serde_json::Value) -> AnthropicParsedContentCore {
    let mut parsed = AnthropicParsedContentCore::default();

    if let Some(arr) = resp.get("content").and_then(|c| c.as_array()) {
        for block in arr {
            let kind = block
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            match kind {
                "text" => {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str())
                        && !t.is_empty()
                    {
                        if !parsed.text.is_empty() {
                            parsed.text.push('\n');
                        }
                        parsed.text.push_str(t);
                    }
                }
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let arguments = block
                        .get("input")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}));

                    if !id.is_empty() && !name.is_empty() {
                        parsed.tool_calls.push(AnthropicToolCallCore {
                            id,
                            name,
                            arguments,
                        });
                    }
                }
                "thinking" => {
                    // Thinking content may be placed under "thinking"; if absent,
                    // fall back to "text" for robustness.
                    if let Some(th) = block
                        .get("thinking")
                        .or_else(|| block.get("text"))
                        .and_then(|v| v.as_str())
                        && !th.is_empty()
                    {
                        match &mut parsed.thinking {
                            Some(acc) => {
                                if !acc.is_empty() {
                                    acc.push('\n');
                                }
                                acc.push_str(th);
                            }
                            None => parsed.thinking = Some(th.to_string()),
                        }
                    }
                }
                _ => {}
            }
        }
    }

    parsed
}

/// Extract a minimal ChatResult (only content + usage/finish_reason) from
/// Anthropic response JSON.
pub fn parse_minimal_chat_result(resp: &serde_json::Value) -> ChatResult {
    let parsed = parse_content_blocks_core(resp);

    // Usage parsing reuses the existing helper.
    let mut text_acc = String::new();
    text_acc.push_str(&parsed.text);

    let usage = parse_usage(resp.get("usage"));
    let finish_reason = resp
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .and_then(|s| parse_finish_reason_core(Some(s)));

    // Populate the core parsed content model so higher layers can
    // reconstruct richer MessageContent (text + tool calls + thinking)
    // without re-parsing provider JSON.
    let parsed_content = ChatParsedContentCore {
        text: parsed.text.clone(),
        tool_calls: parsed
            .tool_calls
            .iter()
            .map(|t| ChatParsedToolCallCore {
                id: Some(t.id.clone()),
                name: t.name.clone(),
                arguments: t.arguments.clone(),
            })
            .collect(),
        thinking: parsed.thinking.clone(),
    };

    ChatResult {
        content: text_acc,
        finish_reason,
        usage,
        metadata: Default::default(),
        parsed_content: Some(parsed_content),
    }
}

/// Parse Anthropic stop_reason into the core `FinishReasonCore` enum.
///
/// This helper is shared by both non-streaming response handling and
/// streaming event converters to ensure consistent semantics across
/// different call paths.
pub fn parse_finish_reason_core(reason: Option<&str>) -> Option<FinishReasonCore> {
    match reason {
        Some("end_turn") => Some(FinishReasonCore::Stop),
        Some("max_tokens") => Some(FinishReasonCore::Length),
        Some("tool_use") => Some(FinishReasonCore::ToolCalls),
        Some("refusal") => Some(FinishReasonCore::ContentFilter),
        // For stop_sequence / pause_turn and any other values, keep the
        // original string as an opaque tag.
        Some(other) => Some(FinishReasonCore::Other(other.to_string())),
        None => None,
    }
}
