//! ChatRequest → ChatInput 映射助手。
//!
//! 该模块负责将聚合层的 `ChatRequest` 映射为 `siumai-core` 中的
//! `ChatInput`，并负责将各 Provider 的 typed `ProviderOptions`
//! 写入 `ChatInput::extra`。

use crate::types::{ChatRequest, ProviderOptions};

/// Helper: map an aggregator-level `ChatRequest` into a minimal
/// `siumai-core` `ChatInput` (OpenAI-style).
///
/// OpenAI / OpenAI-compatible 及其它复用 OpenAI Chat 标准的 provider 共享。
pub fn openai_like_chat_request_to_core_input(
    req: &ChatRequest,
) -> siumai_core::execution::chat::ChatInput {
    use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};

    let messages = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                crate::types::MessageRole::System => ChatRole::System,
                crate::types::MessageRole::User => ChatRole::User,
                crate::types::MessageRole::Assistant => ChatRole::Assistant,
                _ => ChatRole::User,
            };
            let content = m.content.all_text();
            ChatMessageInput { role, content }
        })
        .collect::<Vec<_>>();

    ChatInput {
        messages,
        model: Some(req.common_params.model.clone()),
        max_tokens: req.common_params.max_tokens,
        temperature: req.common_params.temperature,
        top_p: req.common_params.top_p,
        presence_penalty: None,
        frequency_penalty: None,
        stop: req.common_params.stop_sequences.clone(),
        extra: Default::default(),
    }
}

/// Helper: OpenAI Chat 标准专用映射。
///
/// 在 `openai_like_chat_request_to_core_input` 基础上，将
/// `ProviderOptions::OpenAi` 中的 typed 配置写入 `ChatInput::extra`
/// 的 `openai_*` key。
pub fn openai_chat_request_to_core_input(
    req: &ChatRequest,
) -> siumai_core::execution::chat::ChatInput {
    let mut input = openai_like_chat_request_to_core_input(req);

    if let ProviderOptions::OpenAi(ref options) = req.provider_options {
        // Reasoning effort (o1/o3 models)
        if let Some(effort) = options.reasoning_effort {
            if let Ok(v) = serde_json::to_value(effort) {
                input
                    .extra
                    .insert("openai_reasoning_effort".to_string(), v);
            }
        }

        // Service tier preference
        if let Some(tier) = options.service_tier {
            if let Ok(v) = serde_json::to_value(tier) {
                input.extra.insert("openai_service_tier".to_string(), v);
            }
        }

        // Modalities (e.g., ["text","audio"])
        if let Some(ref mods) = options.modalities
            && let Ok(v) = serde_json::to_value(mods)
        {
            input.extra.insert("openai_modalities".to_string(), v);
        }

        // Audio configuration
        if let Some(ref aud) = options.audio
            && let Ok(v) = serde_json::to_value(aud)
        {
            input.extra.insert("openai_audio".to_string(), v);
        }

        // Prediction content
        if let Some(ref pred) = options.prediction
            && let Ok(v) = serde_json::to_value(pred)
        {
            input.extra.insert("openai_prediction".to_string(), v);
        }

        // Web search options
        if let Some(ref ws) = options.web_search_options
            && let Ok(v) = serde_json::to_value(ws)
        {
            input
                .extra
                .insert("openai_web_search_options".to_string(), v);
        }
    }

    input
}

/// Helper: Gemini 风格 ChatRequest → ChatInput 映射。
pub fn gemini_like_chat_request_to_core_input(
    req: &ChatRequest,
) -> siumai_core::execution::chat::ChatInput {
    use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};
    use std::collections::HashMap;

    let messages = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                crate::types::MessageRole::System => ChatRole::System,
                crate::types::MessageRole::User => ChatRole::User,
                crate::types::MessageRole::Assistant => ChatRole::Assistant,
                _ => ChatRole::User,
            };
            let content = m.content.all_text();
            ChatMessageInput { role, content }
        })
        .collect::<Vec<_>>();

    let mut extra: HashMap<String, serde_json::Value> = HashMap::new();
    if let ProviderOptions::Gemini(ref options) = req.provider_options {
        if let Some(ref code) = options.code_execution {
            if let Ok(v) = serde_json::to_value(code) {
                extra.insert("gemini_code_execution".to_string(), v);
            }
        }
        if let Some(ref search) = options.search_grounding {
            if let Ok(v) = serde_json::to_value(search) {
                extra.insert("gemini_search_grounding".to_string(), v);
            }
        }
        if let Some(ref fs) = options.file_search {
            if let Ok(v) = serde_json::to_value(fs) {
                extra.insert("gemini_file_search".to_string(), v);
            }
        }
        if let Some(ref mime) = options.response_mime_type {
            extra.insert(
                "gemini_response_mime_type".to_string(),
                serde_json::json!(mime),
            );
        }
    }

    ChatInput {
        messages,
        model: Some(req.common_params.model.clone()),
        max_tokens: req.common_params.max_tokens,
        temperature: req.common_params.temperature,
        top_p: req.common_params.top_p,
        presence_penalty: None,
        frequency_penalty: None,
        stop: req.common_params.stop_sequences.clone(),
        extra,
    }
}

/// Helper: Anthropic / MiniMaxi 共用的 ChatRequest → ChatInput 映射。
pub fn anthropic_like_chat_request_to_core_input(
    req: &ChatRequest,
) -> siumai_core::execution::chat::ChatInput {
    use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};
    use std::collections::HashMap;

    let messages = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                crate::types::MessageRole::System => ChatRole::System,
                crate::types::MessageRole::User => ChatRole::User,
                crate::types::MessageRole::Assistant => ChatRole::Assistant,
                _ => ChatRole::User,
            };
            let content = m.content.all_text();
            ChatMessageInput { role, content }
        })
        .collect::<Vec<_>>();

    // Map typed Anthropic options into core-level `extra` payload.
    //
    // NOTE: 最终协议 JSON 在这里构造，std 层仅做轻量 rename。
    let mut extra: HashMap<String, serde_json::Value> = HashMap::new();
    if let ProviderOptions::Anthropic(ref options) = req.provider_options {
        // Thinking mode configuration
        if let Some(ref thinking) = options.thinking_mode {
            if thinking.enabled {
                let mut thinking_config = serde_json::json!({ "type": "enabled" });
                if let Some(budget) = thinking.thinking_budget {
                    thinking_config["budget_tokens"] = serde_json::json!(budget);
                }
                extra.insert("anthropic_thinking".to_string(), thinking_config);
            }
        }

        // Structured output configuration
        if let Some(ref rf) = options.response_format {
            let value = match rf {
                crate::types::AnthropicResponseFormat::JsonObject => {
                    serde_json::json!({ "type": "json_object" })
                }
                crate::types::AnthropicResponseFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                } => serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": strict,
                        "schema": schema,
                    }
                }),
            };
            extra.insert("anthropic_response_format".to_string(), value);
        }

        // Prompt caching configuration (v1: message-level caching by message_index).
        //
        // Currently we only apply cache control to user/assistant messages, and
        // the cache type only supports "ephemeral". More fine-grained TTL/cache_key
        // settings may be added in future versions.
        if let Some(ref pc) = options.prompt_caching {
            if pc.enabled && !pc.cache_control.is_empty() {
                let entries: Vec<serde_json::Value> = pc
                    .cache_control
                    .iter()
                    .map(|ctrl| {
                        // `AnthropicCacheType` already implements Serialize (lowercase enum),
                        // so we construct the final protocol shape directly here.
                        let cache_type = serde_json::to_value(&ctrl.cache_type)
                            .unwrap_or_else(|_| serde_json::json!("ephemeral"));
                        serde_json::json!({
                            "index": ctrl.message_index,
                            "cache_control": {
                                "type": cache_type
                            }
                        })
                    })
                    .collect();

                if !entries.is_empty() {
                    extra.insert(
                        "anthropic_prompt_caching".to_string(),
                        serde_json::Value::Array(entries),
                    );
                }
            }
        }
    }

    ChatInput {
        messages,
        model: Some(req.common_params.model.clone()),
        max_tokens: req.common_params.max_tokens,
        temperature: req.common_params.temperature,
        top_p: req.common_params.top_p,
        presence_penalty: None,
        frequency_penalty: None,
        stop: req.common_params.stop_sequences.clone(),
        extra,
    }
}

