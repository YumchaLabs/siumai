//! Alibaba/Qwen prompt cache-control request shaping.
//!
//! Alibaba's OpenAI-compatible chat surface accepts `cache_control` markers on text and image
//! content parts. AI SDK exposes those markers through message/part `providerOptions.alibaba`.

use crate::types::{
    ChatMessage, ChatRequest, ContentPart, MessageContent, MessageRole, ProviderOptionsMap, Warning,
};

const MAX_CACHE_BREAKPOINTS: usize = 4;

/// AI SDK warning emitted when Alibaba prompt-cache breakpoint count exceeds the provider limit.
pub const CACHE_BREAKPOINT_LIMIT_WARNING: &str =
    "Max breakpoint limit exceeded. Only the last 4 cache markers will take effect.";

#[derive(Debug, Clone, Default)]
struct CacheControlCollector {
    breakpoint_count: usize,
}

impl CacheControlCollector {
    fn cache_control(
        &mut self,
        provider_id: &str,
        provider_options: &ProviderOptionsMap,
    ) -> Option<serde_json::Value> {
        let mut out = None;

        for key in crate::standards::openai::compat::metadata::provider_options_keys(provider_id) {
            let Some(options) = provider_options.get_object(&key) else {
                continue;
            };

            if let Some(value) = options
                .get("cacheControl")
                .or_else(|| options.get("cache_control"))
            {
                out = Some(value.clone());
            }
        }

        if out.is_some() {
            self.breakpoint_count += 1;
        }

        out
    }

    fn warnings(&self) -> Vec<Warning> {
        let extra = self.breakpoint_count.saturating_sub(MAX_CACHE_BREAKPOINTS);
        std::iter::repeat_with(|| Warning::other(CACHE_BREAKPOINT_LIMIT_WARNING))
            .take(extra)
            .collect()
    }
}

#[derive(Debug, Clone)]
enum WireCacheControl {
    TextLike(Option<serde_json::Value>),
    UserParts(Vec<Option<serde_json::Value>>),
}

impl WireCacheControl {
    fn has_cache_control(&self) -> bool {
        match self {
            Self::TextLike(cache_control) => cache_control.is_some(),
            Self::UserParts(parts) => parts.iter().any(Option::is_some),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CacheControlPlan {
    messages: Vec<WireCacheControl>,
    collector: CacheControlCollector,
}

impl CacheControlPlan {
    fn has_cache_control(&self) -> bool {
        self.messages
            .iter()
            .any(WireCacheControl::has_cache_control)
    }

    fn warnings(&self) -> Vec<Warning> {
        self.collector.warnings()
    }
}

/// Whether the compat provider id should receive Alibaba/Qwen cache-control handling.
pub fn supports_alibaba_cache_control(provider_id: &str) -> bool {
    matches!(
        provider_id
            .split('.')
            .next()
            .unwrap_or(provider_id)
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "alibaba" | "qwen"
    )
}

/// Compute AI SDK-style prompt-cache warnings for Alibaba/Qwen chat requests.
pub fn cache_control_warnings(provider_id: &str, req: &ChatRequest) -> Vec<Warning> {
    if !supports_alibaba_cache_control(provider_id) {
        return Vec::new();
    }

    build_cache_control_plan(provider_id, &req.messages).warnings()
}

pub(crate) fn apply_cache_controls_to_chat_body(
    provider_id: &str,
    req: &ChatRequest,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    if !supports_alibaba_cache_control(provider_id) {
        return;
    }

    let plan = build_cache_control_plan(provider_id, &req.messages);
    if !plan.has_cache_control() {
        return;
    }

    let Some(messages) = body_obj
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };

    for (message, cache_control) in messages.iter_mut().zip(plan.messages.iter()) {
        apply_message_cache_control(message, cache_control);
    }
}

fn build_cache_control_plan(provider_id: &str, messages: &[ChatMessage]) -> CacheControlPlan {
    let mut plan = CacheControlPlan::default();

    for message in messages {
        let message_cache_control = plan
            .collector
            .cache_control(provider_id, &message.provider_options);

        match message.role {
            MessageRole::System | MessageRole::Developer | MessageRole::Assistant => {
                plan.messages
                    .push(WireCacheControl::TextLike(message_cache_control));
            }
            MessageRole::User => {
                let part_cache_controls = user_part_cache_controls(
                    provider_id,
                    &message.content,
                    message_cache_control,
                    &mut plan.collector,
                );
                plan.messages
                    .push(WireCacheControl::UserParts(part_cache_controls));
            }
            MessageRole::Tool => {
                let tool_cache_controls = tool_result_cache_controls(
                    provider_id,
                    &message.content,
                    message_cache_control,
                    &mut plan.collector,
                );
                plan.messages.extend(
                    tool_cache_controls
                        .into_iter()
                        .map(WireCacheControl::TextLike),
                );
            }
        }
    }

    plan
}

fn user_part_cache_controls(
    provider_id: &str,
    content: &MessageContent,
    message_cache_control: Option<serde_json::Value>,
    collector: &mut CacheControlCollector,
) -> Vec<Option<serde_json::Value>> {
    match content {
        MessageContent::Text(_) => vec![message_cache_control],
        MessageContent::MultiModal(parts) => parts
            .iter()
            .enumerate()
            .filter_map(|(index, part)| {
                let is_last_part = index == parts.len().saturating_sub(1);
                let part_options = cacheable_user_part_provider_options(part)?;
                Some(
                    collector
                        .cache_control(provider_id, part_options)
                        .or_else(|| {
                            is_last_part
                                .then(|| message_cache_control.clone())
                                .flatten()
                        }),
                )
            })
            .collect(),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => vec![message_cache_control],
    }
}

fn cacheable_user_part_provider_options(part: &ContentPart) -> Option<&ProviderOptionsMap> {
    match part {
        ContentPart::Text {
            provider_options, ..
        }
        | ContentPart::Image {
            provider_options, ..
        }
        | ContentPart::File {
            provider_options, ..
        } => Some(provider_options),
        _ => None,
    }
}

fn tool_result_cache_controls(
    provider_id: &str,
    content: &MessageContent,
    message_cache_control: Option<serde_json::Value>,
    collector: &mut CacheControlCollector,
) -> Vec<Option<serde_json::Value>> {
    match content {
        MessageContent::Text(_) => vec![message_cache_control],
        MessageContent::MultiModal(parts) => {
            let tool_results = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolResult {
                        provider_options, ..
                    } => Some(provider_options),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let last = tool_results.len().saturating_sub(1);

            tool_results
                .into_iter()
                .enumerate()
                .map(|(index, provider_options)| {
                    collector
                        .cache_control(provider_id, provider_options)
                        .or_else(|| {
                            (index == last)
                                .then(|| message_cache_control.clone())
                                .flatten()
                        })
                })
                .collect()
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => vec![message_cache_control],
    }
}

fn apply_message_cache_control(message: &mut serde_json::Value, cache_control: &WireCacheControl) {
    let Some(message_obj) = message.as_object_mut() else {
        return;
    };

    match cache_control {
        WireCacheControl::TextLike(Some(cache_control)) => {
            apply_text_like_cache_control(message_obj, cache_control.clone());
        }
        WireCacheControl::TextLike(None) => {}
        WireCacheControl::UserParts(part_cache_controls) => {
            apply_user_part_cache_controls(message_obj, part_cache_controls);
        }
    }
}

fn apply_text_like_cache_control(
    message_obj: &mut serde_json::Map<String, serde_json::Value>,
    cache_control: serde_json::Value,
) {
    let text = message_obj
        .get("content")
        .map(text_from_content_value)
        .unwrap_or_default();

    message_obj.insert(
        "content".to_string(),
        serde_json::Value::Array(vec![text_part_with_cache_control(text, cache_control)]),
    );
}

fn apply_user_part_cache_controls(
    message_obj: &mut serde_json::Map<String, serde_json::Value>,
    part_cache_controls: &[Option<serde_json::Value>],
) {
    if !part_cache_controls.iter().any(Option::is_some) {
        return;
    }

    let Some(content) = message_obj.get_mut("content") else {
        return;
    };

    match content {
        serde_json::Value::String(text) => {
            if let Some(Some(cache_control)) = part_cache_controls.first() {
                *content = serde_json::Value::Array(vec![text_part_with_cache_control(
                    text.clone(),
                    cache_control.clone(),
                )]);
            }
        }
        serde_json::Value::Array(parts) => {
            for (part, cache_control) in parts.iter_mut().zip(part_cache_controls.iter()) {
                let Some(cache_control) = cache_control else {
                    continue;
                };
                let Some(part_obj) = part.as_object_mut() else {
                    continue;
                };
                part_obj.insert("cache_control".to_string(), cache_control.clone());
            }
        }
        _ => {}
    }
}

fn text_part_with_cache_control(
    text: String,
    cache_control: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "type": "text",
        "text": text,
        "cache_control": cache_control,
    })
}

fn text_from_content_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Null => String::new(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
            .collect::<String>(),
        other => other.to_string(),
    }
}
