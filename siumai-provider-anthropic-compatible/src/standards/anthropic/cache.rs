//! Anthropic Prompt Caching Implementation
//!
//! This module implements Anthropic's prompt caching feature which allows
//! caching of frequently used prompts to reduce latency and costs.
//!
//! API Reference: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::{ChatMessage, ContentPart, MessageContent};
use base64::Engine;

/// Cache control configuration for Anthropic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache type - currently only "ephemeral" is supported
    pub r#type: CacheType,
    /// Optional cache TTL in seconds
    pub ttl: Option<u32>,
    /// Optional cache key for manual cache management
    pub cache_key: Option<String>,
}

impl Default for CacheControl {
    fn default() -> Self {
        Self {
            r#type: CacheType::Ephemeral,
            ttl: None,
            cache_key: None,
        }
    }
}

impl CacheControl {
    /// Create a new ephemeral cache control
    pub const fn ephemeral() -> Self {
        Self {
            r#type: CacheType::Ephemeral,
            ttl: None,
            cache_key: None,
        }
    }

    /// Create a cache control with custom TTL
    pub const fn with_ttl(mut self, ttl_seconds: u32) -> Self {
        self.ttl = Some(ttl_seconds);
        self
    }

    /// Create a cache control with custom cache key
    pub fn with_key<S: Into<String>>(mut self, key: S) -> Self {
        self.cache_key = Some(key.into());
        self
    }

    /// Convert to JSON for API requests
    pub fn to_json(&self) -> serde_json::Value {
        let mut json = serde_json::json!({
            "type": self.r#type
        });

        if let Some(ttl) = self.ttl {
            json["ttl"] = serde_json::Value::Number(ttl.into());
        }

        if let Some(ref key) = self.cache_key {
            json["cache_key"] = serde_json::Value::String(key.clone());
        }

        json
    }
}

/// Cache type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CacheType {
    /// Ephemeral cache - automatically managed by Anthropic
    Ephemeral,
}

/// Cache-aware message builder
pub struct CacheAwareMessageBuilder {
    /// Base message
    message: ChatMessage,
    /// Cache control for the message
    cache_control: Option<CacheControl>,
    /// Cache control for individual content parts
    content_cache_controls: HashMap<usize, CacheControl>,
}

impl CacheAwareMessageBuilder {
    /// Create a new cache-aware message builder
    pub fn new(message: ChatMessage) -> Self {
        Self {
            message,
            cache_control: None,
            content_cache_controls: HashMap::new(),
        }
    }

    /// Set cache control for the entire message
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Set cache control for a specific content part (for multimodal messages)
    pub fn with_content_cache_control(
        mut self,
        content_index: usize,
        cache_control: CacheControl,
    ) -> Self {
        self.content_cache_controls
            .insert(content_index, cache_control);
        self
    }

    /// Build the message with cache controls applied
    pub fn build(self) -> Result<serde_json::Value, LlmError> {
        let mut message_json = self.convert_message_to_json()?;

        // Align with Anthropic API + Vercel AI SDK:
        // cache_control is attached to content blocks, not the message object itself.
        //
        // Message-level cache control applies to the last content block.
        if let Some(cache_control) = self.cache_control {
            // Ensure content is an array of blocks.
            if let Some(content) = message_json.get_mut("content")
                && let serde_json::Value::String(s) = content
            {
                *content = serde_json::Value::Array(vec![serde_json::json!({
                    "type": "text",
                    "text": s,
                })]);
            }

            if let Some(content) = message_json.get_mut("content")
                && let Some(arr) = content.as_array_mut()
                && let Some(last) = arr.last_mut()
                && let Some(obj) = last.as_object_mut()
                && !obj.contains_key("cache_control")
            {
                obj.insert("cache_control".to_string(), cache_control.to_json());
            }
        }

        // Apply content-level cache controls
        if !self.content_cache_controls.is_empty()
            && let Some(content) = message_json.get_mut("content")
        {
            if let serde_json::Value::String(s) = content {
                *content = serde_json::Value::Array(vec![serde_json::json!({
                    "type": "text",
                    "text": s,
                })]);
            }
            if let serde_json::Value::Array(content_array) = content {
                for (index, cache_control) in self.content_cache_controls {
                    if let Some(content_item) = content_array.get_mut(index)
                        && let Some(content_obj) = content_item.as_object_mut()
                    {
                        content_obj.insert("cache_control".to_string(), cache_control.to_json());
                    }
                }
            }
        }

        Ok(message_json)
    }

    /// Convert `ChatMessage` to JSON format
    fn convert_message_to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut message_json = serde_json::json!({
            "role": match self.message.role {
                crate::types::MessageRole::System => "system",
                crate::types::MessageRole::User => "user",
                crate::types::MessageRole::Assistant => "assistant",
                crate::types::MessageRole::Developer => "user", // Developer messages are treated as user messages in Anthropic
                crate::types::MessageRole::Tool => "tool",
            }
        });

        // Handle content
        match &self.message.content {
            MessageContent::Text(text) => {
                message_json["content"] = serde_json::Value::String(text.clone());
            }
            MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": text
                            }));
                        }
                        ContentPart::Image { source, detail } => {
                            let (media_type, data) = match source {
                                crate::types::chat::MediaSource::Base64 { data } => {
                                    ("image/jpeg", data.clone())
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    ("image/jpeg", encoded)
                                }
                                crate::types::chat::MediaSource::Url { url } => {
                                    content_parts.push(serde_json::json!({
                                        "type": "text",
                                        "text": format!("[Image: {}]", url)
                                    }));
                                    continue;
                                }
                            };

                            let mut image_part = serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            });
                            if let Some(detail) = detail {
                                image_part["detail"] = serde_json::json!(detail);
                            }
                            content_parts.push(image_part);
                        }
                        ContentPart::Audio { source, .. } => {
                            // Anthropic does not support audio
                            let placeholder = match source {
                                crate::types::chat::MediaSource::Url { url } => {
                                    format!("[Audio: {}]", url)
                                }
                                _ => "[Audio not supported]".to_string(),
                            };
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": placeholder
                            }));
                        }
                        ContentPart::File {
                            source, media_type, ..
                        } => {
                            if media_type == "application/pdf" {
                                let data = match source {
                                    crate::types::chat::MediaSource::Base64 { data } => {
                                        data.clone()
                                    }
                                    crate::types::chat::MediaSource::Binary { data } => {
                                        base64::engine::general_purpose::STANDARD.encode(data)
                                    }
                                    crate::types::chat::MediaSource::Url { url } => {
                                        content_parts.push(serde_json::json!({
                                            "type": "text",
                                            "text": format!("[PDF: {}]", url)
                                        }));
                                        continue;
                                    }
                                };
                                content_parts.push(serde_json::json!({
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data
                                    }
                                }));
                            } else {
                                content_parts.push(serde_json::json!({
                                    "type": "text",
                                    "text": "[File not supported]"
                                }));
                            }
                        }
                        ContentPart::ToolCall { .. } => {}
                        ContentPart::ToolResult { .. } => {}
                        ContentPart::Reasoning { text } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("<thinking>{}</thinking>", text)
                            }));
                        }
                    }
                }
                message_json["content"] = serde_json::Value::Array(content_parts);
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => {
                message_json["content"] =
                    serde_json::Value::String(serde_json::to_string(v).unwrap_or_default());
            }
        }

        Ok(message_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_cache_control_applies_to_last_content_block() {
        let msg = ChatMessage::user("part1")
            .with_content_parts(vec![ContentPart::text("part2")])
            .build();

        let json = CacheAwareMessageBuilder::new(msg)
            .with_cache_control(CacheControl::ephemeral())
            .build()
            .unwrap();

        let content = json["content"].as_array().expect("content array");
        assert!(content[0].get("cache_control").is_none());
        assert_eq!(content[1]["cache_control"]["type"], "ephemeral");
    }
}
