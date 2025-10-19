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

        // Apply message-level cache control
        if let Some(cache_control) = self.cache_control {
            message_json["cache_control"] = cache_control.to_json();
        }

        // Apply content-level cache controls
        if !self.content_cache_controls.is_empty()
            && let Some(content) = message_json.get_mut("content")
        {
            match content {
                serde_json::Value::Array(content_array) => {
                    for (index, cache_control) in self.content_cache_controls {
                        if let Some(content_item) = content_array.get_mut(index)
                            && let Some(content_obj) = content_item.as_object_mut()
                        {
                            content_obj
                                .insert("cache_control".to_string(), cache_control.to_json());
                        }
                    }
                }
                serde_json::Value::String(_) => {
                    // For simple text content, we can't apply part-level cache control
                    // This would need to be converted to array format first
                }
                _ => {}
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
                        ContentPart::Image { image_url, detail } => {
                            let mut image_part = serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg", // Default, should be detected
                                    "data": image_url
                                }
                            });
                            if let Some(detail) = detail {
                                image_part["detail"] = serde_json::Value::String(detail.clone());
                            }
                            content_parts.push(image_part);
                        }
                        ContentPart::Audio { audio_url, format } => {
                            content_parts.push(serde_json::json!({
                                "type": "audio",
                                "source": {
                                    "type": "base64",
                                    "media_type": format,
                                    "data": audio_url
                                }
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

/// Cache statistics from Anthropic API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Number of cache hits
    pub cache_hits: u32,
    /// Number of cache misses
    pub cache_misses: u32,
    /// Total cached tokens
    pub cached_tokens: u32,
    /// Cache creation tokens (for new cache entries)
    pub cache_creation_tokens: u32,
    /// Cache read tokens (for cache hits)
    pub cache_read_tokens: u32,
}

impl CacheStatistics {
    /// Create empty cache statistics
    pub const fn empty() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            cached_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
        }
    }

    /// Parse cache statistics from API response
    pub fn from_response(response: &serde_json::Value) -> Option<Self> {
        let usage = response.get("usage")?;

        Some(Self {
            cache_hits: usage.get("cache_hits")?.as_u64()? as u32,
            cache_misses: usage.get("cache_misses")?.as_u64()? as u32,
            cached_tokens: usage.get("cached_tokens")?.as_u64()? as u32,
            cache_creation_tokens: usage.get("cache_creation_tokens")?.as_u64()? as u32,
            cache_read_tokens: usage.get("cache_read_tokens")?.as_u64()? as u32,
        })
    }

    /// Calculate cache efficiency ratio
    pub fn cache_efficiency(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_requests as f64
        }
    }

    /// Calculate token savings from caching
    pub const fn token_savings(&self) -> u32 {
        // Tokens that would have been processed without caching
        self.cache_read_tokens
    }
}

/// Helper functions for common caching patterns
pub mod patterns {
    use super::*;

    /// Create a system message with caching for long prompts
    pub fn cached_system_message<S: Into<String>>(content: S) -> CacheAwareMessageBuilder {
        let message = ChatMessage {
            role: crate::types::MessageRole::System,
            content: MessageContent::Text(content.into()),
            metadata: crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };

        CacheAwareMessageBuilder::new(message).with_cache_control(CacheControl::ephemeral())
    }

    /// Create a user message with document caching
    pub fn cached_document_message<S: Into<String>>(
        document: S,
        query: S,
    ) -> CacheAwareMessageBuilder {
        let content = format!("Document:\n{}\n\nQuery: {}", document.into(), query.into());
        let message = ChatMessage {
            role: crate::types::MessageRole::User,
            content: MessageContent::Text(content),
            metadata: crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };

        CacheAwareMessageBuilder::new(message).with_cache_control(CacheControl::ephemeral())
    }

    /// Create a conversation with cached context
    pub fn cached_conversation_context(
        context_messages: Vec<ChatMessage>,
        new_message: ChatMessage,
    ) -> Vec<serde_json::Value> {
        let mut result = Vec::new();

        // Add context messages with caching
        for (i, message) in context_messages.into_iter().enumerate() {
            let builder = CacheAwareMessageBuilder::new(message);
            let builder = if i == 0 {
                // Cache the first context message (usually system prompt)
                builder.with_cache_control(CacheControl::ephemeral())
            } else {
                builder
            };

            if let Ok(json) = builder.build() {
                result.push(json);
            }
        }

        // Add new message without caching
        if let Ok(json) = CacheAwareMessageBuilder::new(new_message).build() {
            result.push(json);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_control_creation() {
        let cache_control = CacheControl::ephemeral()
            .with_ttl(3600)
            .with_key("test-key");

        assert_eq!(cache_control.r#type, CacheType::Ephemeral);
        assert_eq!(cache_control.ttl, Some(3600));
        assert_eq!(cache_control.cache_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_cache_control_json() {
        let cache_control = CacheControl::ephemeral().with_ttl(1800);
        let json = cache_control.to_json();

        assert_eq!(json["type"], "ephemeral");
        assert_eq!(json["ttl"], 1800);
    }

    #[test]
    fn test_cache_statistics() {
        let stats = CacheStatistics {
            cache_hits: 8,
            cache_misses: 2,
            cached_tokens: 1000,
            cache_creation_tokens: 100,
            cache_read_tokens: 800,
        };

        assert_eq!(stats.cache_efficiency(), 0.8);
        assert_eq!(stats.token_savings(), 800);
    }

    #[test]
    fn test_cached_system_message() {
        let builder = patterns::cached_system_message("You are a helpful assistant.");
        let json = builder.build().unwrap();

        assert_eq!(json["role"], "system");
        assert_eq!(json["content"], "You are a helpful assistant.");
        assert!(json["cache_control"].is_object());
    }
}
