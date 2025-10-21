//! Anthropic Provider Options
//!
//! This module contains types for Anthropic-specific features including:
//! - Prompt caching configuration
//! - Thinking mode (extended thinking)

use serde::{Deserialize, Serialize};

/// Anthropic-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnthropicOptions {
    /// Prompt caching configuration
    pub prompt_caching: Option<PromptCachingConfig>,
    /// Thinking mode (extended thinking)
    pub thinking_mode: Option<ThinkingModeConfig>,
}

impl AnthropicOptions {
    /// Create new Anthropic options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable prompt caching
    pub fn with_prompt_caching(mut self, config: PromptCachingConfig) -> Self {
        self.prompt_caching = Some(config);
        self
    }

    /// Enable thinking mode
    pub fn with_thinking_mode(mut self, config: ThinkingModeConfig) -> Self {
        self.thinking_mode = Some(config);
        self
    }
}

/// Prompt caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCachingConfig {
    /// Whether prompt caching is enabled
    pub enabled: bool,
    /// Cache control markers
    pub cache_control: Vec<AnthropicCacheControl>,
}

impl Default for PromptCachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_control: vec![],
        }
    }
}

/// Anthropic cache control marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCacheControl {
    /// Cache type
    pub cache_type: AnthropicCacheType,
    /// Message index to apply cache control to
    pub message_index: usize,
}

/// Anthropic cache type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicCacheType {
    /// Ephemeral cache (5 minutes TTL)
    Ephemeral,
}

/// Thinking mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingModeConfig {
    /// Whether thinking mode is enabled
    pub enabled: bool,
    /// Thinking budget (tokens)
    pub thinking_budget: Option<u32>,
}

impl Default for ThinkingModeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thinking_budget: None,
        }
    }
}
