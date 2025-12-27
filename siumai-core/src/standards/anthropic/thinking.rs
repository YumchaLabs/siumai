//! Anthropic Extended Thinking Support
//!
//! This module provides support for Anthropic's extended thinking feature,
//! which allows Claude to show its step-by-step reasoning process.
//!
//! Based on the official Anthropic API documentation:
//! <https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{ChatMessage, ChatResponse, MessageContent};

/// Configuration for extended thinking according to official API documentation
/// <https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Type of thinking configuration (must be "enabled")
    pub r#type: String,
    /// Maximum tokens to use for thinking (required, minimum 1024)
    pub budget_tokens: u32,
}

/// A thinking block from Anthropic's response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingBlock {
    /// The thinking content (summarized in Claude 4 models)
    pub thinking: String,
    /// Encrypted signature for verification
    pub signature: Option<String>,
}

/// A redacted thinking block (encrypted for safety)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedThinkingBlock {
    /// Encrypted thinking data
    pub data: String,
}

impl ThinkingConfig {
    /// Create an enabled thinking configuration with budget tokens
    /// According to the API docs, `budget_tokens` must be >= 1024
    pub fn enabled(budget_tokens: u32) -> Self {
        assert!(budget_tokens >= 1024, "budget_tokens must be >= 1024");
        Self {
            r#type: "enabled".to_string(),
            budget_tokens,
        }
    }

    /// Set budget tokens for thinking
    pub fn with_budget_tokens(mut self, budget_tokens: u32) -> Self {
        assert!(budget_tokens >= 1024, "budget_tokens must be >= 1024");
        self.budget_tokens = budget_tokens;
        self
    }

    /// Check if thinking is enabled (always true for this config)
    pub fn is_enabled(&self) -> bool {
        self.r#type == "enabled"
    }

    /// Convert to request parameters for the API
    pub fn to_request_params(&self) -> serde_json::Value {
        let mut thinking_obj = serde_json::Map::new();
        thinking_obj.insert(
            "type".to_string(),
            serde_json::Value::String(self.r#type.clone()),
        );
        thinking_obj.insert(
            "budget_tokens".to_string(),
            serde_json::Value::Number(self.budget_tokens.into()),
        );
        serde_json::Value::Object(thinking_obj)
    }

    /// Validate the thinking configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.r#type != "enabled" {
            return Err("thinking type must be 'enabled'".to_string());
        }
        if self.budget_tokens < 1024 {
            return Err("budget_tokens must be >= 1024".to_string());
        }
        Ok(())
    }
}

/// Thinking response parser for Anthropic's extended thinking format
pub struct ThinkingResponseParser;

impl ThinkingResponseParser {
    /// Extract thinking content from Anthropic response
    /// According to the API docs, thinking blocks have type "thinking" and contain "thinking" field
    pub fn extract_thinking(response: &serde_json::Value) -> Option<ThinkingBlock> {
        // Check for thinking in the response content array
        if let Some(content) = response.get("content")
            && let Some(content_array) = content.as_array()
        {
            for item in content_array {
                if let Some(item_type) = item.get("type").and_then(|t| t.as_str())
                    && item_type == "thinking"
                {
                    let thinking_text = item
                        .get("thinking")
                        .and_then(|t| t.as_str())
                        .map(std::string::ToString::to_string);
                    let signature = item
                        .get("signature")
                        .and_then(|s| s.as_str())
                        .map(std::string::ToString::to_string);

                    if let Some(thinking) = thinking_text {
                        return Some(ThinkingBlock {
                            thinking,
                            signature,
                        });
                    }
                }
            }
        }
        None
    }

    /// Extract redacted thinking content from response
    pub fn extract_redacted_thinking(
        response: &serde_json::Value,
    ) -> Option<RedactedThinkingBlock> {
        if let Some(content) = response.get("content")
            && let Some(content_array) = content.as_array()
        {
            for item in content_array {
                if let Some(item_type) = item.get("type").and_then(|t| t.as_str())
                    && item_type == "redacted_thinking"
                {
                    let data = item
                        .get("data")
                        .and_then(|d| d.as_str())
                        .map(std::string::ToString::to_string);

                    if let Some(data) = data {
                        return Some(RedactedThinkingBlock { data });
                    }
                }
            }
        }
        None
    }

    /// Parse thinking from streaming response
    pub fn parse_thinking_delta(chunk: &serde_json::Value) -> Option<String> {
        // Check for thinking delta in streaming response
        if let Some(delta) = chunk.get("delta") {
            if let Some(thinking) = delta.get("thinking") {
                return thinking.as_str().map(std::string::ToString::to_string);
            }

            // Check for thinking in content delta
            if let Some(content) = delta.get("content")
                && let Some(content_array) = content.as_array()
            {
                for item in content_array {
                    if let Some(item_type) = item.get("type").and_then(|t| t.as_str())
                        && item_type == "thinking"
                        && let Some(text_delta) = item.get("text")
                    {
                        return text_delta.as_str().map(std::string::ToString::to_string);
                    }
                }
            }
        }

        None
    }

    /// Enhance `ChatResponse` with thinking content
    pub fn enhance_response_with_thinking(
        mut response: ChatResponse,
        thinking_content: Option<String>,
    ) -> ChatResponse {
        if let Some(thinking) = thinking_content {
            // Add thinking to provider_metadata under "anthropic" namespace
            let mut anthropic_meta = response
                .provider_metadata
                .as_ref()
                .and_then(|m| m.get("anthropic").cloned())
                .unwrap_or_default();

            anthropic_meta.insert("thinking".to_string(), serde_json::Value::String(thinking));

            let mut provider_metadata = response.provider_metadata.unwrap_or_default();
            provider_metadata.insert("anthropic".to_string(), anthropic_meta);
            response.provider_metadata = Some(provider_metadata);
        }
        response
    }
}

/// Reasoning analysis utilities
pub struct ReasoningAnalyzer;

impl ReasoningAnalyzer {
    /// Analyze thinking content for reasoning patterns
    pub fn analyze_reasoning(thinking_content: &str) -> ReasoningAnalysis {
        let mut analysis = ReasoningAnalysis::new();

        // Count reasoning steps - look for step indicators in the text
        let step_indicators = [
            "step by step",
            "first",
            "second",
            "third",
            "then",
            "next",
            "finally",
            "1.",
            "2.",
            "3.",
            "step",
            "initially",
            "subsequently",
        ];

        let content_lower = thinking_content.to_lowercase();
        analysis.reasoning_steps = step_indicators
            .iter()
            .map(|indicator| content_lower.matches(indicator).count() as u32)
            .sum::<u32>()
            .max(1); // At least 1 step if any reasoning content exists

        // Detect reasoning patterns
        if thinking_content.contains("Let me think")
            || thinking_content.contains("I need to consider")
        {
            analysis.patterns.push("deliberative".to_string());
        }

        if thinking_content.contains("pros and cons")
            || thinking_content.contains("advantages and disadvantages")
        {
            analysis.patterns.push("comparative".to_string());
        }

        if thinking_content.contains("because")
            || thinking_content.contains("therefore")
            || thinking_content.contains("since")
        {
            analysis.patterns.push("causal".to_string());
        }

        if thinking_content.contains("What if") || thinking_content.contains("Suppose") {
            analysis.patterns.push("hypothetical".to_string());
        }

        // Calculate complexity score
        analysis.complexity_score = Self::calculate_complexity_score(thinking_content);

        // Extract key concepts
        analysis.key_concepts = Self::extract_key_concepts(thinking_content);

        analysis
    }

    /// Calculate a simple complexity score based on length and vocabulary
    fn calculate_complexity_score(thinking_content: &str) -> f32 {
        let word_count = thinking_content.split_whitespace().count() as f32;
        let unique_words = thinking_content
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect::<std::collections::HashSet<_>>()
            .len() as f32;

        if word_count == 0.0 {
            return 0.0;
        }

        let vocabulary_richness = unique_words / word_count;
        let length_factor = (word_count / 100.0).min(5.0); // Cap at 5x

        vocabulary_richness * length_factor
    }

    /// Extract key concepts (simple noun phrase extraction)
    fn extract_key_concepts(thinking_content: &str) -> Vec<String> {
        // Simple extraction: look for capitalized words and common technical terms
        let mut concepts = Vec::new();

        let words: Vec<&str> = thinking_content.split_whitespace().collect();
        for word in words {
            // Remove punctuation
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());

            // Skip common words
            if clean_word.len() < 3 {
                continue;
            }

            // Check for capitalized words (potential proper nouns)
            if clean_word.chars().next().is_some_and(|c| c.is_uppercase()) {
                concepts.push(clean_word.to_string());
            }
        }

        // Deduplicate and limit
        concepts.sort();
        concepts.dedup();
        concepts.into_iter().take(10).collect()
    }
}

/// Analysis results for thinking content
#[derive(Debug, Clone)]
pub struct ReasoningAnalysis {
    /// Number of reasoning steps detected
    pub reasoning_steps: u32,
    /// Detected reasoning patterns
    pub patterns: Vec<String>,
    /// Complexity score (0.0-1.0+)
    pub complexity_score: f32,
    /// Key concepts extracted
    pub key_concepts: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ReasoningAnalysis {
    /// Create a new empty analysis
    pub fn new() -> Self {
        Self {
            reasoning_steps: 0,
            patterns: Vec::new(),
            complexity_score: 0.0,
            key_concepts: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the analysis
    pub fn add_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Default for ReasoningAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Thinking-aware message builder
pub struct ThinkingAwareMessageBuilder {
    /// Base message
    message: ChatMessage,
    /// Whether to include thinking tags
    include_thinking_tags: bool,
}

impl ThinkingAwareMessageBuilder {
    /// Create a new thinking-aware message builder
    pub fn new(message: ChatMessage) -> Self {
        Self {
            message,
            include_thinking_tags: false,
        }
    }

    /// Enable thinking tags in the message content
    pub const fn with_thinking_tags(mut self) -> Self {
        self.include_thinking_tags = true;
        self
    }

    /// Build the message with thinking enhancements
    pub fn build(self) -> ChatMessage {
        if !self.include_thinking_tags {
            return self.message;
        }

	        let mut message = self.message;

	        // Add thinking tags to assistant messages (for context)
	        if message.role == crate::types::MessageRole::Assistant
	            && let MessageContent::Text(text) = &message.content
	        {
	            message.content = MessageContent::Text(format!("<thinking>{}</thinking>", text));
	        }

	        message
	    }
}
