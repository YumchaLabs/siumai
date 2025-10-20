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
            response
                .metadata
                .insert("thinking".to_string(), serde_json::Value::String(thinking));
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

    /// Calculate complexity score based on thinking content
    fn calculate_complexity_score(thinking_content: &str) -> f64 {
        let word_count = thinking_content.split_whitespace().count() as f64;
        let sentence_count = thinking_content.split('.').count() as f64;
        let unique_words = thinking_content
            .split_whitespace()
            .map(str::to_lowercase)
            .collect::<std::collections::HashSet<_>>()
            .len() as f64;

        // Simple complexity heuristic
        let avg_sentence_length = if sentence_count > 0.0 {
            word_count / sentence_count
        } else {
            0.0
        };
        let vocabulary_diversity = if word_count > 0.0 {
            unique_words / word_count
        } else {
            0.0
        };

        (avg_sentence_length * 0.3 + vocabulary_diversity * 0.7).min(10.0)
    }

    /// Extract key concepts from thinking content
    fn extract_key_concepts(thinking_content: &str) -> Vec<String> {
        // Simple keyword extraction - in a real implementation,
        // you might use NLP libraries for better concept extraction
        let keywords = [
            "problem",
            "solution",
            "approach",
            "method",
            "strategy",
            "analysis",
            "conclusion",
            "evidence",
            "reasoning",
            "logic",
            "assumption",
            "hypothesis",
            "theory",
            "principle",
            "concept",
        ];

        let mut concepts = Vec::new();
        let content_lower = thinking_content.to_lowercase();

        for keyword in &keywords {
            if content_lower.contains(keyword) {
                concepts.push(keyword.to_string());
            }
        }

        concepts.sort();
        concepts.dedup();
        concepts
    }
}

/// Reasoning analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningAnalysis {
    /// Number of reasoning steps identified
    pub reasoning_steps: u32,
    /// Reasoning patterns detected
    pub patterns: Vec<String>,
    /// Complexity score (0-10)
    pub complexity_score: f64,
    /// Key concepts extracted
    pub key_concepts: Vec<String>,
    /// Analysis metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ReasoningAnalysis {
    /// Create a new reasoning analysis
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
    pub fn with_metadata<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), json_value);
        }
        self
    }

    /// Get a summary of the reasoning analysis
    pub fn summary(&self) -> String {
        format!(
            "Reasoning Analysis: {} steps, {} patterns ({}), complexity: {:.1}/10",
            self.reasoning_steps,
            self.patterns.len(),
            self.patterns.join(", "),
            self.complexity_score
        )
    }
}

impl Default for ReasoningAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for working with thinking responses
pub mod helpers {
    use super::*;

    /// Create a thinking-enabled message
    pub fn create_thinking_message<S: Into<String>>(
        content: S,
        thinking_config: ThinkingConfig,
    ) -> (ChatMessage, serde_json::Value) {
        let message = ChatMessage {
            role: crate::types::MessageRole::User,
            content: MessageContent::Text(content.into()),
            metadata: crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };

        let params = thinking_config.to_request_params();
        (message, params)
    }

    /// Extract and analyze thinking from response
    pub fn extract_and_analyze_thinking(
        response: &ChatResponse,
    ) -> Option<(String, ReasoningAnalysis)> {
        if let Some(thinking_content) = response.thinking.as_ref() {
            let analysis = ReasoningAnalyzer::analyze_reasoning(thinking_content);
            return Some((thinking_content.to_string(), analysis));
        }
        None
    }

    /// Format thinking content for display
    pub fn format_thinking_for_display(thinking_content: &str, include_analysis: bool) -> String {
        let mut formatted = format!("🤔 **Claude's Thinking Process:**\n\n{thinking_content}");

        if include_analysis {
            let analysis = ReasoningAnalyzer::analyze_reasoning(thinking_content);
            formatted.push_str(&format!("\n\n📊 **Analysis:** {}", analysis.summary()));
        }

        formatted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_config() {
        let config = ThinkingConfig::enabled(1024);

        assert!(config.is_enabled());
        assert_eq!(config.budget_tokens, 1024);

        let params = config.to_request_params();
        assert_eq!(
            params.get("type"),
            Some(&serde_json::Value::String("enabled".to_string()))
        );
        assert_eq!(
            params.get("budget_tokens"),
            Some(&serde_json::Value::Number(1024.into()))
        );
    }

    #[test]
    #[should_panic(expected = "budget_tokens must be >= 1024")]
    fn test_thinking_config_invalid_budget() {
        ThinkingConfig::enabled(500); // Should panic
    }

    #[test]
    fn test_reasoning_analysis() {
        let thinking_content = "Let me think about this step by step. First, I need to consider the pros and cons. Because of this evidence, therefore I conclude...";

        let analysis = ReasoningAnalyzer::analyze_reasoning(thinking_content);

        assert!(analysis.reasoning_steps > 0);
        assert!(analysis.patterns.contains(&"deliberative".to_string()));
        assert!(analysis.patterns.contains(&"comparative".to_string()));
        assert!(analysis.patterns.contains(&"causal".to_string()));
        assert!(analysis.complexity_score > 0.0);
    }

    #[test]
    fn test_thinking_extraction() {
        let response_json = serde_json::json!({
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I need to analyze this carefully...",
                    "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
                },
                {
                    "type": "text",
                    "text": "Based on my analysis..."
                }
            ]
        });

        let thinking = ThinkingResponseParser::extract_thinking(&response_json);
        assert!(thinking.is_some());

        if let Some(thinking_block) = thinking {
            assert_eq!(
                thinking_block.thinking,
                "I need to analyze this carefully..."
            );
            assert!(thinking_block.signature.is_some());
        }
    }
}
