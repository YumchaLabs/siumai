//! Tests for OpenAI StreamDelta thinking field priority handling
//!
//! This module tests the custom deserialization logic that handles multiple
//! thinking field names with priority order: reasoning_content > thinking > reasoning

use serde_json::json;

// Import the private struct for testing
use crate::providers::openai::streaming::extract_thinking_from_multiple_fields;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_field_priority_reasoning_content_first() {
        // Test that reasoning_content has highest priority
        let json_data = json!({
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": "DeepSeek thinking",
            "thinking": "OpenAI thinking",
            "reasoning": "Alternative reasoning"
        });

        let thinking = extract_thinking_from_multiple_fields(&json_data);
        assert_eq!(thinking, Some("DeepSeek thinking".to_string()));
    }

    #[test]
    fn test_thinking_field_priority_thinking_second() {
        // Test that thinking has second priority when reasoning_content is not present
        let json_data = json!({
            "role": "assistant",
            "content": "Hello",
            "thinking": "OpenAI thinking",
            "reasoning": "Alternative reasoning"
        });

        let thinking = extract_thinking_from_multiple_fields(&json_data);
        assert_eq!(thinking, Some("OpenAI thinking".to_string()));
    }

    #[test]
    fn test_thinking_field_priority_reasoning_third() {
        // Test that reasoning has third priority when others are not present
        let json_data = json!({
            "role": "assistant",
            "content": "Hello",
            "reasoning": "Alternative reasoning"
        });

        let thinking = extract_thinking_from_multiple_fields(&json_data);
        assert_eq!(thinking, Some("Alternative reasoning".to_string()));
    }

    #[test]
    fn test_thinking_field_priority_empty_fields_ignored() {
        // Test that empty fields are ignored and next priority is used
        let json_data = json!({
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": "",
            "thinking": "   ",  // Whitespace only
            "reasoning": "Alternative reasoning"
        });

        let thinking = extract_thinking_from_multiple_fields(&json_data);
        assert_eq!(thinking, Some("Alternative reasoning".to_string()));
    }

    #[test]
    fn test_thinking_field_priority_no_thinking_fields() {
        // Test that None is returned when no thinking fields are present
        let json_data = json!({
            "role": "assistant",
            "content": "Hello"
        });

        let thinking = extract_thinking_from_multiple_fields(&json_data);
        assert_eq!(thinking, None);
    }

    #[test]
    fn test_thinking_field_priority_null_values_ignored() {
        // Test that null values are ignored
        let json_data = json!({
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": null,
            "thinking": "OpenAI thinking",
            "reasoning": "Alternative reasoning"
        });

        let thinking = extract_thinking_from_multiple_fields(&json_data);
        assert_eq!(thinking, Some("OpenAI thinking".to_string()));
    }

    #[test]
    fn test_thinking_field_priority_non_string_values_ignored() {
        // Test that non-string values are ignored
        let json_data = json!({
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": 123,
            "thinking": ["array", "value"],
            "reasoning": "Alternative reasoning"
        });

        let thinking = extract_thinking_from_multiple_fields(&json_data);
        assert_eq!(thinking, Some("Alternative reasoning".to_string()));
    }
}
