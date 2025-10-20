//! Tests for Gemini Thinking Functionality
//!
//! This module tests the Google Gemini provider's thinking capabilities,
//! including proper handling of the 'thought' boolean flag according to
//! Google API documentation.

use siumai::providers::gemini::types::{Content, Part, ThinkingConfig};
use siumai::stream::ChatStreamEvent;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that Part correctly handles thinking content
    #[test]
    fn test_part_thinking_content_handling() {
        // Test regular text part
        let text_part = Part::text("Regular text".to_string());
        if let Part::Text { text, thought } = text_part {
            assert_eq!(text, "Regular text");
            assert_eq!(thought, None); // Should default to None (false)
        } else {
            panic!("Expected text part");
        }

        // Test thought summary part
        let thought_part = Part::thought_summary("Thinking content".to_string());
        if let Part::Text { text, thought } = thought_part {
            assert_eq!(text, "Thinking content");
            assert_eq!(thought, Some(true)); // Should be explicitly true
        } else {
            panic!("Expected thought part");
        }
    }

    /// Test stream event creation for thinking vs regular content
    #[test]
    fn test_stream_event_thinking_separation() {
        // Simulate how the streaming logic should work
        let thinking_part = Part::Text {
            text: "Let me think about this...".to_string(),
            thought: Some(true),
        };

        let regular_part = Part::Text {
            text: "Here is my answer.".to_string(),
            thought: Some(false),
        };

        let default_part = Part::Text {
            text: "Default content.".to_string(),
            thought: None, // Should be treated as false
        };

        // Test thinking content creates ThinkingDelta event
        if let Part::Text { text, thought } = thinking_part {
            let should_be_thinking = thought.unwrap_or(false);
            assert!(should_be_thinking);

            let event = if should_be_thinking {
                ChatStreamEvent::ThinkingDelta { delta: text }
            } else {
                ChatStreamEvent::ContentDelta {
                    delta: text,
                    index: None,
                }
            };

            if let ChatStreamEvent::ThinkingDelta { delta } = event {
                assert_eq!(delta, "Let me think about this...");
            } else {
                panic!("Expected ThinkingDelta event");
            }
        }

        // Test regular content creates ContentDelta event
        if let Part::Text { text, thought } = regular_part {
            let should_be_thinking = thought.unwrap_or(false);
            assert!(!should_be_thinking);

            let event = if should_be_thinking {
                ChatStreamEvent::ThinkingDelta { delta: text }
            } else {
                ChatStreamEvent::ContentDelta {
                    delta: text,
                    index: None,
                }
            };

            if let ChatStreamEvent::ContentDelta { delta, .. } = event {
                assert_eq!(delta, "Here is my answer.");
            } else {
                panic!("Expected ContentDelta event");
            }
        }

        // Test default content (None) creates ContentDelta event
        if let Part::Text { text, thought } = default_part {
            let should_be_thinking = thought.unwrap_or(false);
            assert!(!should_be_thinking);

            let event = if should_be_thinking {
                ChatStreamEvent::ThinkingDelta { delta: text }
            } else {
                ChatStreamEvent::ContentDelta {
                    delta: text,
                    index: None,
                }
            };

            if let ChatStreamEvent::ContentDelta { delta, .. } = event {
                assert_eq!(delta, "Default content.");
            } else {
                panic!("Expected ContentDelta event");
            }
        }
    }

    /// Test Content creation helpers
    #[test]
    fn test_content_creation_helpers() {
        // Test user content creation
        let user_content = Content::user_text("Hello, how are you?".to_string());
        assert_eq!(user_content.role, Some("user".to_string()));
        assert_eq!(user_content.parts.len(), 1);

        if let Part::Text { text, thought } = &user_content.parts[0] {
            assert_eq!(text, "Hello, how are you?");
            assert_eq!(*thought, None); // Should default to None
        } else {
            panic!("Expected text part");
        }

        // Test model content creation
        let model_content = Content::model_text("I'm doing well, thank you!".to_string());
        assert_eq!(model_content.role, Some("model".to_string()));
        assert_eq!(model_content.parts.len(), 1);

        // Test system content creation
        let system_content = Content::system_text("You are a helpful assistant.".to_string());
        assert_eq!(system_content.role, None); // System instructions don't have a role
        assert_eq!(system_content.parts.len(), 1);
    }

    /// Test ThinkingConfig validation
    #[test]
    fn test_thinking_config_validation() {
        // Test valid configurations
        let valid_config = ThinkingConfig::with_budget(1024);
        assert!(valid_config.validate().is_ok());

        let dynamic_config = ThinkingConfig::dynamic();
        assert!(dynamic_config.validate().is_ok());

        let disabled_config = ThinkingConfig::disabled();
        assert!(disabled_config.validate().is_ok());

        // Test invalid configuration
        let invalid_config = ThinkingConfig {
            thinking_budget: Some(-2), // Invalid: less than -1
            include_thoughts: None,
        };
        assert!(invalid_config.validate().is_err());
    }

    /// Test Part creation helpers
    #[test]
    fn test_part_creation_helpers() {
        // Test regular text part
        let text_part = Part::text("Regular text".to_string());
        if let Part::Text { text, thought } = text_part {
            assert_eq!(text, "Regular text");
            assert_eq!(thought, None);
        } else {
            panic!("Expected text part");
        }

        // Test thought summary part
        let thought_part = Part::thought_summary("Thinking content".to_string());
        if let Part::Text { text, thought } = thought_part {
            assert_eq!(text, "Thinking content");
            assert_eq!(thought, Some(true));
        } else {
            panic!("Expected thought part");
        }
    }
}
