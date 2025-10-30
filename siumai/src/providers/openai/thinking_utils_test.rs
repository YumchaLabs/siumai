//! Tests for thinking content processing utilities
//!
//! This module tests the handling of `<think>` tags in OpenAI-compatible responses.

#[cfg(test)]
mod tests {
    use super::super::utils::*;

    #[test]
    fn test_contains_thinking_tags() {
        // Test with opening tag
        assert!(contains_thinking_tags("Hello <think>"));

        // Test with closing tag
        assert!(contains_thinking_tags("</think> Hello"));

        // Test with both tags
        assert!(contains_thinking_tags("<think>thinking</think>"));

        // Test without tags
        assert!(!contains_thinking_tags("Hello world"));

        // Test empty string
        assert!(!contains_thinking_tags(""));
    }

    #[test]
    fn test_extract_thinking_content() {
        // Test simple extraction
        let input = "<think>This is thinking</think>Hello world";
        let result = extract_thinking_content(input);
        assert_eq!(result, Some("This is thinking".to_string()));

        // Test multiline thinking
        let input = "<think>\nThis is\nmultiline thinking\n</think>Hello world";
        let result = extract_thinking_content(input);
        assert_eq!(result, Some("This is\nmultiline thinking".to_string()));

        // Test no thinking tags
        let input = "Hello world";
        let result = extract_thinking_content(input);
        assert_eq!(result, None);

        // Test empty thinking
        let input = "<think></think>Hello world";
        let result = extract_thinking_content(input);
        assert_eq!(result, None);

        // Test only whitespace in thinking
        let input = "<think>   </think>Hello world";
        let result = extract_thinking_content(input);
        assert_eq!(result, None);
    }

    #[test]
    fn test_filter_thinking_content() {
        // Test simple filtering
        let input = "<think>This is thinking</think>Hello world";
        let result = filter_thinking_content(input);
        assert_eq!(result, "Hello world");

        // Test multiple thinking blocks
        let input = "<think>First</think>Hello<think>Second</think>World";
        let result = filter_thinking_content(input);
        assert_eq!(result, "HelloWorld");

        // Test multiline thinking
        let input = "<think>\nThis is\nmultiline thinking\n</think>Hello world";
        let result = filter_thinking_content(input);
        assert_eq!(result, "Hello world");

        // Test no thinking tags
        let input = "Hello world";
        let result = filter_thinking_content(input);
        assert_eq!(result, "Hello world");

        // Test only thinking tags
        let input = "<think>Only thinking</think>";
        let result = filter_thinking_content(input);
        assert_eq!(result, "");
    }

    #[test]
    fn test_extract_content_without_thinking() {
        // Test with thinking tags
        let input = "<think>This is thinking</think>Hello world";
        let result = extract_content_without_thinking(input);
        assert_eq!(result, "Hello world");

        // Test without thinking tags
        let input = "Hello world";
        let result = extract_content_without_thinking(input);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_deepseek_style_thinking() {
        // Test DeepSeek-style thinking content
        let input = r#"<think>
The user asked a programming question.
I need to:
1. Analyze the core of the problem
2. Provide a clear explanation
3. Give practical examples
</think>

This is a great programming question. Let me explain..."#;

        let thinking = extract_thinking_content(input);
        assert!(thinking.is_some());
        assert!(
            thinking
                .unwrap()
                .contains("The user asked a programming question")
        );

        let filtered = filter_thinking_content(input);
        assert!(!filtered.contains("<think>"));
        assert!(filtered.contains("This is a great programming question"));
    }

    #[test]
    fn test_edge_cases() {
        // Test malformed tags
        let input = "<think>Incomplete thinking";
        let result = extract_thinking_content(input);
        assert_eq!(result, None);

        // Test nested tags (should not happen but test anyway)
        let input = "<think>Outer <think>Inner</think> thinking</think>";
        let result = extract_thinking_content(input);
        assert!(result.is_some());

        // Test empty input
        let result = extract_thinking_content("");
        assert_eq!(result, None);

        let result = filter_thinking_content("");
        assert_eq!(result, "");
    }
}
