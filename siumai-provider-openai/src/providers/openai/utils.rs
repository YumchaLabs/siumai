//! `OpenAI` Utility Functions
//!
//! Common utility functions for `OpenAI` API interactions.

use super::types::*;
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::types::*;
use reqwest::header::HeaderMap;

/// Build HTTP headers for `OpenAI` API requests
pub fn build_headers(
    api_key: &str,
    organization: Option<&str>,
    project: Option<&str>,
    custom_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    ProviderHeaders::openai(api_key, organization, project, custom_headers)
}

/// Convert tools to OpenAI Chat API format
pub fn convert_tools_to_openai_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    // Keep provider layer thin; delegate to canonical protocol mapping.
    crate::standards::openai::utils::convert_tools_to_openai_format(tools)
}

/// Convert tools to OpenAI Responses API format (flattened)
pub fn convert_tools_to_responses_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    // Keep provider layer thin; delegate to canonical protocol mapping.
    crate::standards::openai::utils::convert_tools_to_responses_format(tools)
}

/// Convert message content to `OpenAI` format
pub fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    crate::standards::openai::utils::convert_message_content_to_openai_value(content)
}

/// Convert messages to `OpenAI` format
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<OpenAiMessage>, LlmError> {
    crate::standards::openai::utils::convert_messages(messages)
}

/// Parse `OpenAI` finish reason
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    crate::standards::openai::utils::parse_finish_reason(reason)
}

/// Get default models for `OpenAI`
pub fn get_default_models() -> Vec<String> {
    use super::model_constants as openai;

    let models = vec![
        openai::gpt_5::GPT_5.to_string(),
        openai::gpt_4_1::GPT_4_1.to_string(),
        openai::gpt_4o::GPT_4O.to_string(),
        openai::gpt_4o::GPT_4O_MINI.to_string(),
        openai::gpt_4_turbo::GPT_4_TURBO.to_string(),
        openai::gpt_4::GPT_4.to_string(),
        openai::o1::O1.to_string(),
        openai::o1::O1_MINI.to_string(),
        openai::o3::O3_MINI.to_string(),
        openai::gpt_3_5::GPT_3_5_TURBO.to_string(),
    ];

    models
}

/// Check if content contains thinking tags (`<think>` or `</think>`)
/// This is used to detect DeepSeek-style thinking content
pub fn contains_thinking_tags(content: &str) -> bool {
    content.contains("<think>") || content.contains("</think>")
}

/// Extract thinking content from `<think>...</think>` tags
/// Returns the content inside the tags, or None if no valid tags found
/// Uses simple string parsing instead of regex for better performance
pub fn extract_thinking_content(content: &str) -> Option<String> {
    let start_tag = "<think>";
    let end_tag = "</think>";

    let start_pos = content.find(start_tag)?;
    let content_start = start_pos + start_tag.len();
    let end_pos = content[content_start..].find(end_tag)?;

    let thinking = content[content_start..content_start + end_pos].trim();
    if thinking.is_empty() {
        None
    } else {
        Some(thinking.to_string())
    }
}

/// Filter out thinking content from text for display purposes
/// Removes `<think>...</think>` tags and their content
/// Uses simple string parsing instead of regex for better performance
pub fn filter_thinking_content(content: &str) -> String {
    let start_tag = "<think>";
    let end_tag = "</think>";

    let mut result = String::new();
    let mut remaining = content;

    while let Some(start_pos) = remaining.find(start_tag) {
        // Add content before the tag
        result.push_str(&remaining[..start_pos]);

        // Find the end tag
        if let Some(end_pos) = remaining[start_pos..].find(end_tag) {
            // Skip the thinking content and continue after the end tag
            let skip_to = start_pos + end_pos + end_tag.len();
            remaining = &remaining[skip_to..];
        } else {
            // No matching end tag, remove everything from start tag onwards
            remaining = "";
            break;
        }
    }

    // Add any remaining content
    result.push_str(remaining);
    result.trim().to_string()
}

/// Extract content without thinking tags
/// If content contains thinking tags, filter them out; otherwise return as-is
pub fn extract_content_without_thinking(content: &str) -> String {
    if contains_thinking_tags(content) {
        filter_thinking_content(content)
    } else {
        content.to_string()
    }
}

/// Determine if a model should default to Responses API (auto mode)
/// Currently only gpt-5 family triggers auto routing
pub fn is_responses_model(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    m.starts_with("gpt-5")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_responses_model_only_gpt5() {
        assert!(is_responses_model("gpt-5"));
        assert!(is_responses_model("gpt-5-mini"));
        assert!(is_responses_model("GPT-5-VISION"));
        assert!(!is_responses_model("gpt-4o"));
        assert!(!is_responses_model("o1"));
        assert!(!is_responses_model(""));
    }

    #[test]
    fn test_extract_thinking_content() {
        // Test basic extraction
        let content = "Some text <think>This is thinking</think> more text";
        assert_eq!(
            extract_thinking_content(content),
            Some("This is thinking".to_string())
        );

        // Test with newlines
        let content = "Text <think>\nMultiline\nthinking\n</think> end";
        assert_eq!(
            extract_thinking_content(content),
            Some("Multiline\nthinking".to_string())
        );

        // Test with no thinking tags
        assert_eq!(extract_thinking_content("No tags here"), None);

        // Test with empty thinking
        assert_eq!(extract_thinking_content("<think></think>"), None);
        assert_eq!(extract_thinking_content("<think>   </think>"), None);

        // Test with only start tag
        assert_eq!(extract_thinking_content("<think>No end tag"), None);
    }

    #[test]
    fn test_filter_thinking_content() {
        // Test basic filtering
        let content = "Before <think>thinking</think> after";
        assert_eq!(filter_thinking_content(content), "Before  after");

        // Test multiple thinking blocks
        let content = "A <think>t1</think> B <think>t2</think> C";
        assert_eq!(filter_thinking_content(content), "A  B  C");

        // Test with no thinking tags
        let content = "No tags here";
        assert_eq!(filter_thinking_content(content), "No tags here");

        // Test with unclosed tag
        let content = "Before <think>unclosed";
        assert_eq!(filter_thinking_content(content), "Before");

        // Test with nested-like content
        let content = "A <think>B <think>C</think> D";
        // Should remove from first <think> to first </think>
        assert_eq!(filter_thinking_content(content), "A  D");
    }

    #[test]
    fn test_contains_thinking_tags() {
        assert!(contains_thinking_tags("<think>content</think>"));
        assert!(contains_thinking_tags("text <think>"));
        assert!(contains_thinking_tags("text </think>"));
        assert!(!contains_thinking_tags("no tags"));
        assert!(!contains_thinking_tags(""));
    }
}

/// Convert provider-agnostic ToolChoice to OpenAI format
///
/// # OpenAI Format
///
/// - `Auto` → `"auto"`
/// - `Required` → `"required"`
/// - `None` → `"none"`
/// - `Tool { name }` → `{"type": "function", "function": {"name": "..."}}`
///
/// # Example
///
/// ```rust
/// use siumai::types::ToolChoice;
/// use siumai::providers::openai::utils::convert_tool_choice;
///
/// let choice = ToolChoice::tool("weather");
/// let openai_format = convert_tool_choice(&choice);
/// ```
pub fn convert_tool_choice(choice: &crate::types::ToolChoice) -> serde_json::Value {
    crate::standards::openai::utils::convert_tool_choice(choice)
}

#[cfg(test)]
mod tool_choice_tests {
    use super::*;

    #[test]
    fn test_extract_content_without_thinking() {
        let content = "Before <think>thinking</think> after";
        assert_eq!(extract_content_without_thinking(content), "Before  after");

        let content = "No thinking tags";
        assert_eq!(
            extract_content_without_thinking(content),
            "No thinking tags"
        );
    }

    #[test]
    fn test_convert_tool_choice() {
        use crate::types::ToolChoice;

        // Test Auto
        let result = convert_tool_choice(&ToolChoice::Auto);
        assert_eq!(result, serde_json::json!("auto"));

        // Test Required
        let result = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(result, serde_json::json!("required"));

        // Test None
        let result = convert_tool_choice(&ToolChoice::None);
        assert_eq!(result, serde_json::json!("none"));

        // Test Tool
        let result = convert_tool_choice(&ToolChoice::tool("weather"));
        assert_eq!(
            result,
            serde_json::json!({
                "type": "function",
                "function": { "name": "weather" }
            })
        );
    }
}
