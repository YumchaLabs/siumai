pub fn convert_tool_choice(choice: &crate::types::ToolChoice) -> Option<serde_json::Value> {
    use crate::types::ToolChoice;

    match choice {
        ToolChoice::Auto => Some(serde_json::json!({
            "type": "auto"
        })),
        ToolChoice::Required => Some(serde_json::json!({
            "type": "any"
        })),
        ToolChoice::None => None, // Anthropic doesn't support 'none', remove tools instead
        ToolChoice::Tool { name } => Some(serde_json::json!({
            "type": "tool",
            "name": name
        })),
    }
}

#[cfg(test)]
mod tool_choice_tests {
    use super::*;

    #[test]
    fn test_convert_tool_choice() {
        use crate::types::ToolChoice;

        // Test Auto
        let result = convert_tool_choice(&ToolChoice::Auto);
        assert_eq!(result, Some(serde_json::json!({"type": "auto"})));

        // Test Required (maps to "any" in Anthropic)
        let result = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(result, Some(serde_json::json!({"type": "any"})));

        // Test None (returns None, tools should be removed)
        let result = convert_tool_choice(&ToolChoice::None);
        assert_eq!(result, None);

        // Test Tool
        let result = convert_tool_choice(&ToolChoice::tool("weather"));
        assert_eq!(
            result,
            Some(serde_json::json!({
                "type": "tool",
                "name": "weather"
            }))
        );
    }
}
