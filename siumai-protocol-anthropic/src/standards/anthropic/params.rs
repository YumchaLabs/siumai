//! Anthropic standard parameters for streaming conversion.

use crate::types::Tool;

/// Structured output mode inferred from the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputMode {
    /// Request-level `responseFormat: { type: "json", schema }` is implemented via the reserved
    /// `json` tool (unsupported models).
    JsonTool,
    /// Request-level `responseFormat: { type: "json", schema }` is implemented via
    /// `output_config.format` (supported models).
    OutputFormat,
}

/// Config for Anthropic streaming conversion.
#[derive(Debug, Clone, Default)]
pub struct AnthropicParams {
    pub structured_output_mode: Option<StructuredOutputMode>,
    pub mark_code_execution_dynamic: bool,
}

impl AnthropicParams {
    pub fn with_structured_output_mode(mut self, mode: StructuredOutputMode) -> Self {
        self.structured_output_mode = Some(mode);
        self
    }

    pub fn with_tools(mut self, tools: &[Tool]) -> Self {
        self.mark_code_execution_dynamic = has_web_tool_20260209_without_code_execution(tools);
        self
    }

    pub const fn should_mark_code_execution_dynamic(&self) -> bool {
        self.mark_code_execution_dynamic
    }
}

fn has_web_tool_20260209_without_code_execution(tools: &[Tool]) -> bool {
    let mut has_web_tool_20260209 = false;
    let mut has_code_execution_tool = false;

    for tool in tools {
        match tool {
            Tool::Function { function } => {
                if function.name == "code_execution" {
                    has_code_execution_tool = true;
                }
            }
            Tool::ProviderDefined(provider_tool)
                if provider_tool.provider() == Some("anthropic") =>
            {
                match provider_tool.tool_type() {
                    Some("web_fetch_20260209") | Some("web_search_20260209") => {
                        has_web_tool_20260209 = true;
                    }
                    Some("code_execution_20250522")
                    | Some("code_execution_20250825")
                    | Some("code_execution_20260120") => {
                        has_code_execution_tool = true;
                    }
                    _ => {}
                }
            }
            Tool::ProviderDefined(_) => {}
        }
    }

    has_web_tool_20260209 && !has_code_execution_tool
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_code_execution_dynamic_for_2026_web_tools_without_code_execution_tool() {
        let params = AnthropicParams::default().with_tools(&[
            crate::tools::anthropic::web_search_20260209(),
            crate::tools::anthropic::web_fetch_20260209(),
        ]);

        assert!(params.should_mark_code_execution_dynamic());
    }

    #[test]
    fn does_not_mark_code_execution_dynamic_when_code_execution_tool_exists() {
        let params = AnthropicParams::default().with_tools(&[
            crate::tools::anthropic::web_fetch_20260209(),
            crate::tools::anthropic::code_execution_20260120(),
        ]);

        assert!(!params.should_mark_code_execution_dynamic());
    }

    #[test]
    fn function_tool_named_code_execution_counts_as_explicit_code_execution() {
        let params = AnthropicParams::default().with_tools(&[
            crate::tools::anthropic::web_search_20260209(),
            Tool::function(
                "code_execution",
                "custom",
                serde_json::json!({ "type": "object" }),
            ),
        ]);

        assert!(!params.should_mark_code_execution_dynamic());
    }
}
