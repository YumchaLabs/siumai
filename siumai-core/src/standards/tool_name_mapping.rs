//! Tool name mapping utilities (Vercel AI SDK aligned).
//!
//! Vercel concept:
//! - Provider-defined tools have a stable provider `id` (e.g. `openai.web_search`)
//! - A call-scoped custom `name` can be used by the client (e.g. `mySearch`)
//! - Providers may operate on provider-native names (e.g. `web_search`)
//!
//! This utility builds a bidirectional mapping between:
//! - custom tool name (client-facing) <-> provider tool name (provider-facing)
#![deny(unsafe_code)]

use std::collections::HashMap;

use crate::types::Tool;

/// Bidirectional mapping between custom tool names and provider tool names.
#[derive(Debug, Clone, Default)]
pub struct ToolNameMapping {
    custom_to_provider: HashMap<String, String>,
    provider_to_custom: HashMap<String, String>,
}

impl ToolNameMapping {
    /// Maps a custom tool name (used by the client) to the provider's tool name.
    /// If the custom tool name does not have a mapping, returns the input name.
    pub fn to_provider_tool_name<'a>(&'a self, custom_tool_name: &'a str) -> &'a str {
        self.custom_to_provider
            .get(custom_tool_name)
            .map(|s| s.as_str())
            .unwrap_or(custom_tool_name)
    }

    /// Maps a provider tool name to the custom tool name used by the client.
    /// If the provider tool name does not have a mapping, returns the input name.
    pub fn to_custom_tool_name<'a>(&'a self, provider_tool_name: &'a str) -> &'a str {
        self.provider_to_custom
            .get(provider_tool_name)
            .map(|s| s.as_str())
            .unwrap_or(provider_tool_name)
    }
}

/// Create mappings for provider-defined tools.
///
/// `provider_tool_names` maps provider tool ids (e.g. `openai.web_search`) to provider-native
/// tool names (e.g. `web_search`).
pub fn create_tool_name_mapping(
    tools: &[Tool],
    provider_tool_names: &[(&str, &str)],
) -> ToolNameMapping {
    let provider_tool_names: HashMap<&str, &str> = provider_tool_names.iter().copied().collect();

    let mut custom_to_provider: HashMap<String, String> = HashMap::new();
    let mut provider_to_custom: HashMap<String, String> = HashMap::new();

    for tool in tools {
        let Tool::ProviderDefined(provider_tool) = tool else {
            continue;
        };

        let Some(provider_tool_name) = provider_tool_names.get(provider_tool.id.as_str()) else {
            continue;
        };

        custom_to_provider.insert(
            provider_tool.name.clone(),
            (*provider_tool_name).to_string(),
        );
        provider_to_custom.insert(
            (*provider_tool_name).to_string(),
            provider_tool.name.clone(),
        );
    }

    ToolNameMapping {
        custom_to_provider,
        provider_to_custom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderDefinedTool;

    const OPENAI_PROVIDER_TOOL_NAMES: &[(&str, &str)] = &[
        ("openai.web_search", "web_search"),
        ("openai.web_search_preview", "web_search_preview"),
        ("openai.file_search", "file_search"),
    ];

    #[test]
    fn creates_mapping_for_provider_defined_tools() {
        let tools = vec![
            Tool::ProviderDefined(ProviderDefinedTool::new("openai.web_search", "mySearch")),
            Tool::ProviderDefined(ProviderDefinedTool::new(
                "openai.web_search_preview",
                "myPreviewSearch",
            )),
        ];

        let mapping = create_tool_name_mapping(&tools, OPENAI_PROVIDER_TOOL_NAMES);
        assert_eq!(mapping.to_provider_tool_name("mySearch"), "web_search");
        assert_eq!(mapping.to_custom_tool_name("web_search"), "mySearch");
        assert_eq!(
            mapping.to_provider_tool_name("myPreviewSearch"),
            "web_search_preview"
        );
        assert_eq!(
            mapping.to_custom_tool_name("web_search_preview"),
            "myPreviewSearch"
        );
    }

    #[test]
    fn ignores_function_tools() {
        let tools = vec![Tool::function(
            "my-function-tool",
            "A function tool",
            serde_json::json!({ "type": "object" }),
        )];

        let mapping = create_tool_name_mapping(&tools, OPENAI_PROVIDER_TOOL_NAMES);
        assert_eq!(
            mapping.to_provider_tool_name("my-function-tool"),
            "my-function-tool"
        );
        assert_eq!(
            mapping.to_custom_tool_name("my-function-tool"),
            "my-function-tool"
        );
    }

    #[test]
    fn returns_input_when_no_mapping_exists() {
        let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
            "openai.web_search",
            "mySearch",
        ))];

        let mapping = create_tool_name_mapping(&tools, OPENAI_PROVIDER_TOOL_NAMES);
        assert_eq!(mapping.to_provider_tool_name("unknown"), "unknown");
        assert_eq!(mapping.to_custom_tool_name("unknown"), "unknown");
    }

    #[test]
    fn ignores_unrecognized_provider_tool_ids() {
        let tools = vec![
            Tool::ProviderDefined(ProviderDefinedTool::new(
                "openai.unknown_tool",
                "unknownTool",
            )),
            Tool::ProviderDefined(ProviderDefinedTool::new(
                "openai.file_search",
                "myFileSearch",
            )),
        ];

        let mapping = create_tool_name_mapping(&tools, OPENAI_PROVIDER_TOOL_NAMES);
        assert_eq!(mapping.to_provider_tool_name("unknownTool"), "unknownTool");
        assert_eq!(mapping.to_provider_tool_name("myFileSearch"), "file_search");
        assert_eq!(mapping.to_custom_tool_name("file_search"), "myFileSearch");
    }

    #[test]
    fn handles_empty_tools_array() {
        let tools: Vec<Tool> = Vec::new();
        let mapping = create_tool_name_mapping(&tools, OPENAI_PROVIDER_TOOL_NAMES);
        assert_eq!(mapping.to_provider_tool_name("any-tool"), "any-tool");
        assert_eq!(mapping.to_custom_tool_name("any-tool"), "any-tool");
    }

    #[test]
    fn handles_mixed_function_and_provider_defined_tools() {
        let tools = vec![
            Tool::function(
                "function-tool",
                "A function tool",
                serde_json::json!({ "type": "object" }),
            ),
            Tool::ProviderDefined(ProviderDefinedTool::new(
                "openai.web_search",
                "provider-tool",
            )),
        ];

        let mapping = create_tool_name_mapping(&tools, OPENAI_PROVIDER_TOOL_NAMES);
        assert_eq!(
            mapping.to_provider_tool_name("function-tool"),
            "function-tool"
        );
        assert_eq!(
            mapping.to_custom_tool_name("function-tool"),
            "function-tool"
        );
        assert_eq!(mapping.to_provider_tool_name("provider-tool"), "web_search");
        assert_eq!(mapping.to_custom_tool_name("web_search"), "provider-tool");
    }
}
