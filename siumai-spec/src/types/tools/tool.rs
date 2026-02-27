//! Tool enum and constructors.

use serde::{Deserialize, Serialize};

use super::{ProviderDefinedTool, ToolFunction};

/// Tool definition for function calling
///
/// This enum represents different types of tools that can be used with LLMs:
/// - User-defined functions
/// - Provider-defined tools (web search, file search, etc.)
///
/// # Examples
///
/// ## User-defined function
///
/// ```rust
/// use siumai::types::Tool;
///
/// let tool = Tool::function(
///     "get_weather".to_string(),
///     "Get weather information".to_string(),
///     serde_json::json!({
///         "type": "object",
///         "properties": {
///             "location": { "type": "string" }
///         }
///     })
/// );
/// ```
///
/// ## Provider-defined tool
///
/// ```rust
/// use siumai::types::Tool;
///
/// // Preferred: use Vercel-aligned factories.
/// let tool = siumai::tools::openai::web_search();
///
/// // Or: config-driven selection by tool id (uses Vercel-aligned default name).
/// let tool = Tool::provider_defined_id("openai.web_search").unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Tool {
    /// User-defined function tool
    Function {
        #[serde(flatten)]
        function: ToolFunction,
    },

    /// Provider-defined tool (web search, file search, code execution, etc.)
    #[serde(rename = "provider-defined", alias = "provider")]
    ProviderDefined(ProviderDefinedTool),
}

impl Tool {
    /// Create a new function tool
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::Tool;
    ///
    /// let tool = Tool::function(
    ///     "get_weather",
    ///     "Get weather information",
    ///     serde_json::json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "location": { "type": "string" }
    ///         }
    ///     })
    /// );
    /// ```
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self::Function {
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
                input_examples: None,
                strict: None,
                provider_options_map: crate::types::ProviderOptionsMap::default(),
            },
        }
    }

    /// Create a provider-defined tool
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::Tool;
    ///
    /// let tool = Tool::provider_defined("openai.web_search", "web_search")
    ///     .with_args(serde_json::json!({
    ///         "searchContextSize": "high"
    ///     }));
    /// ```
    pub fn provider_defined(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ProviderDefined(ProviderDefinedTool::new(id, name))
    }

    /// Create a provider-defined tool from a known tool id using a Vercel-aligned default name.
    ///
    /// This is equivalent to calling the corresponding factory in `siumai::tools::<provider>`,
    /// but works when you only have the tool id as a string.
    ///
    /// Note: some tools require mandatory provider args and therefore cannot be constructed from
    /// an id alone (e.g. `google.file_search`, `google.vertex_rag_store`).
    pub fn provider_defined_id(id: &str) -> Option<Self> {
        crate::tools::provider_defined_tool(id)
    }

    /// Add arguments to a provider-defined tool
    ///
    /// This is a convenience method that only works on ProviderDefined variants.
    /// For Function variants, this method does nothing.
    pub fn with_args(self, args: serde_json::Value) -> Self {
        match self {
            Self::ProviderDefined(mut tool) => {
                tool.args = args;
                Self::ProviderDefined(tool)
            }
            other => other,
        }
    }
}
