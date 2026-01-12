//! Provider-defined tool types.

use super::Tool;

/// Provider-defined tool configuration
///
/// This represents a tool that is defined and executed by the provider,
/// such as web search, file search, code execution, computer use, etc.
///
/// # Format
///
/// The tool ID follows the format `"provider.tool_name"`, for example:
/// - `"openai.web_search"` - OpenAI Responses API web search
/// - `"openai.web_search_preview"` - OpenAI Chat API web search (preview)
/// - `"anthropic.web_search_20250305"` - Anthropic web search (version 20250305)
/// - `"anthropic.computer_20250124"` - Anthropic computer use
/// - `"google.code_execution"` - Google Gemini code execution
///
/// # Examples
///
/// ```rust
/// use siumai::types::ProviderDefinedTool;
///
/// // Create a basic provider-defined tool
/// let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
///
/// // With configuration arguments
/// let tool = ProviderDefinedTool::new("openai.web_search", "web_search")
///     .with_args(serde_json::json!({
///         "searchContextSize": "high"
///     }));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderDefinedTool {
    /// Tool ID in format "provider.tool_name"
    ///
    /// Examples: "openai.web_search", "anthropic.web_search_20250305"
    pub id: String,

    /// Tool name used in the tools map
    ///
    /// Examples: "web_search", "file_search", "code_execution"
    pub name: String,

    /// Provider-specific configuration arguments.
    ///
    /// This is aligned with Vercel AI SDK's `{ type: "provider", id, name, args }` shape.
    pub args: serde_json::Value,
}

impl serde::Serialize for ProviderDefinedTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("ProviderDefinedTool", 3)?;
        st.serialize_field("id", &self.id)?;
        st.serialize_field("name", &self.name)?;
        st.serialize_field("args", &self.args)?;
        st.end()
    }
}

impl<'de> serde::Deserialize<'de> for ProviderDefinedTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::Deserialize;
        use std::collections::HashMap;

        #[derive(Deserialize)]
        struct De {
            id: String,
            name: String,
            #[serde(default)]
            args: Option<serde_json::Value>,
            #[serde(flatten)]
            extra: HashMap<String, serde_json::Value>,
        }

        let mut de = De::deserialize(deserializer)?;

        // Prefer explicit `args` (Vercel shape). If missing, fall back to the legacy flatten
        // format where config keys live at the top-level alongside `id` and `name`.
        let mut args = de
            .args
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

        if !de.extra.is_empty()
            && let serde_json::Value::Object(ref mut map) = args
        {
            for (k, v) in de.extra.drain() {
                map.entry(k).or_insert(v);
            }
        }

        Ok(Self {
            id: de.id,
            name: de.name,
            args,
        })
    }
}

impl ProviderDefinedTool {
    /// Create a new provider-defined tool
    ///
    /// # Arguments
    ///
    /// * `id` - Tool ID in format "provider.tool_name"
    /// * `name` - Tool name used in the tools map
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ProviderDefinedTool;
    ///
    /// let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    /// ```
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            args: serde_json::Value::Object(Default::default()),
        }
    }

    /// Set configuration arguments
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ProviderDefinedTool;
    ///
    /// let tool = ProviderDefinedTool::new("openai.web_search", "web_search")
    ///     .with_args(serde_json::json!({
    ///         "searchContextSize": "high",
    ///         "userLocation": {
    ///             "type": "approximate",
    ///             "city": "San Francisco"
    ///         }
    ///     }));
    /// ```
    pub fn with_args(mut self, args: serde_json::Value) -> Self {
        self.args = args;
        self
    }

    /// Get the provider name from the ID
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ProviderDefinedTool;
    ///
    /// let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    /// assert_eq!(tool.provider(), Some("openai"));
    /// ```
    pub fn provider(&self) -> Option<&str> {
        self.id.split('.').next()
    }

    /// Get the tool type from the ID
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ProviderDefinedTool;
    ///
    /// let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    /// assert_eq!(tool.tool_type(), Some("web_search"));
    /// ```
    pub fn tool_type(&self) -> Option<&str> {
        self.id.split('.').nth(1)
    }

    /// Create a provider-defined tool from a known tool id using a Vercel-aligned default name.
    ///
    /// Prefer `siumai::tools::{openai, anthropic, google, xai}::*` when you can; this helper is
    /// useful for config-driven/dynamic tool selection.
    pub fn from_id(id: &str) -> Option<Self> {
        match crate::tools::provider_defined_tool(id)? {
            Tool::ProviderDefined(pd) => Some(pd),
            _ => None,
        }
    }
}
