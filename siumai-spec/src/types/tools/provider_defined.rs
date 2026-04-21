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

    /// Optional display title.
    pub title: Option<String>,

    /// Provider-specific metadata carried on the stable AI SDK-aligned tool surface.
    ///
    /// This mirrors the common `providerOptions` field on AI SDK `Tool`, even for
    /// `type: "provider"` tools. Provider runtimes may choose to ignore or lower
    /// these options selectively.
    pub provider_options_map: crate::types::ProviderOptionsMap,

    /// Provider-specific configuration arguments.
    ///
    /// This is aligned with Vercel AI SDK's `{ type: "provider", id, name, args }` shape.
    pub args: serde_json::Value,

    /// Whether this provider-defined tool can return deferred results on later turns.
    ///
    /// This mirrors AI SDK's `supportsDeferredResults` provider-tool metadata and is kept on the
    /// stable portable tool shape so higher-level orchestration can reason about deferred tool
    /// outputs without baking the detail into provider request shapers.
    pub supports_deferred_results: Option<bool>,
}

impl serde::Serialize for ProviderDefinedTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let field_count = 3
            + usize::from(self.title.is_some())
            + usize::from(!self.provider_options_map.is_empty())
            + usize::from(self.supports_deferred_results.is_some());
        let mut st = serializer.serialize_struct("ProviderDefinedTool", field_count)?;
        st.serialize_field("id", &self.id)?;
        st.serialize_field("name", &self.name)?;
        if let Some(title) = &self.title {
            st.serialize_field("title", title)?;
        }
        if !self.provider_options_map.is_empty() {
            st.serialize_field("providerOptions", &self.provider_options_map)?;
        }
        st.serialize_field("args", &self.args)?;
        if let Some(supports_deferred_results) = self.supports_deferred_results {
            st.serialize_field("supportsDeferredResults", &supports_deferred_results)?;
        }
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
            title: Option<String>,
            #[serde(default, rename = "providerOptions", alias = "provider_options")]
            provider_options_map: crate::types::ProviderOptionsMap,
            #[serde(default)]
            args: Option<serde_json::Value>,
            #[serde(
                default,
                rename = "supportsDeferredResults",
                alias = "supports_deferred_results"
            )]
            supports_deferred_results: Option<bool>,
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
            title: de.title,
            provider_options_map: de.provider_options_map,
            args,
            supports_deferred_results: de.supports_deferred_results,
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
            title: None,
            provider_options_map: crate::types::ProviderOptionsMap::default(),
            args: serde_json::Value::Object(Default::default()),
            supports_deferred_results: None,
        }
    }

    /// Optional display title.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Attach a display title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Borrow provider-defined-tool provider options.
    pub fn provider_options_map(&self) -> &crate::types::ProviderOptionsMap {
        &self.provider_options_map
    }

    /// Replace provider-defined-tool provider options.
    pub fn with_provider_options_map(
        mut self,
        provider_options_map: crate::types::ProviderOptionsMap,
    ) -> Self {
        self.provider_options_map = provider_options_map;
        self
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

    /// Mark whether this provider-defined tool supports deferred results.
    pub fn with_supports_deferred_results(mut self, supports_deferred_results: bool) -> Self {
        self.supports_deferred_results = Some(supports_deferred_results);
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
