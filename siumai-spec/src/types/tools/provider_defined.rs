//! Provider-defined tool types.

use super::Tool;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ProviderToolType {
    Provider,
}

impl Default for ProviderToolType {
    fn default() -> Self {
        Self::Provider
    }
}

/// AI SDK V4 model-facing provider-tool shape.
///
/// This projection intentionally keeps only `{ type, id, name, args }`. Execution ownership,
/// schemas, deferred-result metadata, title, and provider options remain on the stable portable
/// `ProviderDefinedTool` surface where orchestration and UI layers can consume them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelV4ProviderTool {
    #[serde(rename = "type")]
    marker: ProviderToolType,
    /// Provider tool id in `<provider>.<tool>` format.
    pub id: String,
    /// Tool name unique within this model call.
    pub name: String,
    /// Provider-owned tool configuration arguments.
    pub args: serde_json::Value,
}

impl LanguageModelV4ProviderTool {
    /// Create a model-facing provider tool.
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            marker: ProviderToolType::Provider,
            id: id.into(),
            name: name.into(),
            args,
        }
    }
}

/// Provider-defined tool configuration
///
/// This represents an AI SDK-style `type: "provider"` tool. Most built-in hosted tools are both
/// defined and executed by the provider, but the shape can also represent tools whose schemas are
/// provider-defined while execution is owned by the local runtime.
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

    /// Optional provider-defined input schema.
    pub input_schema: Option<serde_json::Value>,

    /// Optional provider-defined output schema.
    pub output_schema: Option<serde_json::Value>,

    /// Whether this provider tool is executed by the provider.
    ///
    /// This mirrors AI SDK's `isProviderExecuted` flag. Legacy Siumai constructors default to
    /// `true` because the historical provider-defined tool surface represented hosted tools.
    pub is_provider_executed: bool,

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

const fn default_is_provider_executed() -> bool {
    true
}

impl serde::Serialize for ProviderDefinedTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let field_count = 4
            + usize::from(self.title.is_some())
            + usize::from(!self.provider_options_map.is_empty())
            + usize::from(self.input_schema.is_some())
            + usize::from(self.output_schema.is_some())
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
        if let Some(input_schema) = &self.input_schema {
            st.serialize_field("inputSchema", input_schema)?;
        }
        if let Some(output_schema) = &self.output_schema {
            st.serialize_field("outputSchema", output_schema)?;
        }
        st.serialize_field("isProviderExecuted", &self.is_provider_executed)?;
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
            #[serde(default, rename = "inputSchema", alias = "input_schema")]
            input_schema: Option<serde_json::Value>,
            #[serde(default, rename = "outputSchema", alias = "output_schema")]
            output_schema: Option<serde_json::Value>,
            #[serde(
                default = "default_is_provider_executed",
                rename = "isProviderExecuted",
                alias = "is_provider_executed"
            )]
            is_provider_executed: bool,
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
            input_schema: de.input_schema,
            output_schema: de.output_schema,
            is_provider_executed: de.is_provider_executed,
            args,
            supports_deferred_results: de.supports_deferred_results,
        })
    }
}

impl ProviderDefinedTool {
    /// Create a new hosted provider tool.
    ///
    /// This is the legacy constructor and defaults `is_provider_executed` to `true`.
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
            input_schema: None,
            output_schema: None,
            is_provider_executed: true,
            args: serde_json::Value::Object(Default::default()),
            supports_deferred_results: None,
        }
    }

    /// Create a tool whose input/output schemas are provider-defined but execution is local.
    ///
    /// This mirrors AI SDK provider-utils `createProviderDefinedToolFactory(...)`, which sets
    /// `isProviderExecuted: false`.
    pub fn provider_defined(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, name).with_provider_executed(false)
    }

    /// Create a hosted tool that is executed by the provider.
    ///
    /// This mirrors AI SDK provider-utils `createProviderExecutedToolFactory(...)`, which sets
    /// `isProviderExecuted: true`.
    pub fn provider_executed(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, name).with_provider_executed(true)
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

    /// Borrow the optional provider-defined input schema.
    pub fn input_schema(&self) -> Option<&serde_json::Value> {
        self.input_schema.as_ref()
    }

    /// Attach a provider-defined input schema.
    pub fn with_input_schema(mut self, input_schema: serde_json::Value) -> Self {
        self.input_schema = Some(input_schema);
        self
    }

    /// Borrow the optional provider-defined output schema.
    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        self.output_schema.as_ref()
    }

    /// Attach a provider-defined output schema.
    pub fn with_output_schema(mut self, output_schema: serde_json::Value) -> Self {
        self.output_schema = Some(output_schema);
        self
    }

    /// Whether the provider executes this tool.
    pub const fn is_provider_executed(&self) -> bool {
        self.is_provider_executed
    }

    /// Set whether the provider executes this tool.
    pub fn with_provider_executed(mut self, is_provider_executed: bool) -> Self {
        self.is_provider_executed = is_provider_executed;
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

    /// Project this stable provider tool onto the AI SDK V4 model-facing shape.
    pub fn to_language_model_v4(&self) -> LanguageModelV4ProviderTool {
        self.into()
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

impl From<&ProviderDefinedTool> for LanguageModelV4ProviderTool {
    fn from(value: &ProviderDefinedTool) -> Self {
        Self {
            marker: ProviderToolType::Provider,
            id: value.id.clone(),
            name: value.name.clone(),
            args: value.args.clone(),
        }
    }
}

impl From<ProviderDefinedTool> for LanguageModelV4ProviderTool {
    fn from(value: ProviderDefinedTool) -> Self {
        Self::from(&value)
    }
}
