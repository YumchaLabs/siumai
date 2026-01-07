//! Tool calling and function definition types

use serde::{Deserialize, Serialize};

// Deprecated ToolCall and FunctionCall removed. Use ContentPart::ToolCall and tool_result helpers.

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

        if !de.extra.is_empty() {
            if let serde_json::Value::Object(ref mut map) = args {
                for (k, v) in de.extra.drain() {
                    map.entry(k).or_insert(v);
                }
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
}

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
/// let tool = Tool::provider_defined("openai.web_search", "web_search");
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

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// Function name
    pub name: String,
    /// Function description
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// JSON schema for function parameters
    pub parameters: serde_json::Value,

    /// Tool input examples (Vercel-aligned).
    ///
    /// Vercel's AI SDK accepts `inputExamples: [{ input: {...} }]` on function tools and
    /// forwards them to Anthropic as `input_examples: [{...}]`.
    #[serde(
        default,
        rename = "inputExamples",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_examples: Option<Vec<serde_json::Value>>,

    /// Strict mode setting for the tool (Vercel-aligned).
    ///
    /// Providers that support strict mode will use this setting to determine
    /// how the input should be generated. Strict mode will always produce
    /// valid inputs, but it might limit what input schemas the model can use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,

    /// Tool-level provider options (Vercel-aligned).
    ///
    /// This is useful for provider-specific tool configuration knobs such as
    /// Anthropic's `defer_loading` for function tools.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "crate::types::ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: crate::types::ProviderOptionsMap,
}

/// Tool type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "file_search")]
    FileSearch,
    #[serde(rename = "web_search")]
    WebSearch,
}

/// Provider-agnostic tool choice strategy
///
/// This type follows the AI SDK (Vercel) standard and is compatible with
/// OpenAI, Anthropic, Gemini, and other major providers.
///
/// # Examples
///
/// ```rust
/// use siumai::types::ToolChoice;
///
/// // Let the model decide (default)
/// let choice = ToolChoice::Auto;
///
/// // Require the model to call at least one tool
/// let choice = ToolChoice::Required;
///
/// // Prevent the model from calling any tools
/// let choice = ToolChoice::None;
///
/// // Force the model to call a specific tool
/// let choice = ToolChoice::tool("weather");
/// ```
///
/// # Provider Mapping
///
/// Different providers have different representations:
///
/// - **OpenAI**: `"auto"`, `"required"`, `"none"`, or `{"type": "function", "function": {"name": "..."}}`
/// - **Anthropic**: `{"type": "auto"}`, `{"type": "any"}`, tools removed for "none", or `{"type": "tool", "name": "..."}`
/// - **Gemini**: `"AUTO"`, `"ANY"`, `"NONE"`, or function calling mode with specific function
///
/// The provider transformers handle these conversions automatically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// Let the model decide whether to call tools (default)
    ///
    /// The model can choose to call zero or more tools, or respond with text only.
    #[default]
    Auto,

    /// Require the model to call at least one tool
    ///
    /// The model must call one or more tools. It cannot respond with text only.
    /// Note: Not all providers support this mode.
    Required,

    /// Prevent the model from calling any tools
    ///
    /// The model will only respond with text and cannot call any tools.
    /// Note: Some providers (like Anthropic) implement this by removing tools from the request.
    None,

    /// Force the model to call a specific tool
    ///
    /// The model must call the specified tool. The tool name must match one of the
    /// tools provided in the request.
    #[serde(rename = "tool")]
    Tool {
        /// Name of the tool to call
        name: String,
    },
}

impl ToolChoice {
    /// Create a tool choice that forces a specific tool
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ToolChoice;
    ///
    /// let choice = ToolChoice::tool("weather");
    /// ```
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool { name: name.into() }
    }

    /// Check if this is the Auto variant
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Check if this is the Required variant
    pub fn is_required(&self) -> bool {
        matches!(self, Self::Required)
    }

    /// Check if this is the None variant
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if this is the Tool variant
    pub fn is_tool(&self) -> bool {
        matches!(self, Self::Tool { .. })
    }

    /// Get the tool name if this is a Tool variant
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Self::Tool { name } => Some(name),
            _ => None,
        }
    }
}

/// OpenAI-specific built-in tools
///
/// # Deprecation Notice
///
/// This enum is deprecated in favor of the new `hosted_tools` module which provides
/// a more flexible and type-safe API for provider-defined tools.
///
/// ## Migration Guide
///
/// **Old way (deprecated)**:
/// ```rust,ignore
/// use siumai::types::OpenAiBuiltInTool;
///
/// let tool = OpenAiBuiltInTool::WebSearch;
/// ```
///
/// **New way (recommended)**:
/// ```rust
/// use siumai::hosted_tools::openai;
///
/// let tool = openai::web_search().build();
/// ```
///
/// ### Migration Examples
///
/// #### Web Search
/// ```rust,ignore
/// // Old
/// OpenAiBuiltInTool::WebSearchOptions {
///     options: WebSearchOptions {
///         search_context_size: Some("high".into()),
///         user_location: Some(WebSearchUserLocation { ... }),
///     }
/// }
///
/// // New
/// use siumai::hosted_tools::openai;
/// openai::web_search()
///     .with_search_context_size("high")
///     .with_user_location(openai::UserLocation::new("approximate").with_country("US"))
///     .build()
/// ```
///
/// #### File Search
/// ```rust,ignore
/// // Old
/// OpenAiBuiltInTool::FileSearchOptions {
///     vector_store_ids: Some(vec!["vs_123".into()]),
///     max_num_results: Some(10),
///     ranking_options: None,
///     filters: None,
/// }
///
/// // New
/// use siumai::hosted_tools::openai;
/// openai::file_search()
///     .with_vector_store_ids(vec!["vs_123".to_string()])
///     .with_max_num_results(10)
///     .build()
/// ```
///
/// #### Computer Use
/// ```rust,ignore
/// // Old
/// OpenAiBuiltInTool::ComputerUse {
///     display_width: 1920,
///     display_height: 1080,
///     environment: "headless".into(),
/// }
///
/// // New
/// use siumai::hosted_tools::openai;
/// openai::computer_use(1920, 1080, "headless")
/// ```
///
/// See the `hosted_tools` module documentation for more examples.
#[deprecated(
    since = "0.12.0",
    note = "Use `hosted_tools::openai` module instead. See migration guide in the documentation."
)]
#[cfg(any())]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpenAiBuiltInTool {
    /// Web search tool
    WebSearch,
    /// Web search tool with extra provider-specific options
    WebSearchAdvanced { extra: serde_json::Value },
    /// Web search tool with typed options (per Vercel AI examples)
    WebSearchOptions { options: WebSearchOptions },
    /// File search tool
    FileSearch {
        /// Vector store IDs to search
        vector_store_ids: Option<Vec<String>>,
    },
    /// File search tool with extra options
    FileSearchAdvanced {
        /// Vector store IDs to search
        vector_store_ids: Option<Vec<String>>,
        /// Extra provider-specific options
        extra: serde_json::Value,
    },
    /// File search tool with typed options (aligned with OpenAI Responses API)
    FileSearchOptions {
        /// Vector store IDs to search
        vector_store_ids: Option<Vec<String>>,
        /// Maximum results to return (provider may ignore)
        max_num_results: Option<u32>,
        /// Ranking options
        #[serde(skip_serializing_if = "Option::is_none")]
        ranking_options: Option<FileSearchRankingOptions>,
        /// Filters for search
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<FileSearchFilter>,
    },
    /// Computer use tool
    ComputerUse {
        /// Display width
        display_width: u32,
        /// Display height
        display_height: u32,
        /// Environment type
        environment: String,
    },
    /// Computer use tool with advanced options
    ComputerUseAdvanced {
        /// Display width
        display_width: u32,
        /// Display height
        display_height: u32,
        /// Environment type
        environment: String,
        /// Optional display scale
        #[serde(skip_serializing_if = "Option::is_none")]
        display_scale: Option<f32>,
        /// Extra provider-specific options
        extra: serde_json::Value,
    },
}

#[cfg(any())]
impl OpenAiBuiltInTool {
    /// Convert to JSON for API requests
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::WebSearch => serde_json::json!({
                "type": "web_search_preview"
            }),
            Self::WebSearchAdvanced { extra } => {
                let mut json = serde_json::json!({
                    "type": "web_search_preview"
                });
                // merge extra fields
                if let Some(map) = extra.as_object() {
                    for (k, v) in map {
                        json[k] = v.clone();
                    }
                }
                json
            }
            Self::WebSearchOptions { options } => {
                let mut json = serde_json::json!({ "type": "web_search_preview" });
                if let Some(s) = &options.search_context_size {
                    json["searchContextSize"] = serde_json::json!(s);
                }
                if let Some(loc) = &options.user_location {
                    let mut o = serde_json::json!({ "type": loc.r#type });
                    if let Some(country) = &loc.country {
                        o["country"] = serde_json::json!(country);
                    }
                    if let Some(city) = &loc.city {
                        o["city"] = serde_json::json!(city);
                    }
                    if let Some(region) = &loc.region {
                        o["region"] = serde_json::json!(region);
                    }
                    if let Some(tz) = &loc.timezone {
                        o["timezone"] = serde_json::json!(tz);
                    }
                    json["userLocation"] = o;
                }
                json
            }
            Self::FileSearch { vector_store_ids } => {
                let mut json = serde_json::json!({
                    "type": "file_search"
                });
                if let Some(ids) = vector_store_ids {
                    json["vector_store_ids"] = serde_json::Value::Array(
                        ids.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect(),
                    );
                }
                json
            }
            Self::FileSearchAdvanced {
                vector_store_ids,
                extra,
            } => {
                let mut json = serde_json::json!({
                    "type": "file_search"
                });
                if let Some(ids) = vector_store_ids {
                    json["vector_store_ids"] = serde_json::Value::Array(
                        ids.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect(),
                    );
                }
                if let Some(map) = extra.as_object() {
                    for (k, v) in map {
                        json[k] = v.clone();
                    }
                }
                json
            }
            Self::FileSearchOptions {
                vector_store_ids,
                max_num_results,
                ranking_options,
                filters,
            } => {
                let mut json = serde_json::json!({ "type": "file_search" });
                if let Some(ids) = vector_store_ids {
                    json["vector_store_ids"] = serde_json::Value::Array(
                        ids.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect(),
                    );
                }
                if let Some(k) = *max_num_results {
                    json["max_num_results"] = serde_json::json!(k);
                }
                if let Some(ro) = ranking_options {
                    let mut o = serde_json::json!({});
                    if let Some(r) = ro.ranker.as_ref() {
                        o["ranker"] = serde_json::json!(r);
                    }
                    if let Some(th) = ro.score_threshold {
                        o["score_threshold"] = serde_json::json!(th);
                    }
                    json["ranking_options"] = o;
                }
                if let Some(f) = filters {
                    json["filters"] = f.to_json();
                }
                json
            }
            Self::ComputerUse {
                display_width,
                display_height,
                environment,
            } => serde_json::json!({
                "type": "computer_use_preview",
                "display_width": display_width,
                "display_height": display_height,
                "environment": environment
            }),
            Self::ComputerUseAdvanced {
                display_width,
                display_height,
                environment,
                display_scale,
                extra,
            } => {
                let mut json = serde_json::json!({
                    "type": "computer_use_preview",
                    "display_width": display_width,
                    "display_height": display_height,
                    "environment": environment
                });
                if let Some(scale) = display_scale {
                    json["display_scale"] = serde_json::json!(scale);
                }
                if let Some(map) = extra.as_object() {
                    for (k, v) in map {
                        json[k] = v.clone();
                    }
                }
                json
            }
        }
    }
}

/// Typed options for OpenAI web search built-in tool (subset based on Vercel AI docs)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(any())]
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(any())]
pub struct WebSearchUserLocation {
    #[serde(rename = "type")]
    pub r#type: String, // e.g., "approximate"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(any())]
pub struct FileSearchRankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[cfg(any())]
pub enum FileSearchFilter {
    #[serde(rename = "eq")]
    Eq {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "ne")]
    Ne {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "gt")]
    Gt {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "gte")]
    Gte {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "lt")]
    Lt {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "lte")]
    Lte {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "and")]
    And { filters: Vec<FileSearchFilter> },
    #[serde(rename = "or")]
    Or { filters: Vec<FileSearchFilter> },
}

#[cfg(any())]
impl FileSearchFilter {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            FileSearchFilter::Eq { key, value } => serde_json::json!({
                "type": "eq", "key": key, "value": value
            }),
            FileSearchFilter::Ne { key, value } => serde_json::json!({
                "type": "ne", "key": key, "value": value
            }),
            FileSearchFilter::Gt { key, value } => serde_json::json!({
                "type": "gt", "key": key, "value": value
            }),
            FileSearchFilter::Gte { key, value } => serde_json::json!({
                "type": "gte", "key": key, "value": value
            }),
            FileSearchFilter::Lt { key, value } => serde_json::json!({
                "type": "lt", "key": key, "value": value
            }),
            FileSearchFilter::Lte { key, value } => serde_json::json!({
                "type": "lte", "key": key, "value": value
            }),
            FileSearchFilter::And { filters } => serde_json::json!({
                "type": "and",
                "filters": filters.iter().map(|f| f.to_json()).collect::<Vec<_>>()
            }),
            FileSearchFilter::Or { filters } => serde_json::json!({
                "type": "or",
                "filters": filters.iter().map(|f| f.to_json()).collect::<Vec<_>>()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any())]
    fn web_search_advanced_merges_extra() {
        let t = OpenAiBuiltInTool::WebSearchAdvanced {
            extra: serde_json::json!({"region":"us","safe":"moderate"}),
        };
        let v = t.to_json();
        assert_eq!(
            v.get("type").and_then(|s| s.as_str()),
            Some("web_search_preview")
        );
        assert_eq!(v.get("region").and_then(|s| s.as_str()), Some("us"));
        assert_eq!(v.get("safe").and_then(|s| s.as_str()), Some("moderate"));
    }

    #[test]
    #[cfg(any())]
    fn file_search_advanced_includes_ids_and_extra() {
        let t = OpenAiBuiltInTool::FileSearchAdvanced {
            vector_store_ids: Some(vec!["vs1".into(), "vs2".into()]),
            extra: serde_json::json!({"ranker":"semantic"}),
        };
        let v = t.to_json();
        assert_eq!(v.get("type").and_then(|s| s.as_str()), Some("file_search"));
        let ids = v
            .get("vector_store_ids")
            .and_then(|a| a.as_array())
            .cloned()
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(v.get("ranker").and_then(|s| s.as_str()), Some("semantic"));
    }

    #[test]
    #[cfg(any())]
    fn computer_use_advanced_includes_scale_and_extra() {
        let t = OpenAiBuiltInTool::ComputerUseAdvanced {
            display_width: 1280,
            display_height: 720,
            environment: "headless".into(),
            display_scale: Some(1.5),
            extra: serde_json::json!({"cursor":"enabled"}),
        };
        let v = t.to_json();
        assert_eq!(
            v.get("type").and_then(|s| s.as_str()),
            Some("computer_use_preview")
        );
        assert_eq!(v.get("display_width").and_then(|x| x.as_u64()), Some(1280));
        assert_eq!(v.get("display_height").and_then(|x| x.as_u64()), Some(720));
        assert_eq!(
            v.get("environment").and_then(|s| s.as_str()),
            Some("headless")
        );
        assert_eq!(v.get("display_scale").and_then(|x| x.as_f64()), Some(1.5));
        assert_eq!(v.get("cursor").and_then(|s| s.as_str()), Some("enabled"));
    }

    #[test]
    #[cfg(any())]
    fn web_search_options_includes_country_timezone() {
        let t = OpenAiBuiltInTool::WebSearchOptions {
            options: WebSearchOptions {
                search_context_size: Some("high".into()),
                user_location: Some(WebSearchUserLocation {
                    r#type: "approximate".into(),
                    country: Some("US".into()),
                    city: Some("SF".into()),
                    region: Some("CA".into()),
                    timezone: Some("America/Los_Angeles".into()),
                }),
            },
        };
        let v = t.to_json();
        let loc = v.get("userLocation").and_then(|o| o.as_object()).unwrap();
        assert_eq!(loc.get("country").and_then(|x| x.as_str()), Some("US"));
        assert_eq!(
            loc.get("timezone").and_then(|x| x.as_str()),
            Some("America/Los_Angeles")
        );
    }

    #[test]
    #[cfg(any())]
    fn file_search_options_includes_max_num_results_ranking_filters() {
        let t = OpenAiBuiltInTool::FileSearchOptions {
            vector_store_ids: Some(vec!["vs1".into()]),
            max_num_results: Some(15),
            ranking_options: Some(FileSearchRankingOptions {
                ranker: Some("auto".into()),
                score_threshold: Some(0.5),
            }),
            filters: Some(FileSearchFilter::And {
                filters: vec![
                    FileSearchFilter::Eq {
                        key: "doc_type".into(),
                        value: serde_json::json!("pdf"),
                    },
                    FileSearchFilter::Gt {
                        key: "score".into(),
                        value: serde_json::json!(0.1),
                    },
                ],
            }),
        };
        let v = t.to_json();
        assert_eq!(v.get("type").and_then(|s| s.as_str()), Some("file_search"));
        assert_eq!(v.get("max_num_results").and_then(|x| x.as_u64()), Some(15));
        let ro = v
            .get("ranking_options")
            .and_then(|o| o.as_object())
            .unwrap();
        assert_eq!(ro.get("ranker").and_then(|x| x.as_str()), Some("auto"));
        assert_eq!(
            ro.get("score_threshold").and_then(|x| x.as_f64()),
            Some(0.5)
        );
        assert!(v.get("filters").is_some());
    }

    #[test]
    fn provider_defined_tool_new() {
        let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
        assert_eq!(tool.id, "openai.web_search");
        assert_eq!(tool.name, "web_search");
        assert_eq!(tool.args, serde_json::json!({}));
    }

    #[test]
    fn provider_defined_tool_with_args() {
        let tool = ProviderDefinedTool::new("openai.web_search", "web_search").with_args(
            serde_json::json!({
                "searchContextSize": "high"
            }),
        );
        assert_eq!(tool.id, "openai.web_search");
        assert_eq!(tool.name, "web_search");
        assert_eq!(
            tool.args.get("searchContextSize").and_then(|v| v.as_str()),
            Some("high")
        );
    }

    #[test]
    fn provider_defined_tool_provider() {
        let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
        assert_eq!(tool.provider(), Some("openai"));

        let tool2 = ProviderDefinedTool::new("anthropic.web_search_20250305", "web_search");
        assert_eq!(tool2.provider(), Some("anthropic"));

        let tool3 = ProviderDefinedTool::new("invalid", "test");
        assert_eq!(tool3.provider(), Some("invalid"));
    }

    #[test]
    fn provider_defined_tool_tool_type() {
        let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
        assert_eq!(tool.tool_type(), Some("web_search"));

        let tool2 = ProviderDefinedTool::new("google.code_execution", "code_execution");
        assert_eq!(tool2.tool_type(), Some("code_execution"));

        let tool3 = ProviderDefinedTool::new("invalid", "test");
        assert_eq!(tool3.tool_type(), None);
    }

    #[test]
    fn tool_enum_function_variant() {
        let tool = Tool::function(
            "weather".to_string(),
            "Get weather".to_string(),
            serde_json::json!({}),
        );
        match tool {
            Tool::Function { function } => {
                assert_eq!(function.name, "weather");
                assert_eq!(function.description, "Get weather");
            }
            _ => panic!("Expected Function variant"),
        }
    }

    #[test]
    fn tool_enum_provider_defined_variant() {
        let provider_tool = ProviderDefinedTool::new("openai.web_search", "web_search");
        let tool = Tool::ProviderDefined(provider_tool.clone());
        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "openai.web_search");
                assert_eq!(pt.name, "web_search");
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }

    #[test]
    fn tool_enum_serialization() {
        // Test Function variant serialization
        let func_tool = Tool::function(
            "weather".to_string(),
            "Get weather".to_string(),
            serde_json::json!({}),
        );
        let json = serde_json::to_value(&func_tool).unwrap();
        assert_eq!(json.get("type").and_then(|v| v.as_str()), Some("function"));
        assert_eq!(json.get("name").and_then(|v| v.as_str()), Some("weather"));

        // Test ProviderDefined variant serialization
        let provider_tool = ProviderDefinedTool::new("openai.web_search", "web_search");
        let pd_tool = Tool::ProviderDefined(provider_tool);
        let json = serde_json::to_value(&pd_tool).unwrap();
        assert_eq!(
            json.get("type").and_then(|v| v.as_str()),
            Some("provider-defined")
        );
        assert_eq!(
            json.get("id").and_then(|v| v.as_str()),
            Some("openai.web_search")
        );
        assert_eq!(
            json.get("name").and_then(|v| v.as_str()),
            Some("web_search")
        );
        assert_eq!(json.get("args"), Some(&serde_json::json!({})));
    }

    #[test]
    fn provider_defined_tool_deserializes_vercel_shape() {
        let v = serde_json::json!({
            "type": "provider",
            "id": "openai.web_search_preview",
            "name": "web_search_preview",
            "args": {
                "search_context_size": "low"
            }
        });

        let tool: Tool = serde_json::from_value(v).expect("deserialize provider tool");
        let Tool::ProviderDefined(tool) = tool else {
            panic!("expected ProviderDefined");
        };
        assert_eq!(tool.id, "openai.web_search_preview");
        assert_eq!(tool.name, "web_search_preview");
        assert_eq!(
            tool.args
                .get("search_context_size")
                .and_then(|v| v.as_str()),
            Some("low")
        );
    }

    #[test]
    fn provider_defined_tool_deserializes_legacy_flatten_shape() {
        let v = serde_json::json!({
            "type": "provider-defined",
            "id": "openai.web_search",
            "name": "web_search",
            "searchContextSize": "high",
            "userLocation": {
                "type": "approximate",
                "country": "US"
            }
        });

        let tool: Tool = serde_json::from_value(v).expect("deserialize legacy provider tool");
        let Tool::ProviderDefined(tool) = tool else {
            panic!("expected ProviderDefined");
        };
        assert_eq!(tool.id, "openai.web_search");
        assert_eq!(tool.name, "web_search");
        assert_eq!(
            tool.args.get("searchContextSize").and_then(|v| v.as_str()),
            Some("high")
        );
        assert_eq!(
            tool.args
                .get("userLocation")
                .and_then(|v| v.get("country"))
                .and_then(|v| v.as_str()),
            Some("US")
        );
    }

    #[test]
    fn tool_enum_deserialization() {
        // Test Function variant deserialization
        let json = serde_json::json!({
            "type": "function",
            "name": "weather",
            "description": "Get weather",
            "parameters": {}
        });
        let tool: Tool = serde_json::from_value(json).unwrap();
        match tool {
            Tool::Function { function } => {
                assert_eq!(function.name, "weather");
            }
            _ => panic!("Expected Function variant"),
        }

        // Test ProviderDefined variant deserialization
        let json = serde_json::json!({
            "type": "provider-defined",
            "id": "openai.web_search",
            "name": "web_search"
        });
        let tool: Tool = serde_json::from_value(json).unwrap();
        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "openai.web_search");
                assert_eq!(pt.name, "web_search");
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }
}
