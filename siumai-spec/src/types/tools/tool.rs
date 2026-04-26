//! Tool enum and constructors.

use serde::{Deserialize, Serialize};

use super::{
    LanguageModelV4FunctionTool, LanguageModelV4ProviderTool, ProviderDefinedTool, ToolFunction,
};

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

    /// Provider-defined/provider-executed tool (web search, file search, code execution, etc.)
    #[serde(rename = "provider", alias = "provider-defined")]
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
            function: ToolFunction::new(name, description, parameters),
        }
    }

    /// Optional display title for the tool.
    pub fn title(&self) -> Option<&str> {
        match self {
            Self::Function { function } => function.title(),
            Self::ProviderDefined(tool) => tool.title(),
        }
    }

    /// Attach an optional display title.
    pub fn with_title(self, title: impl Into<String>) -> Self {
        let title = title.into();
        match self {
            Self::Function { function } => Self::Function {
                function: function.with_title(title),
            },
            Self::ProviderDefined(tool) => Self::ProviderDefined(tool.with_title(title)),
        }
    }

    /// Create a function tool with AI SDK-style output schema metadata.
    pub fn function_with_output_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self::Function {
            function: ToolFunction::new(name, description, input_schema)
                .with_output_schema(output_schema),
        }
    }

    /// Create a hosted provider tool.
    ///
    /// This legacy constructor defaults to `isProviderExecuted: true`, matching the historical
    /// Siumai hosted-tool surface. Use `provider_defined_with_schema` for AI SDK provider-defined
    /// tools whose execution is local.
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

    /// Create an AI SDK-style provider-defined tool whose execution is local.
    pub fn provider_defined_with_schema(
        id: impl Into<String>,
        name: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self::ProviderDefined(
            ProviderDefinedTool::provider_defined(id, name).with_input_schema(input_schema),
        )
    }

    /// Create an AI SDK-style provider-executed tool with provider-defined schemas.
    pub fn provider_executed_with_schema(
        id: impl Into<String>,
        name: impl Into<String>,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self::ProviderDefined(
            ProviderDefinedTool::provider_executed(id, name)
                .with_input_schema(input_schema)
                .with_output_schema(output_schema),
        )
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

    /// Attach AI SDK-style deferred-result metadata when this is a provider-defined tool.
    pub fn with_supports_deferred_results(self, supports_deferred_results: bool) -> Self {
        match self {
            Self::ProviderDefined(tool) => Self::ProviderDefined(
                tool.with_supports_deferred_results(supports_deferred_results),
            ),
            other => other,
        }
    }

    /// Whether this is a provider tool that is executed by the provider.
    pub fn is_provider_executed(&self) -> Option<bool> {
        match self {
            Self::ProviderDefined(tool) => Some(tool.is_provider_executed()),
            Self::Function { .. } => None,
        }
    }

    /// Set AI SDK-style provider execution ownership when this is a provider tool.
    pub fn with_provider_executed(self, is_provider_executed: bool) -> Self {
        match self {
            Self::ProviderDefined(tool) => {
                Self::ProviderDefined(tool.with_provider_executed(is_provider_executed))
            }
            other => other,
        }
    }

    /// Return the inner function schema when this is a function tool.
    pub const fn function_ref(&self) -> Option<&ToolFunction> {
        match self {
            Self::Function { function } => Some(function),
            Self::ProviderDefined(_) => None,
        }
    }

    /// Return the mutable inner function schema when this is a function tool.
    pub fn function_mut(&mut self) -> Option<&mut ToolFunction> {
        match self {
            Self::Function { function } => Some(function),
            Self::ProviderDefined(_) => None,
        }
    }

    /// AI SDK-style view over the function input schema.
    pub fn input_schema(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Function { function } => Some(function.input_schema()),
            Self::ProviderDefined(tool) => tool.input_schema(),
        }
    }

    /// Mutable AI SDK-style view over the function input schema.
    pub fn input_schema_mut(&mut self) -> Option<&mut serde_json::Value> {
        match self {
            Self::Function { function } => Some(function.input_schema_mut()),
            Self::ProviderDefined(tool) => tool.input_schema.as_mut(),
        }
    }

    /// Replace the function input schema when this is a function tool.
    pub fn with_input_schema(self, input_schema: serde_json::Value) -> Self {
        match self {
            Self::Function { function } => Self::Function {
                function: function.with_input_schema(input_schema),
            },
            Self::ProviderDefined(tool) => {
                Self::ProviderDefined(tool.with_input_schema(input_schema))
            }
        }
    }

    /// Return the optional AI SDK-style function output schema.
    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Function { function } => function.output_schema(),
            Self::ProviderDefined(tool) => tool.output_schema(),
        }
    }

    /// Attach AI SDK-style function output schema metadata when this is a function tool.
    pub fn with_output_schema(self, output_schema: serde_json::Value) -> Self {
        match self {
            Self::Function { function } => Self::Function {
                function: function.with_output_schema(output_schema),
            },
            Self::ProviderDefined(tool) => {
                Self::ProviderDefined(tool.with_output_schema(output_schema))
            }
        }
    }

    /// Return the optional AI SDK-style input examples metadata.
    pub fn input_examples(&self) -> Option<&[serde_json::Value]> {
        self.function_ref().and_then(ToolFunction::input_examples)
    }

    /// Attach AI SDK-style input examples metadata when this is a function tool.
    pub fn with_input_examples(
        self,
        input_examples: impl IntoIterator<Item = serde_json::Value>,
    ) -> Self {
        let input_examples = input_examples.into_iter().collect::<Vec<_>>();
        match self {
            Self::Function { function } => Self::Function {
                function: function.with_input_examples(input_examples),
            },
            other => other,
        }
    }

    /// Return the optional strict-mode setting.
    pub fn strict(&self) -> Option<bool> {
        self.function_ref().and_then(ToolFunction::strict)
    }

    /// Attach an AI SDK-style strict-mode setting when this is a function tool.
    pub fn with_strict(self, strict: bool) -> Self {
        match self {
            Self::Function { function } => Self::Function {
                function: function.with_strict(strict),
            },
            other => other,
        }
    }

    /// Borrow function-tool provider options when this is a function tool.
    pub fn provider_options_map(&self) -> Option<&crate::types::ProviderOptionsMap> {
        match self {
            Self::Function { function } => Some(function.provider_options_map()),
            Self::ProviderDefined(tool) => Some(tool.provider_options_map()),
        }
    }

    /// Replace tool-level provider options when this is a function or provider-defined tool.
    pub fn with_provider_options_map(
        self,
        provider_options_map: crate::types::ProviderOptionsMap,
    ) -> Self {
        match self {
            Self::Function { function } => Self::Function {
                function: function.with_provider_options_map(provider_options_map),
            },
            Self::ProviderDefined(tool) => {
                Self::ProviderDefined(tool.with_provider_options_map(provider_options_map))
            }
        }
    }

    /// Project this tool onto the AI SDK V4 model-facing function-tool shape when applicable.
    pub fn to_language_model_v4_function_tool(&self) -> Option<LanguageModelV4FunctionTool> {
        self.function_ref().map(LanguageModelV4FunctionTool::from)
    }

    /// Project this tool onto the AI SDK V4 model-facing provider-tool shape when applicable.
    pub fn to_language_model_v4_provider_tool(&self) -> Option<LanguageModelV4ProviderTool> {
        match self {
            Self::ProviderDefined(tool) => Some(LanguageModelV4ProviderTool::from(tool)),
            Self::Function { .. } => None,
        }
    }
}
