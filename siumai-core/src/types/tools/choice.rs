//! Tool choice and tool type utilities.

use serde::{Deserialize, Serialize};

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
