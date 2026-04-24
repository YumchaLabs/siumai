//! Tool choice and tool type utilities.

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
    Tool {
        /// Name of the tool to call
        name: String,
    },
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::None => serializer.serialize_str("none"),
            Self::Tool { name } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "tool")?;
                map.serialize_entry("toolName", name)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::from_json_value(value).map_err(serde::de::Error::custom)
    }
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

    fn from_json_value(value: serde_json::Value) -> Result<Self, String> {
        match value {
            serde_json::Value::String(value) => match value.as_str() {
                "auto" => Ok(Self::Auto),
                "required" => Ok(Self::Required),
                "none" => Ok(Self::None),
                other => Err(format!("unsupported tool choice string `{other}`")),
            },
            serde_json::Value::Object(mut object) => {
                if object.get("type").and_then(serde_json::Value::as_str) == Some("tool") {
                    let tool_name = object
                        .remove("toolName")
                        .or_else(|| object.remove("tool_name"))
                        .and_then(|value| value.as_str().map(str::to_string))
                        .ok_or("tool choice object is missing `toolName`")?;
                    return Ok(Self::tool(tool_name));
                }

                if let Some(serde_json::Value::Object(mut legacy_tool)) = object.remove("tool") {
                    let name = legacy_tool
                        .remove("name")
                        .and_then(|value| value.as_str().map(str::to_string))
                        .ok_or("legacy tool choice object is missing `tool.name`")?;
                    return Ok(Self::tool(name));
                }

                Err("unsupported tool choice object shape".to_string())
            }
            _ => Err("tool choice must be a string or object".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ToolChoice;

    #[test]
    fn tool_choice_tool_serializes_ai_sdk_shape() {
        let value = serde_json::to_value(ToolChoice::tool("weather")).expect("serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "type": "tool",
                "toolName": "weather"
            })
        );
    }

    #[test]
    fn tool_choice_roundtrips_ai_sdk_shapes() {
        assert_eq!(
            serde_json::from_value::<ToolChoice>(serde_json::json!("auto"))
                .expect("auto deserializes"),
            ToolChoice::Auto
        );
        assert_eq!(
            serde_json::from_value::<ToolChoice>(serde_json::json!("required"))
                .expect("required deserializes"),
            ToolChoice::Required
        );
        assert_eq!(
            serde_json::from_value::<ToolChoice>(serde_json::json!("none"))
                .expect("none deserializes"),
            ToolChoice::None
        );
        assert_eq!(
            serde_json::from_value::<ToolChoice>(serde_json::json!({
                "type": "tool",
                "toolName": "weather"
            }))
            .expect("tool deserializes"),
            ToolChoice::tool("weather")
        );
    }

    #[test]
    fn tool_choice_still_accepts_legacy_enum_shape() {
        let value = serde_json::json!({
            "tool": {
                "name": "weather"
            }
        });

        assert_eq!(
            serde_json::from_value::<ToolChoice>(value).expect("legacy tool deserializes"),
            ToolChoice::tool("weather")
        );
    }
}
