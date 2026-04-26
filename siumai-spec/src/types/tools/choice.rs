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

    /// Project this high-level tool-choice input onto the model-facing V4 object shape.
    pub fn to_language_model_v4(&self) -> LanguageModelV4ToolChoice {
        self.into()
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

/// AI SDK V4 model-facing tool-choice shape.
///
/// AI SDK accepts high-level tool choices as strings, but `prepareToolChoice(...)` projects them
/// onto this provider-facing object shape before model calls.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LanguageModelV4ToolChoice {
    /// The model may choose whether to call a tool.
    #[default]
    Auto,
    /// The model must call at least one tool.
    Required,
    /// The model must not call a tool.
    None,
    /// The model must call the named tool.
    Tool {
        /// Name of the tool to call.
        tool_name: String,
    },
}

impl LanguageModelV4ToolChoice {
    /// Create a model-facing tool choice that forces a specific tool.
    pub fn tool(tool_name: impl Into<String>) -> Self {
        Self::Tool {
            tool_name: tool_name.into(),
        }
    }

    /// Get the tool name if this is a Tool variant.
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Self::Tool { tool_name } => Some(tool_name),
            _ => None,
        }
    }

    fn from_json_value(value: serde_json::Value) -> Result<Self, String> {
        match value {
            serde_json::Value::String(value) => match value.as_str() {
                "auto" => Ok(Self::Auto),
                "required" => Ok(Self::Required),
                "none" => Ok(Self::None),
                other => Err(format!("unsupported v4 tool choice string `{other}`")),
            },
            serde_json::Value::Object(mut object) => match object
                .remove("type")
                .and_then(|value| value.as_str().map(str::to_string))
                .as_deref()
            {
                Some("auto") => Ok(Self::Auto),
                Some("required") => Ok(Self::Required),
                Some("none") => Ok(Self::None),
                Some("tool") => {
                    let tool_name = object
                        .remove("toolName")
                        .or_else(|| object.remove("tool_name"))
                        .and_then(|value| value.as_str().map(str::to_string))
                        .ok_or("v4 tool choice object is missing `toolName`")?;
                    Ok(Self::tool(tool_name))
                }
                Some(other) => Err(format!("unsupported v4 tool choice type `{other}`")),
                None => Err("v4 tool choice object is missing `type`".to_string()),
            },
            _ => Err("v4 tool choice must be an object or compatibility string".to_string()),
        }
    }
}

impl Serialize for LanguageModelV4ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "auto")?;
                map.end()
            }
            Self::Required => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "required")?;
                map.end()
            }
            Self::None => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "none")?;
                map.end()
            }
            Self::Tool { tool_name } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "tool")?;
                map.serialize_entry("toolName", tool_name)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for LanguageModelV4ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::from_json_value(value).map_err(serde::de::Error::custom)
    }
}

impl From<&ToolChoice> for LanguageModelV4ToolChoice {
    fn from(value: &ToolChoice) -> Self {
        match value {
            ToolChoice::Auto => Self::Auto,
            ToolChoice::Required => Self::Required,
            ToolChoice::None => Self::None,
            ToolChoice::Tool { name } => Self::tool(name.clone()),
        }
    }
}

impl From<ToolChoice> for LanguageModelV4ToolChoice {
    fn from(value: ToolChoice) -> Self {
        Self::from(&value)
    }
}

/// Project an optional high-level tool choice onto the AI SDK model-facing V4 shape.
pub fn prepare_tool_choice(tool_choice: Option<&ToolChoice>) -> LanguageModelV4ToolChoice {
    tool_choice
        .map(LanguageModelV4ToolChoice::from)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{LanguageModelV4ToolChoice, ToolChoice, prepare_tool_choice};

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

    #[test]
    fn language_model_v4_tool_choice_serializes_model_facing_objects() {
        assert_eq!(
            serde_json::to_value(LanguageModelV4ToolChoice::Auto).expect("serialize"),
            serde_json::json!({ "type": "auto" })
        );
        assert_eq!(
            serde_json::to_value(LanguageModelV4ToolChoice::Required).expect("serialize"),
            serde_json::json!({ "type": "required" })
        );
        assert_eq!(
            serde_json::to_value(LanguageModelV4ToolChoice::None).expect("serialize"),
            serde_json::json!({ "type": "none" })
        );
        assert_eq!(
            serde_json::to_value(LanguageModelV4ToolChoice::tool("weather")).expect("serialize"),
            serde_json::json!({
                "type": "tool",
                "toolName": "weather"
            })
        );
    }

    #[test]
    fn prepare_tool_choice_matches_ai_sdk_projection() {
        assert_eq!(prepare_tool_choice(None), LanguageModelV4ToolChoice::Auto);
        assert_eq!(
            prepare_tool_choice(Some(&ToolChoice::Required)),
            LanguageModelV4ToolChoice::Required
        );
        assert_eq!(
            prepare_tool_choice(Some(&ToolChoice::None)),
            LanguageModelV4ToolChoice::None
        );
        assert_eq!(
            prepare_tool_choice(Some(&ToolChoice::tool("weather"))),
            LanguageModelV4ToolChoice::tool("weather")
        );
    }

    #[test]
    fn language_model_v4_tool_choice_accepts_objects_and_compat_strings() {
        assert_eq!(
            serde_json::from_value::<LanguageModelV4ToolChoice>(serde_json::json!({
                "type": "tool",
                "toolName": "weather"
            }))
            .expect("object deserializes"),
            LanguageModelV4ToolChoice::tool("weather")
        );
        assert_eq!(
            serde_json::from_value::<LanguageModelV4ToolChoice>(serde_json::json!("none"))
                .expect("compat string deserializes"),
            LanguageModelV4ToolChoice::None
        );
    }
}
