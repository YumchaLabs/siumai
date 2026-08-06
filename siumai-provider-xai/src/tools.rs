//! Typed xAI hosted tools for the Responses API.
//!
//! Hosted tools are provider-owned request semantics. They deliberately do not reuse the
//! provider-neutral function-tool contract and do not expose a raw JSON escape hatch.

use std::fmt;

use serde::{Deserialize, Serialize};

/// xAI web-search configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct XaiWebSearchTool {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_domains: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_domains: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_image_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_image_understanding: Option<bool>,
}

impl XaiWebSearchTool {
    pub const fn new() -> Self {
        Self {
            allowed_domains: Vec::new(),
            excluded_domains: Vec::new(),
            enable_image_search: None,
            enable_image_understanding: None,
        }
    }

    pub fn with_allowed_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_excluded_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.excluded_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    pub const fn with_image_understanding(mut self, enabled: bool) -> Self {
        self.enable_image_understanding = Some(enabled);
        self
    }

    pub const fn with_image_search(mut self, enabled: bool) -> Self {
        self.enable_image_search = Some(enabled);
        self
    }
}

/// xAI X-search configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct XaiXSearchTool {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_x_handles: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_x_handles: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_image_understanding: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_video_understanding: Option<bool>,
}

impl XaiXSearchTool {
    pub const fn new() -> Self {
        Self {
            allowed_x_handles: Vec::new(),
            excluded_x_handles: Vec::new(),
            from_date: None,
            to_date: None,
            enable_image_understanding: None,
            enable_video_understanding: None,
        }
    }

    pub fn with_allowed_x_handles<I, S>(mut self, handles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_x_handles = handles.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_excluded_x_handles<I, S>(mut self, handles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.excluded_x_handles = handles.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_date_range(
        mut self,
        from_date: impl Into<String>,
        to_date: impl Into<String>,
    ) -> Self {
        self.from_date = Some(from_date.into());
        self.to_date = Some(to_date.into());
        self
    }

    pub const fn with_image_understanding(mut self, enabled: bool) -> Self {
        self.enable_image_understanding = Some(enabled);
        self
    }

    pub const fn with_video_understanding(mut self, enabled: bool) -> Self {
        self.enable_video_understanding = Some(enabled);
        self
    }
}

/// xAI file-search configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct XaiFileSearchTool {
    pub vector_store_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u32>,
}

impl XaiFileSearchTool {
    pub fn new<I, S>(vector_store_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            vector_store_ids: vector_store_ids.into_iter().map(Into::into).collect(),
            max_num_results: None,
        }
    }

    pub const fn with_max_num_results(mut self, maximum: u32) -> Self {
        self.max_num_results = Some(maximum);
        self
    }
}

/// xAI remote-MCP configuration.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct XaiMcpTool {
    pub server_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_tools: Vec<String>,
}

impl XaiMcpTool {
    pub fn new(server_url: impl Into<String>) -> Self {
        Self {
            server_url: server_url.into(),
            server_label: None,
            server_description: None,
            allowed_tools: Vec::new(),
        }
    }

    pub fn with_server_label(mut self, label: impl Into<String>) -> Self {
        self.server_label = Some(label.into());
        self
    }

    pub fn with_server_description(mut self, description: impl Into<String>) -> Self {
        self.server_description = Some(description.into());
        self
    }

    pub fn with_allowed_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = tools.into_iter().map(Into::into).collect();
        self
    }
}

impl fmt::Debug for XaiMcpTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiMcpTool")
            .field("server_url", &"[REDACTED]")
            .field("server_label", &self.server_label)
            .field("server_description", &self.server_description)
            .field("allowed_tools", &self.allowed_tools)
            .finish()
    }
}

/// Provider-owned xAI hosted tool accepted by the Responses API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum XaiResponsesTool {
    WebSearch { options: XaiWebSearchTool },
    XSearch { options: XaiXSearchTool },
    CodeExecution,
    ViewImage,
    ViewXVideo,
    FileSearch { options: XaiFileSearchTool },
    Mcp { options: XaiMcpTool },
}

impl XaiResponsesTool {
    pub const fn web_search() -> Self {
        Self::WebSearch {
            options: XaiWebSearchTool::new(),
        }
    }

    pub const fn web_search_with(options: XaiWebSearchTool) -> Self {
        Self::WebSearch { options }
    }

    pub const fn x_search() -> Self {
        Self::XSearch {
            options: XaiXSearchTool::new(),
        }
    }

    pub const fn x_search_with(options: XaiXSearchTool) -> Self {
        Self::XSearch { options }
    }

    pub const fn code_execution() -> Self {
        Self::CodeExecution
    }

    pub const fn view_image() -> Self {
        Self::ViewImage
    }

    pub const fn view_x_video() -> Self {
        Self::ViewXVideo
    }

    pub fn file_search<I, S>(vector_store_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::FileSearch {
            options: XaiFileSearchTool::new(vector_store_ids),
        }
    }

    pub const fn file_search_with(options: XaiFileSearchTool) -> Self {
        Self::FileSearch { options }
    }

    pub fn mcp(server_url: impl Into<String>) -> Self {
        Self::Mcp {
            options: XaiMcpTool::new(server_url),
        }
    }

    pub const fn mcp_with(options: XaiMcpTool) -> Self {
        Self::Mcp { options }
    }

    pub(crate) fn validate(&self, index: usize) -> Result<(), String> {
        match self {
            Self::WebSearch { options } => {
                if !options.allowed_domains.is_empty() && !options.excluded_domains.is_empty() {
                    return Err(format!(
                        "native_tools[{index}] cannot combine allowed_domains and excluded_domains"
                    ));
                }
                if options.allowed_domains.len() > 5 || options.excluded_domains.len() > 5 {
                    return Err(format!(
                        "native_tools[{index}] web-search domain filters accept at most five entries"
                    ));
                }
            }
            Self::XSearch { options } => {
                if !options.allowed_x_handles.is_empty() && !options.excluded_x_handles.is_empty() {
                    return Err(format!(
                        "native_tools[{index}] cannot combine allowed_x_handles and excluded_x_handles"
                    ));
                }
                if options.allowed_x_handles.len() > 10 || options.excluded_x_handles.len() > 10 {
                    return Err(format!(
                        "native_tools[{index}] X-search handle filters accept at most ten entries"
                    ));
                }
                let from = parse_tool_date(options.from_date.as_deref(), index, "from_date")?;
                let to = parse_tool_date(options.to_date.as_deref(), index, "to_date")?;
                if from.zip(to).is_some_and(|(from, to)| from > to) {
                    return Err(format!(
                        "native_tools[{index}].from_date must not be later than to_date"
                    ));
                }
            }
            Self::FileSearch { options } => {
                if options.vector_store_ids.is_empty() {
                    return Err(format!(
                        "native_tools[{index}].vector_store_ids must not be empty"
                    ));
                }
                if options.max_num_results == Some(0) {
                    return Err(format!(
                        "native_tools[{index}].max_num_results must be greater than zero"
                    ));
                }
            }
            Self::Mcp { options } => {
                let server_url = url::Url::parse(&options.server_url).map_err(|_| {
                    format!("native_tools[{index}].server_url must be an absolute HTTPS URL")
                })?;
                if server_url.scheme() != "https"
                    || !server_url.username().is_empty()
                    || server_url.password().is_some()
                {
                    return Err(format!(
                        "native_tools[{index}].server_url must be an HTTPS URL without embedded credentials"
                    ));
                }
            }
            Self::CodeExecution | Self::ViewImage | Self::ViewXVideo => {}
        }
        Ok(())
    }

    /// Encode the provider-native flat object expected by the xAI Responses API.
    ///
    /// The typed Rust representation keeps each tool's options nested so Serde can reject
    /// unknown fields reliably. Flattening happens only at the wire-codec boundary.
    pub(crate) fn as_value(&self) -> Result<serde_json::Value, serde_json::Error> {
        let (tool_type, options) = match self {
            Self::WebSearch { options } => ("web_search", serde_json::to_value(options)?),
            Self::XSearch { options } => ("x_search", serde_json::to_value(options)?),
            Self::CodeExecution => ("code_interpreter", serde_json::Value::Null),
            Self::ViewImage => ("view_image", serde_json::Value::Null),
            Self::ViewXVideo => ("view_x_video", serde_json::Value::Null),
            Self::FileSearch { options } => ("file_search", serde_json::to_value(options)?),
            Self::Mcp { options } => ("mcp", serde_json::to_value(options)?),
        };

        let mut object = match options {
            serde_json::Value::Null => serde_json::Map::new(),
            serde_json::Value::Object(object) => object,
            _ => {
                return Err(<serde_json::Error as serde::ser::Error>::custom(
                    "xAI hosted-tool options must serialize as an object",
                ));
            }
        };
        object.insert(
            "type".to_string(),
            serde_json::Value::String(tool_type.to_string()),
        );
        Ok(serde_json::Value::Object(object))
    }
}

fn parse_tool_date(
    value: Option<&str>,
    index: usize,
    field: &str,
) -> Result<Option<chrono::NaiveDate>, String> {
    value
        .map(|value| {
            chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .map_err(|_| format!("native_tools[{index}].{field} must use YYYY-MM-DD format"))
        })
        .transpose()
}

pub fn web_search() -> XaiResponsesTool {
    XaiResponsesTool::web_search()
}

pub fn web_search_with(options: XaiWebSearchTool) -> XaiResponsesTool {
    XaiResponsesTool::web_search_with(options)
}

pub fn x_search() -> XaiResponsesTool {
    XaiResponsesTool::x_search()
}

pub fn x_search_with(options: XaiXSearchTool) -> XaiResponsesTool {
    XaiResponsesTool::x_search_with(options)
}

pub fn code_execution() -> XaiResponsesTool {
    XaiResponsesTool::code_execution()
}

pub fn view_image() -> XaiResponsesTool {
    XaiResponsesTool::view_image()
}

pub fn view_x_video() -> XaiResponsesTool {
    XaiResponsesTool::view_x_video()
}

pub fn file_search<I, S>(vector_store_ids: I) -> XaiResponsesTool
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    XaiResponsesTool::file_search(vector_store_ids)
}

pub fn mcp(server_url: impl Into<String>) -> XaiResponsesTool {
    XaiResponsesTool::mcp(server_url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosted_tools_encode_native_snake_case_shapes() {
        let tool = XaiResponsesTool::web_search_with(
            XaiWebSearchTool::new().with_allowed_domains(["example.com"]),
        );
        assert_eq!(
            tool.as_value().expect("serialize hosted tool"),
            serde_json::json!({
                "type": "web_search",
                "allowed_domains": ["example.com"]
            })
        );
    }

    #[test]
    fn code_execution_uses_xai_wire_type() {
        assert_eq!(
            XaiResponsesTool::code_execution()
                .as_value()
                .expect("serialize code execution tool"),
            serde_json::json!({"type": "code_interpreter"})
        );
    }

    #[test]
    fn mcp_debug_redacts_server_url() {
        let tool = XaiResponsesTool::mcp("https://mcp.example.test/path?token=secret");
        let debug = format!("{tool:?}");
        assert!(!debug.contains("token=secret"));
        assert!(debug.contains("[REDACTED]"));
    }
}
