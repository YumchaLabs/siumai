//! Groq built-in and remote-MCP tool helpers.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{GroqLanguageOptions, GroqResponsesOptions};

const MAX_MCP_LABEL_BYTES: usize = 128;
const MAX_MCP_HEADERS: usize = 32;
const MAX_MCP_HEADER_NAME_BYTES: usize = 256;
const MAX_MCP_HEADER_VALUE_BYTES: usize = 8 * 1024;

/// Approval policy for a Groq remote-MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GroqMcpApproval {
    Always,
    Never,
}

/// Provider-owned remote-MCP tool accepted by Groq Responses.
///
/// Headers may contain authorization material. `Debug` therefore reports only their count, and
/// callers must opt in to serialization before the wire codec can observe their values.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct GroqRemoteMcpTool {
    server_label: String,
    server_url: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    header_entries: Vec<GroqMcpHeader>,
    #[serde(skip_serializing_if = "Option::is_none")]
    require_approval: Option<GroqMcpApproval>,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroqMcpHeader {
    name: String,
    value: String,
}

impl GroqRemoteMcpTool {
    pub fn new(server_label: impl Into<String>, server_url: impl Into<String>) -> Self {
        Self {
            server_label: server_label.into(),
            server_url: server_url.into(),
            header_entries: Vec::new(),
            require_approval: None,
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        let name = name.into();
        let value = value.into();
        if let Some(header) = self
            .header_entries
            .iter_mut()
            .find(|header| header.name.eq_ignore_ascii_case(&name))
        {
            header.name = name;
            header.value = value;
        } else {
            self.header_entries.push(GroqMcpHeader { name, value });
        }
        self
    }

    pub const fn with_require_approval(mut self, policy: GroqMcpApproval) -> Self {
        self.require_approval = Some(policy);
        self
    }

    pub fn server_label(&self) -> &str {
        &self.server_label
    }

    pub fn server_url(&self) -> &str {
        &self.server_url
    }

    pub fn require_approval(&self) -> Option<GroqMcpApproval> {
        self.require_approval
    }

    pub(crate) fn validate(&self, index: usize) -> Result<(), String> {
        if invalid_bounded_text(&self.server_label, MAX_MCP_LABEL_BYTES) {
            return Err(format!(
                "remote_mcp_tools[{index}].server_label must be non-empty, control-free, and at most {MAX_MCP_LABEL_BYTES} bytes"
            ));
        }
        let server_url = url::Url::parse(&self.server_url).map_err(|_| {
            format!("remote_mcp_tools[{index}].server_url must be an absolute HTTPS URL")
        })?;
        if server_url.scheme() != "https"
            || !server_url.username().is_empty()
            || server_url.password().is_some()
            || server_url.host_str().is_none()
        {
            return Err(format!(
                "remote_mcp_tools[{index}].server_url must be an HTTPS URL without embedded credentials"
            ));
        }
        if self.header_entries.len() > MAX_MCP_HEADERS {
            return Err(format!(
                "remote_mcp_tools[{index}].headers must contain at most {MAX_MCP_HEADERS} entries"
            ));
        }
        for header in &self.header_entries {
            if invalid_bounded_text(&header.name, MAX_MCP_HEADER_NAME_BYTES)
                || header.value.len() > MAX_MCP_HEADER_VALUE_BYTES
                || header.value.chars().any(char::is_control)
            {
                return Err(format!(
                    "remote_mcp_tools[{index}].headers contains an invalid name or value"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn as_value(&self) -> Result<Value, serde_json::Error> {
        let mut object = serde_json::Map::new();
        object.insert("type".to_string(), Value::String("mcp".to_string()));
        object.insert(
            "server_label".to_string(),
            Value::String(self.server_label.clone()),
        );
        object.insert(
            "server_url".to_string(),
            Value::String(self.server_url.clone()),
        );
        if !self.header_entries.is_empty() {
            let headers = self
                .header_entries
                .iter()
                .map(|header| (header.name.clone(), Value::String(header.value.clone())))
                .collect();
            object.insert("headers".to_string(), Value::Object(headers));
        }
        if let Some(require_approval) = self.require_approval {
            object.insert(
                "require_approval".to_string(),
                serde_json::to_value(require_approval)?,
            );
        }
        Ok(Value::Object(object))
    }
}

impl fmt::Debug for GroqRemoteMcpTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqRemoteMcpTool")
            .field("server_label", &self.server_label)
            .field("server_url", &"[REDACTED]")
            .field("header_count", &self.header_entries.len())
            .field("require_approval", &self.require_approval)
            .finish()
    }
}

fn invalid_bounded_text(value: &str, maximum_bytes: usize) -> bool {
    value.trim().is_empty() || value.len() > maximum_bytes || value.chars().any(char::is_control)
}

/// Create the typed provider option that enables Groq's built-in browser-search tool.
///
/// The tool is provider executed and therefore is not represented as a portable local
/// [`siumai_core::ToolSpec`].
pub fn browser_search() -> GroqLanguageOptions {
    GroqLanguageOptions::new().with_browser_search(true)
}

/// Enable Groq browser search on a Responses model.
pub fn responses_browser_search() -> GroqResponsesOptions {
    GroqResponsesOptions::new().with_browser_search(true)
}

/// Enable Groq code execution on a Responses model.
pub fn responses_code_execution() -> GroqResponsesOptions {
    GroqResponsesOptions::new().with_code_execution(true)
}

/// Configure one Groq remote-MCP server on a Responses model.
pub fn responses_remote_mcp(tool: GroqRemoteMcpTool) -> GroqResponsesOptions {
    GroqResponsesOptions::new().with_remote_mcp_tool(tool)
}

#[cfg(test)]
mod tests {
    use siumai_core::ProviderOptions;

    use super::*;

    #[test]
    fn built_in_and_remote_mcp_helpers_build_typed_options() {
        let options = ProviderOptions::typed(&browser_search()).unwrap();
        assert_eq!(options.namespace().as_str(), "groq");
        assert_eq!(options.value()["browser_search"], true);

        let options = ProviderOptions::typed(&responses_browser_search()).unwrap();
        assert_eq!(options.value()["browser_search"], true);
        let options = ProviderOptions::typed(&responses_code_execution()).unwrap();
        assert_eq!(options.value()["code_execution"], true);

        let tool = GroqRemoteMcpTool::new("docs", "https://mcp.example.com/sse")
            .with_header("Authorization", "Bearer secret")
            .with_require_approval(GroqMcpApproval::Always);
        let debug = format!("{tool:?}");
        assert!(!debug.contains("Bearer secret"));
        assert!(!debug.contains("mcp.example.com"));
        let options = ProviderOptions::typed(&responses_remote_mcp(tool)).unwrap();
        assert_eq!(
            options.value()["remote_mcp_tools"][0]["server_label"],
            "docs"
        );
    }

    #[test]
    fn remote_mcp_rejects_non_https_or_credentialed_urls() {
        for url in [
            "http://example.com/mcp",
            "https://user:secret@example.com/mcp",
        ] {
            let options = responses_remote_mcp(GroqRemoteMcpTool::new("docs", url));
            assert!(ProviderOptions::typed(&options).is_err());
        }
    }
}
