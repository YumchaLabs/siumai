//! Anthropic Provider-Defined Tools
//!
//! Factory functions for creating Anthropic-specific provider-defined tools.
//! These tools are executed by Anthropic's servers.

use crate::types::{ProviderDefinedTool, Tool};

/// Web search configuration builder (2025-03-05 version)
#[derive(Debug, Clone, Default)]
pub struct WebSearch20250305Config {
    max_uses: Option<u32>,
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
}

impl WebSearch20250305Config {
    /// Create a new web search configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of times this tool can be used in a conversation
    pub fn with_max_uses(mut self, max: u32) -> Self {
        self.max_uses = Some(max);
        self
    }

    /// Set the list of allowed domains for search results
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self
    }

    /// Set the list of blocked domains for search results
    pub fn with_blocked_domains(mut self, domains: Vec<String>) -> Self {
        self.blocked_domains = Some(domains);
        self
    }

    /// Build the Tool
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(max) = self.max_uses {
            args["max_uses"] = serde_json::json!(max);
        }

        if let Some(allowed) = self.allowed_domains {
            args["allowed_domains"] = serde_json::json!(allowed);
        }

        if let Some(blocked) = self.blocked_domains {
            args["blocked_domains"] = serde_json::json!(blocked);
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("anthropic.web_search_20250305", "web_search_20250305")
                .with_args(args),
        )
    }
}

/// Create a web search tool (2025-03-05 version)
pub fn web_search_20250305() -> WebSearch20250305Config {
    WebSearch20250305Config::new()
}
