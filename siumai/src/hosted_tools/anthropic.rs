//! Anthropic Provider-Defined Tools
//!
//! This module provides factory functions for creating Anthropic-specific provider-defined tools.
//! These tools are executed by Anthropic's servers.
//!
//! # Examples
//!
//! ```rust
//! use siumai::hosted_tools::anthropic;
//!
//! // Create a web search tool (2025-03-05 version)
//! let web_search = anthropic::web_search_20250305();
//!
//! // Create a web search tool with custom configuration
//! let web_search_custom = anthropic::web_search_20250305()
//!     .with_max_uses(5)
//!     .with_allowed_domains(vec!["github.com".to_string(), "docs.rs".to_string()]);
//! ```

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
///
/// This is Anthropic's web search tool with the 2025-03-05 API version.
///
/// # Example
///
/// ```rust
/// use siumai::hosted_tools::anthropic;
///
/// let tool = anthropic::web_search_20250305()
///     .with_max_uses(3)
///     .with_allowed_domains(vec!["github.com".to_string()]);
/// ```
pub fn web_search_20250305() -> WebSearch20250305Config {
    WebSearch20250305Config::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_20250305_default() {
        let tool = web_search_20250305().build();
        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "anthropic.web_search_20250305");
                assert_eq!(pt.name, "web_search_20250305");
                assert_eq!(pt.provider(), Some("anthropic"));
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }

    #[test]
    fn test_web_search_20250305_with_config() {
        let tool = web_search_20250305()
            .with_max_uses(5)
            .with_allowed_domains(vec!["github.com".to_string(), "docs.rs".to_string()])
            .with_blocked_domains(vec!["spam.com".to_string()])
            .build();

        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "anthropic.web_search_20250305");
                assert_eq!(pt.args.get("max_uses").and_then(|v| v.as_u64()), Some(5));

                let allowed = pt.args.get("allowed_domains").unwrap().as_array().unwrap();
                assert_eq!(allowed.len(), 2);
                assert_eq!(allowed[0].as_str(), Some("github.com"));

                let blocked = pt.args.get("blocked_domains").unwrap().as_array().unwrap();
                assert_eq!(blocked.len(), 1);
                assert_eq!(blocked[0].as_str(), Some("spam.com"));
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }
}
