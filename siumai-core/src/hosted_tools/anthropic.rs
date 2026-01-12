//! Anthropic Provider-Defined Tools
//!
//! Factory functions for creating Anthropic-specific provider-defined tools.
//! These tools are executed by Anthropic's servers.

use crate::types::{ProviderDefinedTool, Tool};

/// User location for Anthropic web search (Vercel-aligned).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl UserLocation {
    /// Create an approximate user location (the only supported type for `web_search_20250305`).
    pub fn approximate() -> Self {
        Self {
            r#type: "approximate".to_string(),
            city: None,
            region: None,
            country: None,
            timezone: None,
        }
    }

    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}

/// Web search configuration builder (2025-03-05 version)
#[derive(Debug, Clone, Default)]
pub struct WebSearch20250305Config {
    max_uses: Option<u32>,
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    user_location: Option<serde_json::Value>,
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

    /// Set the user location for search results (provider-specific).
    ///
    /// This should follow Anthropic's `user_location` shape.
    pub fn with_user_location(mut self, location: serde_json::Value) -> Self {
        self.user_location = Some(location);
        self
    }

    /// Set the user location for search results (typed helper, Vercel-aligned).
    pub fn with_user_location_typed(mut self, location: UserLocation) -> Self {
        if let Ok(v) = serde_json::to_value(location) {
            self.user_location = Some(v);
        }
        self
    }

    /// Build the Tool
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(max) = self.max_uses {
            // Vercel-aligned tool args (SDK shape)
            args["maxUses"] = serde_json::json!(max);
        }

        if let Some(allowed) = self.allowed_domains {
            args["allowedDomains"] = serde_json::json!(allowed);
        }

        if let Some(blocked) = self.blocked_domains {
            args["blockedDomains"] = serde_json::json!(blocked);
        }

        if let Some(loc) = self.user_location {
            args["userLocation"] = loc;
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("anthropic.web_search_20250305", "web_search").with_args(args),
        )
    }
}

/// Create a web search tool (2025-03-05 version)
pub fn web_search_20250305() -> WebSearch20250305Config {
    WebSearch20250305Config::new()
}

/// Web fetch citations configuration (Vercel-aligned).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebFetchCitations {
    pub enabled: bool,
}

impl WebFetchCitations {
    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

/// Web fetch configuration builder (2025-09-10 version; Vertex/Claude beta tool).
#[derive(Debug, Clone, Default)]
pub struct WebFetch20250910Config {
    max_uses: Option<u32>,
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    citations: Option<WebFetchCitations>,
    max_content_tokens: Option<u32>,
}

impl WebFetch20250910Config {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_uses(mut self, max: u32) -> Self {
        self.max_uses = Some(max);
        self
    }

    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self
    }

    pub fn with_blocked_domains(mut self, domains: Vec<String>) -> Self {
        self.blocked_domains = Some(domains);
        self
    }

    pub fn with_citations(mut self, citations: WebFetchCitations) -> Self {
        self.citations = Some(citations);
        self
    }

    pub fn with_max_content_tokens(mut self, max_tokens: u32) -> Self {
        self.max_content_tokens = Some(max_tokens);
        self
    }

    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(max) = self.max_uses {
            args["maxUses"] = serde_json::json!(max);
        }
        if let Some(allowed) = self.allowed_domains {
            args["allowedDomains"] = serde_json::json!(allowed);
        }
        if let Some(blocked) = self.blocked_domains {
            args["blockedDomains"] = serde_json::json!(blocked);
        }
        if let Some(citations) = self.citations {
            args["citations"] = serde_json::to_value(citations)
                .unwrap_or_else(|_| serde_json::json!({"enabled": true}));
        }
        if let Some(max_tokens) = self.max_content_tokens {
            args["maxContentTokens"] = serde_json::json!(max_tokens);
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("anthropic.web_fetch_20250910", "web_fetch").with_args(args),
        )
    }
}

/// Create a web fetch tool (2025-09-10 version).
pub fn web_fetch_20250910() -> WebFetch20250910Config {
    WebFetch20250910Config::new()
}

/// Create a tool search tool (regex variant, 2025-11-19 version).
///
/// This is a provider-hosted tool that allows Claude to discover available tools on demand.
/// Requires the `advanced-tool-use-2025-11-20` beta header.
pub fn tool_search_regex_20251119() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "anthropic.tool_search_regex_20251119",
        "tool_search",
    ))
}

/// Create a tool search tool (BM25 variant, 2025-11-19 version).
///
/// Requires the `advanced-tool-use-2025-11-20` beta header.
pub fn tool_search_bm25_20251119() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "anthropic.tool_search_bm25_20251119",
        "tool_search",
    ))
}

/// Create a code execution tool (2025-05-22 version).
///
/// Requires the `code-execution-2025-05-22` beta header.
pub fn code_execution_20250522() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "anthropic.code_execution_20250522",
        "code_execution",
    ))
}

/// Create a code execution tool (2025-08-25 version).
///
/// Requires the `code-execution-2025-08-25` beta header.
pub fn code_execution_20250825() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "anthropic.code_execution_20250825",
        "code_execution",
    ))
}

/// Create a memory tool (2025-08-18 version).
///
/// Requires the `context-management-2025-06-27` beta header.
pub fn memory_20250818() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "anthropic.memory_20250818",
        "memory",
    ))
}
