//! Web search related types and configurations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Web search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchConfig {
    /// Whether web search is enabled
    pub enabled: bool,
    /// Maximum number of search results to retrieve
    pub max_results: Option<u32>,
    /// Search context size for providers that support it
    pub context_size: Option<WebSearchContextSize>,
    /// Custom search prompt for result integration
    pub search_prompt: Option<String>,
    /// Web search implementation strategy
    pub strategy: WebSearchStrategy,
    /// Provider-specific search parameters
    pub provider_params: HashMap<String, serde_json::Value>,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_results: Some(5),
            context_size: None,
            search_prompt: None,
            strategy: WebSearchStrategy::Auto,
            provider_params: HashMap::new(),
        }
    }
}

/// Web search context size for providers that support it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSearchContextSize {
    Small,
    Medium,
    Large,
}

/// Web search implementation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSearchStrategy {
    /// Automatically choose the best strategy for the provider
    Auto,
    /// Use provider's built-in search tools (`OpenAI` Responses API, xAI Live Search)
    BuiltIn,
    /// Use provider's web search tool (Anthropic `web_search` tool)
    Tool,
    /// Use external search API and inject results into context
    External,
}

/// Web search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchResult {
    /// Search result title
    pub title: String,
    /// Search result URL
    pub url: String,
    /// Search result snippet/description
    pub snippet: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}
