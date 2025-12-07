//! Web Search Functionality (extras crate)
//!
//! This module provides unified web search capabilities across different AI providers.
//! It is a direct extraction from the core `siumai` crate, adapted to depend on the
//! public `siumai` API instead of internal modules.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use siumai::error::LlmError;
use siumai::types::{WebSearchConfig, WebSearchContextSize, WebSearchResult, WebSearchStrategy};

/// Web search capability trait
#[async_trait]
pub trait WebSearchCapability {
    /// Perform a web search
    async fn web_search(
        &self,
        query: String,
        config: Option<WebSearchConfig>,
    ) -> Result<Vec<WebSearchResult>, LlmError>;

    /// Check if web search is supported
    fn supports_web_search(&self) -> bool;

    /// Get the web search strategy used by this provider
    fn web_search_strategy(&self) -> WebSearchStrategy;
}

/// Web search implementation for different providers
pub struct WebSearchProvider {
    /// Provider name
    pub provider: String,
    /// Search configuration
    pub config: WebSearchConfig,
}

impl WebSearchProvider {
    /// Create a new web search provider
    pub const fn new(provider: String, config: WebSearchConfig) -> Self {
        Self { provider, config }
    }

    /// Build search parameters for `OpenAI`
    pub fn build_openai_params(&self, _query: &str) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        if let Some(max_results) = self.config.max_results {
            params.insert(
                "max_results".to_string(),
                serde_json::Value::Number(max_results.into()),
            );
        }

        if let Some(context_size) = &self.config.context_size {
            let size_str = match context_size {
                WebSearchContextSize::Small => "small",
                WebSearchContextSize::Medium => "medium",
                WebSearchContextSize::Large => "large",
            };
            params.insert(
                "search_context_size".to_string(),
                serde_json::Value::String(size_str.to_string()),
            );
        }

        // Merge typed ProviderOptions if present (OpenAI)
        if let siumai::types::ProviderOptions::OpenAi(ref boxed) = self.config.provider_options
            && let Some(ws) = &boxed.web_search_options
        {
            // Map fields to equivalent key/value pairs
            if let Some(sz) = &ws.search_context_size {
                params.insert(
                    "search_context_size".into(),
                    serde_json::Value::String(sz.clone()),
                );
            }
            if let Some(loc) = &ws.user_location
                && let Ok(v) = serde_json::to_value(loc)
            {
                params.insert("user_location".into(), v);
            }
        }

        params
    }

    /// Build search parameters for xAI
    pub fn build_xai_params(&self, query: &str) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        params.insert(
            "query".to_string(),
            serde_json::Value::String(query.to_string()),
        );

        if let Some(max_results) = self.config.max_results {
            params.insert(
                "max_results".to_string(),
                serde_json::Value::Number(max_results.into()),
            );
        }

        // Merge typed ProviderOptions if present (xAI)
        if let siumai::types::ProviderOptions::Xai(ref xai) = self.config.provider_options
            && let Some(sp) = &xai.search_parameters
            && let Ok(v) = serde_json::to_value(sp)
        {
            // xAI expects a structured object under search_parameters; when building
            // key-value params, flatten into a single key for downstream code to serialize.
            params.insert("search_parameters".into(), v);
        }

        params
    }

    /// Build search parameters for Anthropic
    pub fn build_anthropic_params(&self, query: &str) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        params.insert(
            "query".to_string(),
            serde_json::Value::String(query.to_string()),
        );

        if let Some(max_results) = self.config.max_results {
            params.insert(
                "max_results".to_string(),
                serde_json::Value::Number(max_results.into()),
            );
        }

        params
    }

    /// Build search parameters for Gemini
    pub fn build_gemini_params(&self, query: &str) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        params.insert(
            "query".to_string(),
            serde_json::Value::String(query.to_string()),
        );

        if let Some(max_results) = self.config.max_results {
            params.insert(
                "max_results".to_string(),
                serde_json::Value::Number(max_results.into()),
            );
        }

        params
    }

    /// Build search parameters for `OpenRouter`
    pub fn build_openrouter_params(&self, query: &str) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        if let Some(search_prompt) = &self.config.search_prompt {
            params.insert(
                "search_prompt".to_string(),
                serde_json::Value::String(search_prompt.clone()),
            );
        } else {
            // Default search prompt
            params.insert(
                "search_prompt".to_string(),
                serde_json::Value::String(format!("Search for information about: {query}")),
            );
        }

        params
    }
}

/// Web search tool for Anthropic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicWebSearchTool {
    /// Search query
    pub query: String,
    /// Maximum number of results
    pub max_results: Option<u32>,
}

impl AnthropicWebSearchTool {
    /// Create a new web search tool
    pub const fn new(query: String) -> Self {
        Self {
            query,
            max_results: None,
        }
    }

    /// Set maximum results
    pub const fn with_max_results(mut self, max_results: u32) -> Self {
        self.max_results = Some(max_results);
        self
    }

    /// Convert to tool definition
    pub fn to_tool(&self) -> siumai::types::Tool {
        let mut parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        });

        if self.max_results.is_some() {
            parameters["properties"]["max_results"] = serde_json::json!({
                "type": "integer",
                "description": "Maximum number of search results to return"
            });
        }

        siumai::types::Tool::function(
            "web_search".to_string(),
            "Search the web for current information".to_string(),
            parameters,
        )
    }
}

/// xAI Live Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiLiveSearchConfig {
    /// Whether to enable live search
    pub enabled: bool,
    /// Search parameters
    pub search_parameters: HashMap<String, serde_json::Value>,
}

impl Default for XaiLiveSearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            search_parameters: HashMap::new(),
        }
    }
}

impl XaiLiveSearchConfig {
    /// Create a new live search config
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable live search
    pub const fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable live search
    pub const fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Add search parameter
    pub fn with_parameter<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.search_parameters.insert(
            key.to_string(),
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        );
        self
    }

    /// Set maximum results
    pub fn with_max_results(self, max_results: u32) -> Self {
        self.with_parameter("max_results", max_results)
    }

    /// Set search timeout
    pub fn with_timeout(self, timeout_seconds: u32) -> Self {
        self.with_parameter("timeout", timeout_seconds)
    }
}

/// Gemini search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiSearchConfig {
    /// Whether to enable search-augmented generation
    pub enabled: bool,
    /// Search parameters
    pub search_parameters: HashMap<String, serde_json::Value>,
}

impl Default for GeminiSearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            search_parameters: HashMap::new(),
        }
    }
}

impl GeminiSearchConfig {
    /// Create a new search config
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable search
    pub const fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable search
    pub const fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Add search parameter
    pub fn with_parameter<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.search_parameters.insert(
            key.to_string(),
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai::types::Tool;

    #[test]
    fn test_web_search_config() {
        let config = WebSearchConfig {
            enabled: true,
            max_results: Some(5),
            context_size: Some(WebSearchContextSize::Medium),
            search_prompt: Some("Custom search prompt".to_string()),
            strategy: WebSearchStrategy::Auto,
            provider_options: siumai::types::ProviderOptions::None,
        };

        assert!(config.enabled);
        assert_eq!(config.max_results, Some(5));
    }

    #[test]
    fn test_anthropic_web_search_tool() {
        let tool = AnthropicWebSearchTool::new("test query".to_string()).with_max_results(10);

        assert_eq!(tool.query, "test query");
        assert_eq!(tool.max_results, Some(10));

        let tool_def = tool.to_tool();
        match tool_def {
            Tool::Function { function } => {
                assert_eq!(function.name, "web_search");
            }
            _ => panic!("Expected Function variant"),
        }
    }

    #[test]
    fn test_xai_live_search_config() {
        let config = XaiLiveSearchConfig::new()
            .enable()
            .with_max_results(5)
            .with_timeout(30);

        assert!(config.enabled);
        assert_eq!(
            config.search_parameters.get("max_results"),
            Some(&serde_json::Value::Number(5.into()))
        );
        assert_eq!(
            config.search_parameters.get("timeout"),
            Some(&serde_json::Value::Number(30.into()))
        );
    }
}
