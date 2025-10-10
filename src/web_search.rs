//! Web Search Functionality
//!
//! This module provides unified web search capabilities across different AI providers.
//! Each provider implements web search differently:
//! - `OpenAI`: Built-in web search tools via Responses API
//! - Anthropic: `web_search` tool
//!
//! Note: OpenAI Responses API `web_search` is currently not implemented in this crate and will
//! return `UnsupportedOperation`. Track the implementation status in the changelog and README.
//! - xAI: Live Search with `search_parameters`
//! - Gemini: Search-augmented generation
//! - `OpenRouter`: `search_prompt` parameter

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::{WebSearchConfig, WebSearchContextSize, WebSearchResult, WebSearchStrategy};

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

        // Add provider-specific parameters
        for (key, value) in &self.config.provider_params {
            params.insert(key.clone(), value.clone());
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

        // Add provider-specific parameters
        for (key, value) in &self.config.provider_params {
            params.insert(key.clone(), value.clone());
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

        // Add provider-specific parameters
        for (key, value) in &self.config.provider_params {
            params.insert(key.clone(), value.clone());
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

        // Add provider-specific parameters
        for (key, value) in &self.config.provider_params {
            params.insert(key.clone(), value.clone());
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

        // Add provider-specific parameters
        for (key, value) in &self.config.provider_params {
            params.insert(key.clone(), value.clone());
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
    pub fn to_tool(&self) -> crate::types::Tool {
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

        crate::types::Tool::function(
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

    #[test]
    fn test_web_search_config() {
        let config = WebSearchConfig {
            enabled: true,
            max_results: Some(5),
            context_size: Some(WebSearchContextSize::Medium),
            search_prompt: Some("Custom search prompt".to_string()),
            strategy: WebSearchStrategy::Auto,
            provider_params: HashMap::new(),
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
        assert_eq!(tool_def.function.name, "web_search");
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
