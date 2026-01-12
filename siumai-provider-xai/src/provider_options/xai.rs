//! xAI (Grok) provider options.
//!
//! These typed option structs are owned by the xAI provider crate and are serialized into
//! `providerOptions["xai"]` (Vercel-aligned open options map).

use serde::{Deserialize, Serialize};

/// xAI (Grok) specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiOptions {
    /// Reasoning effort for Grok models.
    pub reasoning_effort: Option<String>,
    /// Web search parameters.
    pub search_parameters: Option<XaiSearchParameters>,
}

impl XaiOptions {
    /// Create new xAI options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable web search with configuration.
    pub fn with_search(mut self, params: XaiSearchParameters) -> Self {
        self.search_parameters = Some(params);
        self
    }

    /// Enable web search with default settings.
    pub fn with_default_search(mut self) -> Self {
        self.search_parameters = Some(XaiSearchParameters::default());
        self
    }

    /// Set reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }
}

/// xAI web search parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiSearchParameters {
    /// Search mode.
    pub mode: SearchMode,
    /// Whether to return citations.
    pub return_citations: Option<bool>,
    /// Maximum number of search results.
    pub max_search_results: Option<u32>,
    /// Start date for search (YYYY-MM-DD).
    pub from_date: Option<String>,
    /// End date for search (YYYY-MM-DD).
    pub to_date: Option<String>,
    /// Search sources configuration.
    pub sources: Option<Vec<SearchSource>>,
}

impl Default for XaiSearchParameters {
    fn default() -> Self {
        Self {
            mode: SearchMode::Auto,
            return_citations: Some(true),
            max_search_results: Some(5),
            from_date: None,
            to_date: None,
            sources: None,
        }
    }
}

/// Search mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Automatically decide whether to search.
    Auto,
    /// Always search.
    On,
    /// Never search.
    Off,
}

/// Search source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSource {
    /// Source type.
    #[serde(rename = "type")]
    pub source_type: SearchSourceType,
    /// Country code for localized search.
    pub country: Option<String>,
    /// Allowed websites.
    pub allowed_websites: Option<Vec<String>>,
    /// Excluded websites.
    pub excluded_websites: Option<Vec<String>>,
    /// Enable safe search.
    pub safe_search: Option<bool>,
}

/// Search source type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchSourceType {
    /// Web search.
    Web,
    /// News search.
    News,
    /// X (Twitter) search.
    X,
}
