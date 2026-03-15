//! xAI (Grok) provider options.
//!
//! These typed option structs are owned by the xAI provider crate and are serialized into
//! `providerOptions["xai"]` (Vercel-aligned open options map).

use serde::{Deserialize, Serialize};

/// xAI (Grok) specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiOptions {
    /// Reasoning effort for Grok models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Web search parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
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

/// xAI text-to-speech specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiTtsOptions {
    /// Output sample rate in Hz.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u64>,
    /// Output bit rate in bps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bit_rate: Option<u64>,
}

impl XaiTtsOptions {
    /// Create new xAI TTS options.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sample_rate(mut self, sample_rate: u64) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    pub fn with_bit_rate(mut self, bit_rate: u64) -> Self {
        self.bit_rate = Some(bit_rate);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.sample_rate.is_none() && self.bit_rate.is_none()
    }
}

/// xAI web search parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiSearchParameters {
    /// Search mode.
    pub mode: SearchMode,
    /// Whether to return citations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_citations: Option<bool>,
    /// Maximum number of search results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_search_results: Option<u32>,
    /// Start date for search (YYYY-MM-DD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_date: Option<String>,
    /// End date for search (YYYY-MM-DD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_date: Option<String>,
    /// Search sources configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Allowed websites.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_websites: Option<Vec<String>>,
    /// Excluded websites.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excluded_websites: Option<Vec<String>>,
    /// Enable safe search.
    #[serde(skip_serializing_if = "Option::is_none")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xai_tts_options_serialize_only_present_fields() {
        let value = serde_json::to_value(
            XaiTtsOptions::new()
                .with_sample_rate(44_100)
                .with_bit_rate(192_000),
        )
        .expect("serialize xai tts options");

        assert_eq!(
            value,
            serde_json::json!({
                "sample_rate": 44_100,
                "bit_rate": 192_000
            })
        );
    }
}
