//! Perplexity provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["perplexity"]`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Perplexity search mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerplexitySearchMode {
    /// General web search.
    Web,
    /// Academic search.
    Academic,
    /// Security-focused search.
    Sec,
}

/// Perplexity search recency filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerplexitySearchRecencyFilter {
    /// Search within the last hour.
    Hour,
    /// Search within the last day.
    Day,
    /// Search within the last week.
    Week,
    /// Search within the last month.
    Month,
    /// Search within the last year.
    Year,
}

/// Search context size for hosted web search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerplexitySearchContextSize {
    /// Lower latency, less context.
    Low,
    /// Balanced context size.
    Medium,
    /// Maximum context size.
    High,
}

/// User location hint for hosted web search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerplexityUserLocation {
    /// Approximate latitude.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latitude: Option<f64>,
    /// Approximate longitude.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub longitude: Option<f64>,
    /// ISO country code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// IANA timezone identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl PerplexityUserLocation {
    /// Create a new user-location hint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the country code.
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the timezone identifier.
    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }

    /// Set latitude and longitude.
    pub fn with_coordinates(mut self, latitude: f64, longitude: f64) -> Self {
        self.latitude = Some(latitude);
        self.longitude = Some(longitude);
        self
    }
}

/// Nested hosted web-search options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerplexityWebSearchOptions {
    /// Search context size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<PerplexitySearchContextSize>,
    /// Optional user location hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<PerplexityUserLocation>,
}

impl PerplexityWebSearchOptions {
    /// Create new web-search options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the search context size.
    pub fn with_search_context_size(mut self, size: PerplexitySearchContextSize) -> Self {
        self.search_context_size = Some(size);
        self
    }

    /// Set the user location hint.
    pub fn with_user_location(mut self, location: PerplexityUserLocation) -> Self {
        self.user_location = Some(location);
        self
    }
}

/// Perplexity-specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerplexityOptions {
    /// Search mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_mode: Option<PerplexitySearchMode>,
    /// Search recency filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_recency_filter: Option<PerplexitySearchRecencyFilter>,
    /// Whether to return related questions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_related_questions: Option<bool>,
    /// Whether to return images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_images: Option<bool>,
    /// Whether to disable hosted search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_search: Option<bool>,
    /// Whether to enable the search classifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_search_classifier: Option<bool>,
    /// Allowed domains for hosted search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_domain_filter: Option<Vec<String>>,
    /// Allowed languages for hosted search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_language_filter: Option<Vec<String>>,
    /// Restrict search results after this date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_after_date_filter: Option<String>,
    /// Restrict search results before this date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_before_date_filter: Option<String>,
    /// Restrict results updated after this date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_updated_after_filter: Option<String>,
    /// Restrict results updated before this date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_updated_before_filter: Option<String>,
    /// Restrict image domains.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_domain_filter: Option<Vec<String>>,
    /// Restrict image formats.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_format_filter: Option<Vec<String>>,
    /// Nested web-search settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<PerplexityWebSearchOptions>,
    /// Additional Perplexity-specific parameters.
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl PerplexityOptions {
    /// Create new Perplexity options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the search mode.
    pub fn with_search_mode(mut self, mode: PerplexitySearchMode) -> Self {
        self.search_mode = Some(mode);
        self
    }

    /// Set the search recency filter.
    pub fn with_search_recency_filter(mut self, filter: PerplexitySearchRecencyFilter) -> Self {
        self.search_recency_filter = Some(filter);
        self
    }

    /// Control whether images are returned.
    pub fn with_return_images(mut self, enabled: bool) -> Self {
        self.return_images = Some(enabled);
        self
    }

    /// Control whether related questions are returned.
    pub fn with_return_related_questions(mut self, enabled: bool) -> Self {
        self.return_related_questions = Some(enabled);
        self
    }

    /// Control whether hosted search is disabled.
    pub fn with_disable_search(mut self, disabled: bool) -> Self {
        self.disable_search = Some(disabled);
        self
    }

    /// Control whether the search classifier is enabled.
    pub fn with_search_classifier(mut self, enabled: bool) -> Self {
        self.enable_search_classifier = Some(enabled);
        self
    }

    /// Replace the domain filter list.
    pub fn with_search_domain_filter<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let domains = domains.into_iter().map(Into::into).collect::<Vec<_>>();
        self.search_domain_filter = if domains.is_empty() {
            None
        } else {
            Some(domains)
        };
        self
    }

    /// Replace the language filter list.
    pub fn with_search_language_filter<I, S>(mut self, languages: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let languages = languages.into_iter().map(Into::into).collect::<Vec<_>>();
        self.search_language_filter = if languages.is_empty() {
            None
        } else {
            Some(languages)
        };
        self
    }

    /// Set nested hosted web-search options.
    pub fn with_web_search_options(mut self, options: PerplexityWebSearchOptions) -> Self {
        self.web_search_options = Some(options);
        self
    }

    /// Set the hosted search context size.
    pub fn with_search_context_size(mut self, size: PerplexitySearchContextSize) -> Self {
        let options = self
            .web_search_options
            .take()
            .unwrap_or_default()
            .with_search_context_size(size);
        self.web_search_options = Some(options);
        self
    }

    /// Set the hosted search user location.
    pub fn with_user_location(mut self, location: PerplexityUserLocation) -> Self {
        let options = self
            .web_search_options
            .take()
            .unwrap_or_default()
            .with_user_location(location);
        self.web_search_options = Some(options);
        self
    }

    /// Add a custom Perplexity parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perplexity_options_serialize_known_search_fields() {
        let value = serde_json::to_value(
            PerplexityOptions::new()
                .with_search_mode(PerplexitySearchMode::Academic)
                .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
                .with_return_images(true)
                .with_return_related_questions(true)
                .with_search_context_size(PerplexitySearchContextSize::High)
                .with_user_location(
                    PerplexityUserLocation::new()
                        .with_country("US")
                        .with_timezone("America/New_York"),
                )
                .with_param("someVendorParam", serde_json::json!(true)),
        )
        .expect("options serialize");

        assert_eq!(value["search_mode"], serde_json::json!("academic"));
        assert_eq!(value["search_recency_filter"], serde_json::json!("month"));
        assert_eq!(value["return_images"], serde_json::json!(true));
        assert_eq!(value["return_related_questions"], serde_json::json!(true));
        assert_eq!(
            value["web_search_options"]["search_context_size"],
            serde_json::json!("high")
        );
        assert_eq!(
            value["web_search_options"]["user_location"]["country"],
            serde_json::json!("US")
        );
        assert_eq!(value["someVendorParam"], serde_json::json!(true));
    }
}
