//! Web search options for OpenAI
//!
//! Learn more: https://platform.openai.com/docs/guides/tools-web-search

use serde::{Deserialize, Serialize};

/// Web search options for OpenAI chat completions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAiWebSearchOptions {
    /// User location for the search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocationWrapper>,

    /// High level guidance for the amount of context window space to use
    /// Valid values: "low", "medium", "high" (default: "medium")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
}

/// User location wrapper
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserLocationWrapper {
    /// The type of location (always "approximate")
    #[serde(rename = "type")]
    pub location_type: String,

    /// Approximate location parameters
    pub approximate: WebSearchLocation,
}

/// Web search location
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchLocation {
    /// Two-letter ISO country code (e.g., "US")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,

    /// Region of the user (e.g., "California")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,

    /// City of the user (e.g., "San Francisco")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,

    /// IANA timezone (e.g., "America/Los_Angeles")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl OpenAiWebSearchOptions {
    /// Create new web search options with default values
    pub fn new() -> Self {
        Self {
            user_location: None,
            search_context_size: None,
        }
    }

    /// Set the search context size
    pub fn with_context_size<S: Into<String>>(mut self, size: S) -> Self {
        self.search_context_size = Some(size.into());
        self
    }

    /// Set the user location
    pub fn with_location(mut self, location: WebSearchLocation) -> Self {
        self.user_location = Some(UserLocationWrapper {
            location_type: "approximate".to_string(),
            approximate: location,
        });
        self
    }
}

impl Default for OpenAiWebSearchOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSearchLocation {
    /// Create a new empty location
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the country (two-letter ISO code)
    pub fn with_country<S: Into<String>>(mut self, country: S) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the region
    pub fn with_region<S: Into<String>>(mut self, region: S) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the city
    pub fn with_city<S: Into<String>>(mut self, city: S) -> Self {
        self.city = Some(city.into());
        self
    }

    /// Set the timezone (IANA timezone)
    pub fn with_timezone<S: Into<String>>(mut self, timezone: S) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}
