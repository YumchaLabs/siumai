//! Web search options for OpenAI
//!
//! Configuration for web search functionality in OpenAI models.
//!
//! Learn more: https://platform.openai.com/docs/guides/tools-web-search

use serde::{Deserialize, Serialize};

/// Web search options for OpenAI chat completions
///
/// Configuration for the web search tool, including user location and context size.
/// This is used with the `web_search_options` parameter in chat completions.
///
/// # Example
///
/// ```rust
/// use siumai::types::provider_options::openai::{OpenAiWebSearchOptions, WebSearchLocation};
///
/// let options = OpenAiWebSearchOptions::new()
///     .with_context_size("high")
///     .with_location(WebSearchLocation::new()
///         .with_country("US")
///         .with_city("San Francisco")
///         .with_timezone("America/Los_Angeles")
///     );
/// ```
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
///
/// Wraps the approximate location with a type field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserLocationWrapper {
    /// The type of location (always "approximate")
    #[serde(rename = "type")]
    pub location_type: String,

    /// Approximate location parameters
    pub approximate: WebSearchLocation,
}

/// Web search location
///
/// Approximate location parameters for the search.
///
/// # Example
///
/// ```rust
/// use siumai::types::provider_options::openai::WebSearchLocation;
///
/// let location = WebSearchLocation::new()
///     .with_country("US")
///     .with_region("California")
///     .with_city("San Francisco")
///     .with_timezone("America/Los_Angeles");
/// ```
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
    ///
    /// Valid values: "low", "medium", "high"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_location() {
        let location = WebSearchLocation::new()
            .with_country("US")
            .with_region("California")
            .with_city("San Francisco")
            .with_timezone("America/Los_Angeles");

        assert_eq!(location.country, Some("US".to_string()));
        assert_eq!(location.region, Some("California".to_string()));
        assert_eq!(location.city, Some("San Francisco".to_string()));
        assert_eq!(location.timezone, Some("America/Los_Angeles".to_string()));
    }

    #[test]
    fn test_web_search_options() {
        let options = OpenAiWebSearchOptions::new()
            .with_context_size("high")
            .with_location(
                WebSearchLocation::new()
                    .with_country("US")
                    .with_city("San Francisco"),
            );

        assert_eq!(options.search_context_size, Some("high".to_string()));
        assert!(options.user_location.is_some());

        let location_wrapper = options.user_location.unwrap();
        assert_eq!(location_wrapper.location_type, "approximate");
        assert_eq!(location_wrapper.approximate.country, Some("US".to_string()));
        assert_eq!(
            location_wrapper.approximate.city,
            Some("San Francisco".to_string())
        );
    }

    #[test]
    fn test_web_search_options_serialization() {
        let options = OpenAiWebSearchOptions::new()
            .with_context_size("medium")
            .with_location(WebSearchLocation::new().with_country("US"));

        let json = serde_json::to_value(&options).unwrap();

        assert_eq!(json["search_context_size"], "medium");
        assert_eq!(json["user_location"]["type"], "approximate");
        assert_eq!(json["user_location"]["approximate"]["country"], "US");
    }
}
