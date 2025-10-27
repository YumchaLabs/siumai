//! OpenAI Provider-Defined Tools
//!
//! This module provides factory functions for creating OpenAI-specific provider-defined tools.
//! These tools are executed by OpenAI's servers and include web search, file search, and computer use.
//!
//! # Examples
//!
//! ```rust
//! use siumai::provider_tools::openai;
//!
//! // Create a web search tool with default settings
//! let web_search = openai::web_search();
//!
//! // Create a web search tool with custom configuration
//! let web_search_custom = openai::web_search()
//!     .with_search_context_size("high")
//!     .with_user_location(
//!         openai::UserLocation::new("approximate")
//!             .with_country("US")
//!             .with_city("San Francisco")
//!     );
//!
//! // Create a file search tool
//! let file_search = openai::file_search()
//!     .with_vector_store_ids(vec!["vs_123".to_string()])
//!     .with_max_num_results(10);
//!
//! // Create a computer use tool
//! let computer_use = openai::computer_use(1920, 1080, "headless");
//! ```

use crate::types::{ProviderDefinedTool, Tool};

/// Web search configuration builder
#[derive(Debug, Clone, Default)]
pub struct WebSearchConfig {
    search_context_size: Option<String>,
    user_location: Option<UserLocation>,
}

impl WebSearchConfig {
    /// Create a new web search configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the search context size (e.g., "low", "medium", "high")
    pub fn with_search_context_size(mut self, size: impl Into<String>) -> Self {
        self.search_context_size = Some(size.into());
        self
    }

    /// Set the user location for search results
    pub fn with_user_location(mut self, location: UserLocation) -> Self {
        self.user_location = Some(location);
        self
    }

    /// Build the Tool
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(size) = self.search_context_size {
            args["searchContextSize"] = serde_json::json!(size);
        }

        if let Some(loc) = self.user_location {
            let mut loc_json = serde_json::json!({
                "type": loc.r#type
            });
            if let Some(country) = loc.country {
                loc_json["country"] = serde_json::json!(country);
            }
            if let Some(city) = loc.city {
                loc_json["city"] = serde_json::json!(city);
            }
            if let Some(region) = loc.region {
                loc_json["region"] = serde_json::json!(region);
            }
            if let Some(timezone) = loc.timezone {
                loc_json["timezone"] = serde_json::json!(timezone);
            }
            args["userLocation"] = loc_json;
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("openai.web_search", "web_search").with_args(args),
        )
    }
}

/// User location for web search
#[derive(Debug, Clone)]
pub struct UserLocation {
    r#type: String,
    country: Option<String>,
    city: Option<String>,
    region: Option<String>,
    timezone: Option<String>,
}

impl UserLocation {
    /// Create a new user location with the specified type (e.g., "approximate")
    pub fn new(r#type: impl Into<String>) -> Self {
        Self {
            r#type: r#type.into(),
            country: None,
            city: None,
            region: None,
            timezone: None,
        }
    }

    /// Set the country
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the city
    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    /// Set the region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the timezone
    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}

/// Create a web search tool with default settings
///
/// # Example
///
/// ```rust
/// use siumai::provider_tools::openai;
///
/// let tool = openai::web_search();
/// ```
pub fn web_search() -> WebSearchConfig {
    WebSearchConfig::new()
}

/// File search configuration builder
#[derive(Debug, Clone, Default)]
pub struct FileSearchConfig {
    vector_store_ids: Option<Vec<String>>,
    max_num_results: Option<u32>,
    ranking_options: Option<RankingOptions>,
}

impl FileSearchConfig {
    /// Create a new file search configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vector store IDs to search
    pub fn with_vector_store_ids(mut self, ids: Vec<String>) -> Self {
        self.vector_store_ids = Some(ids);
        self
    }

    /// Set the maximum number of results to return
    pub fn with_max_num_results(mut self, max: u32) -> Self {
        self.max_num_results = Some(max);
        self
    }

    /// Set the ranking options
    pub fn with_ranking_options(mut self, options: RankingOptions) -> Self {
        self.ranking_options = Some(options);
        self
    }

    /// Build the Tool
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(ids) = self.vector_store_ids {
            args["vector_store_ids"] = serde_json::json!(ids);
        }

        if let Some(max) = self.max_num_results {
            args["max_num_results"] = serde_json::json!(max);
        }

        if let Some(ranking) = self.ranking_options {
            let mut ranking_json = serde_json::json!({});
            if let Some(ranker) = ranking.ranker {
                ranking_json["ranker"] = serde_json::json!(ranker);
            }
            if let Some(threshold) = ranking.score_threshold {
                ranking_json["score_threshold"] = serde_json::json!(threshold);
            }
            args["ranking_options"] = ranking_json;
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("openai.file_search", "file_search").with_args(args),
        )
    }
}

/// Ranking options for file search
#[derive(Debug, Clone, Default)]
pub struct RankingOptions {
    ranker: Option<String>,
    score_threshold: Option<f64>,
}

impl RankingOptions {
    /// Create new ranking options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ranker type (e.g., "auto", "default_2024_08_21")
    pub fn with_ranker(mut self, ranker: impl Into<String>) -> Self {
        self.ranker = Some(ranker.into());
        self
    }

    /// Set the score threshold
    pub fn with_score_threshold(mut self, threshold: f64) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
}

/// Create a file search tool with default settings
///
/// # Example
///
/// ```rust
/// use siumai::provider_tools::openai;
///
/// let tool = openai::file_search()
///     .with_vector_store_ids(vec!["vs_123".to_string()]);
/// ```
pub fn file_search() -> FileSearchConfig {
    FileSearchConfig::new()
}

/// Create a computer use tool
///
/// # Arguments
///
/// * `display_width` - Display width in pixels
/// * `display_height` - Display height in pixels
/// * `environment` - Environment type (e.g., "headless", "desktop")
///
/// # Example
///
/// ```rust
/// use siumai::provider_tools::openai;
///
/// let tool = openai::computer_use(1920, 1080, "headless");
/// ```
pub fn computer_use(
    display_width: u32,
    display_height: u32,
    environment: impl Into<String>,
) -> Tool {
    let args = serde_json::json!({
        "display_width": display_width,
        "display_height": display_height,
        "environment": environment.into()
    });

    Tool::ProviderDefined(
        ProviderDefinedTool::new("openai.computer_use", "computer_use").with_args(args),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_default() {
        let tool = web_search().build();
        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "openai.web_search");
                assert_eq!(pt.name, "web_search");
                assert_eq!(pt.provider(), Some("openai"));
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }

    #[test]
    fn test_web_search_with_config() {
        let tool = web_search()
            .with_search_context_size("high")
            .with_user_location(
                UserLocation::new("approximate")
                    .with_country("US")
                    .with_city("SF"),
            )
            .build();

        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "openai.web_search");
                assert_eq!(
                    pt.args.get("searchContextSize").and_then(|v| v.as_str()),
                    Some("high")
                );
                let loc = pt.args.get("userLocation").unwrap();
                assert_eq!(loc.get("country").and_then(|v| v.as_str()), Some("US"));
                assert_eq!(loc.get("city").and_then(|v| v.as_str()), Some("SF"));
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }

    #[test]
    fn test_file_search() {
        let tool = file_search()
            .with_vector_store_ids(vec!["vs_123".to_string()])
            .with_max_num_results(10)
            .build();

        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "openai.file_search");
                assert_eq!(pt.name, "file_search");
                let ids = pt.args.get("vector_store_ids").unwrap().as_array().unwrap();
                assert_eq!(ids.len(), 1);
                assert_eq!(
                    pt.args.get("max_num_results").and_then(|v| v.as_u64()),
                    Some(10)
                );
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }

    #[test]
    fn test_computer_use() {
        let tool = computer_use(1920, 1080, "headless");

        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "openai.computer_use");
                assert_eq!(pt.name, "computer_use");
                assert_eq!(
                    pt.args.get("display_width").and_then(|v| v.as_u64()),
                    Some(1920)
                );
                assert_eq!(
                    pt.args.get("display_height").and_then(|v| v.as_u64()),
                    Some(1080)
                );
                assert_eq!(
                    pt.args.get("environment").and_then(|v| v.as_str()),
                    Some("headless")
                );
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }
}
