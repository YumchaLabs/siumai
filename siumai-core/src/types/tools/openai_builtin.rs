//! Deprecated OpenAI built-in tool types (currently disabled).

use serde::{Deserialize, Serialize};

/// OpenAI-specific built-in tools
///
/// # Deprecation Notice
///
/// This enum is deprecated in favor of the new `hosted_tools` module which provides
/// a more flexible and type-safe API for provider-defined tools.
///
/// ## Migration Guide
///
/// **Old way (deprecated)**:
/// ```rust,ignore
/// use siumai::types::OpenAiBuiltInTool;
///
/// let tool = OpenAiBuiltInTool::WebSearch;
/// ```
///
/// **New way (recommended)**:
/// ```rust
/// use siumai::hosted_tools::openai;
///
/// let tool = openai::web_search().build();
/// ```
///
/// ### Migration Examples
///
/// #### Web Search
/// ```rust,ignore
/// // Old
/// OpenAiBuiltInTool::WebSearchOptions {
///     options: WebSearchOptions {
///         search_context_size: Some("high".into()),
///         user_location: Some(WebSearchUserLocation { ... }),
///     }
/// }
///
/// // New
/// use siumai::hosted_tools::openai;
/// openai::web_search()
///     .with_search_context_size("high")
///     .with_user_location(openai::UserLocation::new("approximate").with_country("US"))
///     .build()
/// ```
///
/// #### File Search
/// ```rust,ignore
/// // Old
/// OpenAiBuiltInTool::FileSearchOptions {
///     vector_store_ids: Some(vec!["vs_123".into()]),
///     max_num_results: Some(10),
///     ranking_options: None,
///     filters: None,
/// }
///
/// // New
/// use siumai::hosted_tools::openai;
/// openai::file_search()
///     .with_vector_store_ids(vec!["vs_123".to_string()])
///     .with_max_num_results(10)
///     .build()
/// ```
///
/// #### Computer Use
/// ```rust,ignore
/// // Old
/// OpenAiBuiltInTool::ComputerUse {
///     display_width: 1920,
///     display_height: 1080,
///     environment: "headless".into(),
/// }
///
/// // New
/// use siumai::hosted_tools::openai;
/// openai::computer_use(1920, 1080, "headless")
/// ```
///
/// See the `hosted_tools` module documentation for more examples.
#[deprecated(
    since = "0.12.0",
    note = "Use `hosted_tools::openai` module instead. See migration guide in the documentation."
)]
#[cfg(any())]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpenAiBuiltInTool {
    /// Web search tool
    WebSearch,
    /// Web search tool with extra provider-specific options
    WebSearchAdvanced { extra: serde_json::Value },
    /// Web search tool with typed options (per Vercel AI examples)
    WebSearchOptions { options: WebSearchOptions },
    /// File search tool
    FileSearch {
        /// Vector store IDs to search
        vector_store_ids: Option<Vec<String>>,
    },
    /// File search tool with extra options
    FileSearchAdvanced {
        /// Vector store IDs to search
        vector_store_ids: Option<Vec<String>>,
        /// Extra provider-specific options
        extra: serde_json::Value,
    },
    /// File search tool with typed options (aligned with OpenAI Responses API)
    FileSearchOptions {
        /// Vector store IDs to search
        vector_store_ids: Option<Vec<String>>,
        /// Maximum results to return (provider may ignore)
        max_num_results: Option<u32>,
        /// Ranking options
        #[serde(skip_serializing_if = "Option::is_none")]
        ranking_options: Option<FileSearchRankingOptions>,
        /// Filters for search
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<FileSearchFilter>,
    },
    /// Computer use tool
    ComputerUse {
        /// Display width
        display_width: u32,
        /// Display height
        display_height: u32,
        /// Environment type
        environment: String,
    },
    /// Computer use tool with advanced options
    ComputerUseAdvanced {
        /// Display width
        display_width: u32,
        /// Display height
        display_height: u32,
        /// Environment type
        environment: String,
        /// Optional display scale
        #[serde(skip_serializing_if = "Option::is_none")]
        display_scale: Option<f32>,
        /// Extra provider-specific options
        extra: serde_json::Value,
    },
}

#[cfg(any())]
impl OpenAiBuiltInTool {
    /// Convert to JSON for API requests
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::WebSearch => serde_json::json!({
                "type": "web_search_preview"
            }),
            Self::WebSearchAdvanced { extra } => {
                let mut json = serde_json::json!({
                    "type": "web_search_preview"
                });
                // merge extra fields
                if let Some(map) = extra.as_object() {
                    for (k, v) in map {
                        json[k] = v.clone();
                    }
                }
                json
            }
            Self::WebSearchOptions { options } => {
                let mut json = serde_json::json!({ "type": "web_search_preview" });
                if let Some(s) = &options.search_context_size {
                    json["searchContextSize"] = serde_json::json!(s);
                }
                if let Some(loc) = &options.user_location {
                    let mut o = serde_json::json!({ "type": loc.r#type });
                    if let Some(country) = &loc.country {
                        o["country"] = serde_json::json!(country);
                    }
                    if let Some(city) = &loc.city {
                        o["city"] = serde_json::json!(city);
                    }
                    if let Some(region) = &loc.region {
                        o["region"] = serde_json::json!(region);
                    }
                    if let Some(tz) = &loc.timezone {
                        o["timezone"] = serde_json::json!(tz);
                    }
                    json["userLocation"] = o;
                }
                json
            }
            Self::FileSearch { vector_store_ids } => {
                let mut json = serde_json::json!({
                    "type": "file_search"
                });
                if let Some(ids) = vector_store_ids {
                    json["vector_store_ids"] = serde_json::Value::Array(
                        ids.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect(),
                    );
                }
                json
            }
            Self::FileSearchAdvanced {
                vector_store_ids,
                extra,
            } => {
                let mut json = serde_json::json!({
                    "type": "file_search"
                });
                if let Some(ids) = vector_store_ids {
                    json["vector_store_ids"] = serde_json::Value::Array(
                        ids.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect(),
                    );
                }
                if let Some(map) = extra.as_object() {
                    for (k, v) in map {
                        json[k] = v.clone();
                    }
                }
                json
            }
            Self::FileSearchOptions {
                vector_store_ids,
                max_num_results,
                ranking_options,
                filters,
            } => {
                let mut json = serde_json::json!({ "type": "file_search" });
                if let Some(ids) = vector_store_ids {
                    json["vector_store_ids"] = serde_json::Value::Array(
                        ids.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect(),
                    );
                }
                if let Some(k) = *max_num_results {
                    json["max_num_results"] = serde_json::json!(k);
                }
                if let Some(ro) = ranking_options {
                    let mut o = serde_json::json!({});
                    if let Some(r) = ro.ranker.as_ref() {
                        o["ranker"] = serde_json::json!(r);
                    }
                    if let Some(th) = ro.score_threshold {
                        o["score_threshold"] = serde_json::json!(th);
                    }
                    json["ranking_options"] = o;
                }
                if let Some(f) = filters {
                    json["filters"] = f.to_json();
                }
                json
            }
            Self::ComputerUse {
                display_width,
                display_height,
                environment,
            } => serde_json::json!({
                "type": "computer_use_preview",
                "display_width": display_width,
                "display_height": display_height,
                "environment": environment
            }),
            Self::ComputerUseAdvanced {
                display_width,
                display_height,
                environment,
                display_scale,
                extra,
            } => {
                let mut json = serde_json::json!({
                    "type": "computer_use_preview",
                    "display_width": display_width,
                    "display_height": display_height,
                    "environment": environment
                });
                if let Some(scale) = display_scale {
                    json["display_scale"] = serde_json::json!(scale);
                }
                if let Some(map) = extra.as_object() {
                    for (k, v) in map {
                        json[k] = v.clone();
                    }
                }
                json
            }
        }
    }
}

/// Typed options for OpenAI web search built-in tool (subset based on Vercel AI docs)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(any())]
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(any())]
pub struct WebSearchUserLocation {
    #[serde(rename = "type")]
    pub r#type: String, // e.g., "approximate"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(any())]
pub struct FileSearchRankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[cfg(any())]
pub enum FileSearchFilter {
    #[serde(rename = "eq")]
    Eq {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "ne")]
    Ne {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "gt")]
    Gt {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "gte")]
    Gte {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "lt")]
    Lt {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "lte")]
    Lte {
        key: String,
        value: serde_json::Value,
    },
    #[serde(rename = "and")]
    And { filters: Vec<FileSearchFilter> },
    #[serde(rename = "or")]
    Or { filters: Vec<FileSearchFilter> },
}

#[cfg(any())]
impl FileSearchFilter {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            FileSearchFilter::Eq { key, value } => serde_json::json!({
                "type": "eq", "key": key, "value": value
            }),
            FileSearchFilter::Ne { key, value } => serde_json::json!({
                "type": "ne", "key": key, "value": value
            }),
            FileSearchFilter::Gt { key, value } => serde_json::json!({
                "type": "gt", "key": key, "value": value
            }),
            FileSearchFilter::Gte { key, value } => serde_json::json!({
                "type": "gte", "key": key, "value": value
            }),
            FileSearchFilter::Lt { key, value } => serde_json::json!({
                "type": "lt", "key": key, "value": value
            }),
            FileSearchFilter::Lte { key, value } => serde_json::json!({
                "type": "lte", "key": key, "value": value
            }),
            FileSearchFilter::And { filters } => serde_json::json!({
                "type": "and",
                "filters": filters.iter().map(|f| f.to_json()).collect::<Vec<_>>()
            }),
            FileSearchFilter::Or { filters } => serde_json::json!({
                "type": "or",
                "filters": filters.iter().map(|f| f.to_json()).collect::<Vec<_>>()
            }),
        }
    }
}
