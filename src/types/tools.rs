//! Tool calling and function definition types

use serde::{Deserialize, Serialize};

/// Tool calling types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<FunctionCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (usually "function")
    pub r#type: String,
    /// Function definition
    pub function: ToolFunction,
}

impl Tool {
    /// Create a new function tool
    pub fn function(name: String, description: String, parameters: serde_json::Value) -> Self {
        Self {
            r#type: "function".to_string(),
            function: ToolFunction {
                name,
                description,
                parameters,
            },
        }
    }
}

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// JSON schema for function parameters
    pub parameters: serde_json::Value,
}

/// Tool type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "file_search")]
    FileSearch,
    #[serde(rename = "web_search")]
    WebSearch,
}

/// OpenAI-specific built-in tools
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
                    if let Some(r) = ro.ranker.as_ref() { o["ranker"] = serde_json::json!(r); }
                    if let Some(th) = ro.score_threshold { o["score_threshold"] = serde_json::json!(th); }
                    json["ranking_options"] = o;
                }
                if let Some(f) = filters { json["filters"] = f.to_json(); }
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
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct FileSearchRankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FileSearchFilter {
    #[serde(rename = "eq")]
    Eq { key: String, value: serde_json::Value },
    #[serde(rename = "ne")]
    Ne { key: String, value: serde_json::Value },
    #[serde(rename = "gt")]
    Gt { key: String, value: serde_json::Value },
    #[serde(rename = "gte")]
    Gte { key: String, value: serde_json::Value },
    #[serde(rename = "lt")]
    Lt { key: String, value: serde_json::Value },
    #[serde(rename = "lte")]
    Lte { key: String, value: serde_json::Value },
    #[serde(rename = "and")]
    And { filters: Vec<FileSearchFilter> },
    #[serde(rename = "or")]
    Or { filters: Vec<FileSearchFilter> },
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_search_advanced_merges_extra() {
        let t = OpenAiBuiltInTool::WebSearchAdvanced {
            extra: serde_json::json!({"region":"us","safe":"moderate"}),
        };
        let v = t.to_json();
        assert_eq!(
            v.get("type").and_then(|s| s.as_str()),
            Some("web_search_preview")
        );
        assert_eq!(v.get("region").and_then(|s| s.as_str()), Some("us"));
        assert_eq!(v.get("safe").and_then(|s| s.as_str()), Some("moderate"));
    }

    #[test]
    fn file_search_advanced_includes_ids_and_extra() {
        let t = OpenAiBuiltInTool::FileSearchAdvanced {
            vector_store_ids: Some(vec!["vs1".into(), "vs2".into()]),
            extra: serde_json::json!({"ranker":"semantic"}),
        };
        let v = t.to_json();
        assert_eq!(v.get("type").and_then(|s| s.as_str()), Some("file_search"));
        let ids = v
            .get("vector_store_ids")
            .and_then(|a| a.as_array())
            .cloned()
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(v.get("ranker").and_then(|s| s.as_str()), Some("semantic"));
    }

    #[test]
    fn computer_use_advanced_includes_scale_and_extra() {
        let t = OpenAiBuiltInTool::ComputerUseAdvanced {
            display_width: 1280,
            display_height: 720,
            environment: "headless".into(),
            display_scale: Some(1.5),
            extra: serde_json::json!({"cursor":"enabled"}),
        };
        let v = t.to_json();
        assert_eq!(
            v.get("type").and_then(|s| s.as_str()),
            Some("computer_use_preview")
        );
        assert_eq!(v.get("display_width").and_then(|x| x.as_u64()), Some(1280));
        assert_eq!(v.get("display_height").and_then(|x| x.as_u64()), Some(720));
        assert_eq!(
            v.get("environment").and_then(|s| s.as_str()),
            Some("headless")
        );
        assert_eq!(v.get("display_scale").and_then(|x| x.as_f64()), Some(1.5));
        assert_eq!(v.get("cursor").and_then(|s| s.as_str()), Some("enabled"));
    }

    #[test]
    fn web_search_options_includes_country_timezone() {
        let t = OpenAiBuiltInTool::WebSearchOptions {
            options: WebSearchOptions {
                search_context_size: Some("high".into()),
                user_location: Some(WebSearchUserLocation {
                    r#type: "approximate".into(),
                    country: Some("US".into()),
                    city: Some("SF".into()),
                    region: Some("CA".into()),
                    timezone: Some("America/Los_Angeles".into()),
                }),
            },
        };
        let v = t.to_json();
        let loc = v.get("userLocation").and_then(|o| o.as_object()).unwrap();
        assert_eq!(loc.get("country").and_then(|x| x.as_str()), Some("US"));
        assert_eq!(loc.get("timezone").and_then(|x| x.as_str()), Some("America/Los_Angeles"));
    }

    #[test]
    fn file_search_options_includes_max_num_results_ranking_filters() {
        let t = OpenAiBuiltInTool::FileSearchOptions {
            vector_store_ids: Some(vec!["vs1".into()]),
            max_num_results: Some(15),
            ranking_options: Some(FileSearchRankingOptions { ranker: Some("auto".into()), score_threshold: Some(0.5) }),
            filters: Some(FileSearchFilter::And { filters: vec![
                FileSearchFilter::Eq { key: "doc_type".into(), value: serde_json::json!("pdf") },
                FileSearchFilter::Gt { key: "score".into(), value: serde_json::json!(0.1) },
            ]}),
        };
        let v = t.to_json();
        assert_eq!(v.get("type").and_then(|s| s.as_str()), Some("file_search"));
        assert_eq!(v.get("max_num_results").and_then(|x| x.as_u64()), Some(15));
        let ro = v.get("ranking_options").and_then(|o| o.as_object()).unwrap();
        assert_eq!(ro.get("ranker").and_then(|x| x.as_str()), Some("auto"));
        assert_eq!(ro.get("score_threshold").and_then(|x| x.as_f64()), Some(0.5));
        assert!(v.get("filters").is_some());
    }
}
