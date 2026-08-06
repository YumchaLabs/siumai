//! xAI (Grok) provider options.
//!
//! These typed option structs are owned by the xAI provider crate and are serialized into the
//! `xai` provider-options namespace. The provider codec normalizes their ergonomic Rust-facing
//! names to the current xAI wire contract.

use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};
use siumai_protocol_openai::chat_completions::API_MODE_ID as CHAT_API_MODE_ID;
use siumai_protocol_openai::responses_next::API_MODE_ID as RESPONSES_API_MODE_ID;

use crate::tools::XaiResponsesTool;

macro_rules! xai_string_enum {
    ($name:ident { $($variant:ident => $wire:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant,)+
            /// Forward-compatible escape hatch for newly introduced upstream string values.
            Other(String),
        }

        impl $name {
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $wire,)+
                    Self::Other(value) => value.as_str(),
                }
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                match value {
                    $($wire => Self::$variant,)+
                    other => Self::Other(other.to_string()),
                }
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                match value.as_str() {
                    $($wire => Self::$variant,)+
                    _ => Self::Other(value),
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Ok(Self::from(value))
            }
        }
    };
}

xai_string_enum!(XaiChatReasoningEffort {
    None => "none",
    Low => "low",
    Medium => "medium",
    High => "high",
});

xai_string_enum!(XaiResponsesReasoningEffort {
    None => "none",
    Low => "low",
    Medium => "medium",
    High => "high",
});

xai_string_enum!(XaiReasoningSummary {
    Auto => "auto",
    Concise => "concise",
    Detailed => "detailed",
});

xai_string_enum!(XaiResponseInclude {
    FileSearchCallResults => "file_search_call.results",
    ReasoningEncryptedContent => "reasoning.encrypted_content",
});

/// xAI chat-completions specific options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiChatOptions {
    /// Reasoning effort for Grok chat models.
    #[serde(
        rename = "reasoningEffort",
        alias = "reasoning_effort",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_effort: Option<XaiChatReasoningEffort>,
    /// Whether to return token logprobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top logprobs to return.
    #[serde(
        rename = "topLogprobs",
        alias = "top_logprobs",
        skip_serializing_if = "Option::is_none"
    )]
    pub top_logprobs: Option<u32>,
    /// Whether to allow parallel tool calls.
    #[serde(
        rename = "parallelToolCalls",
        alias = "parallel_tool_calls",
        skip_serializing_if = "Option::is_none"
    )]
    pub parallel_tool_calls: Option<bool>,
    /// Legacy Chat Completions live-search parameters.
    ///
    /// New integrations should use provider-hosted `web_search` or `x_search` tools through the
    /// Responses API. This field remains only for explicit compatibility with deployments that
    /// still accept the retired Chat search contract.
    #[serde(
        rename = "searchParameters",
        alias = "search_parameters",
        skip_serializing_if = "Option::is_none"
    )]
    pub search_parameters: Option<XaiSearchParameters>,
    /// Stable application-supplied key used to improve prompt-cache affinity.
    #[serde(
        rename = "promptCacheKey",
        alias = "prompt_cache_key",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_cache_key: Option<String>,
}

impl TypedProviderOptions for XaiChatOptions {
    const NAMESPACE: &'static str = "xai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        validate_logprobs(self.logprobs, self.top_logprobs)?;
        if let Some(search) = &self.search_parameters {
            search.validate("search_parameters")?;
        }
        validate_non_empty_id("prompt_cache_key", self.prompt_cache_key.as_deref())?;
        Ok(())
    }
}

impl XaiChatOptions {
    /// Create new xAI chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the legacy Chat Completions live-search contract.
    ///
    /// Prefer [`XaiResponsesOptions::with_native_tool`] for new integrations.
    pub fn with_search(mut self, params: XaiSearchParameters) -> Self {
        self.search_parameters = Some(params);
        self
    }

    /// Enable legacy Chat Completions live search with provider defaults.
    ///
    /// Prefer [`XaiResponsesOptions::with_native_tool`] for new integrations.
    pub fn with_default_search(mut self) -> Self {
        self.search_parameters = Some(XaiSearchParameters::default());
        self
    }

    /// Set reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: impl Into<XaiChatReasoningEffort>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Enable or disable logprobs in the response.
    pub fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    /// Request the number of top logprobs for each output token.
    pub fn with_top_logprobs(mut self, count: u32) -> Self {
        self.top_logprobs = Some(count);
        self.logprobs = Some(true);
        self
    }

    /// Enable or disable parallel tool calls.
    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Attach a stable xAI prompt-cache key to this Chat Completions call.
    pub fn with_prompt_cache_key(mut self, prompt_cache_key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(prompt_cache_key.into());
        self
    }
}

/// xAI Responses-specific provider options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiResponsesOptions {
    /// Reasoning effort for Grok Responses models.
    #[serde(
        rename = "reasoningEffort",
        alias = "reasoning_effort",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_effort: Option<XaiResponsesReasoningEffort>,
    /// Reasoning summary verbosity for Responses-style APIs.
    #[serde(
        rename = "reasoningSummary",
        alias = "reasoning_summary",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_summary: Option<XaiReasoningSummary>,
    /// Whether to return token logprobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top logprobs to return.
    #[serde(
        rename = "topLogprobs",
        alias = "top_logprobs",
        skip_serializing_if = "Option::is_none"
    )]
    pub top_logprobs: Option<u32>,
    /// Whether to allow parallel tool calls.
    #[serde(
        rename = "parallelToolCalls",
        alias = "parallel_tool_calls",
        skip_serializing_if = "Option::is_none"
    )]
    pub parallel_tool_calls: Option<bool>,
    /// Whether to store the response for later retrieval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Previous response id for continuing a response chain.
    #[serde(
        rename = "previousResponseId",
        alias = "previous_response_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub previous_response_id: Option<String>,
    /// Stable application-supplied cache key for Responses prompt-cache affinity.
    #[serde(
        rename = "promptCacheKey",
        alias = "prompt_cache_key",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_cache_key: Option<String>,
    /// Additional response payload sections to include.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<XaiResponseInclude>>,
    /// Provider-hosted tools executed by xAI rather than the local runtime.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_tools: Vec<XaiResponsesTool>,
}

impl XaiResponsesOptions {
    /// Create new xAI Responses options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: impl Into<XaiResponsesReasoningEffort>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Set reasoning summary verbosity.
    pub fn with_reasoning_summary(mut self, summary: impl Into<XaiReasoningSummary>) -> Self {
        self.reasoning_summary = Some(summary.into());
        self
    }

    /// Enable or disable logprobs in the response.
    pub fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    /// Request the number of top logprobs for each output token.
    pub fn with_top_logprobs(mut self, count: u32) -> Self {
        self.top_logprobs = Some(count);
        self.logprobs = Some(true);
        self
    }

    /// Enable or disable parallel tool calls.
    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Control response storage.
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Continue from a previous response id.
    pub fn with_previous_response(mut self, response_id: impl Into<String>) -> Self {
        self.previous_response_id = Some(response_id.into());
        self
    }

    /// Attach a stable xAI prompt-cache key to this Responses call.
    pub fn with_prompt_cache_key(mut self, prompt_cache_key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(prompt_cache_key.into());
        self
    }

    /// Request additional response sections.
    pub fn with_include<I, T>(mut self, include: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<XaiResponseInclude>,
    {
        self.include = Some(include.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_native_tool(mut self, tool: XaiResponsesTool) -> Self {
        self.native_tools.push(tool);
        self
    }

    pub fn with_native_tools<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = XaiResponsesTool>,
    {
        self.native_tools.extend(tools);
        self
    }
}

impl TypedProviderOptions for XaiResponsesOptions {
    const NAMESPACE: &'static str = "xai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(RESPONSES_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        validate_logprobs(self.logprobs, self.top_logprobs)?;
        validate_non_empty_id("previous_response_id", self.previous_response_id.as_deref())?;
        validate_non_empty_id("prompt_cache_key", self.prompt_cache_key.as_deref())?;
        for (index, tool) in self.native_tools.iter().enumerate() {
            tool.validate(index)
                .map_err(|reason| ProviderOptionError::Rejected {
                    path: format!("native_tools[{index}]"),
                    reason,
                })?;
        }
        Ok(())
    }
}

/// Legacy xAI Chat Completions live-search parameters.
///
/// The current provider path for search is the Responses hosted-tool API. This type is retained as
/// a typed compatibility boundary for endpoints that still implement `search_parameters`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiSearchParameters {
    /// Search mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<SearchMode>,
    /// Whether to return citations.
    #[serde(
        rename = "returnCitations",
        alias = "return_citations",
        skip_serializing_if = "Option::is_none"
    )]
    pub return_citations: Option<bool>,
    /// Maximum number of search results.
    #[serde(
        rename = "maxSearchResults",
        alias = "max_search_results",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_search_results: Option<u32>,
    /// Start date for search (YYYY-MM-DD).
    #[serde(
        rename = "fromDate",
        alias = "from_date",
        skip_serializing_if = "Option::is_none"
    )]
    pub from_date: Option<String>,
    /// End date for search (YYYY-MM-DD).
    #[serde(
        rename = "toDate",
        alias = "to_date",
        skip_serializing_if = "Option::is_none"
    )]
    pub to_date: Option<String>,
    /// Search sources configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<SearchSource>>,
}

impl XaiSearchParameters {
    pub fn with_mode(mut self, mode: SearchMode) -> Self {
        self.mode = Some(mode);
        self
    }

    fn validate(&self, path: &str) -> Result<(), ProviderOptionError> {
        if self.max_search_results == Some(0) {
            return Err(ProviderOptionError::Rejected {
                path: format!("{path}.max_search_results"),
                reason: "maximum search results must be greater than zero".to_string(),
            });
        }
        let from = parse_optional_date(self.from_date.as_deref(), path, "from_date")?;
        let to = parse_optional_date(self.to_date.as_deref(), path, "to_date")?;
        if from.zip(to).is_some_and(|(from, to)| from > to) {
            return Err(ProviderOptionError::Rejected {
                path: path.to_string(),
                reason: "from_date must not be later than to_date".to_string(),
            });
        }
        for (index, source) in self.sources.iter().flatten().enumerate() {
            source.validate(path, index)?;
        }
        Ok(())
    }
}

/// Search mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchSource {
    /// Web search source.
    Web(WebSearchSource),
    /// News search source.
    News(NewsSearchSource),
    /// X (Twitter) search source.
    X(XSearchSource),
    /// RSS feed search source.
    Rss(RssSearchSource),
}

impl SearchSource {
    /// Wrap a web source into the discriminated search-source union.
    pub fn web(source: WebSearchSource) -> Self {
        Self::Web(source)
    }

    /// Wrap a news source into the discriminated search-source union.
    pub fn news(source: NewsSearchSource) -> Self {
        Self::News(source)
    }

    /// Wrap an X source into the discriminated search-source union.
    pub fn x(source: XSearchSource) -> Self {
        Self::X(source)
    }

    /// Wrap an RSS source into the discriminated search-source union.
    pub fn rss(source: RssSearchSource) -> Self {
        Self::Rss(source)
    }

    fn validate(&self, path: &str, index: usize) -> Result<(), ProviderOptionError> {
        let source_path = format!("{path}.sources[{index}]");
        match self {
            Self::Web(source)
                if source
                    .allowed_websites
                    .as_ref()
                    .is_some_and(|v| !v.is_empty())
                    && source
                        .excluded_websites
                        .as_ref()
                        .is_some_and(|v| !v.is_empty()) =>
            {
                Err(ProviderOptionError::Rejected {
                    path: source_path,
                    reason: "web source cannot combine allowed and excluded websites".to_string(),
                })
            }
            Self::X(source)
                if source
                    .included_x_handles
                    .as_ref()
                    .is_some_and(|v| !v.is_empty())
                    && source
                        .excluded_x_handles
                        .as_ref()
                        .is_some_and(|v| !v.is_empty()) =>
            {
                Err(ProviderOptionError::Rejected {
                    path: source_path,
                    reason: "X source cannot combine included and excluded handles".to_string(),
                })
            }
            Self::Rss(source) if source.links.is_empty() => Err(ProviderOptionError::Rejected {
                path: source_path,
                reason: "RSS source must include at least one link".to_string(),
            }),
            Self::Web(_) | Self::News(_) | Self::X(_) | Self::Rss(_) => Ok(()),
        }
    }
}

impl From<WebSearchSource> for SearchSource {
    fn from(value: WebSearchSource) -> Self {
        Self::Web(value)
    }
}

impl From<NewsSearchSource> for SearchSource {
    fn from(value: NewsSearchSource) -> Self {
        Self::News(value)
    }
}

impl From<XSearchSource> for SearchSource {
    fn from(value: XSearchSource) -> Self {
        Self::X(value)
    }
}

impl From<RssSearchSource> for SearchSource {
    fn from(value: RssSearchSource) -> Self {
        Self::Rss(value)
    }
}

/// Web search source parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebSearchSource {
    /// Country code for localized search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Allowed websites.
    #[serde(
        rename = "allowedWebsites",
        alias = "allowed_websites",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_websites: Option<Vec<String>>,
    /// Excluded websites.
    #[serde(
        rename = "excludedWebsites",
        alias = "excluded_websites",
        skip_serializing_if = "Option::is_none"
    )]
    pub excluded_websites: Option<Vec<String>>,
    /// Enable safe search.
    #[serde(
        rename = "safeSearch",
        alias = "safe_search",
        skip_serializing_if = "Option::is_none"
    )]
    pub safe_search: Option<bool>,
}

impl WebSearchSource {
    pub fn new() -> Self {
        Self::default()
    }
}

/// News search source parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NewsSearchSource {
    /// Country code for localized search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Excluded websites.
    #[serde(
        rename = "excludedWebsites",
        alias = "excluded_websites",
        skip_serializing_if = "Option::is_none"
    )]
    pub excluded_websites: Option<Vec<String>>,
    /// Enable safe search.
    #[serde(
        rename = "safeSearch",
        alias = "safe_search",
        skip_serializing_if = "Option::is_none"
    )]
    pub safe_search: Option<bool>,
}

impl NewsSearchSource {
    pub fn new() -> Self {
        Self::default()
    }
}

/// X search source parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XSearchSource {
    /// Excluded X handles for X sources.
    pub excluded_x_handles: Option<Vec<String>>,
    /// Included X handles for X sources.
    pub included_x_handles: Option<Vec<String>>,
    /// Minimum favorite count for X posts.
    pub post_favorite_count: Option<u64>,
    /// Minimum view count for X posts.
    pub post_view_count: Option<u64>,
}

impl XSearchSource {
    pub fn new() -> Self {
        Self::default()
    }
}

/// RSS search source parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RssSearchSource {
    /// RSS feed links for RSS sources.
    pub links: Vec<String>,
}

impl RssSearchSource {
    pub fn new<I, S>(links: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            links: links.into_iter().map(Into::into).collect(),
        }
    }
}

impl Serialize for SearchSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            SearchSource::Web(source) => {
                let mut field_count = 1;
                if source.country.is_some() {
                    field_count += 1;
                }
                if source.allowed_websites.is_some() {
                    field_count += 1;
                }
                if source.excluded_websites.is_some() {
                    field_count += 1;
                }
                if source.safe_search.is_some() {
                    field_count += 1;
                }

                let mut map = serializer.serialize_map(Some(field_count))?;
                map.serialize_entry("type", "web")?;
                if let Some(country) = &source.country {
                    map.serialize_entry("country", country)?;
                }
                if let Some(allowed_websites) = &source.allowed_websites {
                    map.serialize_entry("allowedWebsites", allowed_websites)?;
                }
                if let Some(excluded_websites) = &source.excluded_websites {
                    map.serialize_entry("excludedWebsites", excluded_websites)?;
                }
                if let Some(safe_search) = source.safe_search {
                    map.serialize_entry("safeSearch", &safe_search)?;
                }
                map.end()
            }
            SearchSource::News(source) => {
                let mut field_count = 1;
                if source.country.is_some() {
                    field_count += 1;
                }
                if source.excluded_websites.is_some() {
                    field_count += 1;
                }
                if source.safe_search.is_some() {
                    field_count += 1;
                }

                let mut map = serializer.serialize_map(Some(field_count))?;
                map.serialize_entry("type", "news")?;
                if let Some(country) = &source.country {
                    map.serialize_entry("country", country)?;
                }
                if let Some(excluded_websites) = &source.excluded_websites {
                    map.serialize_entry("excludedWebsites", excluded_websites)?;
                }
                if let Some(safe_search) = source.safe_search {
                    map.serialize_entry("safeSearch", &safe_search)?;
                }
                map.end()
            }
            SearchSource::X(source) => {
                let mut field_count = 1;
                if source.excluded_x_handles.is_some() {
                    field_count += 1;
                }
                if source.included_x_handles.is_some() {
                    field_count += 1;
                }
                if source.post_favorite_count.is_some() {
                    field_count += 1;
                }
                if source.post_view_count.is_some() {
                    field_count += 1;
                }

                let mut map = serializer.serialize_map(Some(field_count))?;
                map.serialize_entry("type", "x")?;
                if let Some(excluded_x_handles) = &source.excluded_x_handles {
                    map.serialize_entry("excludedXHandles", excluded_x_handles)?;
                }
                if let Some(included_x_handles) = &source.included_x_handles {
                    map.serialize_entry("includedXHandles", included_x_handles)?;
                }
                if let Some(post_favorite_count) = source.post_favorite_count {
                    map.serialize_entry("postFavoriteCount", &post_favorite_count)?;
                }
                if let Some(post_view_count) = source.post_view_count {
                    map.serialize_entry("postViewCount", &post_view_count)?;
                }
                map.end()
            }
            SearchSource::Rss(source) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "rss")?;
                map.serialize_entry("links", &source.links)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SearchSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "type", rename_all = "lowercase")]
        enum SearchSourceInput {
            Web {
                #[serde(default)]
                country: Option<String>,
                #[serde(default, rename = "allowedWebsites", alias = "allowed_websites")]
                allowed_websites: Option<Vec<String>>,
                #[serde(default, rename = "excludedWebsites", alias = "excluded_websites")]
                excluded_websites: Option<Vec<String>>,
                #[serde(default, rename = "safeSearch", alias = "safe_search")]
                safe_search: Option<bool>,
            },
            News {
                #[serde(default)]
                country: Option<String>,
                #[serde(default, rename = "excludedWebsites", alias = "excluded_websites")]
                excluded_websites: Option<Vec<String>>,
                #[serde(default, rename = "safeSearch", alias = "safe_search")]
                safe_search: Option<bool>,
            },
            X {
                #[serde(default, rename = "excludedXHandles", alias = "excluded_x_handles")]
                excluded_x_handles: Option<Vec<String>>,
                #[serde(default, rename = "includedXHandles", alias = "included_x_handles")]
                included_x_handles: Option<Vec<String>>,
                #[serde(default, rename = "postFavoriteCount", alias = "post_favorite_count")]
                post_favorite_count: Option<u64>,
                #[serde(default, rename = "postViewCount", alias = "post_view_count")]
                post_view_count: Option<u64>,
            },
            Rss {
                links: Vec<String>,
            },
        }

        Ok(match SearchSourceInput::deserialize(deserializer)? {
            SearchSourceInput::Web {
                country,
                allowed_websites,
                excluded_websites,
                safe_search,
            } => SearchSource::Web(WebSearchSource {
                country,
                allowed_websites,
                excluded_websites,
                safe_search,
            }),
            SearchSourceInput::News {
                country,
                excluded_websites,
                safe_search,
            } => SearchSource::News(NewsSearchSource {
                country,
                excluded_websites,
                safe_search,
            }),
            SearchSourceInput::X {
                excluded_x_handles,
                included_x_handles,
                post_favorite_count,
                post_view_count,
            } => SearchSource::X(XSearchSource {
                excluded_x_handles,
                included_x_handles,
                post_favorite_count,
                post_view_count,
            }),
            SearchSourceInput::Rss { links } => SearchSource::Rss(RssSearchSource { links }),
        })
    }
}

fn validate_logprobs(
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
) -> Result<(), ProviderOptionError> {
    if top_logprobs.is_some_and(|count| count > 8) {
        return Err(ProviderOptionError::Rejected {
            path: "top_logprobs".to_string(),
            reason: "top_logprobs must be between zero and eight".to_string(),
        });
    }
    if top_logprobs.is_some() && logprobs == Some(false) {
        return Err(ProviderOptionError::Rejected {
            path: "logprobs".to_string(),
            reason: "logprobs cannot be false when top_logprobs is present".to_string(),
        });
    }
    Ok(())
}

fn validate_non_empty_id(path: &str, value: Option<&str>) -> Result<(), ProviderOptionError> {
    if value.is_some_and(|value| value.trim().is_empty() || value.len() > 1024) {
        return Err(ProviderOptionError::Rejected {
            path: path.to_string(),
            reason: "value must be non-empty and at most 1024 bytes".to_string(),
        });
    }
    Ok(())
}

fn parse_optional_date(
    value: Option<&str>,
    path: &str,
    field: &str,
) -> Result<Option<chrono::NaiveDate>, ProviderOptionError> {
    value
        .map(|value| {
            chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|_| {
                ProviderOptionError::Rejected {
                    path: format!("{path}.{field}"),
                    reason: "date must use YYYY-MM-DD format".to_string(),
                }
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xai_chat_options_typed_builders_serialize_chat_fields() {
        let value = serde_json::to_value(
            XaiChatOptions::new()
                .with_reasoning_effort("high")
                .with_top_logprobs(3)
                .with_prompt_cache_key("cache-123")
                .with_search(XaiSearchParameters {
                    mode: Some(SearchMode::On),
                    return_citations: Some(true),
                    max_search_results: Some(5),
                    from_date: Some("2026-03-01".to_string()),
                    to_date: Some("2026-03-11".to_string()),
                    sources: Some(vec![SearchSource::Web(WebSearchSource {
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec!["example.com".to_string()]),
                        excluded_websites: None,
                        safe_search: Some(true),
                    })]),
                })
                .with_parallel_tool_calls(false),
        )
        .expect("serialize xai options");

        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["topLogprobs"], serde_json::json!(3));
        assert_eq!(value["parallelToolCalls"], serde_json::json!(false));
        assert_eq!(value["promptCacheKey"], serde_json::json!("cache-123"));
        assert_eq!(value["searchParameters"]["mode"], serde_json::json!("on"));
        assert_eq!(
            value["searchParameters"]["returnCitations"],
            serde_json::json!(true)
        );
        assert_eq!(
            value["searchParameters"]["maxSearchResults"],
            serde_json::json!(5)
        );
        assert_eq!(
            value["searchParameters"]["fromDate"],
            serde_json::json!("2026-03-01")
        );
        assert_eq!(
            value["searchParameters"]["toDate"],
            serde_json::json!("2026-03-11")
        );
        assert_eq!(
            value["searchParameters"]["sources"][0]["allowedWebsites"],
            serde_json::json!(["example.com"])
        );
        assert!(
            value["searchParameters"]["sources"][0]
                .get("excludedWebsites")
                .is_none()
        );
        assert_eq!(
            value["searchParameters"]["sources"][0]["safeSearch"],
            serde_json::json!(true)
        );
        assert!(value.get("reasoning_effort").is_none());
        assert!(value.get("top_logprobs").is_none());
        assert!(value.get("search_parameters").is_none());
        assert!(value.get("reasoning_summary").is_none());
        assert!(value.get("store").is_none());
        assert!(value.get("previous_response_id").is_none());
        assert!(value.get("include").is_none());
    }

    #[test]
    fn xai_responses_options_typed_builders_serialize_responses_fields() {
        let value = serde_json::to_value(
            XaiResponsesOptions::new()
                .with_reasoning_effort("medium")
                .with_reasoning_summary("detailed")
                .with_top_logprobs(3)
                .with_parallel_tool_calls(false)
                .with_store(false)
                .with_previous_response("resp_prev_123")
                .with_prompt_cache_key("cache-123")
                .with_include(["file_search_call.results"]),
        )
        .expect("serialize xai responses options");

        assert_eq!(value["reasoningEffort"], serde_json::json!("medium"));
        assert_eq!(value["reasoningSummary"], serde_json::json!("detailed"));
        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["topLogprobs"], serde_json::json!(3));
        assert_eq!(value["parallelToolCalls"], serde_json::json!(false));
        assert_eq!(value["store"], serde_json::json!(false));
        assert_eq!(
            value["previousResponseId"],
            serde_json::json!("resp_prev_123")
        );
        assert_eq!(value["promptCacheKey"], serde_json::json!("cache-123"));
        assert_eq!(
            value["include"],
            serde_json::json!(["file_search_call.results"])
        );
        assert!(value.get("reasoning_effort").is_none());
        assert!(value.get("reasoning_summary").is_none());
        assert!(value.get("top_logprobs").is_none());
        assert!(value.get("previous_response_id").is_none());
    }

    #[test]
    fn xai_responses_options_preserve_explicit_store_true() {
        let value = serde_json::to_value(
            XaiResponsesOptions::new()
                .with_store(false)
                .with_store(true),
        )
        .expect("serialize xai responses options");

        assert_eq!(value["store"], serde_json::json!(true));
    }

    #[test]
    fn xai_string_enums_keep_known_and_forward_compatible_variants() {
        assert_eq!(
            XaiChatReasoningEffort::from("high"),
            XaiChatReasoningEffort::High
        );
        assert_eq!(
            XaiResponsesReasoningEffort::from("medium"),
            XaiResponsesReasoningEffort::Medium
        );
        assert_eq!(
            XaiReasoningSummary::from("verbose"),
            XaiReasoningSummary::Other("verbose".to_string())
        );
        assert_eq!(
            XaiResponseInclude::from("custom.include.path"),
            XaiResponseInclude::Other("custom.include.path".to_string())
        );
    }

    #[test]
    fn xai_search_parameters_deserialize_camel_case_aliases() {
        let value = serde_json::json!({
            "mode": "on",
            "returnCitations": true,
            "maxSearchResults": 7,
            "fromDate": "2026-03-01",
            "toDate": "2026-03-11",
            "sources": [{
                "type": "web",
                "allowedWebsites": ["example.com"],
                "safeSearch": true
            }]
        });

        let params: XaiSearchParameters =
            serde_json::from_value(value).expect("deserialize xai search parameters");

        assert!(matches!(params.mode, Some(SearchMode::On)));
        assert_eq!(params.return_citations, Some(true));
        assert_eq!(params.max_search_results, Some(7));
        assert_eq!(params.from_date.as_deref(), Some("2026-03-01"));
        assert_eq!(params.to_date.as_deref(), Some("2026-03-11"));
        assert_eq!(
            params.sources,
            Some(vec![SearchSource::Web(WebSearchSource {
                country: None,
                allowed_websites: Some(vec!["example.com".to_string()]),
                excluded_websites: None,
                safe_search: Some(true),
            })])
        );
    }

    #[test]
    fn xai_chat_options_deserialize_wire_aliases() {
        let value = serde_json::json!({
            "reasoning_effort": "high",
            "top_logprobs": 3,
            "parallel_tool_calls": false,
            "prompt_cache_key": "cache-123",
            "search_parameters": {
                "mode": "on",
                "return_citations": true,
                "max_search_results": 5,
                "from_date": "2026-03-01",
                "to_date": "2026-03-11"
            }
        });

        let options: XaiChatOptions =
            serde_json::from_value(value).expect("deserialize xai chat options");

        assert_eq!(options.reasoning_effort, Some(XaiChatReasoningEffort::High));
        assert_eq!(options.top_logprobs, Some(3));
        assert_eq!(options.parallel_tool_calls, Some(false));
        assert_eq!(options.prompt_cache_key.as_deref(), Some("cache-123"));
        assert!(matches!(
            options
                .search_parameters
                .as_ref()
                .and_then(|value| value.mode),
            Some(SearchMode::On)
        ));
        assert_eq!(
            options
                .search_parameters
                .as_ref()
                .and_then(|value| value.return_citations),
            Some(true)
        );
        assert_eq!(
            options
                .search_parameters
                .as_ref()
                .and_then(|value| value.max_search_results),
            Some(5)
        );
    }

    #[test]
    fn xai_responses_options_deserialize_wire_aliases() {
        let value = serde_json::json!({
            "reasoning_effort": "medium",
            "reasoning_summary": "detailed",
            "logprobs": true,
            "top_logprobs": 3,
            "store": false,
            "previous_response_id": "resp_prev_123",
            "include": ["file_search_call.results"]
        });

        let options: XaiResponsesOptions =
            serde_json::from_value(value).expect("deserialize xai responses options");

        assert_eq!(
            options.reasoning_effort,
            Some(XaiResponsesReasoningEffort::Medium)
        );
        assert_eq!(
            options.reasoning_summary,
            Some(XaiReasoningSummary::Detailed)
        );
        assert_eq!(options.logprobs, Some(true));
        assert_eq!(options.top_logprobs, Some(3));
        assert_eq!(options.store, Some(false));
        assert_eq!(
            options.previous_response_id.as_deref(),
            Some("resp_prev_123")
        );
        assert_eq!(
            options.include,
            Some(vec![XaiResponseInclude::FileSearchCallResults])
        );
    }

    #[test]
    fn xai_search_parameters_deserialize_wire_aliases() {
        let value = serde_json::json!({
            "mode": "on",
            "return_citations": true,
            "max_search_results": 7,
            "from_date": "2026-03-01",
            "to_date": "2026-03-11",
            "sources": [{
                "type": "web",
                "allowed_websites": ["example.com"],
                "safe_search": true
            }]
        });

        let params: XaiSearchParameters =
            serde_json::from_value(value).expect("deserialize xai search parameters");

        assert!(matches!(params.mode, Some(SearchMode::On)));
        assert_eq!(params.return_citations, Some(true));
        assert_eq!(params.max_search_results, Some(7));
        assert_eq!(params.from_date.as_deref(), Some("2026-03-01"));
        assert_eq!(params.to_date.as_deref(), Some("2026-03-11"));
        assert_eq!(
            params.sources,
            Some(vec![SearchSource::Web(WebSearchSource {
                country: None,
                allowed_websites: Some(vec!["example.com".to_string()]),
                excluded_websites: None,
                safe_search: Some(true),
            })])
        );
    }

    #[test]
    fn xai_search_parameters_serialize_provider_option_field_names() {
        let value = serde_json::to_value(XaiSearchParameters {
            mode: Some(SearchMode::On),
            return_citations: Some(true),
            max_search_results: Some(3),
            from_date: Some("2026-03-01".to_string()),
            to_date: Some("2026-03-11".to_string()),
            sources: Some(vec![
                XSearchSource {
                    excluded_x_handles: Some(vec!["spam".to_string()]),
                    included_x_handles: Some(vec!["openai".to_string(), "deepmind".to_string()]),
                    post_favorite_count: Some(10),
                    post_view_count: Some(99),
                }
                .into(),
                RssSearchSource::new(["https://example.com/feed.xml"]).into(),
            ]),
        })
        .expect("serialize xai search parameters");

        assert_eq!(value["returnCitations"], serde_json::json!(true));
        assert_eq!(value["maxSearchResults"], serde_json::json!(3));
        assert_eq!(value["fromDate"], serde_json::json!("2026-03-01"));
        assert_eq!(value["toDate"], serde_json::json!("2026-03-11"));
        assert_eq!(value["sources"][0]["type"], serde_json::json!("x"));
        assert_eq!(
            value["sources"][0]["includedXHandles"],
            serde_json::json!(["openai", "deepmind"])
        );
        assert_eq!(
            value["sources"][0]["excludedXHandles"],
            serde_json::json!(["spam"])
        );
        assert_eq!(
            value["sources"][0]["postFavoriteCount"],
            serde_json::json!(10)
        );
        assert_eq!(value["sources"][0]["postViewCount"], serde_json::json!(99));
        assert!(value["sources"][0].get("included_x_handles").is_none());
        assert!(value["sources"][0].get("excluded_x_handles").is_none());
        assert_eq!(value["sources"][1]["type"], serde_json::json!("rss"));
        assert_eq!(
            value["sources"][1]["links"],
            serde_json::json!(["https://example.com/feed.xml"])
        );
        assert!(value.get("return_citations").is_none());
        assert!(value.get("max_search_results").is_none());
        assert!(value.get("from_date").is_none());
        assert!(value.get("to_date").is_none());
    }

    #[test]
    fn xai_search_source_deserializes_wire_aliases() {
        let value = serde_json::json!({
            "type": "x",
            "included_x_handles": ["openai", "deepmind"],
            "excluded_x_handles": ["grok"],
            "post_favorite_count": 10,
            "post_view_count": 99
        });

        let source: SearchSource =
            serde_json::from_value(value).expect("deserialize xai search source");

        assert_eq!(
            source,
            SearchSource::X(XSearchSource {
                excluded_x_handles: Some(vec!["grok".to_string()]),
                included_x_handles: Some(vec!["openai".to_string(), "deepmind".to_string()]),
                post_favorite_count: Some(10),
                post_view_count: Some(99),
            })
        );
    }

    #[test]
    fn xai_default_search_parameters_defer_to_provider_defaults() {
        let defaults = XaiSearchParameters::default();

        assert_eq!(defaults.mode, None);
        assert_eq!(defaults.return_citations, None);
        assert_eq!(defaults.max_search_results, None);
    }
}
