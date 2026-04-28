//! xAI provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["xai"]`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

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
    Low => "low",
    High => "high",
});

xai_string_enum!(XaiResponsesReasoningEffort {
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
});

xai_string_enum!(XaiImageResolution {
    OneK => "1k",
    TwoK => "2k",
});

xai_string_enum!(XaiImageQuality {
    Low => "low",
    Medium => "medium",
    High => "high",
});

xai_string_enum!(XaiVideoResolution {
    R480p => "480p",
    R720p => "720p",
});

xai_string_enum!(XaiVideoMode {
    EditVideo => "edit-video",
    ExtendVideo => "extend-video",
    ReferenceToVideo => "reference-to-video",
});

xai_string_enum!(SearchMode {
    Off => "off",
    Auto => "auto",
    On => "on",
});

/// xAI chat-completions specific options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct XaiLanguageModelChatOptions {
    /// Reasoning effort for Grok chat models.
    #[serde(
        default,
        rename = "reasoningEffort",
        alias = "reasoning_effort",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_effort: Option<XaiChatReasoningEffort>,
    /// Whether to return token logprobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top logprobs to return.
    #[serde(
        default,
        rename = "topLogprobs",
        alias = "top_logprobs",
        skip_serializing_if = "Option::is_none"
    )]
    pub top_logprobs: Option<u32>,
    /// Whether to enable parallel function calling during tool use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_function_calling: Option<bool>,
    /// Web search parameters.
    #[serde(
        default,
        rename = "searchParameters",
        alias = "search_parameters",
        skip_serializing_if = "Option::is_none"
    )]
    pub search_parameters: Option<XaiSearchParameters>,
}

impl XaiLanguageModelChatOptions {
    /// Create empty xAI chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: impl Into<XaiChatReasoningEffort>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Enable or disable logprobs.
    pub const fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    /// Request top logprobs and implicitly enable logprobs.
    pub const fn with_top_logprobs(mut self, count: u32) -> Self {
        self.top_logprobs = Some(count);
        self.logprobs = Some(true);
        self
    }

    /// Control parallel function calling.
    pub const fn with_parallel_function_calling(mut self, enabled: bool) -> Self {
        self.parallel_function_calling = Some(enabled);
        self
    }

    /// Set web search parameters.
    pub fn with_search(mut self, params: XaiSearchParameters) -> Self {
        self.search_parameters = Some(params);
        self
    }

    /// Enable web search with default settings.
    pub fn with_default_search(self) -> Self {
        self.with_search(XaiSearchParameters::default())
    }
}

/// Backward-compatible alias for xAI chat options.
pub type XaiOptions = XaiLanguageModelChatOptions;

/// Deprecated AI SDK-compatible alias for xAI chat options.
#[deprecated(note = "Use XaiLanguageModelChatOptions instead.")]
pub type XaiProviderOptions = XaiLanguageModelChatOptions;

/// xAI Responses-specific provider options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct XaiLanguageModelResponsesOptions {
    /// Reasoning effort for Responses models.
    #[serde(
        default,
        rename = "reasoningEffort",
        alias = "reasoning_effort",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_effort: Option<XaiResponsesReasoningEffort>,
    /// Reasoning summary verbosity.
    #[serde(
        default,
        rename = "reasoningSummary",
        alias = "reasoning_summary",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_summary: Option<XaiReasoningSummary>,
    /// Whether to return token logprobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top logprobs to return.
    #[serde(
        default,
        rename = "topLogprobs",
        alias = "top_logprobs",
        skip_serializing_if = "Option::is_none"
    )]
    pub top_logprobs: Option<u32>,
    /// Whether to store the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Previous response id.
    #[serde(
        default,
        rename = "previousResponseId",
        alias = "previous_response_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub previous_response_id: Option<String>,
    /// Additional output data to include.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<XaiResponseInclude>>,
}

impl XaiLanguageModelResponsesOptions {
    /// Create empty xAI Responses options.
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

    /// Enable or disable logprobs.
    pub const fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    /// Request top logprobs and implicitly enable logprobs.
    pub const fn with_top_logprobs(mut self, count: u32) -> Self {
        self.top_logprobs = Some(count);
        self.logprobs = Some(true);
        self
    }

    /// Control response storage.
    pub const fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Continue from a previous response id.
    pub fn with_previous_response(mut self, response_id: impl Into<String>) -> Self {
        self.previous_response_id = Some(response_id.into());
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
}

/// Deprecated AI SDK-compatible alias for xAI Responses options.
#[deprecated(note = "Use XaiLanguageModelResponsesOptions instead.")]
pub type XaiResponsesProviderOptions = XaiLanguageModelResponsesOptions;

/// xAI image-generation specific options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct XaiImageModelOptions {
    /// Output aspect ratio.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "aspectRatio"
    )]
    pub aspect_ratio: Option<String>,
    /// Output image format.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "outputFormat"
    )]
    pub output_format: Option<String>,
    /// Whether to block until the image is fully generated.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "syncMode")]
    pub sync_mode: Option<bool>,
    /// Output resolution hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<XaiImageResolution>,
    /// Output quality hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<XaiImageQuality>,
    /// End-user identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Additional xAI image fields.
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl XaiImageModelOptions {
    /// Create empty xAI image options.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    pub fn with_output_format(mut self, output_format: impl Into<String>) -> Self {
        self.output_format = Some(output_format.into());
        self
    }

    pub const fn with_sync_mode(mut self, sync_mode: bool) -> Self {
        self.sync_mode = Some(sync_mode);
        self
    }

    pub fn with_resolution(mut self, resolution: impl Into<XaiImageResolution>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    pub fn with_quality(mut self, quality: impl Into<XaiImageQuality>) -> Self {
        self.quality = Some(quality.into());
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// Deprecated AI SDK-compatible alias for xAI image options.
#[deprecated(note = "Use XaiImageModelOptions instead.")]
pub type XaiImageProviderOptions = XaiImageModelOptions;

/// xAI video-generation specific options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct XaiVideoModelOptions {
    /// Polling interval in milliseconds.
    #[serde(
        default,
        rename = "pollIntervalMs",
        alias = "poll_interval_ms",
        skip_serializing_if = "Option::is_none"
    )]
    pub poll_interval_ms: Option<u64>,
    /// Polling timeout in milliseconds.
    #[serde(
        default,
        rename = "pollTimeoutMs",
        alias = "poll_timeout_ms",
        skip_serializing_if = "Option::is_none"
    )]
    pub poll_timeout_ms: Option<u64>,
    /// Output resolution hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<XaiVideoResolution>,
    /// Explicit xAI video operation mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<XaiVideoMode>,
    /// Source video URL for edit/extend modes.
    #[serde(
        default,
        rename = "videoUrl",
        alias = "video_url",
        skip_serializing_if = "Option::is_none"
    )]
    pub video_url: Option<String>,
    /// Reference image URLs for reference-to-video mode.
    #[serde(
        default,
        rename = "referenceImageUrls",
        alias = "reference_image_urls",
        skip_serializing_if = "Option::is_none"
    )]
    pub reference_image_urls: Option<Vec<String>>,
    /// Additional xAI video fields.
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl XaiVideoModelOptions {
    /// Create empty xAI video options.
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_poll_interval_ms(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = Some(poll_interval_ms);
        self
    }

    pub const fn with_poll_timeout_ms(mut self, poll_timeout_ms: u64) -> Self {
        self.poll_timeout_ms = Some(poll_timeout_ms);
        self
    }

    pub fn with_resolution(mut self, resolution: impl Into<XaiVideoResolution>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    pub fn with_mode(mut self, mode: impl Into<XaiVideoMode>) -> Self {
        self.mode = Some(mode.into());
        self
    }

    pub fn with_video_url(mut self, video_url: impl Into<String>) -> Self {
        self.video_url = Some(video_url.into());
        self
    }

    pub fn with_reference_image_urls<I, S>(mut self, reference_image_urls: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.reference_image_urls =
            Some(reference_image_urls.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// Deprecated AI SDK-compatible alias for xAI video options.
#[deprecated(note = "Use XaiVideoModelOptions instead.")]
pub type XaiVideoProviderOptions = XaiVideoModelOptions;

/// xAI file-upload specific options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct XaiFilesOptions {
    /// Team identifier.
    #[serde(
        default,
        rename = "teamId",
        alias = "team_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub team_id: Option<String>,
    /// Provider-native file path hint.
    #[serde(
        default,
        rename = "filePath",
        alias = "file_path",
        skip_serializing_if = "Option::is_none"
    )]
    pub file_path: Option<String>,
    /// Additional xAI file fields.
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl XaiFilesOptions {
    /// Create empty xAI files options.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_team_id(mut self, team_id: impl Into<String>) -> Self {
        self.team_id = Some(team_id.into());
        self
    }

    pub fn with_file_path(mut self, file_path: impl Into<String>) -> Self {
        self.file_path = Some(file_path.into());
        self
    }

    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// xAI web search parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiSearchParameters {
    /// Search mode.
    pub mode: SearchMode,
    /// Whether to return citations.
    #[serde(
        default,
        rename = "returnCitations",
        alias = "return_citations",
        skip_serializing_if = "Option::is_none"
    )]
    pub return_citations: Option<bool>,
    /// Start date for search data.
    #[serde(
        default,
        rename = "fromDate",
        alias = "from_date",
        skip_serializing_if = "Option::is_none"
    )]
    pub from_date: Option<String>,
    /// End date for search data.
    #[serde(
        default,
        rename = "toDate",
        alias = "to_date",
        skip_serializing_if = "Option::is_none"
    )]
    pub to_date: Option<String>,
    /// Maximum number of search results.
    #[serde(
        default,
        rename = "maxSearchResults",
        alias = "max_search_results",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_search_results: Option<u32>,
    /// Search source configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<SearchSource>>,
}

impl Default for XaiSearchParameters {
    fn default() -> Self {
        Self {
            mode: SearchMode::Auto,
            return_citations: Some(true),
            from_date: None,
            to_date: None,
            max_search_results: Some(20),
            sources: None,
        }
    }
}

impl XaiSearchParameters {
    /// Create search parameters with AI SDK-aligned defaults.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: impl Into<SearchMode>) -> Self {
        self.mode = mode.into();
        self
    }

    pub const fn with_return_citations(mut self, enabled: bool) -> Self {
        self.return_citations = Some(enabled);
        self
    }

    pub const fn with_max_search_results(mut self, max_search_results: u32) -> Self {
        self.max_search_results = Some(max_search_results);
        self
    }

    pub fn with_from_date(mut self, from_date: impl Into<String>) -> Self {
        self.from_date = Some(from_date.into());
        self
    }

    pub fn with_to_date(mut self, to_date: impl Into<String>) -> Self {
        self.to_date = Some(to_date.into());
        self
    }

    pub fn with_sources<I, S>(mut self, sources: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<SearchSource>,
    {
        self.sources = Some(sources.into_iter().map(Into::into).collect());
        self
    }
}

/// xAI search source configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SearchSource {
    Web(WebSearchSource),
    X(XSearchSource),
    News(NewsSearchSource),
    Rss(RssSearchSource),
}

impl From<WebSearchSource> for SearchSource {
    fn from(value: WebSearchSource) -> Self {
        Self::Web(value)
    }
}

impl From<XSearchSource> for SearchSource {
    fn from(value: XSearchSource) -> Self {
        Self::X(value)
    }
}

impl From<NewsSearchSource> for SearchSource {
    fn from(value: NewsSearchSource) -> Self {
        Self::News(value)
    }
}

impl From<RssSearchSource> for SearchSource {
    fn from(value: RssSearchSource) -> Self {
        Self::Rss(value)
    }
}

/// Web search source.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchSource {
    /// Two-letter country code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Excluded websites.
    #[serde(
        default,
        rename = "excludedWebsites",
        alias = "excluded_websites",
        skip_serializing_if = "Option::is_none"
    )]
    pub excluded_websites: Option<Vec<String>>,
    /// Allowed websites.
    #[serde(
        default,
        rename = "allowedWebsites",
        alias = "allowed_websites",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_websites: Option<Vec<String>>,
    /// Safe-search flag.
    #[serde(
        default,
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

    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    pub fn with_allowed_websites<I, S>(mut self, websites: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_websites = Some(websites.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_excluded_websites<I, S>(mut self, websites: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.excluded_websites = Some(websites.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_safe_search(mut self, safe_search: bool) -> Self {
        self.safe_search = Some(safe_search);
        self
    }
}

/// X/Twitter search source.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct XSearchSource {
    /// Excluded X handles.
    #[serde(
        default,
        rename = "excludedXHandles",
        alias = "excluded_x_handles",
        skip_serializing_if = "Option::is_none"
    )]
    pub excluded_x_handles: Option<Vec<String>>,
    /// Included X handles.
    #[serde(
        default,
        rename = "includedXHandles",
        alias = "included_x_handles",
        alias = "xHandles",
        alias = "x_handles",
        skip_serializing_if = "Option::is_none"
    )]
    pub included_x_handles: Option<Vec<String>>,
    /// Minimum post favorite count.
    #[serde(
        default,
        rename = "postFavoriteCount",
        alias = "post_favorite_count",
        skip_serializing_if = "Option::is_none"
    )]
    pub post_favorite_count: Option<u32>,
    /// Minimum post view count.
    #[serde(
        default,
        rename = "postViewCount",
        alias = "post_view_count",
        skip_serializing_if = "Option::is_none"
    )]
    pub post_view_count: Option<u32>,
}

impl XSearchSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_included_x_handles<I, S>(mut self, handles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.included_x_handles = Some(handles.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_excluded_x_handles<I, S>(mut self, handles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.excluded_x_handles = Some(handles.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_post_favorite_count(mut self, count: u32) -> Self {
        self.post_favorite_count = Some(count);
        self
    }

    pub const fn with_post_view_count(mut self, count: u32) -> Self {
        self.post_view_count = Some(count);
        self
    }
}

/// News search source.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewsSearchSource {
    /// Two-letter country code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Excluded websites.
    #[serde(
        default,
        rename = "excludedWebsites",
        alias = "excluded_websites",
        skip_serializing_if = "Option::is_none"
    )]
    pub excluded_websites: Option<Vec<String>>,
    /// Safe-search flag.
    #[serde(
        default,
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

    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    pub fn with_excluded_websites<I, S>(mut self, websites: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.excluded_websites = Some(websites.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_safe_search(mut self, safe_search: bool) -> Self {
        self.safe_search = Some(safe_search);
        self
    }
}

/// RSS search source.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RssSearchSource {
    /// RSS feed links.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xai_chat_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            XaiLanguageModelChatOptions::new()
                .with_reasoning_effort("high")
                .with_top_logprobs(3)
                .with_parallel_function_calling(false)
                .with_search(
                    XaiSearchParameters::new()
                        .with_mode(SearchMode::On)
                        .with_max_search_results(5)
                        .with_sources(vec![
                            SearchSource::from(
                                WebSearchSource::new()
                                    .with_country("US")
                                    .with_allowed_websites(["example.com"])
                                    .with_safe_search(true),
                            ),
                            SearchSource::from(
                                XSearchSource::new()
                                    .with_included_x_handles(["openai"])
                                    .with_post_view_count(99),
                            ),
                        ]),
                ),
        )
        .expect("options serialize");

        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["topLogprobs"], serde_json::json!(3));
        assert_eq!(value["parallel_function_calling"], serde_json::json!(false));
        assert_eq!(value["searchParameters"]["mode"], serde_json::json!("on"));
        assert_eq!(
            value["searchParameters"]["maxSearchResults"],
            serde_json::json!(5)
        );
        assert_eq!(
            value["searchParameters"]["sources"][0]["allowedWebsites"],
            serde_json::json!(["example.com"])
        );
        assert_eq!(
            value["searchParameters"]["sources"][1]["includedXHandles"],
            serde_json::json!(["openai"])
        );
        assert!(value.get("reasoning_effort").is_none());
        assert!(value.get("top_logprobs").is_none());
        assert!(value.get("search_parameters").is_none());
    }

    #[test]
    fn xai_responses_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            XaiLanguageModelResponsesOptions::new()
                .with_reasoning_effort("medium")
                .with_reasoning_summary("detailed")
                .with_top_logprobs(2)
                .with_store(false)
                .with_previous_response("resp_123")
                .with_include(["file_search_call.results"]),
        )
        .expect("options serialize");

        assert_eq!(value["reasoningEffort"], serde_json::json!("medium"));
        assert_eq!(value["reasoningSummary"], serde_json::json!("detailed"));
        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["topLogprobs"], serde_json::json!(2));
        assert_eq!(value["store"], serde_json::json!(false));
        assert_eq!(value["previousResponseId"], serde_json::json!("resp_123"));
        assert_eq!(
            value["include"],
            serde_json::json!(["file_search_call.results"])
        );
    }

    #[test]
    fn xai_image_video_and_file_options_serialize_to_ai_sdk_shape() {
        let image = serde_json::to_value(
            XaiImageModelOptions::new()
                .with_aspect_ratio("16:9")
                .with_output_format("png")
                .with_sync_mode(true)
                .with_resolution("2k")
                .with_quality("high")
                .with_user("user-123"),
        )
        .expect("image options serialize");

        assert_eq!(image["aspect_ratio"], serde_json::json!("16:9"));
        assert_eq!(image["output_format"], serde_json::json!("png"));
        assert_eq!(image["sync_mode"], serde_json::json!(true));
        assert_eq!(image["resolution"], serde_json::json!("2k"));
        assert_eq!(image["quality"], serde_json::json!("high"));
        assert_eq!(image["user"], serde_json::json!("user-123"));

        let video = serde_json::to_value(
            XaiVideoModelOptions::new()
                .with_poll_interval_ms(1000)
                .with_poll_timeout_ms(60000)
                .with_resolution("720p")
                .with_mode("reference-to-video")
                .with_reference_image_urls(["https://example.com/ref.png"]),
        )
        .expect("video options serialize");

        assert_eq!(video["pollIntervalMs"], serde_json::json!(1000));
        assert_eq!(video["pollTimeoutMs"], serde_json::json!(60000));
        assert_eq!(video["resolution"], serde_json::json!("720p"));
        assert_eq!(video["mode"], serde_json::json!("reference-to-video"));
        assert_eq!(
            video["referenceImageUrls"],
            serde_json::json!(["https://example.com/ref.png"])
        );

        let files = serde_json::to_value(
            XaiFilesOptions::new()
                .with_team_id("team-123")
                .with_file_path("/tmp/a.txt")
                .with_extra_field("retention_days", serde_json::json!(7)),
        )
        .expect("files options serialize");

        assert_eq!(files["teamId"], serde_json::json!("team-123"));
        assert_eq!(files["filePath"], serde_json::json!("/tmp/a.txt"));
        assert_eq!(files["retention_days"], serde_json::json!(7));
    }

    #[test]
    fn xai_options_accept_legacy_aliases() {
        let options: XaiLanguageModelChatOptions = serde_json::from_value(serde_json::json!({
            "reasoning_effort": "high",
            "top_logprobs": 3,
            "search_parameters": {
                "mode": "on",
                "return_citations": true,
                "max_search_results": 5,
                "sources": [{
                    "type": "x",
                    "x_handles": ["openai"],
                    "post_view_count": 10
                }]
            }
        }))
        .expect("options deserialize");

        assert_eq!(options.reasoning_effort, Some(XaiChatReasoningEffort::High));
        assert_eq!(options.top_logprobs, Some(3));
        let source = options
            .search_parameters
            .and_then(|params| params.sources)
            .and_then(|sources| sources.into_iter().next())
            .expect("source");
        assert_eq!(
            source,
            SearchSource::X(XSearchSource {
                included_x_handles: Some(vec!["openai".to_string()]),
                post_view_count: Some(10),
                ..Default::default()
            })
        );
    }

    #[test]
    #[allow(deprecated)]
    fn xai_option_aliases_remain_available() {
        let _: XaiOptions = XaiLanguageModelChatOptions::new();
        let _: XaiProviderOptions = XaiLanguageModelChatOptions::new();
        let _: XaiResponsesProviderOptions = XaiLanguageModelResponsesOptions::new();
        let _: XaiImageProviderOptions = XaiImageModelOptions::new();
        let _: XaiVideoProviderOptions = XaiVideoModelOptions::new();
    }
}
