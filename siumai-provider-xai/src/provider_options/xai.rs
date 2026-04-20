//! xAI (Grok) provider options.
//!
//! These typed option structs are owned by the xAI provider crate and are serialized into
//! `providerOptions["xai"]` (Vercel-aligned open options map).

use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
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
    ReasoningEncryptedContent => "reasoning.encrypted_content",
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

/// xAI image-generation specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiImageOptions {
    /// Output aspect ratio (for example `1:1`, `16:9`, `9:16`).
    #[serde(skip_serializing_if = "Option::is_none", alias = "aspectRatio")]
    pub aspect_ratio: Option<String>,
    /// Output image format (for example `png`, `jpeg`, `webp`).
    #[serde(skip_serializing_if = "Option::is_none", alias = "outputFormat")]
    pub output_format: Option<String>,
    /// Whether to block until the image is fully generated.
    #[serde(skip_serializing_if = "Option::is_none", alias = "syncMode")]
    pub sync_mode: Option<bool>,
    /// Output resolution hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<XaiImageResolution>,
    /// Output quality hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<XaiImageQuality>,
    /// End-user identifier for provider-side attribution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Forward-compatible provider-owned escape hatch for newly introduced options.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl XaiImageOptions {
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

    pub fn with_sync_mode(mut self, sync_mode: bool) -> Self {
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

/// xAI video-generation specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiVideoOptions {
    /// Polling interval in milliseconds.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollIntervalMs",
        alias = "poll_interval_ms"
    )]
    pub poll_interval_ms: Option<u64>,
    /// Polling timeout in milliseconds.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollTimeoutMs",
        alias = "poll_timeout_ms"
    )]
    pub poll_timeout_ms: Option<u64>,
    /// Output resolution hint (`480p` or `720p`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<XaiVideoResolution>,
    /// Explicit xAI video operation mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<XaiVideoMode>,
    /// Source video URL for video editing.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "videoUrl",
        alias = "video_url"
    )]
    pub video_url: Option<String>,
    /// Reference image URLs for reference-to-video generation.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "referenceImageUrls",
        alias = "reference_image_urls"
    )]
    pub reference_image_urls: Option<Vec<String>>,
    /// Forward-compatible provider-owned escape hatch for newly introduced options.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl XaiVideoOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_poll_interval_ms(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = Some(poll_interval_ms);
        self
    }

    pub fn with_poll_timeout_ms(mut self, poll_timeout_ms: u64) -> Self {
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

/// xAI file-upload specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiFilesOptions {
    /// Team identifier forwarded as `team_id` on the xAI multipart upload endpoint.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "teamId",
        alias = "team_id"
    )]
    pub team_id: Option<String>,
    /// Optional provider-native file path hint.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "filePath",
        alias = "file_path"
    )]
    pub file_path: Option<String>,
    /// Forward-compatible provider-owned escape hatch for newly introduced fields.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl XaiFilesOptions {
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

/// xAI chat-completions specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// Whether to enable parallel function calling during tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_function_calling: Option<bool>,
    /// Web search parameters.
    #[serde(
        rename = "searchParameters",
        alias = "search_parameters",
        skip_serializing_if = "Option::is_none"
    )]
    pub search_parameters: Option<XaiSearchParameters>,
}

impl XaiChatOptions {
    /// Create new xAI chat options.
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

    /// Enable or disable parallel function calling.
    pub fn with_parallel_function_calling(mut self, enabled: bool) -> Self {
        self.parallel_function_calling = Some(enabled);
        self
    }
}

/// xAI Responses-specific provider options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// Whether to store the response for later retrieval.
    ///
    /// Vercel parity: `true` is omitted from the payload, `false` is sent explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Previous response id for continuing a response chain.
    #[serde(
        rename = "previousResponseId",
        alias = "previous_response_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub previous_response_id: Option<String>,
    /// Additional response payload sections to include.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<XaiResponseInclude>>,
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

    /// Control response storage.
    ///
    /// `true` clears the explicit override and falls back to the provider default.
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = if store { None } else { Some(false) };
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

/// Backward-compatible alias for the native xAI chat options surface.
pub type XaiOptions = XaiChatOptions;

/// AI SDK-style alias for xAI chat language-model options.
pub type XaiLanguageModelChatOptions = XaiChatOptions;

/// Deprecated AI SDK compatibility alias for xAI chat options.
#[deprecated(note = "Use `XaiLanguageModelChatOptions` instead.")]
pub type XaiProviderOptions = XaiLanguageModelChatOptions;

/// AI SDK-style alias for xAI Responses options.
pub type XaiLanguageModelResponsesOptions = XaiResponsesOptions;

/// Deprecated AI SDK compatibility alias for xAI Responses options.
#[deprecated(note = "Use `XaiLanguageModelResponsesOptions` instead.")]
pub type XaiResponsesProviderOptions = XaiLanguageModelResponsesOptions;

/// AI SDK-style alias for xAI image-model options.
pub type XaiImageModelOptions = XaiImageOptions;

/// Deprecated AI SDK compatibility alias for xAI image options.
#[deprecated(note = "Use `XaiImageModelOptions` instead.")]
pub type XaiImageProviderOptions = XaiImageModelOptions;

/// AI SDK-style alias for xAI video-model options.
pub type XaiVideoModelOptions = XaiVideoOptions;

/// Deprecated AI SDK compatibility alias for xAI video options.
#[deprecated(note = "Use `XaiVideoModelOptions` instead.")]
pub type XaiVideoProviderOptions = XaiVideoModelOptions;

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

impl Default for XaiSearchParameters {
    fn default() -> Self {
        Self {
            mode: SearchMode::Auto,
            return_citations: Some(true),
            max_search_results: Some(20),
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

    /// Deprecated AI SDK alias helper. Normalized to `includedXHandles`.
    pub fn with_legacy_x_handles<I, S>(mut self, handles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.included_x_handles = Some(handles.into_iter().map(Into::into).collect());
        self
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
                #[serde(default, rename = "xHandles", alias = "x_handles")]
                x_handles: Option<Vec<String>>,
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
                x_handles,
                post_favorite_count,
                post_view_count,
            } => SearchSource::X(XSearchSource {
                excluded_x_handles,
                included_x_handles: included_x_handles.or(x_handles),
                post_favorite_count,
                post_view_count,
            }),
            SearchSourceInput::Rss { links } => SearchSource::Rss(RssSearchSource { links }),
        })
    }
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

    #[test]
    fn xai_image_options_serialize_ai_sdk_fields_and_passthrough() {
        let value = serde_json::to_value(
            XaiImageOptions::new()
                .with_aspect_ratio("16:9")
                .with_output_format("png")
                .with_sync_mode(true)
                .with_resolution("2k")
                .with_quality("high")
                .with_user("user-123")
                .with_extra_field("custom", serde_json::json!(true)),
        )
        .expect("serialize xai image options");

        assert_eq!(value["aspect_ratio"], serde_json::json!("16:9"));
        assert_eq!(value["output_format"], serde_json::json!("png"));
        assert_eq!(value["sync_mode"], serde_json::json!(true));
        assert_eq!(value["resolution"], serde_json::json!("2k"));
        assert_eq!(value["quality"], serde_json::json!("high"));
        assert_eq!(value["user"], serde_json::json!("user-123"));
        assert_eq!(value["custom"], serde_json::json!(true));
    }

    #[test]
    fn xai_video_options_deserialize_camel_case_aliases_and_passthrough() {
        let options: XaiVideoOptions = serde_json::from_value(serde_json::json!({
            "pollIntervalMs": 1000,
            "pollTimeoutMs": 60000,
            "resolution": "720p",
            "mode": "extend-video",
            "videoUrl": "https://example.com/video.mp4",
            "referenceImageUrls": [
                "https://example.com/ref-1.png",
                "https://example.com/ref-2.png"
            ],
            "style": "cinematic"
        }))
        .expect("deserialize xai video options");

        assert_eq!(options.poll_interval_ms, Some(1000));
        assert_eq!(options.poll_timeout_ms, Some(60000));
        assert_eq!(options.resolution, Some(XaiVideoResolution::R720p));
        assert_eq!(options.mode, Some(XaiVideoMode::ExtendVideo));
        assert_eq!(
            options.video_url.as_deref(),
            Some("https://example.com/video.mp4")
        );
        assert_eq!(
            options.reference_image_urls,
            Some(vec![
                "https://example.com/ref-1.png".to_string(),
                "https://example.com/ref-2.png".to_string(),
            ])
        );
        assert_eq!(
            options.extra_fields.get("style"),
            Some(&serde_json::json!("cinematic"))
        );
    }

    #[test]
    fn xai_video_options_serialize_ai_sdk_fields_and_passthrough() {
        let value = serde_json::to_value(
            XaiVideoOptions::new()
                .with_poll_interval_ms(1000)
                .with_poll_timeout_ms(60000)
                .with_resolution("720p")
                .with_mode("reference-to-video")
                .with_video_url("https://example.com/video.mp4")
                .with_reference_image_urls([
                    "https://example.com/ref-1.png",
                    "https://example.com/ref-2.png",
                ])
                .with_extra_field("style", serde_json::json!("cinematic")),
        )
        .expect("serialize xai video options");

        assert_eq!(value["pollIntervalMs"], serde_json::json!(1000));
        assert_eq!(value["pollTimeoutMs"], serde_json::json!(60000));
        assert_eq!(value["resolution"], serde_json::json!("720p"));
        assert_eq!(value["mode"], serde_json::json!("reference-to-video"));
        assert_eq!(
            value["videoUrl"],
            serde_json::json!("https://example.com/video.mp4")
        );
        assert_eq!(
            value["referenceImageUrls"],
            serde_json::json!([
                "https://example.com/ref-1.png",
                "https://example.com/ref-2.png"
            ])
        );
        assert_eq!(value["style"], serde_json::json!("cinematic"));
    }

    #[test]
    fn xai_files_options_serialize_aliases_and_passthrough() {
        let value = serde_json::to_value(
            XaiFilesOptions::new()
                .with_team_id("team-123")
                .with_file_path("/uploads/demo.txt")
                .with_extra_field("retention_days", serde_json::json!(7)),
        )
        .expect("serialize xai files options");

        assert_eq!(value["teamId"], serde_json::json!("team-123"));
        assert_eq!(value["filePath"], serde_json::json!("/uploads/demo.txt"));
        assert_eq!(value["retention_days"], serde_json::json!(7));
    }

    #[test]
    fn xai_files_options_deserialize_camel_case_aliases() {
        let options: XaiFilesOptions = serde_json::from_value(serde_json::json!({
            "teamId": "team-123",
            "filePath": "/uploads/demo.txt",
            "custom": true
        }))
        .expect("deserialize xai files options");

        assert_eq!(options.team_id.as_deref(), Some("team-123"));
        assert_eq!(options.file_path.as_deref(), Some("/uploads/demo.txt"));
        assert_eq!(
            options.extra_fields.get("custom"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn xai_chat_options_typed_builders_serialize_chat_fields() {
        let value = serde_json::to_value(
            XaiOptions::new()
                .with_reasoning_effort("high")
                .with_top_logprobs(3)
                .with_search(XaiSearchParameters {
                    mode: SearchMode::On,
                    return_citations: Some(true),
                    max_search_results: Some(5),
                    from_date: Some("2026-03-01".to_string()),
                    to_date: Some("2026-03-11".to_string()),
                    sources: Some(vec![SearchSource::Web(WebSearchSource {
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec!["example.com".to_string()]),
                        excluded_websites: Some(vec!["blocked.example.com".to_string()]),
                        safe_search: Some(true),
                    })]),
                })
                .with_parallel_function_calling(false),
        )
        .expect("serialize xai options");

        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["topLogprobs"], serde_json::json!(3));
        assert_eq!(value["parallel_function_calling"], serde_json::json!(false));
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
        assert_eq!(
            value["searchParameters"]["sources"][0]["excludedWebsites"],
            serde_json::json!(["blocked.example.com"])
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
                .with_store(false)
                .with_previous_response("resp_prev_123")
                .with_include(["file_search_call.results"]),
        )
        .expect("serialize xai responses options");

        assert_eq!(value["reasoningEffort"], serde_json::json!("medium"));
        assert_eq!(value["reasoningSummary"], serde_json::json!("detailed"));
        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["topLogprobs"], serde_json::json!(3));
        assert_eq!(value["store"], serde_json::json!(false));
        assert_eq!(
            value["previousResponseId"],
            serde_json::json!("resp_prev_123")
        );
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
    fn xai_responses_options_with_store_true_omits_explicit_store() {
        let value = serde_json::to_value(
            XaiResponsesOptions::new()
                .with_store(false)
                .with_store(true),
        )
        .expect("serialize xai responses options");

        assert!(value.get("store").is_none());
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
                "excludedWebsites": ["blocked.example.com"],
                "safeSearch": true
            }]
        });

        let params: XaiSearchParameters =
            serde_json::from_value(value).expect("deserialize xai search parameters");

        assert!(matches!(params.mode, SearchMode::On));
        assert_eq!(params.return_citations, Some(true));
        assert_eq!(params.max_search_results, Some(7));
        assert_eq!(params.from_date.as_deref(), Some("2026-03-01"));
        assert_eq!(params.to_date.as_deref(), Some("2026-03-11"));
        assert_eq!(
            params.sources,
            Some(vec![SearchSource::Web(WebSearchSource {
                country: None,
                allowed_websites: Some(vec!["example.com".to_string()]),
                excluded_websites: Some(vec!["blocked.example.com".to_string()]),
                safe_search: Some(true),
            })])
        );
    }

    #[test]
    fn xai_chat_options_deserialize_legacy_snake_case_aliases() {
        let value = serde_json::json!({
            "reasoning_effort": "high",
            "top_logprobs": 3,
            "parallel_function_calling": false,
            "search_parameters": {
                "mode": "on",
                "return_citations": true,
                "max_search_results": 5,
                "from_date": "2026-03-01",
                "to_date": "2026-03-11"
            }
        });

        let options: XaiChatOptions =
            serde_json::from_value(value).expect("deserialize legacy xai chat options");

        assert_eq!(options.reasoning_effort, Some(XaiChatReasoningEffort::High));
        assert_eq!(options.top_logprobs, Some(3));
        assert_eq!(options.parallel_function_calling, Some(false));
        assert!(matches!(
            options.search_parameters.as_ref().map(|value| &value.mode),
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
    fn xai_responses_options_deserialize_legacy_snake_case_aliases() {
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
            serde_json::from_value(value).expect("deserialize legacy xai responses options");

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
    fn xai_search_parameters_deserialize_legacy_snake_case_aliases() {
        let value = serde_json::json!({
            "mode": "on",
            "return_citations": true,
            "max_search_results": 7,
            "from_date": "2026-03-01",
            "to_date": "2026-03-11",
            "sources": [{
                "type": "web",
                "allowed_websites": ["example.com"],
                "excluded_websites": ["blocked.example.com"],
                "safe_search": true
            }]
        });

        let params: XaiSearchParameters =
            serde_json::from_value(value).expect("deserialize legacy xai search parameters");

        assert!(matches!(params.mode, SearchMode::On));
        assert_eq!(params.return_citations, Some(true));
        assert_eq!(params.max_search_results, Some(7));
        assert_eq!(params.from_date.as_deref(), Some("2026-03-01"));
        assert_eq!(params.to_date.as_deref(), Some("2026-03-11"));
        assert_eq!(
            params.sources,
            Some(vec![SearchSource::Web(WebSearchSource {
                country: None,
                allowed_websites: Some(vec!["example.com".to_string()]),
                excluded_websites: Some(vec!["blocked.example.com".to_string()]),
                safe_search: Some(true),
            })])
        );
    }

    #[test]
    fn xai_search_parameters_serialize_ai_sdk_field_names() {
        let value = serde_json::to_value(XaiSearchParameters {
            mode: SearchMode::On,
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
        assert!(value["sources"][0].get("xHandles").is_none());
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
    fn xai_search_source_legacy_x_handles_alias_maps_to_included_x_handles() {
        let source: SearchSource = XSearchSource::new()
            .with_legacy_x_handles(["openai", "deepmind"])
            .into();
        let value = serde_json::to_value(source).expect("serialize xai search source");

        assert_eq!(value["type"], serde_json::json!("x"));
        assert_eq!(
            value["includedXHandles"],
            serde_json::json!(["openai", "deepmind"])
        );
        assert!(value.get("xHandles").is_none());
        assert!(value.get("included_x_handles").is_none());
    }

    #[test]
    fn xai_search_source_deserializes_deprecated_and_legacy_aliases() {
        let value = serde_json::json!({
            "type": "x",
            "x_handles": ["openai", "deepmind"],
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
    fn xai_default_search_parameters_match_ai_sdk_defaults() {
        let defaults = XaiSearchParameters::default();

        assert!(matches!(defaults.mode, SearchMode::Auto));
        assert_eq!(defaults.return_citations, Some(true));
        assert_eq!(defaults.max_search_results, Some(20));
    }
}
