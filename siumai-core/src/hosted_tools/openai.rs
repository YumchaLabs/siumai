//! OpenAI Provider-Defined Tools
//!
//! Factory functions for creating OpenAI-specific provider-defined tools.
//! These tools are executed by OpenAI's servers and include web search,
//! file search, computer use, code interpreter, and image generation.
//!
//! # Examples
//!
//! ```rust
//! use siumai::hosted_tools::openai;
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
//!             .with_city("San Francisco"),
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

/// MCP server configuration builder (OpenAI Responses `mcp` tool).
///
/// Vercel AI SDK tool id: `openai.mcp`, tool name commonly `MCP`.
#[derive(Debug, Clone)]
pub struct McpConfig {
    server_label: String,
    server_url: String,
    server_description: Option<String>,
    require_approval: Option<String>,
}

impl McpConfig {
    /// Create a new MCP tool configuration.
    pub fn new(server_label: impl Into<String>, server_url: impl Into<String>) -> Self {
        Self {
            server_label: server_label.into(),
            server_url: server_url.into(),
            server_description: None,
            require_approval: None,
        }
    }

    /// Set the server description (human readable).
    pub fn with_server_description(mut self, desc: impl Into<String>) -> Self {
        self.server_description = Some(desc.into());
        self
    }

    /// Set the OpenAI tool approval policy.
    ///
    /// Vercel-aligned values include: `"never"`, `"always"`, `"auto"`.
    pub fn with_require_approval(mut self, policy: impl Into<String>) -> Self {
        self.require_approval = Some(policy.into());
        self
    }

    /// Build the Tool.
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({
            "serverLabel": self.server_label,
            "serverUrl": self.server_url,
        });

        if let Some(v) = self.server_description {
            args["serverDescription"] = serde_json::json!(v);
        }
        if let Some(v) = self.require_approval {
            args["requireApproval"] = serde_json::json!(v);
        }

        Tool::ProviderDefined(ProviderDefinedTool::new("openai.mcp", "MCP").with_args(args))
    }
}

/// Create an OpenAI MCP tool configuration (Responses API).
///
/// This tool allows OpenAI to call tools exposed by an MCP server. The model will emit
/// MCP tool calls via the Responses API protocol.
pub fn mcp(server_label: impl Into<String>, server_url: impl Into<String>) -> McpConfig {
    McpConfig::new(server_label, server_url)
}

/// Web search configuration builder
#[derive(Debug, Clone, Default)]
pub struct WebSearchConfig {
    tool_id: Option<&'static str>,
    tool_name: Option<&'static str>,
    search_context_size: Option<String>,
    user_location: Option<UserLocation>,
    external_web_access: Option<bool>,
    allowed_domains: Option<Vec<String>>,
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

    /// Whether to allow external web access (OpenAI Responses `web_search`).
    pub fn with_external_web_access(mut self, enabled: bool) -> Self {
        self.external_web_access = Some(enabled);
        self
    }

    /// Restrict web search to allowed domains (OpenAI Responses `web_search`).
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
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

        if let Some(v) = self.external_web_access {
            args["externalWebAccess"] = serde_json::json!(v);
        }

        if let Some(domains) = self.allowed_domains {
            args["filters"] = serde_json::json!({ "allowedDomains": domains });
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new(
                self.tool_id.unwrap_or("openai.web_search"),
                self.tool_name.unwrap_or("webSearch"),
            )
            .with_args(args),
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
pub fn web_search() -> WebSearchConfig {
    WebSearchConfig::new()
}

/// File search configuration builder
#[derive(Debug, Clone, Default)]
pub struct FileSearchConfig {
    vector_store_ids: Option<Vec<String>>,
    max_num_results: Option<u32>,
    ranking_options: Option<RankingOptions>,
    filters: Option<serde_json::Value>,
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

    /// Set file-search filters (provider-specific schema; Vercel-aligned).
    pub fn with_filters(mut self, filters: serde_json::Value) -> Self {
        self.filters = Some(filters);
        self
    }

    /// Build the Tool
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(ids) = self.vector_store_ids {
            // Vercel-aligned tool args (SDK shape)
            args["vectorStoreIds"] = serde_json::json!(ids);
        }

        if let Some(max) = self.max_num_results {
            args["maxNumResults"] = serde_json::json!(max);
        }

        if let Some(ranking) = self.ranking_options {
            let mut ranking_json = serde_json::json!({});
            if let Some(ranker) = ranking.ranker {
                ranking_json["ranker"] = serde_json::json!(ranker);
            }
            if let Some(threshold) = ranking.score_threshold {
                ranking_json["scoreThreshold"] = serde_json::json!(threshold);
            }
            args["ranking"] = ranking_json;
        }

        if let Some(filters) = self.filters {
            args["filters"] = filters;
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("openai.file_search", "fileSearch").with_args(args),
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
pub fn file_search() -> FileSearchConfig {
    FileSearchConfig::new()
}

/// Create a web search preview tool with default settings (Responses API).
///
/// This mirrors Vercel AI SDK's `openai.web_search_preview`.
pub fn web_search_preview() -> WebSearchConfig {
    WebSearchConfig {
        tool_id: Some("openai.web_search_preview"),
        tool_name: Some("web_search_preview"),
        ..WebSearchConfig::new()
    }
}

/// Create a computer use tool
pub fn computer_use(display_width: u32, display_height: u32, environment: &str) -> Tool {
    let args = serde_json::json!({
        "display_width": display_width,
        "display_height": display_height,
        "environment": environment,
    });

    Tool::ProviderDefined(
        ProviderDefinedTool::new("openai.computer_use", "computer_use").with_args(args),
    )
}

/// Code interpreter container configuration (Vercel-aligned).
#[derive(Debug, Clone)]
pub enum CodeInterpreterContainer {
    /// Use an existing container by ID.
    ContainerId(String),
    /// Use an auto container with uploaded file IDs available to the interpreter.
    AutoWithFiles { file_ids: Vec<String> },
}

/// Code interpreter configuration builder (OpenAI Responses `code_interpreter` tool).
#[derive(Debug, Clone, Default)]
pub struct CodeInterpreterConfig {
    container: Option<CodeInterpreterContainer>,
}

impl CodeInterpreterConfig {
    /// Create a new code interpreter configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use an existing container by ID.
    pub fn with_container_id(mut self, container_id: impl Into<String>) -> Self {
        self.container = Some(CodeInterpreterContainer::ContainerId(container_id.into()));
        self
    }

    /// Use an auto container with uploaded file IDs.
    pub fn with_file_ids(mut self, file_ids: Vec<String>) -> Self {
        self.container = Some(CodeInterpreterContainer::AutoWithFiles { file_ids });
        self
    }

    /// Build the Tool.
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(container) = self.container {
            match container {
                CodeInterpreterContainer::ContainerId(id) => {
                    args["container"] = serde_json::json!(id);
                }
                CodeInterpreterContainer::AutoWithFiles { file_ids } => {
                    args["container"] = serde_json::json!({ "fileIds": file_ids });
                }
            }
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("openai.code_interpreter", "codeExecution").with_args(args),
        )
    }
}

/// Create an OpenAI code interpreter tool configuration (Responses API).
pub fn code_interpreter() -> CodeInterpreterConfig {
    CodeInterpreterConfig::new()
}

/// Inpainting mask configuration for image generation (Vercel-aligned).
#[derive(Debug, Clone, Default)]
pub struct ImageMask {
    file_id: Option<String>,
    image_url: Option<String>,
}

impl ImageMask {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_file_id(mut self, file_id: impl Into<String>) -> Self {
        self.file_id = Some(file_id.into());
        self
    }

    pub fn with_image_url(mut self, image_url: impl Into<String>) -> Self {
        self.image_url = Some(image_url.into());
        self
    }
}

/// Image generation configuration builder (OpenAI Responses `image_generation` tool).
#[derive(Debug, Clone, Default)]
pub struct ImageGenerationConfig {
    background: Option<String>,
    input_fidelity: Option<String>,
    input_image_mask: Option<ImageMask>,
    model: Option<String>,
    moderation: Option<String>,
    output_compression: Option<u32>,
    output_format: Option<String>,
    partial_images: Option<u32>,
    quality: Option<String>,
    size: Option<String>,
}

impl ImageGenerationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_background(mut self, background: impl Into<String>) -> Self {
        self.background = Some(background.into());
        self
    }

    pub fn with_input_fidelity(mut self, fidelity: impl Into<String>) -> Self {
        self.input_fidelity = Some(fidelity.into());
        self
    }

    pub fn with_input_image_mask(mut self, mask: ImageMask) -> Self {
        self.input_image_mask = Some(mask);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_moderation(mut self, moderation: impl Into<String>) -> Self {
        self.moderation = Some(moderation.into());
        self
    }

    pub fn with_output_compression(mut self, compression: u32) -> Self {
        self.output_compression = Some(compression);
        self
    }

    pub fn with_output_format(mut self, format: impl Into<String>) -> Self {
        self.output_format = Some(format.into());
        self
    }

    pub fn with_partial_images(mut self, partial_images: u32) -> Self {
        self.partial_images = Some(partial_images);
        self
    }

    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }

    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(v) = self.background {
            args["background"] = serde_json::json!(v);
        }
        if let Some(v) = self.input_fidelity {
            args["inputFidelity"] = serde_json::json!(v);
        }
        if let Some(mask) = self.input_image_mask {
            let mut m = serde_json::json!({});
            if let Some(v) = mask.file_id {
                m["fileId"] = serde_json::json!(v);
            }
            if let Some(v) = mask.image_url {
                m["imageUrl"] = serde_json::json!(v);
            }
            args["inputImageMask"] = m;
        }
        if let Some(v) = self.model {
            args["model"] = serde_json::json!(v);
        }
        if let Some(v) = self.moderation {
            args["moderation"] = serde_json::json!(v);
        }
        if let Some(v) = self.output_compression {
            args["outputCompression"] = serde_json::json!(v);
        }
        if let Some(v) = self.output_format {
            args["outputFormat"] = serde_json::json!(v);
        }
        if let Some(v) = self.partial_images {
            args["partialImages"] = serde_json::json!(v);
        }
        if let Some(v) = self.quality {
            args["quality"] = serde_json::json!(v);
        }
        if let Some(v) = self.size {
            args["size"] = serde_json::json!(v);
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("openai.image_generation", "generateImage").with_args(args),
        )
    }
}

/// Create an OpenAI image generation tool configuration (Responses API).
pub fn image_generation() -> ImageGenerationConfig {
    ImageGenerationConfig::new()
}

/// Create an OpenAI local shell tool (Responses API).
///
/// This mirrors Vercel AI SDK's `openai.local_shell` tool, which uses `toolName: "shell"`.
pub fn local_shell() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new("openai.local_shell", "shell"))
}

/// Create an OpenAI shell tool (Responses API).
///
/// This mirrors Vercel AI SDK's `openai.shell` tool.
pub fn shell() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new("openai.shell", "shell"))
}

/// Create an OpenAI apply patch tool (Responses API).
pub fn apply_patch() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.apply_patch",
        "apply_patch",
    ))
}
