use crate::types::HttpConfig;
use serde::{Deserialize, Serialize};

/// Gemini-specific configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the Gemini API
    pub base_url: String,
    /// Default model to use
    pub model: String,
    /// Default generation configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<super::GenerationConfig>,
    /// Default safety settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<super::SafetySetting>>,
    /// HTTP timeout in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    /// HTTP configuration (custom headers, proxy, user agent)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_config: Option<HttpConfig>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-1.5-flash".to_string(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: Some(HttpConfig::default()),
        }
    }
}

impl GeminiConfig {
    /// Create a new Gemini configuration with the given API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }
    /// Set the model to use
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }
    /// Set the base URL
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
    /// Set generation configuration
    pub fn with_generation_config(mut self, config: super::GenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }
    /// Set safety settings
    pub fn with_safety_settings(mut self, settings: Vec<super::SafetySetting>) -> Self {
        self.safety_settings = Some(settings);
        self
    }
    /// Set HTTP timeout
    pub const fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set HTTP config (headers/proxy/user-agent)
    pub fn with_http_config(mut self, http: HttpConfig) -> Self {
        self.http_config = Some(http);
        self
    }
}

/// Tool configuration for any Tool specified in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Optional. Function calling config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

/// Configuration for specifying function calling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallingConfig {
    /// Optional. Specifies the mode in which function calling should execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<FunctionCallingMode>,
    /// Optional. A set of function names that, when provided, limits the functions the model will call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Defines the execution behavior for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionCallingMode {
    #[serde(rename = "MODE_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "AUTO")]
    Auto,
    #[serde(rename = "ANY")]
    Any,
    #[serde(rename = "NONE")]
    None,
}

// ================================================================================================
// Embedding options & extension helpers (Gemini-specific convenience APIs)
// ================================================================================================

/// Gemini-specific embedding configuration options
///
/// This struct provides type-safe configuration for Gemini embedding requests,
/// including task type optimization, context titles, and custom dimensions.
///
/// # Example
/// ```rust,no_run
/// use siumai::providers::gemini::types::GeminiEmbeddingOptions;
/// use siumai::types::EmbeddingTaskType;
///
/// let options = GeminiEmbeddingOptions::new()
///     .with_task_type(EmbeddingTaskType::RetrievalQuery)
///     .with_title("Search Context")
///     .with_output_dimensionality(768);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GeminiEmbeddingOptions {
    /// Task type for optimization (Gemini-specific feature)
    pub task_type: Option<crate::types::EmbeddingTaskType>,
    /// Title for additional context (helps with embedding quality)
    pub title: Option<String>,
    /// Custom output dimensions (128-3072, must be supported by model)
    pub output_dimensionality: Option<u32>,
}

impl GeminiEmbeddingOptions {
    /// Create new Gemini embedding options with default values
    pub fn new() -> Self {
        Self::default()
    }
    /// Set task type for optimization
    pub fn with_task_type(mut self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }
    /// Set title for additional context
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    /// Set custom output dimensions
    pub fn with_output_dimensionality(mut self, dimensions: u32) -> Self {
        self.output_dimensionality = Some(dimensions);
        self
    }
    /// Apply these options to an EmbeddingRequest
    pub fn apply_to_request(
        self,
        mut request: crate::types::EmbeddingRequest,
    ) -> crate::types::EmbeddingRequest {
        if let Some(task_type) = self.task_type {
            request = request.with_task_type(task_type);
        }
        if let Some(title) = self.title {
            request = request.with_provider_param("title", serde_json::Value::String(title));
        }
        if let Some(dims) = self.output_dimensionality {
            request.dimensions = Some(dims);
        }
        request
    }
}

/// Extension trait for EmbeddingRequest to add Gemini-specific configuration
pub trait GeminiEmbeddingRequestExt {
    /// Configure this request with Gemini-specific options
    fn with_gemini_config(self, config: GeminiEmbeddingOptions) -> Self;
    /// Quick method to set Gemini task type
    fn with_gemini_task_type(self, task_type: crate::types::EmbeddingTaskType) -> Self;
    /// Quick method to set Gemini title
    fn with_gemini_title(self, title: impl Into<String>) -> Self;
    /// Quick method to set Gemini output dimensions
    fn with_gemini_dimensions(self, dimensions: u32) -> Self;
}

impl GeminiEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_gemini_config(self, config: GeminiEmbeddingOptions) -> Self {
        config.apply_to_request(self)
    }
    fn with_gemini_task_type(self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.with_task_type(task_type)
    }
    fn with_gemini_title(self, title: impl Into<String>) -> Self {
        self.with_provider_param("title", serde_json::Value::String(title.into()))
    }
    fn with_gemini_dimensions(self, dimensions: u32) -> Self {
        let mut request = self;
        request.dimensions = Some(dimensions);
        request
    }
}
