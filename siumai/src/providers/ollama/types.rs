//! Ollama-specific type definitions
//!
//! This module contains type definitions specific to the Ollama API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ollama chat request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    /// Model name
    pub model: String,
    /// Messages in the conversation
    pub messages: Vec<OllamaChatMessage>,
    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Output format (json or schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    /// Additional model options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    /// Keep model loaded duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Should the model think before responding (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
}

/// Ollama generate request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaGenerateRequest {
    /// Model name
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Suffix text (for completion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// Images for multimodal models (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Output format (json or schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    /// Additional model options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    /// System message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Prompt template
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Raw mode (bypass templating)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
    /// Keep model loaded duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Context from previous request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    /// Should the model think before responding (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
}

/// Ollama chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatMessage {
    /// Role of the message sender
    pub role: String,
    /// Content of the message
    pub content: String,
    /// Images for multimodal models (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    /// The model's thinking process (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

/// Ollama tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    /// Type of tool (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: OllamaFunction,
}

/// Ollama function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// Function parameters schema
    pub parameters: serde_json::Value,
}

/// Ollama tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    /// Function being called
    pub function: OllamaFunctionCall,
}

/// Ollama function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    /// Function name
    pub name: String,
    /// Function arguments
    pub arguments: serde_json::Value,
}

/// Ollama chat response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatResponse {
    /// Model used
    pub model: String,
    /// Creation timestamp
    pub created_at: String,
    /// Response message
    pub message: OllamaChatMessage,
    /// Whether the response is complete
    pub done: bool,
    /// Reason for completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Total duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Load duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Prompt evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Prompt evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    /// Evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    /// Evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Ollama generate response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaGenerateResponse {
    /// Model used
    pub model: String,
    /// Creation timestamp
    pub created_at: String,
    /// Generated response text
    pub response: String,
    /// Whether the response is complete
    pub done: bool,
    /// Context for next request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    /// Total duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Load duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Prompt evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Prompt evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    /// Evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    /// Evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Ollama embedding request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaEmbeddingRequest {
    /// Model name
    pub model: String,
    /// Input text or list of texts
    pub input: serde_json::Value,
    /// Truncate input to fit context length
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    /// Additional model options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    /// Keep model loaded duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

/// Ollama embedding response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaEmbeddingResponse {
    /// Model used
    pub model: String,
    /// Generated embeddings
    pub embeddings: Vec<Vec<f64>>,
    /// Total duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Load duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Prompt evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
}

/// Ollama model information
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModel {
    /// Model name
    pub name: String,
    /// Model identifier
    pub model: String,
    /// Last modified timestamp
    pub modified_at: String,
    /// Model size in bytes
    pub size: u64,
    /// Model digest
    pub digest: String,
    /// Model details
    pub details: OllamaModelDetails,
}

/// Ollama model details
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelDetails {
    /// Parent model
    pub parent_model: String,
    /// Model format
    pub format: String,
    /// Model family
    pub family: String,
    /// Model families
    pub families: Vec<String>,
    /// Parameter size
    pub parameter_size: String,
    /// Quantization level
    pub quantization_level: String,
}

/// Ollama models list response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelsResponse {
    /// List of models
    pub models: Vec<OllamaModel>,
}

/// Ollama running models response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaRunningModelsResponse {
    /// List of running models
    pub models: Vec<OllamaRunningModel>,
}

/// Ollama running model information
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaRunningModel {
    /// Model name
    pub name: String,
    /// Model identifier
    pub model: String,
    /// Model size in bytes
    pub size: u64,
    /// Model digest
    pub digest: String,
    /// Model details
    pub details: OllamaModelDetails,
    /// Expiration time
    pub expires_at: String,
    /// VRAM size in bytes
    pub size_vram: u64,
}

/// Ollama version response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaVersionResponse {
    /// Ollama version
    pub version: String,
}

// ================================================================================================
// Embedding Types
// ================================================================================================

/// Ollama-specific embedding configuration options
///
/// This struct provides type-safe configuration for Ollama embedding requests,
/// including truncation control, model keep-alive settings, and custom model options.
///
/// # Example
/// ```rust,no_run
/// use siumai::providers::ollama::OllamaEmbeddingOptions;
/// use std::collections::HashMap;
///
/// let mut model_options = HashMap::new();
/// model_options.insert("temperature".to_string(), serde_json::json!(0.1));
///
/// let options = OllamaEmbeddingOptions::new()
///     .with_truncate(true)
///     .with_keep_alive("5m")
///     .with_options(model_options);
/// ```
#[derive(Debug, Clone, Default)]
pub struct OllamaEmbeddingOptions {
    /// Whether to truncate input to fit context length
    pub truncate: Option<bool>,
    /// How long to keep the model loaded in memory
    pub keep_alive: Option<String>,
    /// Additional model-specific options
    pub options: Option<HashMap<String, serde_json::Value>>,
}

impl OllamaEmbeddingOptions {
    /// Create new Ollama embedding options with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set truncation behavior
    ///
    /// When true, input text will be truncated to fit the model's context length.
    /// When false, requests with text longer than context length will fail.
    pub fn with_truncate(mut self, truncate: bool) -> Self {
        self.truncate = Some(truncate);
        self
    }

    /// Set keep alive duration
    ///
    /// Controls how long the model stays loaded in memory after the request.
    /// Examples: "5m", "10s", "1h", or "0" to unload immediately.
    pub fn with_keep_alive(mut self, duration: impl Into<String>) -> Self {
        self.keep_alive = Some(duration.into());
        self
    }

    /// Set model options
    ///
    /// Additional options to pass to the model. Common options include:
    /// - `temperature`: Controls randomness (0.0 to 1.0)
    /// - `top_p`: Controls diversity via nucleus sampling
    /// - `top_k`: Controls diversity by limiting token choices
    /// - `repeat_penalty`: Penalizes repetition
    pub fn with_options(mut self, options: HashMap<String, serde_json::Value>) -> Self {
        self.options = Some(options);
        self
    }

    /// Add a single model option
    ///
    /// Convenience method to add individual options without creating a HashMap.
    pub fn with_option(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let mut options = self.options.unwrap_or_default();
        options.insert(key.into(), value);
        self.options = Some(options);
        self
    }

    /// Apply these options to an EmbeddingRequest
    ///
    /// This method modifies the provided EmbeddingRequest to include
    /// Ollama-specific parameters.
    pub fn apply_to_request(
        self,
        mut request: crate::types::EmbeddingRequest,
    ) -> crate::types::EmbeddingRequest {
        if let Some(truncate) = self.truncate {
            request = request.with_provider_param("truncate", serde_json::Value::Bool(truncate));
        }
        if let Some(keep_alive) = self.keep_alive {
            request =
                request.with_provider_param("keep_alive", serde_json::Value::String(keep_alive));
        }
        if let Some(options) = self.options {
            request = request.with_provider_param(
                "options",
                serde_json::Value::Object(options.into_iter().collect()),
            );
        }
        request
    }
}

/// Extension trait for EmbeddingRequest to add Ollama-specific configuration
pub trait OllamaEmbeddingRequestExt {
    /// Configure this request with Ollama-specific options
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::types::EmbeddingRequest;
    /// use siumai::providers::ollama::{OllamaEmbeddingOptions, OllamaEmbeddingRequestExt};
    ///
    /// let request = EmbeddingRequest::new(vec!["text".to_string()])
    ///     .with_ollama_config(
    ///         OllamaEmbeddingOptions::new()
    ///             .with_truncate(true)
    ///             .with_keep_alive("5m")
    ///     );
    /// ```
    fn with_ollama_config(self, config: OllamaEmbeddingOptions) -> Self;

    /// Quick method to set Ollama truncation behavior
    fn with_ollama_truncate(self, truncate: bool) -> Self;

    /// Quick method to set Ollama keep alive duration
    fn with_ollama_keep_alive(self, duration: impl Into<String>) -> Self;

    /// Quick method to add a single Ollama model option
    fn with_ollama_option(self, key: impl Into<String>, value: serde_json::Value) -> Self;
}

impl OllamaEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_ollama_config(self, config: OllamaEmbeddingOptions) -> Self {
        config.apply_to_request(self)
    }

    fn with_ollama_truncate(self, truncate: bool) -> Self {
        self.with_provider_param("truncate", serde_json::Value::Bool(truncate))
    }

    fn with_ollama_keep_alive(self, duration: impl Into<String>) -> Self {
        self.with_provider_param("keep_alive", serde_json::Value::String(duration.into()))
    }

    fn with_ollama_option(self, key: impl Into<String>, value: serde_json::Value) -> Self {
        // Get existing options or create new ones
        let mut options = self
            .provider_params
            .get("options")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        options.insert(key.into(), value);

        self.with_provider_param("options", serde_json::Value::Object(options))
    }
}
