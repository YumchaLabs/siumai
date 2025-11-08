//! Google Gemini Provider Options
//!
//! This module contains types for Gemini-specific features including:
//! - Code execution configuration
//! - Search grounding (web search)
//! - Dynamic retrieval settings

use serde::{Deserialize, Serialize};

/// Google Gemini specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiOptions {
    /// Code execution configuration
    pub code_execution: Option<CodeExecutionConfig>,
    /// Search grounding (web search)
    pub search_grounding: Option<SearchGroundingConfig>,
    /// File Search configuration (Gemini File Search tool)
    pub file_search: Option<FileSearchConfig>,
    /// Preferred MIME type for responses (e.g., "application/json")
    pub response_mime_type: Option<String>,
}

impl GeminiOptions {
    /// Create new Gemini options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable code execution
    pub fn with_code_execution(mut self, config: CodeExecutionConfig) -> Self {
        self.code_execution = Some(config);
        self
    }

    /// Enable search grounding
    pub fn with_search_grounding(mut self, config: SearchGroundingConfig) -> Self {
        self.search_grounding = Some(config);
        self
    }

    /// Enable File Search with given store names
    pub fn with_file_search_store_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stores: Vec<String> = names.into_iter().map(Into::into).collect();
        self.file_search = Some(FileSearchConfig {
            file_search_store_names: stores,
        });
        self
    }

    /// Set preferred response MIME type (e.g., application/json)
    pub fn with_response_mime_type(mut self, mime: impl Into<String>) -> Self {
        self.response_mime_type = Some(mime.into());
        self
    }
}

/// Code execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionConfig {
    /// Whether code execution is enabled
    pub enabled: bool,
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Search grounding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchGroundingConfig {
    /// Whether search grounding is enabled
    pub enabled: bool,
    /// Dynamic retrieval configuration
    pub dynamic_retrieval_config: Option<DynamicRetrievalConfig>,
}

impl Default for SearchGroundingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dynamic_retrieval_config: None,
        }
    }
}

/// Dynamic retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRetrievalConfig {
    /// Retrieval mode
    pub mode: DynamicRetrievalMode,
    /// Dynamic threshold
    pub dynamic_threshold: Option<f32>,
}

/// Dynamic retrieval mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DynamicRetrievalMode {
    /// Unspecified mode
    ModeUnspecified,
    /// Dynamic mode
    ModeDynamic,
}

/// File Search configuration (Gemini File Search tool)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchConfig {
    /// Names of File Search stores to use for retrieval
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub file_search_store_names: Vec<String>,
}
