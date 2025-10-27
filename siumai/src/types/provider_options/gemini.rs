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
