//! Model information and capabilities
//!
//! This module defines provider-agnostic model information structures used across
//! the library. Provider-specific model catalogs and constants live in the
//! crate-level `model_catalog` module and are re-exported from `siumai::models`
//! and `siumai::constants`.

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model owner/organization
    pub owned_by: String,
    /// Creation timestamp
    pub created: Option<u64>,
    /// Model capabilities (high-level tags like "chat", "vision", "audio")
    pub capabilities: Vec<String>,
    /// Context window size
    pub context_window: Option<u32>,
    /// Maximum output tokens
    pub max_output_tokens: Option<u32>,
    /// Input cost per token
    pub input_cost_per_token: Option<f64>,
    /// Output cost per token
    pub output_cost_per_token: Option<f64>,
}
