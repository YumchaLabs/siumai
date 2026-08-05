//! MiniMax-specific type definitions
//!
//! This module contains type definitions specific to MiniMax API.

use serde::{Deserialize, Serialize};

/// MiniMax API error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxError {
    /// Error code
    pub code: Option<String>,
    /// Error message
    pub message: String,
    /// Error type
    #[serde(rename = "type")]
    pub error_type: Option<String>,
}

/// MiniMax API error wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxErrorResponse {
    /// Error details
    pub error: MinimaxError,
}
