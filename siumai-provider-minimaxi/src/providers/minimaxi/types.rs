//! MiniMaxi-specific type definitions
//!
//! This module contains type definitions specific to MiniMaxi API.

use serde::{Deserialize, Serialize};

/// MiniMaxi API error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxiError {
    /// Error code
    pub code: Option<String>,
    /// Error message
    pub message: String,
    /// Error type
    #[serde(rename = "type")]
    pub error_type: Option<String>,
}

/// MiniMaxi API error wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxiErrorResponse {
    /// Error details
    pub error: MinimaxiError,
}
