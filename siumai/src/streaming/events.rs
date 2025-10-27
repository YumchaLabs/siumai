//! Streaming Event Types
//!
//! Defines delta types for incremental streaming updates.

use crate::types::ToolType;
use serde::{Deserialize, Serialize};

/// Tool Call Delta
///
/// Represents an incremental update to a tool call during streaming.
/// Tool calls are built up incrementally as the model generates them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Call ID
    pub id: Option<String>,
    /// Tool type
    pub r#type: Option<ToolType>,
    /// Function call delta
    pub function: Option<FunctionCallDelta>,
}

/// Function Call Delta
///
/// Represents an incremental update to a function call during streaming.
/// The function name and arguments are built up piece by piece.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name delta
    pub name: Option<String>,
    /// Arguments delta (JSON string fragment)
    pub arguments: Option<String>,
}
