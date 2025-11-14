//! Core-level finish reason abstraction (provider-agnostic)

/// Minimal finish reasons used by standards and providers.
///
/// This enum is intentionally small; aggregation crates can map it to
/// richer, provider-specific enums (e.g. `crate::types::FinishReason`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReasonCore {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Other(String),
}

impl FinishReasonCore {
    /// Parse a provider-specific finish reason string into the core enum.
    ///
    /// This helper is shared by multiple standards/providers (OpenAI, Groq,
    /// xAI, etc.) to keep string-to-enum mapping consistent.
    pub fn from_str(reason: Option<&str>) -> Option<Self> {
        match reason {
            Some("stop") => Some(Self::Stop),
            Some("length") => Some(Self::Length),
            Some("tool_calls") | Some("function_call") => Some(Self::ToolCalls),
            Some("content_filter") => Some(Self::ContentFilter),
            Some(other) => Some(Self::Other(other.to_string())),
            None => None,
        }
    }
}
