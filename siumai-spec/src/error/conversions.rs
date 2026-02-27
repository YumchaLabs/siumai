//! Type Conversions for LlmError
//!
//! This module contains From trait implementations for converting
//! common error types into LlmError.

use super::types::LlmError;

// From implementations
impl From<serde_json::Error> for LlmError {
    fn from(err: serde_json::Error) -> Self {
        Self::JsonError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let llm_err: LlmError = json_err.into();
        assert!(matches!(llm_err, LlmError::JsonError(_)));
    }
}
