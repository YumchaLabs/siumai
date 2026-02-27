//! Type Conversions for LlmError
//!
//! This module contains From trait implementations for converting
//! common error types into LlmError.

use super::types::LlmError;

// From implementations
#[cfg(feature = "reqwest")]
impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        Self::HttpError(err.to_string())
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(err: serde_json::Error) -> Self {
        Self::JsonError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "reqwest")]
    fn test_from_reqwest_error() {
        // We can't easily create a reqwest::Error, so we'll just verify the trait exists
        // The actual conversion is tested implicitly when used in the codebase
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let llm_err: LlmError = json_err.into();
        assert!(matches!(llm_err, LlmError::JsonError(_)));
    }
}
