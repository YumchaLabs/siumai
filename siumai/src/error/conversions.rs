//! Re-export conversions for core LlmError
#[allow(unused_imports)]
pub use siumai_core::error::conversions::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;

    #[test]
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
