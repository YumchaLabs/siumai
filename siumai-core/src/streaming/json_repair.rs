//! JSON repair utilities for streaming and non-streaming responses
//!
//! This module provides utilities for automatically repairing malformed JSON
//! in LLM responses (both streaming and non-streaming). When the `json-repair`
//! feature is enabled, it uses the `jsonrepair` crate which has a fast path
//! for valid JSON.

/// Parse JSON with automatic repair when json-repair feature is enabled
///
/// This is a drop-in replacement for `serde_json::from_str` that automatically
/// repairs malformed JSON when the `json-repair` feature is enabled.
///
/// This function is used in both streaming (SSE events) and non-streaming
/// (HTTP response body) scenarios.
///
/// # Fast Path
///
/// When `json-repair` is enabled, the `jsonrepair` crate first tries to parse
/// with `serde_json::from_str`. If the JSON is already valid, it returns
/// immediately with zero overhead. Only if parsing fails does it attempt repair.
///
/// # Examples
///
/// ```rust,ignore
/// use siumai::streaming::json_repair::parse_json_with_repair;
///
/// // Valid JSON - fast path, zero overhead
/// let valid = r#"{"name":"John","age":30}"#;
/// let value: serde_json::Value = parse_json_with_repair(valid)?;
///
/// // Invalid JSON - automatically repaired (when json-repair feature enabled)
/// let invalid = r#"{name: 'John', age: 30,}"#;
/// let value: serde_json::Value = parse_json_with_repair(invalid)?;
/// ```
#[cfg(feature = "json-repair")]
#[doc(hidden)]
pub fn parse_json_with_repair<T: serde::de::DeserializeOwned>(
    json_str: &str,
) -> Result<T, serde_json::Error> {
    use jsonrepair::{Options, repair_json};

    // Try direct parse first (jsonrepair will also do this, but we can provide better errors)
    match serde_json::from_str::<T>(json_str) {
        Ok(val) => Ok(val),
        Err(original_err) => {
            // Try to repair the JSON
            let opts = Options::default();
            match repair_json(json_str, &opts) {
                Ok(repaired) => {
                    // Parse the repaired JSON
                    match serde_json::from_str(&repaired) {
                        Ok(val) => {
                            tracing::debug!(
                                "JSON repaired successfully:\nOriginal: {}\nRepaired: {}",
                                json_str,
                                repaired
                            );
                            Ok(val)
                        }
                        Err(_) => {
                            // If repair succeeded but parsing still fails, return original error
                            tracing::warn!(
                                "JSON repair succeeded but parsing failed:\nOriginal: {}\nRepaired: {}",
                                json_str,
                                repaired
                            );
                            Err(original_err)
                        }
                    }
                }
                Err(repair_err) => {
                    // Repair failed, return original error
                    tracing::debug!("JSON repair failed: {}", repair_err);
                    Err(original_err)
                }
            }
        }
    }
}

/// Parse JSON without repair (when json-repair feature is disabled)
///
/// This is a simple wrapper around `serde_json::from_str` for consistency.
#[cfg(not(feature = "json-repair"))]
#[inline]
#[doc(hidden)]
pub fn parse_json_with_repair<T: serde::de::DeserializeOwned>(
    json_str: &str,
) -> Result<T, serde_json::Error> {
    serde_json::from_str(json_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_json() {
        let valid = r#"{"name":"John","age":30}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(valid);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["name"], "John");
        assert_eq!(value["age"], 30);
    }

    #[test]
    #[cfg(feature = "json-repair")]
    fn test_parse_invalid_json_with_repair() {
        // Unquoted keys
        let invalid = r#"{name: "John", age: 30}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(invalid);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["name"], "John");
        assert_eq!(value["age"], 30);
    }

    #[test]
    #[cfg(feature = "json-repair")]
    fn test_parse_single_quotes() {
        let invalid = r#"{'name': 'John', 'age': 30}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(invalid);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["name"], "John");
        assert_eq!(value["age"], 30);
    }

    #[test]
    #[cfg(feature = "json-repair")]
    fn test_parse_trailing_comma() {
        let invalid = r#"{"name":"John","age":30,}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(invalid);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["name"], "John");
        assert_eq!(value["age"], 30);
    }

    #[test]
    #[cfg(feature = "json-repair")]
    fn test_parse_with_comments() {
        let invalid = r#"{"name":"John", /* comment */ "age":30}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(invalid);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["name"], "John");
        assert_eq!(value["age"], 30);
    }

    #[test]
    #[cfg(feature = "json-repair")]
    fn test_parse_markdown_fence() {
        let invalid = r#"```json
{"name":"John","age":30}
```"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(invalid);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["name"], "John");
        assert_eq!(value["age"], 30);
    }

    #[test]
    #[cfg(not(feature = "json-repair"))]
    fn test_parse_invalid_json_without_repair() {
        // Without json-repair feature, invalid JSON should fail
        let invalid = r#"{name: "John", age: 30}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_object() {
        let empty = r#"{}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(empty);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value.is_object());
        assert_eq!(value.as_object().unwrap().len(), 0);
    }

    #[test]
    fn test_parse_empty_array() {
        let empty = r#"[]"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(empty);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value.is_array());
        assert_eq!(value.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_parse_nested_object() {
        let nested = r#"{"user":{"name":"John","age":30}}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(nested);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["user"]["name"], "John");
        assert_eq!(value["user"]["age"], 30);
    }

    #[test]
    #[cfg(feature = "json-repair")]
    fn test_parse_nested_with_errors() {
        let invalid = r#"{user: {name: 'John', age: 30,}}"#;
        let result: Result<serde_json::Value, _> = parse_json_with_repair(invalid);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["user"]["name"], "John");
        assert_eq!(value["user"]["age"], 30);
    }
}
