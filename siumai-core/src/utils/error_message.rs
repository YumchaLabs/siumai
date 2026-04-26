//! AI SDK-style error message utility.

use std::fmt::Display;

/// Convert an optional displayable error-like value into a user-facing message.
///
/// This is the Rust equivalent of AI SDK `getErrorMessage`: missing values map
/// to `"unknown error"`, strings display as-is, `serde_json::Value` displays as
/// compact JSON, and Rust errors use their `Display` implementation.
pub fn get_error_message<E>(error: Option<&E>) -> String
where
    E: Display + ?Sized,
{
    error
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown error".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;

    #[test]
    fn gets_message_from_missing_or_string_values() {
        assert_eq!(get_error_message::<str>(None), "unknown error");
        assert_eq!(
            get_error_message(Some("something went wrong")),
            "something went wrong"
        );
        assert_eq!(get_error_message(Some("")), "");
    }

    #[test]
    fn gets_message_from_json_and_errors() {
        let json = serde_json::json!({ "code": "FAIL", "detail": "oops" });
        assert_eq!(
            get_error_message(Some(&json)),
            r#"{"code":"FAIL","detail":"oops"}"#
        );

        let error = LlmError::InvalidInput("bad input".to_string());
        assert_eq!(get_error_message(Some(&error)), "Invalid input: bad input");
    }
}
