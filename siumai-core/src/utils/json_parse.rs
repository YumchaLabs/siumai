//! AI SDK-style JSON parsing helpers.

use crate::error::LlmError;
use crate::types::{JSONValue, Schema, ValidationResult};

/// Result returned by [`safe_parse_json`] and [`safe_parse_json_with_schema`].
#[derive(Debug, Clone)]
pub enum JsonParseResult<T = JSONValue> {
    /// JSON parsing and optional validation succeeded.
    Success {
        /// Parsed or validated value.
        value: T,
        /// Raw parsed JSON value before validation.
        raw_value: JSONValue,
    },
    /// JSON parsing or optional validation failed.
    Failure {
        /// Parse or validation error.
        error: LlmError,
        /// Raw parsed JSON value when parsing succeeded but validation failed.
        raw_value: Option<JSONValue>,
    },
}

impl<T> JsonParseResult<T> {
    /// Whether parsing succeeded.
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Whether parsing failed.
    pub const fn is_failure(&self) -> bool {
        matches!(self, Self::Failure { .. })
    }

    /// Convert into a standard `Result`, discarding `raw_value`.
    pub fn into_result(self) -> Result<T, LlmError> {
        match self {
            Self::Success { value, .. } => Ok(value),
            Self::Failure { error, .. } => Err(error),
        }
    }
}

/// Parse JSON text using AI SDK `parseJSON`-style security checks.
pub fn parse_json(text: &str) -> Result<JSONValue, LlmError> {
    let value = serde_json::from_str::<JSONValue>(text)
        .map_err(|error| LlmError::ParseError(format!("Failed to parse JSON: {error}")))?;
    reject_forbidden_prototype_properties(&value)?;
    Ok(value)
}

/// Parse and validate JSON text with a runtime schema validator.
pub fn parse_json_with_schema<T>(text: &str, schema: &Schema<T>) -> Result<T, LlmError> {
    let value = parse_json(text)?;
    validate_parsed_json(value, schema)
}

/// Safely parse JSON text, returning an explicit result union.
pub fn safe_parse_json(text: &str) -> JsonParseResult {
    match parse_json(text) {
        Ok(value) => JsonParseResult::Success {
            raw_value: value.clone(),
            value,
        },
        Err(error) => JsonParseResult::Failure {
            error,
            raw_value: None,
        },
    }
}

/// Safely parse and validate JSON text with a runtime schema validator.
pub fn safe_parse_json_with_schema<T>(text: &str, schema: &Schema<T>) -> JsonParseResult<T> {
    match parse_json(text) {
        Ok(value) => match validate_parsed_json(value.clone(), schema) {
            Ok(typed) => JsonParseResult::Success {
                value: typed,
                raw_value: value,
            },
            Err(error) => JsonParseResult::Failure {
                error,
                raw_value: Some(value),
            },
        },
        Err(error) => JsonParseResult::Failure {
            error,
            raw_value: None,
        },
    }
}

/// Return whether input can be parsed by [`parse_json`].
pub fn is_parsable_json(input: &str) -> bool {
    parse_json(input).is_ok()
}

fn validate_parsed_json<T>(value: JSONValue, schema: &Schema<T>) -> Result<T, LlmError> {
    match schema.validate(&value) {
        Some(ValidationResult::Success { value }) => Ok(value),
        Some(ValidationResult::Failure { error }) => Err(error),
        None => Err(LlmError::InvalidInput(
            "Schema does not define a runtime validator".to_string(),
        )),
    }
}

fn reject_forbidden_prototype_properties(value: &JSONValue) -> Result<(), LlmError> {
    match value {
        JSONValue::Object(object) => {
            if object.contains_key("__proto__") {
                return Err(forbidden_prototype_error());
            }

            if let Some(JSONValue::Object(constructor)) = object.get("constructor")
                && constructor.contains_key("prototype")
            {
                return Err(forbidden_prototype_error());
            }

            for value in object.values() {
                reject_forbidden_prototype_properties(value)?;
            }
        }
        JSONValue::Array(values) => {
            for value in values {
                reject_forbidden_prototype_properties(value)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn forbidden_prototype_error() -> LlmError {
    LlmError::ParseError("Object contains forbidden prototype property".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::json_schema_with_validator;

    #[test]
    fn parse_json_parses_values_and_safe_result() {
        let value = parse_json(r#"{"answer":42}"#).expect("parse JSON");
        assert_eq!(value["answer"], serde_json::json!(42));

        let safe = safe_parse_json(r#"{"answer":42}"#);
        assert!(safe.is_success());
        let JsonParseResult::Success { value, raw_value } = safe else {
            panic!("safe parse should succeed");
        };
        assert_eq!(value, raw_value);
    }

    #[test]
    fn parse_json_rejects_invalid_json_and_forbidden_keys() {
        assert!(matches!(
            parse_json("{not json}"),
            Err(LlmError::ParseError(_))
        ));
        assert!(matches!(
            parse_json(r#"{"__proto__":{"polluted":true}}"#),
            Err(LlmError::ParseError(_))
        ));
        assert!(matches!(
            parse_json(r#"{"constructor":{"prototype":{"polluted":true}}}"#),
            Err(LlmError::ParseError(_))
        ));
        assert!(parse_json(r#"{"constructor":null}"#).is_ok());
    }

    #[test]
    fn is_parsable_json_uses_secure_parse_rules() {
        assert!(is_parsable_json(r#"{"ok":true}"#));
        assert!(!is_parsable_json("{not json}"));
        assert!(!is_parsable_json(r#"{"__proto__":{}}"#));
    }

    #[test]
    fn parse_json_with_schema_uses_runtime_validator() {
        let schema =
            json_schema_with_validator(
                serde_json::json!({ "type": "object" }),
                |value| match value.get("answer").and_then(JSONValue::as_str) {
                    Some(answer) => ValidationResult::success(answer.to_string()),
                    None => ValidationResult::failure(LlmError::ParseError(
                        "missing answer".to_string(),
                    )),
                },
            );

        assert_eq!(
            parse_json_with_schema(r#"{"answer":"yes"}"#, &schema).expect("validated JSON"),
            "yes"
        );

        let failed = safe_parse_json_with_schema(r#"{"answer":42}"#, &schema);
        let JsonParseResult::Failure { raw_value, .. } = failed else {
            panic!("schema validation should fail");
        };
        assert_eq!(
            raw_value.expect("raw value")["answer"],
            serde_json::json!(42)
        );
    }
}
