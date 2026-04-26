//! AI SDK-style type validation helpers.

use crate::types::{
    JSONValue, Schema, TypeValidationContext, TypeValidationError, ValidationResult,
};

/// Result returned by [`safe_validate_types`].
#[derive(Debug, Clone)]
pub enum TypeValidationResult<T = JSONValue> {
    /// Runtime schema validation succeeded.
    Success {
        /// Validated value.
        value: T,
        /// Raw input JSON value before validation.
        raw_value: JSONValue,
    },
    /// Runtime schema validation failed.
    Failure {
        /// AI SDK-style validation error carrier.
        error: TypeValidationError,
        /// Raw input JSON value.
        raw_value: JSONValue,
    },
}

impl<T> TypeValidationResult<T> {
    /// Whether validation succeeded.
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Whether validation failed.
    pub const fn is_failure(&self) -> bool {
        matches!(self, Self::Failure { .. })
    }

    /// Convert into a standard `Result`, discarding `raw_value`.
    pub fn into_result(self) -> Result<T, TypeValidationError> {
        match self {
            Self::Success { value, .. } => Ok(value),
            Self::Failure { error, .. } => Err(error),
        }
    }
}

/// Validate a JSON value using a runtime schema validator.
///
/// AI SDK can cast an unvalidated `unknown` value to a TypeScript generic when no
/// validator exists. Rust cannot do that soundly, so schemas without a runtime
/// validator return `TypeValidationError` instead of pretending a typed value exists.
pub fn validate_types<T>(
    value: JSONValue,
    schema: &Schema<T>,
    context: Option<TypeValidationContext>,
) -> Result<T, TypeValidationError> {
    safe_validate_types(value, schema, context).into_result()
}

/// Safely validate a JSON value using a runtime schema validator.
pub fn safe_validate_types<T>(
    value: JSONValue,
    schema: &Schema<T>,
    context: Option<TypeValidationContext>,
) -> TypeValidationResult<T> {
    let raw_value = value.clone();

    match schema.validate(&value) {
        Some(ValidationResult::Success { value }) => {
            TypeValidationResult::Success { value, raw_value }
        }
        Some(ValidationResult::Failure { error }) => TypeValidationResult::Failure {
            error: type_validation_error(raw_value.clone(), error.to_string(), context),
            raw_value,
        },
        None => TypeValidationResult::Failure {
            error: type_validation_error(
                raw_value.clone(),
                "Schema does not define a runtime validator",
                context,
            ),
            raw_value,
        },
    }
}

fn type_validation_error(
    value: JSONValue,
    cause: impl Into<String>,
    context: Option<TypeValidationContext>,
) -> TypeValidationError {
    TypeValidationError::new(value, Some(JSONValue::String(cause.into())), context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;
    use crate::types::{json_schema, json_schema_with_validator};

    #[test]
    fn validate_types_returns_typed_value() {
        let schema =
            json_schema_with_validator(
                serde_json::json!({ "type": "object" }),
                |value| match value.get("answer").and_then(JSONValue::as_str) {
                    Some(answer) => ValidationResult::success(answer.to_string()),
                    None => ValidationResult::failure(LlmError::InvalidInput(
                        "missing answer".to_string(),
                    )),
                },
            );

        let value = validate_types(
            serde_json::json!({ "answer": "yes" }),
            &schema,
            Some(TypeValidationContext {
                field: Some("answer".to_string()),
                entity_name: Some("response".to_string()),
                entity_id: Some("test".to_string()),
            }),
        )
        .expect("valid value");

        assert_eq!(value, "yes");
    }

    #[test]
    fn safe_validate_types_preserves_raw_value_on_failure() {
        let schema =
            json_schema_with_validator(
                serde_json::json!({ "type": "object" }),
                |value| match value.get("answer").and_then(JSONValue::as_str) {
                    Some(answer) => ValidationResult::success(answer.to_string()),
                    None => ValidationResult::failure(LlmError::InvalidInput(
                        "missing answer".to_string(),
                    )),
                },
            );

        let result = safe_validate_types(serde_json::json!({ "answer": 42 }), &schema, None);

        let TypeValidationResult::Failure { error, raw_value } = result else {
            panic!("validation should fail");
        };
        assert_eq!(raw_value["answer"], serde_json::json!(42));
        assert!(error.message.contains("missing answer"));
    }

    #[test]
    fn schema_without_runtime_validator_is_an_explicit_failure() {
        let schema = json_schema(serde_json::json!({ "type": "object" }));
        let result = safe_validate_types(serde_json::json!({ "answer": "yes" }), &schema, None);

        assert!(result.is_failure());
    }
}
