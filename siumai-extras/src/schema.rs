//! JSON Schema validation utilities
//!
//! This module provides utilities for validating JSON values against JSON Schema.
//! It implements the `SchemaValidator` trait from `siumai` core library.
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai_extras::schema::validate_json;
//! use serde_json::json;
//!
//! let schema = json!({
//!     "type": "object",
//!     "properties": {
//!         "name": { "type": "string" },
//!         "age": { "type": "number" }
//!     },
//!     "required": ["name"]
//! });
//!
//! let value = json!({
//!     "name": "Alice",
//!     "age": 30
//! });
//!
//! validate_json(&schema, &value)?;
//! ```

use crate::error::{ExtrasError, Result};
use serde_json::Value;
use siumai::error::LlmError;
use siumai::types::SchemaValidator as SchemaValidatorTrait;

/// Validate a JSON value against a JSON Schema
///
/// ## Arguments
///
/// - `schema`: The JSON Schema to validate against
/// - `instance`: The JSON value to validate
///
/// ## Returns
///
/// - `Ok(())` if validation succeeds
/// - `Err(ExtrasError::SchemaValidation)` if validation fails with error details
/// - `Err(ExtrasError::SchemaCompilation)` if the schema itself is invalid
///
/// ## Example
///
/// ```rust,ignore
/// use siumai_extras::schema::validate_json;
/// use serde_json::json;
///
/// let schema = json!({ "type": "string" });
/// let value = json!("hello");
///
/// validate_json(&schema, &value)?;
/// ```
pub fn validate_json(schema: &Value, instance: &Value) -> Result<()> {
    if !schema.is_object() {
        return Ok(());
    }

    let compiled = jsonschema::validator_for(schema)
        .map_err(|e| ExtrasError::SchemaCompilation(format!("Invalid JSON Schema: {}", e)))?;

    if compiled.validate(instance).is_err() {
        let mut msgs = Vec::new();
        for err in compiled.iter_errors(instance) {
            msgs.push(format!("{} at {}", err, err.instance_path));
            if msgs.len() >= 3 {
                break;
            }
        }
        return Err(ExtrasError::SchemaValidation(msgs.join("; ")));
    }

    Ok(())
}

/// Validate a JSON value against a JSON Schema, returning detailed error messages
///
/// This is similar to `validate_json` but returns all validation errors instead of
/// stopping at the first 3.
///
/// ## Arguments
///
/// - `schema`: The JSON Schema to validate against
/// - `instance`: The JSON value to validate
///
/// ## Returns
///
/// - `Ok(())` if validation succeeds
/// - `Err(ExtrasError::SchemaValidation)` with all error messages if validation fails
pub fn validate_json_detailed(schema: &Value, instance: &Value) -> Result<()> {
    if !schema.is_object() {
        return Ok(());
    }

    let compiled = jsonschema::validator_for(schema)
        .map_err(|e| ExtrasError::SchemaCompilation(format!("Invalid JSON Schema: {}", e)))?;

    if compiled.validate(instance).is_err() {
        let msgs: Vec<String> = compiled
            .iter_errors(instance)
            .map(|err| format!("{} at {}", err, err.instance_path))
            .collect();

        if !msgs.is_empty() {
            return Err(ExtrasError::SchemaValidation(msgs.join("; ")));
        }
    }

    Ok(())
}

/// A reusable JSON Schema validator
///
/// This struct compiles a JSON Schema once and can be used to validate multiple instances.
/// It implements the `SchemaValidator` trait from `siumai` core library.
///
/// ## Example
///
/// ```rust,ignore
/// use siumai_extras::schema::JsonSchemaValidator;
/// use serde_json::json;
///
/// let schema = json!({ "type": "string" });
/// let validator = JsonSchemaValidator::new(&schema)?;
///
/// validator.validate(&json!("hello"))?;
/// validator.validate(&json!("world"))?;
/// ```
pub struct JsonSchemaValidator {
    validator: jsonschema::Validator,
}

impl JsonSchemaValidator {
    /// Create a new schema validator
    ///
    /// ## Arguments
    ///
    /// - `schema`: The JSON Schema to compile
    ///
    /// ## Returns
    ///
    /// - `Ok(JsonSchemaValidator)` if the schema is valid
    /// - `Err(ExtrasError::SchemaCompilation)` if the schema is invalid
    pub fn new(schema: &Value) -> Result<Self> {
        let validator = jsonschema::validator_for(schema)
            .map_err(|e| ExtrasError::SchemaCompilation(format!("Invalid JSON Schema: {}", e)))?;

        Ok(Self { validator })
    }

    /// Validate a JSON value and return all error messages
    ///
    /// ## Arguments
    ///
    /// - `instance`: The JSON value to validate
    ///
    /// ## Returns
    ///
    /// - `Ok(())` if validation succeeds
    /// - `Err(ExtrasError::SchemaValidation)` with all error messages if validation fails
    pub fn validate_detailed(&self, instance: &Value) -> Result<()> {
        if self.validator.validate(instance).is_err() {
            let msgs: Vec<String> = self
                .validator
                .iter_errors(instance)
                .map(|err| format!("{} at {}", err, err.instance_path))
                .collect();

            if !msgs.is_empty() {
                return Err(ExtrasError::SchemaValidation(msgs.join("; ")));
            }
        }

        Ok(())
    }
}

/// Implement the SchemaValidator trait from siumai core library
impl SchemaValidatorTrait for JsonSchemaValidator {
    fn validate(&self, instance: &Value) -> std::result::Result<(), LlmError> {
        if self.validator.validate(instance).is_err() {
            let mut msgs = Vec::new();
            for err in self.validator.iter_errors(instance) {
                msgs.push(format!("{} at {}", err, err.instance_path));
                if msgs.len() >= 3 {
                    break;
                }
            }
            return Err(LlmError::ParseError(msgs.join("; ")));
        }

        Ok(())
    }

    fn is_valid(&self, instance: &Value) -> bool {
        self.validator.validate(instance).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_json_success() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        let value = json!({ "name": "Alice" });
        assert!(validate_json(&schema, &value).is_ok());
    }

    #[test]
    fn test_validate_json_failure() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        let value = json!({ "name": 123 });
        assert!(validate_json(&schema, &value).is_err());
    }

    #[test]
    fn test_schema_validator() {
        let schema = json!({ "type": "string" });
        let validator = JsonSchemaValidator::new(&schema).unwrap();

        // Test using the trait method
        use siumai::types::SchemaValidator;
        assert!(validator.validate(&json!("hello")).is_ok());
        assert!(validator.validate(&json!(123)).is_err());
        assert!(validator.is_valid(&json!("world")));
        assert!(!validator.is_valid(&json!(456)));
    }
}
