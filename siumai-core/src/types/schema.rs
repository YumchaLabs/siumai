//! Schema validation types and traits
//!
//! This module defines the core schema validation interface used by the orchestrator
//! for structured output validation. The actual validation implementation is provided
//! by `siumai-extras` with the `schema` feature.

use crate::error::LlmError;
use serde::{Deserialize, Serialize};

/// A trait for validating JSON values against a schema.
///
/// This trait provides a common interface for schema validation that can be
/// implemented by different validation backends. The default implementation
/// is provided by `siumai-extras` using the `jsonschema` crate.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::SchemaValidator;
/// use serde_json::json;
///
/// struct MyValidator {
///     schema: serde_json::Value,
/// }
///
/// impl SchemaValidator for MyValidator {
///     fn validate(&self, instance: &serde_json::Value) -> Result<(), LlmError> {
///         // Custom validation logic
///         Ok(())
///     }
/// }
/// ```
pub trait SchemaValidator: Send + Sync {
    /// Validate a JSON value against the schema.
    ///
    /// # Arguments
    ///
    /// * `instance` - The JSON value to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` if validation succeeds
    /// * `Err(LlmError::ParseError)` if validation fails
    fn validate(&self, instance: &serde_json::Value) -> Result<(), LlmError>;

    /// Check if a JSON value is valid without returning error details.
    ///
    /// # Arguments
    ///
    /// * `instance` - The JSON value to validate
    ///
    /// # Returns
    ///
    /// * `true` if validation succeeds
    /// * `false` if validation fails
    fn is_valid(&self, instance: &serde_json::Value) -> bool {
        self.validate(instance).is_ok()
    }
}

/// Configuration for structured output with schema validation.
///
/// This struct holds the schema and optional validator for structured output.
/// When used with Agent, it enables automatic validation of the model's output
/// against the provided schema.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::OutputSchema;
/// use serde_json::json;
///
/// let schema = json!({
///     "type": "object",
///     "properties": {
///         "name": {"type": "string"},
///         "age": {"type": "number"}
///     },
///     "required": ["name"]
/// });
///
/// let output_schema = OutputSchema::new(schema)
///     .with_name("person_info")
///     .with_description("Person information");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSchema {
    /// The JSON schema for validation
    pub schema: serde_json::Value,
    /// Optional name for the schema (used by some providers)
    pub name: Option<String>,
    /// Optional description for the schema (used by some providers)
    pub description: Option<String>,
}

impl OutputSchema {
    /// Create a new output schema.
    ///
    /// # Arguments
    ///
    /// * `schema` - The JSON schema for validation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::OutputSchema;
    /// use serde_json::json;
    ///
    /// let schema = json!({"type": "string"});
    /// let output_schema = OutputSchema::new(schema);
    /// ```
    pub fn new(schema: serde_json::Value) -> Self {
        Self {
            schema,
            name: None,
            description: None,
        }
    }

    /// Set the schema name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for the schema
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output_schema = OutputSchema::new(schema)
    ///     .with_name("person_info");
    /// ```
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the schema description.
    ///
    /// # Arguments
    ///
    /// * `description` - The description for the schema
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output_schema = OutputSchema::new(schema)
    ///     .with_description("Person information");
    /// ```
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Validate a JSON value against this schema using a validator.
    ///
    /// # Arguments
    ///
    /// * `instance` - The JSON value to validate
    /// * `validator` - The validator to use
    ///
    /// # Returns
    ///
    /// * `Ok(())` if validation succeeds
    /// * `Err(LlmError)` if validation fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::OutputSchema;
    /// use serde_json::json;
    ///
    /// let output_schema = OutputSchema::new(schema);
    /// let value = json!({"name": "Alice", "age": 30});
    ///
    /// output_schema.validate(&value, &validator)?;
    /// ```
    pub fn validate(
        &self,
        instance: &serde_json::Value,
        validator: &dyn SchemaValidator,
    ) -> Result<(), LlmError> {
        validator.validate(instance)
    }

    /// Check if a JSON value is valid against this schema.
    ///
    /// # Arguments
    ///
    /// * `instance` - The JSON value to validate
    /// * `validator` - The validator to use
    ///
    /// # Returns
    ///
    /// * `true` if validation succeeds
    /// * `false` if validation fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output_schema = OutputSchema::new(schema);
    /// let value = json!({"name": "Alice"});
    ///
    /// if output_schema.is_valid(&value, &validator) {
    ///     println!("Valid!");
    /// }
    /// ```
    pub fn is_valid(&self, instance: &serde_json::Value, validator: &dyn SchemaValidator) -> bool {
        validator.is_valid(instance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Mock validator for testing
    struct MockValidator;

    impl SchemaValidator for MockValidator {
        fn validate(&self, instance: &serde_json::Value) -> Result<(), LlmError> {
            if instance.is_object() {
                Ok(())
            } else {
                Err(LlmError::ParseError("Expected object".to_string()))
            }
        }
    }

    #[test]
    fn test_output_schema_creation() {
        let schema = json!({"type": "object"});
        let output_schema = OutputSchema::new(schema.clone());

        assert_eq!(output_schema.schema, schema);
        assert_eq!(output_schema.name, None);
        assert_eq!(output_schema.description, None);
    }

    #[test]
    fn test_output_schema_with_name() {
        let schema = json!({"type": "object"});
        let output_schema = OutputSchema::new(schema).with_name("test_schema");

        assert_eq!(output_schema.name, Some("test_schema".to_string()));
    }

    #[test]
    fn test_output_schema_with_description() {
        let schema = json!({"type": "object"});
        let output_schema = OutputSchema::new(schema).with_description("Test description");

        assert_eq!(
            output_schema.description,
            Some("Test description".to_string())
        );
    }

    #[test]
    fn test_output_schema_validate() {
        let schema = json!({"type": "object"});
        let output_schema = OutputSchema::new(schema);
        let validator = MockValidator;

        let valid_value = json!({"name": "Alice"});
        assert!(output_schema.validate(&valid_value, &validator).is_ok());

        let invalid_value = json!("not an object");
        assert!(output_schema.validate(&invalid_value, &validator).is_err());
    }

    #[test]
    fn test_output_schema_is_valid() {
        let schema = json!({"type": "object"});
        let output_schema = OutputSchema::new(schema);
        let validator = MockValidator;

        let valid_value = json!({"name": "Alice"});
        assert!(output_schema.is_valid(&valid_value, &validator));

        let invalid_value = json!("not an object");
        assert!(!output_schema.is_valid(&invalid_value, &validator));
    }

    #[test]
    fn test_schema_validator_is_valid_default() {
        let validator = MockValidator;

        assert!(validator.is_valid(&json!({"key": "value"})));
        assert!(!validator.is_valid(&json!("string")));
    }
}
