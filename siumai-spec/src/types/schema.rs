//! Schema validation types and traits
//!
//! This module defines the core schema validation interface used by the orchestrator
//! for structured output validation. The actual validation implementation is provided
//! by `siumai-extras` with the `schema` feature.

use crate::error::LlmError;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, OnceLock};

use super::ai_sdk::{JSONSchema7, JSONValue};

/// Result returned by an optional runtime schema validator.
///
/// This mirrors AI SDK's `ValidationResult` union while keeping the Rust carrier explicit.
#[derive(Debug, Clone)]
pub enum ValidationResult<T = JSONValue> {
    /// Validation succeeded and produced a typed value.
    Success {
        /// Validated value.
        value: T,
    },
    /// Validation failed.
    Failure {
        /// Validation error.
        error: LlmError,
    },
}

impl<T> ValidationResult<T> {
    /// Create a successful validation result.
    pub fn success(value: T) -> Self {
        Self::Success { value }
    }

    /// Create a failed validation result.
    pub fn failure(error: LlmError) -> Self {
        Self::Failure { error }
    }

    /// Whether validation succeeded.
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Whether validation failed.
    pub const fn is_failure(&self) -> bool {
        matches!(self, Self::Failure { .. })
    }

    /// Convert into a standard `Result`.
    pub fn into_result(self) -> Result<T, LlmError> {
        match self {
            Self::Success { value } => Ok(value),
            Self::Failure { error } => Err(error),
        }
    }
}

type SchemaValidateFn<T> = dyn Fn(&JSONValue) -> ValidationResult<T> + Send + Sync + 'static;

/// AI SDK-style schema carrier.
///
/// The JSON Schema is always available for provider requests. Runtime validation is optional,
/// matching AI SDK's `validate?: ...` contract without pretending every schema has a validator.
pub struct Schema<T = JSONValue> {
    json_schema: JSONSchema7,
    validate: Option<Arc<SchemaValidateFn<T>>>,
}

impl<T> Clone for Schema<T> {
    fn clone(&self) -> Self {
        Self {
            json_schema: self.json_schema.clone(),
            validate: self.validate.clone(),
        }
    }
}

impl<T> fmt::Debug for Schema<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Schema")
            .field("json_schema", &self.json_schema)
            .field("has_validate", &self.validate.is_some())
            .finish()
    }
}

impl<T> Schema<T> {
    /// Create a schema from a JSON Schema value.
    pub fn new(json_schema: JSONSchema7) -> Self {
        Self {
            json_schema,
            validate: None,
        }
    }

    /// Create a schema from a JSON Schema value and an optional validator.
    pub fn with_validator<F>(json_schema: JSONSchema7, validate: F) -> Self
    where
        F: Fn(&JSONValue) -> ValidationResult<T> + Send + Sync + 'static,
    {
        Self {
            json_schema,
            validate: Some(Arc::new(validate)),
        }
    }

    /// Return the JSON Schema passed to providers.
    pub const fn json_schema(&self) -> &JSONSchema7 {
        &self.json_schema
    }

    /// Clone the JSON Schema value.
    pub fn into_json_schema(self) -> JSONSchema7 {
        self.json_schema
    }

    /// Whether this schema has a runtime validator.
    pub const fn has_validator(&self) -> bool {
        self.validate.is_some()
    }

    /// Validate a JSON value when a runtime validator is present.
    pub fn validate(&self, value: &JSONValue) -> Option<ValidationResult<T>> {
        self.validate.as_ref().map(|validate| validate(value))
    }
}

/// Lazily-created schema with cached initialization.
///
/// This is the Rust equivalent of AI SDK `lazySchema`: expensive schema construction can be
/// deferred until the first caller needs the concrete schema.
pub struct LazySchema<T = JSONValue> {
    create_schema: Arc<dyn Fn() -> Schema<T> + Send + Sync + 'static>,
    schema: Arc<OnceLock<Schema<T>>>,
}

impl<T> Clone for LazySchema<T> {
    fn clone(&self) -> Self {
        Self {
            create_schema: self.create_schema.clone(),
            schema: self.schema.clone(),
        }
    }
}

impl<T> fmt::Debug for LazySchema<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazySchema")
            .field("initialized", &self.schema.get().is_some())
            .finish_non_exhaustive()
    }
}

impl<T> LazySchema<T> {
    /// Create a lazy schema factory.
    pub fn new<F>(create_schema: F) -> Self
    where
        F: Fn() -> Schema<T> + Send + Sync + 'static,
    {
        Self {
            create_schema: Arc::new(create_schema),
            schema: Arc::new(OnceLock::new()),
        }
    }

    /// Return the cached concrete schema, creating it on first use.
    pub fn schema(&self) -> &Schema<T> {
        self.schema.get_or_init(|| (self.create_schema)())
    }

    /// Clone the cached concrete schema, creating it on first use.
    pub fn into_schema(&self) -> Schema<T> {
        self.schema().clone()
    }

    /// Whether the schema has already been initialized.
    pub fn is_initialized(&self) -> bool {
        self.schema.get().is_some()
    }
}

/// Schema input accepted by `as_schema`.
#[derive(Clone)]
pub enum FlexibleSchema<T = JSONValue> {
    /// Concrete schema.
    Schema(Schema<T>),
    /// Lazily-created schema.
    Lazy(LazySchema<T>),
}

impl<T> fmt::Debug for FlexibleSchema<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(schema) => f.debug_tuple("Schema").field(schema).finish(),
            Self::Lazy(schema) => f.debug_tuple("Lazy").field(schema).finish(),
        }
    }
}

impl<T> FlexibleSchema<T> {
    /// Resolve into a concrete schema.
    pub fn into_schema(self) -> Schema<T> {
        match self {
            Self::Schema(schema) => schema,
            Self::Lazy(schema) => schema.into_schema(),
        }
    }
}

impl<T> From<Schema<T>> for FlexibleSchema<T> {
    fn from(value: Schema<T>) -> Self {
        Self::Schema(value)
    }
}

impl<T> From<LazySchema<T>> for FlexibleSchema<T> {
    fn from(value: LazySchema<T>) -> Self {
        Self::Lazy(value)
    }
}

impl From<JSONSchema7> for FlexibleSchema<JSONValue> {
    fn from(value: JSONSchema7) -> Self {
        Self::Schema(Schema::new(value))
    }
}

/// Create an AI SDK-style schema using JSON Schema.
pub fn json_schema(json_schema: JSONSchema7) -> Schema<JSONValue> {
    Schema::new(json_schema)
}

/// Create an AI SDK-style schema using JSON Schema and a runtime validator.
pub fn json_schema_with_validator<T, F>(json_schema: JSONSchema7, validate: F) -> Schema<T>
where
    F: Fn(&JSONValue) -> ValidationResult<T> + Send + Sync + 'static,
{
    Schema::with_validator(json_schema, validate)
}

/// Create a lazily initialized schema.
pub fn lazy_schema<T, F>(create_schema: F) -> LazySchema<T>
where
    F: Fn() -> Schema<T> + Send + Sync + 'static,
{
    LazySchema::new(create_schema)
}

/// Resolve a flexible schema into a concrete schema.
pub fn as_schema<T>(schema: impl Into<FlexibleSchema<T>>) -> Schema<T> {
    schema.into().into_schema()
}

/// Resolve an optional flexible schema, falling back to an empty object schema.
pub fn as_schema_or_empty<T>(schema: Option<impl Into<FlexibleSchema<T>>>) -> Schema<T> {
    match schema {
        Some(schema) => as_schema(schema),
        None => empty_json_schema(),
    }
}

/// Create the empty object schema used by AI SDK `asSchema(undefined)`.
pub fn empty_json_schema<T>() -> Schema<T> {
    Schema::new(serde_json::json!({
        "properties": {},
        "additionalProperties": false
    }))
}

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

    #[test]
    fn ai_sdk_schema_carrier_exposes_json_schema_and_optional_validator() {
        let schema = json_schema_with_validator(
            json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
            |value| {
                value
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(|name| ValidationResult::success(name.to_string()))
                    .unwrap_or_else(|| {
                        ValidationResult::failure(LlmError::ParseError(
                            "Expected a string name".to_string(),
                        ))
                    })
            },
        );

        assert!(schema.has_validator());
        assert_eq!(schema.json_schema()["type"], json!("object"));

        let result = schema
            .validate(&json!({ "name": "Ada" }))
            .expect("validator is present")
            .into_result()
            .expect("valid value");
        assert_eq!(result, "Ada");

        let error = schema
            .validate(&json!({ "name": 42 }))
            .expect("validator is present")
            .into_result()
            .expect_err("invalid value");
        assert!(error.to_string().contains("Expected a string name"));
    }

    #[test]
    fn lazy_schema_is_cached_and_resolves_through_as_schema() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let lazy = {
            let calls = calls.clone();
            lazy_schema(move || {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                json_schema(json!({ "type": "string" }))
            })
        };

        assert!(!lazy.is_initialized());
        assert_eq!(lazy.schema().json_schema()["type"], json!("string"));
        assert!(lazy.is_initialized());

        let resolved = as_schema(lazy.clone());
        assert_eq!(resolved.json_schema()["type"], json!("string"));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn as_schema_or_empty_matches_ai_sdk_undefined_fallback() {
        let schema = as_schema_or_empty::<JSONValue>(None::<Schema>);
        assert_eq!(schema.json_schema()["properties"], json!({}));
        assert_eq!(schema.json_schema()["additionalProperties"], json!(false));

        let direct = as_schema(json!({ "type": "number" }));
        assert_eq!(direct.json_schema()["type"], json!("number"));
    }
}
