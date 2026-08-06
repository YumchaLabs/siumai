//! Optional standards-based JSON Schema validation for structured output.
//!
//! Enable the `json-schema` feature to compile a schema once and reuse it through
//! [`JsonSchemaValidator`] or [`crate::OutputDescriptor`]. This adapter performs no network or
//! filesystem access: external `$ref` targets are rejected, while references bundled into the
//! schema document continue to work.
//!
//! ```rust
//! use serde::Deserialize;
//! use serde_json::json;
//! use siumai_runtime::{JsonSchemaValidator, OutputDescriptor};
//!
//! #[derive(Deserialize)]
//! struct Person {
//!     name: String,
//! }
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let schema = json!({
//!     "type": "object",
//!     "properties": { "name": { "type": "string" } },
//!     "required": ["name"],
//!     "additionalProperties": false
//! });
//! let validator = JsonSchemaValidator::new(&schema)?;
//! let descriptor = OutputDescriptor::<Person>::new("person", schema, validator)?;
//! # let _ = descriptor;
//! # Ok(())
//! # }
//! ```

use std::error::Error as StdError;
use std::fmt;

use jsonschema::{Retrieve, Uri};
use serde_json::Value;
use thiserror::Error;

use crate::{OutputSchemaValidator, SchemaValidationError};

const MAX_REPORTED_VIOLATIONS: usize = 8;
const MAX_DIAGNOSTIC_CHARS: usize = 512;

/// Compile a schema and validate one JSON instance.
///
/// Prefer [`JsonSchemaValidator`] when validating more than one instance so the schema is compiled
/// only once.
pub fn validate_json(schema: &Value, instance: &Value) -> Result<(), JsonSchemaError> {
    JsonSchemaValidator::new(schema)?
        .validate(instance)
        .map_err(JsonSchemaError::from)
}

/// An error returned while compiling or applying a JSON Schema.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JsonSchemaError {
    /// The schema could not be compiled.
    #[error(transparent)]
    Compilation(#[from] JsonSchemaCompilationError),
    /// The instance did not satisfy the compiled schema.
    #[error(transparent)]
    Validation(#[from] JsonSchemaValidationError),
}

/// A bounded, value-free JSON Schema compilation diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonSchemaCompilationError {
    schema_path: String,
    keyword: String,
}

impl JsonSchemaCompilationError {
    fn from_validator(error: &jsonschema::ValidationError<'_>) -> Self {
        Self {
            schema_path: bounded_path(error.instance_path()),
            keyword: bounded_text(error.kind().keyword()),
        }
    }

    /// JSON Pointer identifying the invalid schema location.
    ///
    /// An empty string denotes the schema document root.
    pub fn schema_path(&self) -> &str {
        &self.schema_path
    }

    /// JSON Schema keyword associated with the compilation failure.
    pub fn keyword(&self) -> &str {
        &self.keyword
    }
}

impl fmt::Display for JsonSchemaCompilationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid JSON Schema at ")?;
        write_path(formatter, &self.schema_path)?;
        write!(formatter, " (keyword {})", self.keyword)
    }
}

impl StdError for JsonSchemaCompilationError {}

/// One value-free JSON Schema violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonSchemaViolation {
    instance_path: String,
    schema_path: String,
    keyword: String,
}

impl JsonSchemaViolation {
    fn from_validator(error: &jsonschema::ValidationError<'_>) -> Self {
        Self {
            instance_path: bounded_path(error.instance_path()),
            schema_path: bounded_path(error.schema_path()),
            keyword: bounded_text(error.kind().keyword()),
        }
    }

    /// JSON Pointer identifying the invalid instance location.
    ///
    /// An empty string denotes the instance document root.
    pub fn instance_path(&self) -> &str {
        &self.instance_path
    }

    /// JSON Pointer identifying the violated schema constraint.
    ///
    /// An empty string denotes the schema document root.
    pub fn schema_path(&self) -> &str {
        &self.schema_path
    }

    /// JSON Schema keyword associated with the violation.
    pub fn keyword(&self) -> &str {
        &self.keyword
    }
}

impl fmt::Display for JsonSchemaViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("instance ")?;
        write_path(formatter, &self.instance_path)?;
        formatter.write_str(" violates ")?;
        write_path(formatter, &self.schema_path)?;
        write!(formatter, " ({})", self.keyword)
    }
}

/// Bounded diagnostics for an instance that did not satisfy a compiled schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonSchemaValidationError {
    violations: Vec<JsonSchemaViolation>,
    additional_violations_omitted: bool,
}

impl JsonSchemaValidationError {
    fn new(violations: Vec<JsonSchemaViolation>, additional_violations_omitted: bool) -> Self {
        debug_assert!(!violations.is_empty());
        Self {
            violations,
            additional_violations_omitted,
        }
    }

    /// Reported violations, capped at an implementation-defined bound.
    pub fn violations(&self) -> &[JsonSchemaViolation] {
        &self.violations
    }

    /// Whether the instance contained more violations than were reported.
    pub const fn additional_violations_omitted(&self) -> bool {
        self.additional_violations_omitted
    }
}

impl fmt::Display for JsonSchemaValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON Schema validation failed")?;

        if let Some(first) = self.violations.first() {
            write!(formatter, ": {first}")?;
        }

        let additional_reported = self.violations.len().saturating_sub(1);
        if additional_reported > 0 {
            write!(
                formatter,
                "; {additional_reported} additional violation(s) reported"
            )?;
        }
        if self.additional_violations_omitted {
            formatter.write_str("; additional violations omitted")?;
        }

        Ok(())
    }
}

impl StdError for JsonSchemaValidationError {}

/// A reusable validator for one JSON Schema document.
///
/// Construction compiles the schema eagerly and performs no external I/O. Bundle external
/// references into the schema document before construction.
pub struct JsonSchemaValidator {
    schema: Value,
    validator: jsonschema::Validator,
}

impl fmt::Debug for JsonSchemaValidator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsonSchemaValidator")
            .finish_non_exhaustive()
    }
}

impl JsonSchemaValidator {
    /// Compile a JSON Schema without resolving external resources.
    ///
    /// Object and boolean schemas are supported. Invalid schemas and unbundled external `$ref`
    /// targets return a bounded [`JsonSchemaCompilationError`].
    pub fn new(schema: &Value) -> Result<Self, JsonSchemaCompilationError> {
        let validator = jsonschema::options()
            .with_retriever(RejectExternalReferences)
            .build(schema)
            .map_err(|error| JsonSchemaCompilationError::from_validator(&error))?;

        Ok(Self {
            schema: schema.clone(),
            validator,
        })
    }

    /// Return the schema compiled by this validator.
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Return whether an instance satisfies the compiled schema without allocating diagnostics.
    pub fn is_valid(&self, instance: &Value) -> bool {
        self.validator.is_valid(instance)
    }

    /// Validate one JSON instance against the compiled schema.
    pub fn validate(&self, instance: &Value) -> Result<(), JsonSchemaValidationError> {
        let mut errors = self.validator.iter_errors(instance);
        let violations = errors
            .by_ref()
            .take(MAX_REPORTED_VIOLATIONS)
            .map(|error| JsonSchemaViolation::from_validator(&error))
            .collect::<Vec<_>>();

        if violations.is_empty() {
            return Ok(());
        }

        let additional_violations_omitted = errors.next().is_some();
        Err(JsonSchemaValidationError::new(
            violations,
            additional_violations_omitted,
        ))
    }

    fn ensure_same_schema(&self, schema: &Value) -> Result<(), SchemaValidationError> {
        if schema == &self.schema {
            Ok(())
        } else {
            Err(SchemaValidationError::new(
                "the output descriptor schema differs from the compiled JSON Schema validator",
            ))
        }
    }
}

impl OutputSchemaValidator for JsonSchemaValidator {
    fn validate_schema(&self, schema: &Value) -> Result<(), SchemaValidationError> {
        self.ensure_same_schema(schema)
    }

    fn validate(&self, schema: &Value, instance: &Value) -> Result<(), SchemaValidationError> {
        self.ensure_same_schema(schema)?;
        JsonSchemaValidator::validate(self, instance)
            .map_err(|error| SchemaValidationError::new(error.to_string()))
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("external JSON Schema references are disabled")]
struct ExternalReferencesDisabled;

#[derive(Debug, Clone, Copy)]
struct RejectExternalReferences;

impl Retrieve for RejectExternalReferences {
    fn retrieve(&self, _uri: &Uri<String>) -> Result<Value, Box<dyn StdError + Send + Sync>> {
        Err(Box::new(ExternalReferencesDisabled))
    }
}

fn bounded_path(path: impl fmt::Display) -> String {
    bounded_text(path.to_string())
}

fn bounded_text(text: impl AsRef<str>) -> String {
    let mut chars = text.as_ref().chars();
    let bounded = chars
        .by_ref()
        .take(MAX_DIAGNOSTIC_CHARS)
        .collect::<String>();
    if chars.next().is_some() {
        format!("{bounded}…")
    } else {
        bounded
    }
}

fn write_path(formatter: &mut fmt::Formatter<'_>, path: &str) -> fmt::Result {
    if path.is_empty() {
        formatter.write_str("the document root")
    } else {
        formatter.write_str(path)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{JsonSchemaError, JsonSchemaValidator, validate_json};
    use crate::OutputSchemaValidator;

    #[test]
    fn validates_object_and_boolean_schemas() {
        let schema = json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        });

        assert!(validate_json(&schema, &json!({ "name": "Ada" })).is_ok());
        assert!(matches!(
            validate_json(&schema, &json!({ "name": 42 })),
            Err(JsonSchemaError::Validation(_))
        ));
        assert!(validate_json(&json!(true), &json!({ "anything": true })).is_ok());
        assert!(matches!(
            validate_json(&json!(false), &json!(null)),
            Err(JsonSchemaError::Validation(_))
        ));
    }

    #[test]
    fn reports_value_free_bounded_violations() {
        let schema = json!({
            "type": "object",
            "properties": { "secret": { "type": "integer" } },
            "required": ["secret"]
        });
        let validator = JsonSchemaValidator::new(&schema).expect("schema should compile");

        let error = validator
            .validate(&json!({ "secret": "must-not-appear" }))
            .expect_err("instance should fail validation");

        assert_eq!(error.violations()[0].instance_path(), "/secret");
        assert_eq!(error.violations()[0].keyword(), "type");
        assert!(!error.to_string().contains("must-not-appear"));
        assert!(!format!("{error:?}").contains("must-not-appear"));
    }

    #[test]
    fn rejects_invalid_schemas_without_echoing_schema_values() {
        let schema = json!({ "type": "private-invalid-type" });

        let error = JsonSchemaValidator::new(&schema).expect_err("schema should be rejected");

        assert!(!error.to_string().contains("private-invalid-type"));
        assert!(!format!("{error:?}").contains("private-invalid-type"));
    }

    #[test]
    fn rejects_unbundled_external_references_without_fetching_them() {
        let schema = json!({ "$ref": "https://credentials.invalid/private-schema.json" });

        let error = JsonSchemaValidator::new(&schema).expect_err("external ref should be rejected");

        assert_eq!(error.keyword(), "$ref");
        assert!(!error.to_string().contains("credentials.invalid"));
        assert!(!format!("{error:?}").contains("credentials.invalid"));
    }

    #[test]
    fn supports_references_bundled_into_the_schema_document() {
        let schema = json!({
            "$defs": {
                "identifier": { "type": "integer" }
            },
            "$ref": "#/$defs/identifier"
        });
        let validator = JsonSchemaValidator::new(&schema).expect("schema should compile");

        assert!(validator.is_valid(&json!(42)));
        assert!(!validator.is_valid(&json!("42")));
    }

    #[test]
    fn runtime_adapter_rejects_schema_substitution() {
        let schema = json!({ "type": "integer" });
        let validator = JsonSchemaValidator::new(&schema).expect("schema should compile");

        let error = OutputSchemaValidator::validate(
            &validator,
            &json!({ "type": "string" }),
            &json!("value"),
        )
        .expect_err("a descriptor must not substitute another schema");

        assert!(error.message().contains("differs"));
    }

    #[test]
    fn debug_output_does_not_include_the_compiled_schema() {
        let validator = JsonSchemaValidator::new(&json!({
            "const": "must-not-appear"
        }))
        .expect("schema should compile");

        assert!(!format!("{validator:?}").contains("must-not-appear"));
    }
}
