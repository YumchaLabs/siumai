//! `OpenAI` Structured Output Support
//!
//! This module implements `OpenAI`'s structured output feature which ensures
//! the model's output conforms to a specified JSON schema.
//!
//! API Reference: <https://platform.openai.com/docs/guides/structured-outputs>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::ChatResponse;

/// Structured output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredOutputConfig {
    /// Whether structured output is enabled
    pub enabled: bool,
    /// JSON schema for the expected output
    pub schema: Option<serde_json::Value>,
    /// Response format configuration
    pub response_format: Option<ResponseFormat>,
    /// Whether to use strict mode
    pub strict: bool,
}

impl Default for StructuredOutputConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            schema: None,
            response_format: None,
            strict: true,
        }
    }
}

impl StructuredOutputConfig {
    /// Create a new structured output configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable structured output
    pub const fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable structured output
    pub const fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Set JSON schema
    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set response format
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Enable or disable strict mode
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Convert to request parameters
    pub fn to_request_params(&self) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        if self.enabled {
            if let Some(ref format) = self.response_format {
                params.insert("response_format".to_string(), format.to_json());
            } else if let Some(ref schema) = self.schema {
                // Default JSON object format with schema
                let format = ResponseFormat::JsonObject {
                    schema: schema.clone(),
                    strict: self.strict,
                };
                params.insert("response_format".to_string(), format.to_json());
            }
        }

        params
    }
}

/// Response format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFormat {
    /// Text format (default)
    Text,
    /// JSON object format
    JsonObject {
        /// JSON schema
        schema: serde_json::Value,
        /// Whether to use strict mode
        strict: bool,
    },
    /// JSON schema format (legacy)
    JsonSchema {
        /// Schema name
        name: String,
        /// JSON schema
        schema: serde_json::Value,
        /// Whether to use strict mode
        strict: bool,
    },
}

impl ResponseFormat {
    /// Create a JSON object format
    pub const fn json_object(schema: serde_json::Value) -> Self {
        Self::JsonObject {
            schema,
            strict: true,
        }
    }

    /// Create a JSON schema format
    pub fn json_schema<S: Into<String>>(name: S, schema: serde_json::Value) -> Self {
        Self::JsonSchema {
            name: name.into(),
            schema,
            strict: true,
        }
    }

    /// Convert to JSON for API requests
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Text => serde_json::json!({
                "type": "text"
            }),
            Self::JsonObject { schema, strict } => serde_json::json!({
                "type": "json_object",
                "json_schema": {
                    "schema": schema,
                    "strict": strict
                }
            }),
            Self::JsonSchema {
                name,
                schema,
                strict,
            } => serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": strict
                }
            }),
        }
    }
}

/// Structured output validator
pub struct StructuredOutputValidator;

impl StructuredOutputValidator {
    /// Validate response against schema
    pub fn validate_response(
        response: &ChatResponse,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        // Extract JSON content from response
        let content = match &response.content {
            crate::types::MessageContent::Text(text) => text,
            crate::types::MessageContent::MultiModal(parts) => {
                // Find the first text part
                for part in parts {
                    if let crate::types::ContentPart::Text { text, .. } = part {
                        return Self::validate_json_string(text, schema);
                    }
                }
                return Err(LlmError::ParseError(
                    "No text content found in multimodal response".to_string(),
                ));
            }
            #[cfg(feature = "structured-messages")]
            crate::types::MessageContent::Json(v) => {
                let s =
                    serde_json::to_string(v).map_err(|e| LlmError::ParseError(e.to_string()))?;
                return Self::validate_json_string(&s, schema);
            }
        };

        Self::validate_json_string(content, schema)
    }

    /// Validate JSON string against schema
    fn validate_json_string(
        json_str: &str,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        // Parse JSON
        let parsed_json: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| LlmError::ParseError(format!("Invalid JSON: {e}")))?;

        // Basic schema validation (in a real implementation, you'd use a proper JSON schema validator)
        Self::basic_schema_validation(&parsed_json, schema)?;

        Ok(parsed_json)
    }

    /// Basic schema validation
    fn basic_schema_validation(
        value: &serde_json::Value,
        schema: &serde_json::Value,
    ) -> Result<(), LlmError> {
        // This is a simplified validation - in production, use a proper JSON schema library
        if let Some(schema_type) = schema.get("type").and_then(|t| t.as_str()) {
            match schema_type {
                "object" => {
                    if !value.is_object() {
                        return Err(LlmError::ParseError("Expected object type".to_string()));
                    }

                    // Check required properties
                    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                        let obj = value.as_object().unwrap();
                        for req_prop in required {
                            if let Some(prop_name) = req_prop.as_str()
                                && !obj.contains_key(prop_name)
                            {
                                return Err(LlmError::ParseError(format!(
                                    "Missing required property: {prop_name}"
                                )));
                            }
                        }
                    }
                }
                "array" => {
                    if !value.is_array() {
                        return Err(LlmError::ParseError("Expected array type".to_string()));
                    }
                }
                "string" => {
                    if !value.is_string() {
                        return Err(LlmError::ParseError("Expected string type".to_string()));
                    }
                }
                "number" => {
                    if !value.is_number() {
                        return Err(LlmError::ParseError("Expected number type".to_string()));
                    }
                }
                "boolean" => {
                    if !value.is_boolean() {
                        return Err(LlmError::ParseError("Expected boolean type".to_string()));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Schema builder for common patterns
pub struct SchemaBuilder;

impl SchemaBuilder {
    /// Create a simple object schema
    pub fn object(properties: HashMap<String, serde_json::Value>) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": properties,
            "additionalProperties": false
        })
    }

    /// Create an object schema with required fields
    pub fn object_with_required(
        properties: HashMap<String, serde_json::Value>,
        required: Vec<String>,
    ) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        })
    }

    /// Create an array schema
    pub fn array(items: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "type": "array",
            "items": items
        })
    }

    /// Create a string schema
    pub fn string() -> serde_json::Value {
        serde_json::json!({
            "type": "string"
        })
    }

    /// Create a string schema with enum values
    pub fn string_enum(values: Vec<String>) -> serde_json::Value {
        serde_json::json!({
            "type": "string",
            "enum": values
        })
    }

    /// Create a number schema
    pub fn number() -> serde_json::Value {
        serde_json::json!({
            "type": "number"
        })
    }

    /// Create an integer schema
    pub fn integer() -> serde_json::Value {
        serde_json::json!({
            "type": "integer"
        })
    }

    /// Create a boolean schema
    pub fn boolean() -> serde_json::Value {
        serde_json::json!({
            "type": "boolean"
        })
    }
}

/// Helper functions for common structured output patterns
pub mod patterns {
    use super::*;

    /// Create a classification response schema
    pub fn classification_schema(categories: Vec<String>) -> serde_json::Value {
        let mut properties = HashMap::new();
        properties.insert(
            "category".to_string(),
            SchemaBuilder::string_enum(categories),
        );
        properties.insert("confidence".to_string(), SchemaBuilder::number());
        properties.insert("reasoning".to_string(), SchemaBuilder::string());

        SchemaBuilder::object_with_required(
            properties,
            vec!["category".to_string(), "confidence".to_string()],
        )
    }

    /// Create an extraction response schema
    pub fn extraction_schema(fields: HashMap<String, serde_json::Value>) -> serde_json::Value {
        SchemaBuilder::object(fields)
    }

    /// Create a summary response schema
    pub fn summary_schema() -> serde_json::Value {
        let mut properties = HashMap::new();
        properties.insert("summary".to_string(), SchemaBuilder::string());
        properties.insert(
            "key_points".to_string(),
            SchemaBuilder::array(SchemaBuilder::string()),
        );
        properties.insert("word_count".to_string(), SchemaBuilder::integer());

        SchemaBuilder::object_with_required(
            properties,
            vec!["summary".to_string(), "key_points".to_string()],
        )
    }

    /// Create a Q&A response schema
    pub fn qa_schema() -> serde_json::Value {
        let mut properties = HashMap::new();
        properties.insert("answer".to_string(), SchemaBuilder::string());
        properties.insert("confidence".to_string(), SchemaBuilder::number());
        properties.insert(
            "sources".to_string(),
            SchemaBuilder::array(SchemaBuilder::string()),
        );

        SchemaBuilder::object_with_required(properties, vec!["answer".to_string()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_output_config() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });

        let config = StructuredOutputConfig::new()
            .enable()
            .with_schema(schema.clone())
            .with_strict(true);

        assert!(config.enabled);
        assert_eq!(config.schema, Some(schema));
        assert!(config.strict);
    }

    #[test]
    fn test_response_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            }
        });

        let format = ResponseFormat::json_object(schema.clone());
        let json = format.to_json();

        assert_eq!(json["type"], "json_object");
        assert_eq!(json["json_schema"]["schema"], schema);
        assert_eq!(json["json_schema"]["strict"], true);
    }

    #[test]
    fn test_schema_builder() {
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), SchemaBuilder::string());
        properties.insert("age".to_string(), SchemaBuilder::integer());

        let schema = SchemaBuilder::object_with_required(properties, vec!["name".to_string()]);

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["name"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::Value::String("name".to_string()))
        );
    }

    #[test]
    fn test_classification_pattern() {
        let categories = vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ];
        let schema = patterns::classification_schema(categories);

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["category"]["enum"].is_array());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::Value::String("category".to_string()))
        );
    }
}
