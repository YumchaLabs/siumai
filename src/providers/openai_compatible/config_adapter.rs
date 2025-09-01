//! Configuration-driven Provider Adapter
//!
//! This module provides a configuration-driven adapter system that allows
//! defining provider-specific behavior through configuration files or structs,
//! similar to Cherry Studio's approach but integrated with our existing systems.

use super::adapter::{ProviderAdapter, ProviderCompatibility};
use super::types::{FieldMappings, ModelConfig, RequestType};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::HttpConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration-driven adapter that can be loaded from files or created programmatically
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigurableAdapter {
    /// Provider identifier
    pub provider_id: String,
    /// Provider display name
    pub name: String,
    /// Base URL for the provider
    pub base_url: String,
    /// Compatibility configuration
    pub compatibility: ProviderCompatibility,
    /// Field mappings for response parsing
    pub field_mappings: EnhancedFieldMappings,
    /// Parameter transformation rules
    pub parameter_rules: ParameterTransformRules,
    /// Model-specific configurations
    pub model_configs: HashMap<String, ModelSpecificConfig>,
    /// Custom headers to add to requests
    pub custom_headers: HashMap<String, String>,
    /// HTTP configuration overrides
    pub http_overrides: HttpConfigOverrides,
}

/// Enhanced field mappings with conditional support
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnhancedFieldMappings {
    /// Thinking content fields (in priority order)
    pub thinking_fields: Vec<String>,
    /// Regular content field
    pub content_field: String,
    /// Tool calls field
    pub tool_calls_field: String,
    /// Role field
    pub role_field: String,
    /// Custom field mappings
    pub custom_mappings: HashMap<String, String>,
    /// Conditional mappings based on model or other criteria
    pub conditional_mappings: Vec<ConditionalMapping>,
}

/// Conditional field mapping
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConditionalMapping {
    /// Condition to check
    pub condition: MappingCondition,
    /// Field mappings to apply when condition is met
    pub mappings: HashMap<String, String>,
}

/// Mapping condition types
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum MappingCondition {
    /// Model name contains string
    ModelContains { value: String },
    /// Model name equals string
    ModelEquals { value: String },
    /// Provider equals string
    ProviderEquals { value: String },
    /// Parameter exists in request
    ParameterExists { parameter: String },
    /// Parameter equals value
    ParameterEquals { parameter: String, value: serde_json::Value },
}

/// Parameter transformation rules
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParameterTransformRules {
    /// Simple parameter name mappings
    pub simple_mappings: HashMap<String, String>,
    /// Conditional parameter transformations
    pub conditional_transformations: Vec<ConditionalTransformation>,
    /// Parameters to remove from requests
    pub remove_parameters: Vec<String>,
    /// Default parameters to add
    pub default_parameters: HashMap<String, serde_json::Value>,
}

/// Conditional parameter transformation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConditionalTransformation {
    /// Condition to check
    pub condition: MappingCondition,
    /// Transformations to apply
    pub transformations: Vec<ParameterTransformation>,
}

/// Parameter transformation types
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ParameterTransformation {
    /// Rename a parameter
    Rename { from: String, to: String },
    /// Transform parameter value
    Transform { parameter: String, transformer: ValueTransformer },
    /// Add a parameter with fixed value
    Add { key: String, value: serde_json::Value },
    /// Remove a parameter
    Remove { parameter: String },
}

/// Value transformation types
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ValueTransformer {
    /// Multiply numeric value
    Multiply { factor: f64 },
    /// Map string values
    StringMap { mappings: HashMap<String, String> },
    /// Convert to boolean
    ToBoolean,
    /// Convert to string
    ToString,
}

/// Model-specific configuration overrides
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSpecificConfig {
    /// Force streaming for this model
    pub force_streaming: Option<bool>,
    /// Whether model supports thinking
    pub supports_thinking: Option<bool>,
    /// Maximum tokens for this model
    pub max_tokens: Option<u32>,
    /// Parameter overrides for this model
    pub parameter_overrides: Option<HashMap<String, serde_json::Value>>,
    /// Field mapping overrides for this model
    pub field_mapping_overrides: Option<HashMap<String, String>>,
}

/// HTTP configuration overrides
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HttpConfigOverrides {
    /// Additional timeout for this provider
    pub additional_timeout_secs: Option<u64>,
    /// Custom user agent
    pub user_agent: Option<String>,
    /// Additional headers
    pub headers: HashMap<String, String>,
}

impl Default for EnhancedFieldMappings {
    fn default() -> Self {
        Self {
            thinking_fields: vec!["thinking".to_string()],
            content_field: "content".to_string(),
            tool_calls_field: "tool_calls".to_string(),
            role_field: "role".to_string(),
            custom_mappings: HashMap::new(),
            conditional_mappings: vec![],
        }
    }
}

impl Default for ParameterTransformRules {
    fn default() -> Self {
        Self {
            simple_mappings: HashMap::new(),
            conditional_transformations: vec![],
            remove_parameters: vec![],
            default_parameters: HashMap::new(),
        }
    }
}

impl Default for HttpConfigOverrides {
    fn default() -> Self {
        Self {
            additional_timeout_secs: None,
            user_agent: None,
            headers: HashMap::new(),
        }
    }
}

impl ConfigurableAdapter {
    /// Create a new configurable adapter
    pub fn new(provider_id: &str, name: &str, base_url: &str) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            name: name.to_string(),
            base_url: base_url.to_string(),
            compatibility: ProviderCompatibility::default(),
            field_mappings: EnhancedFieldMappings::default(),
            parameter_rules: ParameterTransformRules::default(),
            model_configs: HashMap::new(),
            custom_headers: HashMap::new(),
            http_overrides: HttpConfigOverrides::default(),
        }
    }

    /// Load adapter configuration from JSON file
    pub fn from_json_file(path: &std::path::Path) -> Result<Self, LlmError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to read config file: {}", e)))?;
        
        let adapter: Self = serde_json::from_str(&content)
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to parse config: {}", e)))?;
        
        Ok(adapter)
    }

    /// Save adapter configuration to JSON file
    pub fn to_json_file(&self, path: &std::path::Path) -> Result<(), LlmError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }

    /// Create a DeepSeek configuration
    pub fn deepseek() -> Self {
        let mut adapter = Self::new("deepseek", "DeepSeek", "https://api.deepseek.com/v1");

        adapter.compatibility = ProviderCompatibility::deepseek();
        adapter.field_mappings.thinking_fields = vec!["reasoning_content".to_string(), "thinking".to_string()];

        // Add conditional transformations for reasoning models
        let reasoner_condition = MappingCondition::ModelContains {
            value: "reasoner".to_string(),
        };

        // Clean up messages to remove reasoning_content from context
        adapter.parameter_rules.conditional_transformations.push(ConditionalTransformation {
            condition: reasoner_condition.clone(),
            transformations: vec![
                // Remove unsupported parameters for reasoning models
                ParameterTransformation::Remove { parameter: "logprobs".to_string() },
                ParameterTransformation::Remove { parameter: "top_logprobs".to_string() },
                // Set default max_tokens for reasoning models
                ParameterTransformation::Add {
                    key: "max_tokens".to_string(),
                    value: serde_json::Value::Number(serde_json::Number::from(32768))
                },
                // Force streaming for reasoning models
                ParameterTransformation::Add {
                    key: "stream".to_string(),
                    value: serde_json::Value::Bool(true)
                },
            ],
        });

        adapter
    }

    /// Create a SiliconFlow configuration
    pub fn siliconflow() -> Self {
        let mut adapter = Self::new("siliconflow", "SiliconFlow", "https://api.siliconflow.cn");
        
        // SiliconFlow has mixed compatibility depending on the model
        adapter.compatibility.supports_array_content = true;
        adapter.compatibility.supports_stream_options = true;
        
        // For DeepSeek models on SiliconFlow, use reasoning_content
        let deepseek_condition = MappingCondition::ModelContains {
            value: "deepseek".to_string(),
        };
        
        adapter.field_mappings.conditional_mappings.push(ConditionalMapping {
            condition: deepseek_condition.clone(),
            mappings: {
                let mut mappings = HashMap::new();
                mappings.insert("thinking".to_string(), "reasoning_content".to_string());
                mappings
            },
        });
        
        // Parameter transformation for DeepSeek models
        adapter.parameter_rules.conditional_transformations.push(ConditionalTransformation {
            condition: deepseek_condition,
            transformations: vec![
                ParameterTransformation::Rename {
                    from: "thinking_budget".to_string(),
                    to: "reasoning_effort".to_string(),
                },
            ],
        });
        
        adapter
    }

    /// Check if a condition is met
    fn check_condition(&self, condition: &MappingCondition, model: &str, params: &serde_json::Value) -> bool {
        match condition {
            MappingCondition::ModelContains { value } => model.contains(value),
            MappingCondition::ModelEquals { value } => model == value,
            MappingCondition::ProviderEquals { value } => self.provider_id == *value,
            MappingCondition::ParameterExists { parameter } => params.get(parameter).is_some(),
            MappingCondition::ParameterEquals { parameter, value } => {
                params.get(parameter).map_or(false, |v| v == value)
            }
        }
    }

    /// Apply parameter transformations
    fn apply_parameter_transformations(&self, params: &mut serde_json::Value, model: &str) -> Result<(), LlmError> {
        // Apply simple mappings
        for (from, to) in &self.parameter_rules.simple_mappings {
            if let Some(value) = params.get(from).cloned() {
                params.as_object_mut().unwrap().remove(from);
                params[to] = value;
            }
        }

        // Apply conditional transformations
        for conditional in &self.parameter_rules.conditional_transformations {
            if self.check_condition(&conditional.condition, model, params) {
                for transformation in &conditional.transformations {
                    self.apply_transformation(params, transformation)?;
                }
            }
        }

        // Remove parameters
        for param in &self.parameter_rules.remove_parameters {
            params.as_object_mut().unwrap().remove(param);
        }

        // Add default parameters
        for (key, value) in &self.parameter_rules.default_parameters {
            if !params.get(key).is_some() {
                params[key] = value.clone();
            }
        }

        Ok(())
    }

    /// Apply a single parameter transformation
    fn apply_transformation(&self, params: &mut serde_json::Value, transformation: &ParameterTransformation) -> Result<(), LlmError> {
        match transformation {
            ParameterTransformation::Rename { from, to } => {
                if let Some(value) = params.get(from).cloned() {
                    params.as_object_mut().unwrap().remove(from);
                    params[to] = value;
                }
            }
            ParameterTransformation::Transform { parameter, transformer } => {
                if let Some(value) = params.get_mut(parameter) {
                    *value = self.apply_value_transformer(value, transformer)?;
                }
            }
            ParameterTransformation::Add { key, value } => {
                params[key] = value.clone();
            }
            ParameterTransformation::Remove { parameter } => {
                params.as_object_mut().unwrap().remove(parameter);
            }
        }
        Ok(())
    }

    /// Apply value transformer
    fn apply_value_transformer(&self, value: &serde_json::Value, transformer: &ValueTransformer) -> Result<serde_json::Value, LlmError> {
        match transformer {
            ValueTransformer::Multiply { factor } => {
                if let Some(num) = value.as_f64() {
                    Ok(serde_json::Value::Number(serde_json::Number::from_f64(num * factor).unwrap()))
                } else {
                    Err(LlmError::ConfigurationError("Cannot multiply non-numeric value".to_string()))
                }
            }
            ValueTransformer::StringMap { mappings } => {
                if let Some(str_val) = value.as_str() {
                    Ok(serde_json::Value::String(
                        mappings.get(str_val).cloned().unwrap_or_else(|| str_val.to_string())
                    ))
                } else {
                    Ok(value.clone())
                }
            }
            ValueTransformer::ToBoolean => {
                Ok(serde_json::Value::Bool(match value {
                    serde_json::Value::Bool(b) => *b,
                    serde_json::Value::String(s) => s.to_lowercase() == "true",
                    serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
                    _ => false,
                }))
            }
            ValueTransformer::ToString => {
                Ok(serde_json::Value::String(match value {
                    serde_json::Value::String(s) => s.clone(),
                    _ => value.to_string(),
                }))
            }
        }
    }

    /// Get effective field mappings for a model
    fn get_effective_field_mappings(&self, model: &str) -> FieldMappings {
        // Convert to owned strings to avoid lifetime issues
        let thinking_fields: Vec<&'static str> = self.field_mappings.thinking_fields
            .iter()
            .map(|s| Box::leak(s.clone().into_boxed_str()) as &'static str)
            .collect();

        let mut mappings = FieldMappings {
            thinking_fields,
            content_field: Box::leak(self.field_mappings.content_field.clone().into_boxed_str()),
            tool_calls_field: Box::leak(self.field_mappings.tool_calls_field.clone().into_boxed_str()),
            role_field: Box::leak(self.field_mappings.role_field.clone().into_boxed_str()),
        };

        // Apply conditional mappings
        for conditional in &self.field_mappings.conditional_mappings {
            if self.check_condition(&conditional.condition, model, &serde_json::Value::Null) {
                // Apply the conditional mappings
                for (from, to) in &conditional.mappings {
                    match from.as_str() {
                        "thinking" => {
                            // Replace thinking fields with the mapped value
                            mappings.thinking_fields = vec![Box::leak(to.clone().into_boxed_str())];
                        }
                        "content" => mappings.content_field = Box::leak(to.clone().into_boxed_str()),
                        "tool_calls" => mappings.tool_calls_field = Box::leak(to.clone().into_boxed_str()),
                        "role" => mappings.role_field = Box::leak(to.clone().into_boxed_str()),
                        _ => {} // Custom mappings handled elsewhere
                    }
                }
            }
        }

        mappings
    }
}

impl ProviderAdapter for ConfigurableAdapter {
    fn provider_id(&self) -> &'static str {
        // Note: This is a limitation - we need to return a static str
        // In practice, you might want to use a different approach or
        // store provider IDs in a static registry
        Box::leak(self.provider_id.clone().into_boxed_str())
    }

    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        model: &str,
        _request_type: RequestType,
    ) -> Result<(), LlmError> {
        self.apply_parameter_transformations(params, model)
    }

    fn get_field_mappings(&self, model: &str) -> FieldMappings {
        self.get_effective_field_mappings(model)
    }

    fn get_model_config(&self, model: &str) -> ModelConfig {
        if let Some(model_config) = self.model_configs.get(model) {
            let mut config = ModelConfig::default();

            if let Some(force_streaming) = model_config.force_streaming {
                config.force_streaming = force_streaming;
            }

            if let Some(supports_thinking) = model_config.supports_thinking {
                config.supports_thinking = supports_thinking;
            }

            if let Some(max_tokens) = model_config.max_tokens {
                config.max_tokens = Some(max_tokens);
            }

            config
        } else {
            ModelConfig::default()
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Convert our compatibility config to ProviderCapabilities
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_custom_feature("array_content", self.compatibility.supports_array_content)
            .with_custom_feature("stream_options", self.compatibility.supports_stream_options)
            .with_custom_feature("developer_role", self.compatibility.supports_developer_role)
            .with_custom_feature("enable_thinking", self.compatibility.supports_enable_thinking)
            .with_custom_feature("service_tier", self.compatibility.supports_service_tier)
    }

    fn compatibility(&self) -> ProviderCompatibility {
        self.compatibility.clone()
    }

    fn apply_http_config(&self, mut http_config: HttpConfig) -> HttpConfig {
        // Apply HTTP overrides
        if let Some(additional_timeout) = self.http_overrides.additional_timeout_secs {
            if let Some(current_timeout) = http_config.timeout {
                http_config.timeout = Some(current_timeout + std::time::Duration::from_secs(additional_timeout));
            }
        }

        if let Some(user_agent) = &self.http_overrides.user_agent {
            http_config.user_agent = Some(user_agent.clone());
        }

        // Add custom headers
        for (key, value) in &self.http_overrides.headers {
            http_config.headers.insert(key.clone(), value.clone());
        }

        http_config
    }

    fn custom_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in &self.custom_headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                reqwest::header::HeaderValue::from_str(value),
            ) {
                headers.insert(name, val);
            }
        }
        headers
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }
}
