use super::message::{ContentPart, Message, MessageValidationError};
use super::opaque::{OpaqueProviderBudget, OpaqueProviderItemError};
use crate::annotations::{
    ProviderAnnotationBudget, ProviderAnnotationError, ProviderAnnotationUsage,
};
use crate::tool::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use thiserror::Error;

/// Aggregate limits applied while validating one language request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageRequestBudget {
    opaque_provider_items: OpaqueProviderBudget,
    provider_annotations: ProviderAnnotationBudget,
}

impl LanguageRequestBudget {
    pub const fn new(
        opaque_provider_items: OpaqueProviderBudget,
        provider_annotations: ProviderAnnotationBudget,
    ) -> Self {
        Self {
            opaque_provider_items,
            provider_annotations,
        }
    }

    pub const fn opaque_provider_items(self) -> OpaqueProviderBudget {
        self.opaque_provider_items
    }

    pub const fn provider_annotations(self) -> ProviderAnnotationBudget {
        self.provider_annotations
    }
}

impl Default for LanguageRequestBudget {
    fn default() -> Self {
        Self::new(
            OpaqueProviderBudget::default(),
            ProviderAnnotationBudget::default(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub generation: GenerationConfig,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<StructuredOutputSpec>,
}

impl LanguageRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            generation: GenerationConfig::default(),
            tools: Vec::new(),
            tool_choice: None,
            structured_output: None,
        }
    }

    pub fn with_generation(mut self, generation: GenerationConfig) -> Self {
        self.generation = generation;
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Validate cross-field invariants before provider policy and encoding.
    pub fn validate(&self) -> Result<(), LanguageRequestError> {
        self.validate_with_budget(LanguageRequestBudget::default())
    }

    /// Validate with explicit aggregate budgets for retained provider state.
    pub fn validate_with_budget(
        &self,
        budget: LanguageRequestBudget,
    ) -> Result<(), LanguageRequestError> {
        self.generation.validate()?;
        for (message_index, message) in self.messages.iter().enumerate() {
            message
                .validate()
                .map_err(|source| LanguageRequestError::InvalidMessage {
                    message_index,
                    source,
                })?;
        }
        let mut names = BTreeSet::new();
        for tool in &self.tools {
            if !names.insert(tool.name()) {
                return Err(LanguageRequestError::DuplicateTool {
                    name: tool.name().to_string(),
                });
            }
        }
        if let Some(ToolChoice::Named { name }) = &self.tool_choice
            && !names.contains(name.as_str())
        {
            return Err(LanguageRequestError::UnknownToolChoice { name: name.clone() });
        }
        if let Some(output) = &self.structured_output {
            if output.name.trim().is_empty() {
                return Err(LanguageRequestError::EmptyStructuredOutputName);
            }
            if !output.schema.is_object() && !output.schema.is_boolean() {
                return Err(LanguageRequestError::InvalidStructuredOutputSchema);
            }
        }
        budget
            .opaque_provider_items
            .validate(self.messages.iter().flat_map(|message| {
                message
                    .content()
                    .iter()
                    .filter_map(|part| match part.content() {
                        ContentPart::ProviderOpaque(item) => Some(item),
                        _ => None,
                    })
            }))?;

        let mut annotation_usage = ProviderAnnotationUsage::default();
        for message in &self.messages {
            budget
                .provider_annotations
                .validate_node(message.annotations(), &mut annotation_usage)?;
            for part in message.content() {
                budget
                    .provider_annotations
                    .validate_node(part.annotations(), &mut annotation_usage)?;
            }
        }
        for tool in &self.tools {
            budget
                .provider_annotations
                .validate_node(tool.annotations(), &mut annotation_usage)?;
        }
        Ok(())
    }
}

/// Provider-neutral input accepted by ergonomic language entry points.
///
/// Inputs normalize immediately and losslessly into a [`LanguageRequest`].
/// This adapter does not validate the request or retain model, routing, call
/// option, or wire state.
#[derive(Debug, Clone)]
pub struct LanguageInput(LanguageRequest);

impl LanguageInput {
    /// Consume this input and return its canonical request.
    pub fn into_request(self) -> LanguageRequest {
        self.0
    }
}

impl From<LanguageRequest> for LanguageInput {
    fn from(request: LanguageRequest) -> Self {
        Self(request)
    }
}

impl From<Message> for LanguageInput {
    fn from(message: Message) -> Self {
        Self(LanguageRequest::new(vec![message]))
    }
}

impl From<Vec<Message>> for LanguageInput {
    fn from(messages: Vec<Message>) -> Self {
        Self(LanguageRequest::new(messages))
    }
}

impl From<String> for LanguageInput {
    fn from(text: String) -> Self {
        Self(LanguageRequest::new(vec![Message::user(text)]))
    }
}

impl From<&str> for LanguageInput {
    fn from(text: &str) -> Self {
        Self::from(text.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LanguageRequestError {
    #[error(transparent)]
    Generation(#[from] GenerationConfigError),
    #[error("message {message_index} violates the canonical role/content contract: {source}")]
    InvalidMessage {
        message_index: usize,
        source: MessageValidationError,
    },
    #[error("tool `{name}` is defined more than once")]
    DuplicateTool { name: String },
    #[error("tool choice names undefined tool `{name}`")]
    UnknownToolChoice { name: String },
    #[error("structured output name must not be empty")]
    EmptyStructuredOutputName,
    #[error("structured output schema must be a JSON Schema object or boolean")]
    InvalidStructuredOutputSchema,
    #[error(transparent)]
    OpaqueProviderItems(#[from] OpaqueProviderItemError),
    #[error(transparent)]
    ProviderAnnotations(#[from] ProviderAnnotationError),
}

/// Common generation controls with omission preserved explicitly.
///
/// Providers reject controls they cannot encode without loss. Model-specific
/// controls remain in typed provider options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_output_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GenerationConfigError {
    #[error("temperature must be finite and non-negative")]
    InvalidTemperature,
    #[error("top_p must be finite and between 0 and 1 inclusive")]
    InvalidTopP,
    #[error("stop sequence {index} must not be empty")]
    EmptyStopSequence { index: usize },
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<(), GenerationConfigError> {
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(GenerationConfigError::InvalidTemperature);
        }
        if self
            .top_p
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(GenerationConfigError::InvalidTopP);
        }
        if let Some(index) = self
            .stop_sequences
            .iter()
            .position(|sequence| sequence.is_empty())
        {
            return Err(GenerationConfigError::EmptyStopSequence { index });
        }
        Ok(())
    }
}

/// Portable tool selection. Provider-only hosted tools remain typed extensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named { name: String },
}

/// Provider-neutral JSON Schema request shaping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputSpec {
    pub name: String,
    pub description: Option<String>,
    pub schema: Value,
    pub strict: bool,
}
