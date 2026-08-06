//! Strict structured-output request shaping and final consumption.
//!
//! This module deliberately does not attempt to recover JSON from Markdown,
//! prose, truncated fragments, or tool arguments. A final value is successful
//! only after strict JSON parsing, JSON Schema validation, and typed decoding.

use std::error::Error as StdError;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai_core::{
    CallOptions, ContentPart, FinishReason, LanguageIncompleteReason, LanguageRequest,
    LanguageResponse, LanguageResponseStatus, Message, MessagePart, MessageRole,
    PartialStructuredOutput, StructuredOutputSpec,
};
use thiserror::Error;

const REPAIR_INSTRUCTION: &str = "The previous structured-output response was invalid. Return one replacement JSON value that matches the requested schema. Do not call tools and do not wrap the JSON in Markdown or prose.";

/// A JSON Schema validation failure reported by a validator adapter.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct SchemaValidationError {
    message: String,
}

impl SchemaValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            message: if message.trim().is_empty() {
                "schema validation failed".to_string()
            } else {
                message
            },
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Pluggable JSON Schema validation owned by the structured-output consumer.
///
/// The runtime intentionally does not implement a partial JSON Schema engine.
/// Applications or facade integrations can adapt a standards-compliant
/// validator without changing the provider-neutral output contract.
pub trait OutputSchemaValidator: Send + Sync {
    /// Validate that the descriptor's schema can be used by this validator.
    fn validate_schema(&self, _schema: &Value) -> Result<(), SchemaValidationError> {
        Ok(())
    }

    /// Validate one parsed JSON instance against the descriptor's schema.
    fn validate(&self, schema: &Value, instance: &Value) -> Result<(), SchemaValidationError>;
}

impl<F> OutputSchemaValidator for F
where
    F: Fn(&Value, &Value) -> Result<(), SchemaValidationError> + Send + Sync,
{
    fn validate(&self, schema: &Value, instance: &Value) -> Result<(), SchemaValidationError> {
        self(schema, instance)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AnyJsonValidator;

impl OutputSchemaValidator for AnyJsonValidator {
    fn validate(&self, _schema: &Value, _instance: &Value) -> Result<(), SchemaValidationError> {
        Ok(())
    }
}

/// Invalid structured-output descriptor configuration.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum OutputDescriptorError {
    #[error("structured output name must not be empty")]
    EmptyName,
    #[error("structured output schema must be a JSON Schema object or boolean")]
    InvalidSchemaShape,
    #[error("structured output schema is invalid: {0}")]
    InvalidSchema(#[source] SchemaValidationError),
}

/// Whether the runtime may spend one additional model step repairing output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RepairPolicy {
    /// Never issue a repair model call.
    #[default]
    Disabled,
    /// Permit one tool-free repair step for an initial parse/schema failure.
    OneAttempt,
}

impl RepairPolicy {
    /// Maximum number of additional model steps permitted by this policy.
    pub const fn max_additional_steps(self) -> u8 {
        match self {
            Self::Disabled => 0,
            Self::OneAttempt => 1,
        }
    }

    /// Whether this policy permits a repair plan for the supplied failure.
    pub const fn allows(self, failure: &StructuredOutputError) -> bool {
        matches!(self, Self::OneAttempt) && failure.is_repair_eligible()
    }
}

/// Which model attempt produced a structured-output result or failure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum StructuredOutputAttemptKind {
    #[default]
    Initial,
    Repair,
}

impl fmt::Display for StructuredOutputAttemptKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initial => formatter.write_str("initial attempt"),
            Self::Repair => formatter.write_str("repair attempt"),
        }
    }
}

/// Stable classification of a structured-output failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StructuredOutputFailureKind {
    Refusal,
    ContentFilter,
    MissingOutput,
    IncompleteOutput,
    UnexpectedToolCall,
    ProviderFailure,
    TransportFailure,
    Cancelled,
    InvalidJson,
    SchemaMismatch,
    TypedDecodeMismatch,
}

impl StructuredOutputFailureKind {
    /// Only strict JSON parsing and schema mismatches are repair candidates.
    pub const fn can_repair(self) -> bool {
        matches!(self, Self::InvalidJson | Self::SchemaMismatch)
    }
}

impl fmt::Display for StructuredOutputFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Refusal => "refusal",
            Self::ContentFilter => "content filter",
            Self::MissingOutput => "missing output",
            Self::IncompleteOutput => "incomplete output",
            Self::UnexpectedToolCall => "unexpected tool call",
            Self::ProviderFailure => "provider failure",
            Self::TransportFailure => "transport failure",
            Self::Cancelled => "cancellation",
            Self::InvalidJson => "invalid JSON",
            Self::SchemaMismatch => "schema mismatch",
            Self::TypedDecodeMismatch => "typed decode mismatch",
        };
        formatter.write_str(value)
    }
}

#[derive(Debug)]
enum StructuredOutputErrorSource {
    Json(serde_json::Error),
    Schema(SchemaValidationError),
    Transport(siumai_core::Error),
}

/// A strict structured-output failure with the failed response retained.
///
/// Retaining the response lets the runtime account usage from unsuccessful
/// attempts before it decides whether to spend the single repair step.
#[derive(Debug)]
pub struct StructuredOutputError {
    kind: StructuredOutputFailureKind,
    attempt: StructuredOutputAttemptKind,
    message: String,
    details: Box<StructuredOutputErrorDetails>,
}

#[derive(Debug, Default)]
struct StructuredOutputErrorDetails {
    response: Option<Box<LanguageResponse>>,
    raw_output: Option<String>,
    parsed_value: Option<Value>,
    source: Option<StructuredOutputErrorSource>,
}

impl StructuredOutputError {
    /// Convert a model-call transport failure into a non-repairable output error.
    pub fn from_transport(
        source: siumai_core::Error,
        attempt: StructuredOutputAttemptKind,
    ) -> Self {
        Self {
            kind: StructuredOutputFailureKind::TransportFailure,
            attempt,
            message: "the model call failed before a final response was available".to_string(),
            details: Box::new(StructuredOutputErrorDetails {
                source: Some(StructuredOutputErrorSource::Transport(source)),
                ..StructuredOutputErrorDetails::default()
            }),
        }
    }

    pub fn kind(&self) -> StructuredOutputFailureKind {
        self.kind
    }

    pub fn attempt(&self) -> StructuredOutputAttemptKind {
        self.attempt
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn response(&self) -> Option<&LanguageResponse> {
        self.details.response.as_deref()
    }

    pub fn raw_output(&self) -> Option<&str> {
        self.details.raw_output.as_deref()
    }

    pub fn parsed_value(&self) -> Option<&Value> {
        self.details.parsed_value.as_ref()
    }

    /// True only for the initial parse/schema failures that may be repaired.
    pub const fn is_repair_eligible(&self) -> bool {
        matches!(self.attempt, StructuredOutputAttemptKind::Initial) && self.kind.can_repair()
    }

    pub fn into_response(self) -> Option<LanguageResponse> {
        self.details.response.map(|response| *response)
    }

    pub(crate) fn with_model_error_source(mut self, source: siumai_core::Error) -> Self {
        self.details.source = Some(StructuredOutputErrorSource::Transport(source));
        self
    }

    fn response_failure(
        kind: StructuredOutputFailureKind,
        attempt: StructuredOutputAttemptKind,
        message: impl Into<String>,
        response: LanguageResponse,
    ) -> Self {
        Self {
            kind,
            attempt,
            message: message.into(),
            details: Box::new(StructuredOutputErrorDetails {
                response: Some(Box::new(response)),
                ..StructuredOutputErrorDetails::default()
            }),
        }
    }

    fn invalid_json(
        attempt: StructuredOutputAttemptKind,
        response: LanguageResponse,
        raw_output: String,
        source: serde_json::Error,
    ) -> Self {
        Self {
            kind: StructuredOutputFailureKind::InvalidJson,
            attempt,
            message: "the final response is not one complete JSON value".to_string(),
            details: Box::new(StructuredOutputErrorDetails {
                response: Some(Box::new(response)),
                raw_output: Some(raw_output),
                parsed_value: None,
                source: Some(StructuredOutputErrorSource::Json(source)),
            }),
        }
    }

    fn schema_mismatch(
        attempt: StructuredOutputAttemptKind,
        response: LanguageResponse,
        raw_output: String,
        parsed_value: Value,
        source: SchemaValidationError,
    ) -> Self {
        let message = source.to_string();
        Self {
            kind: StructuredOutputFailureKind::SchemaMismatch,
            attempt,
            message,
            details: Box::new(StructuredOutputErrorDetails {
                response: Some(Box::new(response)),
                raw_output: Some(raw_output),
                parsed_value: Some(parsed_value),
                source: Some(StructuredOutputErrorSource::Schema(source)),
            }),
        }
    }

    fn typed_decode_mismatch(
        attempt: StructuredOutputAttemptKind,
        response: LanguageResponse,
        raw_output: String,
        parsed_value: Value,
        source: serde_json::Error,
    ) -> Self {
        Self {
            kind: StructuredOutputFailureKind::TypedDecodeMismatch,
            attempt,
            message: "the schema-valid JSON value cannot be decoded into the requested Rust type"
                .to_string(),
            details: Box::new(StructuredOutputErrorDetails {
                response: Some(Box::new(response)),
                raw_output: Some(raw_output),
                parsed_value: Some(parsed_value),
                source: Some(StructuredOutputErrorSource::Json(source)),
            }),
        }
    }
}

impl fmt::Display for StructuredOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "structured output {} failed ({}): {}",
            self.attempt, self.kind, self.message
        )
    }
}

impl StdError for StructuredOutputError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self.details.source.as_ref()? {
            StructuredOutputErrorSource::Json(source) => Some(source),
            StructuredOutputErrorSource::Schema(source) => Some(source),
            StructuredOutputErrorSource::Transport(source) => Some(source),
        }
    }
}

/// A fully validated structured-output value and its original response.
#[derive(Debug, Clone)]
pub struct StructuredOutputResult<T> {
    value: T,
    json: Value,
    raw_output: String,
    response: LanguageResponse,
    attempt: StructuredOutputAttemptKind,
}

impl<T> StructuredOutputResult<T> {
    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn json(&self) -> &Value {
        &self.json
    }

    pub fn raw_output(&self) -> &str {
        &self.raw_output
    }

    pub fn response(&self) -> &LanguageResponse {
        &self.response
    }

    pub fn attempt(&self) -> StructuredOutputAttemptKind {
        self.attempt
    }

    pub fn was_repaired(&self) -> bool {
        matches!(self.attempt, StructuredOutputAttemptKind::Repair)
    }

    pub fn into_value(self) -> T {
        self.value
    }

    pub fn into_parts(self) -> (T, Value, String, LanguageResponse) {
        (self.value, self.json, self.raw_output, self.response)
    }
}

/// One tool-free repair step with inherited call-scoped controls.
///
/// The step engine must charge this as an additional model step in the same
/// [`RunBudget`](crate::RunBudget) and usage ledger before executing it.
/// `CallOptions` is cloned from the failed attempt, preserving its cancellation
/// token and never extending its deadline.
#[derive(Debug)]
pub struct StructuredOutputRepair {
    request: LanguageRequest,
    call_options: CallOptions,
    initial_failure: StructuredOutputError,
}

impl StructuredOutputRepair {
    pub fn request(&self) -> &LanguageRequest {
        &self.request
    }

    pub fn call_options(&self) -> &CallOptions {
        &self.call_options
    }

    pub fn initial_failure(&self) -> &StructuredOutputError {
        &self.initial_failure
    }

    pub fn into_parts(self) -> (LanguageRequest, CallOptions, StructuredOutputError) {
        (self.request, self.call_options, self.initial_failure)
    }
}

/// The single owner of structured-output schema, shaping, and final validation.
pub struct OutputDescriptor<T> {
    spec: StructuredOutputSpec,
    repair_policy: RepairPolicy,
    validator: Arc<dyn OutputSchemaValidator>,
    output: PhantomData<fn() -> T>,
}

impl<T> Clone for OutputDescriptor<T> {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec.clone(),
            repair_policy: self.repair_policy,
            validator: Arc::clone(&self.validator),
            output: PhantomData,
        }
    }
}

impl<T> fmt::Debug for OutputDescriptor<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OutputDescriptor")
            .field("spec", &self.spec)
            .field("repair_policy", &self.repair_policy)
            .field("output_type", &std::any::type_name::<T>())
            .finish_non_exhaustive()
    }
}

impl<T> OutputDescriptor<T> {
    /// Create a descriptor backed by an application-supplied JSON Schema validator.
    pub fn new<V>(
        name: impl Into<String>,
        schema: Value,
        validator: V,
    ) -> Result<Self, OutputDescriptorError>
    where
        V: OutputSchemaValidator + 'static,
    {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(OutputDescriptorError::EmptyName);
        }
        if !schema.is_object() && !schema.is_boolean() {
            return Err(OutputDescriptorError::InvalidSchemaShape);
        }
        validator
            .validate_schema(&schema)
            .map_err(OutputDescriptorError::InvalidSchema)?;

        Ok(Self {
            spec: StructuredOutputSpec {
                name,
                description: None,
                schema,
                strict: true,
            },
            repair_policy: RepairPolicy::Disabled,
            validator: Arc::new(validator),
            output: PhantomData,
        })
    }

    /// Create a typed JSON descriptor using the always-valid JSON Schema `true`.
    ///
    /// This still performs strict JSON parsing and typed decoding, but intentionally
    /// imposes no additional shape beyond what `T` accepts.
    pub fn typed_json(name: impl Into<String>) -> Result<Self, OutputDescriptorError> {
        Self::new(name, Value::Bool(true), AnyJsonValidator)
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        let description = description.into();
        self.spec.description = if description.trim().is_empty() {
            None
        } else {
            Some(description)
        };
        self
    }

    /// Set the provider-native strictness hint without weakening final validation.
    pub fn with_provider_strict(mut self, strict: bool) -> Self {
        self.spec.strict = strict;
        self
    }

    pub fn with_repair_policy(mut self, repair_policy: RepairPolicy) -> Self {
        self.repair_policy = repair_policy;
        self
    }

    pub fn spec(&self) -> &StructuredOutputSpec {
        &self.spec
    }

    pub fn repair_policy(&self) -> RepairPolicy {
        self.repair_policy
    }

    /// Make this descriptor the sole structured-output shape for one request.
    pub fn shape_request(&self, mut request: LanguageRequest) -> LanguageRequest {
        request.structured_output = Some(self.spec.clone());
        request
    }

    /// Mark a streamed partial value as explicitly not schema-validated.
    pub fn unvalidated_partial(&self, value: Value) -> PartialStructuredOutput {
        PartialStructuredOutput::unvalidated(value)
    }

    /// Strictly consume the initial model response.
    pub fn consume_response(
        &self,
        response: LanguageResponse,
    ) -> Result<StructuredOutputResult<T>, StructuredOutputError>
    where
        T: DeserializeOwned,
    {
        self.consume(response, StructuredOutputAttemptKind::Initial)
    }

    /// Strictly consume the single repair response.
    ///
    /// Failures from this method are never eligible for another repair.
    pub fn consume_repair_response(
        &self,
        response: LanguageResponse,
    ) -> Result<StructuredOutputResult<T>, StructuredOutputError>
    where
        T: DeserializeOwned,
    {
        self.consume(response, StructuredOutputAttemptKind::Repair)
    }

    /// Build the only permitted repair step, consuming the initial failure.
    ///
    /// Disabled policies, non-parse/schema failures, and failures from an
    /// existing repair attempt are returned unchanged.
    pub fn plan_repair(
        &self,
        mut request: LanguageRequest,
        failure: StructuredOutputError,
        call_options: &CallOptions,
    ) -> Result<StructuredOutputRepair, StructuredOutputError> {
        if !self.repair_policy.allows(&failure) {
            return Err(failure);
        }

        if let Some(response) = failure.response()
            && !response.content().is_empty()
        {
            request.messages.push(Message::new(
                MessageRole::Assistant,
                response.content().iter().cloned().map(MessagePart::from),
            ));
        }
        request
            .messages
            .push(Message::text(MessageRole::Developer, REPAIR_INSTRUCTION));
        request.tools.clear();
        request.tool_choice = None;
        request.structured_output = Some(self.spec.clone());

        Ok(StructuredOutputRepair {
            request,
            call_options: call_options.clone(),
            initial_failure: failure,
        })
    }

    fn consume(
        &self,
        response: LanguageResponse,
        attempt: StructuredOutputAttemptKind,
    ) -> Result<StructuredOutputResult<T>, StructuredOutputError>
    where
        T: DeserializeOwned,
    {
        if let Some((kind, message)) = classify_response(&response) {
            return Err(StructuredOutputError::response_failure(
                kind, attempt, message, response,
            ));
        }

        let raw_output = response
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();

        if raw_output.trim().is_empty() {
            return Err(StructuredOutputError::response_failure(
                StructuredOutputFailureKind::MissingOutput,
                attempt,
                "the final response contains no structured-output text",
                response,
            ));
        }

        let json = match serde_json::from_str::<Value>(&raw_output) {
            Ok(json) => json,
            Err(source) => {
                return Err(StructuredOutputError::invalid_json(
                    attempt, response, raw_output, source,
                ));
            }
        };

        if let Err(source) = self.validator.validate(&self.spec.schema, &json) {
            return Err(StructuredOutputError::schema_mismatch(
                attempt, response, raw_output, json, source,
            ));
        }

        let value = match serde_json::from_value::<T>(json.clone()) {
            Ok(value) => value,
            Err(source) => {
                return Err(StructuredOutputError::typed_decode_mismatch(
                    attempt, response, raw_output, json, source,
                ));
            }
        };

        Ok(StructuredOutputResult {
            value,
            json,
            raw_output,
            response,
            attempt,
        })
    }
}

fn classify_response(response: &LanguageResponse) -> Option<(StructuredOutputFailureKind, String)> {
    if response
        .content()
        .iter()
        .any(|part| matches!(part, ContentPart::Refusal { .. }))
    {
        return Some((
            StructuredOutputFailureKind::Refusal,
            "the model refused to produce structured output".to_string(),
        ));
    }

    match response.status() {
        LanguageResponseStatus::Incomplete {
            reason: Some(LanguageIncompleteReason::ContentFilter),
        } => {
            return Some((
                StructuredOutputFailureKind::ContentFilter,
                "the response was blocked by content filtering".to_string(),
            ));
        }
        LanguageResponseStatus::Incomplete { .. } => {
            return Some((
                StructuredOutputFailureKind::IncompleteOutput,
                "the provider ended the response before normal completion".to_string(),
            ));
        }
        LanguageResponseStatus::Failed => {
            return Some((
                StructuredOutputFailureKind::ProviderFailure,
                "the provider reported a failed response".to_string(),
            ));
        }
        LanguageResponseStatus::Cancelled => {
            return Some((
                StructuredOutputFailureKind::Cancelled,
                "the model call was cancelled".to_string(),
            ));
        }
        LanguageResponseStatus::Completed => {}
        _ => {
            return Some((
                StructuredOutputFailureKind::ProviderFailure,
                "the provider returned an unsupported response status".to_string(),
            ));
        }
    }

    match response.finish_reason() {
        FinishReason::Refusal => Some((
            StructuredOutputFailureKind::Refusal,
            "the model refused to produce structured output".to_string(),
        )),
        FinishReason::ContentFilter => Some((
            StructuredOutputFailureKind::ContentFilter,
            "the response was blocked by content filtering".to_string(),
        )),
        FinishReason::Error => Some((
            StructuredOutputFailureKind::ProviderFailure,
            "the provider reported a failed response".to_string(),
        )),
        FinishReason::Cancelled => Some((
            StructuredOutputFailureKind::Cancelled,
            "the model call was cancelled".to_string(),
        )),
        FinishReason::ToolCalls => Some((
            StructuredOutputFailureKind::UnexpectedToolCall,
            "the model requested a tool instead of producing final structured output".to_string(),
        )),
        _ if response
            .content()
            .iter()
            .any(|part| matches!(part, ContentPart::ToolCall(_))) =>
        {
            Some((
                StructuredOutputFailureKind::UnexpectedToolCall,
                "the final response contains an unresolved tool call".to_string(),
            ))
        }
        _ => None,
    }
}
