use std::collections::HashMap;

use crate::types::UiMessage;
use serde_json::Value;
use thiserror::Error;

/// Errors raised while validating or converting UI messages.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum UiMessageError {
    /// A UI message had no parts.
    #[error("UI message `{message_id}` must contain at least one part")]
    EmptyMessageParts { message_id: String },

    /// A UI message part is structurally invalid.
    #[error("UI message `{message_id}` part {part_index} is invalid: {message}")]
    InvalidPart {
        message_id: String,
        part_index: usize,
        message: String,
    },

    /// UI message metadata failed schema validation.
    #[error("UI message `{message_id}` metadata is invalid: {message}")]
    InvalidMetadata { message_id: String, message: String },

    /// Runtime tool-output mapping failed while converting a UI message.
    #[error("failed to convert UI tool `{tool_name}` (`{tool_call_id}`) output: {message}")]
    ToolOutputConversion {
        tool_name: String,
        tool_call_id: String,
        message: String,
    },
}

/// Options controlling `convert_to_model_messages`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConvertUiMessagesOptions {
    /// Drop `input-streaming` and `input-available` tool parts before conversion.
    pub ignore_incomplete_tool_calls: bool,
}

/// Additional schema-aware validation inputs for `validate_ui_messages_with_schemas`.
#[derive(Debug, Clone, Default)]
pub struct ValidateUiMessagesSchemaOptions<'a> {
    /// Optional schema for message-level metadata.
    pub metadata_schema: Option<&'a Value>,
    /// Optional schemas for `data-*` UI parts keyed by their suffix name.
    pub data_schemas: Option<&'a HashMap<String, Value>>,
}

/// Rust result union for AI SDK `SafeValidateUIMessagesResult`.
#[derive(Debug, Clone, PartialEq)]
pub enum SafeValidateUiMessagesResult {
    /// Validation succeeded and returns the validated message list.
    Success { data: Vec<UiMessage> },
    /// Validation failed without throwing.
    Failure { error: UiMessageError },
}

impl SafeValidateUiMessagesResult {
    /// Return whether validation succeeded.
    pub const fn success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Borrow the validated message list when validation succeeded.
    pub fn data(&self) -> Option<&[UiMessage]> {
        match self {
            Self::Success { data } => Some(data),
            Self::Failure { .. } => None,
        }
    }

    /// Borrow the validation error when validation failed.
    pub fn error(&self) -> Option<&UiMessageError> {
        match self {
            Self::Success { .. } => None,
            Self::Failure { error } => Some(error),
        }
    }

    /// Convert the safe result into a standard Rust `Result`.
    pub fn into_result(self) -> Result<Vec<UiMessage>, UiMessageError> {
        match self {
            Self::Success { data } => Ok(data),
            Self::Failure { error } => Err(error),
        }
    }
}

/// AI SDK export spelling for `SafeValidateUIMessagesResult`.
pub type SafeValidateUIMessagesResult = SafeValidateUiMessagesResult;

/// Abstract schema validator used by UI-message schema-aware validation.
pub trait UiSchemaValidator: Send + Sync {
    /// Validate `instance` against `schema`.
    fn validate(&self, schema: &Value, instance: &Value) -> Result<(), String>;
}

impl<F> UiSchemaValidator for F
where
    F: Fn(&Value, &Value) -> Result<(), String> + Send + Sync,
{
    fn validate(&self, schema: &Value, instance: &Value) -> Result<(), String> {
        self(schema, instance)
    }
}
