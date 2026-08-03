//! Bounded compatibility differences for Chat Completions dialects.

use thiserror::Error;

const MAX_FIELD_BYTES: usize = 128;

/// Wire field used for the canonical maximum-output-token control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MaxOutputTokensField {
    MaxTokens,
    MaxCompletionTokens,
}

impl MaxOutputTokensField {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

/// Validated non-standard reasoning field used by a named dialect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningField(String);

impl ReasoningField {
    pub fn new(value: impl Into<String>) -> Result<Self, DialectError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_FIELD_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(DialectError::InvalidReasoningField);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Closed compatibility knobs for one verified or generic profile.
///
/// This value cannot change endpoints, authentication, protected request
/// fields, response terminal semantics, or retry behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatCompletionsDialect {
    developer_role: bool,
    reasoning_input_field: Option<ReasoningField>,
    reasoning_output_field: Option<ReasoningField>,
    max_output_tokens_field: MaxOutputTokensField,
    stream_usage: bool,
}

impl Default for ChatCompletionsDialect {
    fn default() -> Self {
        Self {
            developer_role: false,
            reasoning_input_field: None,
            reasoning_output_field: None,
            max_output_tokens_field: MaxOutputTokensField::MaxTokens,
            stream_usage: true,
        }
    }
}

impl ChatCompletionsDialect {
    pub fn generic() -> Self {
        Self::default()
    }

    pub fn with_developer_role(mut self, supported: bool) -> Self {
        self.developer_role = supported;
        self
    }

    pub fn with_reasoning_input_field(mut self, field: ReasoningField) -> Self {
        self.reasoning_input_field = Some(field);
        self
    }

    pub fn with_reasoning_output_field(mut self, field: ReasoningField) -> Self {
        self.reasoning_output_field = Some(field);
        self
    }

    pub fn with_max_output_tokens_field(mut self, field: MaxOutputTokensField) -> Self {
        self.max_output_tokens_field = field;
        self
    }

    pub fn with_stream_usage(mut self, supported: bool) -> Self {
        self.stream_usage = supported;
        self
    }

    pub fn supports_developer_role(&self) -> bool {
        self.developer_role
    }

    pub fn reasoning_input_field(&self) -> Option<&str> {
        self.reasoning_input_field
            .as_ref()
            .map(ReasoningField::as_str)
    }

    pub fn reasoning_output_field(&self) -> Option<&str> {
        self.reasoning_output_field
            .as_ref()
            .map(ReasoningField::as_str)
    }

    pub fn max_output_tokens_field(&self) -> MaxOutputTokensField {
        self.max_output_tokens_field
    }

    pub fn supports_stream_usage(&self) -> bool {
        self.stream_usage
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DialectError {
    #[error("reasoning field must be 1..=128 ASCII letters, digits, or '_'")]
    InvalidReasoningField,
}
