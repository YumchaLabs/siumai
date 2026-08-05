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

/// Validated non-standard JSON field used by one bounded dialect rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireFieldName(String);

impl WireFieldName {
    pub fn new(value: impl Into<String>) -> Result<Self, DialectError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_FIELD_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(DialectError::InvalidWireFieldName);
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
    video_input: bool,
    reasoning_input_field: Option<WireFieldName>,
    reasoning_output_field: Option<WireFieldName>,
    cache_read_tokens_field: Option<WireFieldName>,
    max_output_tokens_field: MaxOutputTokensField,
    function_tool_strict: Option<bool>,
    stream_usage: bool,
    stream_choice_usage: bool,
}

impl Default for ChatCompletionsDialect {
    fn default() -> Self {
        Self {
            developer_role: false,
            video_input: false,
            reasoning_input_field: None,
            reasoning_output_field: None,
            cache_read_tokens_field: None,
            max_output_tokens_field: MaxOutputTokensField::MaxTokens,
            function_tool_strict: None,
            stream_usage: true,
            stream_choice_usage: false,
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

    /// Allow OpenAI-shaped `video_url` input blocks for a verified dialect.
    pub fn with_video_input(mut self, supported: bool) -> Self {
        self.video_input = supported;
        self
    }

    pub fn with_reasoning_input_field(mut self, field: WireFieldName) -> Self {
        self.reasoning_input_field = Some(field);
        self
    }

    pub fn with_reasoning_output_field(mut self, field: WireFieldName) -> Self {
        self.reasoning_output_field = Some(field);
        self
    }

    /// Map one top-level numeric usage field to canonical cache-read tokens.
    pub fn with_cache_read_tokens_field(mut self, field: WireFieldName) -> Self {
        self.cache_read_tokens_field = Some(field);
        self
    }

    pub fn with_max_output_tokens_field(mut self, field: MaxOutputTokensField) -> Self {
        self.max_output_tokens_field = field;
        self
    }

    /// Set the provider default explicitly on every function-tool definition.
    pub fn with_function_tool_strict(mut self, strict: bool) -> Self {
        self.function_tool_strict = Some(strict);
        self
    }

    pub fn with_stream_usage(mut self, supported: bool) -> Self {
        self.stream_usage = supported;
        self
    }

    /// Read usage nested in the final stream choice for a verified dialect.
    pub fn with_stream_choice_usage(mut self, supported: bool) -> Self {
        self.stream_choice_usage = supported;
        self
    }

    pub fn supports_developer_role(&self) -> bool {
        self.developer_role
    }

    pub fn supports_video_input(&self) -> bool {
        self.video_input
    }

    pub fn reasoning_input_field(&self) -> Option<&str> {
        self.reasoning_input_field
            .as_ref()
            .map(WireFieldName::as_str)
    }

    pub fn reasoning_output_field(&self) -> Option<&str> {
        self.reasoning_output_field
            .as_ref()
            .map(WireFieldName::as_str)
    }

    pub fn cache_read_tokens_field(&self) -> Option<&str> {
        self.cache_read_tokens_field
            .as_ref()
            .map(WireFieldName::as_str)
    }

    pub fn max_output_tokens_field(&self) -> MaxOutputTokensField {
        self.max_output_tokens_field
    }

    pub fn function_tool_strict(&self) -> Option<bool> {
        self.function_tool_strict
    }

    pub fn supports_stream_usage(&self) -> bool {
        self.stream_usage
    }

    pub fn supports_stream_choice_usage(&self) -> bool {
        self.stream_choice_usage
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DialectError {
    #[error("wire field name must be 1..=128 ASCII letters, digits, or '_'")]
    InvalidWireFieldName,
}
