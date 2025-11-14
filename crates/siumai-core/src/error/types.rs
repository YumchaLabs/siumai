//! Core Error Types (moved from siumai)

use thiserror::Error;

/// Error category for better error handling and recovery strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    Network,
    Authentication,
    RateLimit,
    Client,
    Server,
    Parsing,
    Validation,
    Configuration,
    Unsupported,
    Stream,
    Provider,
    Unknown,
}

/// The primary error type for the LLM library.
#[derive(Error, Debug, Clone)]
pub enum LlmError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),
    #[error("JSON error: {0}")]
    JsonError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Missing API key: {0}")]
    MissingApiKey(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("API error: {code} - {message}")]
    ApiError {
        code: u16,
        message: String,
        details: Option<serde_json::Value>,
    },
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),
    #[error("Quota exceeded: {0}")]
    QuotaExceededError(String),
    #[error("Model not supported: {0}")]
    ModelNotSupported(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Request timeout: {0}")]
    TimeoutError(String),
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Provider error ({provider}): {message}")]
    ProviderError {
        provider: String,
        message: String,
        error_code: Option<String>,
    },
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
    #[error("Tool call error: {0}")]
    ToolCallError(String),
    #[error("Tool validation error: {0}")]
    ToolValidationError(String),
    #[error("Unsupported tool type: {0}")]
    UnsupportedToolType(String),
    #[error("Error in {context}: {message}")]
    ContextualError {
        context: String,
        message: String,
        source_error: Option<Box<LlmError>>,
        metadata: std::collections::HashMap<String, String>,
    },
    #[error("Other error: {0}")]
    Other(String),
}

impl LlmError {
    pub fn api_error(code: u16, message: impl Into<String>) -> Self {
        Self::ApiError {
            code,
            message: message.into(),
            details: None,
        }
    }
    pub fn api_error_with_details(
        code: u16,
        message: impl Into<String>,
        details: serde_json::Value,
    ) -> Self {
        Self::ApiError {
            code,
            message: message.into(),
            details: Some(details),
        }
    }
    pub fn provider_error(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ProviderError {
            provider: provider.into(),
            message: message.into(),
            error_code: None,
        }
    }
    pub fn provider_error_with_code(
        provider: impl Into<String>,
        message: impl Into<String>,
        error_code: impl Into<String>,
    ) -> Self {
        Self::ProviderError {
            provider: provider.into(),
            message: message.into(),
            error_code: Some(error_code.into()),
        }
    }
    pub fn contextual_error(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ContextualError {
            context: context.into(),
            message: message.into(),
            source_error: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    pub fn contextual_error_with_source(
        context: impl Into<String>,
        message: impl Into<String>,
        source: LlmError,
    ) -> Self {
        Self::ContextualError {
            context: context.into(),
            message: message.into(),
            source_error: Some(Box::new(source)),
            metadata: std::collections::HashMap::new(),
        }
    }
    pub fn contextual_error_with_metadata(
        context: impl Into<String>,
        message: impl Into<String>,
        metadata: std::collections::HashMap<String, String>,
    ) -> Self {
        Self::ContextualError {
            context: context.into(),
            message: message.into(),
            source_error: None,
            metadata,
        }
    }
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => ErrorCategory::Authentication,
            Self::RateLimitError(_) | Self::QuotaExceededError(_) => ErrorCategory::RateLimit,
            Self::ApiError { code, .. } if (400..500).contains(code) => ErrorCategory::Client,
            Self::ApiError { code, .. } if (500..600).contains(code) => ErrorCategory::Server,
            Self::HttpError(_) | Self::ConnectionError(_) | Self::TimeoutError(_) => {
                ErrorCategory::Network
            }
            Self::JsonError(_) | Self::ParseError(_) => ErrorCategory::Parsing,
            Self::InvalidParameter(_) | Self::InvalidInput(_) => ErrorCategory::Validation,
            Self::UnsupportedOperation(_)
            | Self::UnsupportedToolType(_)
            | Self::ModelNotSupported(_) => ErrorCategory::Unsupported,
            Self::StreamError(_) => ErrorCategory::Stream,
            Self::ProviderError { .. } => ErrorCategory::Provider,
            Self::ConfigurationError(_) => ErrorCategory::Configuration,
            _ => ErrorCategory::Unknown,
        }
    }
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.category(),
            ErrorCategory::Network | ErrorCategory::Server | ErrorCategory::RateLimit
        )
    }
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::ApiError { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// Convenience helper: whether this error represents an authentication failure.
    pub fn is_auth_error(&self) -> bool {
        matches!(self.category(), ErrorCategory::Authentication)
    }

    /// Convenience helper: whether this error represents a rate-limit condition.
    pub fn is_rate_limit_error(&self) -> bool {
        matches!(self.category(), ErrorCategory::RateLimit)
    }

    /// Best-effort maximum retry attempts based on error category.
    ///
    /// This is intentionally conservative and primarily used by higher-level
    /// retry helpers and tests. For most retryable categories, we return 3;
    /// otherwise 0 (no retries).
    pub fn max_retry_attempts(&self) -> u32 {
        if self.is_retryable() { 3 } else { 0 }
    }
}

/// Core Result alias
pub type Result<T> = std::result::Result<T, LlmError>;
