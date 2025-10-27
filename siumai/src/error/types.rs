//! Core Error Types
//!
//! This module defines the core error types used throughout the LLM library:
//! - `LlmError`: The main error enum
//! - `ErrorCategory`: Error categorization for handling strategies
//! - `Result<T>`: Type alias for Result<T, LlmError>

use thiserror::Error;

/// Error category for better error handling and recovery strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Network-related errors (connection, timeout, etc.)
    Network,
    /// Authentication and authorization errors
    Authentication,
    /// Rate limiting and quota errors
    RateLimit,
    /// Client-side errors (4xx HTTP status codes)
    Client,
    /// Server-side errors (5xx HTTP status codes)
    Server,
    /// Data parsing and serialization errors
    Parsing,
    /// Input validation errors
    Validation,
    /// Configuration errors
    Configuration,
    /// Unsupported operations or models
    Unsupported,
    /// Streaming-related errors
    Stream,
    /// Provider-specific errors
    Provider,
    /// Unknown or uncategorized errors
    Unknown,
}

/// The primary error type for the LLM library.
#[derive(Error, Debug, Clone)]
pub enum LlmError {
    /// HTTP request error
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(String),

    /// Response parsing error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Not found error
    #[error("Not found: {0}")]
    NotFound(String),

    /// Missing API key
    #[error("Missing API key: {0}")]
    MissingApiKey(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// API error response
    #[error("API error: {code} - {message}")]
    ApiError {
        code: u16,
        message: String,
        details: Option<serde_json::Value>,
    },

    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Rate limit error
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),

    /// Quota exceeded error
    #[error("Quota exceeded: {0}")]
    QuotaExceededError(String),

    /// Model not supported error
    #[error("Model not supported: {0}")]
    ModelNotSupported(String),

    /// Stream processing error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Timeout error
    #[error("Request timeout: {0}")]
    TimeoutError(String),

    /// Network connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Provider-specific error
    #[error("Provider error ({provider}): {message}")]
    ProviderError {
        provider: String,
        message: String,
        error_code: Option<String>,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Tool call error
    #[error("Tool call error: {0}")]
    ToolCallError(String),

    /// Tool validation error
    #[error("Tool validation error: {0}")]
    ToolValidationError(String),

    /// Unsupported tool type
    #[error("Unsupported tool type: {0}")]
    UnsupportedToolType(String),

    /// Context-aware error with additional metadata
    #[error("Error in {context}: {message}")]
    ContextualError {
        context: String,
        message: String,
        source_error: Option<Box<LlmError>>,
        metadata: std::collections::HashMap<String, String>,
    },

    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}

impl LlmError {
    /// Creates a new API error.
    pub fn api_error(code: u16, message: impl Into<String>) -> Self {
        Self::ApiError {
            code,
            message: message.into(),
            details: None,
        }
    }

    /// Creates a new API error with details.
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

    /// Creates a new provider error.
    pub fn provider_error(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ProviderError {
            provider: provider.into(),
            message: message.into(),
            error_code: None,
        }
    }

    /// Creates a new provider error with an error code.
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

    /// Creates a new contextual error with metadata.
    pub fn contextual_error(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ContextualError {
            context: context.into(),
            message: message.into(),
            source_error: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a contextual error with a source error.
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

    /// Creates a contextual error with metadata.
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

    /// Checks if the error is retryable with more sophisticated logic.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::HttpError(e) => {
                // More comprehensive check for retryable HTTP errors
                let retryable_keywords = [
                    "timeout",
                    "connect",
                    "network",
                    "dns",
                    "socket",
                    "connection reset",
                    "connection refused",
                    "temporary failure",
                ];
                retryable_keywords
                    .iter()
                    .any(|keyword| e.to_lowercase().contains(keyword))
            }
            Self::ApiError { code, .. } => {
                // More nuanced retry logic based on HTTP status codes
                match *code {
                    // 4xx errors that are retryable
                    408 | 429 => true, // Request Timeout, Too Many Requests
                    // 5xx errors are generally retryable
                    500..=599 => true,
                    // Other 4xx errors are not retryable
                    400..=499 => false,
                    // Unexpected codes, be conservative
                    _ => false,
                }
            }
            Self::RateLimitError(_) => true,
            Self::TimeoutError(_) => true,
            Self::ConnectionError(_) => true,
            Self::QuotaExceededError(_) => false, // Don't retry quota errors
            Self::AuthenticationError(_) => false, // Don't retry auth errors
            Self::ModelNotSupported(_) => false,  // Don't retry unsupported models
            Self::InvalidInput(_) | Self::InvalidParameter(_) => false, // Don't retry validation errors
            Self::ContextualError {
                source_error: Some(source),
                ..
            } => source.is_retryable(),
            Self::ContextualError { .. } => false, // Conservative approach for unknown contextual errors
            _ => false,
        }
    }

    /// Checks if the error is an authentication-related error.
    pub const fn is_auth_error(&self) -> bool {
        match self {
            Self::AuthenticationError(_) => true,
            Self::ApiError { code, .. } => *code == 401 || *code == 403,
            _ => false,
        }
    }

    /// Checks if the error is a rate limit error.
    pub const fn is_rate_limit_error(&self) -> bool {
        match self {
            Self::RateLimitError(_) => true,
            Self::ApiError { code, .. } => *code == 429,
            _ => false,
        }
    }

    /// Gets the HTTP status code of the error, if available.
    pub const fn status_code(&self) -> Option<u16> {
        match self {
            Self::ApiError { code, .. } => Some(*code),
            Self::HttpError(_) => None, // Cannot get status code directly from the string form.
            _ => None,
        }
    }

    /// Gets the error category for better error handling.
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::HttpError(_) | Self::ConnectionError(_) | Self::TimeoutError(_) => {
                ErrorCategory::Network
            }
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => ErrorCategory::Authentication,
            Self::RateLimitError(_) | Self::QuotaExceededError(_) => ErrorCategory::RateLimit,
            Self::ApiError { code, .. } => match *code {
                429 => ErrorCategory::RateLimit,
                400..=499 => ErrorCategory::Client,
                500..=599 => ErrorCategory::Server,
                _ => ErrorCategory::Unknown,
            },
            Self::JsonError(_) | Self::ParseError(_) => ErrorCategory::Parsing,
            Self::InvalidInput(_) | Self::InvalidParameter(_) | Self::ToolValidationError(_) => {
                ErrorCategory::Validation
            }
            Self::ConfigurationError(_) => ErrorCategory::Configuration,
            Self::ModelNotSupported(_)
            | Self::UnsupportedOperation(_)
            | Self::UnsupportedToolType(_) => ErrorCategory::Unsupported,
            Self::StreamError(_) => ErrorCategory::Stream,
            Self::ProviderError { .. } | Self::ToolCallError(_) => ErrorCategory::Provider,
            Self::ContextualError {
                source_error: Some(source),
                ..
            } => source.category(),
            Self::ContextualError { .. } => ErrorCategory::Unknown,
            _ => ErrorCategory::Unknown,
        }
    }

    /// Gets a user-friendly error message.
    pub fn user_message(&self) -> String {
        match self {
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => {
                "Authentication failed. Please check your API key.".to_string()
            }
            Self::RateLimitError(_) => {
                "Rate limit exceeded. Please wait before making more requests.".to_string()
            }
            Self::QuotaExceededError(_) => {
                "API quota exceeded. Please check your usage limits.".to_string()
            }
            Self::ModelNotSupported(model) => {
                format!("The model '{model}' is not supported by this provider.")
            }
            Self::ConnectionError(_) | Self::TimeoutError(_) => {
                "Network connection failed. Please check your internet connection and try again."
                    .to_string()
            }
            Self::ApiError {
                code: 500..=599, ..
            } => "The service is temporarily unavailable. Please try again later.".to_string(),
            _ => self.to_string(),
        }
    }

    /// Gets suggested recovery actions for the error with more detailed guidance.
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => {
                vec![
                    "Verify your API key is correct and properly formatted".to_string(),
                    "Check if your API key has the required permissions for this operation"
                        .to_string(),
                    "Ensure your API key is not expired or revoked".to_string(),
                    "Verify you're using the correct API endpoint".to_string(),
                ]
            }
            Self::RateLimitError(_) => {
                vec![
                    "Implement exponential backoff with jitter".to_string(),
                    "Reduce request frequency".to_string(),
                    "Consider upgrading your API plan for higher limits".to_string(),
                    "Use request batching if supported".to_string(),
                ]
            }
            Self::QuotaExceededError(_) => {
                vec![
                    "Check your usage dashboard for current consumption".to_string(),
                    "Upgrade your API plan for higher quotas".to_string(),
                    "Wait for quota reset (usually monthly)".to_string(),
                    "Optimize your requests to use fewer tokens".to_string(),
                ]
            }
            Self::ConnectionError(_) | Self::TimeoutError(_) => {
                vec![
                    "Check your internet connection stability".to_string(),
                    "Retry the request with exponential backoff".to_string(),
                    "Increase timeout settings for large requests".to_string(),
                    "Check if the service is experiencing outages".to_string(),
                ]
            }
            Self::ModelNotSupported(_) => {
                vec![
                    "Use a supported model from the provider's model list".to_string(),
                    "Check the provider's documentation for available models".to_string(),
                    "Verify the model name is spelled correctly".to_string(),
                ]
            }
            Self::ApiError { code: 400, .. } => {
                vec![
                    "Check your request parameters for validity".to_string(),
                    "Ensure all required fields are provided".to_string(),
                    "Verify parameter types and formats".to_string(),
                ]
            }
            Self::ApiError {
                code: 500..=599, ..
            } => {
                vec![
                    "Retry the request after a delay (server error)".to_string(),
                    "Check the service status page for outages".to_string(),
                    "Contact support if the issue persists".to_string(),
                ]
            }
            Self::StreamError(_) => {
                vec![
                    "Retry the streaming request".to_string(),
                    "Check network stability for streaming".to_string(),
                    "Consider using non-streaming mode as fallback".to_string(),
                ]
            }
            Self::InvalidInput(_) | Self::InvalidParameter(_) => {
                vec![
                    "Validate your input parameters".to_string(),
                    "Check parameter constraints and limits".to_string(),
                    "Refer to the API documentation for valid formats".to_string(),
                ]
            }
            _ => vec![
                "Check the error details and documentation".to_string(),
                "Contact support if the issue persists".to_string(),
            ],
        }
    }

    /// Gets the recommended retry delay in seconds based on error type.
    pub const fn recommended_retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimitError(_) => Some(60), // Wait 1 minute for rate limits
            Self::ApiError { code: 429, .. } => Some(30), // Wait 30 seconds for 429
            Self::ApiError {
                code: 500..=599, ..
            } => Some(5), // Wait 5 seconds for server errors
            Self::TimeoutError(_) => Some(10),   // Wait 10 seconds for timeouts
            Self::ConnectionError(_) => Some(5), // Wait 5 seconds for connection errors
            _ => None,                           // No recommended delay for non-retryable errors
        }
    }

    /// Gets the maximum number of retry attempts recommended for this error.
    pub const fn max_retry_attempts(&self) -> u32 {
        match self {
            Self::RateLimitError(_) => 3,
            Self::ApiError { code: 429, .. } => 3,
            Self::ApiError {
                code: 500..=599, ..
            } => 5,
            Self::TimeoutError(_) => 3,
            Self::ConnectionError(_) => 3,
            _ => 0, // No retries for non-retryable errors
        }
    }
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, LlmError>;
