use siumai_core::{
    Error, ErrorKind, InvalidId, LanguageRequestError, LanguageResponseError,
    OpaqueProviderItemError, ProviderAnnotationError,
};
use thiserror::Error;

/// Typed failure produced by the Anthropic Messages codec.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MessagesCodecError {
    #[error("language request is invalid")]
    InvalidLanguageRequest(#[source] LanguageRequestError),
    #[error("Anthropic Messages cannot encode {feature}")]
    Unsupported { feature: &'static str },
    #[error("Anthropic Messages option contains protected field `{path}`")]
    ProtectedOptionField { path: String },
    #[error("Anthropic Messages option `{field}` is invalid: {reason}")]
    InvalidOption {
        field: &'static str,
        reason: &'static str,
    },
    #[error("Anthropic annotation on {node} is invalid")]
    InvalidAnnotation {
        node: &'static str,
        #[source]
        source: ProviderAnnotationError,
    },
    #[error("Anthropic prompt caching accepts at most {maximum} breakpoints; found {actual}")]
    TooManyCacheBreakpoints { actual: usize, maximum: usize },
    #[error("Anthropic prompt-cache TTL order must place 1h entries before 5m entries")]
    InvalidCacheTtlOrder,
    #[error("Anthropic cache annotation conflicts with another annotation on the same wire block")]
    ConflictingCacheAnnotation,
    #[error("Anthropic Messages response violated the protocol: {reason}")]
    ProtocolViolation { reason: &'static str },
    #[error("Anthropic Messages response contained malformed JSON")]
    JsonDecode(#[source] serde_json::Error),
    #[error("Anthropic Messages request could not be serialized")]
    JsonEncode(#[source] serde_json::Error),
    #[error("Anthropic Messages produced an invalid canonical response")]
    InvalidCanonicalResponse(#[source] LanguageResponseError),
    #[error("Anthropic Messages native content could not be retained")]
    InvalidOpaqueItem(#[source] OpaqueProviderItemError),
    #[error("Anthropic Messages returned an invalid model identifier")]
    InvalidModelId(#[source] InvalidId),
    #[error("Anthropic Messages stream ended without message_stop")]
    UnexpectedEof,
}

impl MessagesCodecError {
    pub const fn error_kind(&self) -> ErrorKind {
        match self {
            Self::InvalidLanguageRequest(_)
            | Self::ProtectedOptionField { .. }
            | Self::InvalidOption { .. }
            | Self::InvalidAnnotation { .. }
            | Self::TooManyCacheBreakpoints { .. }
            | Self::InvalidCacheTtlOrder
            | Self::ConflictingCacheAnnotation => ErrorKind::InvalidInput,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::ProtocolViolation { .. }
            | Self::JsonDecode(_)
            | Self::InvalidCanonicalResponse(_)
            | Self::InvalidOpaqueItem(_)
            | Self::InvalidModelId(_) => ErrorKind::Protocol,
            Self::JsonEncode(_) => ErrorKind::Internal,
            Self::UnexpectedEof => ErrorKind::UnexpectedEof,
        }
    }

    const fn public_message(&self) -> &'static str {
        match self {
            Self::InvalidLanguageRequest(_) => "language request is invalid",
            Self::Unsupported { .. } => "request uses an unsupported Anthropic Messages feature",
            Self::ProtectedOptionField { .. } => {
                "Anthropic Messages options contain a protected field"
            }
            Self::InvalidOption { .. } => "Anthropic Messages options are invalid",
            Self::InvalidAnnotation { .. } => "Anthropic Messages annotation is invalid",
            Self::TooManyCacheBreakpoints { .. } => {
                "Anthropic Messages prompt-cache breakpoint limit exceeded"
            }
            Self::InvalidCacheTtlOrder => "Anthropic Messages prompt-cache TTL order is invalid",
            Self::ConflictingCacheAnnotation => "Anthropic Messages cache annotations conflict",
            Self::ProtocolViolation { .. } => {
                "provider returned an invalid Anthropic Messages response"
            }
            Self::JsonDecode(_) => "provider returned malformed Anthropic Messages JSON",
            Self::JsonEncode(_) => "failed to encode Anthropic Messages request",
            Self::InvalidCanonicalResponse(_) => {
                "Anthropic Messages response could not satisfy the canonical response contract"
            }
            Self::InvalidOpaqueItem(_) => {
                "Anthropic Messages native content could not be retained safely"
            }
            Self::InvalidModelId(_) => "provider returned an invalid model identifier",
            Self::UnexpectedEof => "established stream ended without a protocol terminal event",
        }
    }
}

impl From<MessagesCodecError> for Error {
    fn from(error: MessagesCodecError) -> Self {
        if matches!(&error, MessagesCodecError::UnexpectedEof) {
            return Self::unexpected_eof();
        }
        Self::new(error.error_kind(), error.public_message()).with_source(error)
    }
}
