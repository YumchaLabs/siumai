use std::collections::BTreeMap;

use serde_json::Value;
use siumai_core::{
    Error, ErrorKind, InvalidId, InvalidToolCall, LanguageRequestError, LanguageResponseError,
    OpaqueProviderItemError, ProviderAnnotationError, ProviderProvenanceError,
    PublicDiagnosticText, ResponseDiagnostics, SensitiveResponse,
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
    #[error("Anthropic Messages replay scope is incomplete")]
    InvalidProvenance(#[source] ProviderProvenanceError),
    #[error("Anthropic Messages returned an invalid model identifier")]
    InvalidModelId(#[source] InvalidId),
    #[error("Anthropic Messages returned an invalid canonical tool call")]
    InvalidToolCall(#[source] InvalidToolCall),
    #[error("Anthropic Messages streamed tool input exceeded {maximum} bytes")]
    ToolInputTooLarge { maximum: usize },
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
            | Self::ConflictingCacheAnnotation
            | Self::InvalidProvenance(_) => ErrorKind::InvalidInput,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::ProtocolViolation { .. }
            | Self::JsonDecode(_)
            | Self::InvalidCanonicalResponse(_)
            | Self::InvalidOpaqueItem(_)
            | Self::InvalidModelId(_)
            | Self::InvalidToolCall(_) => ErrorKind::Protocol,
            Self::ToolInputTooLarge { .. } => ErrorKind::ResponseLimit,
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
            Self::InvalidProvenance(_) => {
                "Anthropic Messages replay requires an explicit provider replay domain"
            }
            Self::InvalidModelId(_) => "provider returned an invalid model identifier",
            Self::InvalidToolCall(_) => {
                "provider returned a tool call that violated the canonical contract"
            }
            Self::ToolInputTooLarge { .. } => {
                "Anthropic Messages streamed tool input exceeded the byte limit"
            }
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

pub(crate) fn classify_stream_failure(
    envelope: &Value,
    mut diagnostics: ResponseDiagnostics,
) -> Result<Error, MessagesCodecError> {
    let error = envelope.get("error").and_then(Value::as_object).ok_or(
        MessagesCodecError::ProtocolViolation {
            reason: "stream error event omitted its error object",
        },
    )?;
    let error_type = error
        .get("type")
        .and_then(Value::as_str)
        .and_then(public_provider_identifier)
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "stream error event omitted a bounded error type",
        })?;
    let kind = classify_error_type(error_type.as_str());
    diagnostics = diagnostics.with_provider_type(error_type);
    if let Some(request_id) = envelope
        .get("request_id")
        .and_then(Value::as_str)
        .and_then(public_provider_identifier)
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    let body = serde_json::to_vec(envelope).map_err(MessagesCodecError::JsonEncode)?;
    let sensitive = SensitiveResponse::new(BTreeMap::new(), body);
    let body_truncated = diagnostics.body_truncated() || sensitive.was_truncated();
    Ok(
        Error::new(kind, "Anthropic Messages stream reported a provider error")
            .with_diagnostics(diagnostics.with_body_truncated(body_truncated))
            .with_sensitive_response(sensitive),
    )
}

fn classify_error_type(error_type: &str) -> ErrorKind {
    match error_type {
        "invalid_request_error" | "not_found_error" => ErrorKind::InvalidInput,
        "authentication_error" => ErrorKind::Authentication,
        "billing_error" => ErrorKind::QuotaExceeded,
        "permission_error" => ErrorKind::Authorization,
        "request_too_large" => ErrorKind::LimitExceeded,
        "rate_limit_error" => ErrorKind::RateLimited,
        "timeout_error" => ErrorKind::Timeout,
        "api_error" | "overloaded_error" => ErrorKind::Unavailable,
        _ => ErrorKind::Provider,
    }
}

fn public_provider_identifier(value: &str) -> Option<PublicDiagnosticText> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return None;
    }
    PublicDiagnosticText::new(value.to_string()).ok()
}
