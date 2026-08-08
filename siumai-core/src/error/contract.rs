//! Sanitized, matchable errors for the Siumai model contracts.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use crate::provider::{ModelId, ModelOperation, ProviderId, RouteId};

const MAX_PUBLIC_DIAGNOSTIC_TEXT_BYTES: usize = 4096;
/// Largest provider-supplied retry delay retained in public diagnostics.
pub const MAX_RETRY_AFTER_HINT: Duration = Duration::from_secs(24 * 60 * 60);

/// Invalid text requested for a default logging or serialization surface.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DiagnosticTextError {
    #[error("public diagnostic text exceeds {maximum} bytes")]
    TooLong { maximum: usize },
    #[error("public diagnostic text contains control characters")]
    ControlCharacter,
}

/// Bounded text explicitly approved for default diagnostic surfaces.
///
/// This type enforces size and control-character safety, not semantic secrecy.
/// Provider response text, credentials, and payload fragments belong in
/// [`SensitiveResponse`] even when they pass these structural checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct PublicDiagnosticText(String);

impl PublicDiagnosticText {
    pub fn new(value: impl Into<String>) -> Result<Self, DiagnosticTextError> {
        let value = value.into();
        if value.len() > MAX_PUBLIC_DIAGNOSTIC_TEXT_BYTES {
            return Err(DiagnosticTextError::TooLong {
                maximum: MAX_PUBLIC_DIAGNOSTIC_TEXT_BYTES,
            });
        }
        if value.chars().any(char::is_control) {
            return Err(DiagnosticTextError::ControlCharacter);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PublicDiagnosticText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<&'static str> for PublicDiagnosticText {
    fn from(value: &'static str) -> Self {
        let mut sanitized = value
            .chars()
            .map(|character| {
                if character.is_control() {
                    ' '
                } else {
                    character
                }
            })
            .collect::<String>();
        sanitized.truncate(floor_char_boundary(
            &sanitized,
            MAX_PUBLIC_DIAGNOSTIC_TEXT_BYTES,
        ));
        Self(sanitized)
    }
}

impl<'de> Deserialize<'de> for PublicDiagnosticText {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Stable failure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidInput,
    Configuration,
    Authentication,
    Authorization,
    Unsupported,
    RateLimited,
    QuotaExceeded,
    ContextWindowExceeded,
    Timeout,
    Unavailable,
    Cancelled,
    Transport,
    Protocol,
    Provider,
    UnexpectedEof,
    ResponseLimit,
    LimitExceeded,
    PartialResult,
    ProtocolViolation,
    StructuredOutput,
    Tool,
    Internal,
}

/// A provider-neutral resource whose limit or result cardinality was violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResourceKind {
    EmbeddingInputs,
    EmbeddingTokens,
    RerankCandidates,
    ImageOutputs,
    SpeechTextBytes,
    SpeechTextCharacters,
    TranscriptionAudioBytes,
    TranscriptionDurationSeconds,
}

/// Structured, sanitized details for failures callers commonly branch on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ErrorDetail {
    LimitExceeded {
        resource: ResourceKind,
        actual: u64,
        maximum: u64,
    },
    PartialResult {
        resource: ResourceKind,
        expected: u64,
        actual: u64,
    },
}

/// Where a failure occurred.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorContext {
    pub operation: Option<ModelOperation>,
    pub provider: Option<ProviderId>,
    pub route: Option<RouteId>,
    pub model: Option<ModelId>,
}

/// Response details that are safe to display and serialize by default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseDiagnostics {
    status: Option<u16>,
    provider_code: Option<PublicDiagnosticText>,
    provider_type: Option<PublicDiagnosticText>,
    provider_param: Option<PublicDiagnosticText>,
    request_id: Option<PublicDiagnosticText>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "duration_millis"
    )]
    retry_after: Option<Duration>,
    body_truncated: bool,
}

impl ResponseDiagnostics {
    pub fn status(&self) -> Option<u16> {
        self.status
    }

    pub fn provider_code(&self) -> Option<&str> {
        self.provider_code
            .as_ref()
            .map(PublicDiagnosticText::as_str)
    }

    pub fn provider_type(&self) -> Option<&str> {
        self.provider_type
            .as_ref()
            .map(PublicDiagnosticText::as_str)
    }

    pub fn provider_param(&self) -> Option<&str> {
        self.provider_param
            .as_ref()
            .map(PublicDiagnosticText::as_str)
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_ref().map(PublicDiagnosticText::as_str)
    }

    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }

    pub fn body_truncated(&self) -> bool {
        self.body_truncated
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_provider_code(mut self, code: PublicDiagnosticText) -> Self {
        self.provider_code = Some(code);
        self
    }

    pub fn with_provider_type(mut self, provider_type: PublicDiagnosticText) -> Self {
        self.provider_type = Some(provider_type);
        self
    }

    pub fn with_provider_param(mut self, provider_param: PublicDiagnosticText) -> Self {
        self.provider_param = Some(provider_param);
        self
    }

    pub fn with_request_id(mut self, request_id: PublicDiagnosticText) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_retry_after(mut self, retry_after: Duration) -> Self {
        if retry_after <= MAX_RETRY_AFTER_HINT {
            self.retry_after = Some(retry_after);
        }
        self
    }

    /// Remove a retry hint when independent provider signals disagree.
    pub fn without_retry_after(mut self) -> Self {
        self.retry_after = None;
        self
    }

    pub fn with_body_truncated(mut self, body_truncated: bool) -> Self {
        self.body_truncated = body_truncated;
        self
    }
}

mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<u64>::deserialize(deserializer)?.map(Duration::from_millis);
        if value.is_some_and(|duration| duration > super::MAX_RETRY_AFTER_HINT) {
            return Err(serde::de::Error::custom(
                "response retry hint exceeds the maximum duration",
            ));
        }
        Ok(value)
    }
}

/// Explicitly sensitive response material excluded from default diagnostics.
pub struct SensitiveResponse {
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
    headers_truncated: bool,
    body_truncated: bool,
}

/// An underlying error retained for explicit inspection but redacted from
/// default source-chain reporting.
pub struct SensitiveErrorSource {
    source: Box<dyn StdError + Send + Sync + 'static>,
}

impl fmt::Debug for SensitiveErrorSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SensitiveErrorSource([REDACTED])")
    }
}

impl fmt::Display for SensitiveErrorSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("underlying error source is redacted")
    }
}

impl StdError for SensitiveErrorSource {}

impl SensitiveErrorSource {
    /// Explicitly expose the original source to a trusted caller.
    pub fn expose(&self) -> &(dyn StdError + Send + Sync + 'static) {
        self.source.as_ref()
    }
}

pub const DEFAULT_SENSITIVE_BODY_LIMIT: usize = 64 * 1024;
pub const DEFAULT_SENSITIVE_HEADER_COUNT_LIMIT: usize = 64;
pub const DEFAULT_SENSITIVE_HEADER_VALUE_LIMIT: usize = 4 * 1024;

impl fmt::Debug for SensitiveResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SensitiveResponse")
            .field("headers", &"[REDACTED]")
            .field("body", &"[REDACTED]")
            .field("body_bytes", &self.body.len())
            .field("headers_truncated", &self.headers_truncated)
            .field("body_truncated", &self.body_truncated)
            .finish()
    }
}

impl SensitiveResponse {
    pub fn new(headers: BTreeMap<String, String>, body: Vec<u8>) -> Self {
        Self::with_limit(headers, body, DEFAULT_SENSITIVE_BODY_LIMIT)
    }

    pub fn with_limit(
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
        body_maximum: usize,
    ) -> Self {
        Self::with_limits(
            headers,
            body,
            DEFAULT_SENSITIVE_HEADER_COUNT_LIMIT,
            DEFAULT_SENSITIVE_HEADER_VALUE_LIMIT,
            body_maximum,
        )
    }

    pub fn with_limits(
        headers: BTreeMap<String, String>,
        mut body: Vec<u8>,
        header_count_maximum: usize,
        header_value_maximum: usize,
        body_maximum: usize,
    ) -> Self {
        let mut bounded_headers = BTreeMap::new();
        let mut headers_truncated = headers.len() > header_count_maximum;
        for (name, mut value) in headers.into_iter().take(header_count_maximum) {
            if value.len() > header_value_maximum {
                value.truncate(floor_char_boundary(&value, header_value_maximum));
                headers_truncated = true;
            }
            bounded_headers.insert(name, value);
        }
        let body_truncated = body.len() > body_maximum;
        body.truncate(body_maximum);
        Self {
            headers: bounded_headers,
            body,
            headers_truncated,
            body_truncated,
        }
    }

    /// Explicitly expose raw response material to a trusted caller.
    pub fn expose(&self) -> (&BTreeMap<String, String>, &[u8]) {
        (&self.headers, &self.body)
    }

    pub fn was_truncated(&self) -> bool {
        self.headers_truncated || self.body_truncated
    }
}

fn floor_char_boundary(value: &str, maximum: usize) -> usize {
    let mut boundary = maximum.min(value.len());
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

/// The canonical model and runtime error.
pub struct Error {
    kind: ErrorKind,
    message: PublicDiagnosticText,
    context: Box<ErrorContext>,
    detail: Option<Box<ErrorDetail>>,
    diagnostics: Option<Box<ResponseDiagnostics>>,
    sensitive_response: Option<Box<SensitiveResponse>>,
    source: Option<Box<SensitiveErrorSource>>,
}

impl Error {
    /// Create an error with text approved for public diagnostics.
    ///
    /// String literals are accepted directly. Dynamic text must first pass
    /// [`PublicDiagnosticText::new`], making the logging decision explicit.
    pub fn new(kind: ErrorKind, message: impl Into<PublicDiagnosticText>) -> Self {
        Self {
            kind,
            message: message.into(),
            context: Box::new(ErrorContext::default()),
            detail: None,
            diagnostics: None,
            sensitive_response: None,
            source: None,
        }
    }

    pub fn unexpected_eof() -> Self {
        Self::new(
            ErrorKind::UnexpectedEof,
            "established stream ended without a protocol terminal event",
        )
    }

    pub fn cancelled(message: impl Into<PublicDiagnosticText>) -> Self {
        Self::new(ErrorKind::Cancelled, message)
    }

    pub fn limit_exceeded(resource: ResourceKind, actual: u64, maximum: u64) -> Self {
        Self::new(ErrorKind::LimitExceeded, "provider model limit exceeded").with_detail(
            ErrorDetail::LimitExceeded {
                resource,
                actual,
                maximum,
            },
        )
    }

    pub fn partial_result(resource: ResourceKind, expected: u64, actual: u64) -> Self {
        Self::new(
            ErrorKind::PartialResult,
            "provider returned an incomplete result",
        )
        .with_detail(ErrorDetail::PartialResult {
            resource,
            expected,
            actual,
        })
    }

    pub fn protocol_violation(message: impl Into<PublicDiagnosticText>) -> Self {
        Self::new(ErrorKind::ProtocolViolation, message)
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    pub fn context(&self) -> &ErrorContext {
        self.context.as_ref()
    }

    pub fn detail(&self) -> Option<&ErrorDetail> {
        self.detail.as_deref()
    }

    pub fn diagnostics(&self) -> Option<&ResponseDiagnostics> {
        self.diagnostics.as_deref()
    }

    /// Explicitly access raw response details. Never log this value implicitly.
    pub fn sensitive_response(&self) -> Option<&SensitiveResponse> {
        self.sensitive_response.as_deref()
    }

    /// Explicitly access the original error source. Never log it implicitly.
    pub fn sensitive_source(&self) -> Option<&SensitiveErrorSource> {
        self.source.as_deref()
    }

    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = Box::new(context);
        self
    }

    /// Attach the configured Registry route that handled this operation.
    ///
    /// Provider implementations normally do not know which application route
    /// selected their shared runtime, so the Registry adds this field at the
    /// outer model boundary. The rest of the provider/model context is kept
    /// intact.
    pub fn with_route(mut self, route: crate::RouteId) -> Self {
        self.context.route = Some(route);
        self
    }

    pub fn with_detail(mut self, detail: ErrorDetail) -> Self {
        self.detail = Some(Box::new(detail));
        self
    }

    pub fn with_diagnostics(mut self, diagnostics: ResponseDiagnostics) -> Self {
        self.diagnostics = Some(Box::new(diagnostics));
        self
    }

    pub fn with_sensitive_response(mut self, response: SensitiveResponse) -> Self {
        self.sensitive_response = Some(Box::new(response));
        self
    }

    pub fn with_source(mut self, source: impl StdError + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(SensitiveErrorSource {
            source: Box::new(source),
        }));
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Error")
            .field("kind", &self.kind)
            .field("message", &self.message)
            .field("context", &self.context)
            .field("detail", &self.detail)
            .field("diagnostics", &self.diagnostics)
            .field(
                "sensitive_response",
                &self.sensitive_response.as_ref().map(|_| "[REDACTED]"),
            )
            .field("has_source", &self.source.is_some())
            .finish()
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Error", 5)?;
        state.serialize_field("kind", &self.kind)?;
        state.serialize_field("message", &self.message)?;
        state.serialize_field("context", &self.context)?;
        state.serialize_field("detail", &self.detail)?;
        state.serialize_field("diagnostics", &self.diagnostics)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_surfaces_do_not_expose_raw_response_material() {
        let error = Error::new(ErrorKind::Provider, "provider rejected the request")
            .with_sensitive_response(SensitiveResponse::new(
                BTreeMap::from([("authorization".to_string(), "secret-header".to_string())]),
                b"secret-body".to_vec(),
            ));

        let debug = format!("{error:?}");
        let display = error.to_string();
        let serialized = serde_json::to_string(&error).unwrap();
        for surface in [debug, display, serialized] {
            assert!(!surface.contains("secret-header"));
            assert!(!surface.contains("secret-body"));
        }
    }

    #[test]
    fn response_diagnostics_preserve_matchable_provider_fields() {
        let diagnostics = ResponseDiagnostics::default()
            .with_provider_code(PublicDiagnosticText::new("invalid_parameter").unwrap())
            .with_provider_type(PublicDiagnosticText::new("invalid_request_error").unwrap())
            .with_provider_param(PublicDiagnosticText::new("thinking.keep").unwrap())
            .with_retry_after(Duration::from_secs(3));

        assert_eq!(diagnostics.provider_code(), Some("invalid_parameter"));
        assert_eq!(diagnostics.provider_type(), Some("invalid_request_error"));
        assert_eq!(diagnostics.provider_param(), Some("thinking.keep"));
        assert_eq!(diagnostics.retry_after(), Some(Duration::from_secs(3)));
        assert_eq!(
            diagnostics
                .clone()
                .with_retry_after(MAX_RETRY_AFTER_HINT + Duration::from_secs(1))
                .retry_after(),
            Some(Duration::from_secs(3))
        );
        assert_eq!(diagnostics.without_retry_after().retry_after(), None);
        assert!(
            serde_json::from_value::<ResponseDiagnostics>(serde_json::json!({
                "retry_after": (MAX_RETRY_AFTER_HINT + Duration::from_secs(1)).as_millis()
            }))
            .is_err()
        );
    }

    #[test]
    fn sensitive_response_requires_explicit_access() {
        let error = Error::new(ErrorKind::Provider, "safe")
            .with_sensitive_response(SensitiveResponse::new(BTreeMap::new(), b"raw".to_vec()));
        assert_eq!(error.sensitive_response().unwrap().expose().1, b"raw");
    }

    #[test]
    fn sensitive_response_is_bounded_at_construction() {
        let response = SensitiveResponse::with_limit(BTreeMap::new(), vec![7; 16], 4);
        assert_eq!(response.expose().1, &[7; 4]);
        assert!(response.was_truncated());
    }

    #[test]
    fn limit_details_are_matchable_and_safe_to_serialize() {
        let error = Error::limit_exceeded(ResourceKind::ImageOutputs, 5, 4);

        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        assert_eq!(
            error.detail(),
            Some(&ErrorDetail::LimitExceeded {
                resource: ResourceKind::ImageOutputs,
                actual: 5,
                maximum: 4,
            })
        );
        let serialized = serde_json::to_value(&error).unwrap();
        assert_eq!(serialized["detail"]["LimitExceeded"]["maximum"], 4);
    }

    #[test]
    fn sensitive_response_bounds_raw_headers_as_well_as_body() {
        let response = SensitiveResponse::with_limits(
            BTreeMap::from([
                ("first".to_string(), "long-value".to_string()),
                ("second".to_string(), "ignored".to_string()),
            ]),
            Vec::new(),
            1,
            4,
            0,
        );
        let (headers, _) = response.expose();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers["first"], "long");
        assert!(response.was_truncated());
    }

    #[test]
    fn default_source_chain_is_redacted_but_explicit_source_is_retained() {
        #[derive(Debug, thiserror::Error)]
        #[error("secret source payload")]
        struct SecretSource;

        let error =
            Error::new(ErrorKind::Transport, "safe transport failure").with_source(SecretSource);
        let default_source = StdError::source(&error).unwrap().to_string();
        let explicit_source = error.sensitive_source().unwrap().expose().to_string();

        assert!(!default_source.contains("secret source payload"));
        assert!(explicit_source.contains("secret source payload"));
    }

    #[test]
    fn dynamic_default_diagnostic_text_requires_bounded_control_free_opt_in() {
        assert!(PublicDiagnosticText::new("provider\nraw body").is_err());
        assert!(
            PublicDiagnosticText::new("x".repeat(MAX_PUBLIC_DIAGNOSTIC_TEXT_BYTES + 1)).is_err()
        );

        let approved = PublicDiagnosticText::new("request rejected").unwrap();
        let error = Error::new(ErrorKind::Provider, approved);
        assert_eq!(error.message(), "request rejected");
    }

    #[test]
    fn route_context_preserves_provider_context() {
        let error = Error::new(ErrorKind::Provider, "request failed").with_context(ErrorContext {
            operation: Some(ModelOperation::Generate),
            provider: Some(ProviderId::new("openai").unwrap()),
            route: None,
            model: Some(ModelId::new("gpt-5.6-sol").unwrap()),
        });

        let error = error.with_route(RouteId::new("production").unwrap());
        assert_eq!(
            error.context().route.as_ref().map(RouteId::as_str),
            Some("production")
        );
        assert_eq!(
            error.context().provider.as_ref().map(ProviderId::as_str),
            Some("openai")
        );
        assert_eq!(
            error.context().model.as_ref().map(ModelId::as_str),
            Some("gpt-5.6-sol")
        );
    }
}
