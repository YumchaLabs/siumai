use http::header::{CONTENT_TYPE, HeaderName, HeaderValue};
use serde_json::Value;
use siumai_core::{Error, ErrorKind, ModelId};
use siumai_transport::{RequestBuildError, RequestHeaders, RequestTarget};

const ANTHROPIC_VERSION: HeaderName = HeaderName::from_static("anthropic-version");
const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");

/// Read-only inputs available while projecting one encoded Messages request.
///
/// The context exposes protocol addressing inputs without transferring endpoint,
/// authentication, transport, or replay ownership to the projection.
#[derive(Debug, Clone, Copy)]
pub struct MessagesRequestProjectionContext<'a> {
    model: &'a ModelId,
    stream: bool,
    api_version: &'a str,
    default_target: &'a RequestTarget,
    beta_header: Option<&'a str>,
}

impl<'a> MessagesRequestProjectionContext<'a> {
    pub(crate) fn new(
        model: &'a ModelId,
        stream: bool,
        api_version: &'a str,
        default_target: &'a RequestTarget,
        beta_header: Option<&'a str>,
    ) -> Self {
        Self {
            model,
            stream,
            api_version,
            default_target,
            beta_header,
        }
    }

    pub fn model(&self) -> &ModelId {
        self.model
    }

    pub const fn is_streaming(&self) -> bool {
        self.stream
    }

    pub const fn stream(&self) -> bool {
        self.stream
    }

    pub const fn api_version(&self) -> &str {
        self.api_version
    }

    pub const fn default_target(&self) -> &RequestTarget {
        self.default_target
    }

    /// Return the validated, merged `anthropic-beta` value for this call.
    pub const fn beta_header(&self) -> Option<&str> {
        self.beta_header
    }
}

/// Bounded wire projection returned to the compatible execution engine.
///
/// The target remains relative, headers cannot carry authentication or
/// transport-owned fields, and JSON content type remains engine-owned.
pub struct ProjectedMessagesRequest {
    target: RequestTarget,
    body: Value,
    headers: RequestHeaders,
}

impl ProjectedMessagesRequest {
    pub fn new(target: RequestTarget, body: Value, headers: RequestHeaders) -> Result<Self, Error> {
        validate_projection_headers(&headers)?;
        Ok(Self {
            target,
            body,
            headers,
        })
    }

    /// Construct a projected request while validating a caller-computed target.
    pub fn try_new(
        target: impl Into<String>,
        body: Value,
        headers: RequestHeaders,
    ) -> Result<Self, Error> {
        let target = RequestTarget::new(target.into()).map_err(projection_contract_error)?;
        Self::new(target, body, headers)
    }

    pub fn target(&self) -> &RequestTarget {
        &self.target
    }

    pub const fn body(&self) -> &Value {
        &self.body
    }

    pub const fn headers(&self) -> &RequestHeaders {
        &self.headers
    }

    pub fn into_parts(self) -> (RequestTarget, Value, RequestHeaders) {
        (self.target, self.body, self.headers)
    }
}

impl std::fmt::Debug for ProjectedMessagesRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProjectedMessagesRequest")
            .field("target", &self.target)
            .field("body", &"[REDACTED]")
            .field("headers", &self.headers)
            .finish()
    }
}

/// Provider- or dialect-owned projection of one canonical Messages request.
///
/// Implementations may select a relative operation target, place a protocol
/// version in the body instead of a header, and add non-protected protocol
/// headers. They cannot change the endpoint, HTTP method, authentication,
/// content type, transport policy, or replay semantics.
pub trait MessagesRequestProjection: Send + Sync {
    fn project(
        &self,
        context: &MessagesRequestProjectionContext<'_>,
        body: Value,
    ) -> Result<ProjectedMessagesRequest, Error>;
}

/// Native Anthropic Messages projection used by default.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeMessagesRequestProjection;

impl MessagesRequestProjection for NativeMessagesRequestProjection {
    fn project(
        &self,
        context: &MessagesRequestProjectionContext<'_>,
        body: Value,
    ) -> Result<ProjectedMessagesRequest, Error> {
        let mut headers = RequestHeaders::new()
            .try_insert(
                ANTHROPIC_VERSION,
                HeaderValue::from_str(context.api_version()).map_err(invalid_profile_header)?,
            )
            .map_err(projection_contract_error)?;
        if let Some(beta_header) = context.beta_header() {
            headers = headers
                .try_insert(
                    ANTHROPIC_BETA,
                    HeaderValue::from_str(beta_header).map_err(invalid_profile_header)?,
                )
                .map_err(projection_contract_error)?;
        }
        ProjectedMessagesRequest::new(context.default_target().clone(), body, headers)
    }
}

fn validate_projection_headers(headers: &RequestHeaders) -> Result<(), Error> {
    if headers.get(&CONTENT_TYPE).is_some() {
        return Err(projection_contract_error(
            RequestBuildError::ProtectedHeader,
        ));
    }
    Ok(())
}

fn projection_contract_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Messages request projection violates the transport contract",
    )
    .with_source(source)
}

fn invalid_profile_header(source: http::header::InvalidHeaderValue) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "configured Anthropic Messages projection header is invalid",
    )
    .with_source(source)
}
