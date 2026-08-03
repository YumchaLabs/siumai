//! Public configuration and request-construction errors.

use thiserror::Error;

/// Endpoint validation or resolution failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum EndpointError {
    #[error("endpoint URL is invalid")]
    InvalidUrl,
    #[error("endpoint URL scheme is not allowed by its policy")]
    SchemeNotAllowed,
    #[error("endpoint URL must include a host")]
    MissingHost,
    #[error("endpoint URL must not contain user information")]
    UserInfoNotAllowed,
    #[error("endpoint URL must not contain a fragment")]
    FragmentNotAllowed,
    #[error("endpoint hostname is not allowed by its policy")]
    HostNotAllowed,
    #[error("endpoint resolved to an address forbidden by its policy")]
    AddressNotAllowed,
    #[error("endpoint DNS resolution failed")]
    ResolutionFailed,
    #[error("endpoint DNS resolution returned no addresses")]
    NoAddresses,
    #[error("credential audience does not match the endpoint origin")]
    AudienceMismatch,
    #[error("official provider origin is invalid")]
    InvalidOfficialOrigin,
    #[error("endpoint does not match its provider-owned official origin")]
    OfficialOriginMismatch,
}

/// Invalid transport settings.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TransportConfigError {
    #[error("transport limit `{name}` must be greater than zero")]
    ZeroLimit { name: &'static str },
    #[error("maximum in-flight requests cannot exceed maximum connections")]
    InFlightExceedsConnections,
    #[error("transport limit `{name}` exceeds its hard maximum of {maximum}")]
    LimitTooLarge { name: &'static str, maximum: usize },
    #[error("HTTP client construction failed")]
    ClientBuild,
    #[error("transport queue and in-flight capacities overflow")]
    CapacityOverflow,
    #[error("transport admission capacity exceeds its hard maximum of {maximum}")]
    AdmissionCapacityTooLarge { maximum: usize },
    #[error("transport timeout `{name}` must be greater than zero")]
    ZeroTimeout { name: &'static str },
    #[error("transport timeout `{name}` is too large for this platform")]
    TimeoutTooLarge { name: &'static str },
}

/// Invalid immutable request plan.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RequestBuildError {
    #[error("request target is invalid")]
    InvalidTarget,
    #[error("request target exceeds {maximum} bytes")]
    TargetTooLong { maximum: usize },
    #[error("request header is controlled by the transport")]
    ProtectedHeader,
    #[error("request header value is invalid")]
    InvalidHeaderValue,
    #[error("credential query parameter is invalid")]
    InvalidCredentialQuery,
    #[error("credential query exceeds {maximum} parameters")]
    TooManyCredentialQueryParameters { maximum: usize },
    #[error("encoded credential query exceeds {maximum} bytes")]
    CredentialQueryTooLarge { maximum: usize },
    #[error("authenticated request URL exceeds {maximum} bytes")]
    AuthenticatedUrlTooLong { maximum: usize },
    #[error("request has too many headers")]
    TooManyHeaders,
    #[error("request header value is too large")]
    HeaderValueTooLarge,
    #[error("request body exceeds {maximum} bytes")]
    BodyTooLarge { maximum: usize },
    #[error("multipart body exceeds {maximum} parts")]
    TooManyMultipartParts { maximum: usize },
    #[error("multipart field metadata is invalid")]
    InvalidMultipartMetadata,
    #[error("request JSON serialization failed")]
    JsonSerialization,
    #[error("idempotency header is invalid")]
    InvalidIdempotencyHeader,
    #[error("idempotency header conflicts with another request header")]
    IdempotencyHeaderConflict,
}
