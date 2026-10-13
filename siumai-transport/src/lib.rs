//! Provider transport with bounded I/O, explicit replay proof, and endpoint isolation.
//!
//! This crate owns transport mechanics. Protocol crates own wire codecs and
//! provider crates own authentication, model policy, and provider-specific
//! error interpretation.
#![deny(unsafe_code)]

mod auth;
mod endpoint;
mod error;
pub mod framing;
mod limits;
mod proxy;
mod replay;
mod request;
mod resource;
mod settings;
mod transport;
mod websocket;

/// Experimental provider-author integration seams.
///
/// These APIs are intentionally kept outside the stable transport root and
/// facade re-exports. They are for provider crates that need to compose an
/// already configured transport without taking ownership of provider
/// credentials, profile policy, or complete runtime identity. The current
/// seam is limited to configured-versus-prebuilt source resolution for the
/// compatibility engines.
pub mod experimental;

pub use auth::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, CredentialRevision, NoAuth,
};
pub use endpoint::{
    CredentialAudience, EndpointConfig, EndpointPolicy, LocalNetworkGrant, OfficialOrigin,
    Resolver, SystemResolver,
};
pub use error::{EndpointError, RequestBuildError, TransportConfigError};
pub use limits::TransportLimits;
pub use proxy::{HttpTransportRoute, ProxyBasicCredential, ProxyEndpoint};
pub use replay::{IdempotencyHeader, ReplaySafety, RetryPolicy, TransportRetryPolicyError};
pub use request::{
    MultipartBody, MultipartPart, RequestBody, RequestHeaders, RequestPlan, RequestTarget,
};
pub use resource::{
    DownloadedResource, ResourceDownloadOptions, ResourceDownloader, ResourceDownloaderBuilder,
    ResourceProvenance, ResourceUrl, ResourceUrlError,
};
pub use settings::ProviderHttpTransportSettings;
pub use transport::{
    AttemptLoopOutcome, ProviderTransport, ProviderTransportBuilder, ResponseHeaders,
    RetryClassifier, RetryLimit, RetryReason, TransportByteStream, TransportCallId, TransportEvent,
    TransportObserver, TransportResponse, TransportStreamResponse,
};
pub use websocket::{
    WebSocketConnection, WebSocketEndpoint, WebSocketReceiver, WebSocketSender, WebSocketTransport,
    WebSocketTransportBuilder,
};
