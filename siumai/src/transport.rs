//! Curated provider HTTP transport configuration and observation.
//!
//! This namespace exposes the curated stateless-HTTP configuration accepted by
//! facade provider builders, including the Direct default and an explicit
//! trusted CONNECT route. Transport execution, authentication, request plans,
//! raw responses, resource downloaders, and socket primitives remain available
//! only from their owning crates.

pub use siumai_transport::{
    AttemptLoopOutcome, EndpointConfig, EndpointError, EndpointPolicy, HttpTransportRoute,
    LocalNetworkGrant, OfficialOrigin, ProviderHttpTransportSettings, ProxyBasicCredential,
    ProxyEndpoint, RetryLimit, RetryPolicy, RetryReason, TransportCallId, TransportConfigError,
    TransportEvent, TransportLimits, TransportObserver, TransportRetryPolicyError,
};
