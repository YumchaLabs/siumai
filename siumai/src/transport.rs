//! Curated provider HTTP transport configuration and observation.
//!
//! This namespace exposes the Direct configuration values accepted by facade
//! provider builders. Transport execution, authentication, request plans, raw
//! responses, resource downloaders, and socket primitives remain available
//! only from their owning crates.

pub use siumai_transport::{
    AttemptLoopOutcome, EndpointConfig, EndpointError, EndpointPolicy, LocalNetworkGrant,
    OfficialOrigin, ProviderHttpTransportSettings, RetryLimit, RetryPolicy, RetryReason,
    TransportCallId, TransportConfigError, TransportEvent, TransportLimits, TransportObserver,
    TransportRetryPolicyError,
};
