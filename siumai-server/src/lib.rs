//! Bounded server projections over Siumai's provider-neutral runtime.
//!
//! This crate owns the downstream trust boundary, finite ingress limits, safe
//! HTTP responses, and canonical stream projection. It does not own a second
//! tool loop or provider-native protocol codec.

#![deny(unsafe_code)]

#[cfg(feature = "axum")]
pub mod axum;

mod event;
mod gateway;
mod policy;
mod trust;

pub use event::{GatewayEvent, GatewayLoss, GatewayProjectionError};
pub use gateway::{ServerGateway, ServerGatewayError, TrustedApprovalDecider};
pub use policy::{
    DEFAULT_JSON_RESPONSE_LIMIT_BYTES, DEFAULT_REQUEST_BODY_LIMIT_BYTES,
    DEFAULT_SSE_EVENT_LIMIT_BYTES, DEFAULT_UPSTREAM_BODY_LIMIT_BYTES, GatewayErrorDetail,
    GatewayHeaderPolicy, GatewayLimits, GatewayLossPolicy, GatewayPolicy, GatewayPolicyError,
    GatewayStreamPolicy, MAX_RESPONSE_HEADER_RULES, MAX_SERVER_BODY_LIMIT_BYTES,
    MIN_SERVER_RESPONSE_LIMIT_BYTES,
};
pub use trust::ServerTrustContext;
