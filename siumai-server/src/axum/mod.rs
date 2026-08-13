//! Axum ingress and canonical response projection.

mod body;
mod response;
mod sse;

pub use body::{
    GatewayBodyReadError, GatewayBodyRole, read_request_body, read_request_json,
    read_upstream_body, read_upstream_json,
};
pub use response::{gateway_error_response, language_response, run_response};
pub use sse::{language_sse, language_sse_with_policy, run_sse, run_sse_with_policy};
