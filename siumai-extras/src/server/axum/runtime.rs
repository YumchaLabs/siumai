//! Axum runtime helpers for gateway policy enforcement.
//!
//! English-only comments in code as requested.

use std::error::Error as _;

use axum::{
    body::{Body, Bytes, to_bytes},
    http::StatusCode,
    response::Response,
};
use http_body_util::LengthLimitError;
use serde::de::DeserializeOwned;

use crate::server::GatewayBridgePolicy;

/// Body ownership context for policy-driven reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayBodyRole {
    /// Bytes coming from the downstream client into the gateway route.
    DownstreamRequest,
    /// Bytes coming from an upstream provider into the gateway runtime.
    UpstreamResponse,
}

/// Policy-aware body read error.
#[derive(Debug)]
pub enum GatewayBodyReadError {
    /// The configured read limit was exceeded.
    LimitExceeded {
        /// Which side of the gateway owned the body.
        role: GatewayBodyRole,
        /// Effective byte limit that was exceeded.
        limit_bytes: usize,
    },
    /// The body could not be read successfully.
    ReadFailed {
        /// Which side of the gateway owned the body.
        role: GatewayBodyRole,
        /// Low-level read failure detail.
        detail: String,
    },
    /// The body bytes were read, but JSON decoding failed.
    InvalidJson {
        /// Which side of the gateway owned the body.
        role: GatewayBodyRole,
        /// JSON parse failure detail.
        detail: String,
    },
}

impl GatewayBodyReadError {
    /// Return the HTTP status code recommended for this error.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::LimitExceeded {
                role: GatewayBodyRole::DownstreamRequest,
                ..
            } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::LimitExceeded {
                role: GatewayBodyRole::UpstreamResponse,
                ..
            }
            | Self::ReadFailed {
                role: GatewayBodyRole::UpstreamResponse,
                ..
            }
            | Self::InvalidJson {
                role: GatewayBodyRole::UpstreamResponse,
                ..
            } => StatusCode::BAD_GATEWAY,
            Self::ReadFailed {
                role: GatewayBodyRole::DownstreamRequest,
                ..
            }
            | Self::InvalidJson {
                role: GatewayBodyRole::DownstreamRequest,
                ..
            } => StatusCode::BAD_REQUEST,
        }
    }

    /// Return a route-safe message, respecting policy error passthrough.
    pub fn user_message(&self, policy: &GatewayBridgePolicy) -> String {
        if policy.passthrough_runtime_errors {
            match self {
                Self::LimitExceeded { role, limit_bytes } => match role {
                    GatewayBodyRole::DownstreamRequest => {
                        format!("request body exceeded limit of {limit_bytes} bytes")
                    }
                    GatewayBodyRole::UpstreamResponse => {
                        format!("upstream body exceeded limit of {limit_bytes} bytes")
                    }
                },
                Self::ReadFailed { detail, .. } | Self::InvalidJson { detail, .. } => {
                    detail.clone()
                }
            }
        } else {
            match self {
                Self::LimitExceeded { role, .. } => match role {
                    GatewayBodyRole::DownstreamRequest => "request body too large".to_string(),
                    GatewayBodyRole::UpstreamResponse => "upstream body too large".to_string(),
                },
                Self::ReadFailed { role, .. } => match role {
                    GatewayBodyRole::DownstreamRequest => {
                        "failed to read downstream request body".to_string()
                    }
                    GatewayBodyRole::UpstreamResponse => {
                        "failed to read upstream response body".to_string()
                    }
                },
                Self::InvalidJson { role, .. } => match role {
                    GatewayBodyRole::DownstreamRequest => {
                        "invalid downstream request json".to_string()
                    }
                    GatewayBodyRole::UpstreamResponse => {
                        "invalid upstream response json".to_string()
                    }
                },
            }
        }
    }

    /// Convert the error into a plain-text Axum response.
    pub fn to_response(&self, policy: &GatewayBridgePolicy) -> Response<Body> {
        Response::builder()
            .status(self.status_code())
            .header("content-type", "text/plain; charset=utf-8")
            .body(Body::from(self.user_message(policy)))
            .unwrap_or_else(|_| Response::new(Body::from("internal error")))
    }
}

/// Read a downstream request body under the policy-configured limit.
pub async fn read_request_body_with_policy(
    body: Body,
    policy: &GatewayBridgePolicy,
) -> Result<Bytes, GatewayBodyReadError> {
    read_body_with_limit(
        body,
        policy.request_body_limit_bytes,
        GatewayBodyRole::DownstreamRequest,
    )
    .await
}

/// Read a downstream request JSON body under the policy-configured limit.
pub async fn read_request_json_with_policy<T>(
    body: Body,
    policy: &GatewayBridgePolicy,
) -> Result<T, GatewayBodyReadError>
where
    T: DeserializeOwned,
{
    read_json_with_limit(
        body,
        policy.request_body_limit_bytes,
        GatewayBodyRole::DownstreamRequest,
    )
    .await
}

/// Read an upstream response body under the policy-configured limit.
pub async fn read_upstream_body_with_policy(
    body: Body,
    policy: &GatewayBridgePolicy,
) -> Result<Bytes, GatewayBodyReadError> {
    read_body_with_limit(
        body,
        policy.upstream_read_limit_bytes,
        GatewayBodyRole::UpstreamResponse,
    )
    .await
}

/// Read an upstream response JSON body under the policy-configured limit.
pub async fn read_upstream_json_with_policy<T>(
    body: Body,
    policy: &GatewayBridgePolicy,
) -> Result<T, GatewayBodyReadError>
where
    T: DeserializeOwned,
{
    read_json_with_limit(
        body,
        policy.upstream_read_limit_bytes,
        GatewayBodyRole::UpstreamResponse,
    )
    .await
}

async fn read_json_with_limit<T>(
    body: Body,
    limit: Option<usize>,
    role: GatewayBodyRole,
) -> Result<T, GatewayBodyReadError>
where
    T: DeserializeOwned,
{
    let bytes = read_body_with_limit(body, limit, role).await?;
    serde_json::from_slice(&bytes).map_err(|error| GatewayBodyReadError::InvalidJson {
        role,
        detail: error.to_string(),
    })
}

async fn read_body_with_limit(
    body: Body,
    limit: Option<usize>,
    role: GatewayBodyRole,
) -> Result<Bytes, GatewayBodyReadError> {
    let limit = limit.unwrap_or(usize::MAX);
    to_bytes(body, limit).await.map_err(|error| {
        if error
            .source()
            .is_some_and(|source| source.is::<LengthLimitError>())
        {
            GatewayBodyReadError::LimitExceeded {
                role,
                limit_bytes: limit,
            }
        } else {
            GatewayBodyReadError::ReadFailed {
                role,
                detail: error.to_string(),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde::Deserialize;
    use siumai::experimental::bridge::BridgeMode;

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Payload {
        message: String,
    }

    #[tokio::test]
    async fn request_body_limit_is_enforced() {
        let policy =
            GatewayBridgePolicy::new(BridgeMode::BestEffort).with_request_body_limit_bytes(4);

        let error = read_request_body_with_policy(Body::from("12345"), &policy)
            .await
            .expect_err("request body should exceed policy limit");

        assert!(matches!(
            error,
            GatewayBodyReadError::LimitExceeded {
                role: GatewayBodyRole::DownstreamRequest,
                limit_bytes: 4,
            }
        ));
        assert_eq!(error.status_code(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn upstream_body_limit_is_enforced() {
        let policy =
            GatewayBridgePolicy::new(BridgeMode::BestEffort).with_upstream_read_limit_bytes(4);

        let error = read_upstream_body_with_policy(Body::from("12345"), &policy)
            .await
            .expect_err("upstream body should exceed policy limit");

        assert!(matches!(
            error,
            GatewayBodyReadError::LimitExceeded {
                role: GatewayBodyRole::UpstreamResponse,
                limit_bytes: 4,
            }
        ));
        assert_eq!(error.status_code(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn request_json_is_parsed_under_limit() {
        let policy =
            GatewayBridgePolicy::new(BridgeMode::BestEffort).with_request_body_limit_bytes(64);

        let payload: Payload =
            read_request_json_with_policy(Body::from(r#"{"message":"hello"}"#), &policy)
                .await
                .expect("request json");

        assert_eq!(
            payload,
            Payload {
                message: "hello".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn request_json_error_maps_to_bad_request() {
        let policy = GatewayBridgePolicy::new(BridgeMode::BestEffort);

        let error = read_request_json_with_policy::<Payload>(Body::from("not-json"), &policy)
            .await
            .expect_err("should reject invalid json");

        assert!(matches!(
            error,
            GatewayBodyReadError::InvalidJson {
                role: GatewayBodyRole::DownstreamRequest,
                ..
            }
        ));
        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn masked_body_read_errors_hide_internal_details() {
        let policy = GatewayBridgePolicy::new(BridgeMode::BestEffort)
            .with_upstream_read_limit_bytes(4)
            .with_passthrough_runtime_errors(false);

        let error = read_upstream_body_with_policy(Body::from("12345"), &policy)
            .await
            .expect_err("upstream body should exceed policy limit");

        let response = error.to_response(&policy);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body");

        assert_eq!(body, Bytes::from("upstream body too large"));
    }
}
