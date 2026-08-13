use std::error::Error as _;
use std::fmt;

use axum::body::{Body, Bytes, to_bytes};
use axum::http::StatusCode;
use http_body_util::LengthLimitError;
use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::{GatewayErrorDetail, GatewayPolicy};

use super::response::error_response;

/// Which side of the server boundary owns a buffered HTTP body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GatewayBodyRole {
    Request,
    Upstream,
}

impl fmt::Display for GatewayBodyRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Request => "request",
            Self::Upstream => "upstream",
        })
    }
}

/// Typed and source-preserving body read failure with sanitized diagnostics.
#[derive(Error)]
#[non_exhaustive]
pub enum GatewayBodyReadError {
    #[error("gateway {role} body exceeded its configured byte limit")]
    LimitExceeded {
        role: GatewayBodyRole,
        limit_bytes: usize,
    },
    #[error("gateway failed to read the {role} body")]
    ReadFailed {
        role: GatewayBodyRole,
        #[source]
        source: axum::Error,
    },
    #[error("gateway {role} body was not valid JSON")]
    InvalidJson {
        role: GatewayBodyRole,
        #[source]
        source: serde_json::Error,
    },
}

impl fmt::Debug for GatewayBodyReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded { role, limit_bytes } => formatter
                .debug_struct("LimitExceeded")
                .field("role", role)
                .field("limit_bytes", limit_bytes)
                .finish(),
            Self::ReadFailed { role, .. } => formatter
                .debug_struct("ReadFailed")
                .field("role", role)
                .field("source", &"<redacted>")
                .finish(),
            Self::InvalidJson { role, .. } => formatter
                .debug_struct("InvalidJson")
                .field("role", role)
                .field("source", &"<redacted>")
                .finish(),
        }
    }
}

impl GatewayBodyReadError {
    pub const fn role(&self) -> GatewayBodyRole {
        match self {
            Self::LimitExceeded { role, .. }
            | Self::ReadFailed { role, .. }
            | Self::InvalidJson { role, .. } => *role,
        }
    }

    pub const fn status_code(&self) -> StatusCode {
        match self {
            Self::LimitExceeded {
                role: GatewayBodyRole::Request,
                ..
            } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::ReadFailed {
                role: GatewayBodyRole::Request,
                ..
            }
            | Self::InvalidJson {
                role: GatewayBodyRole::Request,
                ..
            } => StatusCode::BAD_REQUEST,
            Self::LimitExceeded {
                role: GatewayBodyRole::Upstream,
                ..
            }
            | Self::ReadFailed {
                role: GatewayBodyRole::Upstream,
                ..
            }
            | Self::InvalidJson {
                role: GatewayBodyRole::Upstream,
                ..
            } => StatusCode::BAD_GATEWAY,
        }
    }

    pub const fn code(&self) -> &'static str {
        match self {
            Self::LimitExceeded {
                role: GatewayBodyRole::Request,
                ..
            } => "request_body_too_large",
            Self::LimitExceeded {
                role: GatewayBodyRole::Upstream,
                ..
            } => "upstream_body_too_large",
            Self::ReadFailed {
                role: GatewayBodyRole::Request,
                ..
            } => "request_body_read_failed",
            Self::ReadFailed {
                role: GatewayBodyRole::Upstream,
                ..
            } => "upstream_body_read_failed",
            Self::InvalidJson {
                role: GatewayBodyRole::Request,
                ..
            } => "invalid_request_json",
            Self::InvalidJson {
                role: GatewayBodyRole::Upstream,
                ..
            } => "invalid_upstream_json",
        }
    }

    pub fn user_message(&self, policy: &GatewayPolicy) -> String {
        match (self, policy.error_detail()) {
            (Self::LimitExceeded { role, limit_bytes }, GatewayErrorDetail::Public) => {
                format!("{role} body exceeded the configured limit of {limit_bytes} bytes")
            }
            (Self::LimitExceeded { role, .. }, GatewayErrorDetail::Minimal) => {
                format!("{role} body is too large")
            }
            (Self::ReadFailed { role, .. }, _) => format!("failed to read {role} body"),
            (Self::InvalidJson { role, .. }, _) => format!("invalid {role} JSON body"),
        }
    }

    pub fn to_response(&self, policy: &GatewayPolicy) -> axum::response::Response {
        error_response(
            self.status_code(),
            self.code(),
            &self.user_message(policy),
            None,
            policy,
        )
    }
}

/// Read a downstream request body under the finite server limit.
pub async fn read_request_body(
    body: Body,
    policy: &GatewayPolicy,
) -> Result<Bytes, GatewayBodyReadError> {
    read_body(
        body,
        policy.limits().request_body_bytes(),
        GatewayBodyRole::Request,
    )
    .await
}

/// Decode downstream request JSON under the finite server limit.
pub async fn read_request_json<T>(
    body: Body,
    policy: &GatewayPolicy,
) -> Result<T, GatewayBodyReadError>
where
    T: DeserializeOwned,
{
    read_json(
        body,
        policy.limits().request_body_bytes(),
        GatewayBodyRole::Request,
    )
    .await
}

/// Read a buffered upstream body under the finite server limit.
pub async fn read_upstream_body(
    body: Body,
    policy: &GatewayPolicy,
) -> Result<Bytes, GatewayBodyReadError> {
    read_body(
        body,
        policy.limits().upstream_body_bytes(),
        GatewayBodyRole::Upstream,
    )
    .await
}

/// Decode buffered upstream JSON under the finite server limit.
pub async fn read_upstream_json<T>(
    body: Body,
    policy: &GatewayPolicy,
) -> Result<T, GatewayBodyReadError>
where
    T: DeserializeOwned,
{
    read_json(
        body,
        policy.limits().upstream_body_bytes(),
        GatewayBodyRole::Upstream,
    )
    .await
}

async fn read_json<T>(
    body: Body,
    limit_bytes: usize,
    role: GatewayBodyRole,
) -> Result<T, GatewayBodyReadError>
where
    T: DeserializeOwned,
{
    let bytes = read_body(body, limit_bytes, role).await?;
    serde_json::from_slice(&bytes)
        .map_err(|source| GatewayBodyReadError::InvalidJson { role, source })
}

async fn read_body(
    body: Body,
    limit_bytes: usize,
    role: GatewayBodyRole,
) -> Result<Bytes, GatewayBodyReadError> {
    to_bytes(body, limit_bytes).await.map_err(|source| {
        if source
            .source()
            .is_some_and(|source| source.is::<LengthLimitError>())
        {
            GatewayBodyReadError::LimitExceeded { role, limit_bytes }
        } else {
            GatewayBodyReadError::ReadFailed { role, source }
        }
    })
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;
    use crate::GatewayLimits;

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Payload {
        message: String,
    }

    fn four_byte_policy() -> GatewayPolicy {
        GatewayPolicy::default().with_limits(
            GatewayLimits::new(
                4,
                4,
                crate::MIN_SERVER_RESPONSE_LIMIT_BYTES,
                crate::MIN_SERVER_RESPONSE_LIMIT_BYTES,
            )
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn request_and_upstream_limits_have_distinct_statuses() {
        let policy = four_byte_policy();
        let request = read_request_body(Body::from("12345"), &policy)
            .await
            .unwrap_err();
        let upstream = read_upstream_body(Body::from("12345"), &policy)
            .await
            .unwrap_err();

        assert_eq!(request.status_code(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(upstream.status_code(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn request_json_is_decoded_within_limit() {
        let policy = GatewayPolicy::default();
        let payload: Payload = read_request_json(Body::from(r#"{"message":"hello"}"#), &policy)
            .await
            .unwrap();

        assert_eq!(payload.message, "hello");
    }

    #[tokio::test]
    async fn invalid_json_diagnostics_do_not_echo_the_body() {
        let policy = GatewayPolicy::default();
        let error = read_request_json::<Payload>(Body::from("secret-not-json"), &policy)
            .await
            .unwrap_err();

        assert_eq!(error.code(), "invalid_request_json");
        assert!(!format!("{error:?}").contains("secret-not-json"));
    }
}
