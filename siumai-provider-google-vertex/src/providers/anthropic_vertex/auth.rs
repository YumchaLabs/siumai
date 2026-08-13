use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{AUTHORIZATION, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{AuthApplier, AuthContext, AuthRefresh, CredentialPatch, RequestBuildError};
use thiserror::Error as ThisError;

const MAX_ACCESS_TOKEN_BYTES: usize = 16 * 1024;

/// Async source for short-lived Google access tokens.
#[async_trait]
pub trait GoogleVertexTokenSource: Send + Sync {
    async fn token(&self) -> Result<String, Error>;
}

/// Google access-token source for the official Anthropic-on-Vertex endpoint.
#[derive(Clone)]
pub enum GoogleVertexCredential {
    AccessToken(SecretString),
    Dynamic(Arc<dyn GoogleVertexTokenSource>),
}

impl GoogleVertexCredential {
    pub fn access_token(value: impl Into<String>) -> Self {
        Self::AccessToken(SecretString::from(value.into()))
    }

    pub fn dynamic(source: Arc<dyn GoogleVertexTokenSource>) -> Self {
        Self::Dynamic(source)
    }

    pub(crate) fn into_auth(self) -> Result<Arc<dyn AuthApplier>, GoogleVertexCredentialError> {
        if let Self::AccessToken(value) = &self {
            validate_access_token(value.expose_secret())?;
        }
        Ok(Arc::new(GoogleBearerAuth { credential: self }))
    }
}

impl fmt::Debug for GoogleVertexCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("GoogleVertexCredential")
            .field(&match self {
                Self::AccessToken(_) => "AccessToken([REDACTED])",
                Self::Dynamic(_) => "Dynamic([REDACTED])",
            })
            .finish()
    }
}

struct GoogleBearerAuth {
    credential: GoogleVertexCredential,
}

impl fmt::Debug for GoogleBearerAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GoogleBearerAuth")
            .field("credential", &"[REDACTED]")
            .finish()
    }
}

#[async_trait]
impl AuthApplier for GoogleBearerAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let token = match &self.credential {
            GoogleVertexCredential::AccessToken(value) => value.expose_secret().to_string(),
            GoogleVertexCredential::Dynamic(source) => source.token().await.map_err(|_| {
                Error::new(
                    ErrorKind::Authentication,
                    "Google access-token source failed",
                )
            })?,
        };
        validate_access_token(&token).map_err(|_| {
            Error::new(
                ErrorKind::Authentication,
                "Google access token is not a valid bearer credential",
            )
        })?;
        let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|source| {
            Error::new(
                ErrorKind::Authentication,
                "Google access token could not be encoded as a request header",
            )
            .with_source(source)
        })?;
        CredentialPatch::new()
            .try_insert(AUTHORIZATION, value)
            .map_err(request_build_error)
    }
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "Google bearer authentication conflicts with the transport contract",
    )
    .with_source(source)
}

fn validate_access_token(value: &str) -> Result<(), GoogleVertexCredentialError> {
    if value.is_empty()
        || value.len() > MAX_ACCESS_TOKEN_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(GoogleVertexCredentialError);
    }
    Ok(())
}

/// Sanitized static access-token validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[error("Google access token must be non-empty, bounded, and contain no control characters")]
pub struct GoogleVertexCredentialError;
