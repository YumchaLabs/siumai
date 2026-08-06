use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{AUTHORIZATION, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, NoAuth, RequestBuildError,
};
use thiserror::Error as ThisError;

const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;

/// Static MiniMax bearer credential shared by all provider protocol engines.
#[derive(Clone)]
pub struct MinimaxCredential {
    secret: Option<SecretString>,
}

impl MinimaxCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self {
            secret: Some(SecretString::from(value.into())),
        }
    }

    pub const fn unauthenticated() -> Self {
        Self { secret: None }
    }

    pub(crate) const fn is_unauthenticated(&self) -> bool {
        self.secret.is_none()
    }

    pub(crate) fn into_auth(self) -> Result<Arc<dyn AuthApplier>, MinimaxCredentialError> {
        match self.secret {
            Some(secret) => {
                validate_secret(secret.expose_secret())?;
                Ok(Arc::new(MinimaxBearerAuth(secret)))
            }
            None => Ok(Arc::new(NoAuth)),
        }
    }
}

impl fmt::Debug for MinimaxCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MinimaxCredential")
            .field(&if self.secret.is_some() {
                "Bearer([REDACTED])"
            } else {
                "Unauthenticated"
            })
            .finish()
    }
}

struct MinimaxBearerAuth(SecretString);

impl fmt::Debug for MinimaxBearerAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MinimaxBearerAuth")
            .field(&"[REDACTED]")
            .finish()
    }
}

#[async_trait]
impl AuthApplier for MinimaxBearerAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let value = HeaderValue::from_str(&format!("Bearer {}", self.0.expose_secret())).map_err(
            |source| {
                Error::new(
                    ErrorKind::Authentication,
                    "MiniMax credential could not be encoded as an authorization header",
                )
                .with_source(source)
            },
        )?;
        CredentialPatch::new()
            .try_insert(AUTHORIZATION, value)
            .map_err(request_build_error)
    }
}

fn validate_secret(value: &str) -> Result<(), MinimaxCredentialError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_CREDENTIAL_BYTES
        || HeaderValue::from_str(value).is_err()
    {
        return Err(MinimaxCredentialError);
    }
    Ok(())
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "MiniMax credential conflicts with the transport contract",
    )
    .with_source(source)
}

/// Sanitized MiniMax static-credential validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[error("credential must be non-empty, bounded, and contain no invalid header characters")]
pub struct MinimaxCredentialError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_debug_never_exposes_secret() {
        let credential = MinimaxCredential::api_key("secret-canary");
        let debug = format!("{credential:?}");
        assert!(debug.contains("REDACTED"));
        assert!(!debug.contains("secret-canary"));
    }

    #[test]
    fn invalid_static_credentials_are_rejected_before_transport_construction() {
        for value in ["", " token", "token ", "line\nbreak"] {
            assert!(MinimaxCredential::api_key(value).into_auth().is_err());
        }
    }
}
