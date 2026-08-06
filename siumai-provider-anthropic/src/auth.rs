use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, NoAuth, RequestBuildError,
};
use thiserror::Error as ThisError;

const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");

/// Anthropic credential source.
#[derive(Clone)]
pub enum AnthropicCredential {
    ApiKey(SecretString),
    /// Intended for deterministic local endpoints and explicitly unauthenticated gateways.
    Unauthenticated,
}

impl AnthropicCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::ApiKey(SecretString::from(value.into()))
    }

    pub const fn unauthenticated() -> Self {
        Self::Unauthenticated
    }

    pub(crate) const fn is_unauthenticated(&self) -> bool {
        matches!(self, Self::Unauthenticated)
    }

    pub(crate) fn into_auth(self) -> Result<Arc<dyn AuthApplier>, AnthropicCredentialError> {
        match self {
            Self::ApiKey(value) => {
                validate_secret(value.expose_secret())?;
                Ok(Arc::new(AnthropicApiKeyAuth { value }))
            }
            Self::Unauthenticated => Ok(Arc::new(NoAuth)),
        }
    }
}

impl fmt::Debug for AnthropicCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("AnthropicCredential")
            .field(&match self {
                Self::ApiKey(_) => "ApiKey([REDACTED])",
                Self::Unauthenticated => "Unauthenticated",
            })
            .finish()
    }
}

struct AnthropicApiKeyAuth {
    value: SecretString,
}

impl fmt::Debug for AnthropicApiKeyAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicApiKeyAuth")
            .field("value", &"[REDACTED]")
            .finish()
    }
}

#[async_trait]
impl AuthApplier for AnthropicApiKeyAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let value = HeaderValue::from_str(self.value.expose_secret()).map_err(|source| {
            Error::new(
                ErrorKind::Authentication,
                "Anthropic API key could not be encoded as a request header",
            )
            .with_source(source)
        })?;
        CredentialPatch::new()
            .try_insert(X_API_KEY, value)
            .map_err(request_build_error)
    }
}

fn validate_secret(value: &str) -> Result<(), AnthropicCredentialError> {
    if value.is_empty() || value.len() > MAX_CREDENTIAL_BYTES || value.chars().any(char::is_control)
    {
        Err(AnthropicCredentialError)
    } else {
        Ok(())
    }
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "Anthropic credential header conflicts with the transport contract",
    )
    .with_source(source)
}

/// Sanitized Anthropic static-credential validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[error("credential must be non-empty, bounded, and contain no control characters")]
pub struct AnthropicCredentialError;
