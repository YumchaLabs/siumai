use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{AUTHORIZATION, HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, NoAuth, RequestBuildError,
};
use thiserror::Error as ThisError;

const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");

/// Explicit authentication strategy for one compatible endpoint.
#[derive(Clone)]
pub enum AnthropicCompatibleCredential {
    ApiKey(SecretString),
    Bearer(SecretString),
    Unauthenticated,
}

impl AnthropicCompatibleCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::ApiKey(SecretString::from(value.into()))
    }

    pub fn bearer(value: impl Into<String>) -> Self {
        Self::Bearer(SecretString::from(value.into()))
    }

    pub const fn unauthenticated() -> Self {
        Self::Unauthenticated
    }

    pub(crate) fn validate(&self) -> Result<(), CredentialError> {
        match self {
            Self::ApiKey(value) | Self::Bearer(value) => validate_secret(value.expose_secret()),
            Self::Unauthenticated => Ok(()),
        }
    }

    pub(crate) fn into_auth(self) -> Arc<dyn AuthApplier> {
        match self {
            Self::ApiKey(value) => Arc::new(StaticHeaderAuth {
                scheme: StaticAuthScheme::ApiKey,
                value,
            }),
            Self::Bearer(value) => Arc::new(StaticHeaderAuth {
                scheme: StaticAuthScheme::Bearer,
                value,
            }),
            Self::Unauthenticated => Arc::new(NoAuth),
        }
    }
}

impl fmt::Debug for AnthropicCompatibleCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("AnthropicCompatibleCredential")
            .field(&match self {
                Self::ApiKey(_) => "ApiKey([REDACTED])",
                Self::Bearer(_) => "Bearer([REDACTED])",
                Self::Unauthenticated => "Unauthenticated",
            })
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
enum StaticAuthScheme {
    ApiKey,
    Bearer,
}

struct StaticHeaderAuth {
    scheme: StaticAuthScheme,
    value: SecretString,
}

impl fmt::Debug for StaticHeaderAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaticHeaderAuth")
            .field("scheme", &self.scheme)
            .field("value", &"[REDACTED]")
            .finish()
    }
}

#[async_trait]
impl AuthApplier for StaticHeaderAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let value = match self.scheme {
            StaticAuthScheme::ApiKey => HeaderValue::from_str(self.value.expose_secret()),
            StaticAuthScheme::Bearer => {
                HeaderValue::from_str(&format!("Bearer {}", self.value.expose_secret()))
            }
        }
        .map_err(|source| {
            Error::new(
                ErrorKind::Authentication,
                "credential could not be encoded as a request header",
            )
            .with_source(source)
        })?;
        let name = match self.scheme {
            StaticAuthScheme::ApiKey => X_API_KEY,
            StaticAuthScheme::Bearer => AUTHORIZATION,
        };
        CredentialPatch::new()
            .try_insert(name, value)
            .map_err(request_build_error)
    }
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "credential header conflicts with the transport contract",
    )
    .with_source(source)
}

fn validate_secret(value: &str) -> Result<(), CredentialError> {
    if value.is_empty() || value.len() > MAX_CREDENTIAL_BYTES || value.chars().any(char::is_control)
    {
        Err(CredentialError)
    } else {
        Ok(())
    }
}

/// Sanitized static-credential validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[error("credential must be non-empty, bounded, and contain no control characters")]
pub struct CredentialError;
