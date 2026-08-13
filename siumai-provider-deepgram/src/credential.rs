use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{AUTHORIZATION, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, NoAuth, RequestBuildError,
};
use thiserror::Error;

const MAX_API_KEY_BYTES: usize = 16 * 1024;

/// Authentication selected for one configured Deepgram provider.
#[derive(Clone)]
#[non_exhaustive]
pub enum DeepgramCredential {
    ApiKey(SecretString),
    Unauthenticated,
}

impl DeepgramCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::ApiKey(SecretString::from(value.into()))
    }

    /// Explicit unauthenticated mode, primarily for trusted test endpoints.
    pub fn unauthenticated() -> Self {
        Self::Unauthenticated
    }

    pub fn from_env() -> Result<Self, DeepgramCredentialError> {
        std::env::var("DEEPGRAM_API_KEY")
            .map(Self::api_key)
            .map_err(|_| DeepgramCredentialError::MissingEnvironmentVariable)
    }

    pub(crate) fn validate(&self) -> Result<(), DeepgramCredentialError> {
        match self {
            Self::ApiKey(value) => validate_api_key(value.expose_secret()),
            Self::Unauthenticated => Ok(()),
        }
    }

    pub(crate) fn is_unauthenticated(&self) -> bool {
        matches!(self, Self::Unauthenticated)
    }

    pub(crate) fn into_auth(self) -> Arc<dyn AuthApplier> {
        match self {
            Self::ApiKey(value) => Arc::new(TokenAuth(value)),
            Self::Unauthenticated => Arc::new(NoAuth),
        }
    }
}

impl fmt::Debug for DeepgramCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("DeepgramCredential")
            .field(&match self {
                Self::ApiKey(_) => "ApiKey([REDACTED])",
                Self::Unauthenticated => "Unauthenticated",
            })
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DeepgramCredentialError {
    #[error("DEEPGRAM_API_KEY is not configured")]
    MissingEnvironmentVariable,
    #[error("Deepgram API key is empty or contains invalid characters")]
    InvalidApiKey,
    #[error("Deepgram API key exceeds the {maximum}-byte limit")]
    ApiKeyTooLarge { maximum: usize },
}

struct TokenAuth(SecretString);

#[async_trait]
impl AuthApplier for TokenAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let value = HeaderValue::from_str(&format!("Token {}", self.0.expose_secret())).map_err(
            |source| {
                Error::new(
                    ErrorKind::Authentication,
                    "Deepgram credential could not be encoded as an authorization header",
                )
                .with_source(source)
            },
        )?;
        CredentialPatch::new()
            .try_insert(AUTHORIZATION, value)
            .map_err(request_build_error)
    }
}

fn validate_api_key(value: &str) -> Result<(), DeepgramCredentialError> {
    if value.len() > MAX_API_KEY_BYTES {
        return Err(DeepgramCredentialError::ApiKeyTooLarge {
            maximum: MAX_API_KEY_BYTES,
        });
    }
    if value.is_empty() || value != value.trim() || value.chars().any(char::is_control) {
        return Err(DeepgramCredentialError::InvalidApiKey);
    }
    Ok(())
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "Deepgram credential conflicts with the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_debug_never_exposes_the_secret() {
        let credential = DeepgramCredential::api_key("canary-secret");
        let debug = format!("{credential:?}");
        assert!(!debug.contains("canary-secret"));
    }

    #[test]
    fn invalid_static_credentials_fail_validation() {
        assert_eq!(
            DeepgramCredential::api_key("").validate(),
            Err(DeepgramCredentialError::InvalidApiKey)
        );
        assert_eq!(
            DeepgramCredential::api_key(" key ").validate(),
            Err(DeepgramCredentialError::InvalidApiKey)
        );
    }
}
