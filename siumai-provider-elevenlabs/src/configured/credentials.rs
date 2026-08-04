use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::HeaderValue;
use http::header::HeaderName;
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{AuthApplier, AuthContext, AuthRefresh, CredentialPatch, RequestBuildError};
use thiserror::Error;

const MAX_API_KEY_BYTES: usize = 16 * 1024;
const XI_API_KEY: HeaderName = HeaderName::from_static("xi-api-key");

/// A redacted, statically validated ElevenLabs API key.
#[derive(Clone)]
pub struct ElevenLabsApiKey {
    secret: SecretString,
}

impl ElevenLabsApiKey {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            secret: SecretString::from(value.into()),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ElevenLabsCredentialError> {
        let value = self.secret.expose_secret();
        if value.trim().is_empty() {
            return Err(ElevenLabsCredentialError::Empty);
        }
        if value.len() > MAX_API_KEY_BYTES {
            return Err(ElevenLabsCredentialError::TooLarge {
                maximum: MAX_API_KEY_BYTES,
            });
        }
        HeaderValue::from_str(value).map_err(|_| ElevenLabsCredentialError::InvalidHeaderValue)?;
        Ok(())
    }
}

impl fmt::Debug for ElevenLabsApiKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElevenLabsApiKey")
            .field("secret", &"[REDACTED]")
            .finish()
    }
}

/// Explicit authentication material captured by one configured provider.
#[derive(Clone)]
pub enum ElevenLabsCredential {
    ApiKey(ElevenLabsApiKey),
}

impl ElevenLabsCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::ApiKey(ElevenLabsApiKey::new(value))
    }

    pub(crate) fn validate(&self) -> Result<(), ElevenLabsCredentialError> {
        match self {
            Self::ApiKey(api_key) => api_key.validate(),
        }
    }

    pub(crate) fn into_auth(self) -> Arc<dyn AuthApplier> {
        match self {
            Self::ApiKey(api_key) => Arc::new(ElevenLabsAuth { api_key }),
        }
    }
}

impl fmt::Debug for ElevenLabsCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ElevenLabsCredential")
            .field(&"ApiKey([REDACTED])")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ElevenLabsCredentialError {
    #[error("ElevenLabs API key must not be empty")]
    Empty,
    #[error("ElevenLabs API key exceeds {maximum} bytes")]
    TooLarge { maximum: usize },
    #[error("ElevenLabs API key cannot be encoded as an HTTP header")]
    InvalidHeaderValue,
}

struct ElevenLabsAuth {
    api_key: ElevenLabsApiKey,
}

#[async_trait]
impl AuthApplier for ElevenLabsAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let value =
            HeaderValue::from_str(self.api_key.secret.expose_secret()).map_err(|source| {
                Error::new(
                    ErrorKind::Authentication,
                    "ElevenLabs API key could not be encoded as a request header",
                )
                .with_source(source)
            })?;
        CredentialPatch::new()
            .try_insert(XI_API_KEY, value)
            .map_err(request_build_error)
    }
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Authentication,
        "ElevenLabs credential violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_validate_synchronously_and_debug_is_redacted() {
        assert_eq!(
            ElevenLabsCredential::api_key(" ").validate(),
            Err(ElevenLabsCredentialError::Empty)
        );

        let credential = ElevenLabsCredential::api_key("canary-elevenlabs-secret");
        assert!(credential.validate().is_ok());
        assert!(!format!("{credential:?}").contains("canary-elevenlabs-secret"));
    }
}
