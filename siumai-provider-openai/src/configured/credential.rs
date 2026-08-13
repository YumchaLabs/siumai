use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{AUTHORIZATION, HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{AuthApplier, AuthContext, AuthRefresh, CredentialPatch, RequestBuildError};
use thiserror::Error;

const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const ORGANIZATION_HEADER: HeaderName = HeaderName::from_static("openai-organization");
const PROJECT_HEADER: HeaderName = HeaderName::from_static("openai-project");

/// Authentication for one configured OpenAI provider runtime.
#[derive(Clone)]
pub enum OpenAiCredential {
    ApiKey(SecretString),
    /// Explicitly unauthenticated mode for a trusted test double or gateway.
    Unauthenticated,
}

impl OpenAiCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::ApiKey(SecretString::from(value.into()))
    }

    pub fn unauthenticated() -> Self {
        Self::Unauthenticated
    }

    pub const fn is_unauthenticated(&self) -> bool {
        matches!(self, Self::Unauthenticated)
    }

    pub(crate) fn validate(&self) -> Result<(), OpenAiCredentialError> {
        match self {
            Self::ApiKey(value) => validate_secret(value.expose_secret()),
            Self::Unauthenticated => Ok(()),
        }
    }

    pub(crate) fn into_auth(
        self,
        organization: Option<String>,
        project: Option<String>,
    ) -> Result<Arc<dyn AuthApplier>, OpenAiCredentialError> {
        let organization =
            validate_optional_header(organization, OpenAiCredentialError::InvalidOrganization)?;
        let project = validate_optional_header(project, OpenAiCredentialError::InvalidProject)?;
        Ok(match self {
            Self::ApiKey(api_key) => Arc::new(OpenAiAuth {
                api_key: Some(api_key),
                organization,
                project,
            }),
            Self::Unauthenticated => Arc::new(OpenAiAuth {
                api_key: None,
                organization,
                project,
            }),
        })
    }
}

impl fmt::Debug for OpenAiCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("OpenAiCredential")
            .field(&match self {
                Self::ApiKey(_) => "ApiKey([REDACTED])",
                Self::Unauthenticated => "Unauthenticated",
            })
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum OpenAiCredentialError {
    #[error("OpenAI API key is empty, malformed, or too large")]
    InvalidApiKey,
    #[error("OpenAI organization header is empty, malformed, or too large")]
    InvalidOrganization,
    #[error("OpenAI project header is empty, malformed, or too large")]
    InvalidProject,
}

struct OpenAiAuth {
    api_key: Option<SecretString>,
    organization: Option<HeaderValue>,
    project: Option<HeaderValue>,
}

impl fmt::Debug for OpenAiAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiAuth")
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field(
                "organization",
                &self.organization.as_ref().map(|_| "[REDACTED]"),
            )
            .field("project", &self.project.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

#[async_trait]
impl AuthApplier for OpenAiAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let mut patch = CredentialPatch::new();
        if let Some(api_key) = &self.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(authentication_error)?;
            patch = patch
                .try_insert(AUTHORIZATION, value)
                .map_err(request_build_error)?;
        }
        if let Some(organization) = &self.organization {
            patch = patch
                .try_insert(ORGANIZATION_HEADER, organization.clone())
                .map_err(request_build_error)?;
        }
        if let Some(project) = &self.project {
            patch = patch
                .try_insert(PROJECT_HEADER, project.clone())
                .map_err(request_build_error)?;
        }
        Ok(patch)
    }
}

fn validate_secret(value: &str) -> Result<(), OpenAiCredentialError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_CREDENTIAL_BYTES
        || HeaderValue::from_str(value).is_err()
    {
        return Err(OpenAiCredentialError::InvalidApiKey);
    }
    Ok(())
}

fn validate_optional_header(
    value: Option<String>,
    error: OpenAiCredentialError,
) -> Result<Option<HeaderValue>, OpenAiCredentialError> {
    value
        .map(|value| {
            if value.trim().is_empty()
                || value != value.trim()
                || value.len() > MAX_CREDENTIAL_BYTES
            {
                return Err(error);
            }
            HeaderValue::from_str(&value).map_err(|_| error)
        })
        .transpose()
}

fn authentication_error(source: http::header::InvalidHeaderValue) -> Error {
    Error::new(
        ErrorKind::Authentication,
        "OpenAI credential could not be encoded as an authorization header",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "OpenAI credential headers conflict with the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_debug_is_redacted() {
        let credential = OpenAiCredential::api_key("canary-secret");
        let debug = format!("{credential:?}");

        assert!(!debug.contains("canary-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn credential_validation_rejects_whitespace() {
        assert!(OpenAiCredential::api_key(" secret").validate().is_err());
        assert!(OpenAiCredential::api_key("secret ").validate().is_err());
    }
}
