use async_trait::async_trait;
use http::HeaderValue;
use http::header::AUTHORIZATION;
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{AuthApplier, AuthContext, AuthRefresh, CredentialPatch, RequestBuildError};

pub(crate) const MAX_API_KEY_BYTES: usize = 16 * 1024;

pub(crate) fn validate_api_key(api_key: &SecretString) -> bool {
    let value = api_key.expose_secret();
    !value.is_empty()
        && value.len() <= MAX_API_KEY_BYTES
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

pub(crate) struct CohereBearerAuth {
    api_key: SecretString,
}

impl CohereBearerAuth {
    pub(crate) fn new(api_key: SecretString) -> Self {
        Self { api_key }
    }
}

#[async_trait]
impl AuthApplier for CohereBearerAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let value = HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
            .map_err(|source| {
                Error::new(
                    ErrorKind::Authentication,
                    "Cohere API key could not be encoded as an authorization header",
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
        "Cohere credential conflicts with the transport contract",
    )
    .with_source(source)
}
