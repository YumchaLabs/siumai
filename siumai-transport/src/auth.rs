//! Provider-owned authentication applied inside an exact endpoint audience.

use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use http::Method;
use http::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Url;
use siumai_core::Error;

use crate::RequestBuildError;
use crate::endpoint::CredentialAudience;
use crate::replay::is_transport_controlled;

const MAX_CREDENTIAL_QUERY_PAIRS: usize = 32;
const MAX_CREDENTIAL_QUERY_BYTES: usize = 32 * 1024;
const MAX_AUTHENTICATED_URL_BYTES: usize = 64 * 1024;

/// Opaque generation identifying credential material within one auth source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CredentialRevision(u64);

impl CredentialRevision {
    pub fn new(generation: u64) -> Self {
        Self(generation)
    }
}

/// Whether an authentication source should reuse or refresh its material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthRefresh {
    Current,
    AfterUnauthorized {
        rejected_revision: CredentialRevision,
    },
}

/// Immutable request information needed by API-key, bearer, or signing auth.
pub struct AuthContext<'a> {
    audience: &'a CredentialAudience,
    method: &'a Method,
    url: &'a Url,
    headers: &'a HeaderMap,
    body: &'a Bytes,
}

impl<'a> AuthContext<'a> {
    pub(crate) fn new(
        audience: &'a CredentialAudience,
        method: &'a Method,
        url: &'a Url,
        headers: &'a HeaderMap,
        body: &'a Bytes,
    ) -> Self {
        Self {
            audience,
            method,
            url,
            headers,
            body,
        }
    }

    pub fn audience(&self) -> &CredentialAudience {
        self.audience
    }

    pub fn method(&self) -> &Method {
        self.method
    }

    /// Explicit access for request signing. The URL may contain sensitive query data.
    pub fn expose_url(&self) -> &Url {
        self.url
    }

    pub fn headers(&self) -> &HeaderMap {
        self.headers
    }

    pub fn body(&self) -> &Bytes {
        self.body
    }
}

impl fmt::Debug for AuthContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthContext")
            .field("audience", self.audience)
            .field("method", self.method)
            .field("url", &"[REDACTED]")
            .field("header_count", &self.headers.len())
            .field("body_bytes", &self.body.len())
            .finish()
    }
}

/// Credential headers whose values stay redacted on default surfaces.
///
/// Header names returned by the selected applier become its exact protected
/// names for that attempt. The transport rejects a collision with protocol
/// request headers before any network submission.
#[derive(Clone, Default)]
pub struct CredentialPatch {
    headers: HeaderMap,
    query: Vec<(String, String)>,
    revision: Option<CredentialRevision>,
}

impl CredentialPatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_insert(
        mut self,
        name: HeaderName,
        mut value: HeaderValue,
    ) -> Result<Self, RequestBuildError> {
        if is_transport_controlled(&name) || matches!(name.as_str(), "cookie" | "set-cookie") {
            return Err(RequestBuildError::ProtectedHeader);
        }
        value.set_sensitive(true);
        self.headers.insert(name, value);
        Ok(self)
    }

    /// Attach a non-secret generation so a 401 refresh can reject exactly the
    /// credential material used for that attempt.
    pub fn with_revision(mut self, revision: CredentialRevision) -> Self {
        self.revision = Some(revision);
        self
    }

    /// Add a query credential when a provider cannot authenticate by header.
    /// Values are intentionally inaccessible and redacted from `Debug`.
    pub fn try_insert_query(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, RequestBuildError> {
        let name = name.into();
        let value = value.into();
        if name.is_empty()
            || name.len() > 256
            || value.len() > 8 * 1024
            || name.chars().any(|character| {
                character.is_control() || matches!(character, '&' | '=' | '#' | '?')
            })
            || value.chars().any(char::is_control)
        {
            return Err(RequestBuildError::InvalidCredentialQuery);
        }
        if self.query.len() >= MAX_CREDENTIAL_QUERY_PAIRS {
            return Err(RequestBuildError::TooManyCredentialQueryParameters {
                maximum: MAX_CREDENTIAL_QUERY_PAIRS,
            });
        }
        self.query.push((name, value));
        let encoded = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(
                self.query
                    .iter()
                    .map(|(name, value)| (name.as_str(), value.as_str())),
            )
            .finish();
        if encoded.len() > MAX_CREDENTIAL_QUERY_BYTES {
            return Err(RequestBuildError::CredentialQueryTooLarge {
                maximum: MAX_CREDENTIAL_QUERY_BYTES,
            });
        }
        Ok(self)
    }

    pub(crate) fn into_parts(
        self,
    ) -> (HeaderMap, Vec<(String, String)>, Option<CredentialRevision>) {
        (self.headers, self.query, self.revision)
    }
}

pub(crate) fn append_credential_query(
    url: &mut Url,
    query: Vec<(String, String)>,
) -> Result<(), RequestBuildError> {
    if !query.is_empty() {
        let mut pairs = url.query_pairs_mut();
        for (name, value) in query {
            pairs.append_pair(&name, &value);
        }
    }
    if url.as_str().len() > MAX_AUTHENTICATED_URL_BYTES {
        return Err(RequestBuildError::AuthenticatedUrlTooLong {
            maximum: MAX_AUTHENTICATED_URL_BYTES,
        });
    }
    Ok(())
}

impl fmt::Debug for CredentialPatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CredentialPatch")
            .field(
                "header_names",
                &self
                    .headers
                    .keys()
                    .map(HeaderName::as_str)
                    .collect::<Vec<_>>(),
            )
            .field("values", &"[REDACTED]")
            .field(
                "query_names",
                &self
                    .query
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("query_values", &"[REDACTED]")
            .field("revision", &self.revision)
            .finish()
    }
}

/// Provider-owned credential source or request signer.
#[async_trait]
pub trait AuthApplier: Send + Sync {
    /// Whether one 401 response may request refreshed credentials. The retry
    /// still requires replay proof and consumes the shared attempt budget.
    fn supports_refresh(&self) -> bool {
        false
    }

    async fn apply(
        &self,
        context: AuthContext<'_>,
        refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error>;
}

/// Authentication strategy for public or otherwise unauthenticated endpoints.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoAuth;

#[async_trait]
impl AuthApplier for NoAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        Ok(CredentialPatch::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_debug_never_prints_values() {
        let patch = CredentialPatch::new()
            .try_insert(
                http::header::AUTHORIZATION,
                HeaderValue::from_static("Bearer canary-secret"),
            )
            .unwrap();
        let debug = format!("{patch:?}");
        assert!(debug.contains("authorization"));
        assert!(!debug.contains("canary-secret"));
    }

    #[test]
    fn credentials_cannot_override_transport_headers() {
        assert_eq!(
            CredentialPatch::new()
                .try_insert(
                    http::header::HOST,
                    HeaderValue::from_static("attacker.invalid")
                )
                .unwrap_err(),
            RequestBuildError::ProtectedHeader
        );
    }

    #[test]
    fn credential_query_count_encoded_size_and_final_url_are_bounded() {
        let mut patch = CredentialPatch::new();
        for index in 0..MAX_CREDENTIAL_QUERY_PAIRS {
            patch = patch
                .try_insert_query(format!("key-{index}"), "value")
                .unwrap();
        }
        assert!(matches!(
            patch.try_insert_query("overflow", "value"),
            Err(RequestBuildError::TooManyCredentialQueryParameters {
                maximum: MAX_CREDENTIAL_QUERY_PAIRS
            })
        ));

        let oversized = CredentialPatch::new()
            .try_insert_query("a", "%".repeat(8 * 1024))
            .unwrap()
            .try_insert_query("b", "%".repeat(8 * 1024));
        assert!(matches!(
            oversized,
            Err(RequestBuildError::CredentialQueryTooLarge {
                maximum: MAX_CREDENTIAL_QUERY_BYTES
            })
        ));

        let mut url = Url::parse(&format!(
            "https://example.com/path?existing={}",
            "a".repeat(MAX_AUTHENTICATED_URL_BYTES)
        ))
        .unwrap();
        assert!(matches!(
            append_credential_query(&mut url, Vec::new()),
            Err(RequestBuildError::AuthenticatedUrlTooLong {
                maximum: MAX_AUTHENTICATED_URL_BYTES
            })
        ));
    }
}
