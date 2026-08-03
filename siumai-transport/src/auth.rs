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

/// Whether an authentication source should reuse or refresh its material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthRefresh {
    Current,
    AfterUnauthorized,
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
#[derive(Clone, Default)]
pub struct CredentialPatch {
    headers: HeaderMap,
    query: Vec<(String, String)>,
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
        if is_transport_controlled(&name)
            || matches!(
                name.as_str(),
                "cookie" | "proxy-authorization" | "set-cookie"
            )
        {
            return Err(RequestBuildError::ProtectedHeader);
        }
        value.set_sensitive(true);
        self.headers.insert(name, value);
        Ok(self)
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
        self.query.push((name, value));
        Ok(self)
    }

    pub(crate) fn into_parts(self) -> (HeaderMap, Vec<(String, String)>) {
        (self.headers, self.query)
    }
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
}
