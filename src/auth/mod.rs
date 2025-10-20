//! Authentication helpers and token providers.
//! This module defines a minimal trait to supply Bearer tokens (e.g., for Vertex AI).

use crate::error::LlmError;
use async_trait::async_trait;

/// An async Bearer token provider.
///
/// Notes:
/// - The interface is async to allow for network calls without blocking.
/// - Implementations should perform caching internally and refresh tokens
///   when necessary.
#[async_trait]
pub trait TokenProvider: Send + Sync {
    /// Returns an access token string suitable for the `Authorization: Bearer <token>` header.
    async fn token(&self) -> Result<String, LlmError>;
}

/// A simple static token provider useful for tests and basic scenarios where
/// the token is managed externally.
pub struct StaticTokenProvider {
    token: String,
}

impl StaticTokenProvider {
    /// Create a new static token provider.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

#[async_trait]
impl TokenProvider for StaticTokenProvider {
    async fn token(&self) -> Result<String, LlmError> {
        Ok(self.token.clone())
    }
}

#[cfg(feature = "gcp")]
pub mod adc;
#[cfg(feature = "gcp")]
pub mod service_account;
