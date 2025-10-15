//! Authentication helpers and token providers.
//! This module defines a minimal trait to supply Bearer tokens (e.g., for Vertex AI).

use crate::error::LlmError;

/// A synchronous Bearer token provider.
///
/// Notes:
/// - The interface is intentionally synchronous to integrate with existing
///   header-builder call sites which are not async.
/// - Implementations may perform caching internally and refresh tokens
///   when necessary.
pub trait TokenProvider: Send + Sync {
    /// Returns an access token string suitable for the `Authorization: Bearer <token>` header.
    fn token(&self) -> Result<String, LlmError>;
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

impl TokenProvider for StaticTokenProvider {
    fn token(&self) -> Result<String, LlmError> {
        Ok(self.token.clone())
    }
}

pub mod adc;
pub mod service_account;
