//! Proof-oriented retry and replay policy.

use std::fmt;
use std::time::Duration;

use http::header::HeaderName;

use crate::RequestBuildError;

/// A provider-declared header that gives one logical operation idempotency.
#[derive(Clone, PartialEq, Eq)]
pub struct IdempotencyHeader(HeaderName);

impl IdempotencyHeader {
    pub fn new(name: HeaderName) -> Result<Self, RequestBuildError> {
        if is_transport_controlled(&name) || is_common_credential_header(&name) {
            return Err(RequestBuildError::InvalidIdempotencyHeader);
        }
        Ok(Self(name))
    }

    pub fn name(&self) -> &HeaderName {
        &self.0
    }
}

impl fmt::Debug for IdempotencyHeader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("IdempotencyHeader")
            .field(&self.0.as_str())
            .finish()
    }
}

/// Closed replay proof supplied by provider-owned operation code.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ReplaySafety {
    /// The request must not be sent again after an attempt starts.
    #[default]
    Never,
    /// Repeating the operation has the same remote semantics.
    SemanticallyIdempotent,
    /// The provider documents idempotency for a stable per-call key.
    IdempotencyKey(IdempotencyHeader),
}

impl ReplaySafety {
    pub(crate) fn permits_replay(&self) -> bool {
        !matches!(self, Self::Never)
    }
}

/// One retry budget shared by network, authentication, and response retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    max_attempts: u8,
    initial_backoff: Duration,
    max_backoff: Duration,
    max_server_delay: Duration,
    jitter: bool,
}

impl RetryPolicy {
    pub fn new(max_attempts: u8) -> Result<Self, TransportRetryPolicyError> {
        if max_attempts == 0 {
            return Err(TransportRetryPolicyError::ZeroAttempts);
        }
        Ok(Self {
            max_attempts,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(8),
            max_server_delay: Duration::from_secs(60),
            jitter: true,
        })
    }

    pub fn max_attempts(self) -> u8 {
        self.max_attempts
    }

    pub fn initial_backoff(self) -> Duration {
        self.initial_backoff
    }

    pub fn max_backoff(self) -> Duration {
        self.max_backoff
    }

    /// Largest standard `Retry-After` delay this policy will honor.
    pub fn max_server_delay(self) -> Duration {
        self.max_server_delay
    }

    pub fn uses_jitter(self) -> bool {
        self.jitter
    }

    pub fn with_backoff(mut self, initial: Duration, maximum: Duration) -> Self {
        self.initial_backoff = initial.min(maximum);
        self.max_backoff = maximum;
        self
    }

    /// Set the independent ceiling for standard server retry advice.
    pub fn with_max_server_delay(mut self, maximum: Duration) -> Self {
        self.max_server_delay = maximum;
        self
    }

    pub fn with_jitter(mut self, enabled: bool) -> Self {
        self.jitter = enabled;
        self
    }

    pub(crate) fn backoff_for(self, completed_attempts: u8) -> Duration {
        let exponent = u32::from(completed_attempts.saturating_sub(1)).min(31);
        let factor = 1_u32.checked_shl(exponent).unwrap_or(u32::MAX);
        self.initial_backoff
            .saturating_mul(factor)
            .min(self.max_backoff)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3).expect("the default retry attempt count is non-zero")
    }
}

/// Invalid retry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransportRetryPolicyError {
    #[error("retry policy must allow at least one total attempt")]
    ZeroAttempts,
}

pub(crate) fn is_transport_controlled(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "content-length"
            | "host"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "sec-websocket-accept"
            | "sec-websocket-extensions"
            | "sec-websocket-key"
            | "sec-websocket-version"
    )
}

/// Whether a header is part of the small provider-independent credential set.
///
/// `HeaderName` already provides ASCII case-insensitive HTTP name semantics.
/// Provider-specific credential names stay out of this list: the selected
/// credential applier declares them exactly through its [`crate::CredentialPatch`].
pub(crate) fn is_common_credential_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "api-key"
            | "authorization"
            | "cookie"
            | "proxy-authenticate"
            | "set-cookie"
            | "www-authenticate"
            | "x-api-key"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authentication_header_cannot_be_an_idempotency_header() {
        let error = IdempotencyHeader::new(http::header::AUTHORIZATION).unwrap_err();
        assert_eq!(error, RequestBuildError::InvalidIdempotencyHeader);
    }

    #[test]
    fn credential_header_matching_is_exact_and_case_insensitive() {
        for name in ["Authorization", "API-Key", "X-API-Key"] {
            let name = HeaderName::from_bytes(name.as_bytes()).unwrap();
            assert!(
                is_common_credential_header(&name),
                "{name} must be protected"
            );
        }

        for name in [
            "x-token-count-mode",
            "x-secret-sampling-mode",
            "x-api-key-count",
            "api_key",
        ] {
            let name = HeaderName::from_bytes(name.as_bytes()).unwrap();
            assert!(
                !is_common_credential_header(&name),
                "{name} must not be classified by a fuzzy credential heuristic"
            );
        }
    }

    #[test]
    fn backoff_is_capped() {
        let policy = RetryPolicy::new(10)
            .unwrap()
            .with_backoff(Duration::from_millis(100), Duration::from_millis(250));
        assert_eq!(policy.backoff_for(1), Duration::from_millis(100));
        assert_eq!(policy.backoff_for(4), Duration::from_millis(250));
    }

    #[test]
    fn server_retry_advice_has_an_independent_ceiling() {
        let policy = RetryPolicy::new(3)
            .unwrap()
            .with_backoff(Duration::from_millis(10), Duration::from_millis(20))
            .with_max_server_delay(Duration::from_secs(2));

        assert_eq!(policy.max_backoff(), Duration::from_millis(20));
        assert_eq!(policy.max_server_delay(), Duration::from_secs(2));
    }
}
