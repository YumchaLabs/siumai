use thiserror::Error;

use super::{ApprovalClaims, ApprovalEnvelope};

/// Authenticates versioned approval claims into an opaque envelope.
///
/// Implementations should use an audited signing or authenticated-encryption
/// construction. The claims `key_id` must identify the key actually used.
pub trait ApprovalSigner: Send + Sync {
    fn sign(&self, claims: &ApprovalClaims) -> Result<ApprovalEnvelope, ApprovalSigningError>;
}

/// Authenticates an opaque envelope and returns its exact signed claims.
///
/// Implementations must reject malformed data, unsupported keys, algorithms,
/// key-ID mismatches, and failed authenticity checks. Claim context and expiry
/// validation remains the responsibility of [`verify_and_consume`](super::verify_and_consume).
pub trait ApprovalVerifier: Send + Sync {
    fn verify(&self, envelope: &ApprovalEnvelope) -> Result<ApprovalClaims, ApprovalVerifierError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ApprovalSigningError {
    #[error("approval signer rejected the claims")]
    Rejected,
    #[error("approval signer is unavailable")]
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ApprovalVerifierError {
    #[error("approval authenticity verification failed")]
    Rejected,
    #[error("approval verifier is unavailable")]
    Unavailable,
}
