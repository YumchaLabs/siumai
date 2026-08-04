//! One-time approval contracts for trusted local tool execution.
//!
//! Authenticity, context binding, and replay prevention are deliberately
//! separate responsibilities:
//!
//! - an [`ApprovalVerifier`] authenticates an opaque envelope and returns its
//!   claims;
//! - [`verify_and_consume`] compares every execution-bound field against a
//!   host-created [`TrustContext`];
//! - an [`ApprovalConsumeStore`] atomically consumes the approval nonce before
//!   local execution starts.
//!
//! This module does not provide production cryptography. Applications should
//! implement the signer and verifier traits with an audited authenticated
//! signing format or authenticated encryption scheme.

mod claims;
mod consume;
mod context;
mod crypto;
mod verify;

pub use claims::{
    APPROVAL_CLAIMS_VERSION, ApprovalClaims, ApprovalClaimsError, ApprovalEnvelope,
    ApprovalEnvelopeError, MAX_APPROVAL_ENVELOPE_BYTES,
};
pub use consume::{
    ApprovalConsumeError, ApprovalConsumeKey, ApprovalConsumeStore, InMemoryApprovalConsumeStore,
};
pub use context::{TrustContext, TrustContextBuildError, TrustContextBuilder};
pub use crypto::{ApprovalSigner, ApprovalSigningError, ApprovalVerifier, ApprovalVerifierError};
pub use verify::{
    ApprovalClaimField, ApprovalVerificationError, VerifiedApproval, verify_and_consume,
    verify_and_consume_at_unix_ms,
};

#[cfg(test)]
mod tests;
