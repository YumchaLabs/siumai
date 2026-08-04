use std::collections::BTreeSet;
use std::fmt;
use std::sync::Mutex;

use serde::Serialize;
use thiserror::Error;

use super::ApprovalClaims;

/// Stable replay key consumed for one authenticated approval.
///
/// Nonce uniqueness is issuer/key scoped. Deliberately omitting tenant and run
/// fields makes accidental nonce reuse fail closed across those boundaries.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ApprovalConsumeKey {
    version: u16,
    issuer: String,
    key_id: String,
    nonce: String,
}

impl ApprovalConsumeKey {
    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn nonce(&self) -> &str {
        &self.nonce
    }

    pub(super) fn from_claims(claims: &ApprovalClaims) -> Self {
        Self {
            version: claims.version(),
            issuer: claims.issuer().to_owned(),
            key_id: claims.key_id().to_owned(),
            nonce: claims.nonce().to_owned(),
        }
    }
}

impl fmt::Debug for ApprovalConsumeKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ApprovalConsumeKey")
            .field("version", &self.version)
            .field("contents", &"<redacted>")
            .finish()
    }
}

/// Atomically records that an approval has been consumed.
///
/// Implementations must perform one atomic insert-if-absent operation. A
/// check-then-insert sequence without a transaction, unique constraint, or
/// compare-and-swap does not satisfy this contract.
pub trait ApprovalConsumeStore: Send + Sync {
    fn consume_once(&self, key: &ApprovalConsumeKey) -> Result<(), ApprovalConsumeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ApprovalConsumeError {
    #[error("approval was already consumed")]
    AlreadyConsumed,
    #[error("approval consume store is unavailable")]
    Unavailable,
}

/// Deterministic process-local consume store for single-process applications.
///
/// Distributed or restart-safe execution requires a durable implementation
/// backed by an atomic database operation.
#[derive(Default)]
pub struct InMemoryApprovalConsumeStore {
    consumed: Mutex<BTreeSet<ApprovalConsumeKey>>,
}

impl ApprovalConsumeStore for InMemoryApprovalConsumeStore {
    fn consume_once(&self, key: &ApprovalConsumeKey) -> Result<(), ApprovalConsumeError> {
        let mut consumed = self
            .consumed
            .lock()
            .map_err(|_| ApprovalConsumeError::Unavailable)?;
        if consumed.insert(key.clone()) {
            Ok(())
        } else {
            Err(ApprovalConsumeError::AlreadyConsumed)
        }
    }
}

impl fmt::Debug for InMemoryApprovalConsumeStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InMemoryApprovalConsumeStore")
            .field("contents", &"<redacted>")
            .finish()
    }
}
