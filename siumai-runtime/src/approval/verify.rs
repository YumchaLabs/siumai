use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use super::{
    APPROVAL_CLAIMS_VERSION, ApprovalClaims, ApprovalConsumeError, ApprovalConsumeKey,
    ApprovalConsumeStore, ApprovalEnvelope, ApprovalVerifier, ApprovalVerifierError, TrustContext,
};

/// Authenticate, fully bind, and atomically consume one approval.
pub fn verify_and_consume(
    verifier: &(impl ApprovalVerifier + ?Sized),
    consume_store: &(impl ApprovalConsumeStore + ?Sized),
    envelope: &ApprovalEnvelope,
    expected: &TrustContext,
    now: SystemTime,
) -> Result<VerifiedApproval, ApprovalVerificationError> {
    let now_unix_ms = now
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ApprovalVerificationError::ClockBeforeUnixEpoch)?
        .as_millis()
        .try_into()
        .map_err(|_| ApprovalVerificationError::ClockOverflow)?;
    verify_and_consume_at_unix_ms(verifier, consume_store, envelope, expected, now_unix_ms)
}

/// Deterministic timestamp-injected variant of [`verify_and_consume`].
pub fn verify_and_consume_at_unix_ms(
    verifier: &(impl ApprovalVerifier + ?Sized),
    consume_store: &(impl ApprovalConsumeStore + ?Sized),
    envelope: &ApprovalEnvelope,
    expected: &TrustContext,
    now_unix_ms: u64,
) -> Result<VerifiedApproval, ApprovalVerificationError> {
    let claims = match verifier.verify(envelope) {
        Ok(claims) => claims,
        Err(ApprovalVerifierError::Rejected) => {
            return Err(ApprovalVerificationError::AuthenticityRejected);
        }
        Err(ApprovalVerifierError::Unavailable) => {
            return Err(ApprovalVerificationError::VerifierUnavailable);
        }
    };

    if claims.version() != APPROVAL_CLAIMS_VERSION {
        return Err(ApprovalVerificationError::UnsupportedVersion {
            version: claims.version(),
        });
    }
    claims
        .validate_shape()
        .map_err(|_| ApprovalVerificationError::MalformedClaims)?;
    if claims.expires_at_unix_ms() <= now_unix_ms {
        return Err(ApprovalVerificationError::Expired);
    }

    validate_context(&claims, expected)?;

    let consume_key = ApprovalConsumeKey::from_claims(&claims);
    match consume_store.consume_once(&consume_key) {
        Ok(()) => Ok(VerifiedApproval {
            claims,
            consumed_at_unix_ms: now_unix_ms,
        }),
        Err(ApprovalConsumeError::AlreadyConsumed) => {
            Err(ApprovalVerificationError::AlreadyConsumed)
        }
        Err(ApprovalConsumeError::Unavailable) => {
            Err(ApprovalVerificationError::ConsumeStoreUnavailable)
        }
    }
}

fn validate_context(
    claims: &ApprovalClaims,
    expected: &TrustContext,
) -> Result<(), ApprovalVerificationError> {
    require_equal(
        claims.issuer() == expected.issuer(),
        ApprovalClaimField::Issuer,
    )?;
    require_equal(
        claims.audience() == expected.audience(),
        ApprovalClaimField::Audience,
    )?;
    require_equal(
        claims.subject() == expected.subject(),
        ApprovalClaimField::Subject,
    )?;
    require_equal(
        claims.tenant() == expected.tenant(),
        ApprovalClaimField::Tenant,
    )?;
    require_equal(
        claims.route() == expected.route(),
        ApprovalClaimField::Route,
    )?;
    require_equal(
        claims.model_target() == expected.model_target(),
        ApprovalClaimField::ModelTarget,
    )?;
    require_equal(
        claims.run_lineage() == expected.run_lineage(),
        ApprovalClaimField::RunLineage,
    )?;
    require_equal(
        claims.checkpoint() == expected.checkpoint(),
        ApprovalClaimField::Checkpoint,
    )?;
    require_equal(
        claims.execution_owner() == expected.execution_owner(),
        ApprovalClaimField::ExecutionOwner,
    )?;
    require_equal(
        claims.binding_identity() == expected.binding_identity(),
        ApprovalClaimField::BindingIdentity,
    )?;
    require_equal(
        claims.tool_call_id() == expected.tool_call_id(),
        ApprovalClaimField::ToolCallId,
    )?;
    require_equal(
        claims.canonical_arguments_digest() == expected.canonical_arguments_digest(),
        ApprovalClaimField::CanonicalArgumentsDigest,
    )?;
    require_equal(
        claims.catalog_fingerprint() == expected.catalog_fingerprint(),
        ApprovalClaimField::CatalogFingerprint,
    )?;
    require_equal(
        claims.policy_fingerprint() == expected.policy_fingerprint(),
        ApprovalClaimField::PolicyFingerprint,
    )?;
    Ok(())
}

fn require_equal(
    matches: bool,
    field: ApprovalClaimField,
) -> Result<(), ApprovalVerificationError> {
    if matches {
        Ok(())
    } else {
        Err(ApprovalVerificationError::ContextMismatch { field })
    }
}

/// The field that failed exact host-context binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApprovalClaimField {
    Issuer,
    Audience,
    Subject,
    Tenant,
    Route,
    ModelTarget,
    RunLineage,
    Checkpoint,
    ExecutionOwner,
    BindingIdentity,
    ToolCallId,
    CanonicalArgumentsDigest,
    CatalogFingerprint,
    PolicyFingerprint,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ApprovalVerificationError {
    #[error("approval authenticity verification failed")]
    AuthenticityRejected,
    #[error("approval verifier is unavailable")]
    VerifierUnavailable,
    #[error("approval claims version {version} is unsupported")]
    UnsupportedVersion { version: u16 },
    #[error("approval claims are malformed")]
    MalformedClaims,
    #[error("approval has expired")]
    Expired,
    #[error("approval does not match trusted field {field:?}")]
    ContextMismatch { field: ApprovalClaimField },
    #[error("approval was already consumed")]
    AlreadyConsumed,
    #[error("approval consume store is unavailable")]
    ConsumeStoreUnavailable,
    #[error("approval verification clock is before the Unix epoch")]
    ClockBeforeUnixEpoch,
    #[error("approval verification clock cannot be represented in milliseconds")]
    ClockOverflow,
}

/// Proof that an authenticated approval matched and was consumed atomically.
///
/// This value is intentionally not `Clone`; the runtime should move it into
/// the exact frozen execution record that it authorizes.
#[derive(Debug)]
pub struct VerifiedApproval {
    claims: ApprovalClaims,
    consumed_at_unix_ms: u64,
}

impl VerifiedApproval {
    pub fn claims(&self) -> &ApprovalClaims {
        &self.claims
    }

    pub fn consumed_at_unix_ms(&self) -> u64 {
        self.consumed_at_unix_ms
    }

    pub fn into_claims(self) -> ApprovalClaims {
        self.claims
    }
}
