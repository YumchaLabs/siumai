use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use super::{
    APPROVAL_CLAIMS_VERSION, ApprovalClaims, ApprovalConsumeError, ApprovalConsumeKey,
    ApprovalConsumeStore, ApprovalEnvelope, ApprovalVerifier, ApprovalVerifierError, TrustContext,
};

/// One opaque approval paired with the exact host-derived context it must bind.
#[derive(Debug, Clone, Copy)]
pub struct ApprovalVerificationInput<'a> {
    envelope: &'a ApprovalEnvelope,
    expected: &'a TrustContext,
}

impl<'a> ApprovalVerificationInput<'a> {
    pub fn new(envelope: &'a ApprovalEnvelope, expected: &'a TrustContext) -> Self {
        Self { envelope, expected }
    }

    pub fn envelope(&self) -> &'a ApprovalEnvelope {
        self.envelope
    }

    pub fn expected(&self) -> &'a TrustContext {
        self.expected
    }
}

/// Authenticate, fully bind, and atomically consume one approval.
pub async fn verify_and_consume(
    verifier: &(impl ApprovalVerifier + ?Sized),
    consume_store: &(impl ApprovalConsumeStore + ?Sized),
    envelope: &ApprovalEnvelope,
    expected: &TrustContext,
    now: SystemTime,
) -> Result<VerifiedApproval, ApprovalVerificationError> {
    let mut verified = verify_and_consume_many(
        verifier,
        consume_store,
        &[ApprovalVerificationInput::new(envelope, expected)],
        now,
    )
    .await?;
    verified
        .pop()
        .ok_or(ApprovalVerificationError::MissingVerificationProof)
}

/// Authenticate, fully bind, and atomically consume a batch of approvals.
///
/// Every envelope is authenticated and bound before the consume store is
/// called. The store must then consume all replay keys or none of them.
pub async fn verify_and_consume_many(
    verifier: &(impl ApprovalVerifier + ?Sized),
    consume_store: &(impl ApprovalConsumeStore + ?Sized),
    inputs: &[ApprovalVerificationInput<'_>],
    now: SystemTime,
) -> Result<Vec<VerifiedApproval>, ApprovalVerificationError> {
    let now_unix_ms = now
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ApprovalVerificationError::ClockBeforeUnixEpoch)?
        .as_millis()
        .try_into()
        .map_err(|_| ApprovalVerificationError::ClockOverflow)?;
    verify_and_consume_many_at_unix_ms(verifier, consume_store, inputs, now_unix_ms).await
}

/// Deterministic timestamp-injected variant of [`verify_and_consume`].
pub async fn verify_and_consume_at_unix_ms(
    verifier: &(impl ApprovalVerifier + ?Sized),
    consume_store: &(impl ApprovalConsumeStore + ?Sized),
    envelope: &ApprovalEnvelope,
    expected: &TrustContext,
    now_unix_ms: u64,
) -> Result<VerifiedApproval, ApprovalVerificationError> {
    let mut verified = verify_and_consume_many_at_unix_ms(
        verifier,
        consume_store,
        &[ApprovalVerificationInput::new(envelope, expected)],
        now_unix_ms,
    )
    .await?;
    verified
        .pop()
        .ok_or(ApprovalVerificationError::MissingVerificationProof)
}

/// Deterministic timestamp-injected variant of [`verify_and_consume_many`].
pub async fn verify_and_consume_many_at_unix_ms(
    verifier: &(impl ApprovalVerifier + ?Sized),
    consume_store: &(impl ApprovalConsumeStore + ?Sized),
    inputs: &[ApprovalVerificationInput<'_>],
    now_unix_ms: u64,
) -> Result<Vec<VerifiedApproval>, ApprovalVerificationError> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let mut claims = Vec::with_capacity(inputs.len());
    let mut consume_keys = Vec::with_capacity(inputs.len());
    let mut unique_keys = BTreeSet::new();

    for input in inputs {
        let authenticated =
            authenticate_and_bind(verifier, input.envelope, input.expected, now_unix_ms)?;
        let consume_key = ApprovalConsumeKey::from_claims(&authenticated);
        if !unique_keys.insert(consume_key.clone()) {
            return Err(ApprovalVerificationError::DuplicateApproval);
        }
        claims.push(authenticated);
        consume_keys.push(consume_key);
    }

    consume_store
        .consume_many_once(&consume_keys)
        .await
        .map_err(map_consume_error)?;

    Ok(claims
        .into_iter()
        .map(|claims| VerifiedApproval {
            claims,
            consumed_at_unix_ms: now_unix_ms,
        })
        .collect())
}

fn authenticate_and_bind(
    verifier: &(impl ApprovalVerifier + ?Sized),
    envelope: &ApprovalEnvelope,
    expected: &TrustContext,
    now_unix_ms: u64,
) -> Result<ApprovalClaims, ApprovalVerificationError> {
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
    Ok(claims)
}

fn map_consume_error(error: ApprovalConsumeError) -> ApprovalVerificationError {
    match error {
        ApprovalConsumeError::AlreadyConsumed => ApprovalVerificationError::AlreadyConsumed,
        ApprovalConsumeError::DuplicateKey => ApprovalVerificationError::DuplicateApproval,
        ApprovalConsumeError::Unavailable => ApprovalVerificationError::ConsumeStoreUnavailable,
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
        claims.run_id() == expected.run_id(),
        ApprovalClaimField::RunId,
    )?;
    require_equal(
        claims.lineage_id() == expected.lineage_id(),
        ApprovalClaimField::LineageId,
    )?;
    require_equal(
        claims.checkpoint_id() == expected.checkpoint_id(),
        ApprovalClaimField::CheckpointId,
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
    RunId,
    LineageId,
    CheckpointId,
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
    #[error("approval verification batch contains the same approval more than once")]
    DuplicateApproval,
    #[error("approval consume store is unavailable")]
    ConsumeStoreUnavailable,
    #[error("approval verification clock is before the Unix epoch")]
    ClockBeforeUnixEpoch,
    #[error("approval verification clock cannot be represented in milliseconds")]
    ClockOverflow,
    #[error("approval verification completed without producing its authorization proof")]
    MissingVerificationProof,
}

/// Proof that an authenticated approval matched and was consumed atomically.
///
/// This value is intentionally not `Clone`; the runtime moves it into the
/// exact authorized tool call that it permits.
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
