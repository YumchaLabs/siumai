use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use siumai_core::{ExecutionOwner, RouteId, ToolBindingIdentity};
use thiserror::Error;

use crate::options::ModelTarget;
use crate::snapshot::{CheckpointId, LineageId, RunId};

use super::context::{TrustContext, TrustContextBuildError, validate_context_value};

pub const APPROVAL_CLAIMS_VERSION: u16 = 2;
pub const MAX_APPROVAL_ENVELOPE_BYTES: usize = 64 * 1024;
const MAX_NONCE_BYTES: usize = 512;
const MAX_KEY_ID_BYTES: usize = 512;

/// Versioned, immutable claims authenticated by an [`ApprovalSigner`](super::ApprovalSigner).
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalClaims {
    version: u16,
    issuer: String,
    audience: String,
    subject: String,
    tenant: String,
    route: Option<RouteId>,
    model_target: ModelTarget,
    run_id: RunId,
    lineage_id: LineageId,
    checkpoint_id: CheckpointId,
    execution_owner: ExecutionOwner,
    binding_identity: ToolBindingIdentity,
    tool_call_id: String,
    canonical_arguments_digest: String,
    catalog_fingerprint: String,
    policy_fingerprint: String,
    expires_at_unix_ms: u64,
    nonce: String,
    key_id: String,
}

impl ApprovalClaims {
    /// Create claims from a host-created trust context.
    ///
    /// `expires_at_unix_ms` is an absolute Unix timestamp in milliseconds.
    /// The nonce must be unpredictable and unique within the issuer/key scope.
    pub fn issue(
        context: &TrustContext,
        expires_at_unix_ms: u64,
        nonce: impl Into<String>,
        key_id: impl Into<String>,
    ) -> Result<Self, ApprovalClaimsError> {
        let claims = Self {
            version: APPROVAL_CLAIMS_VERSION,
            issuer: context.issuer().to_owned(),
            audience: context.audience().to_owned(),
            subject: context.subject().to_owned(),
            tenant: context.tenant().to_owned(),
            route: context.route().cloned(),
            model_target: context.model_target().clone(),
            run_id: context.run_id().clone(),
            lineage_id: context.lineage_id().clone(),
            checkpoint_id: context.checkpoint_id().clone(),
            execution_owner: context.execution_owner().clone(),
            binding_identity: context.binding_identity().clone(),
            tool_call_id: context.tool_call_id().to_owned(),
            canonical_arguments_digest: context.canonical_arguments_digest().to_owned(),
            catalog_fingerprint: context.catalog_fingerprint().to_owned(),
            policy_fingerprint: context.policy_fingerprint().to_owned(),
            expires_at_unix_ms,
            nonce: nonce.into(),
            key_id: key_id.into(),
        };
        claims.validate_shape()?;
        Ok(claims)
    }

    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    pub fn audience(&self) -> &str {
        &self.audience
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    pub fn route(&self) -> Option<&RouteId> {
        self.route.as_ref()
    }

    pub fn model_target(&self) -> &ModelTarget {
        &self.model_target
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn lineage_id(&self) -> &LineageId {
        &self.lineage_id
    }

    pub fn checkpoint_id(&self) -> &CheckpointId {
        &self.checkpoint_id
    }

    pub fn execution_owner(&self) -> &ExecutionOwner {
        &self.execution_owner
    }

    pub fn binding_identity(&self) -> &ToolBindingIdentity {
        &self.binding_identity
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn canonical_arguments_digest(&self) -> &str {
        &self.canonical_arguments_digest
    }

    pub fn catalog_fingerprint(&self) -> &str {
        &self.catalog_fingerprint
    }

    pub fn policy_fingerprint(&self) -> &str {
        &self.policy_fingerprint
    }

    pub fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }

    pub fn nonce(&self) -> &str {
        &self.nonce
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub(super) fn validate_shape(&self) -> Result<(), ApprovalClaimsError> {
        if self.expires_at_unix_ms == 0 {
            return Err(ApprovalClaimsError::InvalidField {
                field: "expires_at_unix_ms",
            });
        }
        if self.route.as_ref() != self.model_target.route() {
            return Err(ApprovalClaimsError::InconsistentRoute);
        }

        validate_claim_value("issuer", &self.issuer, 4096)?;
        validate_claim_value("audience", &self.audience, 4096)?;
        validate_claim_value("subject", &self.subject, 4096)?;
        validate_claim_value("tenant", &self.tenant, 4096)?;
        validate_claim_value("binding_identity.name", &self.binding_identity.name, 4096)?;
        validate_claim_value(
            "binding_identity.fingerprint",
            &self.binding_identity.fingerprint,
            1024,
        )?;
        validate_claim_value("tool_call_id", &self.tool_call_id, 4096)?;
        validate_claim_value(
            "canonical_arguments_digest",
            &self.canonical_arguments_digest,
            1024,
        )?;
        validate_claim_value("catalog_fingerprint", &self.catalog_fingerprint, 1024)?;
        validate_claim_value("policy_fingerprint", &self.policy_fingerprint, 1024)?;
        validate_claim_value("nonce", &self.nonce, MAX_NONCE_BYTES)?;
        validate_claim_value("key_id", &self.key_id, MAX_KEY_ID_BYTES)?;
        Ok(())
    }
}

impl fmt::Debug for ApprovalClaims {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ApprovalClaims")
            .field("version", &self.version)
            .field("expires_at_unix_ms", &self.expires_at_unix_ms)
            .field("contents", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ApprovalClaimsError {
    #[error("invalid approval claim `{field}`")]
    InvalidField { field: &'static str },
    #[error("approval route does not match the full model target")]
    InconsistentRoute,
}

impl From<TrustContextBuildError> for ApprovalClaimsError {
    fn from(error: TrustContextBuildError) -> Self {
        match error {
            TrustContextBuildError::MissingField { field }
            | TrustContextBuildError::InvalidField { field, .. } => Self::InvalidField { field },
        }
    }
}

fn validate_claim_value(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), ApprovalClaimsError> {
    validate_context_value(field, value, max_bytes).map_err(ApprovalClaimsError::from)
}

/// Opaque authenticated approval bytes.
///
/// The envelope format belongs to the signer/verifier implementation. Runtime
/// callers should persist or transport these bytes without inspecting them.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ApprovalEnvelope(Vec<u8>);

impl ApprovalEnvelope {
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self, ApprovalEnvelopeError> {
        let bytes = bytes.into();
        if bytes.is_empty() {
            return Err(ApprovalEnvelopeError::Empty);
        }
        if bytes.len() > MAX_APPROVAL_ENVELOPE_BYTES {
            return Err(ApprovalEnvelopeError::TooLarge {
                actual: bytes.len(),
                maximum: MAX_APPROVAL_ENVELOPE_BYTES,
            });
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

impl<'de> Deserialize<'de> for ApprovalEnvelope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Self::from_bytes(bytes).map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for ApprovalEnvelope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ApprovalEnvelope")
            .field("bytes", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ApprovalEnvelopeError {
    #[error("approval envelope must not be empty")]
    Empty,
    #[error("approval envelope is {actual} bytes; maximum is {maximum}")]
    TooLarge { actual: usize, maximum: usize },
}
