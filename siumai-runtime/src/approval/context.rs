use std::fmt;

use siumai_core::{ExecutionOwner, RouteId, ToolBindingIdentity};
use thiserror::Error;

use crate::options::ModelTarget;
use crate::snapshot::{CheckpointId, LineageId, RunId};

const MAX_CONTEXT_VALUE_BYTES: usize = 4096;
const MAX_FINGERPRINT_BYTES: usize = 1024;

/// Host-authenticated actor identity used to derive exact approval contexts.
///
/// This type intentionally cannot be deserialized. Remote requests may carry
/// similarly named fields, but trusted host code must authenticate them before
/// constructing this value.
#[derive(Clone, PartialEq, Eq)]
pub struct TrustIdentity {
    issuer: String,
    audience: String,
    subject: String,
    tenant: String,
}

impl TrustIdentity {
    pub fn new(
        issuer: impl Into<String>,
        audience: impl Into<String>,
        subject: impl Into<String>,
        tenant: impl Into<String>,
    ) -> Result<Self, TrustContextBuildError> {
        let identity = Self {
            issuer: issuer.into(),
            audience: audience.into(),
            subject: subject.into(),
            tenant: tenant.into(),
        };
        validate_context_value("issuer", &identity.issuer, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("audience", &identity.audience, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("subject", &identity.subject, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("tenant", &identity.tenant, MAX_CONTEXT_VALUE_BYTES)?;
        Ok(identity)
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
}

impl fmt::Debug for TrustIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustIdentity")
            .field("contents", &"<redacted>")
            .finish()
    }
}

/// Host-authenticated values that authorize exactly one frozen tool binding.
///
/// Runtime code derives the execution-bound fields from a snapshot and frozen
/// request. The host supplies only the already-authenticated [`TrustIdentity`].
#[derive(Clone, PartialEq, Eq)]
pub struct TrustContext {
    identity: TrustIdentity,
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
}

impl TrustContext {
    pub fn builder(identity: TrustIdentity) -> TrustContextBuilder {
        TrustContextBuilder::new(identity)
    }

    pub fn identity(&self) -> &TrustIdentity {
        &self.identity
    }

    pub fn issuer(&self) -> &str {
        self.identity.issuer()
    }

    pub fn audience(&self) -> &str {
        self.identity.audience()
    }

    pub fn subject(&self) -> &str {
        self.identity.subject()
    }

    pub fn tenant(&self) -> &str {
        self.identity.tenant()
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
}

impl fmt::Debug for TrustContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustContext")
            .field("contents", &"<redacted>")
            .finish()
    }
}

/// Builder used internally to derive one exact [`TrustContext`].
#[derive(Clone)]
pub struct TrustContextBuilder {
    identity: TrustIdentity,
    model_target: Option<ModelTarget>,
    run_id: Option<RunId>,
    lineage_id: Option<LineageId>,
    checkpoint_id: Option<CheckpointId>,
    execution_owner: Option<ExecutionOwner>,
    binding_identity: Option<ToolBindingIdentity>,
    tool_call_id: Option<String>,
    canonical_arguments_digest: Option<String>,
    catalog_fingerprint: Option<String>,
    policy_fingerprint: Option<String>,
}

impl TrustContextBuilder {
    pub fn new(identity: TrustIdentity) -> Self {
        Self {
            identity,
            model_target: None,
            run_id: None,
            lineage_id: None,
            checkpoint_id: None,
            execution_owner: None,
            binding_identity: None,
            tool_call_id: None,
            canonical_arguments_digest: None,
            catalog_fingerprint: None,
            policy_fingerprint: None,
        }
    }

    pub fn model_target(mut self, model_target: ModelTarget) -> Self {
        self.model_target = Some(model_target);
        self
    }

    pub fn run_id(mut self, run_id: RunId) -> Self {
        self.run_id = Some(run_id);
        self
    }

    pub fn lineage_id(mut self, lineage_id: LineageId) -> Self {
        self.lineage_id = Some(lineage_id);
        self
    }

    pub fn checkpoint_id(mut self, checkpoint_id: CheckpointId) -> Self {
        self.checkpoint_id = Some(checkpoint_id);
        self
    }

    pub fn execution_owner(mut self, execution_owner: ExecutionOwner) -> Self {
        self.execution_owner = Some(execution_owner);
        self
    }

    pub fn binding_identity(mut self, binding_identity: ToolBindingIdentity) -> Self {
        self.binding_identity = Some(binding_identity);
        self
    }

    pub fn tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    pub fn canonical_arguments_digest(mut self, digest: impl Into<String>) -> Self {
        self.canonical_arguments_digest = Some(digest.into());
        self
    }

    pub fn catalog_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.catalog_fingerprint = Some(fingerprint.into());
        self
    }

    pub fn policy_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.policy_fingerprint = Some(fingerprint.into());
        self
    }

    pub fn build(self) -> Result<TrustContext, TrustContextBuildError> {
        let model_target = required(self.model_target, "model_target")?;
        let run_id = required(self.run_id, "run_id")?;
        let lineage_id = required(self.lineage_id, "lineage_id")?;
        let checkpoint_id = required(self.checkpoint_id, "checkpoint_id")?;
        let execution_owner = required(self.execution_owner, "execution_owner")?;
        let binding_identity = required(self.binding_identity, "binding_identity")?;
        let tool_call_id = required(self.tool_call_id, "tool_call_id")?;
        let canonical_arguments_digest = required(
            self.canonical_arguments_digest,
            "canonical_arguments_digest",
        )?;
        let catalog_fingerprint = required(self.catalog_fingerprint, "catalog_fingerprint")?;
        let policy_fingerprint = required(self.policy_fingerprint, "policy_fingerprint")?;

        validate_context_value(
            "binding_identity.name",
            &binding_identity.name,
            MAX_CONTEXT_VALUE_BYTES,
        )?;
        validate_context_value(
            "binding_identity.fingerprint",
            &binding_identity.fingerprint,
            MAX_FINGERPRINT_BYTES,
        )?;
        validate_context_value("tool_call_id", &tool_call_id, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value(
            "canonical_arguments_digest",
            &canonical_arguments_digest,
            MAX_FINGERPRINT_BYTES,
        )?;
        validate_context_value(
            "catalog_fingerprint",
            &catalog_fingerprint,
            MAX_FINGERPRINT_BYTES,
        )?;
        validate_context_value(
            "policy_fingerprint",
            &policy_fingerprint,
            MAX_FINGERPRINT_BYTES,
        )?;

        let route = model_target.route().cloned();
        Ok(TrustContext {
            identity: self.identity,
            route,
            model_target,
            run_id,
            lineage_id,
            checkpoint_id,
            execution_owner,
            binding_identity,
            tool_call_id,
            canonical_arguments_digest,
            catalog_fingerprint,
            policy_fingerprint,
        })
    }
}

impl fmt::Debug for TrustContextBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustContextBuilder")
            .field("contents", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TrustContextBuildError {
    #[error("missing required trust-context field `{field}`")]
    MissingField { field: &'static str },
    #[error("invalid trust-context field `{field}`: {reason}")]
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
}

fn required<T>(value: Option<T>, field: &'static str) -> Result<T, TrustContextBuildError> {
    value.ok_or(TrustContextBuildError::MissingField { field })
}

pub(super) fn validate_context_value(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), TrustContextBuildError> {
    if value.trim().is_empty() {
        return Err(TrustContextBuildError::InvalidField {
            field,
            reason: "must not be empty",
        });
    }
    if value.len() > max_bytes {
        return Err(TrustContextBuildError::InvalidField {
            field,
            reason: "exceeds the allowed byte length",
        });
    }
    if value.chars().any(char::is_control) {
        return Err(TrustContextBuildError::InvalidField {
            field,
            reason: "must not contain control characters",
        });
    }
    Ok(())
}
