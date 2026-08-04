use std::fmt;

use siumai_core::{ExecutionOwner, RouteId, ToolBindingIdentity};
use thiserror::Error;

use crate::options::ModelTarget;

const MAX_CONTEXT_VALUE_BYTES: usize = 4096;
const MAX_FINGERPRINT_BYTES: usize = 1024;

/// Host-authenticated values that authorize exactly one frozen tool binding.
///
/// This type intentionally cannot be deserialized. A remote request may carry
/// similarly named values, but only trusted host code can construct the
/// context used during approval verification.
#[derive(Clone, PartialEq, Eq)]
pub struct TrustContext {
    issuer: String,
    audience: String,
    subject: String,
    tenant: String,
    route: Option<RouteId>,
    model_target: ModelTarget,
    run_lineage: String,
    checkpoint: String,
    execution_owner: ExecutionOwner,
    binding_identity: ToolBindingIdentity,
    tool_call_id: String,
    canonical_arguments_digest: String,
    catalog_fingerprint: String,
    policy_fingerprint: String,
}

impl TrustContext {
    pub fn builder() -> TrustContextBuilder {
        TrustContextBuilder::default()
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

    pub fn run_lineage(&self) -> &str {
        &self.run_lineage
    }

    pub fn checkpoint(&self) -> &str {
        &self.checkpoint
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

/// Builder for a host-created [`TrustContext`].
#[derive(Clone, Default)]
pub struct TrustContextBuilder {
    issuer: Option<String>,
    audience: Option<String>,
    subject: Option<String>,
    tenant: Option<String>,
    model_target: Option<ModelTarget>,
    run_lineage: Option<String>,
    checkpoint: Option<String>,
    execution_owner: Option<ExecutionOwner>,
    binding_identity: Option<ToolBindingIdentity>,
    tool_call_id: Option<String>,
    canonical_arguments_digest: Option<String>,
    catalog_fingerprint: Option<String>,
    policy_fingerprint: Option<String>,
}

impl TrustContextBuilder {
    pub fn issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = Some(issuer.into());
        self
    }

    pub fn audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = Some(audience.into());
        self
    }

    pub fn subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    pub fn model_target(mut self, model_target: ModelTarget) -> Self {
        self.model_target = Some(model_target);
        self
    }

    pub fn run_lineage(mut self, run_lineage: impl Into<String>) -> Self {
        self.run_lineage = Some(run_lineage.into());
        self
    }

    pub fn checkpoint(mut self, checkpoint: impl Into<String>) -> Self {
        self.checkpoint = Some(checkpoint.into());
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
        let issuer = required(self.issuer, "issuer")?;
        let audience = required(self.audience, "audience")?;
        let subject = required(self.subject, "subject")?;
        let tenant = required(self.tenant, "tenant")?;
        let model_target = required(self.model_target, "model_target")?;
        let run_lineage = required(self.run_lineage, "run_lineage")?;
        let checkpoint = required(self.checkpoint, "checkpoint")?;
        let execution_owner = required(self.execution_owner, "execution_owner")?;
        let binding_identity = required(self.binding_identity, "binding_identity")?;
        let tool_call_id = required(self.tool_call_id, "tool_call_id")?;
        let canonical_arguments_digest = required(
            self.canonical_arguments_digest,
            "canonical_arguments_digest",
        )?;
        let catalog_fingerprint = required(self.catalog_fingerprint, "catalog_fingerprint")?;
        let policy_fingerprint = required(self.policy_fingerprint, "policy_fingerprint")?;

        validate_context_value("issuer", &issuer, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("audience", &audience, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("subject", &subject, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("tenant", &tenant, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("run_lineage", &run_lineage, MAX_CONTEXT_VALUE_BYTES)?;
        validate_context_value("checkpoint", &checkpoint, MAX_CONTEXT_VALUE_BYTES)?;
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
            issuer,
            audience,
            subject,
            tenant,
            route,
            model_target,
            run_lineage,
            checkpoint,
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
