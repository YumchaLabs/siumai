use std::fmt;
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Deserializer, Serialize};
use siumai_core::{ToolBindingIdentity, ToolCall};
use thiserror::Error;

use crate::ModelTarget;

use super::{RecoveryPolicy, ToolEffect, ToolExecutionRequest};

const MAX_POLICY_FINGERPRINT_BYTES: usize = 1_024;
const MAX_DENIAL_REASON_BYTES: usize = 4_096;

/// Stable identity of host approval-decision semantics.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ApprovalPolicyFingerprint(String);

impl ApprovalPolicyFingerprint {
    pub fn new(value: impl Into<String>) -> Result<Self, ApprovalDecisionError> {
        let value = value.into();
        validate_text(
            &value,
            "approval policy fingerprint",
            MAX_POLICY_FINGERPRINT_BYTES,
        )?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApprovalPolicyFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApprovalPolicyFingerprint(..)")
    }
}

impl fmt::Display for ApprovalPolicyFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ApprovalPolicyFingerprint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Frozen, non-executable facts presented to a trusted host policy.
#[derive(Clone, PartialEq)]
pub struct ApprovalRequest {
    step: u32,
    ordinal: usize,
    target: ModelTarget,
    call: ToolCall,
    binding: ToolBindingIdentity,
    canonical_arguments_digest: String,
    effect: ToolEffect,
    recovery_policy: RecoveryPolicy,
    has_stable_idempotency_key: bool,
}

impl ApprovalRequest {
    pub(crate) fn from_frozen(
        step: u32,
        ordinal: usize,
        target: ModelTarget,
        request: &ToolExecutionRequest,
    ) -> Self {
        Self {
            step,
            ordinal,
            target,
            call: request.call().clone(),
            binding: request.binding_identity().clone(),
            canonical_arguments_digest: request.canonical_arguments_digest(),
            effect: request.effect(),
            recovery_policy: request.recovery_policy(),
            has_stable_idempotency_key: request.idempotency_key().is_some(),
        }
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn target(&self) -> &ModelTarget {
        &self.target
    }

    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    pub fn binding(&self) -> &ToolBindingIdentity {
        &self.binding
    }

    pub fn canonical_arguments_digest(&self) -> &str {
        &self.canonical_arguments_digest
    }

    pub fn effect(&self) -> ToolEffect {
        self.effect
    }

    pub fn recovery_policy(&self) -> RecoveryPolicy {
        self.recovery_policy
    }

    pub fn has_stable_idempotency_key(&self) -> bool {
        self.has_stable_idempotency_key
    }
}

impl fmt::Debug for ApprovalRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ApprovalRequest")
            .field("step", &self.step)
            .field("ordinal", &self.ordinal)
            .field("target", &self.target)
            .field("call_id", &self.call.id)
            .field("tool", &self.call.name)
            .field("arguments", &"<redacted>")
            .field("binding", &self.binding)
            .field("effect", &self.effect)
            .field("recovery_policy", &self.recovery_policy)
            .field(
                "has_stable_idempotency_key",
                &self.has_stable_idempotency_key,
            )
            .finish()
    }
}

/// Trusted host decision for one binding that requires approval.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApprovalDecision {
    Approve,
    Deny(ApprovalDenial),
    AwaitExternal,
}

impl ApprovalDecision {
    pub fn deny(reason: impl Into<String>) -> Result<Self, ApprovalDecisionError> {
        ApprovalDenial::new(reason).map(Self::Deny)
    }
}

/// Validated, bounded host-provided explanation for a denied tool call.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ApprovalDenial(String);

impl ApprovalDenial {
    pub fn new(reason: impl Into<String>) -> Result<Self, ApprovalDecisionError> {
        let reason = reason.into();
        validate_text(&reason, "approval denial reason", MAX_DENIAL_REASON_BYTES)?;
        Ok(Self(reason))
    }

    pub fn reason(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApprovalDenial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApprovalDenial(..)")
    }
}

impl fmt::Display for ApprovalDenial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ApprovalDenial {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let reason = String::deserialize(deserializer)?;
        Self::new(reason).map_err(serde::de::Error::custom)
    }
}

/// Boxed future returned by an object-safe [`ApprovalDecider`].
pub type ApprovalDecisionFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ApprovalDecision, ApprovalDecisionError>> + Send + 'a>>;

/// Host-owned policy for required local tool calls.
pub trait ApprovalDecider: Send + Sync + 'static {
    fn decide<'a>(&'a self, request: &'a ApprovalRequest) -> ApprovalDecisionFuture<'a>;

    fn fingerprint(&self) -> &ApprovalPolicyFingerprint;
}

/// Safe default policy that never auto-authorizes a required local tool.
#[derive(Clone)]
pub struct ExternalApprovalDecider {
    fingerprint: ApprovalPolicyFingerprint,
}

impl ExternalApprovalDecider {
    pub fn new(fingerprint: ApprovalPolicyFingerprint) -> Self {
        Self { fingerprint }
    }
}

impl Default for ExternalApprovalDecider {
    fn default() -> Self {
        Self {
            fingerprint: ApprovalPolicyFingerprint("siumai.external-approval.v1".to_owned()),
        }
    }
}

impl ApprovalDecider for ExternalApprovalDecider {
    fn decide<'a>(&'a self, _request: &'a ApprovalRequest) -> ApprovalDecisionFuture<'a> {
        Box::pin(async { Ok(ApprovalDecision::AwaitExternal) })
    }

    fn fingerprint(&self) -> &ApprovalPolicyFingerprint {
        &self.fingerprint
    }
}

impl fmt::Debug for ExternalApprovalDecider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalApprovalDecider")
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ApprovalDecisionError {
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("{field} must not contain surrounding whitespace")]
    SurroundingWhitespace { field: &'static str },
    #[error("{field} must not contain control characters")]
    ControlCharacter { field: &'static str },
    #[error("{field} is {actual} bytes; maximum is {maximum}")]
    TooLong {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("approval decision policy is unavailable: {message}")]
    Unavailable { message: String },
}

fn validate_text(
    value: &str,
    field: &'static str,
    maximum: usize,
) -> Result<(), ApprovalDecisionError> {
    if value.is_empty() {
        return Err(ApprovalDecisionError::Empty { field });
    }
    if value != value.trim() {
        return Err(ApprovalDecisionError::SurroundingWhitespace { field });
    }
    if value.len() > maximum {
        return Err(ApprovalDecisionError::TooLong {
            field,
            actual: value.len(),
            maximum,
        });
    }
    if value.chars().any(char::is_control) {
        return Err(ApprovalDecisionError::ControlCharacter { field });
    }
    Ok(())
}
