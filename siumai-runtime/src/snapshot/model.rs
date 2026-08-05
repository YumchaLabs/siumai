use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};
use siumai_core::{
    ContentPart, LanguageRequest, LanguageRequestError, LanguageResponse, Message,
    ToolBindingIdentity, ToolCall, ToolOutcome, ToolResult, Usage, UsageValue,
};
use thiserror::Error;

use crate::tool::{EffectCertainty, RecoveryPolicy, ToolExecutionAttempt, ToolIdempotencyKey};
use crate::{
    ModelTarget, ModelTransitionOutcome, ProjectionPolicy, RunReport, StepModelSelectorIdentity,
    project_history,
};

/// The only snapshot schema version understood by this release.
pub const RUN_SNAPSHOT_SCHEMA_VERSION: u16 = 4;

const MAX_ID_BYTES: usize = 256;
const MAX_FINGERPRINT_BYTES: usize = 1_024;
const MAX_ENGINE_VERSION_BYTES: usize = 128;
const MAX_REASON_CODE_BYTES: usize = 128;
const MAX_REASON_MESSAGE_BYTES: usize = 2_048;
const MAX_APPROVAL_ID_BYTES: usize = 256;
const MAX_PROVIDER_STATE_NAMESPACE_BYTES: usize = 256;
const MAX_CORRELATION_ID_BYTES: usize = 1_024;
const MAX_TOOL_CALL_ID_BYTES: usize = 1_024;

/// Why a run, lineage, or checkpoint identifier is invalid.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid {kind} identifier: {reason}")]
pub struct InvalidSnapshotId {
    kind: &'static str,
    reason: &'static str,
}

impl InvalidSnapshotId {
    const fn new(kind: &'static str, reason: &'static str) -> Self {
        Self { kind, reason }
    }
}

macro_rules! snapshot_id {
    ($name:ident, $kind:literal) => {
        #[doc = concat!("A validated ", $kind, " identifier.")]
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, InvalidSnapshotId> {
                let value = value.into();
                validate_snapshot_id(&value, $kind)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.0)
                    .finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl TryFrom<&str> for $name {
            type Error = InvalidSnapshotId;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl FromStr for $name {
            type Err = InvalidSnapshotId;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

snapshot_id!(RunId, "run");
snapshot_id!(LineageId, "lineage");
snapshot_id!(CheckpointId, "checkpoint");

fn validate_snapshot_id(value: &str, kind: &'static str) -> Result<(), InvalidSnapshotId> {
    if value.is_empty() {
        return Err(InvalidSnapshotId::new(kind, "must not be empty"));
    }
    if value.len() > MAX_ID_BYTES {
        return Err(InvalidSnapshotId::new(kind, "is too long"));
    }
    if !value.is_ascii() {
        return Err(InvalidSnapshotId::new(kind, "must be ASCII"));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(InvalidSnapshotId::new(
            kind,
            "may contain only ASCII letters, digits, '-', '_', '.', and ':'",
        ));
    }
    Ok(())
}

/// A stable digest used to reject incompatible continuations.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct SnapshotFingerprint(String);

impl SnapshotFingerprint {
    pub fn new(value: impl Into<String>) -> Result<Self, RunSnapshotError> {
        let value = value.into();
        validate_bounded_text(&value, "fingerprint", MAX_FINGERPRINT_BYTES, false)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SnapshotFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SnapshotFingerprint(..)")
    }
}

impl fmt::Display for SnapshotFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SnapshotFingerprint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Fingerprints that bind a continuation to its execution policy.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotFingerprints {
    pub options: SnapshotFingerprint,
    pub tool_catalog: SnapshotFingerprint,
    pub approval_policy: SnapshotFingerprint,
    pub model_selector: Option<StepModelSelectorIdentity>,
    pub projection_policy: ProjectionPolicy,
}

impl fmt::Debug for SnapshotFingerprints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SnapshotFingerprints")
            .field("options", &"<redacted>")
            .field("tool_catalog", &"<redacted>")
            .field("approval_policy", &"<redacted>")
            .field("model_selector", &self.model_selector)
            .field("projection_policy", &self.projection_policy)
            .finish()
    }
}

/// Runtime version that created a snapshot.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct SnapshotEngineVersion(String);

impl SnapshotEngineVersion {
    pub fn new(value: impl Into<String>) -> Result<Self, RunSnapshotError> {
        let value = value.into();
        validate_bounded_text(&value, "engine version", MAX_ENGINE_VERSION_BYTES, false)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SnapshotEngineVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("SnapshotEngineVersion")
            .field(&self.0)
            .finish()
    }
}

impl<'de> Deserialize<'de> for SnapshotEngineVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Immutable identity and lineage metadata for one durable checkpoint.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct SnapshotCheckpoint {
    engine_version: SnapshotEngineVersion,
    run_id: RunId,
    lineage_id: LineageId,
    checkpoint_id: CheckpointId,
    parent_checkpoint_id: Option<CheckpointId>,
}

impl SnapshotCheckpoint {
    pub fn new(
        engine_version: SnapshotEngineVersion,
        run_id: RunId,
        lineage_id: LineageId,
        checkpoint_id: CheckpointId,
        parent_checkpoint_id: Option<CheckpointId>,
    ) -> Result<Self, RunSnapshotError> {
        let checkpoint = Self {
            engine_version,
            run_id,
            lineage_id,
            checkpoint_id,
            parent_checkpoint_id,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn engine_version(&self) -> &SnapshotEngineVersion {
        &self.engine_version
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

    pub fn parent_checkpoint_id(&self) -> Option<&CheckpointId> {
        self.parent_checkpoint_id.as_ref()
    }

    fn validate(&self) -> Result<(), RunSnapshotError> {
        if self.parent_checkpoint_id.as_ref() == Some(&self.checkpoint_id) {
            return Err(RunSnapshotError::SelfParentCheckpoint);
        }
        Ok(())
    }
}

impl fmt::Debug for SnapshotCheckpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SnapshotCheckpoint")
            .field("engine_version", &self.engine_version)
            .field("run_id", &self.run_id)
            .field("lineage_id", &self.lineage_id)
            .field("checkpoint_id", &self.checkpoint_id)
            .field("parent_checkpoint_id", &self.parent_checkpoint_id)
            .finish()
    }
}

#[derive(Deserialize)]
struct SnapshotCheckpointWire {
    engine_version: SnapshotEngineVersion,
    run_id: RunId,
    lineage_id: LineageId,
    checkpoint_id: CheckpointId,
    parent_checkpoint_id: Option<CheckpointId>,
}

impl<'de> Deserialize<'de> for SnapshotCheckpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SnapshotCheckpointWire::deserialize(deserializer)?;
        Self::new(
            wire.engine_version,
            wire.run_id,
            wire.lineage_id,
            wire.checkpoint_id,
            wire.parent_checkpoint_id,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Sanitized machine-readable reason retained across process boundaries.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct SnapshotReason {
    code: String,
    message: Option<String>,
}

impl SnapshotReason {
    pub fn new(code: impl Into<String>, message: Option<String>) -> Result<Self, RunSnapshotError> {
        let code = code.into();
        validate_reason_code(&code)?;
        if let Some(message) = &message {
            validate_bounded_text(
                message,
                "snapshot reason message",
                MAX_REASON_MESSAGE_BYTES,
                true,
            )?;
        }
        Ok(Self { code, message })
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    pub(crate) fn runtime_code(code: &'static str) -> Self {
        Self {
            code: code.to_string(),
            message: None,
        }
    }

    fn dispatch_outcome_unknown() -> Self {
        Self {
            code: "dispatch_outcome_unknown".to_string(),
            message: None,
        }
    }
}

impl fmt::Debug for SnapshotReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SnapshotReason")
            .field("code", &self.code)
            .field("message", &self.message.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[derive(Deserialize)]
struct SnapshotReasonWire {
    code: String,
    message: Option<String>,
}

impl<'de> Deserialize<'de> for SnapshotReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SnapshotReasonWire::deserialize(deserializer)?;
        Self::new(wire.code, wire.message).map_err(serde::de::Error::custom)
    }
}

/// A terminal run outcome stored in a continuation.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SnapshotTerminal {
    Completed { reason: Option<SnapshotReason> },
    Cancelled { reason: SnapshotReason },
    Exhausted { reason: SnapshotReason },
    Failed { reason: SnapshotReason },
    Indeterminate { reason: SnapshotReason },
}

impl fmt::Debug for SnapshotTerminal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed { reason } => formatter
                .debug_struct("Completed")
                .field("reason", reason)
                .finish(),
            Self::Cancelled { reason } => formatter
                .debug_struct("Cancelled")
                .field("reason", reason)
                .finish(),
            Self::Exhausted { reason } => formatter
                .debug_struct("Exhausted")
                .field("reason", reason)
                .finish(),
            Self::Failed { reason } => formatter
                .debug_struct("Failed")
                .field("reason", reason)
                .finish(),
            Self::Indeterminate { reason } => formatter
                .debug_struct("Indeterminate")
                .field("reason", reason)
                .finish(),
        }
    }
}

/// Portable approval state. The executable binding remains host-owned.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingApprovalSnapshot {
    pub approval_id: String,
    pub call: ToolCall,
    pub binding: ToolBindingIdentity,
    pub claim_fingerprint: SnapshotFingerprint,
    pub expires_at_unix_ms: Option<u64>,
}

impl fmt::Debug for PendingApprovalSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingApprovalSnapshot")
            .field("approval_id", &self.approval_id)
            .field("call_id", &self.call.id)
            .field("tool_name", &self.call.name)
            .field("binding", &self.binding.name)
            .field("claim_fingerprint", &"<redacted>")
            .field("expires_at_unix_ms", &self.expires_at_unix_ms)
            .finish()
    }
}

/// Codec-tagged provider continuation retained without interpreting its payload.
///
/// Redacted diagnostics do not encrypt serialized bytes. Applications that
/// persist sensitive continuations must add confidentiality and integrity.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderStateSnapshot {
    pub namespace: String,
    pub correlation_id: Option<String>,
    pub encoding: String,
    pub payload: Vec<u8>,
}

impl fmt::Debug for ProviderStateSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderStateSnapshot")
            .field("namespace", &self.namespace)
            .field(
                "correlation_id",
                &self.correlation_id.as_ref().map(|_| "<redacted>"),
            )
            .field("encoding", &self.encoding)
            .field("payload_bytes", &self.payload.len())
            .finish()
    }
}

/// One local tool call whose executable identity has been frozen for resume.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedToolSnapshot {
    ordinal: u32,
    call: ToolCall,
    binding: ToolBindingIdentity,
    recovery_policy: RecoveryPolicy,
    stable_idempotency_key: Option<ToolIdempotencyKey>,
    attempt: ToolExecutionAttempt,
}

impl PreparedToolSnapshot {
    pub fn new(
        ordinal: u32,
        call: ToolCall,
        binding: ToolBindingIdentity,
        recovery_policy: RecoveryPolicy,
        stable_idempotency_key: Option<ToolIdempotencyKey>,
        attempt: ToolExecutionAttempt,
    ) -> Self {
        Self {
            ordinal,
            call,
            binding,
            recovery_policy,
            stable_idempotency_key,
            attempt,
        }
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    pub fn binding(&self) -> &ToolBindingIdentity {
        &self.binding
    }

    pub fn recovery_policy(&self) -> RecoveryPolicy {
        self.recovery_policy
    }

    pub fn stable_idempotency_key(&self) -> Option<&ToolIdempotencyKey> {
        self.stable_idempotency_key.as_ref()
    }

    pub fn attempt(&self) -> ToolExecutionAttempt {
        self.attempt
    }

    fn same_logical_work(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
            && self.call == other.call
            && self.binding == other.binding
            && self.recovery_policy == other.recovery_policy
            && self.stable_idempotency_key == other.stable_idempotency_key
    }
}

impl fmt::Debug for PreparedToolSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedToolSnapshot")
            .field("ordinal", &self.ordinal)
            .field("call_id", &self.call.id)
            .field("tool_name", &self.call.name)
            .field("binding", &self.binding.name)
            .field("recovery_policy", &self.recovery_policy)
            .field(
                "stable_idempotency_key",
                &self.stable_idempotency_key.as_ref().map(|_| "<redacted>"),
            )
            .field("attempt", &self.attempt)
            .finish()
    }
}

/// One result already completed inside an otherwise pending model step.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletedToolSnapshot {
    ordinal: u32,
    result: ToolResult,
}

impl CompletedToolSnapshot {
    pub fn new(ordinal: u32, result: ToolResult) -> Self {
        Self { ordinal, result }
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn result(&self) -> &ToolResult {
        &self.result
    }
}

impl fmt::Debug for CompletedToolSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompletedToolSnapshot")
            .field("ordinal", &self.ordinal)
            .field("call_id", &self.result.call_id)
            .field("tool_name", &self.result.name)
            .field("outcome", &"<redacted>")
            .finish()
    }
}

/// Frozen local-tool work for a model response that is not yet a completed step.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingStepSnapshot {
    index: u32,
    target: ModelTarget,
    response: LanguageResponse,
    prepared: Vec<PreparedToolSnapshot>,
    completed: Vec<CompletedToolSnapshot>,
    pending_approvals: Vec<PendingApprovalSnapshot>,
}

impl PendingStepSnapshot {
    pub fn new(
        index: u32,
        target: ModelTarget,
        response: LanguageResponse,
        prepared: Vec<PreparedToolSnapshot>,
        completed: Vec<CompletedToolSnapshot>,
        pending_approvals: Vec<PendingApprovalSnapshot>,
    ) -> Self {
        Self {
            index,
            target,
            response,
            prepared,
            completed,
            pending_approvals,
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn target(&self) -> &ModelTarget {
        &self.target
    }

    pub fn response(&self) -> &LanguageResponse {
        &self.response
    }

    pub fn prepared(&self) -> &[PreparedToolSnapshot] {
        &self.prepared
    }

    pub fn completed(&self) -> &[CompletedToolSnapshot] {
        &self.completed
    }

    pub fn pending_approvals(&self) -> &[PendingApprovalSnapshot] {
        &self.pending_approvals
    }
}

impl fmt::Debug for PendingStepSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingStepSnapshot")
            .field("index", &self.index)
            .field("target", &"<redacted>")
            .field("response", &"<redacted>")
            .field("prepared", &self.prepared)
            .field("completed", &self.completed)
            .field("pending_approvals", &self.pending_approvals)
            .finish()
    }
}

/// Frozen provider-owned work for a model response awaiting native continuation.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingProviderStepSnapshot {
    index: u32,
    target: ModelTarget,
    response: LanguageResponse,
    provider_state: Vec<ProviderStateSnapshot>,
}

impl PendingProviderStepSnapshot {
    pub fn new(
        index: u32,
        target: ModelTarget,
        response: LanguageResponse,
        provider_state: Vec<ProviderStateSnapshot>,
    ) -> Self {
        Self {
            index,
            target,
            response,
            provider_state,
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn target(&self) -> &ModelTarget {
        &self.target
    }

    pub fn response(&self) -> &LanguageResponse {
        &self.response
    }

    pub fn provider_state(&self) -> &[ProviderStateSnapshot] {
        &self.provider_state
    }
}

impl fmt::Debug for PendingProviderStepSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingProviderStepSnapshot")
            .field("index", &self.index)
            .field("target", &"<redacted>")
            .field("response", &"<redacted>")
            .field("provider_state", &self.provider_state)
            .finish()
    }
}

/// Stable label for validating and reporting continuation transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResumePointKind {
    AwaitingApprovals,
    AwaitingProvider,
    ReadyToDispatch,
    ReadyForModel,
    Terminal,
}

/// Exact continuation cursor for a durable run checkpoint.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResumePoint {
    AwaitingApprovals(PendingStepSnapshot),
    AwaitingProvider(PendingProviderStepSnapshot),
    ReadyToDispatch(PendingStepSnapshot),
    ReadyForModel { next_step: u32, target: ModelTarget },
    Terminal(SnapshotTerminal),
}

impl ResumePoint {
    pub fn kind(&self) -> ResumePointKind {
        match self {
            Self::AwaitingApprovals(_) => ResumePointKind::AwaitingApprovals,
            Self::AwaitingProvider(_) => ResumePointKind::AwaitingProvider,
            Self::ReadyToDispatch(_) => ResumePointKind::ReadyToDispatch,
            Self::ReadyForModel { .. } => ResumePointKind::ReadyForModel,
            Self::Terminal(_) => ResumePointKind::Terminal,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }

    pub fn pending_step(&self) -> Option<&PendingStepSnapshot> {
        match self {
            Self::AwaitingApprovals(step) | Self::ReadyToDispatch(step) => Some(step),
            Self::AwaitingProvider(_) | Self::ReadyForModel { .. } | Self::Terminal(_) => None,
        }
    }

    pub fn provider_step(&self) -> Option<&PendingProviderStepSnapshot> {
        match self {
            Self::AwaitingProvider(step) => Some(step),
            Self::AwaitingApprovals(_)
            | Self::ReadyToDispatch(_)
            | Self::ReadyForModel { .. }
            | Self::Terminal(_) => None,
        }
    }

    pub fn target(&self) -> Option<&ModelTarget> {
        match self {
            Self::AwaitingApprovals(step) | Self::ReadyToDispatch(step) => Some(step.target()),
            Self::AwaitingProvider(step) => Some(step.target()),
            Self::ReadyForModel { target, .. } => Some(target),
            Self::Terminal(_) => None,
        }
    }
}

impl fmt::Debug for ResumePoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AwaitingApprovals(step) => formatter
                .debug_tuple("AwaitingApprovals")
                .field(step)
                .finish(),
            Self::AwaitingProvider(step) => formatter
                .debug_tuple("AwaitingProvider")
                .field(step)
                .finish(),
            Self::ReadyToDispatch(step) => formatter
                .debug_tuple("ReadyToDispatch")
                .field(step)
                .finish(),
            Self::ReadyForModel { next_step, .. } => formatter
                .debug_struct("ReadyForModel")
                .field("next_step", next_step)
                .field("target", &"<redacted>")
                .finish(),
            Self::Terminal(terminal) => formatter.debug_tuple("Terminal").field(terminal).finish(),
        }
    }
}

/// Stable execution-log state for one tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolExecutionStatus {
    Prepared,
    Dispatched,
    Completed,
    Indeterminate,
}

impl fmt::Display for ToolExecutionStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepared => formatter.write_str("Prepared"),
            Self::Dispatched => formatter.write_str("Dispatched"),
            Self::Completed => formatter.write_str("Completed"),
            Self::Indeterminate => formatter.write_str("Indeterminate"),
        }
    }
}

/// Why a dispatched execution cannot be proven complete or absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IndeterminateReason {
    RecoveredAfterDispatch,
    CancellationAfterDispatch,
    DispatchOutcomeUnknown,
    CheckpointFailure,
}

/// One append-only tool execution event.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolExecutionEvent {
    Prepared {
        sequence: u64,
        occurred_at_unix_ms: u64,
        step: u32,
        tool: PreparedToolSnapshot,
    },
    Dispatched {
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: String,
        attempt: ToolExecutionAttempt,
        dispatch_id: Option<String>,
    },
    Completed {
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: String,
        attempt: ToolExecutionAttempt,
        outcome: ToolOutcome,
    },
    Indeterminate {
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: String,
        attempt: ToolExecutionAttempt,
        reason: IndeterminateReason,
    },
}

impl ToolExecutionEvent {
    pub fn prepared(
        sequence: u64,
        occurred_at_unix_ms: u64,
        step: u32,
        tool: PreparedToolSnapshot,
    ) -> Self {
        Self::Prepared {
            sequence,
            occurred_at_unix_ms,
            step,
            tool,
        }
    }

    pub fn dispatched(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        attempt: ToolExecutionAttempt,
        dispatch_id: Option<String>,
    ) -> Self {
        Self::Dispatched {
            sequence,
            occurred_at_unix_ms,
            call_id: call_id.into(),
            attempt,
            dispatch_id,
        }
    }

    pub fn completed(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        attempt: ToolExecutionAttempt,
        outcome: ToolOutcome,
    ) -> Self {
        Self::Completed {
            sequence,
            occurred_at_unix_ms,
            call_id: call_id.into(),
            attempt,
            outcome,
        }
    }

    pub fn indeterminate(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        attempt: ToolExecutionAttempt,
        reason: IndeterminateReason,
    ) -> Self {
        Self::Indeterminate {
            sequence,
            occurred_at_unix_ms,
            call_id: call_id.into(),
            attempt,
            reason,
        }
    }

    pub fn sequence(&self) -> u64 {
        match self {
            Self::Prepared { sequence, .. }
            | Self::Dispatched { sequence, .. }
            | Self::Completed { sequence, .. }
            | Self::Indeterminate { sequence, .. } => *sequence,
        }
    }

    pub fn call_id(&self) -> &str {
        match self {
            Self::Prepared { tool, .. } => &tool.call().id,
            Self::Dispatched { call_id, .. }
            | Self::Completed { call_id, .. }
            | Self::Indeterminate { call_id, .. } => call_id,
        }
    }

    pub fn attempt(&self) -> ToolExecutionAttempt {
        match self {
            Self::Prepared { tool, .. } => tool.attempt(),
            Self::Dispatched { attempt, .. }
            | Self::Completed { attempt, .. }
            | Self::Indeterminate { attempt, .. } => *attempt,
        }
    }

    pub fn prepared_tool(&self) -> Option<&PreparedToolSnapshot> {
        match self {
            Self::Prepared { tool, .. } => Some(tool),
            Self::Dispatched { .. } | Self::Completed { .. } | Self::Indeterminate { .. } => None,
        }
    }

    pub fn status(&self) -> ToolExecutionStatus {
        match self {
            Self::Prepared { .. } => ToolExecutionStatus::Prepared,
            Self::Dispatched { .. } => ToolExecutionStatus::Dispatched,
            Self::Completed { .. } => ToolExecutionStatus::Completed,
            Self::Indeterminate { .. } => ToolExecutionStatus::Indeterminate,
        }
    }
}

impl fmt::Debug for ToolExecutionEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolExecutionEvent")
            .field("sequence", &self.sequence())
            .field("call_id", &self.call_id())
            .field("status", &self.status())
            .field("attempt", &self.attempt())
            .finish()
    }
}

/// Why an execution event cannot be appended to its log.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ToolExecutionTransitionError {
    #[error("execution event sequence must be {expected}, got {actual}")]
    InvalidSequence { expected: u64, actual: u64 },
    #[error("tool call identifier is invalid")]
    InvalidCallId,
    #[error("tool call `{call_id}` has invalid attempt {attempt}")]
    InvalidAttempt { call_id: String, attempt: u32 },
    #[error("tool call `{call_id}` attempt must be {expected}, got {actual}")]
    AttemptMismatch {
        call_id: String,
        expected: u32,
        actual: u32,
    },
    #[error("tool call `{call_id}` exhausted execution attempts")]
    AttemptExhausted { call_id: String },
    #[error("tool call `{call_id}` changes frozen work between attempts")]
    PreparedWorkChanged { call_id: String },
    #[error("tool call `{call_id}` recovery policy does not permit another attempt")]
    RecoveryNotPermitted { call_id: String },
    #[error("tool call `{call_id}` cannot complete successfully before dispatch")]
    DirectSuccessRequiresDispatch { call_id: String },
    #[error("tool call `{call_id}` contains an invalid stable idempotency key")]
    InvalidStableIdempotencyKey { call_id: String },
    #[error("tool call `{call_id}` does not match its frozen binding name")]
    BindingNameMismatch { call_id: String },
    #[error("tool call `{call_id}` cannot transition from {from:?} to {to}")]
    InvalidTransition {
        call_id: String,
        from: Option<ToolExecutionStatus>,
        to: ToolExecutionStatus,
    },
}

/// Whether a checkpointed tool call may be considered for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolReplayDisposition {
    UnknownCall,
    ReadyToDispatch,
    DoNotReplayCompleted,
    HaltIndeterminate,
}

/// Validated append-only execution history.
#[derive(Clone, Default, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ToolExecutionLog {
    events: Vec<ToolExecutionEvent>,
}

impl ToolExecutionLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_events(
        events: Vec<ToolExecutionEvent>,
    ) -> Result<Self, ToolExecutionTransitionError> {
        validate_execution_events(&events)?;
        Ok(Self { events })
    }

    pub fn append(
        &mut self,
        event: ToolExecutionEvent,
    ) -> Result<(), ToolExecutionTransitionError> {
        let mut candidate = self.events.clone();
        candidate.push(event);
        validate_execution_events(&candidate)?;
        self.events = candidate;
        Ok(())
    }

    pub fn events(&self) -> &[ToolExecutionEvent] {
        &self.events
    }

    pub fn next_sequence(&self) -> u64 {
        self.events.len() as u64
    }

    pub fn status(&self, call_id: &str) -> Option<ToolExecutionStatus> {
        self.events
            .iter()
            .rev()
            .find(|event| event.call_id() == call_id)
            .map(ToolExecutionEvent::status)
    }

    pub fn attempt(&self, call_id: &str) -> Option<ToolExecutionAttempt> {
        self.events
            .iter()
            .rev()
            .find(|event| event.call_id() == call_id)
            .map(ToolExecutionEvent::attempt)
    }

    pub fn replay_disposition(&self, call_id: &str) -> ToolReplayDisposition {
        match self.status(call_id) {
            None => ToolReplayDisposition::UnknownCall,
            Some(ToolExecutionStatus::Prepared) => ToolReplayDisposition::ReadyToDispatch,
            Some(ToolExecutionStatus::Completed) => ToolReplayDisposition::DoNotReplayCompleted,
            Some(ToolExecutionStatus::Dispatched | ToolExecutionStatus::Indeterminate) => {
                ToolReplayDisposition::HaltIndeterminate
            }
        }
    }

    pub fn recover_dispatched(
        &mut self,
        observed_at_unix_ms: u64,
    ) -> Result<usize, ToolExecutionTransitionError> {
        let states = execution_states(&self.events)?;
        let dispatched = states
            .iter()
            .filter_map(|(call_id, state)| {
                (state.status == ToolExecutionStatus::Dispatched)
                    .then_some((call_id.clone(), state.prepared.attempt()))
            })
            .collect::<Vec<_>>();

        for (call_id, attempt) in &dispatched {
            self.append(ToolExecutionEvent::indeterminate(
                self.next_sequence(),
                observed_at_unix_ms,
                call_id.clone(),
                *attempt,
                IndeterminateReason::RecoveredAfterDispatch,
            ))?;
        }
        Ok(dispatched.len())
    }

    pub(crate) fn has_prefix(&self, prefix: &Self) -> bool {
        self.events.starts_with(&prefix.events)
    }
}

impl fmt::Debug for ToolExecutionLog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let completed = self
            .events
            .iter()
            .filter(|event| event.status() == ToolExecutionStatus::Completed)
            .count();
        let indeterminate = self
            .events
            .iter()
            .filter(|event| event.status() == ToolExecutionStatus::Indeterminate)
            .count();
        formatter
            .debug_struct("ToolExecutionLog")
            .field("events", &self.events.len())
            .field("completed", &completed)
            .field("indeterminate", &indeterminate)
            .finish()
    }
}

impl<'de> Deserialize<'de> for ToolExecutionLog {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let events = Vec::<ToolExecutionEvent>::deserialize(deserializer)?;
        Self::from_events(events).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone)]
struct ExecutionState {
    status: ToolExecutionStatus,
    step: u32,
    prepared: PreparedToolSnapshot,
}

fn execution_states(
    events: &[ToolExecutionEvent],
) -> Result<BTreeMap<String, ExecutionState>, ToolExecutionTransitionError> {
    let mut states = BTreeMap::<String, ExecutionState>::new();
    for (index, event) in events.iter().enumerate() {
        let expected = index as u64;
        if event.sequence() != expected {
            return Err(ToolExecutionTransitionError::InvalidSequence {
                expected,
                actual: event.sequence(),
            });
        }
        validate_tool_call_id(event.call_id())?;

        let call_id = event.call_id().to_string();
        if event.attempt().get() == 0 {
            return Err(ToolExecutionTransitionError::InvalidAttempt {
                call_id,
                attempt: event.attempt().get(),
            });
        }

        match event {
            ToolExecutionEvent::Prepared { step, tool, .. } => {
                if tool.binding().name != tool.call().name {
                    return Err(ToolExecutionTransitionError::BindingNameMismatch { call_id });
                }
                if tool
                    .stable_idempotency_key()
                    .is_some_and(|key| ToolIdempotencyKey::new(key.as_str()).is_err())
                {
                    return Err(ToolExecutionTransitionError::InvalidStableIdempotencyKey {
                        call_id,
                    });
                }

                match states.get(&call_id) {
                    None => {
                        if tool.attempt() != ToolExecutionAttempt::INITIAL {
                            return Err(ToolExecutionTransitionError::AttemptMismatch {
                                call_id,
                                expected: ToolExecutionAttempt::INITIAL.get(),
                                actual: tool.attempt().get(),
                            });
                        }
                    }
                    Some(previous) if previous.status == ToolExecutionStatus::Indeterminate => {
                        let expected_attempt =
                            previous.prepared.attempt().next().ok_or_else(|| {
                                ToolExecutionTransitionError::AttemptExhausted {
                                    call_id: call_id.clone(),
                                }
                            })?;
                        if tool.attempt() != expected_attempt {
                            return Err(ToolExecutionTransitionError::AttemptMismatch {
                                call_id,
                                expected: expected_attempt.get(),
                                actual: tool.attempt().get(),
                            });
                        }
                        if *step != previous.step || !tool.same_logical_work(&previous.prepared) {
                            return Err(ToolExecutionTransitionError::PreparedWorkChanged {
                                call_id,
                            });
                        }
                        if !previous.prepared.recovery_policy().permits_retry(
                            EffectCertainty::Indeterminate,
                            previous.prepared.stable_idempotency_key().is_some(),
                        ) {
                            return Err(ToolExecutionTransitionError::RecoveryNotPermitted {
                                call_id,
                            });
                        }
                    }
                    Some(previous) => {
                        return Err(ToolExecutionTransitionError::InvalidTransition {
                            call_id,
                            from: Some(previous.status),
                            to: ToolExecutionStatus::Prepared,
                        });
                    }
                }
                states.insert(
                    event.call_id().to_string(),
                    ExecutionState {
                        status: ToolExecutionStatus::Prepared,
                        step: *step,
                        prepared: tool.clone(),
                    },
                );
            }
            ToolExecutionEvent::Dispatched { attempt, .. } => {
                let Some(previous) = states.get_mut(&call_id) else {
                    return Err(ToolExecutionTransitionError::InvalidTransition {
                        call_id,
                        from: None,
                        to: ToolExecutionStatus::Dispatched,
                    });
                };
                if previous.status != ToolExecutionStatus::Prepared {
                    return Err(ToolExecutionTransitionError::InvalidTransition {
                        call_id,
                        from: Some(previous.status),
                        to: ToolExecutionStatus::Dispatched,
                    });
                }
                validate_event_attempt(&call_id, previous.prepared.attempt(), *attempt)?;
                previous.status = ToolExecutionStatus::Dispatched;
            }
            ToolExecutionEvent::Completed {
                attempt, outcome, ..
            } => {
                let Some(previous) = states.get_mut(&call_id) else {
                    return Err(ToolExecutionTransitionError::InvalidTransition {
                        call_id,
                        from: None,
                        to: ToolExecutionStatus::Completed,
                    });
                };
                validate_event_attempt(&call_id, previous.prepared.attempt(), *attempt)?;
                match previous.status {
                    ToolExecutionStatus::Prepared => {
                        if matches!(outcome, ToolOutcome::Success { .. }) {
                            return Err(
                                ToolExecutionTransitionError::DirectSuccessRequiresDispatch {
                                    call_id,
                                },
                            );
                        }
                    }
                    ToolExecutionStatus::Dispatched => {}
                    status => {
                        return Err(ToolExecutionTransitionError::InvalidTransition {
                            call_id,
                            from: Some(status),
                            to: ToolExecutionStatus::Completed,
                        });
                    }
                }
                previous.status = ToolExecutionStatus::Completed;
            }
            ToolExecutionEvent::Indeterminate { attempt, .. } => {
                let Some(previous) = states.get_mut(&call_id) else {
                    return Err(ToolExecutionTransitionError::InvalidTransition {
                        call_id,
                        from: None,
                        to: ToolExecutionStatus::Indeterminate,
                    });
                };
                if previous.status != ToolExecutionStatus::Dispatched {
                    return Err(ToolExecutionTransitionError::InvalidTransition {
                        call_id,
                        from: Some(previous.status),
                        to: ToolExecutionStatus::Indeterminate,
                    });
                }
                validate_event_attempt(&call_id, previous.prepared.attempt(), *attempt)?;
                previous.status = ToolExecutionStatus::Indeterminate;
            }
        }
    }
    Ok(states)
}

fn validate_execution_events(
    events: &[ToolExecutionEvent],
) -> Result<(), ToolExecutionTransitionError> {
    execution_states(events).map(drop)
}

fn validate_event_attempt(
    call_id: &str,
    expected: ToolExecutionAttempt,
    actual: ToolExecutionAttempt,
) -> Result<(), ToolExecutionTransitionError> {
    if actual != expected {
        return Err(ToolExecutionTransitionError::AttemptMismatch {
            call_id: call_id.to_string(),
            expected: expected.get(),
            actual: actual.get(),
        });
    }
    Ok(())
}

fn validate_tool_call_id(call_id: &str) -> Result<(), ToolExecutionTransitionError> {
    if call_id.is_empty()
        || call_id.len() > MAX_TOOL_CALL_ID_BYTES
        || call_id.chars().any(char::is_control)
    {
        return Err(ToolExecutionTransitionError::InvalidCallId);
    }
    Ok(())
}

/// A versioned, portable continuation captured only at a quiescent boundary.
///
/// Serialization is not an encryption boundary. Applications are responsible
/// for protecting snapshots that contain prompts or provider continuation data.
#[derive(Clone, PartialEq, Serialize)]
pub struct RunSnapshot {
    snapshot_version: u16,
    checkpoint: SnapshotCheckpoint,
    fingerprints: SnapshotFingerprints,
    continuation: LanguageRequest,
    report: RunReport,
    deadline_unix_ms: Option<u64>,
    resume_point: ResumePoint,
}

impl RunSnapshot {
    pub fn new(
        checkpoint: SnapshotCheckpoint,
        fingerprints: SnapshotFingerprints,
        continuation: LanguageRequest,
        report: RunReport,
        deadline_unix_ms: Option<u64>,
        resume_point: ResumePoint,
    ) -> Result<Self, RunSnapshotError> {
        let snapshot = Self {
            snapshot_version: RUN_SNAPSHOT_SCHEMA_VERSION,
            checkpoint,
            fingerprints,
            continuation,
            report,
            deadline_unix_ms,
            resume_point,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn snapshot_version(&self) -> u16 {
        self.snapshot_version
    }

    pub fn checkpoint(&self) -> &SnapshotCheckpoint {
        &self.checkpoint
    }

    pub fn engine_version(&self) -> &SnapshotEngineVersion {
        self.checkpoint.engine_version()
    }

    pub fn run_id(&self) -> &RunId {
        self.checkpoint.run_id()
    }

    pub fn lineage_id(&self) -> &LineageId {
        self.checkpoint.lineage_id()
    }

    pub fn checkpoint_id(&self) -> &CheckpointId {
        self.checkpoint.checkpoint_id()
    }

    pub fn parent_checkpoint_id(&self) -> Option<&CheckpointId> {
        self.checkpoint.parent_checkpoint_id()
    }

    pub fn target(&self) -> &ModelTarget {
        self.resume_point
            .target()
            .unwrap_or_else(|| self.report.current_target())
    }

    pub fn fingerprints(&self) -> &SnapshotFingerprints {
        &self.fingerprints
    }

    pub fn report(&self) -> &RunReport {
        &self.report
    }

    /// Complete provider-neutral request state for the next model call.
    pub fn continuation(&self) -> &LanguageRequest {
        &self.continuation
    }

    pub fn history(&self) -> &[Message] {
        self.report.messages()
    }

    pub fn pending_approvals(&self) -> &[PendingApprovalSnapshot] {
        self.resume_point
            .pending_step()
            .map_or(&[], PendingStepSnapshot::pending_approvals)
    }

    pub fn provider_state(&self) -> &[ProviderStateSnapshot] {
        self.resume_point
            .provider_step()
            .map_or(&[], PendingProviderStepSnapshot::provider_state)
    }

    pub fn execution_log(&self) -> &ToolExecutionLog {
        self.report.execution_log()
    }

    pub fn budget(&self) -> &crate::BudgetLedger {
        self.report.budget()
    }

    pub fn usage(&self) -> &Usage {
        self.report.usage()
    }

    pub fn deadline_unix_ms(&self) -> Option<u64> {
        self.deadline_unix_ms
    }

    pub fn resume_point(&self) -> &ResumePoint {
        &self.resume_point
    }

    /// Convert every execution left in `Dispatched` into `Indeterminate`.
    ///
    /// This operation never turns a completed execution back into dispatchable
    /// work and should be applied before resuming a deserialized continuation.
    pub fn recovered_for_resume(
        mut self,
        observed_at_unix_ms: u64,
    ) -> Result<Self, RunSnapshotError> {
        let pending_approvals = self.pending_approvals().len();
        let recovered = self
            .report
            .execution_log_mut()
            .recover_dispatched(observed_at_unix_ms)?;
        if recovered > 0 {
            for _ in 0..pending_approvals {
                self.report.budget_mut().release_pending_approval();
            }
            self.resume_point = ResumePoint::Terminal(SnapshotTerminal::Indeterminate {
                reason: SnapshotReason::dispatch_outcome_unknown(),
            });
        }
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), RunSnapshotError> {
        if self.snapshot_version != RUN_SNAPSHOT_SCHEMA_VERSION {
            return Err(RunSnapshotError::UnsupportedVersion {
                found: self.snapshot_version,
                supported: RUN_SNAPSHOT_SCHEMA_VERSION,
            });
        }
        self.checkpoint.validate()?;
        self.continuation.validate()?;
        if self.continuation.messages != self.report.messages() {
            return Err(RunSnapshotError::ContinuationHistoryMismatch);
        }
        validate_execution_events(self.report.execution_log().events())?;
        let expected_step = validate_report(&self.report)?;
        validate_model_transitions(
            &self.report,
            &self.resume_point,
            self.fingerprints.projection_policy,
            expected_step,
        )?;

        let expected_pending_approvals = match &self.resume_point {
            ResumePoint::AwaitingApprovals(step) => {
                validate_pending_step(step, &self.report, expected_step)?;
                if step.pending_approvals().is_empty() {
                    return Err(RunSnapshotError::MissingPendingApproval);
                }
                step.pending_approvals().len()
            }
            ResumePoint::AwaitingProvider(step) => {
                validate_provider_step(step, expected_step)?;
                0
            }
            ResumePoint::ReadyToDispatch(step) => {
                validate_pending_step(step, &self.report, expected_step)?;
                if !step.pending_approvals().is_empty() {
                    return Err(RunSnapshotError::UnexpectedPendingApproval);
                }
                0
            }
            ResumePoint::ReadyForModel { next_step, target } => {
                if *next_step != expected_step {
                    return Err(RunSnapshotError::ResumeStepMismatch {
                        expected: expected_step,
                        actual: *next_step,
                    });
                }
                if target != self.report.current_target()
                    && self.fingerprints.model_selector.is_none()
                {
                    return Err(RunSnapshotError::MissingModelSelectorIdentity);
                }
                0
            }
            ResumePoint::Terminal(_) => 0,
        };
        let actual_pending_approvals = self.report.budget().pending_approvals();
        if u64::try_from(expected_pending_approvals).unwrap_or(u64::MAX)
            != u64::from(actual_pending_approvals)
        {
            return Err(RunSnapshotError::PendingApprovalBudgetMismatch {
                expected: u64::try_from(expected_pending_approvals).unwrap_or(u64::MAX),
                actual: u64::from(actual_pending_approvals),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_successor(
        &self,
        successor: &Self,
    ) -> Result<(), RunSnapshotSuccessorError> {
        successor.validate()?;
        if self.run_id() != successor.run_id() {
            return Err(RunSnapshotSuccessorError::RunIdChanged);
        }
        if self.lineage_id() != successor.lineage_id() {
            return Err(RunSnapshotSuccessorError::LineageIdChanged);
        }
        if self.engine_version() != successor.engine_version() {
            return Err(RunSnapshotSuccessorError::EngineVersionChanged);
        }
        if self.report.initial_target() != successor.report.initial_target() {
            return Err(RunSnapshotSuccessorError::InitialTargetChanged);
        }
        if self.fingerprints != successor.fingerprints {
            return Err(RunSnapshotSuccessorError::FingerprintsChanged);
        }
        if successor.parent_checkpoint_id() != Some(self.checkpoint_id()) {
            return Err(RunSnapshotSuccessorError::ParentCheckpointMismatch {
                expected: self.checkpoint_id().clone(),
                actual: successor.parent_checkpoint_id().cloned(),
            });
        }
        if successor.checkpoint_id() == self.checkpoint_id() {
            return Err(RunSnapshotSuccessorError::ReusedCheckpoint);
        }
        validate_continuation_successor(self, successor)?;
        if !successor.report.steps().starts_with(self.report.steps()) {
            return Err(RunSnapshotSuccessorError::StepHistoryRegression);
        }
        if !successor
            .report
            .provider_deferred()
            .starts_with(self.report.provider_deferred())
        {
            return Err(RunSnapshotSuccessorError::ProviderHistoryRegression);
        }
        if !successor
            .report
            .execution_log()
            .has_prefix(self.report.execution_log())
        {
            return Err(RunSnapshotSuccessorError::ExecutionLogRegression);
        }
        validate_budget_successor(self.report.budget(), successor.report.budget())?;
        validate_usage_successor(self.report.usage(), successor.report.usage())?;
        validate_deadline_successor(self.deadline_unix_ms, successor.deadline_unix_ms)?;
        validate_resume_successor(&self.resume_point, &successor.resume_point)
    }
}

impl fmt::Debug for RunSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunSnapshot")
            .field("snapshot_version", &self.snapshot_version)
            .field("checkpoint", &self.checkpoint)
            .field("fingerprints", &"<redacted>")
            .field("history_messages", &self.continuation.messages.len())
            .field("generation", &"<redacted>")
            .field("request_tools", &self.continuation.tools.len())
            .field("tool_choice", &self.continuation.tool_choice.is_some())
            .field(
                "structured_output",
                &self.continuation.structured_output.is_some(),
            )
            .field("completed_steps", &self.report.steps().len())
            .field("provider_items", &self.report.provider_deferred().len())
            .field("execution_log", self.report.execution_log())
            .field("budget", self.report.budget())
            .field("usage", &"<redacted>")
            .field("deadline_unix_ms", &self.deadline_unix_ms)
            .field("resume_point", &self.resume_point)
            .finish()
    }
}

#[derive(Deserialize)]
struct RunSnapshotWire {
    snapshot_version: u16,
    checkpoint: SnapshotCheckpoint,
    fingerprints: SnapshotFingerprints,
    continuation: LanguageRequest,
    report: RunReport,
    deadline_unix_ms: Option<u64>,
    resume_point: ResumePoint,
}

impl<'de> Deserialize<'de> for RunSnapshot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RunSnapshotWire::deserialize(deserializer)?;
        let snapshot = Self {
            snapshot_version: wire.snapshot_version,
            checkpoint: wire.checkpoint,
            fingerprints: wire.fingerprints,
            continuation: wire.continuation,
            report: wire.report,
            deadline_unix_ms: wire.deadline_unix_ms,
            resume_point: wire.resume_point,
        };
        snapshot.validate().map_err(serde::de::Error::custom)?;
        Ok(snapshot)
    }
}

/// Invalid serialized or in-memory snapshot state.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RunSnapshotError {
    #[error("unsupported run snapshot version {found}; this release supports {supported}")]
    UnsupportedVersion { found: u16, supported: u16 },
    #[error(transparent)]
    InvalidContinuation(#[from] LanguageRequestError),
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("{field} must not contain surrounding whitespace")]
    SurroundingWhitespace { field: &'static str },
    #[error("{field} must not contain control characters")]
    ControlCharacter { field: &'static str },
    #[error("{field} must not exceed {maximum} bytes")]
    FieldTooLong { field: &'static str, maximum: usize },
    #[error("snapshot reason code contains unsupported characters")]
    InvalidReasonCode,
    #[error("a checkpoint cannot be its own parent")]
    SelfParentCheckpoint,
    #[error("report step index must be {expected}, got {actual}")]
    InvalidReportStepIndex { expected: u32, actual: u32 },
    #[error("report step index overflowed")]
    ReportStepIndexOverflow,
    #[error("report step {step} does not match the active model target")]
    ReportStepTargetMismatch { step: u32 },
    #[error("model transitions cannot target the initial model step")]
    ModelTransitionAtInitialStep,
    #[error("model transition step {actual} does not follow step {previous}")]
    ModelTransitionStepOutOfOrder { previous: u32, actual: u32 },
    #[error("model transition step {actual} exceeds the current report boundary {maximum}")]
    ModelTransitionStepBeyondReport { maximum: u32, actual: u32 },
    #[error("model transition at step {step} does not start from the active target")]
    ModelTransitionSourceMismatch { step: u32 },
    #[error("model transition at step {step} does not change the target")]
    ModelTransitionTargetUnchanged { step: u32 },
    #[error("model transition at step {step} uses a different projection policy")]
    ModelTransitionPolicyMismatch { step: u32 },
    #[error("model transition at step {step} follows a rejected transition")]
    ModelTransitionAfterRejection { step: u32 },
    #[error("rejected model transition at step {step} must be the terminal report boundary")]
    RejectedModelTransitionNotTerminal { step: u32 },
    #[error("resume point at step {step} does not match the active model target")]
    ResumeTargetMismatch { step: u32 },
    #[error("resume step index must be {expected}, got {actual}")]
    ResumeStepMismatch { expected: u32, actual: u32 },
    #[error("continuation messages must exactly match report message history")]
    ContinuationHistoryMismatch,
    #[error("a frozen model transition requires a versioned selector identity")]
    MissingModelSelectorIdentity,
    #[error("a pending step must contain at least one tool call")]
    PendingStepWithoutToolCalls,
    #[error("pending step has {prepared} prepared calls for {response_calls} response tool calls")]
    PreparedCallCountMismatch {
        response_calls: usize,
        prepared: usize,
    },
    #[error("prepared tool at position {position} has ordinal {actual}")]
    PreparedOrdinalMismatch { position: usize, actual: u32 },
    #[error("pending step contains duplicate tool call `{call_id}`")]
    DuplicatePreparedCall { call_id: String },
    #[error("prepared tool at ordinal {ordinal} does not match the response tool call")]
    PreparedCallMismatch { ordinal: u32 },
    #[error("prepared tool `{call_id}` uses binding `{binding}`")]
    PreparedBindingNameMismatch { call_id: String, binding: String },
    #[error("prepared tool `{call_id}` has no Prepared execution event")]
    MissingPreparedEvent { call_id: String },
    #[error("prepared tool `{call_id}` does not exactly match its Prepared execution event")]
    PreparedEventMismatch { call_id: String },
    #[error("completed tool ordinal {ordinal} appears more than once")]
    DuplicateCompletedOrdinal { ordinal: u32 },
    #[error("completed tool ordinal {actual} does not follow ordinal {previous}")]
    CompletedOrdinalOutOfOrder { previous: u32, actual: u32 },
    #[error("completed tool ordinal {ordinal} does not reference a prepared call")]
    CompletedCallNotPrepared { ordinal: u32 },
    #[error("completed tool ordinal {ordinal} has a mismatched result identity")]
    CompletedResultMismatch { ordinal: u32 },
    #[error("completed tool `{call_id}` does not exactly match its Completed execution event")]
    CompletedEventMismatch { call_id: String },
    #[error("prepared tool `{call_id}` has unresolved execution status {status:?}")]
    PreparedCallNotReady {
        call_id: String,
        status: Option<ToolExecutionStatus>,
    },
    #[error("pending approval identifiers must be unique")]
    DuplicateApprovalId,
    #[error("one tool call cannot have multiple pending approvals")]
    DuplicateApprovalCall,
    #[error("pending approval call `{call_id}` does not exactly match a prepared tool")]
    ApprovalPreparedMismatch { call_id: String },
    #[error("provider-state namespaces must be unique")]
    DuplicateProviderStateNamespace,
    #[error("AwaitingApprovals requires at least one pending approval")]
    MissingPendingApproval,
    #[error("ReadyToDispatch cannot retain pending approvals")]
    UnexpectedPendingApproval,
    #[error("pending-approval budget gauge is {actual}, but the resume point contains {expected}")]
    PendingApprovalBudgetMismatch { expected: u64, actual: u64 },
    #[error("AwaitingProvider requires at least one provider-state entry")]
    MissingProviderState,
    #[error(transparent)]
    InvalidExecutionTransition(#[from] ToolExecutionTransitionError),
}

/// Why a candidate checkpoint cannot follow the currently stored checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RunSnapshotSuccessorError {
    #[error(transparent)]
    InvalidSnapshot(#[from] RunSnapshotError),
    #[error("successor changes the run identifier")]
    RunIdChanged,
    #[error("successor changes the lineage identifier")]
    LineageIdChanged,
    #[error("successor changes the runtime engine version")]
    EngineVersionChanged,
    #[error("successor changes the report's initial model target")]
    InitialTargetChanged,
    #[error("successor changes immutable execution-policy fingerprints")]
    FingerprintsChanged,
    #[error("successor parent checkpoint must be {expected}, got {actual:?}")]
    ParentCheckpointMismatch {
        expected: CheckpointId,
        actual: Option<CheckpointId>,
    },
    #[error("successor reuses the current checkpoint identifier")]
    ReusedCheckpoint,
    #[error("successor rewrites or removes report message history")]
    MessageHistoryRegression,
    #[error("successor changes non-history continuation request state")]
    ContinuationStateChanged,
    #[error("successor rewrites or removes model-transition history")]
    ModelTransitionHistoryRegression,
    #[error("successor appends more than one model transition")]
    MultipleModelTransitions,
    #[error("successor appends a model transition outside a ready-for-model boundary")]
    UnexpectedModelTransition,
    #[error("successor omits the transition to the frozen ready-for-model target")]
    MissingModelTransition,
    #[error("successor model transition does not match the frozen selection")]
    ModelTransitionMismatch,
    #[error("successor continuation is not a reproducible history projection")]
    ProjectionResultMismatch,
    #[error("successor rewrites or removes completed step history")]
    StepHistoryRegression,
    #[error("successor rewrites or removes provider-native report history")]
    ProviderHistoryRegression,
    #[error("successor rewrites or removes execution-log history")]
    ExecutionLogRegression,
    #[error("successor budget `{dimension}` regresses from {previous} to {next}")]
    BudgetRegression {
        dimension: &'static str,
        previous: u64,
        next: u64,
    },
    #[error("successor usage `{dimension}` regresses from {previous} to {next}")]
    UsageRegression {
        dimension: &'static str,
        previous: u64,
        next: u64,
    },
    #[error("successor extends or removes deadline {previous:?} with {next:?}")]
    DeadlineRegression {
        previous: Option<u64>,
        next: Option<u64>,
    },
    #[error("resume point cannot transition from {from:?} to {to:?}")]
    InvalidResumeTransition {
        from: ResumePointKind,
        to: ResumePointKind,
    },
    #[error("successor changes immutable pending-step work")]
    PendingStepChanged,
    #[error("successor rewrites or removes completed work in a pending step")]
    PendingStepCompletionRegression,
    #[error("successor adds or changes a pending approval")]
    PendingApprovalRegression,
    #[error("successor changes immutable provider-step work")]
    ProviderStepChanged,
    #[error("successor resume step {actual} is earlier than {minimum}")]
    ResumeStepRegression { minimum: u32, actual: u32 },
    #[error("pending step {step} did not advance before returning to model step {next_step}")]
    PendingStepDidNotComplete { step: u32, next_step: u32 },
}

fn validate_continuation_successor(
    previous: &RunSnapshot,
    successor: &RunSnapshot,
) -> Result<(), RunSnapshotSuccessorError> {
    if !successor
        .report
        .model_transitions()
        .starts_with(previous.report.model_transitions())
    {
        return Err(RunSnapshotSuccessorError::ModelTransitionHistoryRegression);
    }
    let appended =
        &successor.report.model_transitions()[previous.report.model_transitions().len()..];
    if appended.len() > 1 {
        return Err(RunSnapshotSuccessorError::MultipleModelTransitions);
    }

    let frozen = match previous.resume_point() {
        ResumePoint::ReadyForModel { next_step, target } => Some((*next_step, target)),
        _ => None,
    };
    let source = previous.report.current_target();
    let Some(transition) = appended.first() else {
        if frozen.is_some_and(|(_, target)| target != source) {
            return Err(RunSnapshotSuccessorError::MissingModelTransition);
        }
        if !same_continuation_state(&previous.continuation, &successor.continuation) {
            return Err(RunSnapshotSuccessorError::ContinuationStateChanged);
        }
        if !successor
            .continuation
            .messages
            .starts_with(&previous.continuation.messages)
        {
            return Err(RunSnapshotSuccessorError::MessageHistoryRegression);
        }
        return Ok(());
    };

    let Some((next_step, frozen_target)) = frozen else {
        return Err(RunSnapshotSuccessorError::UnexpectedModelTransition);
    };
    if transition.step() != next_step
        || transition.source() != source
        || transition.target() != frozen_target
        || transition.policy() != previous.fingerprints.projection_policy
    {
        return Err(RunSnapshotSuccessorError::ModelTransitionMismatch);
    }

    match project_history(
        previous.continuation.clone(),
        source,
        frozen_target,
        previous.fingerprints.projection_policy,
    ) {
        Ok(projected) if transition.outcome() == ModelTransitionOutcome::Applied => {
            if transition.scope() != projected.scope()
                || transition.losses() != projected.losses()
                || !same_continuation_state(projected.request(), &successor.continuation)
                || !successor
                    .continuation
                    .messages
                    .starts_with(&projected.request().messages)
            {
                return Err(RunSnapshotSuccessorError::ProjectionResultMismatch);
            }
        }
        Err(error) if transition.outcome() == ModelTransitionOutcome::Rejected => {
            if transition.scope() != error.scope()
                || transition.losses() != error.losses()
                || successor.continuation != previous.continuation
            {
                return Err(RunSnapshotSuccessorError::ProjectionResultMismatch);
            }
        }
        _ => return Err(RunSnapshotSuccessorError::ModelTransitionMismatch),
    }
    Ok(())
}

fn same_continuation_state(previous: &LanguageRequest, next: &LanguageRequest) -> bool {
    previous.generation == next.generation
        && previous.tools == next.tools
        && previous.tool_choice == next.tool_choice
        && previous.structured_output == next.structured_output
}

fn validate_report(report: &RunReport) -> Result<u32, RunSnapshotError> {
    let mut expected = 0_u32;
    for step in report.steps() {
        if step.index() != expected {
            return Err(RunSnapshotError::InvalidReportStepIndex {
                expected,
                actual: step.index(),
            });
        }
        expected = expected
            .checked_add(1)
            .ok_or(RunSnapshotError::ReportStepIndexOverflow)?;
    }
    Ok(expected)
}

fn validate_model_transitions(
    report: &RunReport,
    resume_point: &ResumePoint,
    projection_policy: ProjectionPolicy,
    expected_step: u32,
) -> Result<(), RunSnapshotError> {
    let transitions = report.model_transitions();
    let mut transition_index = 0_usize;
    let mut last_transition_step = None;
    let mut active_target = report.initial_target().clone();
    let mut rejected_step = None;

    for step in report.steps() {
        while let Some(transition) = transitions.get(transition_index) {
            if transition.step() != step.index() {
                break;
            }
            validate_model_transition(
                transition,
                projection_policy,
                expected_step,
                &mut last_transition_step,
                &mut active_target,
                &mut rejected_step,
            )?;
            transition_index = transition_index.saturating_add(1);
        }
        if rejected_step.is_some() || step.target() != &active_target {
            return Err(RunSnapshotError::ReportStepTargetMismatch { step: step.index() });
        }
    }

    for transition in &transitions[transition_index..] {
        validate_model_transition(
            transition,
            projection_policy,
            expected_step,
            &mut last_transition_step,
            &mut active_target,
            &mut rejected_step,
        )?;
    }

    if let Some(step) = rejected_step
        && (step != expected_step || !resume_point.is_terminal())
    {
        return Err(RunSnapshotError::RejectedModelTransitionNotTerminal { step });
    }

    match resume_point {
        ResumePoint::AwaitingApprovals(step) | ResumePoint::ReadyToDispatch(step) => {
            if step.target() != &active_target {
                return Err(RunSnapshotError::ResumeTargetMismatch { step: step.index() });
            }
        }
        ResumePoint::AwaitingProvider(step) => {
            if step.target() != &active_target {
                return Err(RunSnapshotError::ResumeTargetMismatch { step: step.index() });
            }
        }
        ResumePoint::ReadyForModel { next_step, target } => {
            if last_transition_step == Some(*next_step) && target != &active_target {
                return Err(RunSnapshotError::ResumeTargetMismatch { step: *next_step });
            }
        }
        ResumePoint::Terminal(_) => {}
    }
    Ok(())
}

fn validate_model_transition(
    transition: &crate::ModelTransitionRecord,
    projection_policy: ProjectionPolicy,
    expected_step: u32,
    last_transition_step: &mut Option<u32>,
    active_target: &mut ModelTarget,
    rejected_step: &mut Option<u32>,
) -> Result<(), RunSnapshotError> {
    let step = transition.step();
    if step == 0 {
        return Err(RunSnapshotError::ModelTransitionAtInitialStep);
    }
    if step > expected_step {
        return Err(RunSnapshotError::ModelTransitionStepBeyondReport {
            maximum: expected_step,
            actual: step,
        });
    }
    if let Some(previous) = *last_transition_step
        && step <= previous
    {
        return Err(RunSnapshotError::ModelTransitionStepOutOfOrder {
            previous,
            actual: step,
        });
    }
    if rejected_step.is_some() {
        return Err(RunSnapshotError::ModelTransitionAfterRejection { step });
    }
    if transition.source() != active_target {
        return Err(RunSnapshotError::ModelTransitionSourceMismatch { step });
    }
    if transition.target() == transition.source() {
        return Err(RunSnapshotError::ModelTransitionTargetUnchanged { step });
    }
    if transition.policy() != projection_policy {
        return Err(RunSnapshotError::ModelTransitionPolicyMismatch { step });
    }

    *last_transition_step = Some(step);
    match transition.outcome() {
        ModelTransitionOutcome::Applied => active_target.clone_from(transition.target()),
        ModelTransitionOutcome::Rejected => *rejected_step = Some(step),
    }
    Ok(())
}

fn validate_pending_step(
    step: &PendingStepSnapshot,
    report: &RunReport,
    expected_step: u32,
) -> Result<(), RunSnapshotError> {
    if step.index() != expected_step {
        return Err(RunSnapshotError::ResumeStepMismatch {
            expected: expected_step,
            actual: step.index(),
        });
    }

    let response_calls = step
        .response()
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    if response_calls.is_empty() {
        return Err(RunSnapshotError::PendingStepWithoutToolCalls);
    }
    if response_calls.len() != step.prepared().len() {
        return Err(RunSnapshotError::PreparedCallCountMismatch {
            response_calls: response_calls.len(),
            prepared: step.prepared().len(),
        });
    }

    let mut prepared_call_ids = BTreeSet::new();
    for (position, prepared) in step.prepared().iter().enumerate() {
        let expected_ordinal =
            u32::try_from(position).map_err(|_| RunSnapshotError::PreparedOrdinalMismatch {
                position,
                actual: prepared.ordinal(),
            })?;
        if prepared.ordinal() != expected_ordinal {
            return Err(RunSnapshotError::PreparedOrdinalMismatch {
                position,
                actual: prepared.ordinal(),
            });
        }
        validate_tool_call_id(&prepared.call().id)?;
        if !prepared_call_ids.insert(prepared.call().id.as_str()) {
            return Err(RunSnapshotError::DuplicatePreparedCall {
                call_id: prepared.call().id.clone(),
            });
        }
        if response_calls[position] != prepared.call() {
            return Err(RunSnapshotError::PreparedCallMismatch {
                ordinal: prepared.ordinal(),
            });
        }
        if prepared.binding().name != prepared.call().name {
            return Err(RunSnapshotError::PreparedBindingNameMismatch {
                call_id: prepared.call().id.clone(),
                binding: prepared.binding().name.clone(),
            });
        }

        let prepared_event = report
            .execution_log()
            .events()
            .iter()
            .rev()
            .find_map(|event| {
                let ToolExecutionEvent::Prepared {
                    step: event_step,
                    tool,
                    ..
                } = event
                else {
                    return None;
                };
                (tool.call().id == prepared.call().id).then_some((event_step, tool))
            });
        let Some((event_step, event_tool)) = prepared_event else {
            return Err(RunSnapshotError::MissingPreparedEvent {
                call_id: prepared.call().id.clone(),
            });
        };
        if *event_step != step.index() || event_tool != prepared {
            return Err(RunSnapshotError::PreparedEventMismatch {
                call_id: prepared.call().id.clone(),
            });
        }
    }

    let mut completed_ordinals = BTreeSet::new();
    let mut previous_completed_ordinal = None;
    for completed in step.completed() {
        if !completed_ordinals.insert(completed.ordinal()) {
            return Err(RunSnapshotError::DuplicateCompletedOrdinal {
                ordinal: completed.ordinal(),
            });
        }
        if let Some(previous) = previous_completed_ordinal
            && completed.ordinal() <= previous
        {
            return Err(RunSnapshotError::CompletedOrdinalOutOfOrder {
                previous,
                actual: completed.ordinal(),
            });
        }
        previous_completed_ordinal = Some(completed.ordinal());
        let prepared = usize::try_from(completed.ordinal())
            .ok()
            .and_then(|ordinal| step.prepared().get(ordinal))
            .filter(|prepared| prepared.ordinal() == completed.ordinal())
            .ok_or(RunSnapshotError::CompletedCallNotPrepared {
                ordinal: completed.ordinal(),
            })?;
        if completed.result().call_id != prepared.call().id
            || completed.result().name != prepared.call().name
        {
            return Err(RunSnapshotError::CompletedResultMismatch {
                ordinal: completed.ordinal(),
            });
        }
        let completed_event = report
            .execution_log()
            .events()
            .iter()
            .rev()
            .find_map(|event| {
                let ToolExecutionEvent::Completed {
                    call_id,
                    attempt,
                    outcome,
                    ..
                } = event
                else {
                    return None;
                };
                (call_id == &prepared.call().id).then_some((attempt, outcome))
            });
        if completed_event.is_none_or(|(attempt, outcome)| {
            *attempt != prepared.attempt() || outcome != &completed.result().outcome
        }) {
            return Err(RunSnapshotError::CompletedEventMismatch {
                call_id: prepared.call().id.clone(),
            });
        }
    }

    for prepared in step.prepared() {
        let actual_status = report.execution_log().status(&prepared.call().id);
        let valid = if completed_ordinals.contains(&prepared.ordinal()) {
            actual_status == Some(ToolExecutionStatus::Completed)
        } else {
            matches!(
                actual_status,
                Some(ToolExecutionStatus::Prepared | ToolExecutionStatus::Dispatched)
            )
        };
        if !valid {
            return Err(RunSnapshotError::PreparedCallNotReady {
                call_id: prepared.call().id.clone(),
                status: actual_status,
            });
        }
    }

    let mut approval_ids = BTreeSet::new();
    let mut approval_calls = BTreeSet::new();
    for approval in step.pending_approvals() {
        validate_bounded_text(
            &approval.approval_id,
            "approval identifier",
            MAX_APPROVAL_ID_BYTES,
            false,
        )?;
        validate_tool_call_id(&approval.call.id)?;
        if !approval_ids.insert(approval.approval_id.as_str()) {
            return Err(RunSnapshotError::DuplicateApprovalId);
        }
        if !approval_calls.insert(approval.call.id.as_str()) {
            return Err(RunSnapshotError::DuplicateApprovalCall);
        }
        let prepared = step
            .prepared()
            .iter()
            .find(|prepared| prepared.call().id == approval.call.id);
        if prepared.is_none_or(|prepared| {
            prepared.call() != &approval.call
                || prepared.binding() != &approval.binding
                || completed_ordinals.contains(&prepared.ordinal())
                || report.execution_log().status(&approval.call.id)
                    != Some(ToolExecutionStatus::Prepared)
        }) {
            return Err(RunSnapshotError::ApprovalPreparedMismatch {
                call_id: approval.call.id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_provider_step(
    step: &PendingProviderStepSnapshot,
    expected_step: u32,
) -> Result<(), RunSnapshotError> {
    if step.index() != expected_step {
        return Err(RunSnapshotError::ResumeStepMismatch {
            expected: expected_step,
            actual: step.index(),
        });
    }
    if step.provider_state().is_empty() {
        return Err(RunSnapshotError::MissingProviderState);
    }

    let mut provider_namespaces = BTreeSet::new();
    for state in step.provider_state() {
        validate_bounded_text(
            &state.namespace,
            "provider-state namespace",
            MAX_PROVIDER_STATE_NAMESPACE_BYTES,
            false,
        )?;
        validate_bounded_text(
            &state.encoding,
            "provider-state encoding",
            MAX_PROVIDER_STATE_NAMESPACE_BYTES,
            false,
        )?;
        if let Some(correlation_id) = &state.correlation_id {
            validate_bounded_text(
                correlation_id,
                "provider correlation identifier",
                MAX_CORRELATION_ID_BYTES,
                false,
            )?;
        }
        if !provider_namespaces.insert(state.namespace.as_str()) {
            return Err(RunSnapshotError::DuplicateProviderStateNamespace);
        }
    }
    Ok(())
}

fn validate_budget_successor(
    previous: &crate::BudgetLedger,
    next: &crate::BudgetLedger,
) -> Result<(), RunSnapshotSuccessorError> {
    for (dimension, previous, next) in [
        (
            "model_steps",
            u64::from(previous.model_steps()),
            u64::from(next.model_steps()),
        ),
        (
            "tool_calls",
            u64::from(previous.tool_calls()),
            u64::from(next.tool_calls()),
        ),
        (
            "argument_bytes",
            previous.argument_bytes(),
            next.argument_bytes(),
        ),
        ("result_bytes", previous.result_bytes(), next.result_bytes()),
        ("known_tokens", previous.known_tokens(), next.known_tokens()),
        (
            "usage_steps_with_unknown_tokens",
            u64::from(previous.usage_steps_with_unknown_tokens()),
            u64::from(next.usage_steps_with_unknown_tokens()),
        ),
        (
            "known_cost_microunits",
            previous.known_cost_microunits(),
            next.known_cost_microunits(),
        ),
    ] {
        if next < previous {
            return Err(RunSnapshotSuccessorError::BudgetRegression {
                dimension,
                previous,
                next,
            });
        }
    }
    Ok(())
}

fn validate_usage_successor(
    previous: &Usage,
    next: &Usage,
) -> Result<(), RunSnapshotSuccessorError> {
    for (dimension, previous, next) in [
        ("input_tokens", previous.input_tokens, next.input_tokens),
        ("output_tokens", previous.output_tokens, next.output_tokens),
        ("total_tokens", previous.total_tokens, next.total_tokens),
        (
            "reasoning_tokens",
            previous.reasoning_tokens,
            next.reasoning_tokens,
        ),
        (
            "cache_read_tokens",
            previous.cache_read_tokens,
            next.cache_read_tokens,
        ),
        (
            "cache_write_tokens",
            previous.cache_write_tokens,
            next.cache_write_tokens,
        ),
        (
            "audio_input_tokens",
            previous.audio_input_tokens,
            next.audio_input_tokens,
        ),
        (
            "audio_output_tokens",
            previous.audio_output_tokens,
            next.audio_output_tokens,
        ),
        (
            "orchestration_tokens",
            previous.orchestration_tokens,
            next.orchestration_tokens,
        ),
    ] {
        if let (UsageValue::Known(previous), UsageValue::Known(next)) = (previous, next)
            && next < previous
        {
            return Err(RunSnapshotSuccessorError::UsageRegression {
                dimension,
                previous,
                next,
            });
        }
    }
    Ok(())
}

fn validate_deadline_successor(
    previous: Option<u64>,
    next: Option<u64>,
) -> Result<(), RunSnapshotSuccessorError> {
    if matches!((previous, next), (Some(_), None))
        || matches!((previous, next), (Some(previous), Some(next)) if next > previous)
    {
        return Err(RunSnapshotSuccessorError::DeadlineRegression { previous, next });
    }
    Ok(())
}

fn validate_resume_successor(
    previous: &ResumePoint,
    next: &ResumePoint,
) -> Result<(), RunSnapshotSuccessorError> {
    match (previous, next) {
        (ResumePoint::AwaitingApprovals(previous), ResumePoint::AwaitingApprovals(next))
        | (ResumePoint::AwaitingApprovals(previous), ResumePoint::ReadyToDispatch(next))
        | (ResumePoint::ReadyToDispatch(previous), ResumePoint::ReadyToDispatch(next)) => {
            validate_pending_step_successor(previous, next)
        }
        (
            ResumePoint::AwaitingApprovals(previous) | ResumePoint::ReadyToDispatch(previous),
            ResumePoint::ReadyForModel { next_step, .. },
        ) => validate_pending_step_advanced(previous.index(), *next_step),
        (
            ResumePoint::AwaitingApprovals(_) | ResumePoint::ReadyToDispatch(_),
            ResumePoint::Terminal(_),
        ) => Ok(()),
        (ResumePoint::AwaitingProvider(previous), ResumePoint::AwaitingProvider(next)) => {
            if previous.index() != next.index()
                || previous.target() != next.target()
                || previous.response() != next.response()
            {
                return Err(RunSnapshotSuccessorError::ProviderStepChanged);
            }
            Ok(())
        }
        (
            ResumePoint::AwaitingProvider(previous),
            ResumePoint::AwaitingApprovals(next) | ResumePoint::ReadyToDispatch(next),
        ) => validate_resume_step_floor(previous.index(), next.index()),
        (ResumePoint::AwaitingProvider(previous), ResumePoint::ReadyForModel { next_step, .. }) => {
            validate_pending_step_advanced(previous.index(), *next_step)
        }
        (ResumePoint::AwaitingProvider(_), ResumePoint::Terminal(_)) => Ok(()),
        (
            ResumePoint::ReadyForModel {
                next_step: previous,
                ..
            },
            ResumePoint::AwaitingApprovals(next) | ResumePoint::ReadyToDispatch(next),
        ) => validate_resume_step_floor(*previous, next.index()),
        (
            ResumePoint::ReadyForModel {
                next_step: previous,
                ..
            },
            ResumePoint::AwaitingProvider(next),
        ) => validate_resume_step_floor(*previous, next.index()),
        (
            ResumePoint::ReadyForModel {
                next_step: previous,
                ..
            },
            ResumePoint::ReadyForModel {
                next_step: next, ..
            },
        ) => validate_resume_step_floor(*previous, *next),
        (ResumePoint::ReadyForModel { .. }, ResumePoint::Terminal(_)) => Ok(()),
        _ => Err(RunSnapshotSuccessorError::InvalidResumeTransition {
            from: previous.kind(),
            to: next.kind(),
        }),
    }
}

fn validate_pending_step_successor(
    previous: &PendingStepSnapshot,
    next: &PendingStepSnapshot,
) -> Result<(), RunSnapshotSuccessorError> {
    if previous.index() != next.index()
        || previous.target() != next.target()
        || previous.response() != next.response()
        || previous.prepared().len() != next.prepared().len()
        || previous
            .prepared()
            .iter()
            .zip(next.prepared())
            .any(|(previous, next)| !previous.same_logical_work(next))
    {
        return Err(RunSnapshotSuccessorError::PendingStepChanged);
    }
    if previous
        .completed()
        .iter()
        .any(|completed| !next.completed().contains(completed))
    {
        return Err(RunSnapshotSuccessorError::PendingStepCompletionRegression);
    }
    if next
        .pending_approvals()
        .iter()
        .any(|approval| !previous.pending_approvals().contains(approval))
    {
        return Err(RunSnapshotSuccessorError::PendingApprovalRegression);
    }
    Ok(())
}

fn validate_pending_step_advanced(
    step: u32,
    next_step: u32,
) -> Result<(), RunSnapshotSuccessorError> {
    if next_step <= step {
        return Err(RunSnapshotSuccessorError::PendingStepDidNotComplete { step, next_step });
    }
    Ok(())
}

fn validate_resume_step_floor(minimum: u32, actual: u32) -> Result<(), RunSnapshotSuccessorError> {
    if actual < minimum {
        return Err(RunSnapshotSuccessorError::ResumeStepRegression { minimum, actual });
    }
    Ok(())
}

fn validate_bounded_text(
    value: &str,
    field: &'static str,
    maximum: usize,
    allow_empty: bool,
) -> Result<(), RunSnapshotError> {
    if !allow_empty && value.is_empty() {
        return Err(RunSnapshotError::EmptyField { field });
    }
    if value != value.trim() {
        return Err(RunSnapshotError::SurroundingWhitespace { field });
    }
    if value.len() > maximum {
        return Err(RunSnapshotError::FieldTooLong { field, maximum });
    }
    if value.chars().any(char::is_control) {
        return Err(RunSnapshotError::ControlCharacter { field });
    }
    Ok(())
}

fn validate_reason_code(code: &str) -> Result<(), RunSnapshotError> {
    validate_bounded_text(code, "snapshot reason code", MAX_REASON_CODE_BYTES, false)?;
    if !code
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(RunSnapshotError::InvalidReasonCode);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{
        ContentPart, ExecutionOwner, FinishReason, LanguageResponse, ModelId, ProviderId, ToolCall,
        ToolOutcome, ToolResult, Usage,
    };

    use super::*;
    use crate::tool::{RecoveryPolicy, ToolExecutionAttempt};

    fn target() -> ModelTarget {
        ModelTarget::new(
            ProviderId::new("snapshot-test").unwrap(),
            ModelId::new("snapshot-model").unwrap(),
        )
    }

    fn call(ordinal: u32) -> ToolCall {
        ToolCall {
            id: format!("call-{ordinal}"),
            name: format!("tool-{ordinal}"),
            arguments: json!({"ordinal": ordinal}),
            owner: ExecutionOwner::Local,
        }
    }

    fn prepared(ordinal: u32) -> PreparedToolSnapshot {
        PreparedToolSnapshot::new(
            ordinal,
            call(ordinal),
            siumai_core::ToolBindingIdentity {
                name: format!("tool-{ordinal}"),
                fingerprint: format!("binding-{ordinal}"),
            },
            RecoveryPolicy::NeverReplay,
            None,
            ToolExecutionAttempt::INITIAL,
        )
    }

    fn completed(ordinal: u32) -> CompletedToolSnapshot {
        CompletedToolSnapshot::new(
            ordinal,
            ToolResult {
                call_id: format!("call-{ordinal}"),
                name: format!("tool-{ordinal}"),
                outcome: ToolOutcome::Success {
                    value: json!({"ordinal": ordinal}),
                },
            },
        )
    }

    fn response(prepared: &[PreparedToolSnapshot]) -> LanguageResponse {
        LanguageResponse::completed(
            prepared
                .iter()
                .map(|tool| ContentPart::ToolCall(tool.call().clone()))
                .collect(),
            FinishReason::ToolCalls,
            Usage::default(),
        )
        .unwrap()
    }

    #[test]
    fn pending_successor_uses_completed_set_inclusion_not_vector_prefixes() {
        let prepared = (0..4).map(prepared).collect::<Vec<_>>();
        let response = response(&prepared);
        let previous = PendingStepSnapshot::new(
            0,
            target(),
            response.clone(),
            prepared.clone(),
            vec![completed(3)],
            Vec::new(),
        );
        let next = PendingStepSnapshot::new(
            0,
            target(),
            response,
            prepared,
            vec![completed(1), completed(3)],
            Vec::new(),
        );

        validate_pending_step_successor(&previous, &next).unwrap();
    }

    #[test]
    fn pending_snapshot_requires_completed_results_sorted_by_ordinal() {
        let prepared = (0..2).map(prepared).collect::<Vec<_>>();
        let response = response(&prepared);
        let mut log = ToolExecutionLog::new();
        for (sequence, tool) in prepared.iter().enumerate() {
            log.append(ToolExecutionEvent::prepared(
                u64::try_from(sequence).unwrap(),
                10,
                0,
                tool.clone(),
            ))
            .unwrap();
        }
        for (offset, ordinal) in [1_u32, 0].into_iter().enumerate() {
            let tool = &prepared[usize::try_from(ordinal).unwrap()];
            let sequence = 2_u64 + u64::try_from(offset).unwrap() * 2;
            log.append(ToolExecutionEvent::dispatched(
                sequence,
                20,
                &tool.call().id,
                tool.attempt(),
                None,
            ))
            .unwrap();
            log.append(ToolExecutionEvent::completed(
                sequence + 1,
                30,
                &tool.call().id,
                tool.attempt(),
                completed(ordinal).result().outcome.clone(),
            ))
            .unwrap();
        }
        let mut report = RunReport::new(target(), Vec::new());
        *report.execution_log_mut() = log;
        let step = PendingStepSnapshot::new(
            0,
            target(),
            response,
            prepared,
            vec![completed(1), completed(0)],
            Vec::new(),
        );

        assert_eq!(
            validate_pending_step(&step, &report, 0).unwrap_err(),
            RunSnapshotError::CompletedOrdinalOutOfOrder {
                previous: 1,
                actual: 0,
            }
        );
    }
}
