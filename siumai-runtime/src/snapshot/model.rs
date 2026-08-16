use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use siumai_core::{
    ContentPart, LanguageRequest, LanguageRequestError, LanguageResponse, Message,
    PartialLanguageOutput, ProviderScope, ToolBindingIdentity, ToolCall, ToolOutcome, ToolResult,
    Usage,
};
use thiserror::Error;

use crate::tool::{EffectCertainty, RecoveryPolicy, ToolExecutionAttempt, ToolIdempotencyKey};
use crate::{
    ModelTarget, ModelTransitionOutcome, ProjectionPolicy, RunReport, StepModelSelectorIdentity,
};

mod successor;
mod wire;

pub use successor::RunSnapshotSuccessorError;
pub use wire::RUN_SNAPSHOT_SCHEMA_VERSION;

const MAX_ID_BYTES: usize = 256;
const MAX_FINGERPRINT_BYTES: usize = 1_024;
const MAX_ENGINE_VERSION_BYTES: usize = 128;
const MAX_REASON_CODE_BYTES: usize = 128;
const MAX_REASON_MESSAGE_BYTES: usize = 2_048;
const MAX_APPROVAL_ID_BYTES: usize = 256;
const MAX_TOOL_CALL_ID_BYTES: usize = 512;

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
        formatter.write_str("<redacted>")
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
#[serde(deny_unknown_fields)]
pub struct SnapshotFingerprints {
    pub(crate) options: SnapshotFingerprint,
    pub(crate) tool_catalog: SnapshotFingerprint,
    pub(crate) approval_policy: SnapshotFingerprint,
    pub(crate) model_selector: Option<StepModelSelectorIdentity>,
    pub(crate) projection_policy: ProjectionPolicy,
}

impl SnapshotFingerprints {
    pub fn options(&self) -> &SnapshotFingerprint {
        &self.options
    }

    pub fn tool_catalog(&self) -> &SnapshotFingerprint {
        &self.tool_catalog
    }

    pub fn approval_policy(&self) -> &SnapshotFingerprint {
        &self.approval_policy
    }

    pub fn model_selector(&self) -> Option<&StepModelSelectorIdentity> {
        self.model_selector.as_ref()
    }

    pub fn projection_policy(&self) -> ProjectionPolicy {
        self.projection_policy
    }
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

/// Durable execution ABI identifier that created a snapshot.
///
/// This value is independent from both the snapshot schema version and the
/// crate's package version. Change the schema version only for incompatible
/// serialized-shape changes; change this ABI when execution or fingerprint
/// interpretation changes. The serialized field remains `engine_version` as
/// the stable durable-wire name.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct SnapshotEngineVersion(String);

impl SnapshotEngineVersion {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, RunSnapshotError> {
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
    pub(crate) fn new(
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
#[serde(deny_unknown_fields)]
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
    pub(crate) fn new(
        code: impl Into<String>,
        message: Option<String>,
    ) -> Result<Self, RunSnapshotError> {
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
#[serde(deny_unknown_fields)]
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

/// Stable label for a terminal run outcome stored in a continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SnapshotTerminalKind {
    Completed,
    Cancelled,
    Exhausted,
    Failed,
    Indeterminate,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum SnapshotTerminalState {
    Completed {
        reason: Option<SnapshotReason>,
    },
    Cancelled {
        reason: SnapshotReason,
        partial: Option<PartialLanguageOutput>,
    },
    Exhausted {
        reason: SnapshotReason,
        partial: Option<PartialLanguageOutput>,
    },
    Failed {
        reason: SnapshotReason,
        partial: Option<PartialLanguageOutput>,
    },
    Indeterminate {
        reason: SnapshotReason,
    },
}

/// A read-only terminal run outcome stored in a continuation.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SnapshotTerminal {
    state: SnapshotTerminalState,
}

impl SnapshotTerminal {
    pub fn kind(&self) -> SnapshotTerminalKind {
        match &self.state {
            SnapshotTerminalState::Completed { .. } => SnapshotTerminalKind::Completed,
            SnapshotTerminalState::Cancelled { .. } => SnapshotTerminalKind::Cancelled,
            SnapshotTerminalState::Exhausted { .. } => SnapshotTerminalKind::Exhausted,
            SnapshotTerminalState::Failed { .. } => SnapshotTerminalKind::Failed,
            SnapshotTerminalState::Indeterminate { .. } => SnapshotTerminalKind::Indeterminate,
        }
    }

    pub fn reason(&self) -> Option<&SnapshotReason> {
        match &self.state {
            SnapshotTerminalState::Completed { reason } => reason.as_ref(),
            SnapshotTerminalState::Cancelled { reason, .. }
            | SnapshotTerminalState::Exhausted { reason, .. }
            | SnapshotTerminalState::Failed { reason, .. }
            | SnapshotTerminalState::Indeterminate { reason } => Some(reason),
        }
    }

    pub fn partial(&self) -> Option<&PartialLanguageOutput> {
        match &self.state {
            SnapshotTerminalState::Cancelled { partial, .. }
            | SnapshotTerminalState::Exhausted { partial, .. }
            | SnapshotTerminalState::Failed { partial, .. } => partial.as_ref(),
            SnapshotTerminalState::Completed { .. }
            | SnapshotTerminalState::Indeterminate { .. } => None,
        }
    }

    pub(crate) fn completed(reason: Option<SnapshotReason>) -> Self {
        Self {
            state: SnapshotTerminalState::Completed { reason },
        }
    }

    pub(crate) fn cancelled(
        reason: SnapshotReason,
        partial: Option<PartialLanguageOutput>,
    ) -> Self {
        Self {
            state: SnapshotTerminalState::Cancelled { reason, partial },
        }
    }

    pub(crate) fn exhausted(
        reason: SnapshotReason,
        partial: Option<PartialLanguageOutput>,
    ) -> Self {
        Self {
            state: SnapshotTerminalState::Exhausted { reason, partial },
        }
    }

    pub(crate) fn failed(reason: SnapshotReason, partial: Option<PartialLanguageOutput>) -> Self {
        Self {
            state: SnapshotTerminalState::Failed { reason, partial },
        }
    }

    pub(crate) fn indeterminate(reason: SnapshotReason) -> Self {
        Self {
            state: SnapshotTerminalState::Indeterminate { reason },
        }
    }
}

impl fmt::Debug for SnapshotTerminal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.state {
            SnapshotTerminalState::Completed { reason } => formatter
                .debug_struct("Completed")
                .field("reason", reason)
                .finish(),
            SnapshotTerminalState::Cancelled { reason, partial } => formatter
                .debug_struct("Cancelled")
                .field("reason", reason)
                .field("has_partial", &partial.is_some())
                .finish(),
            SnapshotTerminalState::Exhausted { reason, partial } => formatter
                .debug_struct("Exhausted")
                .field("reason", reason)
                .field("has_partial", &partial.is_some())
                .finish(),
            SnapshotTerminalState::Failed { reason, partial } => formatter
                .debug_struct("Failed")
                .field("reason", reason)
                .field("has_partial", &partial.is_some())
                .finish(),
            SnapshotTerminalState::Indeterminate { reason } => formatter
                .debug_struct("Indeterminate")
                .field("reason", reason)
                .finish(),
        }
    }
}

/// Portable approval state. The executable binding remains host-owned.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingApprovalSnapshot {
    pub(crate) approval_id: String,
    pub(crate) call: ToolCall,
    pub(crate) binding: ToolBindingIdentity,
    pub(crate) claim_fingerprint: SnapshotFingerprint,
    pub(crate) expires_at_unix_ms: Option<u64>,
}

impl PendingApprovalSnapshot {
    pub fn approval_id(&self) -> &str {
        &self.approval_id
    }

    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    pub fn binding(&self) -> &ToolBindingIdentity {
        &self.binding
    }

    pub fn claim_fingerprint(&self) -> &SnapshotFingerprint {
        &self.claim_fingerprint
    }

    pub fn expires_at_unix_ms(&self) -> Option<u64> {
        self.expires_at_unix_ms
    }
}

impl fmt::Debug for PendingApprovalSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingApprovalSnapshot")
            .field("approval_id", &self.approval_id)
            .field("call_id", &self.call.id())
            .field("tool_name", &self.call.name())
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
#[serde(deny_unknown_fields)]
pub struct ProviderStateSnapshot {
    namespace: String,
    scope: ProviderScope,
    correlation_id: String,
    encoding: String,
    payload: Vec<u8>,
}

impl ProviderStateSnapshot {
    pub(crate) fn new(
        namespace: String,
        scope: ProviderScope,
        correlation_id: String,
        encoding: String,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            namespace,
            scope,
            correlation_id,
            encoding,
            payload,
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn scope(&self) -> &ProviderScope {
        &self.scope
    }

    pub fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    pub fn encoding(&self) -> &str {
        &self.encoding
    }

    /// Return the sensitive provider-owned continuation bytes.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

impl fmt::Debug for ProviderStateSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderStateSnapshot")
            .field("namespace", &self.namespace)
            .field("scope", &"<redacted>")
            .field("correlation_id", &"<redacted>")
            .field("encoding", &self.encoding)
            .field("payload_bytes", &self.payload.len())
            .finish()
    }
}

/// One local tool call whose executable identity has been frozen for resume.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedToolSnapshot {
    ordinal: u32,
    call: ToolCall,
    binding: ToolBindingIdentity,
    recovery_policy: RecoveryPolicy,
    stable_idempotency_key: Option<ToolIdempotencyKey>,
    attempt: ToolExecutionAttempt,
}

impl PreparedToolSnapshot {
    pub(crate) fn new(
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
            .field("call_id", &self.call.id())
            .field("tool_name", &self.call.name())
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
#[serde(deny_unknown_fields)]
pub struct CompletedToolSnapshot {
    ordinal: u32,
    result: ToolResult,
}

impl CompletedToolSnapshot {
    pub(crate) fn new(ordinal: u32, result: ToolResult) -> Self {
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
#[serde(deny_unknown_fields)]
pub struct PendingStepSnapshot {
    index: u32,
    target: ModelTarget,
    response: LanguageResponse,
    prepared: Vec<PreparedToolSnapshot>,
    completed: Vec<CompletedToolSnapshot>,
    pending_approvals: Vec<PendingApprovalSnapshot>,
}

impl PendingStepSnapshot {
    pub(crate) fn new(
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
#[serde(deny_unknown_fields)]
pub struct PendingProviderStepSnapshot {
    index: u32,
    target: ModelTarget,
    response: LanguageResponse,
    provider_state: Vec<ProviderStateSnapshot>,
}

impl PendingProviderStepSnapshot {
    pub(crate) fn new(
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

    pub(crate) fn matches_step(
        &self,
        index: u32,
        target: &ModelTarget,
        response: &LanguageResponse,
    ) -> bool {
        self.index == index && &self.target == target && &self.response == response
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

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum ResumePointState {
    AwaitingApprovals(PendingStepSnapshot),
    AwaitingProvider(PendingProviderStepSnapshot),
    ReadyToDispatch(PendingStepSnapshot),
    ReadyForModel { next_step: u32, target: ModelTarget },
    Terminal(SnapshotTerminal),
}

/// Exact read-only continuation cursor for a durable run checkpoint.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResumePoint {
    state: ResumePointState,
}

impl ResumePoint {
    pub fn kind(&self) -> ResumePointKind {
        match &self.state {
            ResumePointState::AwaitingApprovals(_) => ResumePointKind::AwaitingApprovals,
            ResumePointState::AwaitingProvider(_) => ResumePointKind::AwaitingProvider,
            ResumePointState::ReadyToDispatch(_) => ResumePointKind::ReadyToDispatch,
            ResumePointState::ReadyForModel { .. } => ResumePointKind::ReadyForModel,
            ResumePointState::Terminal(_) => ResumePointKind::Terminal,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(&self.state, ResumePointState::Terminal(_))
    }

    pub fn pending_step(&self) -> Option<&PendingStepSnapshot> {
        match &self.state {
            ResumePointState::AwaitingApprovals(step) | ResumePointState::ReadyToDispatch(step) => {
                Some(step)
            }
            ResumePointState::AwaitingProvider(_)
            | ResumePointState::ReadyForModel { .. }
            | ResumePointState::Terminal(_) => None,
        }
    }

    pub fn provider_step(&self) -> Option<&PendingProviderStepSnapshot> {
        match &self.state {
            ResumePointState::AwaitingProvider(step) => Some(step),
            ResumePointState::AwaitingApprovals(_)
            | ResumePointState::ReadyToDispatch(_)
            | ResumePointState::ReadyForModel { .. }
            | ResumePointState::Terminal(_) => None,
        }
    }

    pub fn target(&self) -> Option<&ModelTarget> {
        match &self.state {
            ResumePointState::AwaitingApprovals(step) | ResumePointState::ReadyToDispatch(step) => {
                Some(step.target())
            }
            ResumePointState::AwaitingProvider(step) => Some(step.target()),
            ResumePointState::ReadyForModel { target, .. } => Some(target),
            ResumePointState::Terminal(_) => None,
        }
    }

    pub fn next_step(&self) -> Option<u32> {
        match &self.state {
            ResumePointState::AwaitingApprovals(step) | ResumePointState::ReadyToDispatch(step) => {
                Some(step.index())
            }
            ResumePointState::AwaitingProvider(step) => Some(step.index()),
            ResumePointState::ReadyForModel { next_step, .. } => Some(*next_step),
            ResumePointState::Terminal(_) => None,
        }
    }

    pub fn terminal(&self) -> Option<&SnapshotTerminal> {
        match &self.state {
            ResumePointState::Terminal(terminal) => Some(terminal),
            ResumePointState::AwaitingApprovals(_)
            | ResumePointState::AwaitingProvider(_)
            | ResumePointState::ReadyToDispatch(_)
            | ResumePointState::ReadyForModel { .. } => None,
        }
    }

    pub(crate) fn awaiting_approvals(step: PendingStepSnapshot) -> Self {
        Self {
            state: ResumePointState::AwaitingApprovals(step),
        }
    }

    pub(crate) fn awaiting_provider(step: PendingProviderStepSnapshot) -> Self {
        Self {
            state: ResumePointState::AwaitingProvider(step),
        }
    }

    pub(crate) fn ready_to_dispatch(step: PendingStepSnapshot) -> Self {
        Self {
            state: ResumePointState::ReadyToDispatch(step),
        }
    }

    pub(crate) fn ready_for_model(next_step: u32, target: ModelTarget) -> Self {
        Self {
            state: ResumePointState::ReadyForModel { next_step, target },
        }
    }

    pub(crate) fn terminal_state(terminal: SnapshotTerminal) -> Self {
        Self {
            state: ResumePointState::Terminal(terminal),
        }
    }

    pub(crate) fn state(&self) -> &ResumePointState {
        &self.state
    }

    pub(crate) fn into_state(self) -> ResumePointState {
        self.state
    }
}

impl fmt::Debug for ResumePoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.state {
            ResumePointState::AwaitingApprovals(step) => formatter
                .debug_tuple("AwaitingApprovals")
                .field(step)
                .finish(),
            ResumePointState::AwaitingProvider(step) => formatter
                .debug_tuple("AwaitingProvider")
                .field(step)
                .finish(),
            ResumePointState::ReadyToDispatch(step) => formatter
                .debug_tuple("ReadyToDispatch")
                .field(step)
                .finish(),
            ResumePointState::ReadyForModel { next_step, .. } => formatter
                .debug_struct("ReadyForModel")
                .field("next_step", next_step)
                .field("target", &"<redacted>")
                .finish(),
            ResumePointState::Terminal(terminal) => {
                formatter.debug_tuple("Terminal").field(terminal).finish()
            }
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

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum ToolExecutionEventState {
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

/// One read-only append-only tool execution event.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolExecutionEvent {
    state: ToolExecutionEventState,
}

impl ToolExecutionEvent {
    pub(crate) fn prepared(
        sequence: u64,
        occurred_at_unix_ms: u64,
        step: u32,
        tool: PreparedToolSnapshot,
    ) -> Self {
        Self {
            state: ToolExecutionEventState::Prepared {
                sequence,
                occurred_at_unix_ms,
                step,
                tool,
            },
        }
    }

    pub(crate) fn dispatched(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        attempt: ToolExecutionAttempt,
    ) -> Self {
        Self {
            state: ToolExecutionEventState::Dispatched {
                sequence,
                occurred_at_unix_ms,
                call_id: call_id.into(),
                attempt,
            },
        }
    }

    pub(crate) fn completed(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        attempt: ToolExecutionAttempt,
        outcome: ToolOutcome,
    ) -> Self {
        Self {
            state: ToolExecutionEventState::Completed {
                sequence,
                occurred_at_unix_ms,
                call_id: call_id.into(),
                attempt,
                outcome,
            },
        }
    }

    pub(crate) fn indeterminate(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        attempt: ToolExecutionAttempt,
        reason: IndeterminateReason,
    ) -> Self {
        Self {
            state: ToolExecutionEventState::Indeterminate {
                sequence,
                occurred_at_unix_ms,
                call_id: call_id.into(),
                attempt,
                reason,
            },
        }
    }

    pub fn sequence(&self) -> u64 {
        match &self.state {
            ToolExecutionEventState::Prepared { sequence, .. }
            | ToolExecutionEventState::Dispatched { sequence, .. }
            | ToolExecutionEventState::Completed { sequence, .. }
            | ToolExecutionEventState::Indeterminate { sequence, .. } => *sequence,
        }
    }

    pub fn occurred_at_unix_ms(&self) -> u64 {
        match &self.state {
            ToolExecutionEventState::Prepared {
                occurred_at_unix_ms,
                ..
            }
            | ToolExecutionEventState::Dispatched {
                occurred_at_unix_ms,
                ..
            }
            | ToolExecutionEventState::Completed {
                occurred_at_unix_ms,
                ..
            }
            | ToolExecutionEventState::Indeterminate {
                occurred_at_unix_ms,
                ..
            } => *occurred_at_unix_ms,
        }
    }

    pub fn call_id(&self) -> &str {
        match &self.state {
            ToolExecutionEventState::Prepared { tool, .. } => tool.call().id(),
            ToolExecutionEventState::Dispatched { call_id, .. }
            | ToolExecutionEventState::Completed { call_id, .. }
            | ToolExecutionEventState::Indeterminate { call_id, .. } => call_id,
        }
    }

    pub fn attempt(&self) -> ToolExecutionAttempt {
        match &self.state {
            ToolExecutionEventState::Prepared { tool, .. } => tool.attempt(),
            ToolExecutionEventState::Dispatched { attempt, .. }
            | ToolExecutionEventState::Completed { attempt, .. }
            | ToolExecutionEventState::Indeterminate { attempt, .. } => *attempt,
        }
    }

    pub fn prepared_tool(&self) -> Option<&PreparedToolSnapshot> {
        match &self.state {
            ToolExecutionEventState::Prepared { tool, .. } => Some(tool),
            ToolExecutionEventState::Dispatched { .. }
            | ToolExecutionEventState::Completed { .. }
            | ToolExecutionEventState::Indeterminate { .. } => None,
        }
    }

    pub fn status(&self) -> ToolExecutionStatus {
        match &self.state {
            ToolExecutionEventState::Prepared { .. } => ToolExecutionStatus::Prepared,
            ToolExecutionEventState::Dispatched { .. } => ToolExecutionStatus::Dispatched,
            ToolExecutionEventState::Completed { .. } => ToolExecutionStatus::Completed,
            ToolExecutionEventState::Indeterminate { .. } => ToolExecutionStatus::Indeterminate,
        }
    }

    pub fn step(&self) -> Option<u32> {
        match &self.state {
            ToolExecutionEventState::Prepared { step, .. } => Some(*step),
            ToolExecutionEventState::Dispatched { .. }
            | ToolExecutionEventState::Completed { .. }
            | ToolExecutionEventState::Indeterminate { .. } => None,
        }
    }

    pub fn outcome(&self) -> Option<&ToolOutcome> {
        match &self.state {
            ToolExecutionEventState::Completed { outcome, .. } => Some(outcome),
            ToolExecutionEventState::Prepared { .. }
            | ToolExecutionEventState::Dispatched { .. }
            | ToolExecutionEventState::Indeterminate { .. } => None,
        }
    }

    pub fn indeterminate_reason(&self) -> Option<IndeterminateReason> {
        match &self.state {
            ToolExecutionEventState::Indeterminate { reason, .. } => Some(*reason),
            ToolExecutionEventState::Prepared { .. }
            | ToolExecutionEventState::Dispatched { .. }
            | ToolExecutionEventState::Completed { .. } => None,
        }
    }

    pub(crate) fn state(&self) -> &ToolExecutionEventState {
        &self.state
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
#[derive(Clone, PartialEq)]
pub struct ToolExecutionLog {
    events: Vec<ToolExecutionEvent>,
    states: BTreeMap<String, ExecutionState>,
}

impl ToolExecutionLog {
    pub(crate) fn new() -> Self {
        Self {
            events: Vec::new(),
            states: BTreeMap::new(),
        }
    }

    pub(crate) fn from_events(
        events: Vec<ToolExecutionEvent>,
    ) -> Result<Self, ToolExecutionTransitionError> {
        let states = execution_states(&events)?;
        Ok(Self { events, states })
    }

    pub(crate) fn append(
        &mut self,
        event: ToolExecutionEvent,
    ) -> Result<(), ToolExecutionTransitionError> {
        apply_execution_event(&mut self.states, self.events.len() as u64, &event)?;
        self.events.push(event);
        Ok(())
    }

    pub fn events(&self) -> &[ToolExecutionEvent] {
        &self.events
    }

    pub(crate) fn next_sequence(&self) -> u64 {
        self.events.len() as u64
    }

    pub fn status(&self, call_id: &str) -> Option<ToolExecutionStatus> {
        self.states.get(call_id).map(|state| state.status)
    }

    pub fn attempt(&self, call_id: &str) -> Option<ToolExecutionAttempt> {
        self.states
            .get(call_id)
            .map(|state| state.prepared.attempt())
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

    pub(crate) fn recover_dispatched(
        &mut self,
        observed_at_unix_ms: u64,
    ) -> Result<usize, ToolExecutionTransitionError> {
        let dispatched = self
            .states
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

impl Serialize for ToolExecutionLog {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.events.serialize(serializer)
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

#[derive(Clone, PartialEq)]
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
        apply_execution_event(&mut states, index as u64, event)?;
    }
    Ok(states)
}

fn apply_execution_event(
    states: &mut BTreeMap<String, ExecutionState>,
    expected_sequence: u64,
    event: &ToolExecutionEvent,
) -> Result<(), ToolExecutionTransitionError> {
    if event.sequence() != expected_sequence {
        return Err(ToolExecutionTransitionError::InvalidSequence {
            expected: expected_sequence,
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

    match event.state() {
        ToolExecutionEventState::Prepared { step, tool, .. } => {
            if tool.binding().name != tool.call().name() {
                return Err(ToolExecutionTransitionError::BindingNameMismatch { call_id });
            }
            if tool
                .stable_idempotency_key()
                .is_some_and(|key| ToolIdempotencyKey::new(key.as_str()).is_err())
            {
                return Err(ToolExecutionTransitionError::InvalidStableIdempotencyKey { call_id });
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
                    let expected_attempt = previous.prepared.attempt().next().ok_or_else(|| {
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
                        return Err(ToolExecutionTransitionError::PreparedWorkChanged { call_id });
                    }
                    if !previous.prepared.recovery_policy().permits_retry(
                        EffectCertainty::Indeterminate,
                        previous.prepared.stable_idempotency_key().is_some(),
                    ) {
                        return Err(ToolExecutionTransitionError::RecoveryNotPermitted { call_id });
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
                call_id,
                ExecutionState {
                    status: ToolExecutionStatus::Prepared,
                    step: *step,
                    prepared: tool.clone(),
                },
            );
        }
        ToolExecutionEventState::Dispatched { attempt, .. } => {
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
        ToolExecutionEventState::Completed {
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
                            ToolExecutionTransitionError::DirectSuccessRequiresDispatch { call_id },
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
        ToolExecutionEventState::Indeterminate { attempt, .. } => {
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
    Ok(())
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
    pub(crate) fn new(
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
        if !self.report.usage_is_settled()
            && (self.report.usage() != &Usage::default()
                || requires_settled_usage(&self.report, &self.resume_point))
        {
            return Err(RunSnapshotError::UsageSettlementMismatch);
        }
        let expected_step = validate_report(&self.report)?;
        validate_model_transitions(
            &self.report,
            &self.resume_point,
            self.fingerprints.projection_policy,
            expected_step,
        )?;

        let expected_pending_approvals = match self.resume_point.state() {
            ResumePointState::AwaitingApprovals(step) => {
                validate_pending_step(step, &self.report, expected_step)?;
                if step.pending_approvals().is_empty() {
                    return Err(RunSnapshotError::MissingPendingApproval);
                }
                step.pending_approvals().len()
            }
            ResumePointState::AwaitingProvider(step) => {
                validate_provider_step(step, &self.report, expected_step)?;
                0
            }
            ResumePointState::ReadyToDispatch(step) => {
                validate_pending_step(step, &self.report, expected_step)?;
                if !step.pending_approvals().is_empty() {
                    return Err(RunSnapshotError::UnexpectedPendingApproval);
                }
                0
            }
            ResumePointState::ReadyForModel { next_step, target } => {
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
                if self.report.provider_deferred_ledger().has_unresolved() {
                    return Err(RunSnapshotError::UnexpectedUnresolvedProviderState);
                }
                0
            }
            ResumePointState::Terminal(_) => 0,
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
}

fn requires_settled_usage(report: &RunReport, resume_point: &ResumePoint) -> bool {
    if !report.steps().is_empty() {
        return true;
    }
    match resume_point.state() {
        ResumePointState::AwaitingApprovals(_)
        | ResumePointState::AwaitingProvider(_)
        | ResumePointState::ReadyToDispatch(_) => true,
        ResumePointState::Terminal(terminal) => {
            terminal.kind() == SnapshotTerminalKind::Completed || terminal.partial().is_some()
        }
        ResumePointState::ReadyForModel { .. } => false,
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
    #[error("report step {step} contains invalid assistant-history omission records")]
    AssistantHistoryOmissionsMismatch { step: u32 },
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
    #[error("snapshot usage observations require settled report usage state")]
    UsageSettlementMismatch,
    #[error("a frozen model transition requires a versioned selector identity")]
    MissingModelSelectorIdentity,
    #[error("a pending step must contain at least one tool call")]
    PendingStepWithoutToolCalls,
    #[error("prepared tool ordinal {actual} does not follow ordinal {previous}")]
    PreparedOrdinalOutOfOrder { previous: u32, actual: u32 },
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
    #[error("completed tool ordinal {ordinal} has a mismatched result identity")]
    CompletedResultMismatch { ordinal: u32 },
    #[error("completed tool `{call_id}` does not exactly match its Completed execution event")]
    CompletedEventMismatch { call_id: String },
    #[error("completed unprepared tool `{call_id}` has local execution journal state {status:?}")]
    CompletedUnpreparedCallHasJournal {
        call_id: String,
        status: ToolExecutionStatus,
    },
    #[error("response tool ordinal {ordinal} is not represented by prepared or completed state")]
    MissingPendingToolState { ordinal: u32 },
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
    #[error("pending provider state does not exactly match the provider-deferred ledger")]
    ProviderStateProjectionMismatch,
    #[error("AwaitingApprovals requires at least one pending approval")]
    MissingPendingApproval,
    #[error("ReadyToDispatch cannot retain pending approvals")]
    UnexpectedPendingApproval,
    #[error("pending-approval budget gauge is {actual}, but the resume point contains {expected}")]
    PendingApprovalBudgetMismatch { expected: u64, actual: u64 },
    #[error("AwaitingProvider requires at least one provider-state entry")]
    MissingProviderState,
    #[error("ready-for-model state cannot retain unresolved provider work")]
    UnexpectedUnresolvedProviderState,
    #[error(transparent)]
    InvalidExecutionTransition(#[from] ToolExecutionTransitionError),
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
        if step.assistant_history_omissions()
            != step.response().project_assistant_history().omissions()
        {
            return Err(RunSnapshotError::AssistantHistoryOmissionsMismatch { step: step.index() });
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

    match resume_point.state() {
        ResumePointState::AwaitingApprovals(step) | ResumePointState::ReadyToDispatch(step) => {
            if step.target() != &active_target {
                return Err(RunSnapshotError::ResumeTargetMismatch { step: step.index() });
            }
        }
        ResumePointState::AwaitingProvider(step) => {
            if step.target() != &active_target {
                return Err(RunSnapshotError::ResumeTargetMismatch { step: step.index() });
            }
        }
        ResumePointState::ReadyForModel { next_step, target } => {
            if last_transition_step == Some(*next_step) && target != &active_target {
                return Err(RunSnapshotError::ResumeTargetMismatch { step: *next_step });
            }
        }
        ResumePointState::Terminal(_) => {}
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
    let mut prepared_call_ids = BTreeSet::new();
    let mut prepared_ordinals = BTreeSet::new();
    let mut previous_prepared_ordinal = None;
    for prepared in step.prepared() {
        if let Some(previous) = previous_prepared_ordinal
            && prepared.ordinal() <= previous
        {
            return Err(RunSnapshotError::PreparedOrdinalOutOfOrder {
                previous,
                actual: prepared.ordinal(),
            });
        }
        previous_prepared_ordinal = Some(prepared.ordinal());
        prepared_ordinals.insert(prepared.ordinal());
        validate_tool_call_id(prepared.call().id())?;
        if !prepared_call_ids.insert(prepared.call().id()) {
            return Err(RunSnapshotError::DuplicatePreparedCall {
                call_id: prepared.call().id().to_owned(),
            });
        }
        let matches_response = usize::try_from(prepared.ordinal())
            .ok()
            .and_then(|ordinal| response_calls.get(ordinal))
            .is_some_and(|call| *call == prepared.call());
        if !matches_response {
            return Err(RunSnapshotError::PreparedCallMismatch {
                ordinal: prepared.ordinal(),
            });
        }
        if prepared.binding().name != prepared.call().name() {
            return Err(RunSnapshotError::PreparedBindingNameMismatch {
                call_id: prepared.call().id().to_owned(),
                binding: prepared.binding().name.clone(),
            });
        }

        let prepared_event = report
            .execution_log()
            .events()
            .iter()
            .rev()
            .find_map(|event| {
                let ToolExecutionEventState::Prepared {
                    step: event_step,
                    tool,
                    ..
                } = event.state()
                else {
                    return None;
                };
                (tool.call().id() == prepared.call().id()).then_some((event_step, tool))
            });
        let Some((event_step, event_tool)) = prepared_event else {
            return Err(RunSnapshotError::MissingPreparedEvent {
                call_id: prepared.call().id().to_owned(),
            });
        };
        if *event_step != step.index() || event_tool != prepared {
            return Err(RunSnapshotError::PreparedEventMismatch {
                call_id: prepared.call().id().to_owned(),
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
        let response_call = usize::try_from(completed.ordinal())
            .ok()
            .and_then(|ordinal| response_calls.get(ordinal))
            .ok_or(RunSnapshotError::CompletedResultMismatch {
                ordinal: completed.ordinal(),
            })?;
        if completed.result().call_id != response_call.id()
            || completed.result().name != response_call.name()
        {
            return Err(RunSnapshotError::CompletedResultMismatch {
                ordinal: completed.ordinal(),
            });
        }
        if let Some(prepared) = step
            .prepared()
            .iter()
            .find(|prepared| prepared.ordinal() == completed.ordinal())
        {
            let completed_event = report
                .execution_log()
                .events()
                .iter()
                .rev()
                .find_map(|event| {
                    let ToolExecutionEventState::Completed {
                        call_id,
                        attempt,
                        outcome,
                        ..
                    } = event.state()
                    else {
                        return None;
                    };
                    (call_id == prepared.call().id()).then_some((attempt, outcome))
                });
            if completed_event.is_none_or(|(attempt, outcome)| {
                *attempt != prepared.attempt() || outcome != &completed.result().outcome
            }) {
                return Err(RunSnapshotError::CompletedEventMismatch {
                    call_id: prepared.call().id().to_owned(),
                });
            }
        } else if let Some(status) = report.execution_log().status(response_call.id()) {
            return Err(RunSnapshotError::CompletedUnpreparedCallHasJournal {
                call_id: response_call.id().to_owned(),
                status,
            });
        }
    }

    for ordinal in 0..response_calls.len() {
        let ordinal = u32::try_from(ordinal)
            .expect("response tool call ordinal is representable in a snapshot");
        if !prepared_ordinals.contains(&ordinal) && !completed_ordinals.contains(&ordinal) {
            return Err(RunSnapshotError::MissingPendingToolState { ordinal });
        }
    }

    for prepared in step.prepared() {
        let actual_status = report.execution_log().status(prepared.call().id());
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
                call_id: prepared.call().id().to_owned(),
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
        validate_tool_call_id(approval.call.id())?;
        if !approval_ids.insert(approval.approval_id.as_str()) {
            return Err(RunSnapshotError::DuplicateApprovalId);
        }
        if !approval_calls.insert(approval.call.id()) {
            return Err(RunSnapshotError::DuplicateApprovalCall);
        }
        let prepared = step
            .prepared()
            .iter()
            .find(|prepared| prepared.call().id() == approval.call.id());
        if prepared.is_none_or(|prepared| {
            prepared.call() != &approval.call
                || prepared.binding() != &approval.binding
                || completed_ordinals.contains(&prepared.ordinal())
                || report.execution_log().status(approval.call.id())
                    != Some(ToolExecutionStatus::Prepared)
        }) {
            return Err(RunSnapshotError::ApprovalPreparedMismatch {
                call_id: approval.call.id().to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_provider_step(
    step: &PendingProviderStepSnapshot,
    report: &RunReport,
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

    report
        .provider_deferred_ledger()
        .validate_pending_projection(step.target().scope(), step.provider_state())
        .map_err(|_| RunSnapshotError::ProviderStateProjectionMismatch)
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
        ContentPart, LanguageCompletionReason, LanguageResponse, ModelId, ProviderId, ToolCall,
        ToolOutcome, ToolResult, Usage,
    };

    use super::*;
    use crate::tool::{RecoveryPolicy, ToolExecutionAttempt, ToolJournal};

    fn target() -> ModelTarget {
        ModelTarget::new(
            ProviderId::new("snapshot-test").unwrap(),
            ModelId::new("snapshot-model").unwrap(),
        )
    }

    fn call(ordinal: u32) -> ToolCall {
        ToolCall::local(
            format!("call-{ordinal}"),
            format!("tool-{ordinal}"),
            json!({"ordinal": ordinal}),
        )
        .expect("valid tool call")
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
        response_for_calls(prepared.iter().map(|tool| tool.call().clone()).collect())
    }

    fn response_for_calls(calls: Vec<ToolCall>) -> LanguageResponse {
        LanguageResponse::completed(
            calls.into_iter().map(ContentPart::ToolCall).collect(),
            LanguageCompletionReason::ToolCalls,
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

        super::successor::validate_pending_step_successor(&previous, &next).unwrap();
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
                tool.call().id(),
                tool.attempt(),
            ))
            .unwrap();
            log.append(ToolExecutionEvent::completed(
                sequence + 1,
                30,
                tool.call().id(),
                tool.attempt(),
                completed(ordinal).result().outcome.clone(),
            ))
            .unwrap();
        }
        let mut report = RunReport::new(target(), Vec::new());
        report.replace_tool_journal(ToolJournal::from_log(log));
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

    #[test]
    fn pending_snapshot_accepts_sparse_prepared_subset_and_unbound_completion() {
        let prepared = vec![prepared(1)];
        let response = response_for_calls(vec![call(0), call(1)]);
        let unbound_result = CompletedToolSnapshot::new(
            0,
            ToolResult {
                call_id: "call-0".to_string(),
                name: "tool-0".to_string(),
                outcome: ToolOutcome::ExecutionFailed {
                    message: "unknown local tool".to_string(),
                    retryable: false,
                    details: None,
                },
            },
        );
        let mut log = ToolExecutionLog::new();
        log.append(ToolExecutionEvent::prepared(0, 10, 0, prepared[0].clone()))
            .unwrap();
        let mut report = RunReport::new(target(), Vec::new());
        report.replace_tool_journal(ToolJournal::from_log(log));
        let step = PendingStepSnapshot::new(
            0,
            target(),
            response,
            prepared,
            vec![unbound_result],
            Vec::new(),
        );

        validate_pending_step(&step, &report, 0).unwrap();
    }
}
