use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};
use siumai_core::{Message, ToolBindingIdentity, ToolCall, ToolOutcome, Usage};
use thiserror::Error;

use crate::ModelTarget;

/// The only snapshot schema version understood by this release.
pub const RUN_SNAPSHOT_SCHEMA_VERSION: u16 = 1;

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
}

impl fmt::Debug for SnapshotFingerprints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SnapshotFingerprints")
            .field("options", &"<redacted>")
            .field("tool_catalog", &"<redacted>")
            .field("approval_policy", &"<redacted>")
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

/// A quiescent point from which a run can be resumed.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SnapshotSuspension {
    AwaitingApprovals,
    AwaitingProvider,
    Paused { reason: SnapshotReason },
}

impl fmt::Debug for SnapshotSuspension {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AwaitingApprovals => formatter.write_str("AwaitingApprovals"),
            Self::AwaitingProvider => formatter.write_str("AwaitingProvider"),
            Self::Paused { reason } => formatter
                .debug_struct("Paused")
                .field("reason", reason)
                .finish(),
        }
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

/// A snapshot is constructible only from a suspended or terminal state.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SnapshotState {
    Suspended(SnapshotSuspension),
    Terminal(SnapshotTerminal),
}

impl SnapshotState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }
}

impl fmt::Debug for SnapshotState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Suspended(state) => formatter.debug_tuple("Suspended").field(state).finish(),
            Self::Terminal(state) => formatter.debug_tuple("Terminal").field(state).finish(),
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

/// One bounded budget dimension captured at a checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotBudgetCounter {
    pub consumed: u64,
    pub limit: Option<u64>,
}

impl SnapshotBudgetCounter {
    pub const fn new(consumed: u64, limit: Option<u64>) -> Self {
        Self { consumed, limit }
    }

    fn validate(self, dimension: &'static str) -> Result<(), RunSnapshotError> {
        if let Some(limit) = self.limit
            && self.consumed > limit
        {
            return Err(RunSnapshotError::BudgetExceeded {
                dimension,
                consumed: self.consumed,
                limit,
            });
        }
        Ok(())
    }
}

/// Portable budget ledger owned by the snapshot contract.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotBudgetLedger {
    pub model_steps: SnapshotBudgetCounter,
    pub tool_calls: SnapshotBudgetCounter,
    pub argument_bytes: SnapshotBudgetCounter,
    pub result_bytes: SnapshotBudgetCounter,
    pub snapshot_bytes: SnapshotBudgetCounter,
    pub pending_approvals: SnapshotBudgetCounter,
    pub known_tokens: SnapshotBudgetCounter,
    pub known_cost_microunits: SnapshotBudgetCounter,
    pub wall_time_millis: SnapshotBudgetCounter,
}

impl SnapshotBudgetLedger {
    fn validate(&self) -> Result<(), RunSnapshotError> {
        self.model_steps.validate("model_steps")?;
        self.tool_calls.validate("tool_calls")?;
        self.argument_bytes.validate("argument_bytes")?;
        self.result_bytes.validate("result_bytes")?;
        self.snapshot_bytes.validate("snapshot_bytes")?;
        self.pending_approvals.validate("pending_approvals")?;
        self.known_tokens.validate("known_tokens")?;
        self.known_cost_microunits
            .validate("known_cost_microunits")?;
        self.wall_time_millis.validate("wall_time_millis")
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
        call: ToolCall,
        binding: ToolBindingIdentity,
        idempotency_key: Option<String>,
    },
    Dispatched {
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: String,
        dispatch_id: Option<String>,
    },
    Completed {
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: String,
        outcome: ToolOutcome,
    },
    Indeterminate {
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: String,
        reason: IndeterminateReason,
    },
}

impl ToolExecutionEvent {
    pub fn prepared(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call: ToolCall,
        binding: ToolBindingIdentity,
        idempotency_key: Option<String>,
    ) -> Self {
        Self::Prepared {
            sequence,
            occurred_at_unix_ms,
            call,
            binding,
            idempotency_key,
        }
    }

    pub fn dispatched(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        dispatch_id: Option<String>,
    ) -> Self {
        Self::Dispatched {
            sequence,
            occurred_at_unix_ms,
            call_id: call_id.into(),
            dispatch_id,
        }
    }

    pub fn completed(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        outcome: ToolOutcome,
    ) -> Self {
        Self::Completed {
            sequence,
            occurred_at_unix_ms,
            call_id: call_id.into(),
            outcome,
        }
    }

    pub fn indeterminate(
        sequence: u64,
        occurred_at_unix_ms: u64,
        call_id: impl Into<String>,
        reason: IndeterminateReason,
    ) -> Self {
        Self::Indeterminate {
            sequence,
            occurred_at_unix_ms,
            call_id: call_id.into(),
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
            Self::Prepared { call, .. } => &call.id,
            Self::Dispatched { call_id, .. }
            | Self::Completed { call_id, .. }
            | Self::Indeterminate { call_id, .. } => call_id,
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
    #[error("tool call `{call_id}` was prepared more than once")]
    DuplicatePreparation { call_id: String },
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
        let dispatched = self
            .events
            .iter()
            .filter_map(|event| {
                let call_id = event.call_id();
                (event.status() == ToolExecutionStatus::Dispatched
                    && states.get(call_id) == Some(&ToolExecutionStatus::Dispatched))
                .then(|| call_id.to_string())
            })
            .collect::<Vec<_>>();

        for call_id in &dispatched {
            self.append(ToolExecutionEvent::indeterminate(
                self.next_sequence(),
                observed_at_unix_ms,
                call_id.clone(),
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

fn execution_states(
    events: &[ToolExecutionEvent],
) -> Result<BTreeMap<String, ToolExecutionStatus>, ToolExecutionTransitionError> {
    validate_execution_events(events)?;
    Ok(events
        .iter()
        .map(|event| (event.call_id().to_string(), event.status()))
        .collect())
}

fn validate_execution_events(
    events: &[ToolExecutionEvent],
) -> Result<(), ToolExecutionTransitionError> {
    let mut states = BTreeMap::<String, ToolExecutionStatus>::new();
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
        let previous = states.get(&call_id).copied();
        let next = event.status();
        match (previous, next) {
            (None, ToolExecutionStatus::Prepared) => {}
            (Some(ToolExecutionStatus::Prepared), ToolExecutionStatus::Dispatched) => {}
            (Some(ToolExecutionStatus::Dispatched), ToolExecutionStatus::Completed) => {}
            (Some(ToolExecutionStatus::Dispatched), ToolExecutionStatus::Indeterminate) => {}
            (Some(_), ToolExecutionStatus::Prepared) => {
                return Err(ToolExecutionTransitionError::DuplicatePreparation { call_id });
            }
            (from, to) => {
                return Err(ToolExecutionTransitionError::InvalidTransition { call_id, from, to });
            }
        }
        states.insert(call_id, next);
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

/// Validated data needed to create a quiescent snapshot.
#[derive(Clone, PartialEq)]
pub struct RunSnapshotParts {
    pub engine_version: SnapshotEngineVersion,
    pub run_id: RunId,
    pub lineage_id: LineageId,
    pub checkpoint_id: CheckpointId,
    pub parent_checkpoint_id: Option<CheckpointId>,
    pub target: ModelTarget,
    pub fingerprints: SnapshotFingerprints,
    pub history: Vec<Message>,
    pub pending_approvals: Vec<PendingApprovalSnapshot>,
    pub provider_state: Vec<ProviderStateSnapshot>,
    pub execution_log: ToolExecutionLog,
    pub budget: SnapshotBudgetLedger,
    pub usage: Usage,
    pub deadline_unix_ms: Option<u64>,
    pub state: SnapshotState,
}

/// A versioned, portable continuation captured only at a quiescent boundary.
///
/// Serialization is not an encryption boundary. Applications are responsible
/// for protecting snapshots that contain prompts or provider continuation data.
#[derive(Clone, PartialEq, Serialize)]
pub struct RunSnapshot {
    snapshot_version: u16,
    engine_version: SnapshotEngineVersion,
    run_id: RunId,
    lineage_id: LineageId,
    checkpoint_id: CheckpointId,
    parent_checkpoint_id: Option<CheckpointId>,
    target: ModelTarget,
    fingerprints: SnapshotFingerprints,
    history: Vec<Message>,
    pending_approvals: Vec<PendingApprovalSnapshot>,
    provider_state: Vec<ProviderStateSnapshot>,
    execution_log: ToolExecutionLog,
    budget: SnapshotBudgetLedger,
    usage: Usage,
    deadline_unix_ms: Option<u64>,
    state: SnapshotState,
}

impl RunSnapshot {
    pub fn new(parts: RunSnapshotParts) -> Result<Self, RunSnapshotError> {
        let snapshot = Self {
            snapshot_version: RUN_SNAPSHOT_SCHEMA_VERSION,
            engine_version: parts.engine_version,
            run_id: parts.run_id,
            lineage_id: parts.lineage_id,
            checkpoint_id: parts.checkpoint_id,
            parent_checkpoint_id: parts.parent_checkpoint_id,
            target: parts.target,
            fingerprints: parts.fingerprints,
            history: parts.history,
            pending_approvals: parts.pending_approvals,
            provider_state: parts.provider_state,
            execution_log: parts.execution_log,
            budget: parts.budget,
            usage: parts.usage,
            deadline_unix_ms: parts.deadline_unix_ms,
            state: parts.state,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn snapshot_version(&self) -> u16 {
        self.snapshot_version
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

    pub fn target(&self) -> &ModelTarget {
        &self.target
    }

    pub fn fingerprints(&self) -> &SnapshotFingerprints {
        &self.fingerprints
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub fn pending_approvals(&self) -> &[PendingApprovalSnapshot] {
        &self.pending_approvals
    }

    pub fn provider_state(&self) -> &[ProviderStateSnapshot] {
        &self.provider_state
    }

    pub fn execution_log(&self) -> &ToolExecutionLog {
        &self.execution_log
    }

    pub fn budget(&self) -> &SnapshotBudgetLedger {
        &self.budget
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn deadline_unix_ms(&self) -> Option<u64> {
        self.deadline_unix_ms
    }

    pub fn state(&self) -> &SnapshotState {
        &self.state
    }

    pub fn into_parts(self) -> RunSnapshotParts {
        RunSnapshotParts {
            engine_version: self.engine_version,
            run_id: self.run_id,
            lineage_id: self.lineage_id,
            checkpoint_id: self.checkpoint_id,
            parent_checkpoint_id: self.parent_checkpoint_id,
            target: self.target,
            fingerprints: self.fingerprints,
            history: self.history,
            pending_approvals: self.pending_approvals,
            provider_state: self.provider_state,
            execution_log: self.execution_log,
            budget: self.budget,
            usage: self.usage,
            deadline_unix_ms: self.deadline_unix_ms,
            state: self.state,
        }
    }

    /// Convert every execution left in `Dispatched` into `Indeterminate`.
    ///
    /// This operation never turns a completed execution back into dispatchable
    /// work and should be applied before resuming a deserialized continuation.
    pub fn recovered_for_resume(
        mut self,
        observed_at_unix_ms: u64,
    ) -> Result<Self, RunSnapshotError> {
        let recovered = self.execution_log.recover_dispatched(observed_at_unix_ms)?;
        if recovered > 0 {
            self.state = SnapshotState::Terminal(SnapshotTerminal::Indeterminate {
                reason: SnapshotReason::dispatch_outcome_unknown(),
            });
        }
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn is_compatible_successor(&self, successor: &Self) -> bool {
        self.run_id == successor.run_id
            && self.lineage_id == successor.lineage_id
            && self.engine_version == successor.engine_version
            && self.target == successor.target
            && self.fingerprints == successor.fingerprints
    }

    fn validate(&self) -> Result<(), RunSnapshotError> {
        if self.snapshot_version != RUN_SNAPSHOT_SCHEMA_VERSION {
            return Err(RunSnapshotError::UnsupportedVersion {
                found: self.snapshot_version,
                supported: RUN_SNAPSHOT_SCHEMA_VERSION,
            });
        }
        if self.parent_checkpoint_id.as_ref() == Some(&self.checkpoint_id) {
            return Err(RunSnapshotError::SelfParentCheckpoint);
        }
        self.budget.validate()?;
        validate_execution_events(self.execution_log.events())?;

        let mut approval_ids = BTreeSet::new();
        let mut approval_calls = BTreeSet::new();
        for approval in &self.pending_approvals {
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
            if self.execution_log.status(&approval.call.id) != Some(ToolExecutionStatus::Prepared) {
                return Err(RunSnapshotError::ApprovalCallNotPrepared {
                    call_id: approval.call.id.clone(),
                });
            }
        }

        let mut provider_namespaces = BTreeSet::new();
        for state in &self.provider_state {
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

        match &self.state {
            SnapshotState::Suspended(SnapshotSuspension::AwaitingApprovals)
                if self.pending_approvals.is_empty() =>
            {
                Err(RunSnapshotError::MissingPendingApproval)
            }
            SnapshotState::Suspended(SnapshotSuspension::AwaitingProvider)
                if self.provider_state.is_empty() =>
            {
                Err(RunSnapshotError::MissingProviderState)
            }
            _ => Ok(()),
        }
    }
}

impl fmt::Debug for RunSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunSnapshot")
            .field("snapshot_version", &self.snapshot_version)
            .field("engine_version", &self.engine_version)
            .field("run_id", &self.run_id)
            .field("lineage_id", &self.lineage_id)
            .field("checkpoint_id", &self.checkpoint_id)
            .field("parent_checkpoint_id", &self.parent_checkpoint_id)
            .field("target", &"<redacted>")
            .field("fingerprints", &"<redacted>")
            .field("history_messages", &self.history.len())
            .field("pending_approvals", &self.pending_approvals.len())
            .field("provider_state_entries", &self.provider_state.len())
            .field("execution_log", &self.execution_log)
            .field("budget", &self.budget)
            .field("usage", &"<redacted>")
            .field("deadline_unix_ms", &self.deadline_unix_ms)
            .field("state", &self.state)
            .finish()
    }
}

#[derive(Deserialize)]
struct RunSnapshotWire {
    snapshot_version: u16,
    engine_version: SnapshotEngineVersion,
    run_id: RunId,
    lineage_id: LineageId,
    checkpoint_id: CheckpointId,
    parent_checkpoint_id: Option<CheckpointId>,
    target: ModelTarget,
    fingerprints: SnapshotFingerprints,
    history: Vec<Message>,
    pending_approvals: Vec<PendingApprovalSnapshot>,
    provider_state: Vec<ProviderStateSnapshot>,
    execution_log: ToolExecutionLog,
    budget: SnapshotBudgetLedger,
    usage: Usage,
    deadline_unix_ms: Option<u64>,
    state: SnapshotState,
}

impl<'de> Deserialize<'de> for RunSnapshot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RunSnapshotWire::deserialize(deserializer)?;
        let snapshot = Self {
            snapshot_version: wire.snapshot_version,
            engine_version: wire.engine_version,
            run_id: wire.run_id,
            lineage_id: wire.lineage_id,
            checkpoint_id: wire.checkpoint_id,
            parent_checkpoint_id: wire.parent_checkpoint_id,
            target: wire.target,
            fingerprints: wire.fingerprints,
            history: wire.history,
            pending_approvals: wire.pending_approvals,
            provider_state: wire.provider_state,
            execution_log: wire.execution_log,
            budget: wire.budget,
            usage: wire.usage,
            deadline_unix_ms: wire.deadline_unix_ms,
            state: wire.state,
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
    #[error("budget `{dimension}` consumed {consumed}, exceeding limit {limit}")]
    BudgetExceeded {
        dimension: &'static str,
        consumed: u64,
        limit: u64,
    },
    #[error("pending approval identifiers must be unique")]
    DuplicateApprovalId,
    #[error("one tool call cannot have multiple pending approvals")]
    DuplicateApprovalCall,
    #[error("pending approval call `{call_id}` is not in Prepared state")]
    ApprovalCallNotPrepared { call_id: String },
    #[error("provider-state namespaces must be unique")]
    DuplicateProviderStateNamespace,
    #[error("AwaitingApprovals requires at least one pending approval")]
    MissingPendingApproval,
    #[error("AwaitingProvider requires at least one provider-state entry")]
    MissingProviderState,
    #[error(transparent)]
    InvalidExecutionTransition(#[from] ToolExecutionTransitionError),
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
