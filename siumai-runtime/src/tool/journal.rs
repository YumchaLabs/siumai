//! Runtime-owned durable local-tool lifecycle journal.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use siumai_core::ToolResult;
use thiserror::Error;

use crate::snapshot::{
    IndeterminateReason, PreparedToolSnapshot, ToolExecutionEvent, ToolExecutionLog,
    ToolExecutionTransitionError,
};

use super::{ApprovalPolicy, EffectCertainty, ToolExecutionRequest, ToolSet};

/// Failure to mutate or restore the durable local-tool journal.
#[derive(Debug, Error)]
pub(crate) enum ToolJournalError {
    #[error("system clock is before the Unix epoch")]
    Clock,
    #[error("tool ordinal cannot be represented in durable state")]
    Ordinal,
    #[error("restored frozen tool work no longer matches its binding")]
    FrozenWorkMismatch,
    #[error("tool recovery policy does not permit another attempt")]
    RecoveryNotPermitted,
    #[error("restored frozen tool request is invalid")]
    InvalidRestoredRequest,
    #[error(transparent)]
    Transition(#[from] ToolExecutionTransitionError),
}

/// The only runtime writer for durable local-tool execution state.
#[derive(Clone, PartialEq)]
pub(crate) struct ToolJournal {
    log: ToolExecutionLog,
}

impl Default for ToolJournal {
    fn default() -> Self {
        Self {
            log: ToolExecutionLog::new(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolJournalWire {
    events: ToolExecutionLog,
}

#[derive(Serialize)]
struct ToolJournalWireRef<'a> {
    events: &'a ToolExecutionLog,
}

impl Serialize for ToolJournal {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ToolJournalWireRef { events: &self.log }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ToolJournal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ToolJournalWire::deserialize(deserializer)?;
        Ok(Self { log: wire.events })
    }
}

impl std::fmt::Debug for ToolJournal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.log.fmt(formatter)
    }
}

impl ToolJournal {
    #[cfg(test)]
    pub(crate) fn from_log(log: ToolExecutionLog) -> Self {
        Self { log }
    }

    pub(crate) fn view(&self) -> &ToolExecutionLog {
        &self.log
    }

    pub(crate) fn prepare(
        &mut self,
        step: u32,
        ordinal: usize,
        request: &ToolExecutionRequest,
    ) -> Result<PreparedToolSnapshot, ToolJournalError> {
        let ordinal = u32::try_from(ordinal).map_err(|_| ToolJournalError::Ordinal)?;
        let prepared = PreparedToolSnapshot::new(
            ordinal,
            request.call().clone(),
            request.binding_identity().clone(),
            request.recovery_policy(),
            request.idempotency_key().cloned(),
            request.attempt(),
        );
        self.append(ToolExecutionEvent::prepared(
            self.log.next_sequence(),
            unix_millis()?,
            step,
            prepared.clone(),
        ))?;
        Ok(prepared)
    }

    pub(crate) fn dispatch(
        &mut self,
        request: &ToolExecutionRequest,
    ) -> Result<(), ToolJournalError> {
        self.append(ToolExecutionEvent::dispatched(
            self.log.next_sequence(),
            unix_millis()?,
            request.call_id(),
            request.attempt(),
        ))
    }

    pub(crate) fn complete(
        &mut self,
        request: &ToolExecutionRequest,
        result: &ToolResult,
    ) -> Result<(), ToolJournalError> {
        self.append(ToolExecutionEvent::completed(
            self.log.next_sequence(),
            unix_millis()?,
            &result.call_id,
            request.attempt(),
            result.outcome.clone(),
        ))
    }

    pub(crate) fn mark_indeterminate(
        &mut self,
        call_id: &str,
        attempt: super::ToolExecutionAttempt,
        reason: IndeterminateReason,
    ) -> Result<(), ToolJournalError> {
        self.append(ToolExecutionEvent::indeterminate(
            self.log.next_sequence(),
            unix_millis()?,
            call_id,
            attempt,
            reason,
        ))
    }

    pub(crate) fn recover_dispatched(&mut self) -> Result<usize, ToolJournalError> {
        self.log
            .recover_dispatched(unix_millis()?)
            .map_err(Into::into)
    }

    pub(crate) fn restore(
        &self,
        tools: &ToolSet,
        prepared: &PreparedToolSnapshot,
    ) -> Result<ToolExecutionRequest, ToolJournalError> {
        let mut request = tools
            .resolve_frozen(prepared.call().clone(), prepared.binding())
            .map_err(|_| ToolJournalError::FrozenWorkMismatch)?;
        while request.attempt() < prepared.attempt() {
            request = request
                .next_attempt(EffectCertainty::Indeterminate)
                .map_err(|_| ToolJournalError::RecoveryNotPermitted)?;
        }
        if request.attempt() != prepared.attempt()
            || request.recovery_policy() != prepared.recovery_policy()
            || request.idempotency_key() != prepared.stable_idempotency_key()
        {
            return Err(ToolJournalError::FrozenWorkMismatch);
        }
        request
            .validate()
            .map_err(|_| ToolJournalError::InvalidRestoredRequest)?;
        Ok(request)
    }

    pub(crate) fn retry(
        &mut self,
        tools: &ToolSet,
        step: u32,
        prepared: &PreparedToolSnapshot,
        certainty: EffectCertainty,
    ) -> Result<Option<(PreparedToolSnapshot, ToolExecutionRequest)>, ToolJournalError> {
        let request = self.restore(tools, prepared)?;
        if request.approval_policy() == ApprovalPolicy::Required
            || !request.permits_retry(certainty)
        {
            return Ok(None);
        }
        let next = request
            .next_attempt(certainty)
            .map_err(|_| ToolJournalError::RecoveryNotPermitted)?;
        let prepared = self.prepare(step, prepared.ordinal() as usize, &next)?;
        Ok(Some((prepared, next)))
    }

    pub(crate) fn has_prefix(&self, prefix: &Self) -> bool {
        self.log.has_prefix(&prefix.log)
    }

    fn append(&mut self, event: ToolExecutionEvent) -> Result<(), ToolJournalError> {
        self.log.append(event).map_err(Into::into)
    }
}

fn unix_millis() -> Result<u64, ToolJournalError> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ToolJournalError::Clock)?
        .as_millis();
    u64::try_from(millis).map_err(|_| ToolJournalError::Clock)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{ToolCall, ToolOutcome, ToolSpec};

    use super::*;
    use crate::tool::{ApprovalPolicy, ToolBinding, ToolEffect};

    fn request() -> ToolExecutionRequest {
        let binding = ToolBinding::from_fn(
            ToolSpec::new("lookup", None, json!({ "type": "object" })).unwrap(),
            "v1",
            |_| Ok(()),
            |_| async { Ok(ToolOutcome::Success { value: json!(true) }) },
        )
        .unwrap()
        .with_effect(ToolEffect::ReadOnly)
        .with_approval_policy(ApprovalPolicy::NotRequired);
        ToolSet::from_bindings([binding])
            .unwrap()
            .resolve(ToolCall::local("call-1", "lookup", json!({})).unwrap())
            .unwrap()
    }

    #[test]
    fn journal_owns_sequence_and_rejects_success_before_dispatch() {
        let request = request();
        let mut journal = ToolJournal::default();
        journal.prepare(0, 0, &request).unwrap();
        let result = ToolResult {
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
            outcome: ToolOutcome::Success { value: json!(true) },
        };
        assert!(matches!(
            journal.complete(&request, &result),
            Err(ToolJournalError::Transition(
                ToolExecutionTransitionError::DirectSuccessRequiresDispatch { .. }
            ))
        ));
        assert_eq!(journal.view().events().len(), 1);
        assert_eq!(journal.view().events()[0].sequence(), 0);
    }

    #[test]
    fn dispatch_then_complete_is_the_only_success_path() {
        let request = request();
        let mut journal = ToolJournal::default();
        journal.prepare(0, 0, &request).unwrap();
        journal.dispatch(&request).unwrap();
        journal
            .complete(
                &request,
                &ToolResult {
                    call_id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    outcome: ToolOutcome::Success { value: json!(true) },
                },
            )
            .unwrap();
        assert_eq!(journal.view().events().len(), 3);
    }
}
