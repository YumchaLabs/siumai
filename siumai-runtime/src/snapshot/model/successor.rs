use siumai_core::{LanguageRequest, Usage, UsageValue};
use thiserror::Error;

use super::{
    CheckpointId, PendingStepSnapshot, ResumePoint, ResumePointKind, RunSnapshot, RunSnapshotError,
};
use crate::{ModelTransitionOutcome, project_history};

/// Why a candidate checkpoint cannot follow the currently stored checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RunSnapshotSuccessorError {
    #[error(transparent)]
    InvalidSnapshot(#[from] RunSnapshotError),
    #[error("successor changes the lineage identifier")]
    LineageIdChanged,
    #[error("successor changes the durable execution ABI")]
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
    #[error("successor removes, reorders, or changes provider-native observation identity")]
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
    #[error("successor regresses settled usage state")]
    UsageSettlementRegression,
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

impl RunSnapshot {
    pub(crate) fn validate_successor(
        &self,
        successor: &Self,
    ) -> Result<(), RunSnapshotSuccessorError> {
        if self.resume_point().is_terminal() {
            return Ok(());
        }
        successor.validate()?;
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
        if !provider_deferred_keys_advance(
            self.report.provider_deferred(),
            successor.report.provider_deferred(),
        ) {
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
        validate_usage_successor(
            self.report.usage(),
            self.report.usage_is_settled(),
            successor.report.usage(),
            successor.report.usage_is_settled(),
        )?;
        validate_deadline_successor(self.deadline_unix_ms, successor.deadline_unix_ms)?;
        validate_resume_successor(&self.resume_point, &successor.resume_point)
    }
}

fn provider_deferred_keys_advance(
    previous: &[crate::ProviderDeferredObservation],
    successor: &[crate::ProviderDeferredObservation],
) -> bool {
    successor.len() >= previous.len()
        && previous
            .iter()
            .zip(successor)
            .all(|(previous, successor)| previous.has_same_key(successor))
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
    previous_settled: bool,
    next: &Usage,
    next_settled: bool,
) -> Result<(), RunSnapshotSuccessorError> {
    if previous_settled && !next_settled {
        return Err(RunSnapshotSuccessorError::UsageSettlementRegression);
    }
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

pub(super) fn validate_pending_step_successor(
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
