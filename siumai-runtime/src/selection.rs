//! Synchronous policy for selecting the model used by a later run step.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use siumai_core::{Error, LanguageModel, LanguageRequest};

use crate::snapshot::SnapshotFingerprint;
use crate::{
    ModelTarget, ModelTransitionOutcome, ModelTransitionRecord, ProjectionPolicy, RunReport,
    StepRecord, project_history,
};

/// Stable identity for a model-selection policy.
///
/// Durable runtimes persist this value and reject resumes under a different
/// policy implementation or configuration.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepModelSelectorIdentity {
    version: u16,
    fingerprint: SnapshotFingerprint,
}

impl StepModelSelectorIdentity {
    /// Construct a selector identity from a host-owned semantic version and
    /// stable configuration fingerprint.
    pub const fn new(version: u16, fingerprint: SnapshotFingerprint) -> Self {
        Self {
            version,
            fingerprint,
        }
    }

    /// Semantic version of the selection algorithm and its input contract.
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Stable digest of the selector configuration and model catalog.
    pub const fn fingerprint(&self) -> &SnapshotFingerprint {
        &self.fingerprint
    }
}

impl std::fmt::Debug for StepModelSelectorIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StepModelSelectorIdentity")
            .field("version", &self.version)
            .field("fingerprint", &"<redacted>")
            .finish()
    }
}

/// Read-only context supplied when a completed step requires another model call.
#[derive(Clone, Copy)]
pub struct StepModelContext<'a> {
    next_step: u32,
    current_target: &'a ModelTarget,
    previous_step: &'a StepRecord,
    report: &'a RunReport,
}

impl<'a> StepModelContext<'a> {
    pub(crate) fn new(
        next_step: u32,
        current_target: &'a ModelTarget,
        previous_step: &'a StepRecord,
        report: &'a RunReport,
    ) -> Self {
        Self {
            next_step,
            current_target,
            previous_step,
            report,
        }
    }

    /// Index of the model step about to be established.
    pub const fn next_step(&self) -> u32 {
        self.next_step
    }

    /// Target that produced the previous completed step.
    pub const fn current_target(&self) -> &'a ModelTarget {
        self.current_target
    }

    /// Previous completed model step.
    pub const fn previous_step(&self) -> &'a StepRecord {
        self.previous_step
    }

    /// Complete run report before the next step is selected.
    pub const fn report(&self) -> &'a RunReport {
        self.report
    }
}

impl std::fmt::Debug for StepModelContext<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StepModelContext")
            .field("next_step", &self.next_step)
            .field("current_target", &self.current_target)
            .field("previous_step", &self.previous_step.index())
            .finish_non_exhaustive()
    }
}

/// Selects a fully configured model handle for an already-required next step.
///
/// Selection is synchronous and must not perform I/O. The runtime derives the
/// destination [`ModelTarget`] from the returned model and owns all history
/// projection, option precedence, budgeting, and streaming behavior.
pub trait StepModelSelector: Send + Sync + 'static {
    /// Stable identity used to bind durable continuations to this policy.
    fn identity(&self) -> &StepModelSelectorIdentity;

    fn select(&self, context: StepModelContext<'_>) -> Result<Arc<dyn LanguageModel>, Error>;
}

/// Versioned adapter for ergonomic closure-based model selection.
pub struct VersionedStepModelSelector<F> {
    identity: StepModelSelectorIdentity,
    select: F,
}

impl<F> VersionedStepModelSelector<F> {
    pub fn new(identity: StepModelSelectorIdentity, select: F) -> Self {
        Self { identity, select }
    }
}

impl<F> std::fmt::Debug for VersionedStepModelSelector<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VersionedStepModelSelector")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl<F> StepModelSelector for VersionedStepModelSelector<F>
where
    F: for<'a> Fn(StepModelContext<'a>) -> Result<Arc<dyn LanguageModel>, Error>
        + Send
        + Sync
        + 'static,
{
    fn identity(&self) -> &StepModelSelectorIdentity {
        &self.identity
    }

    fn select(&self, context: StepModelContext<'_>) -> Result<Arc<dyn LanguageModel>, Error> {
        (self.select)(context)
    }
}

pub(crate) struct SelectedStepModel {
    pub(crate) model: Arc<dyn LanguageModel>,
    pub(crate) target: ModelTarget,
}

pub(crate) enum PreparedStepModel {
    Ready(Box<PreparedStepModelReady>),
    Rejected {
        transition: Box<ModelTransitionRecord>,
    },
}

pub(crate) struct PreparedStepModelReady {
    pub(crate) model: Arc<dyn LanguageModel>,
    pub(crate) target: ModelTarget,
    pub(crate) request: LanguageRequest,
    pub(crate) transition: Option<ModelTransitionRecord>,
}

pub(crate) fn select_step_model(
    selector: Option<&dyn StepModelSelector>,
    fallback: Arc<dyn LanguageModel>,
    next_step: u32,
    current_target: &ModelTarget,
    previous_step: &StepRecord,
    report: &RunReport,
) -> Result<SelectedStepModel, Error> {
    let model = match selector {
        Some(selector) => selector.select(StepModelContext::new(
            next_step,
            current_target,
            previous_step,
            report,
        ))?,
        None => fallback,
    };
    let target = ModelTarget::from_model(model.as_ref());
    Ok(SelectedStepModel { model, target })
}

pub(crate) fn prepare_selected_step_model(
    selected: SelectedStepModel,
    request: LanguageRequest,
    step: u32,
    source: &ModelTarget,
    policy: ProjectionPolicy,
) -> PreparedStepModel {
    if selected.target == *source {
        return PreparedStepModel::Ready(Box::new(PreparedStepModelReady {
            model: selected.model,
            target: selected.target,
            request,
            transition: None,
        }));
    }

    match project_history(request, source, &selected.target, policy) {
        Ok(projected) => {
            let (request, scope, losses) = projected.into_parts();
            let transition = ModelTransitionRecord::new(
                step,
                source.clone(),
                selected.target.clone(),
                policy,
                scope,
                ModelTransitionOutcome::Applied,
                losses,
            );
            PreparedStepModel::Ready(Box::new(PreparedStepModelReady {
                model: selected.model,
                target: selected.target,
                request,
                transition: Some(transition),
            }))
        }
        Err(error) => PreparedStepModel::Rejected {
            transition: Box::new(ModelTransitionRecord::new(
                step,
                source.clone(),
                selected.target,
                error.policy(),
                error.scope(),
                ModelTransitionOutcome::Rejected,
                error.into_losses(),
            )),
        },
    }
}
