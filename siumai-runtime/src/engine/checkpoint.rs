use std::future::{Future, ready};
use std::pin::Pin;

use siumai_core::{LanguageRequest, LanguageResponse, ToolBindingIdentity, ToolCall};

use crate::snapshot::{
    CompletedToolSnapshot, PendingProviderStepSnapshot, PreparedToolSnapshot, SnapshotTerminal,
};
use crate::{ModelTarget, RunReport};

/// Why the engine requires a state commit before it may continue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CheckpointBoundary {
    Quiescent,
    BeforeDispatch,
    AfterDispatch,
}

/// Whether the engine may cross the committed boundary in this invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CheckpointControl {
    Continue,
    Pause,
}

/// Frozen approval identity retained without carrying executable authority.
#[derive(Clone)]
pub(crate) struct PendingApprovalCheckpoint {
    call: ToolCall,
    binding: ToolBindingIdentity,
    canonical_arguments_digest: String,
}

impl PendingApprovalCheckpoint {
    pub(crate) fn new(
        call: ToolCall,
        binding: ToolBindingIdentity,
        canonical_arguments_digest: String,
    ) -> Self {
        Self {
            call,
            binding,
            canonical_arguments_digest,
        }
    }

    pub(crate) fn call(&self) -> &ToolCall {
        &self.call
    }

    pub(crate) fn binding(&self) -> &ToolBindingIdentity {
        &self.binding
    }

    pub(crate) fn canonical_arguments_digest(&self) -> &str {
        &self.canonical_arguments_digest
    }
}

/// Canonical pending-tool state before durable encoding.
pub(crate) struct PendingToolsCheckpoint {
    index: u32,
    target: ModelTarget,
    response: LanguageResponse,
    prepared: Vec<PreparedToolSnapshot>,
    completed: Vec<CompletedToolSnapshot>,
    pending_approvals: Vec<PendingApprovalCheckpoint>,
    can_progress: bool,
}

impl PendingToolsCheckpoint {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        index: u32,
        target: ModelTarget,
        response: LanguageResponse,
        prepared: Vec<PreparedToolSnapshot>,
        completed: Vec<CompletedToolSnapshot>,
        pending_approvals: Vec<PendingApprovalCheckpoint>,
        can_progress: bool,
    ) -> Self {
        Self {
            index,
            target,
            response,
            prepared,
            completed,
            pending_approvals,
            can_progress,
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        u32,
        ModelTarget,
        LanguageResponse,
        Vec<PreparedToolSnapshot>,
        Vec<CompletedToolSnapshot>,
        Vec<PendingApprovalCheckpoint>,
        bool,
    ) {
        (
            self.index,
            self.target,
            self.response,
            self.prepared,
            self.completed,
            self.pending_approvals,
            self.can_progress,
        )
    }
}

/// Provider-neutral continuation state at a checkpoint boundary.
pub(crate) enum EngineCheckpointState {
    PendingTools(PendingToolsCheckpoint),
    AwaitingProvider(PendingProviderStepSnapshot),
    ReadyForModel { next_step: u32, target: ModelTarget },
    Terminal(SnapshotTerminal),
}

/// Provider-neutral candidate state committed at the runtime persistence seam.
pub(crate) struct EngineCheckpoint {
    boundary: CheckpointBoundary,
    continuation: LanguageRequest,
    report: RunReport,
    state: EngineCheckpointState,
}

impl EngineCheckpoint {
    pub(crate) fn new(
        boundary: CheckpointBoundary,
        continuation: LanguageRequest,
        report: RunReport,
        state: EngineCheckpointState,
    ) -> Self {
        Self {
            boundary,
            continuation,
            report,
            state,
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        CheckpointBoundary,
        LanguageRequest,
        RunReport,
        EngineCheckpointState,
    ) {
        (self.boundary, self.continuation, self.report, self.state)
    }
}

pub(crate) type EngineCheckpointFuture<'a, E> =
    Pin<Box<dyn Future<Output = Result<CheckpointControl, E>> + Send + 'a>>;

/// Internal seam between runtime semantics and state persistence.
pub(crate) trait EngineCheckpointPort: Send {
    type Error: Send + 'static;

    const ENABLED: bool;

    fn max_concurrent_tools(&self, configured: usize) -> usize {
        configured
    }

    fn commit<'a>(
        &'a mut self,
        checkpoint: EngineCheckpoint,
    ) -> EngineCheckpointFuture<'a, Self::Error>;
}

/// Immediate-ACK adapter used by ordinary in-process runs.
pub(crate) struct EphemeralCheckpointPort;

impl EngineCheckpointPort for EphemeralCheckpointPort {
    type Error = std::convert::Infallible;

    const ENABLED: bool = false;

    fn commit<'a>(
        &'a mut self,
        checkpoint: EngineCheckpoint,
    ) -> EngineCheckpointFuture<'a, Self::Error> {
        let _ = checkpoint.into_parts();
        Box::pin(ready(Ok(CheckpointControl::Continue)))
    }
}
