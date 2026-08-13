//! Explicit multi-step language and local-tool execution.

use std::fmt;
use std::sync::Arc;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use siumai_core::{CallOptions, Error, LanguageModel, LanguageRequest, ToolOutcome};

use crate::engine::{StepEngine, ToolHandling};
use crate::run::established_run_stream;
use crate::tool::{ApprovalDecider, ExternalApprovalDecider, ToolSet};
use crate::{ProjectionPolicy, RunStream, RunTerminal, Runtime, StepModelSelector, StepOptions};

/// Runtime action after a known non-success tool outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolOutcomeAction {
    Continue,
    Stop,
}

/// Explicit continuation policy for known tool outcomes.
///
/// Success always continues. Denial, execution failure, and cancellation stop
/// by default so a host must opt in before feeding those outcomes to another
/// model step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutcomePolicy {
    denied: ToolOutcomeAction,
    execution_failed: ToolOutcomeAction,
    cancelled: ToolOutcomeAction,
}

impl ToolOutcomePolicy {
    pub fn with_denied(mut self, action: ToolOutcomeAction) -> Self {
        self.denied = action;
        self
    }

    pub fn with_execution_failed(mut self, action: ToolOutcomeAction) -> Self {
        self.execution_failed = action;
        self
    }

    pub fn with_cancelled(mut self, action: ToolOutcomeAction) -> Self {
        self.cancelled = action;
        self
    }

    pub fn action(&self, outcome: &ToolOutcome) -> ToolOutcomeAction {
        match outcome {
            ToolOutcome::Success { .. } => ToolOutcomeAction::Continue,
            ToolOutcome::Denied { .. } => self.denied,
            ToolOutcome::ExecutionFailed { .. } => self.execution_failed,
            ToolOutcome::Cancelled { .. } => self.cancelled,
            _ => ToolOutcomeAction::Stop,
        }
    }
}

impl Default for ToolOutcomePolicy {
    fn default() -> Self {
        Self {
            denied: ToolOutcomeAction::Stop,
            execution_failed: ToolOutcomeAction::Stop,
            cancelled: ToolOutcomeAction::Stop,
        }
    }
}

/// Clone-cheap configuration for one explicit tool loop.
#[derive(Clone)]
pub struct ToolLoop {
    runtime: Runtime,
    model: Arc<dyn LanguageModel>,
    tools: ToolSet,
    step_options: StepOptions,
    outcome_policy: ToolOutcomePolicy,
    approval_decider: Arc<dyn ApprovalDecider>,
    model_selector: Option<Arc<dyn StepModelSelector>>,
    projection_policy: ProjectionPolicy,
}

impl fmt::Debug for ToolLoop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolLoop")
            .field(
                "target",
                &crate::ModelTarget::from_model(self.model.as_ref()),
            )
            .field("tools", &self.tools.len())
            .field("outcome_policy", &self.outcome_policy)
            .field("model_selector", &self.model_selector.is_some())
            .field("projection_policy", &self.projection_policy)
            .field(
                "approval_policy_fingerprint",
                self.approval_decider.fingerprint(),
            )
            .finish_non_exhaustive()
    }
}

impl ToolLoop {
    pub fn new(model: Arc<dyn LanguageModel>, tools: ToolSet) -> Self {
        Self {
            runtime: Runtime::default(),
            model,
            tools,
            step_options: StepOptions::default(),
            outcome_policy: ToolOutcomePolicy::default(),
            approval_decider: Arc::new(ExternalApprovalDecider::default()),
            model_selector: None,
            projection_policy: ProjectionPolicy::Strict,
        }
    }

    pub fn with_runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = runtime;
        self
    }

    pub fn with_step_options(mut self, options: StepOptions) -> Self {
        self.step_options = options;
        self
    }

    pub fn with_tools(mut self, tools: ToolSet) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_outcome_policy(mut self, policy: ToolOutcomePolicy) -> Self {
        self.outcome_policy = policy;
        self
    }

    /// Install the trusted host policy for bindings marked as requiring approval.
    pub fn with_approval_decider(mut self, decider: Arc<dyn ApprovalDecider>) -> Self {
        self.approval_decider = decider;
        self
    }

    /// Select the model used by each already-required later step.
    pub fn with_model_selector<S>(mut self, selector: S) -> Self
    where
        S: StepModelSelector,
    {
        self.model_selector = Some(Arc::new(selector));
        self
    }

    /// Install a shared selector without adding another allocation layer.
    pub fn with_shared_model_selector(mut self, selector: Arc<dyn StepModelSelector>) -> Self {
        self.model_selector = Some(selector);
        self
    }

    /// Set the loss policy for actual model-target transitions.
    ///
    /// The default is [`ProjectionPolicy::Strict`]. Blocking state is rejected
    /// under every policy.
    pub fn with_projection_policy(mut self, policy: ProjectionPolicy) -> Self {
        self.projection_policy = policy;
        self
    }

    /// Establish the first model stream and return the shared run projection.
    ///
    /// Validation, request shaping, and first-step handshake failures remain
    /// outer errors. Every failure after establishment becomes the run's one
    /// terminal event.
    pub async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<RunStream, Error> {
        let engine = StepEngine::establish(
            self.runtime.clone(),
            Arc::clone(&self.model),
            self.tools.clone(),
            request,
            self.step_options.clone(),
            options,
            self.outcome_policy,
            Arc::clone(&self.approval_decider),
            self.model_selector.clone(),
            self.projection_policy,
            ToolHandling::Execute,
        )
        .await?;
        let cancellation = engine.cancellation().clone();
        let initial_report = engine.report().clone();
        Ok(established_run_stream(
            cancellation,
            engine.into_stream(),
            initial_report,
        ))
    }

    /// Collect [`ToolLoop::stream`] without running a second execution path.
    pub async fn run(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<RunTerminal, Error> {
        let mut stream = self.stream(request, options).await?;
        while let Some(event) = stream.next().await {
            if let crate::RunEvent::Terminal(terminal) = event {
                return Ok(terminal);
            }
        }
        Err(Error::unexpected_eof())
    }
}

impl Runtime {
    /// Configure an explicit local-tool loop on this runtime.
    pub fn tool_loop(&self, model: Arc<dyn LanguageModel>, tools: ToolSet) -> ToolLoop {
        ToolLoop::new(model, tools).with_runtime(self.clone())
    }
}
