//! Orchestrator Builder
//!
//! A convenience builder for configuring stop conditions and options,
//! then running multi‑step orchestration in a single call.
//!
//! This wraps `generate`, `generate_stream` and `generate_stream_owned` with a
//! chainable API. It does not hold a reference to the model; pass the model
//! when calling `run`/`run_stream`.

use std::sync::Arc;

use siumai::error::LlmError;
use siumai::streaming::ChatStreamEvent;
use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, Tool};

use super::prepare_step::{PrepareStepFn, PrepareStepResult, ToolChoice};
use super::stop_condition::StopCondition;
use super::types::{OrchestratorOptions, OrchestratorStreamOptions, StepResult};

/// Builder for configuring and running the orchestrator.
#[derive(Default)]
pub struct OrchestratorBuilder {
    stop_conditions: Vec<Box<dyn StopCondition>>,
    options: OrchestratorOptions,
    stream_options: OrchestratorStreamOptions,
}

// Default is derived

impl OrchestratorBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum steps (applies to both non‑stream and stream variants).
    pub fn max_steps(mut self, max: usize) -> Self {
        self.options.max_steps = max;
        self.stream_options.max_steps = max;
        self
    }

    /// Add a stop condition.
    pub fn stop_when(mut self, cond: Box<dyn StopCondition>) -> Self {
        self.stop_conditions.push(cond);
        self
    }

    /// Add multiple stop conditions.
    pub fn stop_when_all(mut self, conds: Vec<Box<dyn StopCondition>>) -> Self {
        self.stop_conditions.extend(conds);
        self
    }

    /// On each step finish callback (non‑stream).
    pub fn on_step_finish<F>(mut self, cb: F) -> Self
    where
        F: Fn(&StepResult) + Send + Sync + 'static,
    {
        self.options.on_step_finish = Some(Arc::new(cb));
        self
    }

    /// Final finish callback with all steps (non‑stream).
    pub fn on_finish<F>(mut self, cb: F) -> Self
    where
        F: Fn(&[StepResult]) + Send + Sync + 'static,
    {
        self.options.on_finish = Some(Arc::new(cb));
        self
    }

    /// Tool approval callback (applies to both variants).
    #[allow(clippy::type_complexity)]
    pub fn on_tool_approval<F>(mut self, cb: F) -> Self
    where
        F: Fn(&str, &serde_json::Value) -> super::types::ToolApproval + Send + Sync + 'static,
    {
        let arc: Arc<dyn Fn(&str, &serde_json::Value) -> super::types::ToolApproval + Send + Sync> =
            Arc::new(cb);
        self.options.on_tool_approval = Some(arc.clone());
        self.stream_options.on_tool_approval = Some(arc);
        self
    }

    /// Preliminary tool result callback (applies to both variants).
    #[allow(clippy::type_complexity)]
    pub fn on_preliminary_tool_result<F>(mut self, cb: F) -> Self
    where
        F: Fn(&str, &str, &serde_json::Value) + Send + Sync + 'static,
    {
        let arc: Arc<dyn Fn(&str, &str, &serde_json::Value) + Send + Sync> = Arc::new(cb);
        self.options.on_preliminary_tool_result = Some(arc.clone());
        self.stream_options.on_preliminary_tool_result = Some(arc);
        self
    }

    /// Prepare‑step callback for dynamic tool/message selection (non‑stream).
    pub fn prepare_step(mut self, f: PrepareStepFn) -> Self {
        self.options.prepare_step = Some(f);
        self
    }

    /// Set a default tool choice strategy for all steps (non‑stream).
    ///
    /// This is syntactic sugar over `prepare_step` that:
    /// - Applies the given `ToolChoice` when no per‑step override is set
    /// - Composes with an existing `prepare_step` callback if present
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        let existing: Option<PrepareStepFn> = self.options.prepare_step.take();
        let choice_clone = choice.clone();
        let prepare: PrepareStepFn = Arc::new(move |ctx| {
            let mut result = if let Some(ref f) = existing {
                f(ctx)
            } else {
                PrepareStepResult::default()
            };
            if result.tool_choice.is_none() {
                result.tool_choice = Some(choice_clone.clone());
            }
            result
        });
        self.options.prepare_step = Some(prepare);
        self
    }

    /// Restrict the active tools for all steps (non‑stream).
    ///
    /// This is syntactic sugar over `prepare_step` that:
    /// - Applies the given `active_tools` when no per‑step override is set
    /// - Composes with an existing `prepare_step` callback if present
    pub fn active_tools(mut self, tools: Vec<String>) -> Self {
        let existing: Option<PrepareStepFn> = self.options.prepare_step.take();
        let tools_clone = tools.clone();
        let prepare: PrepareStepFn = Arc::new(move |ctx| {
            let mut result = if let Some(ref f) = existing {
                f(ctx)
            } else {
                PrepareStepResult::default()
            };
            if result.active_tools.is_none() {
                result.active_tools = Some(tools_clone.clone());
            }
            result
        });
        self.options.prepare_step = Some(prepare);
        self
    }

    /// Set telemetry configuration (applies to both variants).
    pub fn telemetry(mut self, cfg: siumai::telemetry::TelemetryConfig) -> Self {
        let some = Some(cfg);
        self.stream_options.telemetry = some.clone();
        self.options.telemetry = some;
        self
    }

    /// Set agent‑level CommonParams (applies to both variants).
    pub fn common_params(mut self, params: siumai::types::CommonParams) -> Self {
        let some = Some(params);
        self.stream_options.common_params = some.clone();
        self.options.common_params = some;
        self
    }

    /// Streaming on_chunk callback (stream variant only).
    pub fn on_chunk<F>(mut self, cb: F) -> Self
    where
        F: Fn(&ChatStreamEvent) + Send + Sync + 'static,
    {
        self.stream_options.on_chunk = Some(Arc::new(cb));
        self
    }

    /// Streaming abort callback (stream variant only).
    pub fn on_abort<F>(mut self, cb: F) -> Self
    where
        F: Fn(&[StepResult]) + Send + Sync + 'static,
    {
        self.stream_options.on_abort = Some(Arc::new(cb));
        self
    }

    /// Run non‑stream orchestration.
    pub async fn run(
        &self,
        model: &impl ChatCapability,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        resolver: Option<&dyn super::types::ToolResolver>,
    ) -> Result<(siumai::types::ChatResponse, Vec<StepResult>), LlmError> {
        // Convert owned Box<dyn StopCondition> into a temporary Vec<&dyn StopCondition>
        let mut refs: Vec<&dyn StopCondition> = Vec::with_capacity(self.stop_conditions.len());
        for c in &self.stop_conditions {
            refs.push(&**c);
        }
        super::generate(
            model,
            messages,
            tools,
            resolver,
            &refs,
            self.options.clone(),
        )
        .await
    }

    /// Run streaming orchestration (first step streaming, then follow‑ups).
    pub async fn run_stream(
        &self,
        model: &impl ChatCapability,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<super::stream::StreamOrchestration, LlmError> {
        // Basic streaming path does not execute tools (matches current impl)
        super::generate_stream(model, messages, tools, None, self.stream_options.clone()).await
    }

    /// Run streaming orchestration (owned model + resolver, with tool execution).
    pub async fn run_stream_owned<M, R>(
        &self,
        model: M,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        resolver: Option<R>,
    ) -> Result<super::stream::StreamOrchestration, LlmError>
    where
        M: ChatCapability + Send + Sync + 'static,
        R: super::types::ToolResolver + Send + Sync + 'static,
    {
        super::generate_stream_owned(
            model,
            messages,
            tools,
            resolver,
            self.stream_options.clone(),
        )
        .await
    }
}
