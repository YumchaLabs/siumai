//! Workflow abstractions built on top of the orchestrator and agents.
//!
//! This module provides a lightweight workflow API that composes:
//! - The high-level `Orchestrator<M>` facade
//! - Multiple `ToolLoopAgent<M>` workers
//! - Unified structured output decoding configuration (`OutputDecodeConfig`)
//!
//! The design is inspired by Vercel AI SDK's orchestrator + workers model:
//! a primary orchestrator coordinates specialized workers that can each
//! focus on a subtask (planning, coding, retrieval, etc.).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use siumai::error::LlmError;
use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, ChatResponse, Tool};

use super::{PrepareStepFn, StepResult, ToolApproval, ToolChoice, ToolLoopAgent, ToolResolver};
use crate::structured_output::{OutputDecodeConfig, decode_typed};

use super::{Orchestrator, OrchestratorBuilder};

/// Default worker ID for planner agents.
pub const WORKER_PLANNER: &str = "planner";
/// Default worker ID for coding / implementation agents.
pub const WORKER_CODER: &str = "coder";
/// Default worker ID for research / retrieval agents.
pub const WORKER_RESEARCHER: &str = "researcher";

/// Per-worker wrapper that ties an agent to a worker identifier.
pub struct Worker<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    /// Worker identifier (e.g. "planner", "coder").
    pub id: String,
    /// Underlying agent for this worker.
    pub agent: ToolLoopAgent<M>,
}

impl<M> Worker<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    /// Create a new worker from an ID and a `ToolLoopAgent`.
    pub fn new(id: impl Into<String>, agent: ToolLoopAgent<M>) -> Self {
        Self {
            id: id.into(),
            agent,
        }
    }
}

/// Workflow state with per-worker outputs and captured steps.
#[derive(Debug, Default, Clone)]
pub struct WorkflowState {
    /// Structured outputs produced by workers keyed by worker ID.
    pub worker_outputs: HashMap<String, Value>,
    /// Aggregated steps taken inside each worker across invocations.
    ///
    /// Steps are appended on each worker invocation to provide a simple
    /// per-worker "memory" of tool calls and model interactions.
    pub worker_steps: HashMap<String, Vec<StepResult>>,
    /// Steps taken by the top-level orchestrator.
    ///
    /// This mirrors the `steps` returned from `Orchestrator::run` and is
    /// stored here so downstream consumers (memory, logging, analytics) can
    /// access all orchestration traces via WorkflowState.
    pub orchestration_steps: Vec<StepResult>,
    /// Arbitrary workflow-level metadata (user id, session id, trace ids, etc.).
    ///
    /// This can be populated by callers or higher-level frameworks that wrap
    /// the Workflow API.
    pub metadata: Map<String, Value>,
}

/// Pluggable workflow memory interface.
///
/// This allows workflows to persist and reload their state across runs
/// (e.g. to implement long-term memory, logging, or analytics).
///
/// Implementations can store state in-memory, on disk, or in external
/// services like Redis / databases.
#[async_trait]
pub trait WorkflowMemory: Send + Sync {
    /// Load previously stored workflow state for the given key.
    async fn load(&self, key: &str) -> Result<Option<WorkflowState>, LlmError>;

    /// Persist workflow state for the given key.
    async fn save(&self, key: &str, state: &WorkflowState) -> Result<(), LlmError>;
}

/// Simple in-memory implementation of `WorkflowMemory`.
///
/// This is primarily useful for demos, tests, or single-process applications.
#[derive(Default)]
pub struct InMemoryWorkflowMemory {
    inner: Mutex<HashMap<String, WorkflowState>>,
}

impl InMemoryWorkflowMemory {
    /// Create a new in-memory workflow memory store.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl WorkflowMemory for InMemoryWorkflowMemory {
    async fn load(&self, key: &str) -> Result<Option<WorkflowState>, LlmError> {
        let map = self.inner.lock().map_err(|e| {
            LlmError::InternalError(format!("Failed to lock in-memory workflow memory: {e}"))
        })?;
        Ok(map.get(key).cloned())
    }

    async fn save(&self, key: &str, state: &WorkflowState) -> Result<(), LlmError> {
        let mut map = self.inner.lock().map_err(|e| {
            LlmError::InternalError(format!("Failed to lock in-memory workflow memory: {e}"))
        })?;
        map.insert(key.to_string(), state.clone());
        Ok(())
    }
}

/// Builder for constructing a `Workflow<M>` from a model, tools and workers.
///
/// This mirrors Vercel AI SDK's workflow helpers: start from a model and tools,
/// configure orchestrator options, then register workers that can be invoked
/// via `worker:<id>` tools.
pub struct WorkflowBuilder<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    model: M,
    tools: Vec<Tool>,
    builder: OrchestratorBuilder,
    workers: HashMap<String, Worker<M>>,
    memory: Option<Arc<dyn WorkflowMemory>>,
}

impl<M> WorkflowBuilder<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    /// Create a new workflow builder from a model and tools.
    pub fn new(model: M, tools: Vec<Tool>) -> Self {
        Self {
            model,
            tools,
            builder: OrchestratorBuilder::new(),
            workers: HashMap::new(),
            memory: None,
        }
    }

    /// Set maximum steps for the underlying orchestrator.
    pub fn max_steps(mut self, max: usize) -> Self {
        self.builder = self.builder.max_steps(max);
        self
    }

    /// Add a stop condition for the orchestrator.
    pub fn stop_when(mut self, cond: Box<dyn super::StopCondition>) -> Self {
        self.builder = self.builder.stop_when(cond);
        self
    }

    /// Add multiple stop conditions for the orchestrator.
    pub fn stop_when_all(mut self, conds: Vec<Box<dyn super::StopCondition>>) -> Self {
        self.builder = self.builder.stop_when_all(conds);
        self
    }

    /// On each step finish callback (non-stream).
    pub fn on_step_finish<F>(mut self, cb: F) -> Self
    where
        F: Fn(&StepResult) + Send + Sync + 'static,
    {
        self.builder = self.builder.on_step_finish(cb);
        self
    }

    /// Final finish callback with all steps (non-stream).
    pub fn on_finish<F>(mut self, cb: F) -> Self
    where
        F: Fn(&[StepResult]) + Send + Sync + 'static,
    {
        self.builder = self.builder.on_finish(cb);
        self
    }

    /// Tool approval callback (applies to both non-stream and stream variants).
    #[allow(clippy::type_complexity)]
    pub fn on_tool_approval<F>(mut self, cb: F) -> Self
    where
        F: Fn(&str, &serde_json::Value) -> ToolApproval + Send + Sync + 'static,
    {
        self.builder = self.builder.on_tool_approval(cb);
        self
    }

    /// Preliminary tool result callback (applies to both variants).
    #[allow(clippy::type_complexity)]
    pub fn on_preliminary_tool_result<F>(mut self, cb: F) -> Self
    where
        F: Fn(&str, &str, &serde_json::Value) + Send + Sync + 'static,
    {
        self.builder = self.builder.on_preliminary_tool_result(cb);
        self
    }

    /// Prepare-step callback for dynamic tool/message selection (non-stream).
    pub fn prepare_step(mut self, f: PrepareStepFn) -> Self {
        self.builder = self.builder.prepare_step(f);
        self
    }

    /// Set telemetry configuration (applies to both variants).
    pub fn telemetry(mut self, cfg: siumai::observability::telemetry::TelemetryConfig) -> Self {
        self.builder = self.builder.telemetry(cfg);
        self
    }

    /// Set agent-level CommonParams (applies to both variants).
    pub fn common_params(mut self, params: siumai::types::CommonParams) -> Self {
        self.builder = self.builder.common_params(params);
        self
    }

    /// Streaming on_chunk callback (stream variant only).
    pub fn on_chunk<F>(mut self, cb: F) -> Self
    where
        F: Fn(&siumai::streaming::ChatStreamEvent) + Send + Sync + 'static,
    {
        self.builder = self.builder.on_chunk(cb);
        self
    }

    /// Streaming abort callback (stream variant only).
    pub fn on_abort<F>(mut self, cb: F) -> Self
    where
        F: Fn(&[StepResult]) + Send + Sync + 'static,
    {
        self.builder = self.builder.on_abort(cb);
        self
    }

    /// Set a default tool choice strategy for all steps in this workflow (non-stream).
    ///
    /// This composes with any existing prepare_step callback and applies
    /// when no per-step override is provided.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.builder = self.builder.tool_choice(choice);
        self
    }

    /// Restrict the active tools for all steps in this workflow (non-stream).
    ///
    /// This composes with any existing prepare_step callback and applies
    /// when no per-step override is provided.
    pub fn active_tools(mut self, tools: Vec<String>) -> Self {
        self.builder = self.builder.active_tools(tools);
        self
    }

    /// Register a worker (builder style).
    pub fn with_worker(mut self, id: impl Into<String>, agent: ToolLoopAgent<M>) -> Self {
        let worker = Worker::new(id, agent);
        self.workers.insert(worker.id.clone(), worker);
        self
    }

    /// Register an already-constructed worker (builder style).
    pub fn register_worker(mut self, worker: Worker<M>) -> Self {
        self.workers.insert(worker.id.clone(), worker);
        self
    }

    /// Convenience: register a planner worker using the default planner ID.
    pub fn with_planner_agent(self, agent: ToolLoopAgent<M>) -> Self {
        self.with_worker(WORKER_PLANNER, agent)
    }

    /// Convenience: register a coder worker using the default coder ID.
    pub fn with_coder_agent(self, agent: ToolLoopAgent<M>) -> Self {
        self.with_worker(WORKER_CODER, agent)
    }

    /// Convenience: register a researcher worker using the default researcher ID.
    pub fn with_researcher_agent(self, agent: ToolLoopAgent<M>) -> Self {
        self.with_worker(WORKER_RESEARCHER, agent)
    }

    /// Attach a workflow memory implementation.
    ///
    /// When used with [`Workflow::run_with_memory`], this allows
    /// loading/saving workflow state across runs using a caller-provided key.
    pub fn with_memory(mut self, memory: Arc<dyn WorkflowMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Build the final `Workflow<M>` instance.
    pub fn build(self) -> Workflow<M> {
        Workflow {
            orchestrator: Orchestrator {
                model: self.model,
                tools: self.tools,
                builder: self.builder,
            },
            workers: self.workers,
            memory: self.memory,
        }
    }
}

/// High-level workflow that coordinates an orchestrator and multiple workers.
pub struct Workflow<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    orchestrator: Orchestrator<M>,
    workers: HashMap<String, Worker<M>>,
    memory: Option<Arc<dyn WorkflowMemory>>,
}

impl<M> Workflow<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    /// Create a new workflow from an orchestrator.
    ///
    /// The orchestrator defines the main control loop; workers are registered
    /// separately and can be invoked via special "worker:" tools.
    pub fn new(orchestrator: Orchestrator<M>) -> Self {
        Self {
            orchestrator,
            workers: HashMap::new(),
            memory: None,
        }
    }

    /// Register a worker with this workflow (builder style).
    ///
    /// Workers can be invoked via tool calls with names of the form
    /// `\"worker:<id>\"`. The tool arguments should include an `input` field
    /// containing the user message for the worker.
    pub fn register_worker(mut self, worker: Worker<M>) -> Self {
        self.workers.insert(worker.id.clone(), worker);
        self
    }

    /// Convenience helper to register a worker from id + agent.
    pub fn with_worker(mut self, id: impl Into<String>, agent: ToolLoopAgent<M>) -> Self {
        let worker = Worker::new(id, agent);
        self.workers.insert(worker.id.clone(), worker);
        self
    }

    /// Access the underlying orchestrator.
    pub fn orchestrator(&self) -> &Orchestrator<M> {
        &self.orchestrator
    }

    /// Run the workflow with the given messages and optional base tool resolver.
    ///
    /// `base_resolver` is used for non-worker tools; worker tools use the
    /// registered `ToolLoopAgent` instances.
    ///
    /// Returns `(final_response, steps, workflow_state)`.
    pub async fn run(
        &self,
        messages: Vec<ChatMessage>,
        base_resolver: Option<&dyn ToolResolver>,
    ) -> Result<(ChatResponse, Vec<StepResult>, WorkflowState), LlmError> {
        self.run_internal(messages, base_resolver, None).await
    }

    /// Run the workflow with a memory key, loading/saving state via
    /// the configured `WorkflowMemory` implementation.
    ///
    /// If no memory implementation is attached, this returns a
    /// `ConfigurationError`.
    pub async fn run_with_memory(
        &self,
        session_key: &str,
        messages: Vec<ChatMessage>,
        base_resolver: Option<&dyn ToolResolver>,
    ) -> Result<(ChatResponse, Vec<StepResult>, WorkflowState), LlmError> {
        if self.memory.is_none() {
            return Err(LlmError::ConfigurationError(
                "WorkflowMemory is not configured. Use WorkflowBuilder::with_memory to attach one."
                    .into(),
            ));
        }
        self.run_internal(messages, base_resolver, Some(session_key))
            .await
    }

    async fn run_internal(
        &self,
        messages: Vec<ChatMessage>,
        base_resolver: Option<&dyn ToolResolver>,
        session_key: Option<&str>,
    ) -> Result<(ChatResponse, Vec<StepResult>, WorkflowState), LlmError> {
        let state = Mutex::new(WorkflowState::default());

        // If memory is configured and a session key is provided, attempt to
        // load previous state into the working state before running.
        if let (Some(memory), Some(key)) = (&self.memory, session_key) {
            if let Some(prev) = memory.load(key).await? {
                let mut guard = state.lock().unwrap();
                *guard = prev;
            }
        }

        // Local resolver that routes "worker:<id>" tools to the corresponding
        // worker agent, and delegates others to the base resolver.
        struct WorkflowToolResolver<'a, M>
        where
            M: ChatCapability + Send + Sync + 'static,
        {
            workers: &'a HashMap<String, Worker<M>>,
            base: Option<&'a dyn ToolResolver>,
            state: &'a Mutex<WorkflowState>,
        }

        #[async_trait::async_trait]
        impl<'a, M> ToolResolver for WorkflowToolResolver<'a, M>
        where
            M: ChatCapability + Send + Sync + 'static,
        {
            async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
                // Worker tools: "worker:<id>"
                if let Some(worker_id) = name.strip_prefix("worker:") {
                    let worker = self.workers.get(worker_id).ok_or_else(|| {
                        LlmError::ToolCallError(format!("Unknown worker: {}", worker_id))
                    })?;

                    // Extract input for the worker; convention: arguments.input is the user text.
                    let input = arguments
                        .get("input")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();

                    let msgs = vec![ChatMessage::user(input).build()];

                    // Delegate tool execution inside the worker to the base resolver if present.
                    let base = self.base.ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "Workflow worker requires a base ToolResolver for nested tools".into(),
                        )
                    })?;

                    let result = worker.agent.generate(msgs, base).await?;

                    // Store structured output in workflow state if present.
                    if let Some(output) = &result.output {
                        let mut guard = self.state.lock().unwrap();
                        let worker_key = worker_id.to_string();
                        guard
                            .worker_outputs
                            .insert(worker_key.clone(), output.clone());
                        // Append worker steps to per-worker history.
                        let entry = guard.worker_steps.entry(worker_key).or_default();
                        entry.extend(result.steps.clone());
                    }

                    // Return a generic JSON payload summarizing the worker result.
                    Ok(json!({
                        "worker_id": worker_id,
                        "text": result.text(),
                        "output": result.output,
                    }))
                } else if let Some(base) = self.base {
                    // Delegate non-worker tools to base resolver.
                    base.call_tool(name, arguments).await
                } else {
                    Err(LlmError::ConfigurationError(format!(
                        "No ToolResolver available for tool: {name}"
                    )))
                }
            }
        }

        let resolver = WorkflowToolResolver {
            workers: &self.workers,
            base: base_resolver,
            state: &state,
        };

        let (resp, steps) = self.orchestrator.run(messages, Some(&resolver)).await?;

        {
            // Capture top-level orchestrator steps into workflow state for
            // downstream memory/logging/analytics usage.
            let mut guard = state.lock().unwrap();
            guard.orchestration_steps = steps.clone();
        }

        let final_state = state.into_inner().map_err(|e| {
            LlmError::InternalError(format!("Failed to finalize workflow state: {e}"))
        })?;

        // Persist state if memory is configured and a session key was provided.
        if let (Some(memory), Some(key)) = (&self.memory, session_key) {
            memory.save(key, &final_state).await?;
        }

        Ok((resp, steps, final_state))
    }

    /// Run the workflow and decode the final response into a typed `T` using
    /// a structured output configuration.
    pub async fn run_typed<T: DeserializeOwned>(
        &self,
        messages: Vec<ChatMessage>,
        base_resolver: Option<&dyn ToolResolver>,
        cfg: OutputDecodeConfig,
    ) -> Result<(T, Vec<StepResult>, WorkflowState), LlmError> {
        let (resp, steps, state) = self.run(messages, base_resolver).await?;
        let text = resp
            .content_text()
            .ok_or_else(|| LlmError::ParseError("No text content in response".into()))?;
        let value = decode_typed::<T>(text, &cfg)?;
        Ok((value, steps, state))
    }
}
