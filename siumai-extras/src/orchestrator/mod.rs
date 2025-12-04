//! Orchestrator for multi-step tool calling.
//!
//! This module provides a flexible orchestration system for multi-step tool calling with LLMs.
//! It supports:
//! - Flexible stop conditions (step count, specific tool calls, custom conditions)
//! - Dynamic step preparation (modify tools, messages, etc. before each step)
//! - Tool approval workflows
//! - Streaming and non-streaming execution
//! - Full telemetry integration
//!
//! # Example
//!
//! ```rust,ignore
//! use siumai_extras::orchestrator::{generate, step_count_is, OrchestratorOptions};
//!
//! let (response, steps) = generate(
//!     &model,
//!     messages,
//!     Some(tools),
//!     Some(&resolver),
//!     vec![step_count_is(10)],
//!     OrchestratorOptions::default(),
//! ).await?;
//! ```

// Public modules
pub mod agent;
pub mod builder;
pub mod prepare_step;
pub mod stop_condition;
pub mod types;
pub mod workflow;

// Private modules
mod generate;
mod stream;
mod validation;

// Test modules
#[cfg(test)]
mod tests;

// Re-export public types
pub use agent::ToolLoopAgent;
pub use prepare_step::{PrepareStepContext, PrepareStepFn, PrepareStepResult, ToolChoice};
pub use stop_condition::{
    StopCondition, all_of, any_of, custom_condition, has_no_tool_calls, has_text_response,
    has_tool_call, has_tool_result, step_count_is,
};
pub use types::{
    AgentResult, OrchestratorOptions, OrchestratorStreamOptions, StepResult, ToolApproval,
    ToolExecutionResult, ToolResolver,
};
pub use workflow::{
    InMemoryWorkflowMemory, WORKER_CODER, WORKER_PLANNER, WORKER_RESEARCHER, Worker, Workflow,
    WorkflowBuilder, WorkflowMemory, WorkflowState,
};

// Re-export main functions
pub use builder::OrchestratorBuilder;
pub use generate::generate;
pub use stream::{StreamOrchestration, generate_stream, generate_stream_owned};

use crate::structured_output::{OutputDecodeConfig, decode_typed};
use serde::de::DeserializeOwned;
use siumai::error::LlmError;
use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, ChatResponse, Tool};

/// High-level orchestrator facade that binds a model and tools, inspired by
/// Vercel AI SDK's `Orchestrator`.
///
/// This wraps the lower-level `generate` / `generate_stream` / `generate_stream_owned`
/// functions and `OrchestratorBuilder` into a single struct with a fluent API.
///
/// Typical usage:
///
/// ```rust,ignore
/// use siumai::prelude::*;
/// use siumai_extras::orchestrator::{Orchestrator, ToolResolver, step_count_is};
///
/// # struct MyResolver;
/// # #[async_trait::async_trait]
/// # impl ToolResolver for MyResolver {
/// #     async fn call_tool(
/// #         &self,
/// #         _name: &str,
/// #         _args: serde_json::Value,
/// #     ) -> Result<serde_json::Value, siumai::error::LlmError> {
/// #         Ok(serde_json::json!({}))
/// #     }
/// # }
/// #
/// # async fn demo(model: impl ChatCapability) -> Result<(), siumai::error::LlmError> {
/// let tools = vec![
///     Tool::function(
///         "get_weather",
///         "Get weather for a city",
///         serde_json::json!({
///             "type": "object",
///             "properties": { "city": { "type": "string" } },
///             "required": ["city"]
///         }),
///     ),
/// ];
///
/// let orchestrator = Orchestrator::new(model, tools)
///     .max_steps(10);
///
/// let messages = vec![ChatMessage::user("What's the weather in Tokyo?").build()];
/// let resolver = MyResolver;
///
/// let (response, steps) = orchestrator.run(messages, Some(&resolver)).await?;
/// println!("Answer: {}", response.content_text().unwrap_or_default());
/// println!("Steps taken: {}", steps.len());
/// # Ok(())
/// # }
/// ```
pub struct Orchestrator<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    model: M,
    tools: Vec<Tool>,
    builder: OrchestratorBuilder,
}

impl<M> Orchestrator<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    /// Create a new orchestrator binding a model and a list of tools.
    pub fn new(model: M, tools: Vec<Tool>) -> Self {
        Self {
            model,
            tools,
            builder: OrchestratorBuilder::new(),
        }
    }

    /// Set maximum steps (applies to both non-stream and stream variants).
    pub fn max_steps(mut self, max: usize) -> Self {
        self.builder = self.builder.max_steps(max);
        self
    }

    /// Add a stop condition.
    pub fn stop_when(mut self, cond: Box<dyn StopCondition>) -> Self {
        self.builder = self.builder.stop_when(cond);
        self
    }

    /// Add multiple stop conditions.
    pub fn stop_when_all(mut self, conds: Vec<Box<dyn StopCondition>>) -> Self {
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

    /// Tool approval callback (applies to both variants).
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

    /// Set a default tool choice strategy for all steps (non-stream).
    ///
    /// This is a thin wrapper over `OrchestratorBuilder::tool_choice` and
    /// composes with any existing `prepare_step` callback.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.builder = self.builder.tool_choice(choice);
        self
    }

    /// Restrict the active tools for all steps (non-stream).
    ///
    /// This is a thin wrapper over `OrchestratorBuilder::active_tools` and
    /// composes with any existing `prepare_step` callback.
    pub fn active_tools(mut self, tools: Vec<String>) -> Self {
        self.builder = self.builder.active_tools(tools);
        self
    }

    /// Set telemetry configuration (applies to both variants).
    pub fn telemetry(mut self, cfg: siumai::telemetry::TelemetryConfig) -> Self {
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

    /// Access the underlying tools.
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Access the underlying model by reference.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Run non-stream orchestration using the bound model and tools.
    pub async fn run(
        &self,
        messages: Vec<ChatMessage>,
        resolver: Option<&dyn ToolResolver>,
    ) -> Result<(ChatResponse, Vec<StepResult>), LlmError> {
        self.builder
            .run(&self.model, messages, Some(self.tools.clone()), resolver)
            .await
    }

    /// Run streaming orchestration without executing tools (stream glue only).
    pub async fn run_stream_basic(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<StreamOrchestration, LlmError> {
        self.builder
            .run_stream(&self.model, messages, Some(self.tools.clone()))
            .await
    }

    /// Run streaming orchestration with tool execution in the background.
    ///
    /// This consumes a clone of the model and resolver and returns a
    /// `StreamOrchestration` handle.
    pub async fn run_stream_owned<R>(
        &self,
        messages: Vec<ChatMessage>,
        resolver: Option<R>,
    ) -> Result<StreamOrchestration, LlmError>
    where
        M: Clone,
        R: ToolResolver + Send + Sync + 'static,
    {
        self.builder
            .run_stream_owned(
                self.model.clone(),
                messages,
                Some(self.tools.clone()),
                resolver,
            )
            .await
    }

    /// Run non-stream orchestration and decode the final response into a typed
    /// value `T` using structured output configuration.
    ///
    /// This is a thin wrapper over `run(...)` plus `OutputDecodeConfig`,
    /// mirroring Vercel AI SDK's `experimental_output` behavior.
    pub async fn run_typed<T: DeserializeOwned>(
        &self,
        messages: Vec<ChatMessage>,
        resolver: Option<&dyn ToolResolver>,
        cfg: OutputDecodeConfig,
    ) -> Result<(T, Vec<StepResult>), LlmError> {
        let (resp, steps) = self.run(messages, resolver).await?;
        let text = resp
            .content_text()
            .ok_or_else(|| LlmError::ParseError("No text content in response".into()))?;
        let value = decode_typed::<T>(text, &cfg)?;
        Ok((value, steps))
    }

    /// Convenience helper: run orchestration and decode into `T` from a simple
    /// `OutputSchema`. Shape defaults to `Object`, mode defaults to `Auto`.
    pub async fn run_typed_with_schema<T: DeserializeOwned>(
        &self,
        messages: Vec<ChatMessage>,
        resolver: Option<&dyn ToolResolver>,
        schema: siumai::types::OutputSchema,
    ) -> Result<(T, Vec<StepResult>), LlmError> {
        let cfg = OutputDecodeConfig::from_schema(schema);
        self.run_typed(messages, resolver, cfg).await
    }
}
