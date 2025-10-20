//! Agent abstraction for reusable multi-step tool calling.

use std::sync::Arc;

use super::generate::generate;
use super::stop_condition::{StopCondition, step_count_is};
use super::stream::{StreamOrchestration, generate_stream_owned};
use super::types::{OrchestratorOptions, OrchestratorStreamOptions, StepResult, ToolResolver};
use crate::error::LlmError;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, Tool};

/// A reusable agent that can generate text, stream responses, and use tools across multiple steps.
///
/// Unlike single-step functions, agents can iteratively call tools and make decisions based on
/// intermediate results. This is ideal for building autonomous AI systems that need to perform
/// complex, multi-step tasks.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::orchestrator::{ToolLoopAgent, step_count_is};
/// use siumai::providers::openai::OpenAiClient;
///
/// let agent = ToolLoopAgent::new(
///     OpenAiClient::new("gpt-4o"),
///     vec![weather_tool, calculator_tool],
///     vec![step_count_is(10)],
/// )
/// .with_system("You are a helpful assistant.");
///
/// // Generate a response
/// let (response, steps) = agent.generate(
///     vec![ChatMessage::user("What's the weather in NYC?").build()],
///     &resolver,
/// ).await?;
///
/// // Stream a response
/// let orchestration = agent.stream(
///     vec![ChatMessage::user("Calculate 100 * 25").build()],
///     &resolver,
/// ).await?;
/// ```
pub struct ToolLoopAgent<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    /// The language model to use.
    model: M,
    /// Tools available to the agent.
    tools: Vec<Tool>,
    /// Stop conditions for the agent loop.
    stop_conditions: Vec<Box<dyn StopCondition>>,
    /// Optional system message.
    system: Option<String>,
    /// Optional agent ID for tracking.
    id: Option<String>,
    /// Orchestrator options (callbacks, telemetry, etc.).
    options: OrchestratorOptions,
}

impl<M> ToolLoopAgent<M>
where
    M: ChatCapability + Send + Sync + 'static,
{
    /// Create a new ToolLoopAgent.
    ///
    /// # Arguments
    ///
    /// * `model` - The language model to use
    /// * `tools` - Tools available to the agent
    /// * `stop_conditions` - Conditions that determine when to stop the loop
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = ToolLoopAgent::new(
    ///     model,
    ///     vec![weather_tool, calculator_tool],
    ///     vec![step_count_is(20)],
    /// );
    /// ```
    pub fn new(model: M, tools: Vec<Tool>, stop_conditions: Vec<Box<dyn StopCondition>>) -> Self {
        Self {
            model,
            tools,
            stop_conditions,
            system: None,
            id: None,
            options: OrchestratorOptions::default(),
        }
    }

    /// Create a new ToolLoopAgent with default stop condition (20 steps).
    ///
    /// # Arguments
    ///
    /// * `model` - The language model to use
    /// * `tools` - Tools available to the agent
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = ToolLoopAgent::with_defaults(model, vec![weather_tool]);
    /// ```
    pub fn with_defaults(model: M, tools: Vec<Tool>) -> Self {
        Self::new(model, tools, vec![step_count_is(20)])
    }

    /// Set the system message for the agent.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_system("You are a helpful assistant.");
    /// ```
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the agent ID for tracking.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_id("weather-agent");
    /// ```
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the orchestrator options.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_options(OrchestratorOptions {
    ///     max_steps: 10,
    ///     on_step_finish: Some(Arc::new(|step| {
    ///         println!("Step finished: {:?}", step);
    ///     })),
    ///     ..Default::default()
    /// });
    /// ```
    pub fn with_options(mut self, options: OrchestratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Set a callback to be called when each step finishes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.on_step_finish(Arc::new(|step| {
    ///     println!("Step finished with {} tool calls", step.tool_calls.len());
    /// }));
    /// ```
    pub fn on_step_finish(mut self, callback: Arc<dyn Fn(&StepResult) + Send + Sync>) -> Self {
        self.options.on_step_finish = Some(callback);
        self
    }

    /// Set a callback to be called when all steps finish.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.on_finish(Arc::new(|steps| {
    ///     println!("Completed in {} steps", steps.len());
    /// }));
    /// ```
    pub fn on_finish(mut self, callback: Arc<dyn Fn(&[StepResult]) + Send + Sync>) -> Self {
        self.options.on_finish = Some(callback);
        self
    }

    /// Get the agent ID.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Get the tools available to the agent.
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Generate a response (non-streaming).
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history
    /// * `resolver` - Tool resolver for executing tool calls
    ///
    /// # Returns
    ///
    /// Returns a tuple of (final ChatResponse, all StepResults)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (response, steps) = agent.generate(
    ///     vec![ChatMessage::user("What's the weather?").build()],
    ///     &resolver,
    /// ).await?;
    /// ```
    pub async fn generate(
        &self,
        mut messages: Vec<ChatMessage>,
        resolver: &dyn ToolResolver,
    ) -> Result<(ChatResponse, Vec<StepResult>), LlmError> {
        // Prepend system message if set
        if let Some(ref system) = self.system {
            messages.insert(0, ChatMessage::system(system).build());
        }

        let stop_refs: Vec<&dyn StopCondition> =
            self.stop_conditions.iter().map(|b| b.as_ref()).collect();

        generate(
            &self.model,
            messages,
            Some(self.tools.clone()),
            Some(resolver),
            &stop_refs,
            self.options.clone(),
        )
        .await
    }

    /// Stream a response.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history
    /// * `resolver` - Tool resolver for executing tool calls
    ///
    /// # Returns
    ///
    /// Returns a `StreamOrchestration` containing the stream, steps receiver, and cancel handle
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let orchestration = agent.stream(
    ///     vec![ChatMessage::user("Calculate 100 * 25").build()],
    ///     resolver,
    /// ).await?;
    ///
    /// // Consume the stream
    /// while let Some(event) = orchestration.stream.next().await {
    ///     // Handle event
    /// }
    ///
    /// // Get final steps
    /// let steps = orchestration.steps.await?;
    /// ```
    pub async fn stream<R>(
        self,
        mut messages: Vec<ChatMessage>,
        resolver: R,
    ) -> Result<StreamOrchestration, LlmError>
    where
        R: ToolResolver + Send + Sync + 'static,
    {
        // Prepend system message if set
        if let Some(ref system) = self.system {
            messages.insert(0, ChatMessage::system(system).build());
        }

        let stream_options = OrchestratorStreamOptions {
            max_steps: self.options.max_steps,
            on_chunk: None,
            on_step_finish: self.options.on_step_finish.clone(),
            on_finish: self.options.on_finish.clone(),
            on_tool_approval: self.options.on_tool_approval.clone(),
            on_abort: None,
            telemetry: self.options.telemetry.clone(),
        };

        generate_stream_owned(
            self.model,
            messages,
            Some(self.tools),
            Some(resolver),
            stream_options,
        )
        .await
    }
}

impl<M> Clone for ToolLoopAgent<M>
where
    M: ChatCapability + Send + Sync + Clone + 'static,
{
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            tools: self.tools.clone(),
            stop_conditions: vec![], // Note: StopCondition is not Clone, so we can't clone this
            system: self.system.clone(),
            id: self.id.clone(),
            options: OrchestratorOptions {
                max_steps: self.options.max_steps,
                on_step_finish: self.options.on_step_finish.clone(),
                on_finish: self.options.on_finish.clone(),
                on_tool_approval: self.options.on_tool_approval.clone(),
                prepare_step: self.options.prepare_step.clone(),
                telemetry: self.options.telemetry.clone(),
            },
        }
    }
}
