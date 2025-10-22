//! Agent abstraction for reusable multi-step tool calling.

use std::sync::Arc;

use super::generate::generate;
use super::stop_condition::{StopCondition, step_count_is};
use super::stream::{StreamOrchestration, generate_stream_owned};
use super::types::{
    AgentResult, OrchestratorOptions, OrchestratorStreamOptions, StepResult, ToolResolver,
};
use crate::error::LlmError;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, CommonParams, OutputSchema, Tool};

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
    /// Agent-level model parameters (temperature, max_tokens, etc.).
    ///
    /// These parameters will be applied to all chat requests made by the agent.
    /// Similar to Vercel AI SDK's agent-level parameter configuration.
    common_params: Option<CommonParams>,
    /// Optional output schema for structured output validation.
    ///
    /// When set, the agent will attempt to extract and parse structured output
    /// from the final response. The schema can be validated using a SchemaValidator
    /// from `siumai-extras` with the `schema` feature.
    ///
    /// Similar to Vercel AI SDK's `experimental_output` parameter.
    output_schema: Option<OutputSchema>,
    /// Agent-level tool choice setting.
    ///
    /// Controls how the model should use tools. Can be overridden per-step
    /// using the `prepare_step` callback.
    ///
    /// Similar to Vercel AI SDK's agent-level `toolChoice` parameter.
    tool_choice: Option<super::prepare_step::ToolChoice>,
    /// Agent-level active tools filter.
    ///
    /// Limits which tools are available to the agent. Can be overridden per-step
    /// using the `prepare_step` callback.
    ///
    /// Similar to Vercel AI SDK's agent-level `activeTools` parameter.
    active_tools: Option<Vec<String>>,
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
            common_params: None,
            output_schema: None,
            tool_choice: None,
            active_tools: None,
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

    /// Set agent-level model parameters.
    ///
    /// These parameters will be applied to all chat requests made by the agent.
    /// Similar to Vercel AI SDK's agent-level parameter configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_common_params(CommonParams {
    ///     temperature: Some(0.7),
    ///     max_tokens: Some(1000),
    ///     ..Default::default()
    /// });
    /// ```
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = Some(params);
        self
    }

    /// Set the temperature parameter.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_temperature(0.7);
    /// ```
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        let mut params = self.common_params.take().unwrap_or_default();
        params.temperature = Some(temperature);
        self.common_params = Some(params);
        self
    }

    /// Set the max_tokens parameter.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_max_tokens(1000);
    /// ```
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        let mut params = self.common_params.take().unwrap_or_default();
        params.max_tokens = Some(max_tokens);
        self.common_params = Some(params);
        self
    }

    /// Set the top_p parameter.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_top_p(0.9);
    /// ```
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        let mut params = self.common_params.take().unwrap_or_default();
        params.top_p = Some(top_p);
        self.common_params = Some(params);
        self
    }

    /// Set the seed parameter for deterministic generation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_seed(42);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        let mut params = self.common_params.take().unwrap_or_default();
        params.seed = Some(seed);
        self.common_params = Some(params);
        self
    }

    /// Set the output schema for structured output validation.
    ///
    /// When set, the agent will attempt to extract and parse structured output
    /// from the final response. The schema can be validated using a SchemaValidator
    /// from `siumai-extras` with the `schema` feature.
    ///
    /// Similar to Vercel AI SDK's `experimental_output` parameter.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serde_json::json;
    ///
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "name": {"type": "string"},
    ///         "age": {"type": "number"}
    ///     },
    ///     "required": ["name"]
    /// });
    ///
    /// let agent = agent.with_output_schema(
    ///     OutputSchema::new(schema)
    ///         .with_name("person_info")
    ///         .with_description("Person information")
    /// );
    /// ```
    pub fn with_output_schema(mut self, schema: OutputSchema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set the agent-level tool choice.
    ///
    /// Controls how the model should use tools. Can be overridden per-step
    /// using the `prepare_step` callback.
    ///
    /// Similar to Vercel AI SDK's agent-level `toolChoice` parameter.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::orchestrator::ToolChoice;
    ///
    /// let agent = agent.with_tool_choice(ToolChoice::Required);
    /// ```
    pub fn with_tool_choice(mut self, tool_choice: super::prepare_step::ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set the agent-level active tools filter.
    ///
    /// Limits which tools are available to the agent. Can be overridden per-step
    /// using the `prepare_step` callback.
    ///
    /// Similar to Vercel AI SDK's agent-level `activeTools` parameter.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = agent.with_active_tools(vec!["weather".to_string(), "calculator".to_string()]);
    /// ```
    pub fn with_active_tools(mut self, tools: Vec<String>) -> Self {
        self.active_tools = Some(tools);
        self
    }

    /// Get the agent ID.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Get the output schema if set.
    pub fn output_schema(&self) -> Option<&OutputSchema> {
        self.output_schema.as_ref()
    }

    /// Get the agent-level tool choice if set.
    pub fn tool_choice(&self) -> Option<&super::prepare_step::ToolChoice> {
        self.tool_choice.as_ref()
    }

    /// Get the agent-level active tools if set.
    pub fn active_tools(&self) -> Option<&[String]> {
        self.active_tools.as_deref()
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
    /// Returns an `AgentResult` containing the final response, all steps, and
    /// optionally the extracted structured output (if `output_schema` was set).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = agent.generate(
    ///     vec![ChatMessage::user("What's the weather?").build()],
    ///     &resolver,
    /// ).await?;
    ///
    /// // Access the final response
    /// println!("Response: {}", result.text().unwrap_or(""));
    ///
    /// // Access structured output (if schema was set)
    /// if let Some(output) = result.output {
    ///     println!("Structured output: {}", output);
    /// }
    /// ```
    pub async fn generate(
        &self,
        mut messages: Vec<ChatMessage>,
        resolver: &dyn ToolResolver,
    ) -> Result<AgentResult, LlmError> {
        // Prepend system message if set
        if let Some(ref system) = self.system {
            messages.insert(0, ChatMessage::system(system).build());
        }

        let stop_refs: Vec<&dyn StopCondition> =
            self.stop_conditions.iter().map(|b| b.as_ref()).collect();

        // Merge agent-level common_params with options
        let mut opts = self.options.clone();
        if let Some(ref common_params) = self.common_params {
            opts.common_params = Some(common_params.clone());
        }

        // Apply agent-level tool_choice and active_tools via prepare_step
        if self.tool_choice.is_some() || self.active_tools.is_some() {
            let agent_tool_choice = self.tool_choice.clone();
            let agent_active_tools = self.active_tools.clone();
            let existing_prepare_step = opts.prepare_step.clone();

            opts.prepare_step = Some(Arc::new(move |ctx| {
                // First call the existing prepare_step if any
                let mut result = if let Some(ref prepare) = existing_prepare_step {
                    prepare(ctx)
                } else {
                    super::prepare_step::PrepareStepResult::default()
                };

                // Apply agent-level settings if not overridden by prepare_step
                if result.tool_choice.is_none() {
                    result.tool_choice = agent_tool_choice.clone();
                }
                if result.active_tools.is_none() {
                    result.active_tools = agent_active_tools.clone();
                }

                result
            }));
        }

        let (response, steps) = generate(
            &self.model,
            messages,
            Some(self.tools.clone()),
            Some(resolver),
            &stop_refs,
            opts,
        )
        .await?;

        // Extract structured output if schema is set
        let output = if let Some(ref schema) = self.output_schema {
            Self::extract_output(&response, schema)?
        } else {
            None
        };

        Ok(AgentResult::with_output(response, steps, output))
    }

    /// Extract structured output from the response.
    ///
    /// This method attempts to parse JSON from the response text.
    /// The actual schema validation should be done by the user using
    /// a SchemaValidator from `siumai-extras`.
    fn extract_output(
        response: &ChatResponse,
        _schema: &OutputSchema,
    ) -> Result<Option<serde_json::Value>, LlmError> {
        // Try to get text content
        let text = match response.content_text() {
            Some(t) => t,
            None => return Ok(None),
        };

        // Try to parse as JSON
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(value) => Ok(Some(value)),
            Err(_) => {
                // If the entire text is not JSON, try to extract JSON from markdown code blocks
                if let Some(json_str) = Self::extract_json_from_markdown(text) {
                    match serde_json::from_str::<serde_json::Value>(json_str) {
                        Ok(value) => Ok(Some(value)),
                        Err(e) => Err(LlmError::ParseError(format!(
                            "Failed to parse extracted JSON: {}",
                            e
                        ))),
                    }
                } else {
                    Err(LlmError::ParseError(
                        "Response does not contain valid JSON".to_string(),
                    ))
                }
            }
        }
    }

    /// Extract JSON from markdown code blocks (```json ... ```).
    fn extract_json_from_markdown(text: &str) -> Option<&str> {
        // Look for ```json ... ``` or ``` ... ```
        let json_start = text
            .find("```json\n")
            .map(|i| i + 8)
            .or_else(|| text.find("```\n").map(|i| i + 4))?;

        let remaining = &text[json_start..];
        let json_end = remaining.find("\n```")?;

        Some(&remaining[..json_end])
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
            on_preliminary_tool_result: self.options.on_preliminary_tool_result.clone(),
            on_abort: None,
            telemetry: self.options.telemetry.clone(),
            common_params: self.common_params.clone(),
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
                on_preliminary_tool_result: self.options.on_preliminary_tool_result.clone(),
                prepare_step: self.options.prepare_step.clone(),
                telemetry: self.options.telemetry.clone(),
                common_params: self.options.common_params.clone(),
            },
            common_params: self.common_params.clone(),
            output_schema: self.output_schema.clone(),
            tool_choice: self.tool_choice.clone(),
            active_tools: self.active_tools.clone(),
        }
    }
}
