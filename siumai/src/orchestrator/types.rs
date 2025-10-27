#![allow(clippy::type_complexity)]
//! Core types for orchestrator module.

use std::sync::Arc;

use serde_json::Value;

use super::prepare_step::PrepareStepFn;
use crate::error::LlmError;
use crate::streaming::ChatStreamEvent;
use crate::telemetry::TelemetryConfig;
use crate::types::{
    ChatMessage, ChatResponse, CommonParams, ContentPart, FinishReason, Usage, Warning,
};

/// Result of a single step during orchestration.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Messages contributed in this step (assistant + tool outputs).
    pub messages: Vec<ChatMessage>,
    /// Finish reason returned by the model for this step.
    pub finish_reason: Option<FinishReason>,
    /// Usage reported by the provider for this step.
    pub usage: Option<Usage>,
    /// Tool calls requested by the model in this step (extracted from content).
    pub tool_calls: Vec<ContentPart>,
    /// Tool results generated in this step (extracted from tool messages).
    ///
    /// This field provides convenient access to tool results without needing to
    /// filter through messages. It contains the same data as the tool messages
    /// in the `messages` field, but in a more accessible format.
    pub tool_results: Vec<ContentPart>,
    /// Warnings from the model provider (e.g., unsupported settings).
    ///
    /// This field contains warnings from the ChatResponse, providing visibility
    /// into non-fatal issues during generation.
    pub warnings: Option<Vec<Warning>>,
}

impl StepResult {
    /// Merge usage from all steps.
    ///
    /// This follows Vercel AI SDK's approach: simply sum all usage fields across steps.
    /// Each step's usage is treated independently and added together.
    pub fn merge_usage(steps: &[StepResult]) -> Option<Usage> {
        let mut acc: Option<Usage> = None;
        for s in steps.iter() {
            if let Some(u) = &s.usage {
                match &mut acc {
                    Some(t) => {
                        // Simple addition for all fields (Vercel AI style)
                        t.merge(u);
                    }
                    None => {
                        // First step: clone directly without modification
                        acc = Some(u.clone());
                    }
                }
            }
        }
        acc
    }

    /// Get the text content from the assistant message in this step.
    ///
    /// Returns the text content from the first assistant message in this step's messages.
    /// Returns `None` if there are no assistant messages or if the message has no text content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::orchestrator::StepResult;
    ///
    /// let step: StepResult = /* ... */;
    /// if let Some(text) = step.text() {
    ///     println!("Assistant said: {}", text);
    /// }
    /// ```
    pub fn text(&self) -> Option<&str> {
        self.messages
            .iter()
            .find(|msg| matches!(msg.role, crate::types::MessageRole::Assistant))
            .and_then(|msg| msg.content_text())
    }

    /// Get all tool calls from this step.
    ///
    /// Returns a reference to the tool calls vector.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::orchestrator::StepResult;
    ///
    /// let step: StepResult = /* ... */;
    /// for tool_call in step.tool_calls() {
    ///     if let Some(info) = tool_call.as_tool_call() {
    ///         println!("Tool: {}", info.tool_name);
    ///     }
    /// }
    /// ```
    pub fn tool_calls(&self) -> &[ContentPart] {
        &self.tool_calls
    }

    /// Check if this step has any tool calls.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::orchestrator::StepResult;
    ///
    /// let step: StepResult = /* ... */;
    /// if step.has_tool_calls() {
    ///     println!("This step made tool calls");
    /// }
    /// ```
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Get all tool results from this step.
    ///
    /// Returns a reference to the tool results vector.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::orchestrator::StepResult;
    ///
    /// let step: StepResult = /* ... */;
    /// for tool_result in step.tool_results() {
    ///     if let Some(info) = tool_result.as_tool_result() {
    ///         println!("Tool: {}, Output: {:?}", info.tool_name, info.output);
    ///     }
    /// }
    /// ```
    pub fn tool_results(&self) -> &[ContentPart] {
        &self.tool_results
    }

    /// Check if this step has any tool results.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::orchestrator::StepResult;
    ///
    /// let step: StepResult = /* ... */;
    /// if step.has_tool_results() {
    ///     println!("This step has tool results");
    /// }
    /// ```
    pub fn has_tool_results(&self) -> bool {
        !self.tool_results.is_empty()
    }
}

/// Result of an agent's generate() call.
///
/// This structure contains the final response, all steps taken, and optionally
/// the extracted structured output if an output schema was configured.
///
/// Similar to Vercel AI SDK's agent result structure.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The final chat response from the model.
    pub response: ChatResponse,
    /// All steps taken during the agent's execution.
    pub steps: Vec<StepResult>,
    /// Extracted structured output (if output_schema was set).
    ///
    /// This field contains the parsed JSON output from the final response.
    /// It can be validated using a SchemaValidator from `siumai-extras`.
    pub output: Option<serde_json::Value>,
}

impl AgentResult {
    /// Create a new AgentResult.
    pub fn new(response: ChatResponse, steps: Vec<StepResult>) -> Self {
        Self {
            response,
            steps,
            output: None,
        }
    }

    /// Create a new AgentResult with structured output.
    pub fn with_output(
        response: ChatResponse,
        steps: Vec<StepResult>,
        output: Option<serde_json::Value>,
    ) -> Self {
        Self {
            response,
            steps,
            output,
        }
    }

    /// Get the final text response.
    pub fn text(&self) -> Option<&str> {
        self.response.content_text()
    }

    /// Get all tool calls from all steps.
    pub fn all_tool_calls(&self) -> Vec<&ContentPart> {
        self.steps
            .iter()
            .flat_map(|step| step.tool_calls.iter())
            .collect()
    }

    /// Get all tool results from all steps.
    pub fn all_tool_results(&self) -> Vec<&ContentPart> {
        self.steps
            .iter()
            .flat_map(|step| step.tool_results.iter())
            .collect()
    }

    /// Get total usage across all steps.
    pub fn total_usage(&self) -> Option<Usage> {
        StepResult::merge_usage(&self.steps)
    }

    /// Get all warnings from all steps.
    pub fn all_warnings(&self) -> Vec<&Warning> {
        self.steps
            .iter()
            .filter_map(|step| step.warnings.as_ref())
            .flat_map(|warnings| warnings.iter())
            .collect()
    }
}

/// Tool approval decision.
#[derive(Debug, Clone)]
pub enum ToolApproval {
    /// Approve tool call with given arguments (can be same as original).
    Approve(Value),
    /// Modify arguments before execution.
    Modify(Value),
    /// Deny tool call with reason; orchestrator will emit an error result as tool message.
    Deny { reason: String },
}

/// Tool execution result - can be preliminary (intermediate) or final.
#[derive(Debug, Clone)]
pub enum ToolExecutionResult {
    /// Preliminary result during tool execution (e.g., progress update).
    /// The tool is still running and will produce more results.
    Preliminary {
        /// The intermediate output value
        output: Value,
    },
    /// Final result of tool execution.
    /// This is the last result from the tool.
    Final {
        /// The final output value
        output: Value,
    },
}

impl ToolExecutionResult {
    /// Create a preliminary result
    pub fn preliminary(output: Value) -> Self {
        Self::Preliminary { output }
    }

    /// Create a final result
    pub fn final_result(output: Value) -> Self {
        Self::Final { output }
    }

    /// Check if this is a preliminary result
    pub fn is_preliminary(&self) -> bool {
        matches!(self, Self::Preliminary { .. })
    }

    /// Check if this is a final result
    pub fn is_final(&self) -> bool {
        matches!(self, Self::Final { .. })
    }

    /// Get the output value
    pub fn output(&self) -> &Value {
        match self {
            Self::Preliminary { output } | Self::Final { output } => output,
        }
    }

    /// Consume and get the output value
    pub fn into_output(self) -> Value {
        match self {
            Self::Preliminary { output } | Self::Final { output } => output,
        }
    }
}

/// A tool resolver abstraction that supports both simple and streaming tool execution.
///
/// # Simple Tool Execution
///
/// For simple tools that return a single result, implement `call_tool`:
///
/// ```rust,ignore
/// #[async_trait::async_trait]
/// impl ToolResolver for MyResolver {
///     async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
///         // Execute tool and return result
///         Ok(json!({"result": "done"}))
///     }
/// }
/// ```
///
/// # Streaming Tool Execution
///
/// For long-running tools that can provide progress updates, implement `call_tool_stream`:
///
/// ```rust,ignore
/// use futures::stream::{self, BoxStream};
///
/// #[async_trait::async_trait]
/// impl ToolResolver for MyResolver {
///     async fn call_tool_stream(
///         &self,
///         name: &str,
///         arguments: Value,
///     ) -> Result<BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError> {
///         Ok(Box::pin(stream::iter(vec![
///             Ok(ToolExecutionResult::preliminary(json!({"progress": 50}))),
///             Ok(ToolExecutionResult::preliminary(json!({"progress": 100}))),
///             Ok(ToolExecutionResult::final_result(json!({"result": "done"}))),
///         ])))
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait ToolResolver: Send + Sync {
    /// Execute a tool by name with structured JSON arguments.
    /// Returns a structured JSON value as tool output.
    ///
    /// This is the simple, non-streaming version. For streaming execution,
    /// override `call_tool_stream` instead.
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError>;

    /// Execute a tool with streaming support.
    ///
    /// This method allows tools to return intermediate results (preliminary results)
    /// during execution, which is useful for long-running operations that can provide
    /// progress updates.
    ///
    /// The default implementation wraps `call_tool` to return a single final result.
    /// Override this method to provide true streaming support.
    ///
    /// # Returns
    ///
    /// A stream of `ToolExecutionResult` where:
    /// - `Preliminary` results are intermediate progress updates
    /// - `Final` result is the last result (must be exactly one)
    async fn call_tool_stream(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        // Default implementation: wrap call_tool in a stream
        let result = self.call_tool(name, arguments).await?;
        Ok(Box::pin(futures::stream::once(async move {
            Ok(ToolExecutionResult::final_result(result))
        })))
    }
}

/// Orchestrator options for non-streaming generate.
#[derive(Clone)]
pub struct OrchestratorOptions {
    /// Maximum steps to perform (including the final response step).
    pub max_steps: usize,
    /// Step-finish callback.
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    /// Finish callback with all steps.
    pub on_finish: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    /// Optional tool approval callback. Allows approve/deny/modify tool arguments.
    pub on_tool_approval: Option<Arc<dyn Fn(&str, &Value) -> ToolApproval + Send + Sync>>,
    /// Optional preliminary tool result callback.
    /// Called when a tool returns intermediate results during execution.
    /// Receives: (tool_name, tool_call_id, preliminary_output)
    pub on_preliminary_tool_result: Option<Arc<dyn Fn(&str, &str, &Value) + Send + Sync>>,
    /// Optional prepare step callback for dynamic step configuration.
    pub prepare_step: Option<PrepareStepFn>,
    /// Optional telemetry configuration.
    pub telemetry: Option<TelemetryConfig>,
    /// Agent-level model parameters (temperature, max_tokens, etc.).
    ///
    /// These parameters will be applied to all chat requests made during orchestration.
    /// Similar to Vercel AI SDK's agent-level parameter configuration.
    pub common_params: Option<CommonParams>,
}

impl Default for OrchestratorOptions {
    fn default() -> Self {
        Self {
            max_steps: 8,
            on_step_finish: None,
            on_finish: None,
            on_tool_approval: None,
            on_preliminary_tool_result: None,
            prepare_step: None,
            telemetry: None,
            common_params: None,
        }
    }
}

/// Orchestrator options for streaming generate.
pub struct OrchestratorStreamOptions {
    pub max_steps: usize,
    pub on_chunk: Option<Arc<dyn Fn(&ChatStreamEvent) + Send + Sync>>,
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    pub on_finish: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    pub on_tool_approval: Option<Arc<dyn Fn(&str, &Value) -> ToolApproval + Send + Sync>>,
    /// Optional preliminary tool result callback.
    /// Called when a tool returns intermediate results during execution.
    /// Receives: (tool_name, tool_call_id, preliminary_output)
    pub on_preliminary_tool_result: Option<Arc<dyn Fn(&str, &str, &Value) + Send + Sync>>,
    pub on_abort: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    pub telemetry: Option<TelemetryConfig>,
    /// Agent-level model parameters (temperature, max_tokens, etc.).
    ///
    /// These parameters will be applied to all chat requests made during orchestration.
    /// Similar to Vercel AI SDK's agent-level parameter configuration.
    pub common_params: Option<CommonParams>,
}

impl Default for OrchestratorStreamOptions {
    fn default() -> Self {
        Self {
            max_steps: 8,
            on_chunk: None,
            on_step_finish: None,
            on_finish: None,
            on_tool_approval: None,
            on_preliminary_tool_result: None,
            on_abort: None,
            telemetry: None,
            common_params: None,
        }
    }
}
