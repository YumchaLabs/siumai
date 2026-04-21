#![allow(clippy::type_complexity)]
//! Core types for orchestrator module.

use std::sync::Arc;

use serde_json::{Map, Value};

use super::prepare_step::PrepareStepFn;
use super::stop_condition::StopCondition;
use siumai::experimental::observability::telemetry::TelemetryConfig;
use siumai::prelude::unified::*;
use siumai::tooling::{ToolExecutionOptions, ToolRuntimeMetadata};
use siumai::types::ProviderMetadataMap;
use std::collections::HashMap;

pub use siumai::tooling::ToolExecutionResult;

// ---------------------------------------------------------------------------
// ToolResolver adapters
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ToolResolver for siumai::tooling::ExecutableTools {
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        self.execute(name, arguments).await
    }

    async fn call_tool_with_context(
        &self,
        name: &str,
        arguments: Value,
        context: &OrchestratorContext,
    ) -> Result<Value, LlmError> {
        self.execute_with_options(
            name,
            arguments,
            ToolExecutionOptions::default().with_context(
                context
                    .as_map()
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
        )
        .await
    }

    async fn call_tool_with_runtime_options(
        &self,
        name: &str,
        arguments: Value,
        options: ToolExecutionOptions,
    ) -> Result<Value, LlmError> {
        self.execute_with_options(name, arguments, options).await
    }

    async fn call_tool_stream(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        self.execute_stream(name, arguments, ToolExecutionOptions::default())
            .await
    }

    async fn call_tool_stream_with_context(
        &self,
        name: &str,
        arguments: Value,
        context: &OrchestratorContext,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        self.execute_stream(
            name,
            arguments,
            ToolExecutionOptions::default().with_context(
                context
                    .as_map()
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
        )
        .await
    }

    async fn call_tool_stream_with_runtime_options(
        &self,
        name: &str,
        arguments: Value,
        options: ToolExecutionOptions,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        self.execute_stream(name, arguments, options).await
    }

    fn runtime_tool_metadata(&self, name: &str) -> Option<ToolRuntimeMetadata> {
        self.runtime_metadata(name)
    }
}

/// Stable model identity for a single orchestrator step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepModelInfo {
    /// Canonical provider id (for example, `openai`).
    pub provider: String,
    /// Provider-specific model id (for example, `gpt-4o-mini`).
    pub model_id: String,
}

impl StepModelInfo {
    /// Create step model metadata from the bound language model.
    pub fn from_language_model(model: &(impl LanguageModel + ?Sized)) -> Self {
        Self {
            provider: model.provider_id().to_string(),
            model_id: model.model_id().to_string(),
        }
    }
}

/// Shared trait-object model handle used for per-step model overrides.
pub type StepLanguageModel = Arc<dyn LanguageModel>;

/// Standardized tool-call view derived from a step result.
#[derive(Debug, Clone, Copy)]
pub struct StepToolCallView<'a> {
    /// Original content part backing this tool call.
    pub part: &'a ContentPart,
    /// Tool call id.
    pub tool_call_id: &'a str,
    /// Tool name.
    pub tool_name: &'a str,
    /// Parsed tool input.
    pub input: &'a Value,
    /// Whether the provider executed the tool directly.
    pub provider_executed: Option<bool>,
    /// Whether the call is dynamic relative to the declared request tool set.
    pub dynamic: bool,
    /// Whether the tool call is invalid.
    pub invalid: Option<bool>,
    /// Optional invalidity/error payload.
    pub error: Option<&'a Value>,
    /// Optional human-readable title.
    pub title: Option<&'a str>,
}

/// Standardized tool-result view derived from a step result.
#[derive(Debug, Clone, Copy)]
pub struct StepToolResultView<'a> {
    /// Original content part backing this tool result.
    pub part: &'a ContentPart,
    /// Tool call id.
    pub tool_call_id: &'a str,
    /// Tool name.
    pub tool_name: &'a str,
    /// Resolved tool input from the matching tool call when available.
    pub input: Option<&'a Value>,
    /// Whether the provider executed the tool directly.
    pub provider_executed: Option<bool>,
    /// Whether the result is dynamic relative to the declared request tool set.
    pub dynamic: bool,
    /// Whether this is a preliminary result.
    pub preliminary: Option<bool>,
    /// Optional human-readable title.
    pub title: Option<&'a str>,
}

/// User-defined runtime context flowing through the entire orchestration.
///
/// This intentionally uses an open JSON object so extras can align with AI SDK's
/// runtime `context` contract without introducing generic type explosion.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrchestratorContext {
    values: Map<String, Value>,
}

impl OrchestratorContext {
    /// Create an empty context object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context wrapper from a JSON object map.
    pub fn from_map(values: Map<String, Value>) -> Self {
        Self { values }
    }

    /// Borrow the underlying JSON object.
    pub fn as_map(&self) -> &Map<String, Value> {
        &self.values
    }

    /// Consume the wrapper and return the underlying JSON object.
    pub fn into_map(self) -> Map<String, Value> {
        self.values
    }

    /// Look up a context value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Insert or replace a context value.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        self.values.insert(key.into(), value)
    }
}

impl From<Map<String, Value>> for OrchestratorContext {
    fn from(values: Map<String, Value>) -> Self {
        Self::from_map(values)
    }
}

/// Result of a single step during orchestration.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Stable identifier for the overall orchestration call this step belongs to.
    pub call_id: String,
    /// Zero-based index of this step within the orchestration.
    pub step_number: usize,
    /// Information about the language model that produced this step.
    pub model: StepModelInfo,
    /// The normalized chat request issued for this step.
    pub request: ChatRequest,
    /// The assistant response returned by the model for this step.
    pub response: ChatResponse,
    /// Raw finish reason returned by the provider when available.
    pub raw_finish_reason: Option<String>,
    /// Identifier from telemetry settings for grouping related operations.
    pub function_id: Option<String>,
    /// Additional telemetry metadata associated with this step.
    pub metadata: Option<HashMap<String, Value>>,
    /// User-defined runtime context after `prepare_step` overrides for this step.
    pub context: OrchestratorContext,
    /// Unified content generated in this step (assistant output + tool results).
    pub content: Vec<ContentPart>,
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
    /// Provider-specific metadata from the model response (Vercel-aligned).
    ///
    /// Shape: `{ "provider_id": { "key": value, ... }, ... }`
    ///
    /// This is useful for workflows that need to forward provider state between steps
    /// (e.g., Anthropic container IDs for code execution / skills).
    pub provider_metadata: Option<ProviderMetadataMap>,
}

impl StepResult {
    /// Normalize a message content payload into explicit content parts.
    pub fn normalize_content(content: &MessageContent) -> Vec<ContentPart> {
        match content {
            MessageContent::Text(text) if !text.is_empty() => vec![ContentPart::text(text.clone())],
            MessageContent::MultiModal(parts) => parts.clone(),
            _ => Vec::new(),
        }
    }

    /// Compose the unified step content from the model response and tool results.
    pub fn compose_content(
        response: &ChatResponse,
        tool_results: &[ContentPart],
    ) -> Vec<ContentPart> {
        let mut content = Self::normalize_content(&response.content);
        content.extend(tool_results.iter().cloned());
        content
    }

    /// Merge usage from all steps.
    ///
    /// This follows Vercel AI SDK's approach: sum all normalized usage fields across steps,
    /// but do not carry provider-native `raw` usage into the aggregated total.
    pub fn merge_usage(steps: &[StepResult]) -> Option<Usage> {
        let mut acc = Usage::unknown();
        let mut saw_usage = false;
        for s in steps.iter() {
            if let Some(u) = &s.usage {
                saw_usage = true;
                acc.merge(u);
            }
        }
        saw_usage.then_some(acc)
    }

    /// Get the concatenated top-level text content from this step.
    ///
    /// This follows the AI SDK `StepResult.text` semantic more closely by joining all top-level
    /// `text` parts from `content` in order. When no explicit text parts exist, it falls back to
    /// the assistant message text if available.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai_extras::orchestrator::StepResult;
    ///
    /// let step: StepResult = /* ... */;
    /// if let Some(text) = step.text() {
    ///     println!("Assistant said: {}", text);
    /// }
    /// ```
    pub fn text(&self) -> Option<String> {
        let text_parts = self
            .content
            .iter()
            .filter_map(|part| part.as_text())
            .collect::<Vec<_>>();

        if !text_parts.is_empty() {
            return Some(text_parts.concat());
        }

        self.messages
            .iter()
            .find(|msg| matches!(msg.role, MessageRole::Assistant))
            .and_then(|msg| msg.content_text().map(ToString::to_string))
    }

    /// Get all unified content parts from this step.
    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    /// Get reasoning and reasoning-file parts from this step.
    pub fn reasoning_parts(&self) -> Vec<&ContentPart> {
        self.content
            .iter()
            .filter(|part| part.is_reasoning())
            .collect()
    }

    /// Get the concatenated textual reasoning from this step.
    pub fn reasoning_text(&self) -> Option<String> {
        let reasoning = self
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Reasoning { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        (!reasoning.is_empty()).then(|| reasoning.join("\n"))
    }

    /// Get regular file parts from this step.
    pub fn files(&self) -> Vec<&ContentPart> {
        self.content.iter().filter(|part| part.is_file()).collect()
    }

    /// Get source parts from this step.
    pub fn sources(&self) -> Vec<&ContentPart> {
        self.content
            .iter()
            .filter(|part| part.is_source())
            .collect()
    }

    /// Get all raw tool-call content parts from this step.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai_extras::orchestrator::StepResult;
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

    fn request_declares_tool_name(&self, tool_name: &str) -> bool {
        self.request.tools.as_ref().is_some_and(|tools| {
            tools.iter().any(|tool| match tool {
                Tool::Function { function } => function.name == tool_name,
                Tool::ProviderDefined(provider_tool) => provider_tool.name == tool_name,
            })
        })
    }

    fn tool_call_view_from_part<'a>(
        &'a self,
        part: &'a ContentPart,
    ) -> Option<StepToolCallView<'a>> {
        let info = part.as_tool_call()?;
        Some(StepToolCallView {
            part,
            tool_call_id: info.tool_call_id,
            tool_name: info.tool_name,
            input: info.input,
            provider_executed: info.provider_executed.copied(),
            dynamic: info
                .dynamic
                .copied()
                .unwrap_or_else(|| !self.request_declares_tool_name(info.tool_name)),
            invalid: info.invalid.copied(),
            error: info.error,
            title: info.title,
        })
    }

    fn resolve_tool_call_input<'a>(&'a self, tool_call_id: &str) -> Option<(&'a Value, bool)> {
        self.tool_calls.iter().find_map(|part| {
            let view = self.tool_call_view_from_part(part)?;
            (view.tool_call_id == tool_call_id).then_some((view.input, view.dynamic))
        })
    }

    fn tool_result_view_from_part<'a>(
        &'a self,
        part: &'a ContentPart,
    ) -> Option<StepToolResultView<'a>> {
        let info = part.as_tool_result()?;
        let (fallback_input, dynamic_from_call) = self
            .resolve_tool_call_input(info.tool_call_id)
            .map_or((None, None), |(input, dynamic)| {
                (Some(input), Some(dynamic))
            });

        Some(StepToolResultView {
            part,
            tool_call_id: info.tool_call_id,
            tool_name: info.tool_name,
            input: info.input.or(fallback_input),
            provider_executed: info.provider_executed.copied(),
            dynamic: info
                .dynamic
                .copied()
                .or(dynamic_from_call)
                .unwrap_or_else(|| !self.request_declares_tool_name(info.tool_name)),
            preliminary: info.preliminary.copied(),
            title: info.title,
        })
    }

    /// Get standardized tool-call projections for this step.
    pub fn tool_call_views(&self) -> Vec<StepToolCallView<'_>> {
        self.tool_calls
            .iter()
            .filter_map(|part| self.tool_call_view_from_part(part))
            .collect()
    }

    /// Get the statically resolvable tool calls for this step.
    pub fn static_tool_calls(&self) -> Vec<StepToolCallView<'_>> {
        self.tool_call_views()
            .into_iter()
            .filter(|call| !call.dynamic)
            .collect()
    }

    /// Get the dynamically resolved tool calls for this step.
    pub fn dynamic_tool_calls(&self) -> Vec<StepToolCallView<'_>> {
        self.tool_call_views()
            .into_iter()
            .filter(|call| call.dynamic)
            .collect()
    }

    /// Check if this step has any tool calls.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai_extras::orchestrator::StepResult;
    ///
    /// let step: StepResult = /* ... */;
    /// if step.has_tool_calls() {
    ///     println!("This step made tool calls");
    /// }
    /// ```
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Get all raw tool-result content parts from this step.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai_extras::orchestrator::StepResult;
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

    /// Get standardized tool-result projections for this step.
    pub fn tool_result_views(&self) -> Vec<StepToolResultView<'_>> {
        self.tool_results
            .iter()
            .filter_map(|part| self.tool_result_view_from_part(part))
            .collect()
    }

    /// Get the statically resolvable tool results for this step.
    pub fn static_tool_results(&self) -> Vec<StepToolResultView<'_>> {
        self.tool_result_views()
            .into_iter()
            .filter(|result| !result.dynamic)
            .collect()
    }

    /// Get the dynamically resolved tool results for this step.
    pub fn dynamic_tool_results(&self) -> Vec<StepToolResultView<'_>> {
        self.tool_result_views()
            .into_iter()
            .filter(|result| result.dynamic)
            .collect()
    }

    /// Check if this step has any tool results.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai_extras::orchestrator::StepResult;
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

/// Final completion event emitted by the orchestrator's `on_finish` callback.
///
/// This mirrors AI SDK's `onFinish` shape at a high level:
/// - `step` is the final step result
/// - `steps` contains the full step history
/// - `total_usage` is the normalized aggregate across all steps
#[derive(Debug, Clone)]
pub struct OrchestratorFinishEvent {
    /// The final chat response returned by the orchestrator.
    pub response: ChatResponse,
    /// The final step result.
    pub step: StepResult,
    /// All steps completed during orchestration.
    pub steps: Vec<StepResult>,
    /// Aggregated normalized usage across all steps.
    pub total_usage: Option<Usage>,
    /// Final runtime context after all steps.
    pub context: OrchestratorContext,
}

impl OrchestratorFinishEvent {
    /// Create a finish event from the final response and all recorded steps.
    pub fn from_response_and_steps(response: ChatResponse, steps: Vec<StepResult>) -> Option<Self> {
        let step = steps.last().cloned()?;
        let total_usage = StepResult::merge_usage(&steps);
        let context = step.context.clone();

        Some(Self {
            response,
            step,
            steps,
            total_usage,
            context,
        })
    }

    /// Get the final text response.
    pub fn text(&self) -> Option<&str> {
        self.response.content_text()
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
    Deny {
        /// Human-readable reason explaining why the tool call was denied.
        reason: String,
    },
}

/// Tool execution result - can be preliminary (intermediate) or final.
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
    /// Return runtime-only metadata for a tool when available.
    fn runtime_tool_metadata(&self, _name: &str) -> Option<ToolRuntimeMetadata> {
        None
    }

    /// Execute a tool by name with structured JSON arguments.
    /// Returns a structured JSON value as tool output.
    ///
    /// This is the simple, non-streaming version. For streaming execution,
    /// override `call_tool_stream` instead.
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError>;

    /// Execute a tool with access to the current orchestration context.
    ///
    /// The default implementation preserves backward compatibility by delegating
    /// to `call_tool`.
    async fn call_tool_with_context(
        &self,
        name: &str,
        arguments: Value,
        _context: &OrchestratorContext,
    ) -> Result<Value, LlmError> {
        self.call_tool(name, arguments).await
    }

    /// Execute a tool with the richer shared execution options.
    ///
    /// Default implementations preserve backward compatibility by forwarding only the
    /// shared context object into `call_tool_with_context(...)`.
    async fn call_tool_with_runtime_options(
        &self,
        name: &str,
        arguments: Value,
        options: ToolExecutionOptions,
    ) -> Result<Value, LlmError> {
        let context = OrchestratorContext::from_map(options.context.into_iter().collect());
        self.call_tool_with_context(name, arguments, &context).await
    }

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

    /// Execute a tool with streaming support and access to the current context.
    ///
    /// The default implementation preserves backward compatibility by delegating
    /// to `call_tool_stream`.
    async fn call_tool_stream_with_context(
        &self,
        name: &str,
        arguments: Value,
        _context: &OrchestratorContext,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        self.call_tool_stream(name, arguments).await
    }

    /// Execute a tool stream with the richer shared execution options.
    ///
    /// Default implementations preserve backward compatibility by forwarding only the
    /// shared context object into `call_tool_stream_with_context(...)`.
    async fn call_tool_stream_with_runtime_options(
        &self,
        name: &str,
        arguments: Value,
        options: ToolExecutionOptions,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        let context = OrchestratorContext::from_map(options.context.into_iter().collect());
        self.call_tool_stream_with_context(name, arguments, &context)
            .await
    }
}

/// Orchestrator options for non-streaming generate.
#[derive(Clone)]
pub struct OrchestratorOptions {
    /// Maximum steps to perform (including the final response step).
    pub max_steps: usize,
    /// Step-finish callback.
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    /// Finish callback with the final response, last step, all steps, and aggregated usage.
    pub on_finish: Option<Arc<dyn Fn(&OrchestratorFinishEvent) + Send + Sync>>,
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
    /// User-defined runtime context flowing through prepare-step and tool execution.
    pub context: OrchestratorContext,
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
            context: OrchestratorContext::default(),
            common_params: None,
        }
    }
}

/// Orchestrator options for streaming generate.
#[derive(Clone)]
pub struct OrchestratorStreamOptions {
    /// Maximum steps to perform (including the final response step).
    pub max_steps: usize,
    /// Optional streaming chunk callback invoked for each `ChatStreamEvent`.
    pub on_chunk: Option<Arc<dyn Fn(&ChatStreamEvent) + Send + Sync>>,
    /// Step-finish callback.
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    /// Finish callback with the final response, last step, all steps, and aggregated usage.
    pub on_finish: Option<Arc<dyn Fn(&OrchestratorFinishEvent) + Send + Sync>>,
    /// Optional tool approval callback. Allows approve/deny/modify tool arguments.
    pub on_tool_approval: Option<Arc<dyn Fn(&str, &Value) -> ToolApproval + Send + Sync>>,
    /// Optional preliminary tool result callback.
    /// Called when a tool returns intermediate results during execution.
    /// Receives: (tool_name, tool_call_id, preliminary_output)
    pub on_preliminary_tool_result: Option<Arc<dyn Fn(&str, &str, &Value) + Send + Sync>>,
    /// Optional prepare step callback for dynamic step configuration.
    pub prepare_step: Option<PrepareStepFn>,
    /// Stop conditions that are evaluated after each streamed step.
    pub stop_conditions: Vec<Arc<dyn StopCondition>>,
    /// Optional abort callback invoked when streaming is cancelled.
    pub on_abort: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    /// Optional telemetry configuration.
    pub telemetry: Option<TelemetryConfig>,
    /// User-defined runtime context flowing through prepare-step and tool execution.
    pub context: OrchestratorContext,
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
            prepare_step: None,
            stop_conditions: Vec::new(),
            on_abort: None,
            telemetry: None,
            context: OrchestratorContext::default(),
            common_params: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OrchestratorContext, ToolResolver};
    use futures::StreamExt;
    use serde_json::json;
    use siumai::tooling::{ToolExecutionOptions, tool};
    use siumai::types::{Tool, UserContent};

    #[tokio::test]
    async fn executable_tools_resolver_preserves_streamed_tool_execution() {
        let tools = siumai::tooling::ExecutableTools::from_tools([tool(Tool::function(
            "search",
            "Search tool",
            json!({ "type": "object" }),
        ))
        .with_execute_stream_fn(|_args, options| {
            assert!(options.context.contains_key("tenant"));
            Box::pin(futures::stream::iter(vec![
                Ok(json!({ "progress": 50 })),
                Ok(json!({ "progress": 100 })),
            ]))
        })]);

        let mut context = OrchestratorContext::default();
        context.insert("tenant", json!("acme"));

        let results = ToolResolver::call_tool_stream_with_context(
            &tools,
            "search",
            json!({ "q": "rust" }),
            &context,
        )
        .await
        .expect("stream tool")
        .collect::<Vec<_>>()
        .await;

        assert_eq!(results.len(), 3);
        assert!(results[0].as_ref().expect("preliminary").is_preliminary());
        assert!(results[1].as_ref().expect("preliminary").is_preliminary());
        assert!(results[2].as_ref().expect("final").is_final());
        assert_eq!(
            results[2].as_ref().expect("final").output(),
            &json!({ "progress": 100 })
        );
    }

    #[tokio::test]
    async fn executable_tools_resolver_passes_runtime_options_through_shared_tooling() {
        let tools = siumai::tooling::ExecutableTools::from_tools([tool(Tool::function(
            "search",
            "Search tool",
            json!({ "type": "object" }),
        ))
        .with_execute_with_options_fn(|args, options| async move {
            assert_eq!(args["q"], json!("rust"));
            assert_eq!(options.tool_call_id, "call_runtime");
            assert_eq!(options.messages.len(), 1);
            assert_eq!(options.context.get("tenant"), Some(&json!("acme")));
            Ok(json!({ "ok": true }))
        })]);

        let result = ToolResolver::call_tool_with_runtime_options(
            &tools,
            "search",
            json!({ "q": "rust" }),
            ToolExecutionOptions::new("call_runtime")
                .with_messages(vec![siumai::types::ModelMessage::User(
                    siumai::types::UserModelMessage::new(UserContent::text("hello")),
                )])
                .with_context(
                    [("tenant".to_string(), json!("acme"))]
                        .into_iter()
                        .collect(),
                ),
        )
        .await
        .expect("execute tool");

        assert_eq!(result, json!({ "ok": true }));
    }
}
