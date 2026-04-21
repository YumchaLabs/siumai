//! Tool runtime (schema + execution binding).
//!
//! This module provides a single cohesive tool system without introducing additional crates:
//! - `Tool` remains the portable, spec-level schema value (in `siumai-spec`)
//! - `ExecutableTool` binds a `Tool` to an async execution function
//! - `ExecutableTools` provides name-based lookup and execution
//!
//! Higher-level orchestration (multi-step loops, approvals, stop conditions) stays in `siumai-extras`.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::LlmError;
use crate::types::{
    CancelHandle, ChatMessage, Context, ModelMessage, ModelMessageConversionError, Tool,
    ToolResultOutput,
};

/// Async execution function signature for tools.
pub type ToolExecuteFn =
    Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value, LlmError>> + Send + Sync>;

/// Async execution function signature for tools that need AI SDK-style execution options.
pub type ToolExecuteWithOptionsFn = Arc<
    dyn Fn(Value, ToolExecutionOptions) -> BoxFuture<'static, Result<Value, LlmError>>
        + Send
        + Sync,
>;

/// Raw streaming execution output for tools that produce intermediate values.
pub type ToolExecuteValueStream = BoxStream<'static, Result<Value, LlmError>>;

/// Streaming execution function signature for tools that emit raw intermediate values.
///
/// `execute_tool(...)` normalizes this raw stream into `ToolExecutionResult` events by emitting
/// every streamed value as `preliminary` and replaying the last value as `final`.
pub type ToolExecuteStreamFn =
    Arc<dyn Fn(Value, ToolExecutionOptions) -> ToolExecuteValueStream + Send + Sync>;

/// AI SDK-style execution options passed into runtime tool execution helpers.
#[derive(Debug, Clone, Default)]
pub struct ToolExecutionOptions {
    pub tool_call_id: String,
    pub messages: Vec<ModelMessage>,
    pub abort_signal: Option<CancelHandle>,
    pub context: Context,
}

impl ToolExecutionOptions {
    /// Create empty execution options for a tool call id.
    pub fn new(tool_call_id: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            ..Self::default()
        }
    }

    /// Attach prompt/model messages that initiated the tool call.
    pub fn with_messages(mut self, messages: Vec<ModelMessage>) -> Self {
        self.messages = messages;
        self
    }

    /// Convert stable chat messages into shared `ModelMessage` values and attach them.
    pub fn try_with_chat_messages(
        mut self,
        messages: &[ChatMessage],
    ) -> Result<Self, ModelMessageConversionError> {
        self.messages = model_messages_from_chat_messages(messages)?;
        Ok(self)
    }

    /// Attach a cancellation handle that tool implementations may observe.
    pub fn with_abort_signal(mut self, abort_signal: CancelHandle) -> Self {
        self.abort_signal = Some(abort_signal);
        self
    }

    /// Replace the user-defined runtime context object.
    pub fn with_context(mut self, context: Context) -> Self {
        self.context = context;
        self
    }
}

/// Convert stable chat messages into shared `ModelMessage` values.
pub fn model_messages_from_chat_messages(
    messages: &[ChatMessage],
) -> Result<Vec<ModelMessage>, ModelMessageConversionError> {
    messages
        .iter()
        .cloned()
        .map(ModelMessage::try_from)
        .collect()
}

/// Normalized tool execution result for AI SDK-style helper/runtime parity.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolExecutionResult {
    /// Preliminary/intermediate output while the tool is still running.
    Preliminary { output: Value },
    /// Final output of the tool execution.
    Final { output: Value },
}

impl ToolExecutionResult {
    /// Create a preliminary result.
    pub fn preliminary(output: Value) -> Self {
        Self::Preliminary { output }
    }

    /// Create a final result.
    pub fn final_result(output: Value) -> Self {
        Self::Final { output }
    }

    /// Check whether this result is preliminary.
    pub fn is_preliminary(&self) -> bool {
        matches!(self, Self::Preliminary { .. })
    }

    /// Check whether this result is final.
    pub fn is_final(&self) -> bool {
        matches!(self, Self::Final { .. })
    }

    /// Borrow the output payload.
    pub fn output(&self) -> &Value {
        match self {
            Self::Preliminary { output } | Self::Final { output } => output,
        }
    }

    /// Consume the result and return its output payload.
    pub fn into_output(self) -> Value {
        match self {
            Self::Preliminary { output } | Self::Final { output } => output,
        }
    }
}

/// Normalized tool execution stream returned by AI SDK-style helper/runtime wrappers.
pub type ToolExecutionStream = BoxStream<'static, Result<ToolExecutionResult, LlmError>>;

/// Context passed to runtime tool-result model-output mappers.
#[derive(Debug, Clone)]
pub struct ToolModelOutputContext {
    pub tool_call_id: String,
    pub input: Value,
    pub output: Value,
}

/// Runtime tool-result model-output mapping function.
pub type ToolModelOutputFn =
    Arc<dyn Fn(ToolModelOutputContext) -> Result<ToolResultOutput, LlmError> + Send + Sync>;

/// Runtime context shared by AI SDK-style tool callbacks.
#[derive(Debug, Clone, Default)]
pub struct ToolRuntimeContext {
    pub tool_call_id: String,
    pub messages: Vec<ChatMessage>,
    pub context: serde_json::Map<String, Value>,
}

/// Runtime context passed when a tool-input delta becomes available.
#[derive(Debug, Clone, Default)]
pub struct ToolInputDeltaContext {
    pub tool_call_id: String,
    pub input_text_delta: String,
    pub messages: Vec<ChatMessage>,
    pub context: serde_json::Map<String, Value>,
}

/// Runtime context passed when a full tool input becomes available.
#[derive(Debug, Clone, Default)]
pub struct ToolInputAvailableContext {
    pub tool_call_id: String,
    pub input: Value,
    pub messages: Vec<ChatMessage>,
    pub context: serde_json::Map<String, Value>,
}

/// Runtime approval function for AI SDK-style tool execution gating.
pub type ToolNeedsApprovalFn = Arc<
    dyn Fn(ToolInputAvailableContext) -> BoxFuture<'static, Result<bool, LlmError>> + Send + Sync,
>;

/// Runtime callback invoked when streaming tool input starts.
pub type ToolInputStartFn =
    Arc<dyn Fn(ToolRuntimeContext) -> BoxFuture<'static, Result<(), LlmError>> + Send + Sync>;

/// Runtime callback invoked when a streaming tool-input delta arrives.
pub type ToolInputDeltaFn =
    Arc<dyn Fn(ToolInputDeltaContext) -> BoxFuture<'static, Result<(), LlmError>> + Send + Sync>;

/// Runtime callback invoked when a full tool input becomes available.
pub type ToolInputAvailableFn = Arc<
    dyn Fn(ToolInputAvailableContext) -> BoxFuture<'static, Result<(), LlmError>> + Send + Sync,
>;

/// Runtime approval policy for a tool.
#[derive(Clone)]
pub enum ToolNeedsApproval {
    /// Always require approval before execution.
    Always,
    /// Decide at runtime using the provided callback.
    Check(ToolNeedsApprovalFn),
}

impl std::fmt::Debug for ToolNeedsApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => f.write_str("Always"),
            Self::Check(_) => f.write_str("Check(..)"),
        }
    }
}

/// Runtime-only AI SDK tool metadata that should not leak into the stable wire schema.
#[derive(Clone, Default)]
pub struct ToolRuntimeMetadata {
    dynamic: bool,
    context_schema: Option<Value>,
    needs_approval: Option<ToolNeedsApproval>,
    on_input_start: Option<ToolInputStartFn>,
    on_input_delta: Option<ToolInputDeltaFn>,
    on_input_available: Option<ToolInputAvailableFn>,
}

impl std::fmt::Debug for ToolRuntimeMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRuntimeMetadata")
            .field("dynamic", &self.dynamic)
            .field("has_context_schema", &self.context_schema.is_some())
            .field("has_needs_approval", &self.needs_approval.is_some())
            .field("has_on_input_start", &self.on_input_start.is_some())
            .field("has_on_input_delta", &self.on_input_delta.is_some())
            .field("has_on_input_available", &self.on_input_available.is_some())
            .finish()
    }
}

impl ToolRuntimeMetadata {
    /// Whether the tool is dynamic/runtime-defined.
    pub const fn dynamic(&self) -> bool {
        self.dynamic
    }

    /// Optional context schema metadata carried for type/system parity.
    pub fn context_schema(&self) -> Option<&Value> {
        self.context_schema.as_ref()
    }

    /// Whether this tool has any approval gating configured.
    pub const fn has_needs_approval(&self) -> bool {
        self.needs_approval.is_some()
    }

    /// Whether this tool has an input-start callback.
    pub const fn has_on_input_start(&self) -> bool {
        self.on_input_start.is_some()
    }

    /// Whether this tool has an input-delta callback.
    pub const fn has_on_input_delta(&self) -> bool {
        self.on_input_delta.is_some()
    }

    /// Whether this tool has an input-available callback.
    pub const fn has_on_input_available(&self) -> bool {
        self.on_input_available.is_some()
    }

    /// Evaluate whether this tool requires approval for the given input.
    pub async fn needs_approval(
        &self,
        context: ToolInputAvailableContext,
    ) -> Result<bool, LlmError> {
        match &self.needs_approval {
            None => Ok(false),
            Some(ToolNeedsApproval::Always) => Ok(true),
            Some(ToolNeedsApproval::Check(callback)) => callback(context).await,
        }
    }

    /// Invoke the input-start callback when configured.
    pub async fn invoke_on_input_start(&self, context: ToolRuntimeContext) -> Result<(), LlmError> {
        match &self.on_input_start {
            Some(callback) => callback(context).await,
            None => Ok(()),
        }
    }

    /// Invoke the input-delta callback when configured.
    pub async fn invoke_on_input_delta(
        &self,
        context: ToolInputDeltaContext,
    ) -> Result<(), LlmError> {
        match &self.on_input_delta {
            Some(callback) => callback(context).await,
            None => Ok(()),
        }
    }

    /// Invoke the input-available callback when configured.
    pub async fn invoke_on_input_available(
        &self,
        context: ToolInputAvailableContext,
    ) -> Result<(), LlmError> {
        match &self.on_input_available {
            Some(callback) => callback(context).await,
            None => Ok(()),
        }
    }
}

/// A tool definition with an optional bound executor.
#[derive(Clone)]
pub struct ExecutableTool {
    tool: Tool,
    execute: Option<ToolExecuteFn>,
    execute_with_options: Option<ToolExecuteWithOptionsFn>,
    execute_stream: Option<ToolExecuteStreamFn>,
    to_model_output: Option<ToolModelOutputFn>,
    runtime_metadata: ToolRuntimeMetadata,
}

impl std::fmt::Debug for ExecutableTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutableTool")
            .field("name", &self.name())
            .field("has_execute", &self.has_execute())
            .field(
                "has_execute_with_options",
                &self.execute_with_options.is_some(),
            )
            .field("has_execute_stream", &self.execute_stream.is_some())
            .field("has_to_model_output", &self.to_model_output.is_some())
            .field("runtime_metadata", &self.runtime_metadata)
            .finish()
    }
}

impl ExecutableTool {
    /// Create a tool wrapper without an executor.
    pub fn new(tool: Tool) -> Self {
        Self {
            tool,
            execute: None,
            execute_with_options: None,
            execute_stream: None,
            to_model_output: None,
            runtime_metadata: ToolRuntimeMetadata::default(),
        }
    }

    /// Bind an executor to an existing tool schema.
    pub fn with_execute(mut self, execute: ToolExecuteFn) -> Self {
        self.execute_with_options = None;
        self.execute_stream = None;
        self.execute = Some(execute);
        self
    }

    /// Bind an executor that receives AI SDK-style execution options.
    pub fn with_execute_with_options(mut self, execute: ToolExecuteWithOptionsFn) -> Self {
        self.execute = None;
        self.execute_stream = None;
        self.execute_with_options = Some(execute);
        self
    }

    /// Bind a streaming executor that emits raw intermediate values.
    pub fn with_execute_stream(mut self, execute: ToolExecuteStreamFn) -> Self {
        self.execute = None;
        self.execute_with_options = None;
        self.execute_stream = Some(execute);
        self
    }

    /// Bind an executor that receives AI SDK-style execution options.
    pub fn with_execute_with_options_fn<F, Fut>(mut self, execute: F) -> Self
    where
        F: Fn(Value, ToolExecutionOptions) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, LlmError>> + Send + 'static,
    {
        self.execute = None;
        self.execute_stream = None;
        self.execute_with_options = Some(Arc::new(move |args, options| {
            Box::pin(execute(args, options))
        }));
        self
    }

    /// Bind a streaming executor that emits raw intermediate values.
    pub fn with_execute_stream_fn<F, S>(mut self, execute: F) -> Self
    where
        F: Fn(Value, ToolExecutionOptions) -> S + Send + Sync + 'static,
        S: futures::Stream<Item = Result<Value, LlmError>> + Send + 'static,
    {
        self.execute = None;
        self.execute_with_options = None;
        self.execute_stream = Some(Arc::new(move |args, options| {
            Box::pin(execute(args, options))
        }));
        self
    }

    /// Bind a runtime tool-result model-output mapper to an existing tool schema.
    pub fn with_to_model_output(mut self, to_model_output: ToolModelOutputFn) -> Self {
        self.to_model_output = Some(to_model_output);
        self
    }

    /// Mark the tool as dynamic/runtime-defined for AI SDK parity.
    pub fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.runtime_metadata.dynamic = dynamic;
        self
    }

    /// Carry context-schema metadata for runtime parity without enforcing validation.
    pub fn with_context_schema(mut self, context_schema: Value) -> Self {
        self.runtime_metadata.context_schema = Some(context_schema);
        self
    }

    /// Configure whether this tool requires approval before execution.
    pub fn with_needs_approval(mut self, needs_approval: bool) -> Self {
        self.runtime_metadata.needs_approval = needs_approval.then_some(ToolNeedsApproval::Always);
        self
    }

    /// Configure a runtime approval predicate for this tool.
    pub fn with_needs_approval_fn<F, Fut>(mut self, needs_approval: F) -> Self
    where
        F: Fn(ToolInputAvailableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<bool, LlmError>> + Send + 'static,
    {
        self.runtime_metadata.needs_approval =
            Some(ToolNeedsApproval::Check(Arc::new(move |context| {
                Box::pin(needs_approval(context))
            })));
        self
    }

    /// Configure a callback invoked when streaming tool input starts.
    pub fn with_on_input_start(mut self, on_input_start: ToolInputStartFn) -> Self {
        self.runtime_metadata.on_input_start = Some(on_input_start);
        self
    }

    /// Configure a callback invoked when a streaming tool-input delta arrives.
    pub fn with_on_input_delta(mut self, on_input_delta: ToolInputDeltaFn) -> Self {
        self.runtime_metadata.on_input_delta = Some(on_input_delta);
        self
    }

    /// Configure a callback invoked when a full tool input becomes available.
    pub fn with_on_input_available(mut self, on_input_available: ToolInputAvailableFn) -> Self {
        self.runtime_metadata.on_input_available = Some(on_input_available);
        self
    }

    /// Configure a callback invoked when streaming tool input starts.
    pub fn with_on_input_start_fn<F, Fut>(mut self, on_input_start: F) -> Self
    where
        F: Fn(ToolRuntimeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), LlmError>> + Send + 'static,
    {
        self.runtime_metadata.on_input_start =
            Some(Arc::new(move |context| Box::pin(on_input_start(context))));
        self
    }

    /// Configure a callback invoked when a streaming tool-input delta arrives.
    pub fn with_on_input_delta_fn<F, Fut>(mut self, on_input_delta: F) -> Self
    where
        F: Fn(ToolInputDeltaContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), LlmError>> + Send + 'static,
    {
        self.runtime_metadata.on_input_delta =
            Some(Arc::new(move |context| Box::pin(on_input_delta(context))));
        self
    }

    /// Configure a callback invoked when a full tool input becomes available.
    pub fn with_on_input_available_fn<F, Fut>(mut self, on_input_available: F) -> Self
    where
        F: Fn(ToolInputAvailableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), LlmError>> + Send + 'static,
    {
        self.runtime_metadata.on_input_available = Some(Arc::new(move |context| {
            Box::pin(on_input_available(context))
        }));
        self
    }

    /// Replace the function input schema on the portable tool definition.
    pub fn with_input_schema(mut self, input_schema: Value) -> Self {
        self.tool = self.tool.with_input_schema(input_schema);
        self
    }

    /// Attach AI SDK-style function output schema metadata to the portable tool definition.
    pub fn with_output_schema(mut self, output_schema: Value) -> Self {
        self.tool = self.tool.with_output_schema(output_schema);
        self
    }

    /// Bind a runtime tool-result model-output mapper from a closure.
    pub fn with_to_model_output_fn<F>(mut self, to_model_output: F) -> Self
    where
        F: Fn(ToolModelOutputContext) -> Result<ToolResultOutput, LlmError> + Send + Sync + 'static,
    {
        self.to_model_output = Some(Arc::new(to_model_output));
        self
    }

    /// Create a JSON-based function tool with an executor.
    pub fn function<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, LlmError>> + Send + 'static,
    {
        let tool = Tool::function(name, description, parameters);
        let exec: ToolExecuteFn = Arc::new(move |args: Value| Box::pin(execute(args)));
        Self {
            tool,
            execute: Some(exec),
            execute_with_options: None,
            execute_stream: None,
            to_model_output: None,
            runtime_metadata: ToolRuntimeMetadata::default(),
        }
    }

    /// Create a JSON-based function tool with an executor and output schema metadata.
    pub fn function_with_output_schema<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        output_schema: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, LlmError>> + Send + 'static,
    {
        Self::function(name, description, input_schema, execute).with_output_schema(output_schema)
    }

    /// Create a typed function tool.
    ///
    /// `TArgs` is deserialized from JSON tool call arguments.
    /// `TOut` is serialized into JSON tool result output.
    pub fn typed_function<TArgs, TOut, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: F,
    ) -> Self
    where
        TArgs: DeserializeOwned + Send + 'static,
        TOut: Serialize + Send + 'static,
        F: Fn(TArgs) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<TOut, LlmError>> + Send + 'static,
    {
        let tool = Tool::function(name, description, parameters);
        let exec: ToolExecuteFn = Arc::new(move |args: Value| {
            let parsed: Result<TArgs, LlmError> = serde_json::from_value(args).map_err(|e| {
                LlmError::InvalidParameter(format!("Failed to parse tool arguments: {e}"))
            });
            match parsed {
                Ok(parsed) => {
                    let fut = execute(parsed);
                    Box::pin(async move {
                        let out = fut.await?;
                        serde_json::to_value(out).map_err(|e| {
                            LlmError::InternalError(format!(
                                "Failed to serialize tool output as JSON: {e}"
                            ))
                        })
                    })
                }
                Err(e) => Box::pin(async move { Err(e) }),
            }
        });

        Self {
            tool,
            execute: Some(exec),
            execute_with_options: None,
            execute_stream: None,
            to_model_output: None,
            runtime_metadata: ToolRuntimeMetadata::default(),
        }
    }

    /// Create a typed function tool with AI SDK-style output schema metadata.
    pub fn typed_function_with_output_schema<TArgs, TOut, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        output_schema: Value,
        execute: F,
    ) -> Self
    where
        TArgs: DeserializeOwned + Send + 'static,
        TOut: Serialize + Send + 'static,
        F: Fn(TArgs) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<TOut, LlmError>> + Send + 'static,
    {
        Self::typed_function::<TArgs, TOut, _, _>(name, description, input_schema, execute)
            .with_output_schema(output_schema)
    }

    /// Return the portable tool schema (for sending to the model).
    pub const fn tool(&self) -> &Tool {
        &self.tool
    }

    /// Tool name used in tool calls.
    pub fn name(&self) -> &str {
        match &self.tool {
            Tool::Function { function } => function.name.as_str(),
            Tool::ProviderDefined(t) => t.name.as_str(),
        }
    }

    /// Access runtime-only AI SDK-style tool metadata.
    pub const fn runtime_metadata(&self) -> &ToolRuntimeMetadata {
        &self.runtime_metadata
    }

    /// Whether this tool exposes any executable runtime binding.
    pub const fn has_execute(&self) -> bool {
        self.execute.is_some()
            || self.execute_with_options.is_some()
            || self.execute_stream.is_some()
    }

    /// Execute the tool as a normalized preliminary/final stream with AI SDK-style options.
    pub async fn execute_stream(
        &self,
        args: Value,
        options: ToolExecutionOptions,
    ) -> Result<ToolExecutionStream, LlmError> {
        execute_tool(self, args, options).await
    }

    /// Execute the tool with JSON arguments.
    pub async fn execute_json(&self, args: Value) -> Result<Value, LlmError> {
        self.execute_json_with_options(args, ToolExecutionOptions::default())
            .await
    }

    /// Execute the tool with JSON arguments and AI SDK-style execution options.
    pub async fn execute_json_with_options(
        &self,
        args: Value,
        options: ToolExecutionOptions,
    ) -> Result<Value, LlmError> {
        let mut stream = self.execute_stream(args, options).await?;
        let mut final_output = None;

        while let Some(item) = stream.next().await {
            match item? {
                ToolExecutionResult::Final { output } => final_output = Some(output),
                ToolExecutionResult::Preliminary { .. } => {}
            }
        }

        final_output.ok_or_else(|| {
            LlmError::InternalError(format!(
                "Tool '{}' did not emit a final execution result.",
                self.name()
            ))
        })
    }

    /// Convert a runtime tool result into a stable model-facing output.
    pub fn to_model_output(
        &self,
        context: ToolModelOutputContext,
    ) -> Result<Option<ToolResultOutput>, LlmError> {
        match &self.to_model_output {
            Some(mapper) => mapper(context).map(Some),
            None => Ok(None),
        }
    }
}

/// Alias aligned with AI SDK `ToolSet`.
pub type ToolSet = ExecutableTools;

/// AI SDK-style helper for wrapping a portable `Tool` into an executable runtime carrier.
pub fn tool(tool: impl Into<ExecutableTool>) -> ExecutableTool {
    tool.into()
}

/// AI SDK-style helper for marking a runtime-defined tool.
pub fn dynamic_tool(tool: impl Into<ExecutableTool>) -> ExecutableTool {
    tool.into().with_dynamic(true)
}

/// AI SDK-style helper for checking whether a tool has an execute binding.
pub fn is_executable_tool(tool: Option<&ExecutableTool>) -> bool {
    tool.is_some_and(ExecutableTool::has_execute)
}

/// Execute a tool and normalize its outputs into preliminary/final events.
pub async fn execute_tool(
    tool: &ExecutableTool,
    input: Value,
    options: ToolExecutionOptions,
) -> Result<ToolExecutionStream, LlmError> {
    if let Some(execute_stream) = &tool.execute_stream {
        let tool_name = tool.name().to_string();
        let mut stream = execute_stream(input, options);

        return Ok(Box::pin(try_stream! {
            let mut last_output = None;

            while let Some(output) = stream.next().await {
                let output = output?;
                last_output = Some(output.clone());
                yield ToolExecutionResult::preliminary(output);
            }

            let final_output = last_output.ok_or_else(|| {
                LlmError::InternalError(format!(
                    "Tool '{}' returned an empty execution stream.",
                    tool_name
                ))
            })?;

            yield ToolExecutionResult::final_result(final_output);
        }));
    }

    if let Some(execute_with_options) = &tool.execute_with_options {
        let output = execute_with_options(input, options).await?;
        return Ok(Box::pin(stream::once(async move {
            Ok(ToolExecutionResult::final_result(output))
        })));
    }

    if let Some(execute) = &tool.execute {
        let output = execute(input).await?;
        return Ok(Box::pin(stream::once(async move {
            Ok(ToolExecutionResult::final_result(output))
        })));
    }

    Err(LlmError::UnsupportedOperation(format!(
        "Tool '{}' does not have an executor bound.",
        tool.name()
    )))
}

impl From<Tool> for ExecutableTool {
    fn from(tool: Tool) -> Self {
        Self::new(tool)
    }
}

/// A collection of executable tools with name-based lookup.
#[derive(Clone, Default)]
pub struct ExecutableTools {
    tools: Vec<ExecutableTool>,
    index_by_name: HashMap<String, usize>,
}

impl std::fmt::Debug for ExecutableTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutableTools")
            .field("len", &self.tools.len())
            .finish()
    }
}

impl ExecutableTools {
    /// Create an empty tool collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from an iterator of tools. Later duplicates override earlier ones by name.
    pub fn from_tools(tools: impl IntoIterator<Item = ExecutableTool>) -> Self {
        let mut out = Self::new();
        for tool in tools {
            out.insert(tool);
        }
        out
    }

    /// Insert (or replace) a tool by name.
    pub fn insert(&mut self, tool: ExecutableTool) {
        let name = tool.name().to_string();
        if let Some(&idx) = self.index_by_name.get(&name) {
            self.tools[idx] = tool;
            return;
        }
        let idx = self.tools.len();
        self.tools.push(tool);
        self.index_by_name.insert(name, idx);
    }

    /// Return tool schemas for model calls.
    pub fn schemas(&self) -> Vec<Tool> {
        self.tools.iter().map(|t| t.tool().clone()).collect()
    }

    /// Find a tool by name.
    pub fn get(&self, name: &str) -> Option<&ExecutableTool> {
        let idx = self.index_by_name.get(name).copied()?;
        self.tools.get(idx)
    }

    /// Return runtime-only AI SDK-style metadata by tool name.
    pub fn runtime_metadata(&self, name: &str) -> Option<ToolRuntimeMetadata> {
        self.get(name).map(|tool| tool.runtime_metadata().clone())
    }

    /// Execute a tool by name.
    pub async fn execute(&self, name: &str, args: Value) -> Result<Value, LlmError> {
        self.execute_with_options(name, args, ToolExecutionOptions::default())
            .await
    }

    /// Execute a tool by name with AI SDK-style execution options.
    pub async fn execute_with_options(
        &self,
        name: &str,
        args: Value,
        options: ToolExecutionOptions,
    ) -> Result<Value, LlmError> {
        let tool = self
            .get(name)
            .ok_or_else(|| LlmError::NotFound(format!("Tool not found: '{name}'")))?;
        tool.execute_json_with_options(args, options).await
    }

    /// Execute a tool by name as a normalized preliminary/final stream.
    pub async fn execute_stream(
        &self,
        name: &str,
        args: Value,
        options: ToolExecutionOptions,
    ) -> Result<ToolExecutionStream, LlmError> {
        let tool = self
            .get(name)
            .ok_or_else(|| LlmError::NotFound(format!("Tool not found: '{name}'")))?;
        tool.execute_stream(args, options).await
    }

    /// Convert a runtime tool result into a stable model-facing output by tool name.
    pub fn to_model_output(
        &self,
        name: &str,
        context: ToolModelOutputContext,
    ) -> Result<Option<ToolResultOutput>, LlmError> {
        match self.get(name) {
            Some(tool) => tool.to_model_output(context),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[tokio::test]
    async fn typed_tool_parses_args_and_serializes_output() {
        #[derive(Deserialize)]
        struct Args {
            x: i64,
            y: i64,
        }

        #[derive(Serialize)]
        struct Out {
            sum: i64,
        }

        let tool = ExecutableTool::typed_function::<Args, Out, _, _>(
            "add",
            "Add two integers",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "x": { "type": "integer" },
                    "y": { "type": "integer" }
                },
                "required": ["x", "y"]
            }),
            |args| async move {
                Ok(Out {
                    sum: args.x + args.y,
                })
            },
        );

        let out = tool
            .execute_json(serde_json::json!({ "x": 1, "y": 2 }))
            .await
            .unwrap();

        assert_eq!(out, serde_json::json!({ "sum": 3 }));
    }

    #[tokio::test]
    async fn tool_set_executes_by_name() {
        let mut tools = ExecutableTools::new();
        tools.insert(ExecutableTool::function(
            "echo",
            "Echo input",
            serde_json::json!({"type":"object"}),
            |v| async move { Ok(v) },
        ));

        let out = tools
            .execute("echo", serde_json::json!({"a":1}))
            .await
            .unwrap();
        assert_eq!(out, serde_json::json!({"a":1}));
    }

    #[tokio::test]
    async fn execute_tool_normalizes_streaming_outputs() {
        let tool = tool(Tool::function(
            "search",
            "Search tool",
            serde_json::json!({ "type": "object" }),
        ))
        .with_execute_stream_fn(|_args, options| {
            assert_eq!(options.tool_call_id, "call_1");
            Box::pin(futures::stream::iter(vec![
                Ok(serde_json::json!({ "progress": 50 })),
                Ok(serde_json::json!({ "progress": 100 })),
            ]))
        });

        let results = execute_tool(
            &tool,
            serde_json::json!({ "q": "rust" }),
            ToolExecutionOptions::new("call_1"),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].as_ref().unwrap(),
            &ToolExecutionResult::preliminary(serde_json::json!({ "progress": 50 }))
        );
        assert_eq!(
            results[1].as_ref().unwrap(),
            &ToolExecutionResult::preliminary(serde_json::json!({ "progress": 100 }))
        );
        assert_eq!(
            results[2].as_ref().unwrap(),
            &ToolExecutionResult::final_result(serde_json::json!({ "progress": 100 }))
        );
    }

    #[tokio::test]
    async fn tool_set_executes_streams_with_options() {
        let tools = ExecutableTools::from_tools([tool(Tool::function(
            "search",
            "Search tool",
            serde_json::json!({ "type": "object" }),
        ))
        .with_execute_with_options_fn(|args, options| async move {
            assert_eq!(args["q"], serde_json::json!("rust"));
            assert_eq!(options.tool_call_id, "call_2");
            Ok(serde_json::json!({ "ok": true }))
        })]);

        let out = tools
            .execute_with_options(
                "search",
                serde_json::json!({ "q": "rust" }),
                ToolExecutionOptions::new("call_2"),
            )
            .await
            .unwrap();

        assert_eq!(out, serde_json::json!({ "ok": true }));
    }

    #[test]
    fn tool_execution_options_can_project_chat_messages() {
        let options = ToolExecutionOptions::new("call_3")
            .try_with_chat_messages(&[crate::types::ChatMessage::user("hello").build()])
            .expect("chat messages should convert");

        assert_eq!(options.messages.len(), 1);
        assert!(matches!(
            options.messages[0],
            crate::types::ModelMessage::User(_)
        ));
    }

    #[test]
    fn tool_builder_keeps_output_schema_on_portable_function_tool() {
        #[derive(Deserialize)]
        struct Args {
            city: String,
        }

        #[derive(Serialize)]
        struct Out {
            forecast: String,
        }

        let tool = ExecutableTool::typed_function_with_output_schema::<Args, Out, _, _>(
            "weather",
            "Weather tool",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "forecast": { "type": "string" }
                },
                "required": ["forecast"]
            }),
            |args| async move {
                Ok(Out {
                    forecast: format!("sunny:{}", args.city),
                })
            },
        );

        let schema = tool
            .tool()
            .output_schema()
            .expect("output schema should be attached");

        assert_eq!(
            schema,
            &serde_json::json!({
                "type": "object",
                "properties": {
                    "forecast": { "type": "string" }
                },
                "required": ["forecast"]
            })
        );
    }

    #[test]
    fn tool_set_uses_runtime_model_output_mapper() {
        let tools = ExecutableTools::from_tools([ExecutableTool::new(Tool::function(
            "weather",
            "Weather tool",
            serde_json::json!({ "type": "object" }),
        ))
        .with_to_model_output_fn(|ctx| {
            Ok(ToolResultOutput::content(vec![
                crate::types::ToolResultContentPart::text(format!(
                    "{}:{}",
                    ctx.tool_call_id, ctx.output["temp"]
                )),
            ]))
        })]);

        let output = tools
            .to_model_output(
                "weather",
                ToolModelOutputContext {
                    tool_call_id: "call_1".to_string(),
                    input: serde_json::json!({ "city": "Tokyo" }),
                    output: serde_json::json!({ "temp": 18 }),
                },
            )
            .expect("map ok")
            .expect("mapper should exist");

        assert_eq!(
            output,
            ToolResultOutput::content(vec![crate::types::ToolResultContentPart::text("call_1:18")])
        );
    }

    #[tokio::test]
    async fn tool_runtime_metadata_supports_callbacks_and_dynamic_flags() {
        let tool = ExecutableTool::function(
            "dangerous",
            "Dangerous tool",
            serde_json::json!({ "type": "object" }),
            |args| async move { Ok(args) },
        )
        .with_dynamic(true)
        .with_context_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "role": { "type": "string" }
            }
        }))
        .with_needs_approval_fn(|context| async move {
            Ok(context
                .context
                .get("role")
                .and_then(|value| value.as_str())
                .is_some_and(|role| role != "admin"))
        })
        .with_on_input_available_fn(|_context| async move { Ok(()) });

        let metadata = tool.runtime_metadata();
        assert!(metadata.dynamic());
        assert!(metadata.context_schema().is_some());
        assert!(metadata.has_needs_approval());
        assert!(metadata.has_on_input_available());
        assert!(
            metadata
                .needs_approval(ToolInputAvailableContext {
                    tool_call_id: "call_1".to_string(),
                    input: serde_json::json!({}),
                    messages: Vec::new(),
                    context: serde_json::Map::from_iter([(
                        "role".to_string(),
                        serde_json::json!("viewer"),
                    )]),
                })
                .await
                .unwrap()
        );
    }

    #[test]
    fn executable_tool_helpers_match_ai_sdk_style_facade() {
        let executable = tool(Tool::function(
            "weather",
            "Weather tool",
            serde_json::json!({ "type": "object" }),
        ));
        assert!(!is_executable_tool(Some(&executable)));

        let executable =
            executable.with_execute(Arc::new(|args| Box::pin(async move { Ok(args) })));
        assert!(is_executable_tool(Some(&executable)));
        assert!(
            dynamic_tool(executable.clone())
                .runtime_metadata()
                .dynamic()
        );
    }
}
