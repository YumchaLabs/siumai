use std::future::Future;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use futures::stream::{self};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::LlmError;
use crate::types::{Tool, ToolResultOutput};

use super::context::{
    ToolExecuteFn, ToolExecuteStreamFn, ToolExecuteWithOptionsFn, ToolExecutionOptions,
    ToolExecutionResult, ToolExecutionStream, ToolInputAvailableContext, ToolInputAvailableFn,
    ToolInputDeltaContext, ToolInputDeltaFn, ToolInputStartFn, ToolModelOutputContext,
    ToolModelOutputFn, ToolNeedsApproval, ToolNeedsApprovalContext, ToolRuntimeContext,
    ToolRuntimeMetadata,
};

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
        F: Fn(ToolNeedsApprovalContext) -> Fut + Send + Sync + 'static,
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
