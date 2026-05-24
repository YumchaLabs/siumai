use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use serde_json::Value;

use crate::error::LlmError;
use crate::types::{
    CancelHandle, ChatMessage, Context, ModelMessage, ModelMessageConversionError, ToolResultOutput,
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

/// AI SDK-style tool execution function alias.
///
/// This is the Rust facade equivalent of provider-utils `ToolExecuteFunction`: it receives the
/// parsed tool input and the shared `ToolExecutionOptions`, then resolves one final JSON value.
pub type ToolExecuteFunction = ToolExecuteWithOptionsFn;

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

/// Runtime context shared by AI SDK-style tool input-start callbacks.
///
/// This is an alias of the shared execution-options carrier so callback/runtime
/// metadata stays aligned with the actual tool execution contract.
pub type ToolRuntimeContext = ToolExecutionOptions;

/// Runtime context passed when tool-approval policy is evaluated.
#[derive(Debug, Clone, Default)]
pub struct ToolNeedsApprovalContext {
    pub tool_call_id: String,
    pub input: Value,
    pub messages: Vec<ModelMessage>,
    pub context: Context,
}

impl ToolNeedsApprovalContext {
    /// Build approval context from a parsed input plus shared execution options.
    pub fn from_execution_options(input: Value, options: &ToolExecutionOptions) -> Self {
        Self {
            tool_call_id: options.tool_call_id.clone(),
            input,
            messages: options.messages.clone(),
            context: options.context.clone(),
        }
    }
}

/// Runtime context passed when a tool-input delta becomes available.
#[derive(Debug, Clone, Default)]
pub struct ToolInputDeltaContext {
    pub tool_call_id: String,
    pub input_text_delta: String,
    pub messages: Vec<ModelMessage>,
    pub abort_signal: Option<CancelHandle>,
    pub context: Context,
}

impl ToolInputDeltaContext {
    /// Build input-delta context from the shared execution options.
    pub fn from_execution_options(
        input_text_delta: impl Into<String>,
        options: &ToolExecutionOptions,
    ) -> Self {
        Self {
            tool_call_id: options.tool_call_id.clone(),
            input_text_delta: input_text_delta.into(),
            messages: options.messages.clone(),
            abort_signal: options.abort_signal.clone(),
            context: options.context.clone(),
        }
    }
}

/// Runtime context passed when a full tool input becomes available.
#[derive(Debug, Clone, Default)]
pub struct ToolInputAvailableContext {
    pub tool_call_id: String,
    pub input: Value,
    pub messages: Vec<ModelMessage>,
    pub abort_signal: Option<CancelHandle>,
    pub context: Context,
}

impl ToolInputAvailableContext {
    /// Build input-available context from a parsed input plus shared execution options.
    pub fn from_execution_options(input: Value, options: &ToolExecutionOptions) -> Self {
        Self {
            tool_call_id: options.tool_call_id.clone(),
            input,
            messages: options.messages.clone(),
            abort_signal: options.abort_signal.clone(),
            context: options.context.clone(),
        }
    }
}

/// Runtime approval function for AI SDK-style tool execution gating.
pub type ToolNeedsApprovalFn = Arc<
    dyn Fn(ToolNeedsApprovalContext) -> BoxFuture<'static, Result<bool, LlmError>> + Send + Sync,
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
    pub(super) dynamic: bool,
    pub(super) context_schema: Option<Value>,
    pub(super) needs_approval: Option<ToolNeedsApproval>,
    pub(super) on_input_start: Option<ToolInputStartFn>,
    pub(super) on_input_delta: Option<ToolInputDeltaFn>,
    pub(super) on_input_available: Option<ToolInputAvailableFn>,
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
        context: ToolNeedsApprovalContext,
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
