use crate::types::{ModelMessage, SystemPrompt, Tool, ToolResultOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::errors::ai_sdk_error_message;
use super::{
    Context, GenerateTextEndEvent, GenerateTextStartEvent, GenerateTextStepEndEvent,
    GenerateTextStepStartEvent, JSONSchema7, JSONValue, StreamTextChunkEvent, ToolCall, ToolOutput,
};

/// Common model information used across AI SDK callback events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CallbackModelInfo {
    /// Provider identifier.
    pub provider: String,
    /// Model identifier.
    pub model_id: String,
}

impl CallbackModelInfo {
    /// Create callback model information.
    pub fn new(provider: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
        }
    }
}

/// AI SDK tool approval status discriminator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ToolApprovalStatusType {
    /// The tool does not require approval.
    NotApplicable,
    /// The tool is automatically approved.
    Approved,
    /// The tool is automatically denied.
    Denied,
    /// The tool requires user approval.
    UserApproval,
}

/// Object-form AI SDK tool approval status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolApprovalStatusDetails {
    /// Approval status discriminator.
    #[serde(rename = "type")]
    pub status_type: ToolApprovalStatusType,
    /// Optional approval/denial reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl ToolApprovalStatusDetails {
    /// Create a detailed approval status.
    pub const fn new(status_type: ToolApprovalStatusType) -> Self {
        Self {
            status_type,
            reason: None,
        }
    }

    /// Attach an approval/denial reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// AI SDK tool approval status.
///
/// Upstream also treats `undefined` as not-applicable; Rust represents that with
/// `Option<ToolApprovalStatus>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolApprovalStatus {
    /// String status form.
    Simple(ToolApprovalStatusType),
    /// Object status form.
    Detailed(ToolApprovalStatusDetails),
}

impl ToolApprovalStatus {
    /// Create a string-form not-applicable status.
    pub const fn not_applicable() -> Self {
        Self::Simple(ToolApprovalStatusType::NotApplicable)
    }

    /// Create a string-form approved status.
    pub const fn approved() -> Self {
        Self::Simple(ToolApprovalStatusType::Approved)
    }

    /// Create a string-form denied status.
    pub const fn denied() -> Self {
        Self::Simple(ToolApprovalStatusType::Denied)
    }

    /// Create a string-form user-approval status.
    pub const fn user_approval() -> Self {
        Self::Simple(ToolApprovalStatusType::UserApproval)
    }

    /// Create an object-form status with an optional reason.
    pub fn detailed(status_type: ToolApprovalStatusType, reason: Option<String>) -> Self {
        Self::Detailed(ToolApprovalStatusDetails {
            status_type,
            reason,
        })
    }
}

/// Static per-tool approval configuration.
///
/// AI SDK also accepts approval functions. Rust keeps executable callbacks out of
/// the spec layer and exposes the serializable per-tool status map honestly.
pub type ToolApprovalConfiguration = HashMap<String, ToolApprovalStatus>;

/// Passive options passed to a generic AI SDK tool approval function.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolApprovalDecisionContext<NAME = String, INPUT = JSONValue> {
    /// Tool call that needs approval.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Tools available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool context snapshot.
    pub tools_context: Context,
    /// Runtime context snapshot.
    pub runtime_context: Context,
    /// Messages sent to the model before the assistant tool-call response.
    pub messages: Vec<ModelMessage>,
}

/// AI SDK `NoSuchToolError` data carried into tool-call repair callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NoSuchToolError {
    /// Tool name from the failed call.
    pub tool_name: String,
    /// Available tool names when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_tools: Option<Vec<String>>,
    /// Human-readable error message.
    pub message: String,
}

impl NoSuchToolError {
    /// Create a `NoSuchToolError` with the upstream default message shape.
    pub fn new(tool_name: impl Into<String>, available_tools: Option<Vec<String>>) -> Self {
        let tool_name = tool_name.into();
        let message = match available_tools.as_ref() {
            Some(available_tools) => format!(
                "Model tried to call unavailable tool '{tool_name}'. Available tools: {}.",
                available_tools.join(", ")
            ),
            None => format!(
                "Model tried to call unavailable tool '{tool_name}'. No tools are available."
            ),
        };

        Self {
            tool_name,
            available_tools,
            message,
        }
    }
}

/// AI SDK `InvalidToolInputError` data carried into tool-call repair callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidToolInputError {
    /// Tool name from the failed call.
    pub tool_name: String,
    /// Raw tool input text that failed parsing or validation.
    pub tool_input: String,
    /// Human-readable error message.
    pub message: String,
    /// Provider/application error payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl InvalidToolInputError {
    /// Create an invalid-tool-input error.
    pub fn new(
        tool_name: impl Into<String>,
        tool_input: impl Into<String>,
        cause: Option<JSONValue>,
    ) -> Self {
        let tool_name = tool_name.into();
        let message = format!(
            "Invalid input for tool {tool_name}: {}",
            ai_sdk_error_message(cause.as_ref())
        );

        Self {
            tool_name,
            tool_input: tool_input.into(),
            message,
            cause,
        }
    }
}

/// Passive repair error union accepted by AI SDK `ToolCallRepairFunction`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolCallRepairFunctionError {
    /// The requested tool name is not available.
    NoSuchTool(NoSuchToolError),
    /// The tool input failed schema validation or parsing.
    InvalidToolInput(InvalidToolInputError),
}

impl From<NoSuchToolError> for ToolCallRepairFunctionError {
    fn from(error: NoSuchToolError) -> Self {
        Self::NoSuchTool(error)
    }
}

impl From<InvalidToolInputError> for ToolCallRepairFunctionError {
    fn from(error: InvalidToolInputError) -> Self {
        Self::InvalidToolInput(error)
    }
}

/// AI SDK `ToolCallRepairError` wrapper thrown when repair itself fails.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRepairError {
    /// Human-readable error message.
    pub message: String,
    /// Original parse/availability error that triggered repair.
    pub original_error: ToolCallRepairFunctionError,
    /// Provider/application repair failure payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl ToolCallRepairError {
    /// Create a repair error wrapper.
    pub fn new(original_error: ToolCallRepairFunctionError, cause: Option<JSONValue>) -> Self {
        let message = format!(
            "Error repairing tool call: {}",
            ai_sdk_error_message(cause.as_ref())
        );

        Self {
            message,
            original_error,
            cause,
        }
    }
}

/// Passive options passed to AI SDK `ToolCallRepairFunction`.
///
/// Upstream receives an `inputSchema(toolName)` function. Rust stores the known
/// input schemas by tool name instead of pretending to serialize that callback.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRepairContext<NAME = String, INPUT = JSONValue> {
    /// Optional system prompt override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Messages in the current generation step.
    pub messages: Vec<ModelMessage>,
    /// Tool call that failed to parse or validate.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Tools available to the model.
    pub tools: Vec<Tool>,
    /// Input schemas keyed by tool name.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub input_schemas: HashMap<String, JSONSchema7>,
    /// Error that caused repair to be attempted.
    pub error: ToolCallRepairFunctionError,
}

/// Passive repair result returned by an AI SDK `ToolCallRepairFunction`.
pub type ToolCallRepairResult<NAME = String, INPUT = JSONValue> = Option<ToolCall<NAME, INPUT>>;

/// Event passed to AI SDK tool execution start callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionStartEvent<NAME = String, INPUT = JSONValue> {
    /// Unique generation call identifier.
    pub call_id: String,
    /// Messages sent to the model before the assistant tool-call response.
    pub messages: Vec<ModelMessage>,
    /// Tool call that is about to execute.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Validated tool context when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_context: Option<JSONValue>,
}

/// Event passed to AI SDK tool execution end callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionEndEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Unique generation call identifier.
    pub call_id: String,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Messages sent to the model before the assistant tool-call response.
    pub messages: Vec<ModelMessage>,
    /// Tool call that finished executing.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Validated tool context when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_context: Option<JSONValue>,
    /// Successful or failed tool output.
    pub tool_output: ToolOutput<NAME, INPUT, OUTPUT>,
}

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextStartEvent instead.")]
pub type OnStartEvent<OUTPUT = JSONValue> = GenerateTextStartEvent<OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextStepStartEvent instead.")]
pub type OnStepStartEvent<
    OUTPUT = JSONValue,
    NAME = String,
    INPUT = JSONValue,
    ToolOutputValue = ToolResultOutput,
> = GenerateTextStepStartEvent<OUTPUT, NAME, INPUT, ToolOutputValue>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use StreamTextChunkEvent instead.")]
pub type OnChunkEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    StreamTextChunkEvent<NAME, INPUT, OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextStepEndEvent instead.")]
pub type OnStepFinishEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    GenerateTextStepEndEvent<NAME, INPUT, OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextEndEvent instead.")]
pub type OnFinishEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    GenerateTextEndEvent<NAME, INPUT, OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use ToolExecutionStartEvent instead.")]
pub type OnToolCallStartEvent<NAME = String, INPUT = JSONValue> =
    ToolExecutionStartEvent<NAME, INPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use ToolExecutionEndEvent instead.")]
pub type OnToolCallFinishEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolExecutionEndEvent<NAME, INPUT, OUTPUT>;
