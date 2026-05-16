use crate::types::FlexibleSchema;
use crate::types::chat::{
    UiCustomPart, UiDataPart, UiFilePart, UiMessage, UiMessagePart, UiMessageRole,
    UiReasoningFilePart, UiReasoningPart, UiSourceDocumentPart, UiSourceUrlPart, UiTextPart,
    UiToolInvocation, UiToolKind, UiToolPart, UiToolPartState,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{JSONValue, ToolCall};

/// Headers used by AI SDK UI message streams.
pub const UI_MESSAGE_STREAM_HEADERS: &[(&str, &str)] = &[
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache"),
    ("connection", "keep-alive"),
    ("x-vercel-ai-ui-message-stream", "v1"),
    ("x-accel-buffering", "no"),
];

/// AI SDK UI data-part schema map.
pub type UIDataPartSchemas = HashMap<String, FlexibleSchema<JSONValue>>;

/// AI SDK UI data type to schema map. Rust keeps this as the same runtime map.
pub type UIDataTypesToSchemas = UIDataPartSchemas;

/// AI SDK inferred UI data parts. Rust exposes the resolved JSON-value map directly.
pub type InferUIDataParts = HashMap<String, JSONValue>;

/// AI SDK `UIDataTypes` map.
pub type UIDataTypes = HashMap<String, JSONValue>;

/// AI SDK inferred UI message metadata. Rust resolves the generic helper to JSON values.
pub type InferUIMessageMetadata = JSONValue;

/// AI SDK inferred UI message data map. Rust resolves the generic helper to `UIDataTypes`.
pub type InferUIMessageData = UIDataTypes;

/// AI SDK inferred UI message tools map. Rust resolves the generic helper to `UITools`.
pub type InferUIMessageTools = UITools;

/// AI SDK inferred UI message tool outputs. Rust resolves the generic helper to JSON values.
pub type InferUIMessageToolOutputs = JSONValue;

/// AI SDK inferred UI message tool call.
pub type InferUIMessageToolCall = ToolCall<String, JSONValue>;

/// AI SDK inferred UI message part.
pub type InferUIMessagePart = UiMessagePart;

/// AI SDK `UITool` passive input/output carrier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UITool {
    /// Tool input value.
    pub input: JSONValue,
    /// Tool output value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<JSONValue>,
}

/// AI SDK inferred UI tool carrier. Rust exposes the resolved JSON-value tool shape directly.
pub type InferUITool = UITool;

/// AI SDK `UITools` map.
pub type UITools = HashMap<String, UITool>;

/// AI SDK inferred UI tools map. Rust exposes the resolved JSON-value tool map directly.
pub type InferUITools = UITools;

/// AI SDK-compatible alias for `UIMessage`.
pub type UIMessage = UiMessage;

/// AI SDK-compatible alias for `UIMessagePart`.
pub type UIMessagePart = UiMessagePart;

/// AI SDK-compatible alias for `TextUIPart`.
pub type TextUIPart = UiTextPart;

/// AI SDK-compatible alias for `CustomContentUIPart`.
pub type CustomContentUIPart = UiCustomPart;

/// AI SDK-compatible alias for `ReasoningUIPart`.
pub type ReasoningUIPart = UiReasoningPart;

/// AI SDK-compatible alias for `FileUIPart`.
pub type FileUIPart = UiFilePart;

/// AI SDK-compatible alias for `ReasoningFileUIPart`.
pub type ReasoningFileUIPart = UiReasoningFilePart;

/// AI SDK-compatible alias for `SourceUrlUIPart`.
pub type SourceUrlUIPart = UiSourceUrlPart;

/// AI SDK-compatible alias for `SourceDocumentUIPart`.
pub type SourceDocumentUIPart = UiSourceDocumentPart;

/// AI SDK-compatible alias for `DataUIPart`.
pub type DataUIPart = UiDataPart;

/// AI SDK-compatible alias for `ToolUIPart`.
pub type ToolUIPart = UiToolPart;

/// AI SDK-compatible alias for `DynamicToolUIPart`.
pub type DynamicToolUIPart = UiToolPart;

/// AI SDK-compatible alias for `UIToolInvocation`.
pub type UIToolInvocation = UiToolInvocation;

/// AI SDK-compatible alias for `StepStartUIPart`.
///
/// Siumai represents this as the `UiMessagePart::StepStart` unit variant.
pub type StepStartUIPart = UiMessagePart;

/// Check whether a UI message part is a text part.
pub fn is_text_ui_part(part: &UiMessagePart) -> bool {
    matches!(part, UiMessagePart::Text(_))
}

/// Check whether a UI message part is a custom content part.
pub fn is_custom_content_ui_part(part: &UiMessagePart) -> bool {
    matches!(part, UiMessagePart::Custom(_))
}

/// Check whether a UI message part is a file part.
pub fn is_file_ui_part(part: &UiMessagePart) -> bool {
    matches!(part, UiMessagePart::File(_))
}

/// Check whether a UI message part is a reasoning-file part.
pub fn is_reasoning_file_ui_part(part: &UiMessagePart) -> bool {
    matches!(part, UiMessagePart::ReasoningFile(_))
}

/// Check whether a UI message part is a reasoning part.
pub fn is_reasoning_ui_part(part: &UiMessagePart) -> bool {
    matches!(part, UiMessagePart::Reasoning(_))
}

/// Check whether a UI message part is a data part.
pub fn is_data_ui_part(part: &UiMessagePart) -> bool {
    matches!(part, UiMessagePart::Data(_))
}

/// Check whether a UI message part is a static tool part.
pub fn is_static_tool_ui_part(part: &UiMessagePart) -> bool {
    matches!(
        part,
        UiMessagePart::Tool(UiToolPart {
            kind: UiToolKind::Static { .. },
            ..
        })
    )
}

/// Check whether a UI message part is a dynamic tool part.
pub fn is_dynamic_tool_ui_part(part: &UiMessagePart) -> bool {
    matches!(
        part,
        UiMessagePart::Tool(UiToolPart {
            kind: UiToolKind::Dynamic { .. },
            ..
        })
    )
}

/// Check whether a UI message part is any tool part.
pub fn is_tool_ui_part(part: &UiMessagePart) -> bool {
    matches!(part, UiMessagePart::Tool(_))
}

/// Return the static tool name for a `tool-*` UI part.
pub fn get_static_tool_name(part: &UiMessagePart) -> Option<&str> {
    match part {
        UiMessagePart::Tool(tool_part) => match &tool_part.kind {
            UiToolKind::Static { tool_name } => Some(tool_name.as_str()),
            UiToolKind::Dynamic { .. } => None,
        },
        _ => None,
    }
}

/// Return the resolved tool name for static or dynamic tool UI parts.
pub fn get_tool_name(part: &UiMessagePart) -> Option<&str> {
    match part {
        UiMessagePart::Tool(tool_part) => Some(tool_part.tool_name()),
        _ => None,
    }
}

/// Deprecated AI SDK helper spelling. Use `get_tool_name`.
pub fn get_tool_or_dynamic_tool_name(part: &UiMessagePart) -> Option<&str> {
    get_tool_name(part)
}

/// Check whether the last assistant message's final step has completed non-provider tool calls.
pub fn last_assistant_message_is_complete_with_tool_calls(messages: &[UiMessage]) -> bool {
    let Some(message) = messages.last() else {
        return false;
    };
    if message.role != UiMessageRole::Assistant {
        return false;
    }

    let last_step_start_index = message
        .parts
        .iter()
        .rposition(|part| matches!(part, UiMessagePart::StepStart));
    let start = last_step_start_index.map_or(0, |index| index + 1);

    let tool_parts = message.parts[start..].iter().filter_map(|part| match part {
        UiMessagePart::Tool(tool_part) if tool_part.provider_executed != Some(true) => {
            Some(tool_part)
        }
        _ => None,
    });

    let mut found = false;
    for tool_part in tool_parts {
        found = true;
        if !matches!(
            tool_part.state,
            UiToolPartState::OutputAvailable | UiToolPartState::OutputError
        ) {
            return false;
        }
    }

    found
}

/// Check whether the last assistant message's final step has completed approval responses.
pub fn last_assistant_message_is_complete_with_approval_responses(messages: &[UiMessage]) -> bool {
    let Some(message) = messages.last() else {
        return false;
    };
    if message.role != UiMessageRole::Assistant {
        return false;
    }

    let last_step_start_index = message
        .parts
        .iter()
        .rposition(|part| matches!(part, UiMessagePart::StepStart));
    let start = last_step_start_index.map_or(0, |index| index + 1);

    let mut found_approval_response = false;
    for part in &message.parts[start..] {
        let UiMessagePart::Tool(tool_part) = part else {
            continue;
        };

        found_approval_response |= matches!(tool_part.state, UiToolPartState::ApprovalResponded);

        if !matches!(
            tool_part.state,
            UiToolPartState::OutputAvailable
                | UiToolPartState::OutputError
                | UiToolPartState::ApprovalResponded
        ) {
            return false;
        }
    }

    found_approval_response
}

/// Passive message input accepted by AI SDK `CreateUIMessage`.
///
/// Upstream models this as `Omit<UIMessage, "id" | "role"> & { id?: ...; role?: ... }`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CreateUIMessage<MessagePart = UiMessagePart> {
    /// Optional UI message id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional role, defaulted by the caller/runtime when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<UiMessageRole>,
    /// Optional UI-only metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JSONValue>,
    /// Renderable UI parts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<MessagePart>,
}

impl<MessagePart> Default for CreateUIMessage<MessagePart> {
    fn default() -> Self {
        Self {
            id: None,
            role: None,
            metadata: None,
            parts: Vec::new(),
        }
    }
}

impl<MessagePart> CreateUIMessage<MessagePart> {
    /// Create an empty create-message payload.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the optional message id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the optional message role.
    pub fn with_role(mut self, role: UiMessageRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Set UI-only metadata.
    pub fn with_metadata(mut self, metadata: impl Into<JSONValue>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    /// Set renderable UI parts.
    pub fn with_parts(mut self, parts: impl IntoIterator<Item = MessagePart>) -> Self {
        self.parts = parts.into_iter().collect();
        self
    }
}

/// Serializable subset of AI SDK `ChatRequestOptions`.
///
/// Browser `Headers` objects are represented as a plain string map.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatRequestOptions {
    /// Additional headers passed to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Additional JSON body properties sent to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Request metadata passed through the UI transport.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JSONValue>,
}

impl ChatRequestOptions {
    /// Create empty chat request options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request headers.
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = Some(headers.into_iter().collect());
        self
    }

    /// Set additional request body properties.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Set request metadata.
    pub fn with_metadata(mut self, metadata: impl Into<JSONValue>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }
}

/// AI SDK `ChatStatus` values.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ChatStatus {
    Submitted,
    Streaming,
    Ready,
    Error,
}

/// Passive snapshot of AI SDK `ChatState`.
///
/// Upstream also includes mutation/snapshot functions. Rust keeps only the serializable state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatState<Message = UiMessage> {
    /// Current chat status.
    pub status: ChatStatus,
    /// Error payload when the chat is in an error state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JSONValue>,
    /// Current UI messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
}

impl<Message> ChatState<Message> {
    /// Create a ready chat-state snapshot.
    pub fn ready(messages: Vec<Message>) -> Self {
        Self {
            status: ChatStatus::Ready,
            error: None,
            messages,
        }
    }

    /// Attach an error payload and mark the chat as errored.
    pub fn with_error(mut self, error: impl Into<JSONValue>) -> Self {
        self.status = ChatStatus::Error;
        self.error = Some(error.into());
        self
    }
}

/// Serializable subset of AI SDK `ChatInit`.
///
/// Function-valued callbacks, `generateId`, schema validators, and transport objects are
/// intentionally deferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatInit<Message = UiMessage> {
    /// Optional chat id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Initial messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
}

impl<Message> ChatInit<Message> {
    /// Create empty chat initialization options.
    pub fn new() -> Self {
        Self {
            id: None,
            messages: Vec::new(),
        }
    }

    /// Set the chat id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set initial messages.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }
}

/// AI SDK chat transport send trigger.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ChatTransportTrigger {
    SubmitMessage,
    RegenerateMessage,
}

/// Passive options passed to AI SDK `ChatTransport.sendMessages`.
///
/// `AbortSignal` is runtime-only and intentionally omitted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatTransportSendMessagesOptions<Message = UiMessage> {
    /// New submission or regeneration.
    pub trigger: ChatTransportTrigger,
    /// Chat session id.
    pub chat_id: String,
    /// Message id for regeneration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Conversation history.
    pub messages: Vec<Message>,
    /// Additional request options.
    #[serde(flatten)]
    pub request_options: ChatRequestOptions,
}

/// Passive options passed to AI SDK `ChatTransport.reconnectToStream`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatTransportReconnectToStreamOptions {
    /// Chat session id.
    pub chat_id: String,
    /// Additional request options.
    #[serde(flatten)]
    pub request_options: ChatRequestOptions,
}

/// Serializable subset of AI SDK `HttpChatTransportInitOptions`.
///
/// Custom `fetch` and request-preparation callbacks are intentionally deferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HttpChatTransportInitOptions {
    /// Chat API URL. Upstream defaults to `/api/chat` when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Browser credentials mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// HTTP headers sent with requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Extra body object sent with requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl HttpChatTransportInitOptions {
    /// Create empty HTTP chat transport options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chat API URL.
    pub fn with_api(mut self, api: impl Into<String>) -> Self {
        self.api = Some(api.into());
        self
    }

    /// Set the credentials mode.
    pub fn with_credentials(mut self, credentials: RequestCredentials) -> Self {
        self.credentials = Some(credentials);
        self
    }
}

/// Passive input payload supplied to AI SDK `PrepareSendMessagesRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PrepareSendMessagesRequestOptions<Message = UiMessage> {
    /// Chat id.
    pub id: String,
    /// Conversation history.
    pub messages: Vec<Message>,
    /// Request metadata from `ChatRequestOptions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_metadata: Option<JSONValue>,
    /// Merged body before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Credentials mode before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Headers before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// API URL before final preparation.
    pub api: String,
    /// New submission or regeneration.
    pub trigger: ChatTransportTrigger,
    /// Message id for regeneration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
}

/// Passive return payload from AI SDK `PrepareSendMessagesRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PreparedSendMessagesRequest {
    /// Final request body.
    pub body: JSONValue,
    /// Final request headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Final request credentials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Final API URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
}

/// Passive input payload supplied to AI SDK `PrepareReconnectToStreamRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PrepareReconnectToStreamRequestOptions {
    /// Chat id.
    pub id: String,
    /// Request metadata from `ChatRequestOptions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_metadata: Option<JSONValue>,
    /// Merged body before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Credentials mode before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Headers before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// API URL before final preparation.
    pub api: String,
}

/// Passive return payload from AI SDK `PrepareReconnectToStreamRequest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PreparedReconnectToStreamRequest {
    /// Final request headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Final request credentials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Final API URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
}

/// Serializable subset of AI SDK `CompletionRequestOptions`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CompletionRequestOptions {
    /// Additional headers passed to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Additional JSON body properties sent to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl CompletionRequestOptions {
    /// Create empty completion request options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request headers.
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = Some(headers.into_iter().collect());
        self
    }

    /// Set additional request body properties.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }
}

/// Browser request credentials mode used by AI SDK UI helpers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum RequestCredentials {
    Omit,
    SameOrigin,
    Include,
}

/// AI SDK `useCompletion` stream protocol.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum CompletionStreamProtocol {
    Data,
    Text,
}

/// Serializable subset of AI SDK `UseCompletionOptions`.
///
/// Function-valued options (`onFinish`, `onError`) and custom `fetch` are intentionally deferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UseCompletionOptions {
    /// Completion API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Shared completion id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Initial prompt input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_input: Option<String>,
    /// Initial completion text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_completion: Option<String>,
    /// Browser credentials mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// HTTP headers sent with the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Extra JSON body object sent with the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Streaming protocol, defaulted by the UI runtime when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_protocol: Option<CompletionStreamProtocol>,
}

impl UseCompletionOptions {
    /// Create empty completion hook options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the completion API endpoint.
    pub fn with_api(mut self, api: impl Into<String>) -> Self {
        self.api = Some(api.into());
        self
    }

    /// Set the shared completion id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the initial prompt input.
    pub fn with_initial_input(mut self, initial_input: impl Into<String>) -> Self {
        self.initial_input = Some(initial_input.into());
        self
    }

    /// Set the initial completion text.
    pub fn with_initial_completion(mut self, initial_completion: impl Into<String>) -> Self {
        self.initial_completion = Some(initial_completion.into());
        self
    }

    /// Set browser credentials mode.
    pub fn with_credentials(mut self, credentials: RequestCredentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set request headers.
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = Some(headers.into_iter().collect());
        self
    }

    /// Set extra request body properties.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Set stream protocol.
    pub fn with_stream_protocol(mut self, stream_protocol: CompletionStreamProtocol) -> Self {
        self.stream_protocol = Some(stream_protocol);
        self
    }
}

/// Passive serializable subset of AI SDK `UIMessageStreamOptions`.
///
/// Function-valued options such as `generateMessageId`, `onFinish`, `messageMetadata`, and
/// `onError` are intentionally not represented here because they are runtime callbacks, not data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageStreamOptions<Message = UiMessage> {
    /// Original messages. When present, AI SDK assumes persistence mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_messages: Option<Vec<Message>>,
    /// Whether reasoning parts should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_reasoning: Option<bool>,
    /// Whether source parts should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_sources: Option<bool>,
    /// Whether the finish event should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_finish: Option<bool>,
    /// Whether the message start event should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_start: Option<bool>,
}

impl<Message> Default for UiMessageStreamOptions<Message> {
    fn default() -> Self {
        Self {
            original_messages: None,
            send_reasoning: None,
            send_sources: None,
            send_finish: None,
            send_start: None,
        }
    }
}

impl<Message> UiMessageStreamOptions<Message> {
    /// Create empty UI message stream options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the original messages.
    pub fn with_original_messages(
        mut self,
        original_messages: impl IntoIterator<Item = Message>,
    ) -> Self {
        self.original_messages = Some(original_messages.into_iter().collect());
        self
    }

    /// Set whether reasoning parts should be sent.
    pub fn with_send_reasoning(mut self, send_reasoning: bool) -> Self {
        self.send_reasoning = Some(send_reasoning);
        self
    }

    /// Set whether source parts should be sent.
    pub fn with_send_sources(mut self, send_sources: bool) -> Self {
        self.send_sources = Some(send_sources);
        self
    }

    /// Set whether finish events should be sent.
    pub fn with_send_finish(mut self, send_finish: bool) -> Self {
        self.send_finish = Some(send_finish);
        self
    }

    /// Set whether start events should be sent.
    pub fn with_send_start(mut self, send_start: bool) -> Self {
        self.send_start = Some(send_start);
        self
    }
}

/// AI SDK export spelling for `UIMessageStreamOptions`.
pub type UIMessageStreamOptions<Message = UiMessage> = UiMessageStreamOptions<Message>;
