//! AI SDK-style UI message helpers.

use std::collections::HashMap;

use crate::tooling::{ExecutableTools, ToolModelOutputContext};
use crate::types::{
    ChatMessage, ChatRequest, ContentPart, FilePartSource, MediaSource, MessageContent,
    MessageRole, ProviderOptionsMap, ToolResultOutput, UiDataPart, UiFilePart, UiMessage,
    UiMessagePart, UiMessageRole, UiReasoningFilePart, UiToolInvocationState, UiToolKind,
    UiToolPart, UiToolPartState,
};
use serde_json::Value;
use thiserror::Error;

/// Errors raised while validating or converting UI messages.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum UiMessageError {
    /// A UI message had no parts.
    #[error("UI message `{message_id}` must contain at least one part")]
    EmptyMessageParts { message_id: String },

    /// A UI message part is structurally invalid.
    #[error("UI message `{message_id}` part {part_index} is invalid: {message}")]
    InvalidPart {
        message_id: String,
        part_index: usize,
        message: String,
    },

    /// UI message metadata failed schema validation.
    #[error("UI message `{message_id}` metadata is invalid: {message}")]
    InvalidMetadata { message_id: String, message: String },

    /// Runtime tool-output mapping failed while converting a UI message.
    #[error("failed to convert UI tool `{tool_name}` (`{tool_call_id}`) output: {message}")]
    ToolOutputConversion {
        tool_name: String,
        tool_call_id: String,
        message: String,
    },
}

/// Options controlling `convert_to_model_messages`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConvertUiMessagesOptions {
    /// Drop `input-streaming` and `input-available` tool parts before conversion.
    pub ignore_incomplete_tool_calls: bool,
}

/// Additional schema-aware validation inputs for `validate_ui_messages_with_schemas`.
#[derive(Debug, Clone, Default)]
pub struct ValidateUiMessagesSchemaOptions<'a> {
    /// Optional schema for message-level metadata.
    pub metadata_schema: Option<&'a Value>,
    /// Optional schemas for `data-*` UI parts keyed by their suffix name.
    pub data_schemas: Option<&'a HashMap<String, Value>>,
}

/// Rust result union for AI SDK `SafeValidateUIMessagesResult`.
#[derive(Debug, Clone, PartialEq)]
pub enum SafeValidateUiMessagesResult {
    /// Validation succeeded and returns the validated message list.
    Success { data: Vec<UiMessage> },
    /// Validation failed without throwing.
    Failure { error: UiMessageError },
}

impl SafeValidateUiMessagesResult {
    /// Return whether validation succeeded.
    pub const fn success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Borrow the validated message list when validation succeeded.
    pub fn data(&self) -> Option<&[UiMessage]> {
        match self {
            Self::Success { data } => Some(data),
            Self::Failure { .. } => None,
        }
    }

    /// Borrow the validation error when validation failed.
    pub fn error(&self) -> Option<&UiMessageError> {
        match self {
            Self::Success { .. } => None,
            Self::Failure { error } => Some(error),
        }
    }

    /// Convert the safe result into a standard Rust `Result`.
    pub fn into_result(self) -> Result<Vec<UiMessage>, UiMessageError> {
        match self {
            Self::Success { data } => Ok(data),
            Self::Failure { error } => Err(error),
        }
    }
}

/// AI SDK export spelling for `SafeValidateUIMessagesResult`.
pub type SafeValidateUIMessagesResult = SafeValidateUiMessagesResult;

/// Abstract schema validator used by UI-message schema-aware validation.
pub trait UiSchemaValidator: Send + Sync {
    /// Validate `instance` against `schema`.
    fn validate(&self, schema: &Value, instance: &Value) -> Result<(), String>;
}

impl<F> UiSchemaValidator for F
where
    F: Fn(&Value, &Value) -> Result<(), String> + Send + Sync,
{
    fn validate(&self, schema: &Value, instance: &Value) -> Result<(), String> {
        self(schema, instance)
    }
}

/// Validate a batch of UI messages.
pub fn validate_ui_messages(messages: &[UiMessage]) -> Result<(), UiMessageError> {
    for message in messages {
        if message.parts.is_empty() {
            return Err(UiMessageError::EmptyMessageParts {
                message_id: message.id.clone(),
            });
        }

        for (part_index, part) in message.parts.iter().enumerate() {
            if let Err(message_text) = validate_ui_message_part(part) {
                return Err(UiMessageError::InvalidPart {
                    message_id: message.id.clone(),
                    part_index,
                    message: message_text,
                });
            }
        }
    }

    Ok(())
}

/// Validate UI messages and return a data-carrying success/failure union instead of an error.
pub fn safe_validate_ui_messages(messages: &[UiMessage]) -> SafeValidateUiMessagesResult {
    match validate_ui_messages(messages) {
        Ok(()) => SafeValidateUiMessagesResult::Success {
            data: messages.to_vec(),
        },
        Err(error) => SafeValidateUiMessagesResult::Failure { error },
    }
}

/// Validate UI messages structurally and, optionally, against metadata/data/tool schemas.
pub fn validate_ui_messages_with_schemas(
    messages: &[UiMessage],
    options: ValidateUiMessagesSchemaOptions<'_>,
    tools: Option<&ExecutableTools>,
    validator: &dyn UiSchemaValidator,
) -> Result<(), UiMessageError> {
    validate_ui_messages(messages)?;

    for message in messages {
        if let (Some(schema), Some(metadata)) = (options.metadata_schema, message.metadata.as_ref())
        {
            validator
                .validate(schema, metadata)
                .map_err(|message_text| UiMessageError::InvalidMetadata {
                    message_id: message.id.clone(),
                    message: message_text,
                })?;
        }

        for (part_index, part) in message.parts.iter().enumerate() {
            match part {
                UiMessagePart::Data(part) => {
                    let Some(data_schemas) = options.data_schemas else {
                        continue;
                    };
                    let Some(schema) = data_schemas.get(&part.data_type) else {
                        return Err(UiMessageError::InvalidPart {
                            message_id: message.id.clone(),
                            part_index,
                            message: format!("no schema found for data part `{}`", part.data_type),
                        });
                    };

                    validator
                        .validate(schema, &part.data)
                        .map_err(|message_text| UiMessageError::InvalidPart {
                            message_id: message.id.clone(),
                            part_index,
                            message: format!(
                                "data part `{}` failed schema validation: {message_text}",
                                part.data_type
                            ),
                        })?;
                }
                UiMessagePart::Tool(part) => {
                    let Some(tools) = tools else {
                        continue;
                    };
                    if !matches!(part.kind, UiToolKind::Static { .. }) {
                        continue;
                    }

                    let Some(tool) = tools.get(part.tool_name()) else {
                        return Err(UiMessageError::InvalidPart {
                            message_id: message.id.clone(),
                            part_index,
                            message: format!(
                                "no tool schema found for tool `{}`",
                                part.tool_name()
                            ),
                        });
                    };

                    let invocation = part
                        .invocation()
                        .expect("tool part already passed structural validation");

                    let input_instance = match &invocation.state {
                        UiToolInvocationState::InputAvailable { input }
                        | UiToolInvocationState::OutputAvailable { input, .. } => Some(input),
                        UiToolInvocationState::OutputError { input, .. } => input.as_ref(),
                        UiToolInvocationState::InputStreaming { .. }
                        | UiToolInvocationState::ApprovalRequested { .. }
                        | UiToolInvocationState::ApprovalResponded { .. }
                        | UiToolInvocationState::OutputDenied { .. } => None,
                    };

                    if let (Some(schema), Some(input)) =
                        (tool.tool().input_schema(), input_instance)
                    {
                        validator.validate(schema, input).map_err(|message_text| {
                            UiMessageError::InvalidPart {
                                message_id: message.id.clone(),
                                part_index,
                                message: format!(
                                    "tool `{}` input failed schema validation: {message_text}",
                                    part.tool_name()
                                ),
                            }
                        })?;
                    }

                    if let (Some(schema), UiToolInvocationState::OutputAvailable { output, .. }) =
                        (tool.tool().output_schema(), &invocation.state)
                    {
                        validator.validate(schema, output).map_err(|message_text| {
                            UiMessageError::InvalidPart {
                                message_id: message.id.clone(),
                                part_index,
                                message: format!(
                                    "tool `{}` output failed schema validation: {message_text}",
                                    part.tool_name()
                                ),
                            }
                        })?;
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Validate UI messages with schemas and return a success/failure union instead of an error.
pub fn safe_validate_ui_messages_with_schemas(
    messages: &[UiMessage],
    options: ValidateUiMessagesSchemaOptions<'_>,
    tools: Option<&ExecutableTools>,
    validator: &dyn UiSchemaValidator,
) -> SafeValidateUiMessagesResult {
    match validate_ui_messages_with_schemas(messages, options, tools, validator) {
        Ok(()) => SafeValidateUiMessagesResult::Success {
            data: messages.to_vec(),
        },
        Err(error) => SafeValidateUiMessagesResult::Failure { error },
    }
}

/// Convert UI messages into stable model messages (`ChatMessage`).
pub fn convert_to_model_messages(
    messages: &[UiMessage],
) -> Result<Vec<ChatMessage>, UiMessageError> {
    convert_to_model_messages_with(messages, ConvertUiMessagesOptions::default(), |_part| {
        Ok(None)
    })
}

/// Convert UI messages into stable model messages (`ChatMessage`) with a data-part converter.
pub fn convert_to_model_messages_with<F>(
    messages: &[UiMessage],
    options: ConvertUiMessagesOptions,
    mut convert_data_part: F,
) -> Result<Vec<ChatMessage>, UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    convert_to_model_messages_inner(messages, options, None, &mut convert_data_part)
}

/// Convert UI messages into stable model messages with runtime tool-output mapping support.
pub fn convert_to_model_messages_with_tooling<F>(
    messages: &[UiMessage],
    options: ConvertUiMessagesOptions,
    tools: &ExecutableTools,
    mut convert_data_part: F,
) -> Result<Vec<ChatMessage>, UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    convert_to_model_messages_inner(messages, options, Some(tools), &mut convert_data_part)
}

/// Convert UI messages directly into a `ChatRequest`.
pub fn convert_to_chat_request(messages: &[UiMessage]) -> Result<ChatRequest, UiMessageError> {
    Ok(ChatRequest::new(convert_to_model_messages(messages)?))
}

/// Convert UI messages directly into a `ChatRequest` with a data-part converter.
pub fn convert_to_chat_request_with<F>(
    messages: &[UiMessage],
    options: ConvertUiMessagesOptions,
    convert_data_part: F,
) -> Result<ChatRequest, UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    Ok(ChatRequest::new(convert_to_model_messages_with(
        messages,
        options,
        convert_data_part,
    )?))
}

/// Convert UI messages directly into a `ChatRequest` with runtime tool-output mapping support.
pub fn convert_to_chat_request_with_tooling<F>(
    messages: &[UiMessage],
    options: ConvertUiMessagesOptions,
    tools: &ExecutableTools,
    convert_data_part: F,
) -> Result<ChatRequest, UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    Ok(ChatRequest::new(convert_to_model_messages_with_tooling(
        messages,
        options,
        tools,
        convert_data_part,
    )?))
}

fn convert_to_model_messages_inner<F>(
    messages: &[UiMessage],
    options: ConvertUiMessagesOptions,
    tools: Option<&ExecutableTools>,
    convert_data_part: &mut F,
) -> Result<Vec<ChatMessage>, UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    validate_ui_messages(messages)?;

    let mut model_messages = Vec::new();

    for message in messages {
        match message.role {
            UiMessageRole::System => {
                model_messages.push(convert_system_message(message));
            }
            UiMessageRole::User => {
                model_messages.push(convert_user_message(message, convert_data_part, options)?);
            }
            UiMessageRole::Assistant => {
                convert_assistant_message(
                    message,
                    &mut model_messages,
                    convert_data_part,
                    options,
                    tools,
                )?;
            }
        }
    }

    Ok(model_messages)
}

fn validate_ui_message_part(part: &UiMessagePart) -> Result<(), String> {
    part.validate()
}

fn convert_system_message(message: &UiMessage) -> ChatMessage {
    let mut content = String::new();
    let mut provider_options = ProviderOptionsMap::default();

    for part in &message.parts {
        if let UiMessagePart::Text(part) = part {
            content.push_str(&part.text);
            provider_options.merge_overrides(part.provider_metadata.clone());
        }
    }

    let mut message = ChatMessage::system(content).build();
    *message.provider_options_mut() = provider_options;
    message
}

fn convert_user_message<F>(
    message: &UiMessage,
    convert_data_part: &mut F,
    _options: ConvertUiMessagesOptions,
) -> Result<ChatMessage, UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    let mut content = Vec::new();

    for part in &message.parts {
        match part {
            UiMessagePart::Text(part) => content.push(convert_text_part(part)),
            UiMessagePart::File(part) => content.push(convert_user_file_part(part)),
            UiMessagePart::Data(part) => {
                if let Some(part) = convert_data_part(part)? {
                    content.push(part);
                }
            }
            _ => {}
        }
    }

    Ok(build_message_from_parts(MessageRole::User, content))
}

fn convert_assistant_message<F>(
    message: &UiMessage,
    model_messages: &mut Vec<ChatMessage>,
    convert_data_part: &mut F,
    options: ConvertUiMessagesOptions,
    tools: Option<&ExecutableTools>,
) -> Result<(), UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    let mut block = Vec::new();

    for part in &message.parts {
        if let UiMessagePart::Tool(tool_part) = part
            && options.ignore_incomplete_tool_calls
            && tool_part.is_incomplete_input_state()
        {
            continue;
        }

        match part {
            UiMessagePart::Text(_)
            | UiMessagePart::Custom(_)
            | UiMessagePart::Reasoning(_)
            | UiMessagePart::ReasoningFile(_)
            | UiMessagePart::File(_)
            | UiMessagePart::Tool(_)
            | UiMessagePart::Data(_) => block.push(part),
            UiMessagePart::StepStart => {
                flush_assistant_block(&block, model_messages, convert_data_part, tools)?;
                block.clear();
            }
            UiMessagePart::SourceUrl(_) | UiMessagePart::SourceDocument(_) => {}
        }
    }

    flush_assistant_block(&block, model_messages, convert_data_part, tools)?;
    Ok(())
}

fn flush_assistant_block<F>(
    block: &[&UiMessagePart],
    model_messages: &mut Vec<ChatMessage>,
    convert_data_part: &mut F,
    tools: Option<&ExecutableTools>,
) -> Result<(), UiMessageError>
where
    F: FnMut(&UiDataPart) -> Result<Option<ContentPart>, UiMessageError>,
{
    if block.is_empty() {
        return Ok(());
    }

    let mut assistant_content = Vec::new();

    for part in block {
        match *part {
            UiMessagePart::Text(part) => assistant_content.push(convert_text_part(part)),
            UiMessagePart::Custom(part) => assistant_content.push(convert_custom_part(part)),
            UiMessagePart::Reasoning(part) => assistant_content.push(convert_reasoning_part(part)),
            UiMessagePart::ReasoningFile(part) => {
                assistant_content.push(convert_reasoning_file_part(part))
            }
            UiMessagePart::File(part) => assistant_content.push(convert_assistant_file_part(part)),
            UiMessagePart::Tool(part) => {
                if !matches!(part.state, UiToolPartState::InputStreaming) {
                    assistant_content.push(convert_tool_call_part(part));

                    if let Some(approval) = part.approval.as_ref() {
                        assistant_content.push(ContentPart::tool_approval_request(
                            approval.id.clone(),
                            part.tool_call_id.clone(),
                        ));
                    }

                    if part.provider_executed == Some(true)
                        && !matches!(part.state, UiToolPartState::ApprovalResponded)
                        && matches!(
                            part.state,
                            UiToolPartState::OutputAvailable | UiToolPartState::OutputError
                        )
                    {
                        assistant_content.push(convert_tool_result_part(part, true, tools)?);
                    }
                }
            }
            UiMessagePart::Data(part) => {
                if let Some(part) = convert_data_part(part)? {
                    assistant_content.push(part);
                }
            }
            UiMessagePart::SourceUrl(_)
            | UiMessagePart::SourceDocument(_)
            | UiMessagePart::StepStart => {}
        }
    }

    model_messages.push(build_message_from_parts(
        MessageRole::Assistant,
        assistant_content,
    ));

    let mut tool_content = Vec::new();

    for part in block.iter().filter_map(|part| match *part {
        UiMessagePart::Tool(part) => Some(part),
        _ => None,
    }) {
        if part.provider_executed == Some(true)
            && part
                .approval
                .as_ref()
                .and_then(|approval| approval.approved)
                .is_none()
        {
            continue;
        }

        if let Some(approval) = part.approval.as_ref()
            && let Some(approved) = approval.approved
        {
            tool_content.push(ContentPart::ToolApprovalResponse {
                approval_id: approval.id.clone(),
                approved,
                reason: approval.reason.clone(),
                provider_executed: part.provider_executed,
                provider_options: ProviderOptionsMap::default(),
            });
        }

        if part.provider_executed == Some(true) {
            continue;
        }

        match part.state {
            UiToolPartState::OutputDenied => {
                tool_content.push(convert_tool_denied_result_part(part));
            }
            UiToolPartState::OutputAvailable | UiToolPartState::OutputError => {
                tool_content.push(convert_tool_result_part(part, false, tools)?);
            }
            _ => {}
        }
    }

    if !tool_content.is_empty() {
        model_messages.push(build_message_from_parts(MessageRole::Tool, tool_content));
    }

    Ok(())
}

fn convert_text_part(part: &crate::types::UiTextPart) -> ContentPart {
    let mut text_part = ContentPart::text(part.text.clone());
    if let Some(provider_options) = text_part.provider_options_mut() {
        provider_options.merge_overrides(part.provider_metadata.clone());
    }
    text_part
}

fn convert_custom_part(part: &crate::types::UiCustomPart) -> ContentPart {
    let mut custom_part = ContentPart::custom(part.kind.clone());
    if let Some(provider_options) = custom_part.provider_options_mut() {
        provider_options.merge_overrides(part.provider_metadata.clone());
    }
    custom_part
}

fn convert_reasoning_part(part: &crate::types::UiReasoningPart) -> ContentPart {
    let mut reasoning_part = ContentPart::reasoning(part.text.clone());
    if let Some(provider_options) = reasoning_part.provider_options_mut() {
        provider_options.merge_overrides(part.provider_metadata.clone());
    }
    reasoning_part
}

fn convert_user_file_part(part: &UiFilePart) -> ContentPart {
    convert_assistant_file_part(part)
}

fn convert_assistant_file_part(part: &UiFilePart) -> ContentPart {
    let source = if let Some(provider_reference) = &part.provider_reference {
        FilePartSource::provider_reference(provider_reference.clone())
    } else {
        FilePartSource::url(part.url.clone())
    };

    let mut file_part = ContentPart::File {
        source,
        media_type: part.media_type.clone(),
        filename: part.filename.clone(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    };
    if let Some(provider_options) = file_part.provider_options_mut() {
        provider_options.merge_overrides(part.provider_metadata.clone());
    }
    file_part
}

fn convert_reasoning_file_part(part: &UiReasoningFilePart) -> ContentPart {
    let mut reasoning_file_part = ContentPart::ReasoningFile {
        source: MediaSource::url(part.url.clone()),
        media_type: part.media_type.clone(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    };
    if let Some(provider_options) = reasoning_file_part.provider_options_mut() {
        provider_options.merge_overrides(part.provider_metadata.clone());
    }
    reasoning_file_part
}

fn convert_tool_call_part(part: &UiToolPart) -> ContentPart {
    let input = match part.state {
        UiToolPartState::OutputError => part
            .input
            .clone()
            .or_else(|| part.raw_input.clone())
            .unwrap_or(Value::Null),
        _ => part.input.clone().unwrap_or(Value::Null),
    };

    ContentPart::ToolCall {
        tool_call_id: part.tool_call_id.clone(),
        tool_name: part.tool_name().to_string(),
        arguments: input,
        provider_executed: part.provider_executed,
        dynamic: matches!(&part.kind, UiToolKind::Dynamic { .. }).then_some(true),
        invalid: None,
        error: None,
        title: part.title.clone(),
        provider_options: part.call_provider_metadata.clone(),
        provider_metadata: None,
    }
}

fn convert_tool_result_part(
    part: &UiToolPart,
    provider_executed: bool,
    tools: Option<&ExecutableTools>,
) -> Result<ContentPart, UiMessageError> {
    let provider_options = if !part.result_provider_metadata.is_empty() {
        part.result_provider_metadata.clone()
    } else {
        part.call_provider_metadata.clone()
    };

    let output = match part.state {
        UiToolPartState::OutputError => {
            let error_text = part
                .error_text
                .clone()
                .unwrap_or_else(|| "Tool execution failed.".to_string());
            if provider_executed {
                ToolResultOutput::error_json(Value::String(error_text))
            } else {
                ToolResultOutput::error_text(error_text)
            }
        }
        UiToolPartState::OutputAvailable => {
            let raw_output = part.output.clone().unwrap_or(Value::Null);
            match map_tool_output_with_runtime_tools(tools, part, &raw_output)? {
                Some(output) => output,
                None => ui_tool_output_to_tool_result(&raw_output),
            }
        }
        _ => ToolResultOutput::json(Value::Null),
    };

    Ok(ContentPart::ToolResult {
        tool_call_id: part.tool_call_id.clone(),
        tool_name: part.tool_name().to_string(),
        output,
        input: part.input.clone(),
        provider_executed: provider_executed.then_some(true),
        dynamic: matches!(&part.kind, UiToolKind::Dynamic { .. }).then_some(true),
        preliminary: part.preliminary,
        title: part.title.clone(),
        provider_options,
        provider_metadata: None,
    })
}

fn convert_tool_denied_result_part(part: &UiToolPart) -> ContentPart {
    let denied_reason = part
        .approval
        .as_ref()
        .and_then(|approval| approval.reason.clone())
        .unwrap_or_else(|| "Tool execution denied.".to_string());

    ContentPart::ToolResult {
        tool_call_id: part.tool_call_id.clone(),
        tool_name: part.tool_name().to_string(),
        output: ToolResultOutput::error_text(denied_reason),
        input: part.input.clone(),
        provider_executed: part.provider_executed,
        dynamic: matches!(&part.kind, UiToolKind::Dynamic { .. }).then_some(true),
        preliminary: part.preliminary,
        title: part.title.clone(),
        provider_options: part.call_provider_metadata.clone(),
        provider_metadata: None,
    }
}

fn ui_tool_output_to_tool_result(output: &Value) -> ToolResultOutput {
    serde_json::from_value::<ToolResultOutput>(output.clone()).unwrap_or_else(|_| match output {
        Value::String(text) => ToolResultOutput::text(text.clone()),
        other => ToolResultOutput::json(other.clone()),
    })
}

fn map_tool_output_with_runtime_tools(
    tools: Option<&ExecutableTools>,
    part: &UiToolPart,
    output: &Value,
) -> Result<Option<ToolResultOutput>, UiMessageError> {
    let Some(tools) = tools else {
        return Ok(None);
    };

    tools
        .to_model_output(
            part.tool_name(),
            ToolModelOutputContext {
                tool_call_id: part.tool_call_id.clone(),
                input: part.input.clone().unwrap_or(Value::Null),
                output: output.clone(),
            },
        )
        .map_err(|err| UiMessageError::ToolOutputConversion {
            tool_name: part.tool_name().to_string(),
            tool_call_id: part.tool_call_id.clone(),
            message: err.to_string(),
        })
}

fn build_message_from_parts(role: MessageRole, parts: Vec<ContentPart>) -> ChatMessage {
    let content = if parts.is_empty() {
        MessageContent::Text(String::new())
    } else if parts.len() == 1 {
        match parts.into_iter().next().expect("checked single part") {
            ContentPart::Text {
                text,
                provider_options,
                provider_metadata: None,
            } if provider_options.is_empty() && !matches!(role, MessageRole::Tool) => {
                MessageContent::Text(text)
            }
            part => MessageContent::MultiModal(vec![part]),
        }
    } else {
        MessageContent::MultiModal(parts)
    };

    ChatMessage {
        role,
        content,
        provider_options: ProviderOptionsMap::default(),
        metadata: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        ConvertUiMessagesOptions, SafeValidateUiMessagesResult, ValidateUiMessagesSchemaOptions,
        convert_to_model_messages, convert_to_model_messages_with,
        convert_to_model_messages_with_tooling, safe_validate_ui_messages,
        safe_validate_ui_messages_with_schemas, validate_ui_messages,
        validate_ui_messages_with_schemas,
    };
    use crate::tooling::{ExecutableTool, ExecutableTools};
    use crate::types::{
        ChatMessage, ContentPart, ProviderOptionsMap, ProviderReference, Tool, ToolResultOutput,
        UiFilePart, UiMessage, UiMessagePart, UiToolApproval, UiToolPart, UiToolPartState,
    };

    #[test]
    fn validate_rejects_missing_output_error_text() {
        let message = UiMessage::assistant(
            "msg_1",
            vec![UiMessagePart::Tool(UiToolPart::named(
                "search",
                "call_1",
                UiToolPartState::OutputError,
            ))],
        );

        let err = validate_ui_messages(&[message]).expect_err("validation should fail");
        assert!(format!("{err}").contains("errorText is required"));
    }

    #[test]
    fn safe_validate_returns_ai_sdk_style_result_union() {
        let ok = safe_validate_ui_messages(&[UiMessage::user(
            "user",
            vec![UiMessagePart::text("hello")],
        )]);
        assert!(ok.success());
        assert_eq!(ok.data().expect("validated messages")[0].id, "user");
        assert!(ok.clone().into_result().is_ok());

        let failed = safe_validate_ui_messages(&[UiMessage::user("empty", Vec::new())]);
        assert!(!failed.success());
        assert!(matches!(
            failed.error(),
            Some(crate::ui::UiMessageError::EmptyMessageParts { message_id })
                if message_id == "empty"
        ));
        assert!(matches!(
            failed.into_result(),
            Err(crate::ui::UiMessageError::EmptyMessageParts { message_id })
                if message_id == "empty"
        ));
    }

    #[test]
    fn safe_schema_validate_preserves_schema_errors() {
        let mut data_schemas = HashMap::new();
        data_schemas.insert("other".to_string(), serde_json::json!({ "kind": "other" }));

        let result = safe_validate_ui_messages_with_schemas(
            &[UiMessage::user(
                "user",
                vec![UiMessagePart::data(
                    "weather",
                    serde_json::json!({ "city": "Tokyo" }),
                )],
            )],
            ValidateUiMessagesSchemaOptions {
                metadata_schema: None,
                data_schemas: Some(&data_schemas),
            },
            None,
            &|_schema: &serde_json::Value, _instance: &serde_json::Value| Ok(()),
        );

        let SafeValidateUiMessagesResult::Failure { error } = result else {
            panic!("schema validation should fail");
        };
        assert!(format!("{error}").contains("no schema found for data part `weather`"));
    }

    #[test]
    fn convert_merges_system_text_parts_and_provider_metadata() {
        let mut provider_metadata = ProviderOptionsMap::default();
        provider_metadata.insert(
            "openai",
            serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
        );

        let mut text_part = crate::types::UiTextPart::new("sys-");
        text_part.provider_metadata = provider_metadata.clone();

        let messages = vec![UiMessage::system(
            "sys",
            vec![
                UiMessagePart::Text(text_part),
                UiMessagePart::text("prompt"),
            ],
        )];

        let converted = convert_to_model_messages(&messages).expect("convert ok");
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content_text(), Some("sys-prompt"));
        assert_eq!(converted[0].provider_options(), &provider_metadata);
    }

    #[test]
    fn convert_maps_user_provider_reference_file() {
        let mut file_part = UiFilePart::new("https://example.com/ignored.pdf", "application/pdf");
        file_part.provider_reference = Some(ProviderReference::single("openai", "file_123"));

        let messages = vec![UiMessage::user(
            "user",
            vec![UiMessagePart::File(file_part)],
        )];

        let converted = convert_to_model_messages(&messages).expect("convert ok");
        let ChatMessage { content, .. } = &converted[0];
        let crate::types::MessageContent::MultiModal(parts) = content else {
            panic!("expected multimodal user content");
        };
        let ContentPart::File { source, .. } = &parts[0] else {
            panic!("expected file part");
        };
        assert!(source.is_provider_reference());
        assert_eq!(
            source
                .as_provider_reference()
                .and_then(|reference| reference.get("openai")),
            Some("file_123")
        );
    }

    #[test]
    fn convert_splits_assistant_tool_blocks() {
        let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
        tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
        tool.output = Some(serde_json::json!({ "temp": 18 }));

        let messages = vec![UiMessage::assistant(
            "assistant",
            vec![
                UiMessagePart::text("Before "),
                UiMessagePart::Tool(tool),
                UiMessagePart::step_start(),
                UiMessagePart::text("After"),
            ],
        )];

        let converted = convert_to_model_messages(&messages).expect("convert ok");
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, crate::types::MessageRole::Assistant);
        assert_eq!(converted[1].role, crate::types::MessageRole::Tool);
        assert_eq!(converted[2].role, crate::types::MessageRole::Assistant);
    }

    #[test]
    fn convert_ignores_incomplete_tool_calls_when_requested() {
        let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::InputAvailable);
        tool.input = Some(serde_json::json!({ "city": "Tokyo" }));

        let messages = vec![UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool), UiMessagePart::text("done")],
        )];

        let converted = convert_to_model_messages_with(
            &messages,
            ConvertUiMessagesOptions {
                ignore_incomplete_tool_calls: true,
            },
            |_part| Ok(None),
        )
        .expect("convert ok");

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content_text(), Some("done"));
    }

    #[test]
    fn convert_data_parts_with_callback() {
        let messages = vec![UiMessage::user(
            "user",
            vec![UiMessagePart::data(
                "weather",
                serde_json::json!({ "city": "Tokyo" }),
            )],
        )];

        let converted = convert_to_model_messages_with(
            &messages,
            ConvertUiMessagesOptions::default(),
            |part| {
                Ok(Some(ContentPart::text(format!(
                    "city={}",
                    part.data["city"]
                ))))
            },
        )
        .expect("convert ok");

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content_text(), Some("city=\"Tokyo\""));
    }

    #[test]
    fn convert_tool_approval_response_preserves_provider_executed() {
        let mut tool = UiToolPart::dynamic("shell", "call_1", UiToolPartState::ApprovalResponded);
        tool.input = Some(serde_json::json!({ "command": "ls" }));
        tool.provider_executed = Some(true);
        tool.approval = Some(UiToolApproval {
            id: "approval_1".to_string(),
            approved: Some(true),
            reason: Some("ok".to_string()),
        });

        let converted = convert_to_model_messages(&[UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool)],
        )])
        .expect("convert ok");

        let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
            panic!("expected tool message");
        };
        let ContentPart::ToolApprovalResponse {
            provider_executed, ..
        } = &parts[0]
        else {
            panic!("expected tool approval response");
        };
        assert_eq!(*provider_executed, Some(true));
    }

    #[test]
    fn explicit_tool_result_output_shape_roundtrips_from_ui_output() {
        let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
        tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
        tool.output = Some(serde_json::json!({
            "type": "content",
            "value": [
                { "type": "text", "text": "sunny" }
            ]
        }));

        let converted = convert_to_model_messages(&[UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool)],
        )])
        .expect("convert ok");

        let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
            panic!("expected tool message");
        };
        let ContentPart::ToolResult { output, .. } = &parts[0] else {
            panic!("expected tool result");
        };
        assert_eq!(
            output,
            &ToolResultOutput::content(vec![crate::types::ToolResultContentPart::text("sunny")])
        );
    }

    #[test]
    fn provider_executed_output_error_uses_error_json() {
        let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputError);
        tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
        tool.error_text = Some("boom".to_string());
        tool.provider_executed = Some(true);

        let converted = convert_to_model_messages(&[UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool)],
        )])
        .expect("convert ok");

        let crate::types::MessageContent::MultiModal(parts) = &converted[0].content else {
            panic!("expected assistant multimodal content");
        };
        let ContentPart::ToolResult { output, .. } = &parts[1] else {
            panic!("expected assistant tool result");
        };
        assert_eq!(
            output,
            &ToolResultOutput::error_json(serde_json::json!("boom"))
        );
    }

    #[test]
    fn local_output_error_uses_error_text() {
        let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputError);
        tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
        tool.error_text = Some("boom".to_string());

        let converted = convert_to_model_messages(&[UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool)],
        )])
        .expect("convert ok");

        let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
            panic!("expected tool multimodal content");
        };
        let ContentPart::ToolResult { output, .. } = &parts[0] else {
            panic!("expected tool result");
        };
        assert_eq!(output, &ToolResultOutput::error_text("boom"));
    }

    #[test]
    fn runtime_tool_mapper_overrides_default_ui_tool_output_conversion() {
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

        let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
        tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
        tool.output = Some(serde_json::json!({ "temp": 18 }));

        let converted = convert_to_model_messages_with_tooling(
            &[UiMessage::assistant(
                "assistant",
                vec![UiMessagePart::Tool(tool)],
            )],
            ConvertUiMessagesOptions::default(),
            &tools,
            |_part| Ok(None),
        )
        .expect("convert ok");

        let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
            panic!("expected tool multimodal content");
        };
        let ContentPart::ToolResult { output, .. } = &parts[0] else {
            panic!("expected tool result");
        };
        assert_eq!(
            output,
            &ToolResultOutput::content(vec![crate::types::ToolResultContentPart::text(
                "call_1:18"
            )])
        );
    }

    #[test]
    fn schema_validation_rejects_invalid_tool_output_against_tool_schema() {
        let tools =
            ExecutableTools::from_tools([ExecutableTool::new(Tool::function_with_output_schema(
                "weather",
                "Weather tool",
                serde_json::json!({ "kind": "input" }),
                serde_json::json!({ "kind": "output" }),
            ))]);

        let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
        tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
        tool.output = Some(serde_json::json!({ "temp": 18 }));

        let err = validate_ui_messages_with_schemas(
            &[UiMessage::assistant(
                "assistant",
                vec![UiMessagePart::Tool(tool)],
            )],
            ValidateUiMessagesSchemaOptions::default(),
            Some(&tools),
            &|schema: &serde_json::Value, instance: &serde_json::Value| match schema["kind"]
                .as_str()
            {
                Some("input")
                    if instance
                        .get("city")
                        .and_then(|value| value.as_str())
                        .is_some() =>
                {
                    Ok(())
                }
                Some("output")
                    if instance
                        .get("forecast")
                        .and_then(|value| value.as_str())
                        .is_some() =>
                {
                    Ok(())
                }
                Some(kind) => Err(format!("expected valid {kind} payload")),
                None => Ok(()),
            },
        )
        .expect_err("schema validation should fail");

        assert!(format!("{err}").contains("output failed schema validation"));
    }

    #[test]
    fn schema_validation_rejects_data_parts_without_matching_schema() {
        let mut data_schemas = HashMap::new();
        data_schemas.insert("other".to_string(), serde_json::json!({ "kind": "other" }));

        let err = validate_ui_messages_with_schemas(
            &[UiMessage::user(
                "user",
                vec![UiMessagePart::data(
                    "weather",
                    serde_json::json!({ "city": "Tokyo" }),
                )],
            )],
            ValidateUiMessagesSchemaOptions {
                metadata_schema: None,
                data_schemas: Some(&data_schemas),
            },
            None,
            &|_schema: &serde_json::Value, _instance: &serde_json::Value| Ok(()),
        )
        .expect_err("missing data schema should fail");

        assert!(format!("{err}").contains("no schema found for data part `weather`"));
    }
}
