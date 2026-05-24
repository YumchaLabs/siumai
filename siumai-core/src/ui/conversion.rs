use crate::tooling::{ExecutableTools, ToolModelOutputContext};
use crate::types::{
    ChatMessage, ChatRequest, ContentPart, FilePartSource, MediaSource, MessageContent,
    MessageRole, ProviderOptionsMap, ToolResultOutput, UiDataPart, UiFilePart, UiMessage,
    UiMessagePart, UiMessageRole, UiReasoningFilePart, UiToolKind, UiToolPart, UiToolPartState,
};
use serde_json::Value;

use super::types::{ConvertUiMessagesOptions, UiMessageError};
use super::validation::validate_ui_messages;

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

fn ui_request_options_from_metadata(provider_metadata: &ProviderOptionsMap) -> ProviderOptionsMap {
    let mut provider_options = ProviderOptionsMap::default();
    provider_options.merge_overrides(provider_metadata.clone());
    provider_options
}

// UI message providerMetadata names are request metadata at this adapter boundary. These helpers
// are the only place UI conversion should manufacture legacy `ContentPart` request carriers, and
// they deliberately leave response-side provider metadata empty.
fn ui_request_text_part(
    text: impl Into<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options,
        provider_metadata: None,
    }
}

fn ui_request_custom_part(
    kind: impl Into<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Custom {
        kind: kind.into(),
        provider_options,
        provider_metadata: None,
    }
}

fn ui_request_reasoning_part(
    text: impl Into<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Reasoning {
        text: text.into(),
        provider_options,
        provider_metadata: None,
    }
}

fn ui_request_file_part(
    source: FilePartSource,
    media_type: impl Into<String>,
    filename: Option<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::File {
        source,
        media_type: media_type.into(),
        filename,
        provider_options,
        provider_metadata: None,
    }
}

fn ui_request_reasoning_file_part(
    source: MediaSource,
    media_type: impl Into<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::ReasoningFile {
        source,
        media_type: media_type.into(),
        provider_options,
        provider_metadata: None,
    }
}

fn ui_request_tool_call_part(part: &UiToolPart, input: Value) -> ContentPart {
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

fn ui_request_tool_result_part(
    part: &UiToolPart,
    output: ToolResultOutput,
    provider_executed: Option<bool>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::ToolResult {
        tool_call_id: part.tool_call_id.clone(),
        tool_name: part.tool_name().to_string(),
        output,
        input: part.input.clone(),
        provider_executed,
        dynamic: matches!(&part.kind, UiToolKind::Dynamic { .. }).then_some(true),
        preliminary: part.preliminary,
        title: part.title.clone(),
        provider_options,
        provider_metadata: None,
    }
}

fn convert_text_part(part: &crate::types::UiTextPart) -> ContentPart {
    ui_request_text_part(
        part.text.clone(),
        ui_request_options_from_metadata(&part.provider_metadata),
    )
}

fn convert_custom_part(part: &crate::types::UiCustomPart) -> ContentPart {
    ui_request_custom_part(
        part.kind.clone(),
        ui_request_options_from_metadata(&part.provider_metadata),
    )
}

fn convert_reasoning_part(part: &crate::types::UiReasoningPart) -> ContentPart {
    ui_request_reasoning_part(
        part.text.clone(),
        ui_request_options_from_metadata(&part.provider_metadata),
    )
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

    ui_request_file_part(
        source,
        part.media_type.clone(),
        part.filename.clone(),
        ui_request_options_from_metadata(&part.provider_metadata),
    )
}

fn convert_reasoning_file_part(part: &UiReasoningFilePart) -> ContentPart {
    ui_request_reasoning_file_part(
        MediaSource::url(part.url.clone()),
        part.media_type.clone(),
        ui_request_options_from_metadata(&part.provider_metadata),
    )
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

    ui_request_tool_call_part(part, input)
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

    Ok(ui_request_tool_result_part(
        part,
        output,
        provider_executed.then_some(true),
        provider_options,
    ))
}

fn convert_tool_denied_result_part(part: &UiToolPart) -> ContentPart {
    let denied_reason = part
        .approval
        .as_ref()
        .and_then(|approval| approval.reason.clone())
        .unwrap_or_else(|| "Tool call execution denied.".to_string());

    ui_request_tool_result_part(
        part,
        ToolResultOutput::error_text(denied_reason),
        part.provider_executed,
        part.call_provider_metadata.clone(),
    )
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
