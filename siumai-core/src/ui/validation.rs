use crate::tooling::ExecutableTools;
use crate::types::{UiMessage, UiMessagePart, UiToolInvocationState, UiToolKind};

use super::types::{
    SafeValidateUiMessagesResult, UiMessageError, UiSchemaValidator,
    ValidateUiMessagesSchemaOptions,
};

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

fn validate_ui_message_part(part: &UiMessagePart) -> Result<(), String> {
    part.validate()
}
