//! Response-side adapter for projecting legacy chat `ContentPart` payloads into generated output.
//!
//! The legacy stable `ContentPart` enum is a compatibility carrier with request-side
//! `providerOptions` and response-side `providerMetadata` on several variants. This module is the
//! explicit response compatibility seam: it preserves response metadata, ignores request options,
//! and rejects ambiguous legacy carriers instead of treating `ContentPart` as the canonical output
//! model.

use crate::types::{ChatResponse, ContentPart, FilePartSource, MediaSource, MessageContent};

use super::{
    CustomOutput, FileOutput, GenerateTextContentPart, GenerateTextContentPartProjectionError,
    GeneratedFile, ReasoningFileOutput, ReasoningOutput, Source, TextOutput, ToolCall, ToolResult,
};

/// Project a stable response content part into the canonical non-V4 generated output content.
///
/// This is a response-side projection. It preserves `providerMetadata`, intentionally ignores
/// request-side `providerOptions`, and fails when a legacy `ContentPart` would lose information.
pub fn project_response_content_part_to_generate_text_content_part(
    part: &ContentPart,
) -> Result<GenerateTextContentPart, GenerateTextContentPartProjectionError> {
    match part {
        ContentPart::Text {
            text,
            provider_metadata,
            ..
        } => {
            let mut output = TextOutput::new(text.clone());
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::Custom {
            kind,
            provider_metadata,
            ..
        } => {
            let mut output = CustomOutput::new(kind.clone());
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::Reasoning {
            text,
            provider_metadata,
            ..
        } => {
            let mut output = ReasoningOutput::new(text.clone());
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::ReasoningFile {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            let file = generated_file_from_media_source(source, media_type, "reasoning-file")?;
            let mut output = ReasoningFileOutput::new(file);
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::Source { .. } => Ok(Source::try_from(part)
            .map(GenerateTextContentPart::Source)
            .map_err(
                |_| GenerateTextContentPartProjectionError::UnsupportedContentPart {
                    part_type: "source",
                    reason: "source content could not be projected",
                },
            )?),
        ContentPart::File {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            let file = generated_file_from_file_source(source, media_type, "file")?;
            let mut output = FileOutput::new(file);
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            provider_executed,
            dynamic,
            invalid,
            error,
            title,
            provider_metadata,
            ..
        } => {
            let mut output =
                ToolCall::new(tool_call_id.clone(), tool_name.clone(), arguments.clone());
            if let Some(provider_executed) = provider_executed {
                output = output.with_provider_executed(*provider_executed);
            }
            if let Some(dynamic) = dynamic {
                output = output.with_dynamic(*dynamic);
            }
            if let Some(invalid) = invalid {
                output = output.with_invalid(*invalid);
            }
            if let Some(error) = error {
                output = output.with_error(error.clone());
            }
            if let Some(title) = title {
                output = output.with_title(title.clone());
            }
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            output,
            input,
            provider_executed,
            dynamic,
            preliminary,
            title,
            provider_metadata,
            ..
        } => {
            let Some(input) = input.clone() else {
                return Err(
                    GenerateTextContentPartProjectionError::UnsupportedContentPart {
                        part_type: "tool-result",
                        reason: "tool-result generated output requires original input",
                    },
                );
            };
            let mut projected = ToolResult::new(
                tool_call_id.clone(),
                tool_name.clone(),
                input,
                output.clone(),
            );
            if let Some(provider_executed) = provider_executed {
                projected = projected.with_provider_executed(*provider_executed);
            }
            if let Some(dynamic) = dynamic {
                projected = projected.with_dynamic(*dynamic);
            }
            if let Some(preliminary) = preliminary {
                projected = projected.with_preliminary(*preliminary);
            }
            if let Some(title) = title {
                projected = projected.with_title(title.clone());
            }
            if let Some(provider_metadata) = provider_metadata.clone() {
                projected = projected.with_provider_metadata(provider_metadata);
            }
            Ok(projected.into())
        }
        ContentPart::Image { .. } => Err(unsupported_generated_output_part(
            "image",
            "image content is ambiguous in generated text output projection",
        )),
        ContentPart::Audio { .. } => Err(unsupported_generated_output_part(
            "audio",
            "audio content is ambiguous in generated text output projection",
        )),
        ContentPart::ToolApprovalRequest { .. } => Err(unsupported_generated_output_part(
            "tool-approval-request",
            "tool approval request output requires the original tool call",
        )),
        ContentPart::ToolApprovalResponse { .. } => Err(unsupported_generated_output_part(
            "tool-approval-response",
            "tool approval response output requires the original tool call",
        )),
    }
}

/// Project stable response content into canonical non-V4 generated output content parts.
pub fn project_response_content_to_generate_text_content_parts(
    content: &MessageContent,
) -> Result<Vec<GenerateTextContentPart>, GenerateTextContentPartProjectionError> {
    match content {
        MessageContent::Text(text) => Ok(vec![TextOutput::new(text.clone()).into()]),
        MessageContent::MultiModal(parts) => parts
            .iter()
            .map(project_response_content_part_to_generate_text_content_part)
            .collect(),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => Err(
            GenerateTextContentPartProjectionError::UnsupportedMessageContent {
                content_type: "json",
                reason: "structured JSON content is not generated text output",
            },
        ),
    }
}

/// Project a chat response into canonical non-V4 generated output content parts.
pub fn project_chat_response_to_generate_text_content_parts(
    response: &ChatResponse,
) -> Result<Vec<GenerateTextContentPart>, GenerateTextContentPartProjectionError> {
    project_response_content_to_generate_text_content_parts(&response.content)
}

fn generated_file_from_media_source(
    source: &MediaSource,
    media_type: &str,
    part_type: &'static str,
) -> Result<GeneratedFile, GenerateTextContentPartProjectionError> {
    source.as_base64().map_or_else(
        || {
            Err(
                GenerateTextContentPartProjectionError::UnsupportedContentPart {
                    part_type,
                    reason: "generated file output requires base64 or binary data",
                },
            )
        },
        |base64| Ok(GeneratedFile::from_base64(base64, media_type)),
    )
}

fn generated_file_from_file_source(
    source: &FilePartSource,
    media_type: &str,
    part_type: &'static str,
) -> Result<GeneratedFile, GenerateTextContentPartProjectionError> {
    source.as_base64().map_or_else(
        || {
            Err(
                GenerateTextContentPartProjectionError::UnsupportedContentPart {
                    part_type,
                    reason: "generated file output requires base64 or binary data",
                },
            )
        },
        |base64| Ok(GeneratedFile::from_base64(base64, media_type)),
    )
}

fn unsupported_generated_output_part(
    part_type: &'static str,
    reason: &'static str,
) -> GenerateTextContentPartProjectionError {
    GenerateTextContentPartProjectionError::UnsupportedContentPart { part_type, reason }
}
