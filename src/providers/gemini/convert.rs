//! Gemini request conversion helpers (pure functions)
//!
//! These helpers convert unified ChatMessage/Tool structures into
//! Gemini's typed request structures without performing HTTP calls.

use crate::error::LlmError;
use crate::types::{ChatMessage, MessageContent, Tool};

use super::types::{
    Content, FunctionCall, FunctionDeclaration, GeminiConfig, GeminiTool, GenerateContentRequest,
    Part,
};

/// Parse data URL to extract MIME type and base64 data
fn parse_data_url(data_url: &str) -> Option<(String, String)> {
    if let Some(comma_pos) = data_url.find(',') {
        let header = &data_url[5..comma_pos]; // Skip "data:"
        let data = &data_url[comma_pos + 1..];

        // Extract MIME type
        let mime_type = if let Some(semicolon_pos) = header.find(';') {
            header[..semicolon_pos].to_string()
        } else {
            header.to_string()
        };

        Some((mime_type, data.to_string()))
    } else {
        None
    }
}

/// Guess MIME type by URL/extension via mime_guess; fallback to octet-stream
fn guess_mime_type(url: &str) -> String {
    crate::utils::mime::guess_mime_from_path_or_url(url)
        .unwrap_or_else(|| "application/octet-stream".to_string())
}

/// Convert `ChatMessage` to Gemini Content
pub fn convert_message_to_content(message: &ChatMessage) -> Result<Content, LlmError> {
    let role = match message.role {
        crate::types::MessageRole::User => Some("user".to_string()),
        crate::types::MessageRole::Assistant => Some("model".to_string()),
        crate::types::MessageRole::System => None, // System messages handled separately
        _ => {
            return Err(LlmError::InvalidInput(format!(
                "Unsupported role: {:?}",
                message.role
            )));
        }
    };

    let mut parts = Vec::new();

    match &message.content {
        MessageContent::Text(text) => {
            if !text.is_empty() {
                parts.push(Part::Text {
                    text: text.clone(),
                    thought: None,
                });
            }
        }
        MessageContent::MultiModal(content_parts) => {
            for content_part in content_parts {
                match content_part {
                    crate::types::ContentPart::Text { text } => {
                        if !text.is_empty() {
                            parts.push(Part::Text {
                                text: text.clone(),
                                thought: None,
                            });
                        }
                    }
                    crate::types::ContentPart::Image {
                        image_url,
                        detail: _,
                    } => {
                        if image_url.starts_with("data:") {
                            if let Some((mime_type, data)) = parse_data_url(image_url) {
                                parts.push(Part::InlineData {
                                    inline_data: super::types::Blob { mime_type, data },
                                });
                            }
                        } else if image_url.starts_with("gs://")
                            || image_url.starts_with("https://")
                        {
                            parts.push(Part::FileData {
                                file_data: super::types::FileData {
                                    file_uri: image_url.clone(),
                                    mime_type: Some(guess_mime_type(image_url)),
                                },
                            });
                        }
                    }
                    crate::types::ContentPart::Audio {
                        audio_url,
                        format: _,
                    } => {
                        if audio_url.starts_with("data:") {
                            if let Some((mime_type, data)) = parse_data_url(audio_url) {
                                parts.push(Part::InlineData {
                                    inline_data: super::types::Blob { mime_type, data },
                                });
                            }
                        } else if audio_url.starts_with("gs://")
                            || audio_url.starts_with("https://")
                        {
                            parts.push(Part::FileData {
                                file_data: super::types::FileData {
                                    file_uri: audio_url.clone(),
                                    mime_type: Some(guess_mime_type(audio_url)),
                                },
                            });
                        }
                    }
                }
            }
        }
    }

    // Tool calls
    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            if let Some(function) = &tool_call.function {
                let args = serde_json::from_str(&function.arguments).ok();
                parts.push(Part::FunctionCall {
                    function_call: FunctionCall {
                        name: function.name.clone(),
                        args,
                    },
                });
            }
        }
    }

    // Tool results
    if let Some(tool_call_id) = &message.tool_call_id {
        let response = match &message.content {
            MessageContent::Text(text) => serde_json::json!(text),
            _ => serde_json::json!({}),
        };
        parts.push(Part::FunctionResponse {
            function_response: super::types::FunctionResponse {
                name: tool_call_id.clone(),
                response,
            },
        });
    }

    if parts.is_empty() {
        return Err(LlmError::InvalidInput("Message has no content".to_string()));
    }

    Ok(Content { role, parts })
}

/// Convert Tools to Gemini Tools
pub fn convert_tools_to_gemini(tools: &[Tool]) -> Result<Vec<GeminiTool>, LlmError> {
    let mut gemini_tools = Vec::new();
    let mut function_declarations = Vec::new();

    for tool in tools {
        if tool.r#type == "function" {
            let parameters = tool.function.parameters.clone();
            function_declarations.push(FunctionDeclaration {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: Some(parameters),
                response: None,
            });
        } else {
            return Err(LlmError::UnsupportedOperation(format!(
                "Tool type {} not supported by Gemini",
                tool.r#type
            )));
        }
    }

    if !function_declarations.is_empty() {
        gemini_tools.push(GeminiTool::FunctionDeclarations {
            function_declarations,
        });
    }

    Ok(gemini_tools)
}

/// Build the request body for Gemini API from unified request
pub fn build_request_body(
    config: &GeminiConfig,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
) -> Result<GenerateContentRequest, LlmError> {
    let mut contents = Vec::new();
    let mut system_instruction = None;

    for message in messages {
        if message.role == crate::types::MessageRole::System {
            // Extract system instruction text (flatten multimodal to text)
            let system_text = match &message.content {
                MessageContent::Text(text) => text.clone(),
                MessageContent::MultiModal(parts) => parts
                    .iter()
                    .filter_map(|part| {
                        if let crate::types::ContentPart::Text { text } = part {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            };
            if !system_text.is_empty() {
                system_instruction = Some(Content {
                    role: None,
                    parts: vec![Part::Text {
                        text: system_text,
                        thought: None,
                    }],
                });
            }
        } else {
            contents.push(convert_message_to_content(message)?);
        }
    }

    let gemini_tools = if let Some(ts) = tools {
        if !ts.is_empty() {
            Some(convert_tools_to_gemini(ts)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(GenerateContentRequest {
        model: config.model.clone(),
        contents,
        system_instruction,
        tools: gemini_tools,
        tool_config: None,
        safety_settings: config.safety_settings.clone(),
        generation_config: config.generation_config.clone(),
        cached_content: None,
    })
}
