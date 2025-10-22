//! Gemini request conversion helpers (pure functions)
//!
//! These helpers convert unified ChatMessage/Tool structures into
//! Gemini's typed request structures without performing HTTP calls.

use crate::error::LlmError;
use crate::types::{ChatMessage, MessageContent, Tool};
use base64::Engine;

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
    crate::utils::guess_mime_from_path_or_url(url)
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
                    crate::types::ContentPart::Image { source, .. }
                    | crate::types::ContentPart::Audio { source, .. }
                    | crate::types::ContentPart::File { source, .. } => {
                        match source {
                            crate::types::chat::MediaSource::Url { url } => {
                                if url.starts_with("data:") {
                                    // Parse data URL
                                    if let Some((mime_type, data)) = parse_data_url(url) {
                                        parts.push(Part::InlineData {
                                            inline_data: super::types::Blob { mime_type, data },
                                        });
                                    }
                                } else if url.starts_with("gs://") || url.starts_with("https://") {
                                    // File URL
                                    parts.push(Part::FileData {
                                        file_data: super::types::FileData {
                                            file_uri: url.clone(),
                                            mime_type: Some(guess_mime_type(url)),
                                        },
                                    });
                                }
                            }
                            crate::types::chat::MediaSource::Base64 { data } => {
                                // Inline base64 data
                                let mime_type = match content_part {
                                    crate::types::ContentPart::Image { .. } => "image/jpeg",
                                    crate::types::ContentPart::Audio { media_type, .. } => {
                                        media_type.as_deref().unwrap_or("audio/wav")
                                    }
                                    crate::types::ContentPart::File { media_type, .. } => {
                                        media_type.as_str()
                                    }
                                    _ => "application/octet-stream",
                                };
                                parts.push(Part::InlineData {
                                    inline_data: super::types::Blob {
                                        mime_type: mime_type.to_string(),
                                        data: data.clone(),
                                    },
                                });
                            }
                            crate::types::chat::MediaSource::Binary { data } => {
                                // Convert binary to base64
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                let mime_type = match content_part {
                                    crate::types::ContentPart::Image { .. } => "image/jpeg",
                                    crate::types::ContentPart::Audio { media_type, .. } => {
                                        media_type.as_deref().unwrap_or("audio/wav")
                                    }
                                    crate::types::ContentPart::File { media_type, .. } => {
                                        media_type.as_str()
                                    }
                                    _ => "application/octet-stream",
                                };
                                parts.push(Part::InlineData {
                                    inline_data: super::types::Blob {
                                        mime_type: mime_type.to_string(),
                                        data: encoded,
                                    },
                                });
                            }
                        }
                    }
                }
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(v) => {
            parts.push(Part::Text {
                text: serde_json::to_string(v).unwrap_or_default(),
                thought: None,
            });
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
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => v.clone(),
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
                #[cfg(feature = "structured-messages")]
                MessageContent::Json(v) => serde_json::to_string(v).unwrap_or_default(),
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

    // Merge common_params and generation_config
    let mut merged_generation_config = config.generation_config.clone().unwrap_or_default();

    // Common params take precedence over generation_config defaults
    if let Some(temp) = config.common_params.temperature {
        merged_generation_config.temperature = Some(temp);
    }
    if let Some(max_tokens) = config.common_params.max_tokens {
        merged_generation_config.max_output_tokens = Some(max_tokens as i32);
    }
    if let Some(top_p) = config.common_params.top_p {
        merged_generation_config.top_p = Some(top_p);
    }
    if let Some(stops) = &config.common_params.stop_sequences {
        merged_generation_config.stop_sequences = Some(stops.clone());
    }

    Ok(GenerateContentRequest {
        model: config.model.clone(),
        contents,
        system_instruction,
        tools: gemini_tools,
        tool_config: None,
        safety_settings: config.safety_settings.clone(),
        generation_config: Some(merged_generation_config),
        cached_content: None,
    })
}

/// Convert provider-agnostic ToolChoice to Gemini format
///
/// # Gemini Format
///
/// Gemini uses `tool_config` with `function_calling_config`:
/// - `Auto` → `{"mode": "AUTO"}`
/// - `Required` → `{"mode": "ANY"}`
/// - `None` → `{"mode": "NONE"}`
/// - `Tool { name }` → `{"mode": "ANY", "allowed_function_names": ["..."]}`
///
/// # Example
///
/// ```rust
/// use siumai::types::ToolChoice;
/// use siumai::providers::gemini::convert::convert_tool_choice;
///
/// let choice = ToolChoice::tool("weather");
/// let gemini_format = convert_tool_choice(&choice);
/// ```
pub fn convert_tool_choice(choice: &crate::types::ToolChoice) -> serde_json::Value {
    use crate::types::ToolChoice;

    match choice {
        ToolChoice::Auto => serde_json::json!({
            "function_calling_config": {
                "mode": "AUTO"
            }
        }),
        ToolChoice::Required => serde_json::json!({
            "function_calling_config": {
                "mode": "ANY"
            }
        }),
        ToolChoice::None => serde_json::json!({
            "function_calling_config": {
                "mode": "NONE"
            }
        }),
        ToolChoice::Tool { name } => serde_json::json!({
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [name]
            }
        }),
    }
}

#[cfg(test)]
mod tool_choice_tests {
    use super::*;

    #[test]
    fn test_convert_tool_choice() {
        use crate::types::ToolChoice;

        // Test Auto
        let result = convert_tool_choice(&ToolChoice::Auto);
        assert_eq!(
            result,
            serde_json::json!({
                "function_calling_config": {
                    "mode": "AUTO"
                }
            })
        );

        // Test Required (maps to "ANY" in Gemini)
        let result = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(
            result,
            serde_json::json!({
                "function_calling_config": {
                    "mode": "ANY"
                }
            })
        );

        // Test None
        let result = convert_tool_choice(&ToolChoice::None);
        assert_eq!(
            result,
            serde_json::json!({
                "function_calling_config": {
                    "mode": "NONE"
                }
            })
        );

        // Test Tool
        let result = convert_tool_choice(&ToolChoice::tool("weather"));
        assert_eq!(
            result,
            serde_json::json!({
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": ["weather"]
                }
            })
        );
    }
}
