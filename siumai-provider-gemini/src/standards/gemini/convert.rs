//! Gemini request conversion helpers (protocol layer)
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
        crate::types::MessageRole::Tool => Some("user".to_string()), // Tool results are sent as user functionResponse parts
        crate::types::MessageRole::System | crate::types::MessageRole::Developer => {
            return Err(LlmError::InvalidInput(
                "System/developer messages should be handled by build_request_body()".to_string(),
            ));
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
                    crate::types::ContentPart::ToolCall { .. } => {
                        // Tool calls are handled separately in convert_messages
                        // Skip them here as they're not part of content array
                    }
                    crate::types::ContentPart::ToolResult { .. } => {
                        // Tool results are handled separately in convert_messages
                        // Skip them here as they're not part of content array
                    }
                    crate::types::ContentPart::Reasoning { .. } => {
                        // Reasoning/thinking is handled separately
                        // Skip it here as it's not part of content array
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

    // Tool calls (extract from content)
    let tool_calls = message.tool_calls();
    for part in tool_calls {
        if let crate::types::ContentPart::ToolCall {
            tool_name,
            arguments,
            ..
        } = part
        {
            parts.push(Part::FunctionCall {
                function_call: FunctionCall {
                    name: tool_name.clone(),
                    args: Some(arguments.clone()),
                },
            });
        }
    }

    // Tool results (extract from content)
    let tool_results = message.tool_results();
    for part in tool_results {
        if let crate::types::ContentPart::ToolResult {
            tool_name, output, ..
        } = part
        {
            let response = output.to_json_value();
            parts.push(Part::FunctionResponse {
                function_response: super::types::FunctionResponse {
                    name: tool_name.clone(),
                    response,
                },
            });
        }
    }

    if parts.is_empty() {
        return Err(LlmError::InvalidInput("Message has no content".to_string()));
    }

    Ok(Content { role, parts })
}

/// Convert Tools to Gemini Tools
pub fn convert_tools_to_gemini(model: &str, tools: &[Tool]) -> Result<Vec<GeminiTool>, LlmError> {
    let mut gemini_tools = Vec::new();
    let mut function_declarations = Vec::new();

    fn is_gemini_2_or_newer(model: &str) -> bool {
        // Follow Vercel AI SDK heuristics (best-effort; model IDs are strings).
        model.contains("gemini-2") || model.contains("gemini-3") || model.ends_with("-latest")
    }

    fn supports_dynamic_retrieval(model: &str) -> bool {
        // Vercel AI SDK: only gemini-1.5-flash (excluding -8b) supports dynamic retrieval config.
        model.contains("gemini-1.5-flash") && !model.contains("-8b")
    }

    fn supports_file_search(model: &str) -> bool {
        // Vercel AI SDK: File Search is available on Gemini 2.5 models.
        model.contains("gemini-2.5")
    }

    fn parse_dynamic_retrieval_config(
        args: &serde_json::Value,
    ) -> Option<super::types::DynamicRetrievalConfig> {
        let obj = args.as_object()?;
        let mode = obj.get("mode").and_then(|v| v.as_str());
        let dynamic_threshold = obj.get("dynamicThreshold").and_then(|v| v.as_f64());

        if mode.is_none() && dynamic_threshold.is_none() {
            return None;
        }

        let mode = match mode {
            Some("MODE_DYNAMIC") => super::types::DynamicRetrievalMode::Dynamic,
            Some("MODE_UNSPECIFIED") => super::types::DynamicRetrievalMode::Unspecified,
            _ => super::types::DynamicRetrievalMode::Unspecified,
        };

        Some(super::types::DynamicRetrievalConfig {
            mode,
            dynamic_threshold: dynamic_threshold.map(|v| v as f32),
        })
    }

    for tool in tools {
        match tool {
            Tool::Function { function } => {
                let parameters = function.parameters.clone();
                function_declarations.push(FunctionDeclaration {
                    name: function.name.clone(),
                    description: function.description.clone(),
                    parameters: Some(parameters),
                    response: None,
                });
            }
            Tool::ProviderDefined(provider_tool) => {
                // Handle Google/Gemini provider-defined tools
                if matches!(provider_tool.provider(), Some("google" | "gemini")) {
                    match provider_tool.tool_type() {
                        Some("code_execution") => {
                            // Enable code execution tool
                            function_declarations.shrink_to_fit();
                            gemini_tools.push(GeminiTool::CodeExecution {
                                code_execution: super::types::CodeExecution {},
                            });
                        }
                        Some("google_search") => {
                            if is_gemini_2_or_newer(model) {
                                // Gemini 2.0+ Google Search grounding
                                gemini_tools.push(GeminiTool::GoogleSearch {
                                    google_search: super::types::GoogleSearch {},
                                });
                            } else {
                                // Legacy grounding (Gemini 1.5): googleSearchRetrieval
                                let dynamic_retrieval_config = if supports_dynamic_retrieval(model)
                                {
                                    parse_dynamic_retrieval_config(&provider_tool.args)
                                } else {
                                    None
                                };
                                gemini_tools.push(GeminiTool::GoogleSearchRetrieval {
                                    google_search_retrieval: super::types::GoogleSearchRetrieval {
                                        dynamic_retrieval_config,
                                    },
                                });
                            }
                        }
                        Some("google_search_retrieval") => {
                            // Gemini 1.5 Google Search Retrieval (legacy)
                            let dynamic_retrieval_config =
                                parse_dynamic_retrieval_config(&provider_tool.args);
                            gemini_tools.push(GeminiTool::GoogleSearchRetrieval {
                                google_search_retrieval: super::types::GoogleSearchRetrieval {
                                    dynamic_retrieval_config,
                                },
                            });
                        }
                        Some("google_maps") => {
                            if is_gemini_2_or_newer(model) {
                                gemini_tools.push(GeminiTool::GoogleMaps {
                                    google_maps: super::types::GoogleMaps {},
                                });
                            }
                        }
                        Some("url_context") => {
                            if is_gemini_2_or_newer(model) {
                                gemini_tools.push(GeminiTool::UrlContext {
                                    url_context: super::types::UrlContext {},
                                });
                            }
                        }
                        Some("enterprise_web_search") => {
                            if is_gemini_2_or_newer(model) {
                                gemini_tools.push(GeminiTool::EnterpriseWebSearch {
                                    enterprise_web_search: super::types::EnterpriseWebSearch {},
                                });
                            }
                        }
                        Some("vertex_rag_store") => {
                            if is_gemini_2_or_newer(model) {
                                let obj = provider_tool.args.as_object();
                                let rag_corpus = obj
                                    .and_then(|o| o.get("ragCorpus"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let similarity_top_k = obj
                                    .and_then(|o| o.get("topK"))
                                    .and_then(|v| v.as_u64())
                                    .map(|v| v as u32);

                                if let Some(rag_corpus) = rag_corpus {
                                    gemini_tools.push(GeminiTool::Retrieval {
                                        retrieval: super::types::Retrieval {
                                            vertex_rag_store: super::types::VertexRagStore {
                                                rag_resources: super::types::VertexRagResources {
                                                    rag_corpus,
                                                },
                                                similarity_top_k,
                                            },
                                        },
                                    });
                                }
                            }
                        }
                        Some("file_search") => {
                            if supports_file_search(model) {
                                let obj = provider_tool.args.as_object();
                                let file_search_store_names = obj
                                    .and_then(|o| o.get("fileSearchStoreNames"))
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                            .collect::<Vec<_>>()
                                    });
                                let top_k = obj
                                    .and_then(|o| o.get("topK"))
                                    .and_then(|v| v.as_u64())
                                    .map(|v| v as u32);
                                let metadata_filter =
                                    obj.and_then(|o| o.get("metadataFilter")).cloned();

                                gemini_tools.push(GeminiTool::FileSearch {
                                    file_search: super::types::FileSearch {
                                        file_search_store_names,
                                        top_k,
                                        metadata_filter,
                                    },
                                });
                            }
                        }
                        _ => {
                            // Unknown Google tool type; ignore for now
                        }
                    }
                    continue;
                }
                // Ignore provider-defined tools from other providers
                continue;
            }
        }
    }

    if !function_declarations.is_empty() {
        gemini_tools.push(GeminiTool::FunctionDeclarations {
            function_declarations,
        });
    }

    Ok(gemini_tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_google_provider_defined_tools() {
        // code_execution
        let tools = vec![Tool::provider_defined(
            "google.code_execution",
            "code_execution",
        )];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::CodeExecution { .. }))
        );

        // google_search
        let tools = vec![Tool::provider_defined(
            "google.google_search",
            "google_search",
        )];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleSearch { .. }))
        );

        // google_search_retrieval
        let tools = vec![Tool::provider_defined(
            "google.google_search_retrieval",
            "google_search_retrieval",
        )];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleSearchRetrieval { .. }))
        );

        // google_maps
        let tools = vec![Tool::provider_defined("google.google_maps", "google_maps")];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleMaps { .. }))
        );

        // url_context
        let tools = vec![Tool::provider_defined("google.url_context", "url_context")];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::UrlContext { .. }))
        );

        // enterprise_web_search
        let tools = vec![Tool::provider_defined(
            "google.enterprise_web_search",
            "enterprise_web_search",
        )];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::EnterpriseWebSearch { .. }))
        );

        // vertex_rag_store
        let tools = vec![Tool::provider_defined("google.vertex_rag_store", "vertex_rag_store")
            .with_args(serde_json::json!({ "ragCorpus": "projects/p/locations/l/ragCorpora/c", "topK": 3 }))];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::Retrieval { .. }))
        );

        // file_search
        let tools = vec![
            Tool::provider_defined("google.file_search", "file_search").with_args(
                serde_json::json!({
                    "fileSearchStoreNames": ["fileSearchStores/abc123"],
                    "topK": 5
                }),
            ),
        ];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::FileSearch { .. }))
        );
    }

    #[test]
    fn google_search_maps_to_legacy_retrieval_on_gemini_1_5() {
        let tools = vec![Tool::provider_defined(
            "google.google_search",
            "google_search",
        )];
        let mapped = convert_tools_to_gemini("gemini-1.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleSearchRetrieval { .. })),
            "gemini-1.5 should use googleSearchRetrieval for google_search"
        );
        assert!(
            !mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleSearch { .. })),
            "gemini-1.5 should not emit googleSearch"
        );
    }

    #[test]
    fn file_search_is_ignored_when_model_does_not_support_it() {
        let tools = vec![
            Tool::provider_defined("google.file_search", "file_search").with_args(
                serde_json::json!({
                    "fileSearchStoreNames": ["fileSearchStores/abc123"],
                    "topK": 5
                }),
            ),
        ];

        let mapped = convert_tools_to_gemini("gemini-2.0-flash", &tools).expect("map ok");
        assert!(
            !mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::FileSearch { .. })),
            "file_search should be dropped for non-2.5 models"
        );
    }

    #[test]
    fn google_maps_is_ignored_on_gemini_1_5() {
        let tools = vec![Tool::provider_defined("google.google_maps", "google_maps")];
        let mapped = convert_tools_to_gemini("gemini-1.5-pro", &tools).expect("map ok");
        assert!(
            !mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleMaps { .. })),
            "google_maps should be dropped for gemini-1.5 models"
        );
    }
}

/// Build the request body for Gemini API from unified request
pub fn build_request_body(
    config: &GeminiConfig,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
) -> Result<GenerateContentRequest, LlmError> {
    let mut contents = Vec::new();
    let mut system_text_parts: Vec<String> = Vec::new();
    let mut system_messages_allowed = true;

    for message in messages {
        match message.role {
            crate::types::MessageRole::System | crate::types::MessageRole::Developer => {
                if !system_messages_allowed {
                    return Err(LlmError::InvalidParameter(
                        "System/developer messages are only supported at the beginning of the conversation".to_string(),
                    ));
                }

                // Extract system instruction text (flatten multimodal to text).
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

                if !system_text.trim().is_empty() {
                    system_text_parts.push(system_text);
                }
            }
            _ => {
                system_messages_allowed = false;
                contents.push(convert_message_to_content(message)?);
            }
        }
    }

    // Prefer unified common_params.model as the single source of truth,
    // fallback to config.model for backward compatibility.
    let model = if !config.common_params.model.is_empty() {
        config.common_params.model.clone()
    } else {
        config.model.clone()
    };

    // Merge common_params and generation_config
    let mut merged_generation_config = config.generation_config.clone().unwrap_or_default();

    // Common params take precedence over generation_config defaults
    if let Some(temp) = config.common_params.temperature {
        let v = (temp as f64 * 1_000_000.0).round() / 1_000_000.0;
        merged_generation_config.temperature = Some(v);
    }
    if let Some(max_tokens) = config.common_params.max_tokens {
        merged_generation_config.max_output_tokens = Some(max_tokens as i32);
    }
    if let Some(top_p) = config.common_params.top_p {
        let v = (top_p as f64 * 1_000_000.0).round() / 1_000_000.0;
        merged_generation_config.top_p = Some(v);
    }
    if let Some(stops) = &config.common_params.stop_sequences {
        merged_generation_config.stop_sequences = Some(stops.clone());
    }

    let gemini_tools = if let Some(ts) = tools {
        if !ts.is_empty() {
            Some(convert_tools_to_gemini(&model, ts)?)
        } else {
            None
        }
    } else {
        None
    };

    // Align with Vercel AI SDK:
    // - System instruction is supported as a dedicated field only for Gemini models.
    // - Gemma models do not support `systemInstruction`; inline the system text into the first user message.
    let system_text = system_text_parts
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut system_instruction = if system_text.is_empty() {
        None
    } else {
        Some(Content {
            role: None,
            parts: vec![Part::Text {
                text: system_text.clone(),
                thought: None,
            }],
        })
    };

    fn is_gemma_model(model: &str) -> bool {
        model.to_ascii_lowercase().starts_with("gemma-")
    }

    if is_gemma_model(&model) && !system_text.is_empty() {
        let prefix = format!("{system_text}\n\n");

        // Prefer inserting into the first user message, otherwise create one.
        if let Some(first) = contents.first_mut() {
            if first.role.as_deref() == Some("user") {
                first.parts.insert(
                    0,
                    Part::Text {
                        text: prefix,
                        thought: None,
                    },
                );
            } else {
                contents.insert(
                    0,
                    Content {
                        role: Some("user".to_string()),
                        parts: vec![Part::Text {
                            text: prefix,
                            thought: None,
                        }],
                    },
                );
            }
        } else {
            contents.push(Content {
                role: Some("user".to_string()),
                parts: vec![Part::Text {
                    text: prefix,
                    thought: None,
                }],
            });
        }

        system_instruction = None;
    }

    Ok(GenerateContentRequest {
        model,
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
/// Gemini uses `toolConfig` with `functionCallingConfig`:
/// - `Auto` → `{"mode": "AUTO"}`
/// - `Required` → `{"mode": "ANY"}`
/// - `None` → `{"mode": "NONE"}`
/// - `Tool { name }` → `{"mode": "ANY", "allowedFunctionNames": ["..."]}`
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
            "functionCallingConfig": {
                "mode": "AUTO"
            }
        }),
        ToolChoice::Required => serde_json::json!({
            "functionCallingConfig": {
                "mode": "ANY"
            }
        }),
        ToolChoice::None => serde_json::json!({
            "functionCallingConfig": {
                "mode": "NONE"
            }
        }),
        ToolChoice::Tool { name } => serde_json::json!({
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [name]
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
                "functionCallingConfig": {
                    "mode": "AUTO"
                }
            })
        );

        // Test Required (maps to "ANY" in Gemini)
        let result = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(
            result,
            serde_json::json!({
                "functionCallingConfig": {
                    "mode": "ANY"
                }
            })
        );

        // Test None
        let result = convert_tool_choice(&ToolChoice::None);
        assert_eq!(
            result,
            serde_json::json!({
                "functionCallingConfig": {
                    "mode": "NONE"
                }
            })
        );

        // Test Tool
        let result = convert_tool_choice(&ToolChoice::tool("weather"));
        assert_eq!(
            result,
            serde_json::json!({
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["weather"]
                }
            })
        );
    }
}

#[cfg(test)]
mod system_and_tool_message_tests {
    use super::*;

    #[test]
    fn rejects_system_messages_after_conversation_started() {
        let cfg = GeminiConfig::default();
        let messages = vec![
            crate::types::ChatMessage::user("hi").build(),
            crate::types::ChatMessage::system("sys").build(),
        ];

        let err = build_request_body(&cfg, &messages, None).unwrap_err();
        assert!(matches!(err, LlmError::InvalidParameter(_)));
    }

    #[test]
    fn gemma_inlines_system_instruction_into_first_user_message() {
        let mut cfg = GeminiConfig::default();
        cfg.common_params.model = "gemma-2b-it".to_string();

        let messages = vec![
            crate::types::ChatMessage::system("sys-1").build(),
            crate::types::ChatMessage::system("sys-2").build(),
            crate::types::ChatMessage::user("hi").build(),
        ];

        let req = build_request_body(&cfg, &messages, None).unwrap();
        assert!(req.system_instruction.is_none());
        assert!(!req.contents.is_empty());
        assert_eq!(req.contents[0].role.as_deref(), Some("user"));

        let first_part = req.contents[0].parts.first().unwrap();
        match first_part {
            Part::Text { text, .. } => {
                assert!(text.contains("sys-1"));
                assert!(text.contains("sys-2"));
                assert!(text.ends_with("\n\n"));
            }
            _ => panic!("expected text part"),
        }
    }

    #[test]
    fn tool_role_messages_are_supported_as_function_responses() {
        let cfg = GeminiConfig::default();
        let messages =
            vec![crate::types::ChatMessage::tool_result_text("call_1", "search", "ok").build()];

        let req = build_request_body(&cfg, &messages, None).unwrap();
        assert_eq!(req.contents.len(), 1);
        assert_eq!(req.contents[0].role.as_deref(), Some("user"));

        let first_part = req.contents[0].parts.first().unwrap();
        assert!(matches!(first_part, Part::FunctionResponse { .. }));
    }
}
