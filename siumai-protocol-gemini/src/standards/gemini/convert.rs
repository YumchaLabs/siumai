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
use crate::types::{ToolResultContentPart, ToolResultOutput};

fn extract_thought_signature(
    provider_metadata: &Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Option<String> {
    let map = provider_metadata.as_ref()?;
    for v in map.values() {
        if let Some(sig) = v.get("thoughtSignature").and_then(|s| s.as_str()) {
            let sig = sig.trim();
            if !sig.is_empty() {
                return Some(sig.to_string());
            }
        }
    }
    // Also accept a flat shape: { "thoughtSignature": "..." }
    if let Some(sig) = map
        .get("thoughtSignature")
        .and_then(|s| s.as_str())
        .map(str::trim)
        && !sig.is_empty()
    {
        return Some(sig.to_string());
    }
    None
}

fn is_empty_object_schema(json_schema: &serde_json::Value) -> bool {
    let Some(obj) = json_schema.as_object() else {
        return false;
    };
    if obj.get("type").and_then(|v| v.as_str()) != Some("object") {
        return false;
    }
    let has_no_properties = obj
        .get("properties")
        .and_then(|v| v.as_object())
        .is_none_or(|m| m.is_empty());
    let has_no_additional_properties = obj.get("additionalProperties").is_none();
    has_no_properties && has_no_additional_properties
}

/// Convert JSON Schema 7 into an OpenAPI Schema 3.0 object (Vercel-aligned).
///
/// This is intentionally a minimal conversion that matches the semantics used in
/// `repo-ref/ai/packages/google/src/convert-json-schema-to-openapi-schema.ts`.
fn convert_json_schema_to_openapi_schema(
    json_schema: &serde_json::Value,
    is_root: bool,
) -> Option<serde_json::Value> {
    if json_schema.is_null() {
        return None;
    }

    // Handle empty object schemas: undefined at root, preserved when nested.
    if is_empty_object_schema(json_schema) {
        if is_root {
            return None;
        }

        let mut out = serde_json::Map::new();
        out.insert("type".to_string(), serde_json::json!("object"));
        if let Some(desc) = json_schema
            .as_object()
            .and_then(|o| o.get("description"))
            .and_then(|v| v.as_str())
        {
            out.insert("description".to_string(), serde_json::json!(desc));
        }
        return Some(serde_json::Value::Object(out));
    }

    if json_schema.is_boolean() {
        return Some(serde_json::json!({ "type": "boolean", "properties": {} }));
    }

    let obj = json_schema.as_object()?;

    let mut result = serde_json::Map::new();

    if let Some(desc) = obj.get("description") {
        result.insert("description".to_string(), desc.clone());
    }
    if let Some(required) = obj.get("required") {
        result.insert("required".to_string(), required.clone());
    }
    if let Some(format) = obj.get("format") {
        result.insert("format".to_string(), format.clone());
    }

    if let Some(const_value) = obj.get("const")
        && !const_value.is_null()
    {
        result.insert(
            "enum".to_string(),
            serde_json::Value::Array(vec![const_value.clone()]),
        );
    }

    // Handle type.
    if let Some(type_value) = obj.get("type") {
        match type_value {
            serde_json::Value::String(s) => {
                result.insert("type".to_string(), serde_json::Value::String(s.clone()));
            }
            serde_json::Value::Array(arr) => {
                let mut has_null = false;
                let mut non_null_types: Vec<String> = Vec::new();
                for v in arr {
                    let Some(s) = v.as_str() else {
                        continue;
                    };
                    if s == "null" {
                        has_null = true;
                    } else {
                        non_null_types.push(s.to_string());
                    }
                }

                if non_null_types.is_empty() {
                    result.insert("type".to_string(), serde_json::json!("null"));
                } else {
                    result.insert(
                        "anyOf".to_string(),
                        serde_json::Value::Array(
                            non_null_types
                                .into_iter()
                                .map(|t| serde_json::json!({ "type": t }))
                                .collect(),
                        ),
                    );
                    if has_null {
                        result.insert("nullable".to_string(), serde_json::json!(true));
                    }
                }
            }
            _ => {}
        }
    }

    // Handle enum.
    if let Some(enum_values) = obj.get("enum") {
        result.insert("enum".to_string(), enum_values.clone());
    }

    // Handle properties (omit entries that convert to undefined).
    if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
        let mut mapped = serde_json::Map::new();
        for (k, v) in props {
            if let Some(converted) = convert_json_schema_to_openapi_schema(v, false) {
                mapped.insert(k.clone(), converted);
            }
        }
        result.insert("properties".to_string(), serde_json::Value::Object(mapped));
    }

    // Handle items.
    if let Some(items) = obj.get("items") {
        match items {
            serde_json::Value::Array(arr) => {
                result.insert(
                    "items".to_string(),
                    serde_json::Value::Array(
                        arr.iter()
                            .map(|v| {
                                convert_json_schema_to_openapi_schema(v, false)
                                    .unwrap_or(serde_json::Value::Null)
                            })
                            .collect(),
                    ),
                );
            }
            _ => {
                if let Some(converted) = convert_json_schema_to_openapi_schema(items, false) {
                    result.insert("items".to_string(), converted);
                }
            }
        }
    }

    // Handle allOf.
    if let Some(all_of) = obj.get("allOf").and_then(|v| v.as_array()) {
        result.insert(
            "allOf".to_string(),
            serde_json::Value::Array(
                all_of
                    .iter()
                    .map(|v| {
                        convert_json_schema_to_openapi_schema(v, false)
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect(),
            ),
        );
    }

    // Handle anyOf (nullable special-case).
    if let Some(any_of) = obj.get("anyOf").and_then(|v| v.as_array()) {
        let has_null_type = any_of.iter().any(|schema| {
            schema
                .as_object()
                .and_then(|o| o.get("type"))
                .and_then(|v| v.as_str())
                .is_some_and(|t| t == "null")
        });

        if has_null_type {
            let non_null_schemas: Vec<&serde_json::Value> = any_of
                .iter()
                .filter(|schema| {
                    schema
                        .as_object()
                        .and_then(|o| o.get("type"))
                        .and_then(|v| v.as_str())
                        .is_none_or(|t| t != "null")
                })
                .collect();

            if non_null_schemas.len() == 1 {
                if let Some(converted) =
                    convert_json_schema_to_openapi_schema(non_null_schemas[0], false)
                    && let Some(converted_obj) = converted.as_object()
                {
                    result.insert("nullable".to_string(), serde_json::json!(true));
                    for (k, v) in converted_obj {
                        result.insert(k.clone(), v.clone());
                    }
                }
            } else {
                result.insert(
                    "anyOf".to_string(),
                    serde_json::Value::Array(
                        non_null_schemas
                            .iter()
                            .map(|v| {
                                convert_json_schema_to_openapi_schema(v, false)
                                    .unwrap_or(serde_json::Value::Null)
                            })
                            .collect(),
                    ),
                );
                result.insert("nullable".to_string(), serde_json::json!(true));
            }
        } else {
            result.insert(
                "anyOf".to_string(),
                serde_json::Value::Array(
                    any_of
                        .iter()
                        .map(|v| {
                            convert_json_schema_to_openapi_schema(v, false)
                                .unwrap_or(serde_json::Value::Null)
                        })
                        .collect(),
                ),
            );
        }
    }

    // Handle oneOf.
    if let Some(one_of) = obj.get("oneOf").and_then(|v| v.as_array()) {
        result.insert(
            "oneOf".to_string(),
            serde_json::Value::Array(
                one_of
                    .iter()
                    .map(|v| {
                        convert_json_schema_to_openapi_schema(v, false)
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect(),
            ),
        );
    }

    if let Some(min_length) = obj.get("minLength") {
        result.insert("minLength".to_string(), min_length.clone());
    }

    Some(serde_json::Value::Object(result))
}

/// Convert a JSON Schema 7 root schema into a Gemini-compatible OpenAPI Schema 3.0 object.
///
/// This mirrors Vercel AI SDK's conversion used for `responseFormat` and function tool schemas.
pub fn convert_json_schema_to_openapi_schema_root(
    json_schema: &serde_json::Value,
) -> Option<serde_json::Value> {
    convert_json_schema_to_openapi_schema(json_schema, true)
}

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
                    thought_signature: None,
                });
            }
        }
        MessageContent::MultiModal(content_parts) => {
            for content_part in content_parts {
                match content_part {
                    crate::types::ContentPart::Text {
                        text,
                        provider_metadata,
                    } => {
                        if !text.is_empty() {
                            let thought_signature = extract_thought_signature(provider_metadata);
                            parts.push(Part::Text {
                                text: text.clone(),
                                thought: None,
                                thought_signature,
                            });
                        }
                    }
                    crate::types::ContentPart::Image {
                        source,
                        provider_metadata,
                        ..
                    }
                    | crate::types::ContentPart::Audio {
                        source,
                        provider_metadata,
                        ..
                    }
                    | crate::types::ContentPart::File {
                        source,
                        provider_metadata,
                        ..
                    } => {
                        let thought_signature = extract_thought_signature(provider_metadata);
                        match source {
                            crate::types::chat::MediaSource::Url { url } => {
                                // Vercel AI SDK alignment: assistant messages cannot reference URL-based file data.
                                if role.as_deref() == Some("model") {
                                    return Err(LlmError::InvalidParameter(
                                        "File data URLs in assistant messages are not supported"
                                            .to_string(),
                                    ));
                                }

                                if url.starts_with("data:") {
                                    // Parse data URL
                                    if let Some((mime_type, data)) = parse_data_url(url) {
                                        parts.push(Part::InlineData {
                                            inline_data: super::types::Blob { mime_type, data },
                                            thought_signature,
                                        });
                                    }
                                } else if url.starts_with("gs://")
                                    || url.starts_with("https://")
                                    || url.starts_with("http://")
                                {
                                    // File URL
                                    let mime_type = match content_part {
                                        crate::types::ContentPart::File { media_type, .. } => Some(
                                            if media_type.trim().eq_ignore_ascii_case("image/*") {
                                                "image/jpeg".to_string()
                                            } else {
                                                media_type.clone()
                                            },
                                        ),
                                        crate::types::ContentPart::Audio { media_type, .. } => {
                                            media_type
                                                .clone()
                                                .or_else(|| Some(guess_mime_type(url).to_string()))
                                        }
                                        _ => Some(guess_mime_type(url)),
                                    };
                                    parts.push(Part::FileData {
                                        file_data: super::types::FileData {
                                            file_uri: url.clone(),
                                            mime_type,
                                        },
                                        thought_signature,
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
                                    thought_signature,
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
                                    thought_signature,
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
                    crate::types::ContentPart::Reasoning {
                        text,
                        provider_metadata,
                    } => {
                        // Vercel-aligned: assistant reasoning is represented as a "thought" text part.
                        if role.as_deref() == Some("model") && !text.trim().is_empty() {
                            let thought_signature = extract_thought_signature(provider_metadata);
                            parts.push(Part::Text {
                                text: text.clone(),
                                thought: Some(true),
                                thought_signature,
                            });
                        }
                    }
                    crate::types::ContentPart::ToolApprovalResponse { .. } => {
                        // Tool approval is an out-of-band workflow; Gemini request does not accept it.
                        // Ignore it at request conversion time to keep behavior consistent with other providers.
                    }
                    crate::types::ContentPart::ToolApprovalRequest { .. } => {
                        // Tool approval is an out-of-band workflow; Gemini request does not accept it.
                        // Ignore it at request conversion time to keep behavior consistent with other providers.
                    }
                    crate::types::ContentPart::Source { .. } => {
                        // Sources are not part of Gemini request content; ignore them.
                    }
                }
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(v) => {
            parts.push(Part::Text {
                text: serde_json::to_string(v).unwrap_or_default(),
                thought: None,
                thought_signature: None,
            });
        }
    }

    // Tool calls (extract from content)
    let tool_calls = message.tool_calls();
    for part in tool_calls {
        if let crate::types::ContentPart::ToolCall {
            tool_name,
            arguments,
            provider_metadata,
            ..
        } = part
        {
            let thought_signature = extract_thought_signature(provider_metadata);
            parts.push(Part::FunctionCall {
                function_call: FunctionCall {
                    name: tool_name.clone(),
                    args: Some(arguments.clone()),
                },
                thought_signature,
            });
        }
    }

    // Tool results (extract from content)
    let tool_results = message.tool_results();
    for part in tool_results {
        if let crate::types::ContentPart::ToolResult {
            tool_name,
            output,
            provider_metadata,
            ..
        } = part
        {
            let thought_signature = extract_thought_signature(provider_metadata);

            fn push_function_response_part(
                parts: &mut Vec<Part>,
                tool_name: &str,
                content: serde_json::Value,
                thought_signature: Option<String>,
            ) {
                parts.push(Part::FunctionResponse {
                    function_response: super::types::FunctionResponse {
                        name: tool_name.to_string(),
                        response: serde_json::json!({
                            "name": tool_name,
                            "content": content
                        }),
                    },
                    thought_signature,
                });
            }

            match output {
                ToolResultOutput::Text { value } => {
                    push_function_response_part(
                        &mut parts,
                        tool_name,
                        serde_json::Value::String(value.clone()),
                        thought_signature,
                    );
                }
                ToolResultOutput::Json { value } => {
                    push_function_response_part(
                        &mut parts,
                        tool_name,
                        value.clone(),
                        thought_signature,
                    );
                }
                ToolResultOutput::ExecutionDenied { reason } => {
                    let msg = reason
                        .clone()
                        .unwrap_or_else(|| "Tool execution denied.".to_string());
                    push_function_response_part(
                        &mut parts,
                        tool_name,
                        serde_json::Value::String(msg),
                        thought_signature,
                    );
                }
                ToolResultOutput::ErrorText { value } => {
                    push_function_response_part(
                        &mut parts,
                        tool_name,
                        serde_json::Value::String(value.clone()),
                        thought_signature,
                    );
                }
                ToolResultOutput::ErrorJson { value } => {
                    push_function_response_part(
                        &mut parts,
                        tool_name,
                        value.clone(),
                        thought_signature,
                    );
                }
                ToolResultOutput::Content { value } => {
                    for content_part in value {
                        match content_part {
                            ToolResultContentPart::Text { text } => {
                                push_function_response_part(
                                    &mut parts,
                                    tool_name,
                                    serde_json::Value::String(text.clone()),
                                    thought_signature.clone(),
                                );
                            }
                            ToolResultContentPart::Image {
                                source: crate::types::chat::MediaSource::Base64 { data },
                                ..
                            } => {
                                parts.push(Part::InlineData {
                                    inline_data: super::types::Blob {
                                        mime_type: "image/jpeg".to_string(),
                                        data: data.clone(),
                                    },
                                    thought_signature: thought_signature.clone(),
                                });
                                parts.push(Part::Text {
                                    text: "Tool executed successfully and returned this image as a response"
                                        .to_string(),
                                    thought: None,
                                    thought_signature: thought_signature.clone(),
                                });
                            }
                            content_part @ ToolResultContentPart::Image { .. } => {
                                let as_json =
                                    serde_json::to_string(&content_part).unwrap_or_default();
                                parts.push(Part::Text {
                                    text: as_json,
                                    thought: None,
                                    thought_signature: thought_signature.clone(),
                                });
                            }
                            _ => {
                                let as_json =
                                    serde_json::to_string(content_part).unwrap_or_default();
                                parts.push(Part::Text {
                                    text: as_json,
                                    thought: None,
                                    thought_signature: thought_signature.clone(),
                                });
                            }
                        }
                    }
                }
            }
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
        // Vercel AI SDK: treat explicit "latest" aliases as Gemini 2+ for provider tools.
        let is_latest = matches!(
            model,
            "gemini-flash-latest" | "gemini-flash-lite-latest" | "gemini-pro-latest"
        );
        model.contains("gemini-2") || model.contains("gemini-3") || is_latest
    }

    fn supports_dynamic_retrieval(model: &str) -> bool {
        // Vercel AI SDK: only gemini-1.5-flash (excluding -8b) supports dynamic retrieval config.
        model.contains("gemini-1.5-flash") && !model.contains("-8b")
    }

    fn supports_file_search(model: &str) -> bool {
        // Vercel AI SDK: File Search is available on Gemini 2.5 models and Gemini 3 models.
        model.contains("gemini-2.5") || model.contains("gemini-3")
    }

    fn parse_dynamic_retrieval_config(
        args: &serde_json::Value,
    ) -> Option<super::types::DynamicRetrievalConfig> {
        let obj = args.as_object()?;
        let mode = obj.get("mode").and_then(|v| v.as_str());
        let dynamic_threshold = match obj.get("dynamicThreshold") {
            Some(serde_json::Value::Number(n)) => Some(n.clone()),
            _ => None,
        };

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
            dynamic_threshold,
        })
    }

    let has_provider_tools = tools.iter().any(|t| matches!(t, Tool::ProviderDefined(_)));

    for tool in tools {
        match tool {
            Tool::Function { function } => {
                if has_provider_tools {
                    // Vercel AI SDK: when provider-defined tools are present, function tools are ignored.
                    continue;
                }
                let parameters = convert_json_schema_to_openapi_schema(&function.parameters, true);
                function_declarations.push(FunctionDeclaration {
                    name: function.name.clone(),
                    description: function.description.clone(),
                    parameters,
                    response: None,
                });
            }
            Tool::ProviderDefined(provider_tool) => {
                // Handle Google/Gemini provider-defined tools
                if matches!(provider_tool.provider(), Some("google" | "gemini")) {
                    match provider_tool.tool_type() {
                        Some("code_execution") => {
                            // Vercel AI SDK: Code Execution is only supported on Gemini 2.0+.
                            if is_gemini_2_or_newer(model) {
                                function_declarations.shrink_to_fit();
                                gemini_tools.push(GeminiTool::CodeExecution {
                                    code_execution: super::types::CodeExecution {},
                                });
                            }
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
                                    .ok_or_else(|| {
                                        LlmError::InvalidInput(
                                            "google.vertex_rag_store requires `ragCorpus`"
                                                .to_string(),
                                        )
                                    })?
                                    .to_string();

                                let similarity_top_k = match obj.and_then(|o| o.get("topK")) {
                                    None => None,
                                    Some(v) => {
                                        let n = v.as_u64().ok_or_else(|| {
                                            LlmError::InvalidInput(
                                                "google.vertex_rag_store `topK` must be a positive integer"
                                                    .to_string(),
                                            )
                                        })?;
                                        if n == 0 {
                                            return Err(LlmError::InvalidInput(
                                                "google.vertex_rag_store `topK` must be a positive integer"
                                                    .to_string(),
                                            ));
                                        }
                                        Some(n as u32)
                                    }
                                };

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
                                    })
                                    .filter(|arr| !arr.is_empty())
                                    .ok_or_else(|| {
                                        LlmError::InvalidInput(
                                            "google.file_search requires non-empty `fileSearchStoreNames`"
                                                .to_string(),
                                        )
                                    })?;

                                let top_k = match obj.and_then(|o| o.get("topK")) {
                                    None => None,
                                    Some(v) => {
                                        let n = v.as_u64().ok_or_else(|| {
                                            LlmError::InvalidInput(
                                                "google.file_search `topK` must be a positive integer"
                                                    .to_string(),
                                            )
                                        })?;
                                        if n == 0 {
                                            return Err(LlmError::InvalidInput(
                                                "google.file_search `topK` must be a positive integer"
                                                    .to_string(),
                                            ));
                                        }
                                        Some(n as u32)
                                    }
                                };

                                let metadata_filter = match obj.and_then(|o| o.get("metadataFilter"))
                                {
                                    None => None,
                                    Some(v) => Some(
                                        v.as_str()
                                            .ok_or_else(|| {
                                                LlmError::InvalidInput(
                                                    "google.file_search `metadataFilter` must be a string expression"
                                                        .to_string(),
                                                )
                                            })?
                                            .to_string(),
                                    ),
                                };

                                gemini_tools.push(GeminiTool::FileSearch {
                                    file_search: super::types::FileSearch {
                                        file_search_store_names: Some(file_search_store_names),
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

    if !has_provider_tools && !function_declarations.is_empty() {
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
        let tools = vec![crate::tools::google::code_execution()];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::CodeExecution { .. }))
        );

        // google_search
        let tools = vec![crate::tools::google::google_search()];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleSearch { .. }))
        );

        // google_search_retrieval
        let tools = vec![crate::tools::google::google_search_retrieval()];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleSearchRetrieval { .. }))
        );

        // google_maps
        let tools = vec![crate::tools::google::google_maps()];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleMaps { .. }))
        );

        // url_context
        let tools = vec![crate::tools::google::url_context()];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::UrlContext { .. }))
        );

        // enterprise_web_search
        let tools = vec![crate::tools::google::enterprise_web_search()];
        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::EnterpriseWebSearch { .. }))
        );

        // vertex_rag_store
        let tools = vec![
            Tool::provider_defined("google.vertex_rag_store", "vertex_rag_store").with_args(
                serde_json::json!({
                    "ragCorpus": "projects/p/locations/l/ragCorpora/c",
                    "topK": 3
                }),
            ),
        ];
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
        let tools = vec![crate::tools::google::google_search()];
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
            "file_search should be dropped for models before gemini-2.5 and gemini-3"
        );
    }

    #[test]
    fn file_search_is_allowed_on_gemini_3_models() {
        let tools = vec![
            Tool::provider_defined("google.file_search", "file_search").with_args(
                serde_json::json!({
                    "fileSearchStoreNames": ["fileSearchStores/abc123"],
                    "topK": 5
                }),
            ),
        ];

        let mapped = convert_tools_to_gemini("gemini-3-pro-preview", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::FileSearch { .. })),
            "file_search should be allowed for gemini-3 models"
        );
    }

    #[test]
    fn google_maps_is_ignored_on_gemini_1_5() {
        let tools = vec![crate::tools::google::google_maps()];
        let mapped = convert_tools_to_gemini("gemini-1.5-pro", &tools).expect("map ok");
        assert!(
            !mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleMaps { .. })),
            "google_maps should be dropped for gemini-1.5 models"
        );
    }

    #[test]
    fn function_tools_are_ignored_when_provider_defined_tools_are_present() {
        let tools = vec![
            Tool::function(
                "testFunction",
                "A test function",
                serde_json::json!({ "type": "object", "properties": {} }),
            ),
            crate::tools::google::google_search(),
        ];

        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        assert!(
            mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::GoogleSearch { .. })),
            "expected provider tool to be mapped"
        );
        assert!(
            !mapped
                .iter()
                .any(|t| matches!(t, GeminiTool::FunctionDeclarations { .. })),
            "function tools should be ignored when provider tools are present"
        );
    }

    #[test]
    fn empty_object_schema_is_omitted_for_function_declarations() {
        let tools = vec![Tool::function(
            "testFunction",
            "A test function",
            serde_json::json!({ "type": "object", "properties": {} }),
        )];

        let mapped = convert_tools_to_gemini("gemini-2.5-flash", &tools).expect("map ok");
        let decls = mapped
            .iter()
            .find_map(|t| match t {
                GeminiTool::FunctionDeclarations {
                    function_declarations,
                } => Some(function_declarations),
                _ => None,
            })
            .expect("function declarations");
        assert_eq!(decls.len(), 1);
        assert!(decls[0].parameters.is_none());
    }

    #[test]
    fn google_file_search_requires_store_names() {
        let tools = vec![Tool::provider_defined("google.file_search", "file_search")];
        let err = convert_tools_to_gemini("gemini-2.5-flash", &tools).unwrap_err();
        assert!(
            err.to_string()
                .contains("google.file_search requires non-empty `fileSearchStoreNames`"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn google_file_search_rejects_non_string_metadata_filter() {
        let tool = Tool::provider_defined("google.file_search", "file_search").with_args(
            serde_json::json!({
                "fileSearchStoreNames": ["fileSearchStores/abc123"],
                "metadataFilter": { "source": "test" }
            }),
        );

        let err = convert_tools_to_gemini("gemini-2.5-flash", &[tool]).unwrap_err();
        assert!(
            err.to_string()
                .contains("google.file_search `metadataFilter` must be a string expression"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn google_vertex_rag_store_requires_rag_corpus() {
        let tools = vec![Tool::provider_defined(
            "google.vertex_rag_store",
            "vertex_rag_store",
        )];
        let err = convert_tools_to_gemini("gemini-2.5-flash", &tools).unwrap_err();
        assert!(
            err.to_string()
                .contains("google.vertex_rag_store requires `ragCorpus`"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn google_vertex_rag_store_rejects_non_positive_top_k() {
        let tools = vec![
            Tool::provider_defined("google.vertex_rag_store", "vertex_rag_store").with_args(
                serde_json::json!({
                    "ragCorpus": "projects/p/locations/l/ragCorpora/c",
                    "topK": 0
                }),
            ),
        ];
        let err = convert_tools_to_gemini("gemini-2.5-flash", &tools).unwrap_err();
        assert!(
            err.to_string()
                .contains("google.vertex_rag_store `topK` must be a positive integer"),
            "unexpected error: {err:?}"
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
                            if let crate::types::ContentPart::Text { text, .. } = part {
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
        let v = (temp * 1_000_000.0).round() / 1_000_000.0;
        merged_generation_config.temperature = Some(v);
    }
    if let Some(max_tokens) = config.common_params.max_tokens {
        merged_generation_config.max_output_tokens = Some(max_tokens as i32);
    }
    if let Some(top_p) = config.common_params.top_p {
        let v = (top_p * 1_000_000.0).round() / 1_000_000.0;
        merged_generation_config.top_p = Some(v);
    }
    if let Some(top_k) = config.common_params.top_k {
        let rounded = top_k.round();
        if (top_k - rounded).abs() < 1e-9 && rounded >= 0.0 && rounded <= i32::MAX as f64 {
            merged_generation_config.top_k = Some(rounded as i32);
        }
    }
    if let Some(stops) = &config.common_params.stop_sequences {
        merged_generation_config.stop_sequences = Some(stops.clone());
    }
    if let Some(seed) = config.common_params.seed
        && seed <= i32::MAX as u64
    {
        merged_generation_config.seed = Some(seed as i32);
    }
    if let Some(fp) = config.common_params.frequency_penalty {
        merged_generation_config.frequency_penalty = Some(fp);
    }
    if let Some(pp) = config.common_params.presence_penalty {
        merged_generation_config.presence_penalty = Some(pp);
    }

    let gemini_tools = if let Some(ts) = tools {
        if ts.is_empty() {
            None
        } else {
            let mapped = convert_tools_to_gemini(&model, ts)?;
            if mapped.is_empty() {
                None
            } else {
                Some(mapped)
            }
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
                thought_signature: None,
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
                        thought_signature: None,
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
                            thought_signature: None,
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
                    thought_signature: None,
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
        match first_part {
            Part::FunctionResponse {
                function_response, ..
            } => {
                assert_eq!(function_response.name, "search");
                assert_eq!(
                    function_response.response,
                    serde_json::json!({ "name": "search", "content": "ok" })
                );
            }
            _ => panic!("expected functionResponse"),
        }
    }

    #[test]
    fn tool_result_content_maps_text_and_image_parts_like_vercel() {
        let cfg = GeminiConfig::default();

        let msg = crate::types::ChatMessage {
            role: crate::types::MessageRole::Tool,
            content: MessageContent::MultiModal(vec![
                crate::types::ContentPart::tool_result_content(
                    "call_1",
                    "generate_image",
                    vec![
                        ToolResultContentPart::text("Generated image:"),
                        ToolResultContentPart::image_base64("aGVsbG8="),
                    ],
                ),
            ]),
            metadata: Default::default(),
        };

        let req = build_request_body(&cfg, &[msg], None).unwrap();
        assert_eq!(req.contents.len(), 1);
        assert_eq!(req.contents[0].role.as_deref(), Some("user"));
        assert_eq!(req.contents[0].parts.len(), 3);

        // 1) Function response for the text part.
        match &req.contents[0].parts[0] {
            Part::FunctionResponse {
                function_response, ..
            } => {
                assert_eq!(function_response.name, "generate_image");
                assert_eq!(
                    function_response.response,
                    serde_json::json!({ "name": "generate_image", "content": "Generated image:" })
                );
            }
            _ => panic!("expected functionResponse"),
        }

        // 2) Inline data for the image part (best-effort; mime type is not carried in ToolResultContentPart).
        match &req.contents[0].parts[1] {
            Part::InlineData { inline_data, .. } => {
                assert_eq!(inline_data.mime_type, "image/jpeg");
                assert_eq!(inline_data.data, "aGVsbG8=");
            }
            _ => panic!("expected inlineData"),
        }

        // 3) Sentinel text (matches Vercel AI SDK behavior).
        match &req.contents[0].parts[2] {
            Part::Text { text, .. } => {
                assert_eq!(
                    text,
                    "Tool executed successfully and returned this image as a response"
                );
            }
            _ => panic!("expected text part"),
        }
    }

    #[test]
    fn rejects_assistant_messages_with_url_based_files() {
        let cfg = GeminiConfig::default();
        let messages = vec![
            crate::types::ChatMessage::assistant("hi")
                .with_file_url("https://example.com/a.bin", "application/octet-stream")
                .build(),
        ];

        let err = build_request_body(&cfg, &messages, None).unwrap_err();
        assert!(matches!(err, LlmError::InvalidParameter(_)));
        assert!(
            err.to_string()
                .contains("File data URLs in assistant messages are not supported")
        );
    }
}
