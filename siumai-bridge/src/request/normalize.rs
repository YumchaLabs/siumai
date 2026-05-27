//! Request bridge normalization from protocol JSON into `ChatRequest`.

#![allow(dead_code)]

use crate::customize::apply_request_remapper;
use crate::lifecycle::{new_bridge_report, new_request_normalize_context, reject_if_needed};
use crate::request::legacy_content;
#[cfg(feature = "anthropic")]
use base64::Engine;
use std::collections::BTreeSet;
#[cfg(feature = "openai")]
use std::collections::HashMap;
#[cfg(feature = "anthropic")]
use std::time::Duration;

use crate::{BridgeOptions, BridgeResult, BridgeTarget};
#[cfg(any(feature = "openai", feature = "anthropic"))]
use serde_json::json;
use serde_json::{Map, Value};
use siumai_core::LlmError;
#[cfg(feature = "anthropic")]
use siumai_core::types::CacheControl;
#[cfg(feature = "openai")]
use siumai_core::types::chat::ImageDetail;
#[cfg(not(any(feature = "openai", feature = "anthropic")))]
use siumai_core::types::chat::ResponseFormat;
#[cfg(any(feature = "openai", feature = "anthropic"))]
use siumai_core::types::chat::{FilePartSource, MediaSource, ProviderReference, ResponseFormat};
use siumai_core::types::{
    ChatMessage, ChatRequest, ContentPart, MessageContent, MessageMetadata, MessageRole,
    ProviderOptionsMap, Tool, ToolFunction, ToolResultOutput,
};
#[cfg(any(feature = "openai", feature = "anthropic"))]
use siumai_core::types::{ToolChoice, ToolResultContentPart};

#[cfg(any(feature = "google", feature = "google-vertex"))]
mod gemini_generate_content;

#[cfg(feature = "anthropic")]
mod anthropic_messages;
#[cfg(feature = "openai")]
mod openai_chat_completions;
#[cfg(feature = "openai")]
mod openai_responses;

fn normalize_request_with_options(
    mut request: ChatRequest,
    source: BridgeTarget,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatRequest>, LlmError> {
    let ctx = new_request_normalize_context(source, &options);
    let mut report = new_bridge_report(Some(source), source, options.mode);

    if let Some(remapper) = options.primitive_remapper.as_deref() {
        apply_request_remapper(&mut request, &ctx, remapper);
    }
    if let Some(hook) = options.request_hook.as_deref() {
        hook.transform_request(&ctx, &mut request, &mut report)?;
    }

    let action = options.loss_policy.request_action(&ctx, &report);
    if reject_if_needed(&mut report, action, "request normalization", source) {
        return Ok(BridgeResult::rejected(report));
    }

    Ok(BridgeResult::new(request, report))
}

#[cfg(feature = "anthropic")]
pub fn bridge_anthropic_messages_json_to_chat_request(
    value: &Value,
) -> Result<ChatRequest, LlmError> {
    anthropic_messages::parse_json_to_chat_request(value)
}

#[cfg(feature = "anthropic")]
pub fn bridge_anthropic_messages_json_to_chat_request_with_options(
    value: &Value,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatRequest>, LlmError> {
    let request = anthropic_messages::parse_json_to_chat_request(value)?;
    normalize_request_with_options(request, BridgeTarget::AnthropicMessages, options)
}

#[cfg(feature = "openai")]
pub fn bridge_openai_responses_json_to_chat_request(
    value: &Value,
) -> Result<ChatRequest, LlmError> {
    openai_responses::parse_json_to_chat_request(value)
}

#[cfg(feature = "openai")]
pub fn bridge_openai_responses_json_to_chat_request_with_options(
    value: &Value,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatRequest>, LlmError> {
    let request = openai_responses::parse_json_to_chat_request(value)?;
    normalize_request_with_options(request, BridgeTarget::OpenAiResponses, options)
}

#[cfg(feature = "openai")]
pub fn bridge_openai_chat_completions_json_to_chat_request(
    value: &Value,
) -> Result<ChatRequest, LlmError> {
    openai_chat_completions::parse_json_to_chat_request(value)
}

#[cfg(feature = "openai")]
pub fn bridge_openai_chat_completions_json_to_chat_request_with_options(
    value: &Value,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatRequest>, LlmError> {
    let request = openai_chat_completions::parse_json_to_chat_request(value)?;
    normalize_request_with_options(request, BridgeTarget::OpenAiChatCompletions, options)
}

#[cfg(any(feature = "google", feature = "google-vertex"))]
pub fn bridge_gemini_generate_content_json_to_chat_request(
    value: &Value,
) -> Result<ChatRequest, LlmError> {
    gemini_generate_content::parse_json_to_chat_request(value)
}

#[cfg(any(feature = "google", feature = "google-vertex"))]
pub fn bridge_gemini_generate_content_json_to_chat_request_with_options(
    value: &Value,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatRequest>, LlmError> {
    let request = gemini_generate_content::parse_json_to_chat_request(value)?;
    normalize_request_with_options(request, BridgeTarget::GeminiGenerateContent, options)
}

fn parse_openai_function_tool(function_obj: &Map<String, Value>) -> Tool {
    let name = optional_string(function_obj, "name").unwrap_or_default();
    let description = optional_string(function_obj, "description").unwrap_or_default();
    let parameters = openai_function_input_schema(function_obj);
    let mut tool = Tool::function(name, description, parameters);

    if let Tool::Function { function } = &mut tool {
        populate_openai_function_tool_metadata(function, function_obj);
    }

    tool
}

fn openai_function_input_schema(function_obj: &Map<String, Value>) -> Value {
    function_obj
        .get("parameters")
        .or_else(|| function_obj.get("inputSchema"))
        .or_else(|| function_obj.get("input_schema"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()))
}

fn populate_openai_function_tool_metadata(
    function: &mut ToolFunction,
    function_obj: &Map<String, Value>,
) {
    function.strict = function_obj.get("strict").and_then(Value::as_bool);
    function.output_schema = function_obj
        .get("outputSchema")
        .or_else(|| function_obj.get("output_schema"))
        .cloned();

    if let Some(input_examples) = function_obj
        .get("inputExamples")
        .or_else(|| function_obj.get("input_examples"))
        .and_then(Value::as_array)
    {
        function.input_examples = Some(input_examples.clone());
    }
}

fn parse_openai_provider_defined_tool(
    wire_type: &str,
    obj: &Map<String, Value>,
    skip_type_key: Option<&str>,
) -> Tool {
    let provider_id = openai_provider_tool_id_from_wire_type(wire_type);
    let mut tool = default_provider_defined_tool(&provider_id).unwrap_or_else(|| {
        Tool::provider_defined(provider_id.clone(), default_openai_tool_name(wire_type))
    });

    if let Tool::ProviderDefined(provider_tool) = &mut tool {
        if let Some(name) = optional_string(obj, "name")
            && !name.is_empty()
        {
            provider_tool.name = name;
        }

        let mut skip = vec!["name"];
        skip.push(skip_type_key.unwrap_or("type"));
        provider_tool.args = collect_remaining_object_fields(obj, &skip);
    }

    tool
}

fn required_string(obj: &Map<String, Value>, key: &str, label: &str) -> Result<String, LlmError> {
    obj.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| LlmError::ParseError(format!("{label} is missing `{key}`")))
}

fn optional_string(obj: &Map<String, Value>, key: &str) -> Option<String> {
    obj.get(key).and_then(Value::as_str).map(str::to_string)
}

fn optional_bool(obj: &Map<String, Value>, key: &str) -> Option<bool> {
    obj.get(key).and_then(Value::as_bool)
}

fn optional_f64(obj: &Map<String, Value>, key: &str) -> Option<f64> {
    obj.get(key).and_then(Value::as_f64)
}

fn optional_u32(obj: &Map<String, Value>, key: &str) -> Option<u32> {
    obj.get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn optional_u64(obj: &Map<String, Value>, key: &str) -> Option<u64> {
    obj.get(key).and_then(Value::as_u64)
}

fn optional_stop_sequences(value: Option<&Value>) -> Result<Option<Vec<String>>, LlmError> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        Value::String(value) => Ok(Some(vec![value.clone()])),
        Value::Array(values) => Ok(Some(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect(),
        )),
        Value::Null => Ok(None),
        _ => Err(LlmError::ParseError(
            "stop sequences must be a string or array of strings".to_string(),
        )),
    }
}

fn expect_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, LlmError> {
    value
        .as_object()
        .ok_or_else(|| LlmError::ParseError(format!("{label} must be a JSON object")))
}

fn expect_array<'a>(value: &'a Value, label: &str) -> Result<&'a [Value], LlmError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| LlmError::ParseError(format!("{label} must be a JSON array")))
}

fn parse_embedded_json(value: &Value) -> Result<Value, LlmError> {
    match value {
        Value::String(text) => match serde_json::from_str(text) {
            Ok(parsed) => Ok(parsed),
            Err(_) => Ok(Value::String(text.clone())),
        },
        other => Ok(other.clone()),
    }
}

fn parse_json_schema_response_format(value: &Value) -> Option<ResponseFormat> {
    let obj = value.as_object()?;
    let type_name = obj.get("type").and_then(Value::as_str);
    let owner = obj
        .get("json_schema")
        .and_then(Value::as_object)
        .unwrap_or(obj);

    if type_name == Some("json_object")
        || (type_name == Some("json") && !owner.contains_key("schema"))
    {
        let mut format = ResponseFormat::json_object();
        if let Some(name) = owner.get("name").and_then(Value::as_str) {
            format = format.with_name(name.to_string());
        }
        if let Some(description) = owner.get("description").and_then(Value::as_str) {
            format = format.with_description(description.to_string());
        }
        return Some(format);
    }

    if type_name != Some("json_schema") && !owner.contains_key("schema") {
        return None;
    }

    let schema = owner.get("schema")?.clone();
    let mut format = ResponseFormat::json_schema(schema);
    if let Some(name) = owner.get("name").and_then(Value::as_str) {
        format = format.with_name(name.to_string());
    }
    if let Some(description) = owner.get("description").and_then(Value::as_str) {
        format = format.with_description(description.to_string());
    }
    if let Some(strict) = owner.get("strict").and_then(Value::as_bool) {
        format = format.with_strict(strict);
    }
    Some(format)
}

fn strip_wrapped_thinking(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let inner = trimmed
        .strip_prefix("<thinking>")
        .and_then(|value| value.strip_suffix("</thinking>"))?;
    Some(inner.to_string())
}

fn parse_text_like_content_parts(text: &str) -> Vec<ContentPart> {
    if let Some(reasoning) = strip_wrapped_thinking(text) {
        vec![legacy_content::request_reasoning_part(
            reasoning,
            ProviderOptionsMap::default(),
        )]
    } else {
        vec![legacy_content::request_text_part(
            text,
            ProviderOptionsMap::default(),
        )]
    }
}

fn parse_tool_result_output_from_string(text: &str, is_error: bool) -> ToolResultOutput {
    match serde_json::from_str::<Value>(text) {
        Ok(value) => {
            if is_error {
                ToolResultOutput::error_json(value)
            } else {
                ToolResultOutput::json(value)
            }
        }
        Err(_) => {
            if is_error {
                ToolResultOutput::error_text(text)
            } else {
                ToolResultOutput::text(text)
            }
        }
    }
}

fn collect_reasoning_summary(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::Array(items)) => {
            let joined = items
                .iter()
                .filter_map(|value| value.as_object())
                .filter_map(|obj| obj.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n");
            Some(joined)
        }
        Some(Value::String(text)) => Some(text.clone()),
        _ => None,
    }
}

fn message_from_parts(role: MessageRole, parts: Vec<ContentPart>) -> ChatMessage {
    let content = if parts.is_empty() {
        MessageContent::Text(String::new())
    } else if parts.len() == 1 {
        match parts.into_iter().next().expect("checked len") {
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

fn text_message(role: MessageRole, text: impl Into<String>) -> ChatMessage {
    ChatMessage {
        role,
        content: MessageContent::Text(text.into()),
        provider_options: ProviderOptionsMap::default(),
        metadata: Default::default(),
    }
}

fn compact_adjacent_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut compacted = Vec::with_capacity(messages.len());

    for message in messages {
        if let Some(last) = compacted.last_mut()
            && can_merge_adjacent_messages(last, &message)
        {
            merge_message_into(last, message);
            continue;
        }
        compacted.push(message);
    }

    compacted
}

fn can_merge_adjacent_messages(previous: &ChatMessage, current: &ChatMessage) -> bool {
    previous.role == current.role
        && matches!(previous.role, MessageRole::Assistant | MessageRole::Tool)
        && is_empty_message_metadata(&previous.metadata)
        && is_empty_message_metadata(&current.metadata)
}

fn is_empty_message_metadata(metadata: &MessageMetadata) -> bool {
    metadata.id.is_none()
        && metadata.timestamp.is_none()
        && metadata.cache_control.is_none()
        && metadata.custom.is_empty()
}

fn merge_message_into(previous: &mut ChatMessage, current: ChatMessage) {
    let mut parts = message_content_into_parts(std::mem::replace(
        &mut previous.content,
        MessageContent::Text(String::new()),
    ));
    parts.extend(message_content_into_parts(current.content));
    previous.content = message_from_parts(previous.role.clone(), parts).content;
}

fn message_content_into_parts(content: MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::Text(text) => {
            if text.is_empty() {
                Vec::new()
            } else {
                parse_text_like_content_parts(&text)
            }
        }
        MessageContent::MultiModal(parts) => parts,
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => vec![legacy_content::request_text_part(
            serde_json::to_string(&value).unwrap_or_default(),
            ProviderOptionsMap::default(),
        )],
    }
}

fn collect_remaining_object_fields(obj: &Map<String, Value>, skip: &[&str]) -> Value {
    let skip = skip.iter().copied().collect::<BTreeSet<_>>();
    Value::Object(
        obj.iter()
            .filter(|(key, _)| !skip.contains(key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn strip_data_url_prefix(value: &str) -> String {
    if let Some((_, encoded)) = value.split_once(",")
        && value.starts_with("data:")
    {
        return encoded.to_string();
    }
    value.to_string()
}

fn infer_document_media_type(title: Option<&str>, url: Option<&str>) -> String {
    let candidate = title.or(url).unwrap_or_default().to_ascii_lowercase();
    if candidate.ends_with(".txt") || candidate.ends_with(".md") || candidate.ends_with(".text") {
        "text/plain".to_string()
    } else {
        "application/pdf".to_string()
    }
}

fn openai_provider_tool_id_from_wire_type(wire_type: &str) -> String {
    #[cfg(feature = "openai")]
    {
        match wire_type {
            "computer_use_preview" => {
                return siumai_protocol_openai::tool_catalog::openai::COMPUTER_USE_ID.to_string();
            }
            "web_search" => {
                return siumai_protocol_openai::tool_catalog::openai::WEB_SEARCH_ID.to_string();
            }
            "web_search_preview" => {
                return siumai_protocol_openai::tool_catalog::openai::WEB_SEARCH_PREVIEW_ID
                    .to_string();
            }
            "file_search" => {
                return siumai_protocol_openai::tool_catalog::openai::FILE_SEARCH_ID.to_string();
            }
            "code_interpreter" => {
                return siumai_protocol_openai::tool_catalog::openai::CODE_INTERPRETER_ID
                    .to_string();
            }
            "image_generation" => {
                return siumai_protocol_openai::tool_catalog::openai::IMAGE_GENERATION_ID
                    .to_string();
            }
            "local_shell" => {
                return siumai_protocol_openai::tool_catalog::openai::LOCAL_SHELL_ID.to_string();
            }
            "shell" => return siumai_protocol_openai::tool_catalog::openai::SHELL_ID.to_string(),
            "mcp" => return siumai_protocol_openai::tool_catalog::openai::MCP_ID.to_string(),
            "apply_patch" => {
                return siumai_protocol_openai::tool_catalog::openai::APPLY_PATCH_ID.to_string();
            }
            _ => {}
        }
    }

    match wire_type {
        "computer_use_preview" => "openai.computer_use".to_string(),
        "web_search" => "openai.web_search".to_string(),
        "web_search_preview" => "openai.web_search_preview".to_string(),
        "file_search" => "openai.file_search".to_string(),
        "code_interpreter" => "openai.code_interpreter".to_string(),
        "image_generation" => "openai.image_generation".to_string(),
        "local_shell" => "openai.local_shell".to_string(),
        "shell" => "openai.shell".to_string(),
        "mcp" => "openai.mcp".to_string(),
        "apply_patch" => "openai.apply_patch".to_string(),
        other => format!("openai.{other}"),
    }
}

fn default_openai_tool_name(wire_type: &str) -> String {
    match wire_type {
        "computer_use_preview" => "computer_use".to_string(),
        "web_search" => "webSearch".to_string(),
        "file_search" => "fileSearch".to_string(),
        "code_interpreter" => "codeExecution".to_string(),
        "image_generation" => "generateImage".to_string(),
        "local_shell" | "shell" => "shell".to_string(),
        "mcp" => "MCP".to_string(),
        other => other.to_string(),
    }
}

fn default_provider_defined_tool(provider_id: &str) -> Option<Tool> {
    #[cfg(feature = "openai")]
    if let Some(tool) =
        siumai_protocol_openai::tool_catalog::openai::provider_defined_tool(provider_id)
    {
        return Some(tool);
    }

    #[cfg(feature = "anthropic")]
    if let Some(tool) =
        siumai_protocol_anthropic::tool_catalog::anthropic::provider_defined_tool(provider_id)
    {
        return Some(tool);
    }

    #[cfg(any(feature = "google", feature = "google-vertex"))]
    if let Some(tool) =
        siumai_protocol_gemini::tool_catalog::google::provider_defined_tool(provider_id)
    {
        return Some(tool);
    }

    None
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "anthropic")]
    use super::anthropic_messages::parse_anthropic_image_part;
    #[cfg(feature = "openai")]
    use super::{
        openai_chat_completions::parse_openai_file_part,
        openai_responses::parse_openai_responses_image_part,
    };
    #[cfg(any(feature = "anthropic", feature = "openai"))]
    use siumai_core::types::ContentPart;

    #[cfg(feature = "openai")]
    #[test]
    fn openai_file_part_file_id_normalizes_to_provider_reference() {
        let obj = serde_json::json!({
            "file": {
                "file_id": "file-123",
                "filename": "doc.pdf"
            }
        });

        let part = parse_openai_file_part(obj.as_object().expect("object")).expect("parse part");
        let ContentPart::File {
            source,
            media_type,
            filename,
            ..
        } = part
        else {
            panic!("expected file part");
        };

        assert_eq!(media_type, "application/pdf");
        assert_eq!(filename.as_deref(), Some("doc.pdf"));
        assert_eq!(
            source
                .as_provider_reference()
                .and_then(|provider_reference| provider_reference.get("openai")),
            Some("file-123")
        );
        assert!(source.as_media_source().is_none());
    }

    #[cfg(feature = "openai")]
    #[test]
    fn openai_responses_image_file_id_normalizes_to_image_provider_reference() {
        let obj = serde_json::json!({
            "type": "input_image",
            "file_id": "file-456"
        });
        let part = parse_openai_responses_image_part(obj.as_object().expect("object"));
        let ContentPart::Image { source, .. } = part else {
            panic!("expected image part");
        };

        assert_eq!(
            source
                .as_provider_reference()
                .and_then(|provider_reference| provider_reference.get("openai")),
            Some("file-456")
        );
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn anthropic_image_file_source_normalizes_to_provider_reference() {
        let obj = serde_json::json!({
            "type": "image",
            "source": {
                "type": "file",
                "file_id": "file-anthropic"
            }
        });

        let part =
            parse_anthropic_image_part(obj.as_object().expect("object")).expect("parse image part");
        let ContentPart::Image { source, .. } = part else {
            panic!("expected image part");
        };

        assert_eq!(
            source
                .as_provider_reference()
                .and_then(|provider_reference| provider_reference.get("anthropic")),
            Some("file-anthropic")
        );
    }
}
