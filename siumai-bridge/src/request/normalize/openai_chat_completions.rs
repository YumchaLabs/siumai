//! OpenAI Chat Completions request JSON normalization.

use super::*;

#[cfg(feature = "openai")]
pub(super) fn parse_json_to_chat_request(value: &Value) -> Result<ChatRequest, LlmError> {
    let obj = expect_object(value, "OpenAI Chat Completions request")?;
    let mut request = ChatRequest::new(Vec::new());

    request.common_params.model = required_string(obj, "model", "OpenAI Chat Completions request")?;
    request.common_params.temperature = optional_f64(obj, "temperature");
    request.common_params.top_p = optional_f64(obj, "top_p");
    request.common_params.frequency_penalty = optional_f64(obj, "frequency_penalty");
    request.common_params.presence_penalty = optional_f64(obj, "presence_penalty");
    request.common_params.max_tokens = optional_u32(obj, "max_tokens");
    request.common_params.max_completion_tokens = optional_u32(obj, "max_completion_tokens");
    request.common_params.seed = optional_u64(obj, "seed");
    request.common_params.stop_sequences =
        optional_stop_sequences(obj.get("stop").or_else(|| obj.get("stop_sequences")))?;
    request.stream = optional_bool(obj, "stream").unwrap_or(false);

    if let Some(value) = obj.get("response_format")
        && let Some(parsed) = parse_json_schema_response_format(value)
    {
        request.response_format = Some(parsed);
    }

    if let Some(value) = obj.get("tools") {
        let tools = parse_openai_chat_tools(value)?;
        if !tools.is_empty() {
            request.tools = Some(tools);
        }
    }
    if let Some(choice) = obj.get("tool_choice") {
        request.tool_choice = parse_openai_chat_tool_choice(choice);
    }

    let mut tool_names_by_call_id = HashMap::new();
    if let Some(messages) = obj.get("messages") {
        for value in expect_array(messages, "OpenAI Chat Completions request.messages")? {
            request.messages.push(parse_openai_chat_message(
                value,
                &mut tool_names_by_call_id,
            )?);
        }
    }

    Ok(request)
}

fn parse_openai_chat_message(
    value: &Value,
    tool_names_by_call_id: &mut HashMap<String, String>,
) -> Result<ChatMessage, LlmError> {
    let obj = expect_object(value, "OpenAI Chat Completions message")?;
    let role = required_string(obj, "role", "OpenAI Chat Completions message")?;

    match role.as_str() {
        "system" => parse_openai_chat_role_message(obj, MessageRole::System),
        "developer" => parse_openai_chat_role_message(obj, MessageRole::Developer),
        "user" => parse_openai_chat_role_message(obj, MessageRole::User),
        "assistant" => parse_openai_chat_assistant_message(obj, tool_names_by_call_id),
        "tool" => parse_openai_chat_tool_message(obj, tool_names_by_call_id),
        other => Err(LlmError::ParseError(format!(
            "unsupported OpenAI Chat Completions role `{other}`"
        ))),
    }
}

fn parse_openai_chat_role_message(
    obj: &Map<String, Value>,
    role: MessageRole,
) -> Result<ChatMessage, LlmError> {
    let mut parts = Vec::new();
    if let Some(content) = obj.get("content") {
        parts = parse_openai_chat_content_parts(content)?;
    }
    Ok(message_from_parts(role, parts))
}

fn parse_openai_chat_assistant_message(
    obj: &Map<String, Value>,
    tool_names_by_call_id: &mut HashMap<String, String>,
) -> Result<ChatMessage, LlmError> {
    let mut parts = Vec::new();
    let has_tool_calls = obj
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty());

    if let Some(content) = obj.get("content")
        && !content.is_null()
    {
        let skip_empty_tool_call_content =
            has_tool_calls && content.as_str().is_some_and(|text| text.is_empty());
        if !skip_empty_tool_call_content {
            parts.extend(parse_openai_chat_content_parts(content)?);
        }
    }

    if let Some(tool_calls) = obj.get("tool_calls") {
        for value in expect_array(tool_calls, "OpenAI Chat Completions assistant.tool_calls")? {
            let tool_call = parse_openai_chat_tool_call(value)?;
            if let ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                ..
            } = &tool_call
            {
                tool_names_by_call_id.insert(tool_call_id.clone(), tool_name.clone());
            }
            parts.push(tool_call);
        }
    }

    Ok(message_from_parts(MessageRole::Assistant, parts))
}

fn parse_openai_chat_tool_message(
    obj: &Map<String, Value>,
    tool_names_by_call_id: &HashMap<String, String>,
) -> Result<ChatMessage, LlmError> {
    let tool_call_id =
        required_string(obj, "tool_call_id", "OpenAI Chat Completions tool message")?;
    let content = obj
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tool_name = tool_names_by_call_id
        .get(&tool_call_id)
        .cloned()
        .unwrap_or_default();

    Ok(message_from_parts(
        MessageRole::Tool,
        vec![legacy_content::request_tool_result_part(
            tool_call_id,
            tool_name,
            parse_tool_result_output_from_string(&content, false),
            None,
            None,
            ProviderOptionsMap::default(),
        )],
    ))
}

fn parse_openai_chat_tool_call(value: &Value) -> Result<ContentPart, LlmError> {
    let obj = expect_object(value, "OpenAI Chat Completions tool_call")?;
    let tool_call_id = required_string(obj, "id", "OpenAI Chat Completions tool_call")?;
    let tool_type = optional_string(obj, "type").unwrap_or_else(|| "function".to_string());

    if tool_type != "function" {
        return Ok(legacy_content::request_tool_call_part(
            tool_call_id,
            tool_type.clone(),
            collect_remaining_object_fields(obj, &["id", "type", "function"]),
            None,
            None,
            ProviderOptionsMap::default(),
        ));
    }

    let function = obj
        .get("function")
        .ok_or_else(|| {
            LlmError::ParseError(
                "OpenAI Chat Completions tool_call.function is required".to_string(),
            )
        })
        .and_then(|value| expect_object(value, "OpenAI Chat Completions tool_call.function"))?;
    let name = required_string(
        function,
        "name",
        "OpenAI Chat Completions tool_call.function",
    )?;
    let arguments = function
        .get("arguments")
        .map(parse_embedded_json)
        .transpose()?
        .unwrap_or_else(|| Value::Object(Map::new()));

    Ok(legacy_content::request_tool_call_part(
        tool_call_id,
        name,
        arguments,
        None,
        None,
        ProviderOptionsMap::default(),
    ))
}

fn parse_openai_chat_content_parts(value: &Value) -> Result<Vec<ContentPart>, LlmError> {
    if let Some(text) = value.as_str() {
        return Ok(parse_text_like_content_parts(text));
    }

    let mut parts = Vec::new();
    for part in expect_array(value, "OpenAI Chat Completions content")? {
        parts.push(parse_openai_chat_content_part(part)?);
    }
    Ok(parts)
}

fn parse_openai_chat_content_part(value: &Value) -> Result<ContentPart, LlmError> {
    let obj = expect_object(value, "OpenAI Chat Completions content part")?;
    let kind = required_string(obj, "type", "OpenAI Chat Completions content part")?;

    match kind.as_str() {
        "text" => Ok(legacy_content::request_text_part(
            optional_string(obj, "text").unwrap_or_default(),
            ProviderOptionsMap::default(),
        )),
        "image_url" => parse_openai_image_url_part(obj),
        "input_audio" => parse_openai_input_audio_part(obj),
        "file" => parse_openai_file_part(obj),
        other => Err(LlmError::ParseError(format!(
            "unsupported OpenAI Chat Completions content part `{other}`"
        ))),
    }
}

fn parse_openai_image_url_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    match obj.get("image_url") {
        Some(Value::String(url)) => Ok(legacy_content::request_image_part(
            FilePartSource::url(url),
            None,
            None,
            ProviderOptionsMap::default(),
        )),
        Some(Value::Object(image)) => {
            let url = required_string(image, "url", "OpenAI image_url content part")?;
            let detail = image
                .get("detail")
                .and_then(Value::as_str)
                .map(ImageDetail::from);
            Ok(legacy_content::request_image_part(
                FilePartSource::url(url),
                None,
                detail,
                ProviderOptionsMap::default(),
            ))
        }
        _ => Err(LlmError::ParseError(
            "OpenAI image_url content part is missing image_url".to_string(),
        )),
    }
}

fn parse_openai_input_audio_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    let audio = obj
        .get("input_audio")
        .ok_or_else(|| {
            LlmError::ParseError("OpenAI input_audio part is missing input_audio".to_string())
        })
        .and_then(|value| expect_object(value, "OpenAI input_audio part"))?;
    let data = required_string(audio, "data", "OpenAI input_audio part")?;
    let format = optional_string(audio, "format").unwrap_or_else(|| "wav".to_string());
    let media_type = match format.as_str() {
        "mp3" => "audio/mpeg".to_string(),
        _ => "audio/wav".to_string(),
    };

    Ok(legacy_content::request_audio_part(
        MediaSource::Base64 { data },
        Some(media_type),
        ProviderOptionsMap::default(),
    ))
}

pub(super) fn parse_openai_file_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    let file = obj
        .get("file")
        .ok_or_else(|| LlmError::ParseError("OpenAI file part is missing file".to_string()))
        .and_then(|value| expect_object(value, "OpenAI file part.file"))?;

    if let Some(file_id) = optional_string(file, "file_id") {
        return Ok(legacy_content::request_file_part(
            FilePartSource::provider_reference(ProviderReference::single("openai", file_id)),
            "application/pdf",
            optional_string(file, "filename"),
            ProviderOptionsMap::default(),
        ));
    }
    if let Some(file_data) = optional_string(file, "file_data") {
        return Ok(legacy_content::request_file_part(
            FilePartSource::base64(strip_data_url_prefix(&file_data)),
            "application/pdf",
            optional_string(file, "filename"),
            ProviderOptionsMap::default(),
        ));
    }

    Err(LlmError::ParseError(
        "OpenAI file part requires file_id or file_data".to_string(),
    ))
}

fn parse_openai_chat_tools(value: &Value) -> Result<Vec<Tool>, LlmError> {
    let mut tools = Vec::new();
    for tool in expect_array(value, "OpenAI Chat Completions request.tools")? {
        tools.push(parse_openai_chat_tool(tool)?);
    }
    Ok(tools)
}

fn parse_openai_chat_tool(value: &Value) -> Result<Tool, LlmError> {
    let obj = expect_object(value, "OpenAI Chat Completions request.tools[]")?;
    let kind = required_string(obj, "type", "OpenAI Chat Completions request.tools[]")?;

    if kind == "function" {
        let function = obj
            .get("function")
            .ok_or_else(|| {
                LlmError::ParseError(
                    "OpenAI Chat Completions function tool is missing `function`".to_string(),
                )
            })
            .and_then(|value| {
                expect_object(value, "OpenAI Chat Completions request.tools[].function")
            })?;
        return Ok(parse_openai_function_tool(function));
    }

    Ok(parse_openai_provider_defined_tool(kind.as_str(), obj, None))
}

fn parse_openai_chat_tool_choice(value: &Value) -> Option<ToolChoice> {
    match value {
        Value::String(choice) => match choice.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "required" => Some(ToolChoice::Required),
            "none" => Some(ToolChoice::None),
            _ => None,
        },
        Value::Object(obj) => {
            if obj.get("type").and_then(Value::as_str) == Some("function") {
                obj.get("function")
                    .and_then(Value::as_object)
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .map(ToolChoice::tool)
            } else {
                obj.get("type")
                    .and_then(Value::as_str)
                    .map(ToolChoice::tool)
            }
        }
        _ => None,
    }
}
