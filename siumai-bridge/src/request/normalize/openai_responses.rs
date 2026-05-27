//! OpenAI Responses request JSON normalization.

use super::*;

#[derive(Debug, Default)]
struct ResponsesToolRegistry {
    tool_names_by_wire_type: HashMap<String, String>,
}

impl ResponsesToolRegistry {
    fn from_tools(tools: &[Tool]) -> Self {
        let mut tool_names_by_wire_type = HashMap::new();
        for tool in tools {
            let Tool::ProviderDefined(provider_tool) = tool else {
                continue;
            };
            if provider_tool.provider() != Some("openai") {
                continue;
            }
            if let Some(wire_type) = openai_responses_wire_type_for_tool(tool) {
                tool_names_by_wire_type.insert(wire_type, provider_tool.name.clone());
            }
        }
        Self {
            tool_names_by_wire_type,
        }
    }

    fn resolve_name(&self, raw_name: &str) -> String {
        self.tool_names_by_wire_type
            .get(raw_name)
            .cloned()
            .unwrap_or_else(|| raw_name.to_string())
    }

    fn tool_name_for_type(&self, wire_type: &str) -> String {
        self.resolve_name(wire_type)
    }
}

#[cfg(feature = "openai")]
pub(super) fn parse_json_to_chat_request(value: &Value) -> Result<ChatRequest, LlmError> {
    let obj = expect_object(value, "OpenAI Responses request")?;
    let mut request = ChatRequest::new(Vec::new());
    let mut openai_options = Map::new();

    request.common_params.model = required_string(obj, "model", "OpenAI Responses request")?;
    request.common_params.temperature = optional_f64(obj, "temperature");
    request.common_params.top_p = optional_f64(obj, "top_p");
    request.common_params.max_completion_tokens = optional_u32(obj, "max_output_tokens");
    request.stream = optional_bool(obj, "stream").unwrap_or(false);

    if let Some(store) = optional_bool(obj, "store") {
        openai_options.insert("store".to_string(), Value::Bool(store));
    }
    if let Some(parallel_tool_calls) = optional_bool(obj, "parallel_tool_calls")
        .or_else(|| optional_bool(obj, "parallelToolCalls"))
    {
        openai_options.insert(
            "parallelToolCalls".to_string(),
            Value::Bool(parallel_tool_calls),
        );
    }
    if let Some(system_message_mode) = optional_string(obj, "system_message_mode")
        .or_else(|| optional_string(obj, "systemMessageMode"))
    {
        openai_options.insert(
            "systemMessageMode".to_string(),
            Value::String(system_message_mode),
        );
    }
    if let Some(reasoning) = obj.get("reasoning").and_then(Value::as_object)
        && let Some(effort) = reasoning.get("effort").and_then(Value::as_str)
    {
        openai_options.insert(
            "reasoningEffort".to_string(),
            Value::String(effort.to_string()),
        );
    }
    if let Some(text) = obj.get("text").and_then(Value::as_object)
        && let Some(format) = text.get("format")
        && let Some(parsed) = parse_json_schema_response_format(format)
    {
        request.response_format = Some(parsed);
    }

    let mut tools = if let Some(value) = obj.get("tools") {
        parse_openai_responses_tools(value)?
    } else {
        Vec::new()
    };
    let tool_registry = ResponsesToolRegistry::from_tools(&tools);

    if let Some(choice) = obj.get("tool_choice") {
        request.tool_choice = parse_openai_responses_tool_choice(choice, &tool_registry);
    }
    if !tools.is_empty() {
        request.tools = Some(std::mem::take(&mut tools));
    }

    if let Some(instructions) = optional_string(obj, "instructions")
        && !instructions.trim().is_empty()
    {
        let role = match openai_options
            .get("systemMessageMode")
            .and_then(Value::as_str)
        {
            Some("developer") => MessageRole::Developer,
            _ => MessageRole::System,
        };
        request.messages.push(text_message(role, instructions));
    }

    if let Some(input) = obj.get("input") {
        let items = expect_array(input, "OpenAI Responses request.input")?;
        let mut index = 0usize;
        let mut call_names = HashMap::new();
        while index < items.len() {
            if should_skip_item_reference_for_approval(items, index) {
                index += 1;
                continue;
            }
            if let Some(message) =
                parse_openai_responses_input_item(&items[index], &tool_registry, &mut call_names)?
            {
                request.messages.push(message);
            }
            index += 1;
        }
        request.messages = compact_adjacent_messages(std::mem::take(&mut request.messages));
    }

    if !openai_options.is_empty() {
        request
            .provider_options_map
            .insert("openai", Value::Object(openai_options));
    }

    Ok(request)
}

fn parse_openai_responses_tools(value: &Value) -> Result<Vec<Tool>, LlmError> {
    let mut tools = Vec::new();
    for tool in expect_array(value, "OpenAI Responses request.tools")? {
        tools.push(parse_openai_responses_tool(tool)?);
    }
    Ok(tools)
}

fn parse_openai_responses_tool(value: &Value) -> Result<Tool, LlmError> {
    let obj = expect_object(value, "OpenAI Responses request.tools[]")?;
    let kind = required_string(obj, "type", "OpenAI Responses request.tools[]")?;

    if kind == "function" {
        let name = required_string(obj, "name", "OpenAI Responses function tool")?;
        let description = optional_string(obj, "description").unwrap_or_default();
        let parameters = openai_function_input_schema(obj);
        let mut tool = Tool::function(name, description, parameters);
        if let Tool::Function { function } = &mut tool {
            populate_openai_function_tool_metadata(function, obj);
        }
        return Ok(tool);
    }

    Ok(parse_openai_provider_defined_tool(
        kind.as_str(),
        obj,
        Some("type"),
    ))
}

fn parse_openai_responses_tool_choice(
    value: &Value,
    registry: &ResponsesToolRegistry,
) -> Option<ToolChoice> {
    match value {
        Value::String(choice) => match choice.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "required" => Some(ToolChoice::Required),
            "none" => Some(ToolChoice::None),
            _ => None,
        },
        Value::Object(obj) => obj
            .get("type")
            .and_then(Value::as_str)
            .map(|kind| match kind {
                "function" => obj
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToolChoice::tool)
                    .unwrap_or(ToolChoice::Auto),
                other => ToolChoice::tool(registry.tool_name_for_type(other)),
            }),
        _ => None,
    }
}

fn should_skip_item_reference_for_approval(items: &[Value], index: usize) -> bool {
    let Some(current) = items.get(index).and_then(Value::as_object) else {
        return false;
    };
    if current.get("type").and_then(Value::as_str) != Some("item_reference") {
        return false;
    }
    let Some(id) = current.get("id").and_then(Value::as_str) else {
        return false;
    };
    let Some(next) = items.get(index + 1).and_then(Value::as_object) else {
        return false;
    };

    next.get("type").and_then(Value::as_str) == Some("mcp_approval_response")
        && next
            .get("approval_request_id")
            .or_else(|| next.get("approvalRequestId"))
            .and_then(Value::as_str)
            == Some(id)
}

fn parse_openai_responses_input_item(
    value: &Value,
    registry: &ResponsesToolRegistry,
    call_names: &mut HashMap<String, String>,
) -> Result<Option<ChatMessage>, LlmError> {
    let obj = expect_object(value, "OpenAI Responses input item")?;

    if obj.contains_key("role") || obj.get("type").and_then(Value::as_str) == Some("message") {
        return Ok(Some(parse_openai_responses_message_item(obj)?));
    }

    let Some(kind) = obj.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };

    match kind {
        "item_reference" => {
            let id = required_string(obj, "id", "OpenAI Responses item_reference")?;
            let mut message = text_message(MessageRole::Assistant, String::new());
            message.metadata.id = Some(id);
            Ok(Some(message))
        }
        "reasoning" => Ok(Some(parse_openai_responses_reasoning_item(obj)?)),
        "function_call" => {
            let message = parse_openai_responses_function_call_item(obj, registry)?;
            if let Some(ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                ..
            }) = message
                .content
                .as_multimodal()
                .and_then(|parts| parts.first())
            {
                call_names.insert(tool_call_id.clone(), tool_name.clone());
            }
            Ok(Some(message))
        }
        "local_shell_call" | "shell_call" | "apply_patch_call" => {
            let message = parse_openai_responses_provider_call_item(obj, registry, kind)?;
            if let Some(ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                ..
            }) = message
                .content
                .as_multimodal()
                .and_then(|parts| parts.first())
            {
                call_names.insert(tool_call_id.clone(), tool_name.clone());
            }
            Ok(Some(message))
        }
        "function_call_output" => Ok(Some(parse_openai_responses_function_call_output_item(
            obj, call_names,
        )?)),
        "local_shell_call_output" | "shell_call_output" | "apply_patch_call_output" => Ok(Some(
            parse_openai_responses_provider_call_output_item(obj, registry, kind)?,
        )),
        "mcp_approval_response" => Ok(Some(parse_openai_responses_approval_item(obj)?)),
        _ => Ok(None),
    }
}

fn parse_openai_responses_message_item(obj: &Map<String, Value>) -> Result<ChatMessage, LlmError> {
    let raw_role = required_string(obj, "role", "OpenAI Responses message item")?;
    let role = match raw_role.as_str() {
        "system" => MessageRole::System,
        "developer" => MessageRole::Developer,
        "assistant" => MessageRole::Assistant,
        "user" => MessageRole::User,
        "tool" => MessageRole::Tool,
        other => {
            return Err(LlmError::ParseError(format!(
                "unsupported OpenAI Responses message role `{other}`"
            )));
        }
    };

    let parts = match obj.get("content") {
        Some(Value::String(text)) => parse_text_like_content_parts(text),
        Some(Value::Array(parts)) => parse_openai_responses_message_content(parts, &role)?,
        Some(Value::Null) | None => Vec::new(),
        _ => {
            return Err(LlmError::ParseError(
                "OpenAI Responses message content must be a string or array".to_string(),
            ));
        }
    };

    let mut message = message_from_parts(role, parts);
    if let Some(id) = optional_string(obj, "id")
        && !id.is_empty()
    {
        message.metadata.id = Some(id);
    }
    Ok(message)
}

fn parse_openai_responses_message_content(
    parts: &[Value],
    role: &MessageRole,
) -> Result<Vec<ContentPart>, LlmError> {
    let mut out = Vec::new();
    for value in parts {
        let obj = expect_object(value, "OpenAI Responses message content part")?;
        let kind = required_string(obj, "type", "OpenAI Responses message content part")?;

        match kind.as_str() {
            "input_text" | "output_text" | "text" => {
                out.extend(parse_text_like_content_parts(
                    &optional_string(obj, "text").unwrap_or_default(),
                ));
            }
            "input_image" | "output_image" => {
                out.push(parse_openai_responses_image_part(obj));
            }
            "input_file" => {
                out.push(parse_openai_responses_file_part(obj)?);
            }
            "tool_use" => {
                let tool_call_id =
                    required_string(obj, "id", "OpenAI Responses tool_use content part")?;
                let tool_name =
                    required_string(obj, "name", "OpenAI Responses tool_use content part")?;
                let arguments = obj
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new()));
                out.push(legacy_content::request_tool_call_part(
                    tool_call_id,
                    tool_name,
                    arguments,
                    None,
                    None,
                    ProviderOptionsMap::default(),
                ));
            }
            other if matches!(role, MessageRole::Assistant) && other == "tool_call" => {
                let tool_call_id =
                    required_string(obj, "id", "OpenAI Responses assistant tool_call part")?;
                let tool_name =
                    required_string(obj, "name", "OpenAI Responses assistant tool_call part")?;
                let arguments = obj
                    .get("arguments")
                    .map(parse_embedded_json)
                    .transpose()?
                    .unwrap_or_else(|| Value::Object(Map::new()));
                out.push(legacy_content::request_tool_call_part(
                    tool_call_id,
                    tool_name,
                    arguments,
                    None,
                    None,
                    ProviderOptionsMap::default(),
                ));
            }
            other => {
                return Err(LlmError::ParseError(format!(
                    "unsupported OpenAI Responses message content part `{other}`"
                )));
            }
        }
    }
    Ok(out)
}

pub(super) fn parse_openai_responses_image_part(obj: &Map<String, Value>) -> ContentPart {
    let provider_options = openai_image_detail_provider_options(obj);

    if let Some(file_id) = optional_string(obj, "file_id") {
        return legacy_content::request_image_part(
            FilePartSource::provider_reference(ProviderReference::single("openai", file_id)),
            None,
            None,
            provider_options,
        );
    }

    let image_url = optional_string(obj, "image_url").unwrap_or_default();
    let source = if image_url.starts_with("data:") {
        FilePartSource::base64(strip_data_url_prefix(&image_url))
    } else {
        FilePartSource::url(image_url)
    };

    legacy_content::request_image_part(source, None, None, provider_options)
}

fn parse_openai_responses_file_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    if let Some(file_id) = optional_string(obj, "file_id") {
        return Ok(legacy_content::request_file_part(
            FilePartSource::provider_reference(ProviderReference::single("openai", file_id)),
            "application/pdf",
            optional_string(obj, "filename"),
            ProviderOptionsMap::default(),
        ));
    }
    if let Some(file_url) = optional_string(obj, "file_url") {
        return Ok(legacy_content::request_file_part(
            FilePartSource::url(file_url),
            infer_document_media_type(None, None),
            optional_string(obj, "filename"),
            ProviderOptionsMap::default(),
        ));
    }
    if let Some(file_data) = optional_string(obj, "file_data") {
        return Ok(legacy_content::request_file_part(
            FilePartSource::base64(strip_data_url_prefix(&file_data)),
            "application/pdf",
            optional_string(obj, "filename"),
            ProviderOptionsMap::default(),
        ));
    }

    Err(LlmError::ParseError(
        "OpenAI Responses input_file part requires file_id, file_url, or file_data".to_string(),
    ))
}

fn parse_openai_responses_reasoning_item(
    obj: &Map<String, Value>,
) -> Result<ChatMessage, LlmError> {
    let text = collect_reasoning_summary(obj.get("summary")).unwrap_or_default();
    let mut openai_options = Map::new();
    if let Some(item_id) = optional_string(obj, "id")
        && !item_id.is_empty()
    {
        openai_options.insert("itemId".to_string(), Value::String(item_id));
    }
    if let Some(encrypted) = obj.get("encrypted_content")
        && !encrypted.is_null()
    {
        openai_options.insert("reasoningEncryptedContent".to_string(), encrypted.clone());
    }

    let mut provider_options = ProviderOptionsMap::default();
    if !openai_options.is_empty() {
        provider_options.insert("openai", Value::Object(openai_options));
    }

    Ok(message_from_parts(
        MessageRole::Assistant,
        vec![legacy_content::request_reasoning_part(
            text,
            provider_options,
        )],
    ))
}

fn parse_openai_responses_function_call_item(
    obj: &Map<String, Value>,
    registry: &ResponsesToolRegistry,
) -> Result<ChatMessage, LlmError> {
    let tool_call_id = required_string(obj, "call_id", "OpenAI Responses function_call item")?;
    let raw_name = required_string(obj, "name", "OpenAI Responses function_call item")?;
    let tool_name = registry.resolve_name(&raw_name);
    let arguments = obj
        .get("arguments")
        .map(parse_embedded_json)
        .transpose()?
        .unwrap_or_else(|| Value::Object(Map::new()));
    let provider_options = openai_item_id_provider_options(obj);

    Ok(message_from_parts(
        MessageRole::Assistant,
        vec![legacy_content::request_tool_call_part(
            tool_call_id,
            tool_name,
            arguments,
            None,
            None,
            provider_options,
        )],
    ))
}

fn parse_openai_responses_provider_call_item(
    obj: &Map<String, Value>,
    registry: &ResponsesToolRegistry,
    kind: &str,
) -> Result<ChatMessage, LlmError> {
    let tool_call_id = required_string(obj, "call_id", "OpenAI Responses provider call item")?;
    let tool_name = registry.tool_name_for_type(openai_responses_provider_call_wire_type(kind));
    let payload_key = openai_responses_provider_call_payload_key(kind);
    let mut arguments = Map::new();
    arguments.insert(
        payload_key.to_string(),
        normalize_openai_provider_call_payload(
            kind,
            obj.get(payload_key).unwrap_or(&Value::Object(Map::new())),
        ),
    );
    let provider_options = openai_item_id_provider_options(obj);

    Ok(message_from_parts(
        MessageRole::Assistant,
        vec![legacy_content::request_tool_call_part(
            tool_call_id,
            tool_name,
            Value::Object(arguments),
            None,
            Some(true),
            provider_options,
        )],
    ))
}

fn parse_openai_responses_function_call_output_item(
    obj: &Map<String, Value>,
    call_names: &HashMap<String, String>,
) -> Result<ChatMessage, LlmError> {
    let tool_call_id =
        required_string(obj, "call_id", "OpenAI Responses function_call_output item")?;
    let output = parse_openai_responses_tool_output(
        obj.get("output").unwrap_or(&Value::String(String::new())),
        false,
    )?;
    let tool_name = call_names.get(&tool_call_id).cloned().unwrap_or_default();

    Ok(message_from_parts(
        MessageRole::Tool,
        vec![legacy_content::request_tool_result_part(
            tool_call_id,
            tool_name,
            output,
            None,
            None,
            ProviderOptionsMap::default(),
        )],
    ))
}

fn parse_openai_responses_provider_call_output_item(
    obj: &Map<String, Value>,
    registry: &ResponsesToolRegistry,
    kind: &str,
) -> Result<ChatMessage, LlmError> {
    let tool_call_id =
        required_string(obj, "call_id", "OpenAI Responses provider call output item")?;
    let tool_name = registry.tool_name_for_type(match kind {
        "local_shell_call_output" => "local_shell",
        "shell_call_output" => "shell",
        "apply_patch_call_output" => "apply_patch",
        other => other,
    });

    let output = match kind {
        "local_shell_call_output" => ToolResultOutput::json(json!({
            "output": normalize_openai_shell_output_value(
                obj.get("output").unwrap_or(&Value::Null)
            )
        })),
        "shell_call_output" => ToolResultOutput::json(json!({
            "output": normalize_openai_shell_output_value(
                obj.get("output").unwrap_or(&Value::Null)
            )
        })),
        "apply_patch_call_output" => ToolResultOutput::json(json!({
            "status": obj.get("status").cloned().unwrap_or(Value::Null),
            "output": obj.get("output").cloned().unwrap_or(Value::Null)
        })),
        _ => ToolResultOutput::text(String::new()),
    };

    Ok(message_from_parts(
        MessageRole::Tool,
        vec![legacy_content::request_tool_result_part(
            tool_call_id,
            tool_name,
            output,
            None,
            Some(true),
            ProviderOptionsMap::default(),
        )],
    ))
}

fn parse_openai_responses_approval_item(obj: &Map<String, Value>) -> Result<ChatMessage, LlmError> {
    let approval_id = required_string(
        obj,
        "approval_request_id",
        "OpenAI Responses mcp_approval_response item",
    )?;
    let approved = obj.get("approve").and_then(Value::as_bool).unwrap_or(false);

    Ok(message_from_parts(
        MessageRole::Tool,
        vec![ContentPart::ToolApprovalResponse {
            approval_id,
            approved,
            reason: optional_string(obj, "reason"),
            provider_executed: Some(true),
            provider_options: ProviderOptionsMap::default(),
        }],
    ))
}

fn parse_openai_responses_tool_output(
    value: &Value,
    is_error: bool,
) -> Result<ToolResultOutput, LlmError> {
    match value {
        Value::String(text) => Ok(parse_tool_result_output_from_string(text, is_error)),
        Value::Array(items) => Ok(ToolResultOutput::content(
            parse_openai_responses_tool_output_parts(items)?,
        )),
        other => Ok(if is_error {
            ToolResultOutput::error_json(other.clone())
        } else {
            ToolResultOutput::json(other.clone())
        }),
    }
}

fn parse_openai_responses_tool_output_parts(
    items: &[Value],
) -> Result<Vec<ToolResultContentPart>, LlmError> {
    let mut parts = Vec::new();
    for value in items {
        let obj = expect_object(value, "OpenAI Responses tool output part")?;
        let kind = required_string(obj, "type", "OpenAI Responses tool output part")?;
        match kind.as_str() {
            "input_text" | "output_text" | "text" => {
                parts.push(ToolResultContentPart::text(
                    optional_string(obj, "text").unwrap_or_default(),
                ));
            }
            "input_image" | "output_image" => {
                if let Some(file_id) = optional_string(obj, "file_id") {
                    parts.push(ToolResultContentPart::image_file_reference(
                        ProviderReference::single("openai", file_id),
                    ));
                } else {
                    parts.push(ToolResultContentPart::image_url(
                        optional_string(obj, "image_url").unwrap_or_default(),
                    ));
                }
            }
            "input_file" => {
                parts.push(if let Some(url) = optional_string(obj, "file_url") {
                    ToolResultContentPart::file_url(url)
                } else if let Some(file_id) = optional_string(obj, "file_id") {
                    ToolResultContentPart::file_reference(ProviderReference::single(
                        "openai", file_id,
                    ))
                } else {
                    ToolResultContentPart::file_data(
                        optional_string(obj, "file_data")
                            .map(|value| strip_data_url_prefix(&value))
                            .unwrap_or_default(),
                        "application/pdf",
                        optional_string(obj, "filename"),
                    )
                });
            }
            other => {
                return Err(LlmError::ParseError(format!(
                    "unsupported OpenAI Responses tool output part `{other}`"
                )));
            }
        }
    }
    Ok(parts)
}

fn openai_responses_wire_type_for_tool(tool: &Tool) -> Option<String> {
    let Tool::ProviderDefined(provider_tool) = tool else {
        return None;
    };
    if provider_tool.provider() != Some("openai") {
        return None;
    }
    match provider_tool.tool_type()? {
        "computer_use" => Some("computer_use_preview".to_string()),
        other => Some(other.to_string()),
    }
}

fn openai_responses_provider_call_wire_type(kind: &str) -> &str {
    match kind {
        "local_shell_call" | "local_shell_call_output" => "local_shell",
        "shell_call" | "shell_call_output" => "shell",
        "apply_patch_call" | "apply_patch_call_output" => "apply_patch",
        other => other,
    }
}

fn openai_responses_provider_call_payload_key(kind: &str) -> &str {
    match kind {
        "apply_patch_call" => "operation",
        _ => "action",
    }
}

fn normalize_openai_provider_call_payload(kind: &str, value: &Value) -> Value {
    match kind {
        "shell_call" => normalize_openai_shell_action(value),
        _ => value.clone(),
    }
}

fn normalize_openai_shell_action(value: &Value) -> Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };

    let mut out = Map::new();
    if let Some(commands) = obj.get("commands") {
        out.insert("commands".to_string(), commands.clone());
    }
    if let Some(timeout_ms) = obj.get("timeout_ms") {
        out.insert("timeoutMs".to_string(), timeout_ms.clone());
    }
    if let Some(max_output_length) = obj.get("max_output_length") {
        out.insert("maxOutputLength".to_string(), max_output_length.clone());
    }

    for (key, inner) in obj {
        if matches!(
            key.as_str(),
            "commands" | "timeout_ms" | "max_output_length"
        ) {
            continue;
        }
        out.insert(key.clone(), inner.clone());
    }

    Value::Object(out)
}

fn normalize_openai_shell_output_value(value: &Value) -> Value {
    let Some(items) = value.as_array() else {
        return value.clone();
    };

    Value::Array(
        items
            .iter()
            .map(normalize_openai_shell_output_item)
            .collect(),
    )
}

fn normalize_openai_shell_output_item(value: &Value) -> Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };

    let mut out = obj.clone();
    if let Some(outcome) = obj.get("outcome").and_then(Value::as_object) {
        let mut normalized_outcome = outcome.clone();
        if let Some(exit_code) = normalized_outcome.remove("exit_code") {
            normalized_outcome.insert("exitCode".to_string(), exit_code);
        }
        out.insert("outcome".to_string(), Value::Object(normalized_outcome));
    }

    Value::Object(out)
}

fn openai_item_id_provider_options(obj: &Map<String, Value>) -> ProviderOptionsMap {
    let mut provider_options = ProviderOptionsMap::default();
    let Some(item_id) = obj.get("id").and_then(Value::as_str) else {
        return provider_options;
    };

    provider_options.insert(
        "openai",
        json!({
            "itemId": item_id,
        }),
    );
    provider_options
}

fn openai_image_detail_provider_options(obj: &Map<String, Value>) -> ProviderOptionsMap {
    let mut provider_options = ProviderOptionsMap::default();
    let Some(detail) = obj.get("detail").and_then(Value::as_str) else {
        return provider_options;
    };

    provider_options.insert(
        "openai",
        json!({
            "imageDetail": detail,
        }),
    );
    provider_options
}

fn infer_image_media_type(obj: &Map<String, Value>) -> String {
    let Some(image_url) = obj.get("image_url").and_then(Value::as_str) else {
        return "image/*".to_string();
    };

    if image_url.starts_with("data:")
        && let Some(without_prefix) = image_url.strip_prefix("data:")
        && let Some((media_type, _)) = without_prefix.split_once(';')
        && !media_type.is_empty()
    {
        return media_type.to_string();
    }

    "image/*".to_string()
}
