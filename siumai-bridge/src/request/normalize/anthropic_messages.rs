//! Anthropic Messages request JSON normalization.

use super::*;

#[cfg(feature = "anthropic")]
pub(super) fn parse_json_to_chat_request(value: &Value) -> Result<ChatRequest, LlmError> {
    let obj = expect_object(value, "Anthropic Messages request")?;
    let mut request = ChatRequest::new(Vec::new());
    let mut anthropic_options = Map::new();

    request.common_params.model = required_string(obj, "model", "Anthropic Messages request")?;
    request.common_params.temperature = optional_f64(obj, "temperature");
    request.common_params.top_p = optional_f64(obj, "top_p");
    request.common_params.top_k = optional_f64(obj, "top_k");
    request.common_params.max_tokens = optional_u32(obj, "max_tokens");
    request.common_params.stop_sequences =
        optional_stop_sequences(obj.get("stop_sequences").or_else(|| obj.get("stop")))?;
    request.stream = optional_bool(obj, "stream").unwrap_or(false);

    if let Some(system) = obj.get("system") {
        request
            .messages
            .extend(parse_anthropic_system_messages(system)?);
    }

    if let Some(messages) = obj.get("messages") {
        for value in expect_array(messages, "Anthropic Messages request.messages")? {
            request.messages.push(parse_anthropic_message(value)?);
        }
    }

    let mut tools = if let Some(value) = obj.get("tools") {
        parse_anthropic_tools(value)?
    } else {
        Vec::new()
    };

    if let Some(format) = extract_anthropic_reserved_json_tool(&mut tools) {
        request.response_format = Some(format);
        anthropic_options.insert(
            "structuredOutputMode".to_string(),
            Value::String("jsonTool".to_string()),
        );
    }

    if let Some(choice) = obj.get("tool_choice") {
        let (tool_choice, disable_parallel_tool_use) = parse_anthropic_tool_choice(choice)?;
        request.tool_choice = tool_choice;
        if disable_parallel_tool_use {
            anthropic_options.insert("disableParallelToolUse".to_string(), Value::Bool(true));
        }
    }

    if !tools.is_empty() {
        request.tools = Some(tools);
    }

    if request.response_format.is_none() {
        let output_config_format = obj
            .get("output_config")
            .and_then(Value::as_object)
            .and_then(|output_config| output_config.get("format"));
        let legacy_output_format = obj.get("output_format");

        if let Some(value) = output_config_format.or(legacy_output_format)
            && let Some(format) = parse_json_schema_response_format(value)
        {
            request.response_format = Some(format);
            anthropic_options.insert(
                "structuredOutputMode".to_string(),
                Value::String("outputFormat".to_string()),
            );
        }
    }

    if let Some(thinking) = obj.get("thinking")
        && thinking.is_object()
    {
        anthropic_options.insert(
            "thinking".to_string(),
            map_anthropic_wire_json_to_sdk_shape(thinking),
        );
    }
    if let Some(cache_control) = obj.get("cache_control")
        && cache_control.is_object()
    {
        anthropic_options.insert(
            "cacheControl".to_string(),
            map_anthropic_wire_json_to_sdk_shape(cache_control),
        );
    }
    if let Some(metadata) = obj.get("metadata")
        && metadata.is_object()
    {
        let metadata = map_anthropic_wire_json_to_sdk_shape(metadata);
        if metadata
            .as_object()
            .is_some_and(|metadata| !metadata.is_empty())
        {
            anthropic_options.insert("metadata".to_string(), metadata);
        }
    }
    if let Some(output_config) = obj.get("output_config").and_then(Value::as_object) {
        if let Some(effort) = output_config.get("effort").and_then(Value::as_str) {
            anthropic_options.insert("effort".to_string(), Value::String(effort.to_string()));
        }
        if let Some(task_budget) = output_config.get("task_budget").and_then(Value::as_object) {
            anthropic_options.insert(
                "taskBudget".to_string(),
                map_anthropic_wire_json_to_sdk_shape(&Value::Object(task_budget.clone())),
            );
        }
    }
    if let Some(mcp_servers) = obj.get("mcp_servers")
        && mcp_servers.is_array()
    {
        anthropic_options.insert(
            "mcpServers".to_string(),
            map_anthropic_wire_json_to_sdk_shape(mcp_servers),
        );
    }
    if let Some(container) = obj.get("container")
        && let Some(mapped_container) = map_anthropic_wire_container_to_sdk_shape(container)
    {
        anthropic_options.insert("container".to_string(), mapped_container);
    }
    if let Some(context_management) = obj.get("context_management")
        && context_management.is_object()
    {
        anthropic_options.insert(
            "contextManagement".to_string(),
            map_anthropic_wire_json_to_sdk_shape(context_management),
        );
    }
    if let Some(speed) = obj.get("speed")
        && !speed.is_null()
    {
        anthropic_options.insert("speed".to_string(), speed.clone());
    }

    if !anthropic_options.is_empty() {
        request
            .provider_options_map
            .insert("anthropic", Value::Object(anthropic_options));
    }

    Ok(request)
}

#[cfg(feature = "anthropic")]
fn map_anthropic_wire_json_to_sdk_shape(value: &Value) -> Value {
    fn map_key(key: &str) -> &str {
        match key {
            "authorization_token" => "authorizationToken",
            "allowed_domains" => "allowedDomains",
            "allowed_tools" => "allowedTools",
            "blocked_domains" => "blockedDomains",
            "budget_tokens" => "budgetTokens",
            "clear_at_least" => "clearAtLeast",
            "clear_tool_inputs" => "clearToolInputs",
            "display_height_px" => "displayHeightPx",
            "display_width_px" => "displayWidthPx",
            "exclude_tools" => "excludeTools",
            "max_uses" => "maxUses",
            "pause_after_compaction" => "pauseAfterCompaction",
            "provider_reference" => "providerReference",
            "skill_id" => "skillId",
            "tool_configuration" => "toolConfiguration",
            "user_id" => "userId",
            "user_location" => "userLocation",
            _ => key,
        }
    }

    match value {
        Value::Object(obj) => Value::Object(
            obj.iter()
                .map(|(key, value)| {
                    (
                        map_key(key).to_string(),
                        map_anthropic_wire_json_to_sdk_shape(value),
                    )
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(map_anthropic_wire_json_to_sdk_shape)
                .collect(),
        ),
        _ => value.clone(),
    }
}

#[cfg(feature = "anthropic")]
fn restore_anthropic_container_skill_sdk_shape(skill: &mut Value) {
    let Some(skill_obj) = skill.as_object_mut() else {
        return;
    };

    if skill_obj.get("type").and_then(Value::as_str) != Some("custom") {
        return;
    }

    if skill_obj.contains_key("providerReference") {
        return;
    }

    let Some(Value::String(skill_id)) = skill_obj.remove("skillId") else {
        return;
    };

    skill_obj.insert(
        "providerReference".to_string(),
        json!({ "anthropic": skill_id }),
    );
}

#[cfg(feature = "anthropic")]
fn map_anthropic_wire_container_to_sdk_shape(value: &Value) -> Option<Value> {
    match value {
        Value::String(id) => {
            if id.trim().is_empty() {
                None
            } else {
                Some(json!({ "id": id }))
            }
        }
        Value::Object(obj) if obj.is_empty() => None,
        Value::Object(_) => {
            let mut mapped = map_anthropic_wire_json_to_sdk_shape(value);
            if let Some(skills) = mapped
                .as_object_mut()
                .and_then(|container| container.get_mut("skills"))
                .and_then(Value::as_array_mut)
            {
                for skill in skills {
                    restore_anthropic_container_skill_sdk_shape(skill);
                }
            }
            Some(mapped)
        }
        _ => None,
    }
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_system_messages(value: &Value) -> Result<Vec<ChatMessage>, LlmError> {
    match value {
        Value::String(text) => Ok(vec![text_message(MessageRole::System, text.clone())]),
        Value::Array(blocks) => {
            let mut messages = Vec::new();
            for value in blocks {
                let block = expect_object(value, "Anthropic system block")?;
                let kind = required_string(block, "type", "Anthropic system block")?;
                if kind != "text" {
                    return Err(LlmError::ParseError(format!(
                        "unsupported Anthropic system block `{kind}`"
                    )));
                }
                let text = optional_string(block, "text").unwrap_or_default();
                let (role, text) = strip_developer_prefix(&text);
                let mut message = text_message(role, text);
                message.metadata.cache_control =
                    block.get("cache_control").and_then(parse_cache_control);
                messages.push(message);
            }
            Ok(messages)
        }
        _ => Err(LlmError::ParseError(
            "Anthropic `system` must be a string or array".to_string(),
        )),
    }
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_message(value: &Value) -> Result<ChatMessage, LlmError> {
    let obj = expect_object(value, "Anthropic message")?;
    let role = required_string(obj, "role", "Anthropic message")?;
    let default_content = Value::String(String::new());
    let content = obj.get("content").unwrap_or(&default_content);

    let parts = parse_anthropic_message_parts(content)?;
    let all_tool_results = !parts.is_empty()
        && parts
            .iter()
            .all(|part| matches!(part, ContentPart::ToolResult { .. }));

    let normalized_role = match role.as_str() {
        "assistant" => MessageRole::Assistant,
        "user" if all_tool_results => MessageRole::Tool,
        "user" => MessageRole::User,
        other => {
            return Err(LlmError::ParseError(format!(
                "unsupported Anthropic message role `{other}`"
            )));
        }
    };

    Ok(message_from_parts(normalized_role, parts))
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_message_parts(value: &Value) -> Result<Vec<ContentPart>, LlmError> {
    if let Some(text) = value.as_str() {
        return Ok(parse_text_like_content_parts(text));
    }

    let mut parts = Vec::new();

    for value in expect_array(value, "Anthropic message.content")? {
        let obj = expect_object(value, "Anthropic message content block")?;
        let kind = required_string(obj, "type", "Anthropic message content block")?;
        let cache_control = obj.get("cache_control").and_then(parse_cache_control);

        match kind.as_str() {
            "text" => {
                let mut block_parts = parse_text_like_content_parts(
                    &optional_string(obj, "text").unwrap_or_default(),
                );
                if let Some(cache_control) = cache_control.as_ref() {
                    for part in &mut block_parts {
                        apply_anthropic_part_cache_control(part, cache_control);
                    }
                }
                parts.extend(block_parts);
            }
            "image" => {
                let mut part = parse_anthropic_image_part(obj)?;
                if let Some(cache_control) = cache_control.as_ref() {
                    apply_anthropic_part_cache_control(&mut part, cache_control);
                }
                parts.push(part);
            }
            "document" => {
                let mut part = parse_anthropic_document_part(obj)?;
                if let Some(cache_control) = cache_control.as_ref() {
                    apply_anthropic_part_cache_control(&mut part, cache_control);
                }
                parts.push(part);
            }
            "tool_use" => {
                let tool_call_id = required_string(obj, "id", "Anthropic tool_use block")?;
                let tool_name = required_string(obj, "name", "Anthropic tool_use block")?;
                let arguments = obj
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new()));
                let mut part = legacy_content::request_tool_call_part(
                    tool_call_id,
                    tool_name,
                    arguments,
                    None,
                    None,
                    ProviderOptionsMap::default(),
                );
                if let Some(cache_control) = cache_control.as_ref() {
                    apply_anthropic_part_cache_control(&mut part, cache_control);
                }
                parts.push(part);
            }
            "tool_result" => {
                let mut part = parse_anthropic_tool_result_part(obj)?;
                if let Some(cache_control) = cache_control.as_ref() {
                    apply_anthropic_part_cache_control(&mut part, cache_control);
                }
                parts.push(part);
            }
            "thinking" => {
                let text = optional_string(obj, "thinking").unwrap_or_default();
                let mut part =
                    legacy_content::request_reasoning_part(text, ProviderOptionsMap::default());
                if let Some(signature) = optional_string(obj, "signature")
                    && !signature.is_empty()
                {
                    merge_anthropic_part_provider_options(
                        &mut part,
                        Map::from_iter([("signature".to_string(), Value::String(signature))]),
                    );
                }
                if let Some(cache_control) = cache_control.as_ref() {
                    apply_anthropic_part_cache_control(&mut part, cache_control);
                }
                parts.push(part);
            }
            "redacted_thinking" => {
                let data = optional_string(obj, "data");
                let mut part = legacy_content::request_reasoning_part(
                    String::new(),
                    ProviderOptionsMap::default(),
                );
                if let Some(data) = data
                    && !data.is_empty()
                {
                    merge_anthropic_part_provider_options(
                        &mut part,
                        Map::from_iter([("redactedData".to_string(), Value::String(data))]),
                    );
                }
                if let Some(cache_control) = cache_control.as_ref() {
                    apply_anthropic_part_cache_control(&mut part, cache_control);
                }
                parts.push(part);
            }
            other => {
                return Err(LlmError::ParseError(format!(
                    "unsupported Anthropic content block `{other}`"
                )));
            }
        }
    }

    Ok(parts)
}

#[cfg(feature = "anthropic")]
pub(super) fn parse_anthropic_image_part(
    obj: &Map<String, Value>,
) -> Result<ContentPart, LlmError> {
    let source = obj
        .get("source")
        .ok_or_else(|| LlmError::ParseError("Anthropic image block is missing source".to_string()))
        .and_then(|value| expect_object(value, "Anthropic image block.source"))?;
    let source_type = required_string(source, "type", "Anthropic image block.source")?;

    let media_source = match source_type.as_str() {
        "url" => FilePartSource::url(required_string(
            source,
            "url",
            "Anthropic image block.source",
        )?),
        "base64" => FilePartSource::base64(required_string(
            source,
            "data",
            "Anthropic image block.source",
        )?),
        "file" => FilePartSource::provider_reference(ProviderReference::single(
            "anthropic",
            required_string(source, "file_id", "Anthropic image block.source")?,
        )),
        other => {
            return Err(LlmError::ParseError(format!(
                "unsupported Anthropic image source `{other}`"
            )));
        }
    };

    Ok(legacy_content::request_image_part(
        media_source,
        optional_string(obj, "media_type").or_else(|| optional_string(obj, "mime_type")),
        None,
        ProviderOptionsMap::default(),
    ))
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_document_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    let source = obj
        .get("source")
        .ok_or_else(|| {
            LlmError::ParseError("Anthropic document block is missing source".to_string())
        })
        .and_then(|value| expect_object(value, "Anthropic document block.source"))?;
    let source_type = required_string(source, "type", "Anthropic document block.source")?;
    let title = optional_string(obj, "title");
    let context = optional_string(obj, "context");

    let (media_source, media_type) = match source_type.as_str() {
        "url" => {
            let url = required_string(source, "url", "Anthropic document block.source")?;
            let media_type = infer_document_media_type(title.as_deref(), Some(url.as_str()));
            (FilePartSource::url(url), media_type)
        }
        "base64" => (
            FilePartSource::base64(required_string(
                source,
                "data",
                "Anthropic document block.source",
            )?),
            optional_string(source, "media_type").unwrap_or_else(|| "application/pdf".to_string()),
        ),
        "text" => (
            FilePartSource::binary(
                optional_string(source, "data")
                    .unwrap_or_default()
                    .into_bytes(),
            ),
            optional_string(source, "media_type").unwrap_or_else(|| "text/plain".to_string()),
        ),
        "file" => (
            FilePartSource::provider_reference(ProviderReference::single(
                "anthropic",
                required_string(source, "file_id", "Anthropic document block.source")?,
            )),
            optional_string(source, "media_type")
                .unwrap_or_else(|| infer_document_media_type(title.as_deref(), title.as_deref())),
        ),
        other => {
            return Err(LlmError::ParseError(format!(
                "unsupported Anthropic document source `{other}`"
            )));
        }
    };

    let mut provider_options = ProviderOptionsMap::default();
    let mut anthropic = Map::new();
    if obj
        .get("citations")
        .and_then(Value::as_object)
        .and_then(|citations| citations.get("enabled"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        anthropic.insert("citations".to_string(), json!({ "enabled": true }));
    }
    if let Some(title) = title.as_ref()
        && !title.is_empty()
    {
        anthropic.insert("title".to_string(), Value::String(title.clone()));
    }
    if let Some(context) = context.as_ref()
        && !context.is_empty()
    {
        anthropic.insert("context".to_string(), Value::String(context.clone()));
    }
    if !anthropic.is_empty() {
        provider_options.insert("anthropic", Value::Object(anthropic));
    }

    Ok(legacy_content::request_file_part(
        media_source,
        media_type,
        title,
        provider_options,
    ))
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_tool_result_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    let tool_call_id = required_string(obj, "tool_use_id", "Anthropic tool_result block")?;
    let default_output = Value::String(String::new());
    let output_value = obj.get("content").unwrap_or(&default_output);
    let is_error = obj
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let output = parse_anthropic_tool_result_output(output_value, is_error)?;

    Ok(legacy_content::request_tool_result_part(
        tool_call_id,
        String::new(),
        output,
        None,
        None,
        ProviderOptionsMap::default(),
    ))
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_tool_result_output(
    value: &Value,
    is_error: bool,
) -> Result<ToolResultOutput, LlmError> {
    match value {
        Value::String(text) => Ok(parse_tool_result_output_from_string(text, is_error)),
        Value::Array(parts) => Ok(ToolResultOutput::content(
            parse_anthropic_tool_result_content(parts)?,
        )),
        other => Ok(if is_error {
            ToolResultOutput::error_json(other.clone())
        } else {
            ToolResultOutput::json(other.clone())
        }),
    }
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_tool_result_content(
    parts: &[Value],
) -> Result<Vec<ToolResultContentPart>, LlmError> {
    let mut out = Vec::new();
    for value in parts {
        let obj = expect_object(value, "Anthropic tool_result content block")?;
        let kind = required_string(obj, "type", "Anthropic tool_result content block")?;
        match kind.as_str() {
            "text" => {
                out.push(ToolResultContentPart::text(
                    optional_string(obj, "text").unwrap_or_default(),
                ));
            }
            "image" => {
                let source = obj
                    .get("source")
                    .ok_or_else(|| {
                        LlmError::ParseError(
                            "Anthropic tool_result image block is missing source".to_string(),
                        )
                    })
                    .and_then(|value| expect_object(value, "Anthropic tool_result image source"))?;
                let source_type =
                    required_string(source, "type", "Anthropic tool_result image source")?;
                let media_source = match source_type.as_str() {
                    "url" => MediaSource::Url {
                        url: required_string(source, "url", "Anthropic tool_result image source")?,
                    },
                    "base64" => MediaSource::Base64 {
                        data: required_string(
                            source,
                            "data",
                            "Anthropic tool_result image source",
                        )?,
                    },
                    other => {
                        return Err(LlmError::ParseError(format!(
                            "unsupported Anthropic tool_result image source `{other}`"
                        )));
                    }
                };
                match media_source {
                    MediaSource::Url { url } => out.push(ToolResultContentPart::image_url(url)),
                    MediaSource::Base64 { data } => {
                        out.push(ToolResultContentPart::image_data(data, "image/*"))
                    }
                    MediaSource::Binary { data } => out.push(ToolResultContentPart::image_data(
                        base64::engine::general_purpose::STANDARD.encode(data),
                        "image/*",
                    )),
                }
            }
            other => {
                return Err(LlmError::ParseError(format!(
                    "unsupported Anthropic tool_result content block `{other}`"
                )));
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_tools(value: &Value) -> Result<Vec<Tool>, LlmError> {
    let mut tools = Vec::new();
    for value in expect_array(value, "Anthropic request.tools")? {
        tools.push(parse_anthropic_tool(value)?);
    }
    Ok(tools)
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_tool(value: &Value) -> Result<Tool, LlmError> {
    let obj = expect_object(value, "Anthropic request.tools[]")?;
    if let Some(kind) = obj.get("type").and_then(Value::as_str)
        && let Some(provider_id) = anthropic_provider_tool_id_from_wire_type(kind)
    {
        let mut tool = default_provider_defined_tool(&provider_id).unwrap_or_else(|| {
            Tool::provider_defined(provider_id.clone(), default_anthropic_tool_name(kind))
        });

        if let Tool::ProviderDefined(provider_tool) = &mut tool {
            if let Some(name) = optional_string(obj, "name")
                && !name.is_empty()
            {
                provider_tool.name = name;
            }
            provider_tool.args = map_anthropic_wire_json_to_sdk_shape(
                &collect_remaining_object_fields(obj, &["type", "name"]),
            );
        }
        return Ok(tool);
    }

    let name = required_string(obj, "name", "Anthropic function tool")?;
    let description = optional_string(obj, "description").unwrap_or_default();
    let input_schema = obj
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut tool = Tool::function(name, description, input_schema);

    if let Tool::Function { function } = &mut tool {
        function.strict = obj.get("strict").and_then(Value::as_bool);
        if let Some(input_examples) = obj.get("input_examples").and_then(Value::as_array) {
            function.input_examples = Some(input_examples.clone());
        }

        let mut anthropic = Map::new();
        if let Some(value) = obj.get("defer_loading").and_then(Value::as_bool) {
            anthropic.insert("deferLoading".to_string(), Value::Bool(value));
        }
        if let Some(value) = obj.get("eager_input_streaming").and_then(Value::as_bool) {
            anthropic.insert("eagerInputStreaming".to_string(), Value::Bool(value));
        }
        if let Some(value) = obj.get("cache_control")
            && value.is_object()
        {
            anthropic.insert("cacheControl".to_string(), value.clone());
        }
        if let Some(value) = obj.get("allowed_callers")
            && value.is_array()
        {
            anthropic.insert("allowedCallers".to_string(), value.clone());
        }
        if !anthropic.is_empty() {
            function
                .provider_options_map
                .insert("anthropic", Value::Object(anthropic));
        }
    }

    Ok(tool)
}

#[cfg(feature = "anthropic")]
fn extract_anthropic_reserved_json_tool(tools: &mut Vec<Tool>) -> Option<ResponseFormat> {
    let index = tools.iter().position(|tool| {
        matches!(
            tool,
            Tool::Function { function }
                if function.name == "json" && !function.parameters.is_null()
        )
    })?;
    let tool = tools.remove(index);
    match tool {
        Tool::Function { function } => {
            let mut format = ResponseFormat::json_schema(function.parameters);
            if !function.description.is_empty() {
                format = format.with_description(function.description);
            }
            Some(format)
        }
        _ => None,
    }
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_tool_choice(value: &Value) -> Result<(Option<ToolChoice>, bool), LlmError> {
    let obj = expect_object(value, "Anthropic tool_choice")?;
    let disable_parallel_tool_use = obj
        .get("disable_parallel_tool_use")
        .or_else(|| obj.get("disableParallelToolUse"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let tool_choice = match obj.get("type").and_then(Value::as_str) {
        Some("auto") => Some(ToolChoice::Auto),
        Some("any") => Some(ToolChoice::Required),
        Some("tool") => obj
            .get("name")
            .and_then(Value::as_str)
            .map(ToolChoice::tool),
        Some(_) | None => None,
    };

    Ok((tool_choice, disable_parallel_tool_use))
}

#[cfg(feature = "anthropic")]
fn strip_developer_prefix(text: &str) -> (MessageRole, String) {
    let prefix = "Developer instructions: ";
    if let Some(stripped) = text.strip_prefix(prefix) {
        (MessageRole::Developer, stripped.to_string())
    } else {
        (MessageRole::System, text.to_string())
    }
}

#[cfg(feature = "anthropic")]
fn parse_cache_control(value: &Value) -> Option<CacheControl> {
    let obj = value.as_object()?;
    let ttl = obj
        .get("ttl")
        .and_then(Value::as_u64)
        .map(Duration::from_secs);
    match obj.get("type").and_then(Value::as_str) {
        Some("ephemeral") | None => Some(if ttl.is_some() {
            CacheControl::Persistent { ttl }
        } else {
            CacheControl::Ephemeral
        }),
        Some("persistent") => Some(CacheControl::Persistent { ttl }),
        _ => None,
    }
}

#[cfg(feature = "anthropic")]
fn cache_control_to_json(cache: &CacheControl) -> Value {
    match cache {
        CacheControl::Ephemeral => json!({ "type": "ephemeral" }),
        CacheControl::Persistent { ttl } => {
            let mut obj = json!({ "type": "ephemeral" });
            if let Some(ttl) = ttl {
                obj["ttl"] = json!(ttl.as_secs());
            }
            obj
        }
    }
}

#[cfg(feature = "anthropic")]
fn merge_anthropic_part_provider_options(part: &mut ContentPart, extra: Map<String, Value>) {
    if extra.is_empty() {
        return;
    }

    let Some(provider_options) = part.provider_options_mut() else {
        return;
    };

    let mut anthropic = provider_options
        .get("anthropic")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    anthropic.extend(extra);
    provider_options.insert("anthropic", Value::Object(anthropic));
}

#[cfg(feature = "anthropic")]
fn apply_anthropic_part_cache_control(part: &mut ContentPart, cache_control: &CacheControl) {
    merge_anthropic_part_provider_options(
        part,
        Map::from_iter([(
            "cacheControl".to_string(),
            cache_control_to_json(cache_control),
        )]),
    );
}

#[cfg(feature = "anthropic")]
fn anthropic_provider_tool_id_from_wire_type(wire_type: &str) -> Option<String> {
    siumai_protocol_anthropic::tool_catalog::anthropic::SERVER_TOOL_SPECS
        .iter()
        .find(|spec| spec.tool_type == wire_type)
        .map(|spec| spec.id.to_string())
}

#[cfg(feature = "anthropic")]
fn default_anthropic_tool_name(wire_type: &str) -> String {
    siumai_protocol_anthropic::tool_catalog::anthropic::SERVER_TOOL_SPECS
        .iter()
        .find(|spec| spec.tool_type == wire_type)
        .map(|spec| spec.tool_name.to_string())
        .unwrap_or_else(|| wire_type.to_string())
}
