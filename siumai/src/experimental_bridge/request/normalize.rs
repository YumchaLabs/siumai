//! Request bridge normalization from protocol JSON into `ChatRequest`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

use serde_json::{Map, Value, json};
use siumai_core::LlmError;
use siumai_core::types::chat::{ImageDetail, MediaSource, ResponseFormat};
use siumai_core::types::{
    CacheControl, ChatMessage, ChatRequest, ContentPart, MessageContent, MessageMetadata,
    MessageRole, Tool, ToolChoice, ToolResultContentPart, ToolResultOutput,
};

#[derive(Debug, Default)]
struct AnthropicMessageParseState {
    part_cache_controls: Map<String, Value>,
    document_citations: Map<String, Value>,
    document_metadata: Map<String, Value>,
    thinking_signatures: BTreeMap<usize, String>,
    redacted_thinking_data: Option<String>,
}

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

#[cfg(feature = "anthropic")]
pub fn bridge_anthropic_messages_json_to_chat_request(
    value: &Value,
) -> Result<ChatRequest, LlmError> {
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

    if request.response_format.is_none()
        && let Some(value) = obj.get("output_format")
        && let Some(format) = parse_json_schema_response_format(value)
    {
        request.response_format = Some(format);
        anthropic_options.insert(
            "structuredOutputMode".to_string(),
            Value::String("outputFormat".to_string()),
        );
    }

    if let Some(thinking) = obj.get("thinking")
        && thinking.is_object()
    {
        anthropic_options.insert("thinking".to_string(), thinking.clone());
    }
    if let Some(output_config) = obj.get("output_config").and_then(Value::as_object)
        && let Some(effort) = output_config.get("effort").and_then(Value::as_str)
    {
        anthropic_options.insert("effort".to_string(), Value::String(effort.to_string()));
    }
    if let Some(mcp_servers) = obj.get("mcp_servers")
        && mcp_servers.is_array()
    {
        anthropic_options.insert("mcpServers".to_string(), mcp_servers.clone());
    }

    if !anthropic_options.is_empty() {
        request
            .provider_options_map
            .insert("anthropic", Value::Object(anthropic_options));
    }

    Ok(request)
}

#[cfg(feature = "openai")]
pub fn bridge_openai_responses_json_to_chat_request(
    value: &Value,
) -> Result<ChatRequest, LlmError> {
    let obj = expect_object(value, "OpenAI Responses request")?;
    let mut request = ChatRequest::new(Vec::new());
    let mut openai_options = Map::new();
    let mut file_id_prefixes = BTreeSet::new();

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
            if let Some(message) = parse_openai_responses_input_item(
                &items[index],
                &tool_registry,
                &mut call_names,
                &mut file_id_prefixes,
            )? {
                request.messages.push(message);
            }
            index += 1;
        }
        request.messages = compact_adjacent_messages(std::mem::take(&mut request.messages));
    }

    if !file_id_prefixes.is_empty() {
        openai_options.insert(
            "fileIdPrefixes".to_string(),
            Value::Array(file_id_prefixes.into_iter().map(Value::String).collect()),
        );
    }
    if !openai_options.is_empty() {
        request
            .provider_options_map
            .insert("openai", Value::Object(openai_options));
    }

    Ok(request)
}

#[cfg(feature = "openai")]
pub fn bridge_openai_chat_completions_json_to_chat_request(
    value: &Value,
) -> Result<ChatRequest, LlmError> {
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
        vec![ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            output: parse_tool_result_output_from_string(&content, false),
            provider_executed: None,
            provider_metadata: None,
        }],
    ))
}

fn parse_openai_chat_tool_call(value: &Value) -> Result<ContentPart, LlmError> {
    let obj = expect_object(value, "OpenAI Chat Completions tool_call")?;
    let tool_call_id = required_string(obj, "id", "OpenAI Chat Completions tool_call")?;
    let tool_type = optional_string(obj, "type").unwrap_or_else(|| "function".to_string());

    if tool_type != "function" {
        return Ok(ContentPart::tool_call(
            tool_call_id,
            tool_type.clone(),
            collect_remaining_object_fields(obj, &["id", "type", "function"]),
            None,
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

    Ok(ContentPart::tool_call(tool_call_id, name, arguments, None))
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
        "text" => Ok(ContentPart::text(
            optional_string(obj, "text").unwrap_or_default(),
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
        Some(Value::String(url)) => Ok(ContentPart::Image {
            source: MediaSource::Url { url: url.clone() },
            detail: None,
            provider_metadata: None,
        }),
        Some(Value::Object(image)) => {
            let url = required_string(image, "url", "OpenAI image_url content part")?;
            let detail = image
                .get("detail")
                .and_then(Value::as_str)
                .map(ImageDetail::from);
            Ok(ContentPart::Image {
                source: MediaSource::Url { url },
                detail,
                provider_metadata: None,
            })
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

    Ok(ContentPart::Audio {
        source: MediaSource::Base64 { data },
        media_type: Some(media_type),
        provider_metadata: None,
    })
}

fn parse_openai_file_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    let file = obj
        .get("file")
        .ok_or_else(|| LlmError::ParseError("OpenAI file part is missing file".to_string()))
        .and_then(|value| expect_object(value, "OpenAI file part.file"))?;

    if let Some(file_id) = optional_string(file, "file_id") {
        return Ok(ContentPart::File {
            source: MediaSource::Base64 { data: file_id },
            media_type: "application/pdf".to_string(),
            filename: optional_string(file, "filename"),
            provider_metadata: None,
        });
    }
    if let Some(file_data) = optional_string(file, "file_data") {
        return Ok(ContentPart::File {
            source: MediaSource::Base64 {
                data: strip_data_url_prefix(&file_data),
            },
            media_type: "application/pdf".to_string(),
            filename: optional_string(file, "filename"),
            provider_metadata: None,
        });
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

fn parse_openai_function_tool(function_obj: &Map<String, Value>) -> Tool {
    let name = optional_string(function_obj, "name").unwrap_or_default();
    let description = optional_string(function_obj, "description").unwrap_or_default();
    let parameters = function_obj
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut tool = Tool::function(name, description, parameters);

    if let Tool::Function { function } = &mut tool {
        function.strict = function_obj.get("strict").and_then(Value::as_bool);
    }

    tool
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
        let parameters = obj
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        let mut tool = Tool::function(name, description, parameters);
        if let Tool::Function { function } = &mut tool {
            function.strict = obj.get("strict").and_then(Value::as_bool);
        }
        return Ok(tool);
    }

    Ok(parse_openai_provider_defined_tool(
        kind.as_str(),
        obj,
        Some("type"),
    ))
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
    file_id_prefixes: &mut BTreeSet<String>,
) -> Result<Option<ChatMessage>, LlmError> {
    let obj = expect_object(value, "OpenAI Responses input item")?;

    if obj.contains_key("role") || obj.get("type").and_then(Value::as_str) == Some("message") {
        return Ok(Some(parse_openai_responses_message_item(
            obj,
            file_id_prefixes,
        )?));
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

fn parse_openai_responses_message_item(
    obj: &Map<String, Value>,
    file_id_prefixes: &mut BTreeSet<String>,
) -> Result<ChatMessage, LlmError> {
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
        Some(Value::Array(parts)) => {
            parse_openai_responses_message_content(parts, &role, file_id_prefixes)?
        }
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
    file_id_prefixes: &mut BTreeSet<String>,
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
                out.push(parse_openai_responses_image_part(obj, file_id_prefixes));
            }
            "input_file" => {
                out.push(parse_openai_responses_file_part(obj, file_id_prefixes)?);
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
                out.push(ContentPart::tool_call(
                    tool_call_id,
                    tool_name,
                    arguments,
                    None,
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
                out.push(ContentPart::tool_call(
                    tool_call_id,
                    tool_name,
                    arguments,
                    None,
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

fn parse_openai_responses_image_part(
    obj: &Map<String, Value>,
    file_id_prefixes: &mut BTreeSet<String>,
) -> ContentPart {
    if let Some(file_id) = optional_string(obj, "file_id") {
        record_openai_file_id_prefix(file_id_prefixes, &file_id);
        return ContentPart::File {
            source: MediaSource::Base64 { data: file_id },
            media_type: "image/*".to_string(),
            filename: None,
            provider_metadata: None,
        };
    }

    ContentPart::Image {
        source: MediaSource::Url {
            url: optional_string(obj, "image_url").unwrap_or_default(),
        },
        detail: obj
            .get("detail")
            .and_then(Value::as_str)
            .map(ImageDetail::from),
        provider_metadata: None,
    }
}

fn parse_openai_responses_file_part(
    obj: &Map<String, Value>,
    file_id_prefixes: &mut BTreeSet<String>,
) -> Result<ContentPart, LlmError> {
    if let Some(file_id) = optional_string(obj, "file_id") {
        record_openai_file_id_prefix(file_id_prefixes, &file_id);
        return Ok(ContentPart::File {
            source: MediaSource::Base64 { data: file_id },
            media_type: "application/pdf".to_string(),
            filename: optional_string(obj, "filename"),
            provider_metadata: None,
        });
    }
    if let Some(file_url) = optional_string(obj, "file_url") {
        return Ok(ContentPart::File {
            source: MediaSource::Url { url: file_url },
            media_type: infer_document_media_type(None, None),
            filename: optional_string(obj, "filename"),
            provider_metadata: None,
        });
    }
    if let Some(file_data) = optional_string(obj, "file_data") {
        return Ok(ContentPart::File {
            source: MediaSource::Base64 {
                data: strip_data_url_prefix(&file_data),
            },
            media_type: "application/pdf".to_string(),
            filename: optional_string(obj, "filename"),
            provider_metadata: None,
        });
    }

    Err(LlmError::ParseError(
        "OpenAI Responses input_file part requires file_id, file_url, or file_data".to_string(),
    ))
}

fn parse_openai_responses_reasoning_item(
    obj: &Map<String, Value>,
) -> Result<ChatMessage, LlmError> {
    let item_id = required_string(obj, "id", "OpenAI Responses reasoning item")?;
    let text = collect_reasoning_summary(obj.get("summary")).unwrap_or_default();
    let mut provider_metadata = HashMap::new();
    let mut openai_meta = Map::new();
    openai_meta.insert("itemId".to_string(), Value::String(item_id));
    if let Some(encrypted) = obj.get("encrypted_content")
        && !encrypted.is_null()
    {
        openai_meta.insert("reasoningEncryptedContent".to_string(), encrypted.clone());
    }
    provider_metadata.insert("openai".to_string(), Value::Object(openai_meta));

    Ok(message_from_parts(
        MessageRole::Assistant,
        vec![ContentPart::Reasoning {
            text,
            provider_metadata: Some(provider_metadata),
        }],
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
    let provider_metadata = openai_item_id_metadata(obj);

    Ok(message_from_parts(
        MessageRole::Assistant,
        vec![ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            provider_executed: None,
            provider_metadata,
        }],
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
    let provider_metadata = openai_item_id_metadata(obj);

    Ok(message_from_parts(
        MessageRole::Assistant,
        vec![ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments: Value::Object(arguments),
            provider_executed: None,
            provider_metadata,
        }],
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
        vec![ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            output,
            provider_executed: None,
            provider_metadata: None,
        }],
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
        vec![ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            output,
            provider_executed: None,
            provider_metadata: None,
        }],
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
        vec![ContentPart::tool_approval_response(approval_id, approved)],
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
                    parts.push(ToolResultContentPart::File {
                        source: MediaSource::Base64 { data: file_id },
                        media_type: "image/*".to_string(),
                        filename: None,
                    });
                } else {
                    parts.push(ToolResultContentPart::Image {
                        source: MediaSource::Url {
                            url: optional_string(obj, "image_url").unwrap_or_default(),
                        },
                        detail: obj
                            .get("detail")
                            .and_then(Value::as_str)
                            .map(ImageDetail::from),
                    });
                }
            }
            "input_file" => {
                parts.push(match optional_string(obj, "file_url") {
                    Some(url) => ToolResultContentPart::File {
                        source: MediaSource::Url { url },
                        media_type: infer_document_media_type(None, None),
                        filename: optional_string(obj, "filename"),
                    },
                    None => ToolResultContentPart::File {
                        source: MediaSource::Base64 {
                            data: optional_string(obj, "file_id")
                                .or_else(|| {
                                    optional_string(obj, "file_data")
                                        .map(|value| strip_data_url_prefix(&value))
                                })
                                .unwrap_or_default(),
                        },
                        media_type: "application/pdf".to_string(),
                        filename: optional_string(obj, "filename"),
                    },
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

fn parse_anthropic_message(value: &Value) -> Result<ChatMessage, LlmError> {
    let obj = expect_object(value, "Anthropic message")?;
    let role = required_string(obj, "role", "Anthropic message")?;
    let default_content = Value::String(String::new());
    let content = obj.get("content").unwrap_or(&default_content);

    let (parts, parse_state) = parse_anthropic_message_parts(content)?;
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

    let mut message = message_from_parts(normalized_role, parts);
    if !parse_state.part_cache_controls.is_empty() {
        message.metadata.custom.insert(
            "anthropic_content_cache_controls".to_string(),
            Value::Object(parse_state.part_cache_controls),
        );
    }
    if !parse_state.document_citations.is_empty() {
        message.metadata.custom.insert(
            "anthropic_document_citations".to_string(),
            Value::Object(parse_state.document_citations),
        );
    }
    if !parse_state.document_metadata.is_empty() {
        message.metadata.custom.insert(
            "anthropic_document_metadata".to_string(),
            Value::Object(parse_state.document_metadata),
        );
    }
    if parse_state.thinking_signatures.len() == 1 {
        if let Some(value) = parse_state.thinking_signatures.values().next() {
            message.metadata.custom.insert(
                "anthropic_thinking_signature".to_string(),
                Value::String(value.clone()),
            );
        }
    } else if !parse_state.thinking_signatures.is_empty() {
        message.metadata.custom.insert(
            "anthropic_thinking_signatures".to_string(),
            Value::Object(
                parse_state
                    .thinking_signatures
                    .into_iter()
                    .map(|(index, signature)| (index.to_string(), Value::String(signature)))
                    .collect(),
            ),
        );
    }
    if let Some(redacted) = parse_state.redacted_thinking_data {
        message.metadata.custom.insert(
            "anthropic_redacted_thinking_data".to_string(),
            Value::String(redacted),
        );
    }

    Ok(message)
}

fn parse_anthropic_message_parts(
    value: &Value,
) -> Result<(Vec<ContentPart>, AnthropicMessageParseState), LlmError> {
    if let Some(text) = value.as_str() {
        return Ok((
            parse_text_like_content_parts(text),
            AnthropicMessageParseState::default(),
        ));
    }

    let mut parts = Vec::new();
    let mut state = AnthropicMessageParseState::default();

    for value in expect_array(value, "Anthropic message.content")? {
        let obj = expect_object(value, "Anthropic message content block")?;
        let kind = required_string(obj, "type", "Anthropic message content block")?;
        let index = parts.len();

        if let Some(cache_control) = obj.get("cache_control").and_then(Value::as_object) {
            state
                .part_cache_controls
                .insert(index.to_string(), Value::Object(cache_control.clone()));
        }

        match kind.as_str() {
            "text" => {
                parts.extend(parse_text_like_content_parts(
                    &optional_string(obj, "text").unwrap_or_default(),
                ));
            }
            "image" => parts.push(parse_anthropic_image_part(obj)?),
            "document" => {
                let part = parse_anthropic_document_part(obj, index, &mut state)?;
                parts.push(part);
            }
            "tool_use" => {
                let tool_call_id = required_string(obj, "id", "Anthropic tool_use block")?;
                let tool_name = required_string(obj, "name", "Anthropic tool_use block")?;
                let arguments = obj
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new()));
                parts.push(ContentPart::tool_call(
                    tool_call_id,
                    tool_name,
                    arguments,
                    None,
                ));
            }
            "tool_result" => {
                parts.push(parse_anthropic_tool_result_part(obj)?);
            }
            "thinking" => {
                let text = optional_string(obj, "thinking").unwrap_or_default();
                parts.push(ContentPart::reasoning(text));
                if let Some(signature) = optional_string(obj, "signature")
                    && !signature.is_empty()
                {
                    state.thinking_signatures.insert(index, signature);
                }
            }
            "redacted_thinking" => {
                parts.push(ContentPart::reasoning(String::new()));
                if state.redacted_thinking_data.is_none() {
                    state.redacted_thinking_data = optional_string(obj, "data");
                }
            }
            other => {
                return Err(LlmError::ParseError(format!(
                    "unsupported Anthropic content block `{other}`"
                )));
            }
        }
    }

    Ok((parts, state))
}

fn parse_anthropic_image_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    let source = obj
        .get("source")
        .ok_or_else(|| LlmError::ParseError("Anthropic image block is missing source".to_string()))
        .and_then(|value| expect_object(value, "Anthropic image block.source"))?;
    let source_type = required_string(source, "type", "Anthropic image block.source")?;

    let media_source = match source_type.as_str() {
        "url" => MediaSource::Url {
            url: required_string(source, "url", "Anthropic image block.source")?,
        },
        "base64" => MediaSource::Base64 {
            data: required_string(source, "data", "Anthropic image block.source")?,
        },
        other => {
            return Err(LlmError::ParseError(format!(
                "unsupported Anthropic image source `{other}`"
            )));
        }
    };

    Ok(ContentPart::Image {
        source: media_source,
        detail: None,
        provider_metadata: None,
    })
}

fn parse_anthropic_document_part(
    obj: &Map<String, Value>,
    index: usize,
    state: &mut AnthropicMessageParseState,
) -> Result<ContentPart, LlmError> {
    if obj
        .get("citations")
        .and_then(Value::as_object)
        .and_then(|citations| citations.get("enabled"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        state
            .document_citations
            .insert(index.to_string(), json!({ "enabled": true }));
    }

    if obj.get("title").is_some() || obj.get("context").is_some() {
        let mut meta = Map::new();
        if let Some(title) = optional_string(obj, "title")
            && !title.is_empty()
        {
            meta.insert("title".to_string(), Value::String(title));
        }
        if let Some(context) = optional_string(obj, "context")
            && !context.is_empty()
        {
            meta.insert("context".to_string(), Value::String(context));
        }
        if !meta.is_empty() {
            state
                .document_metadata
                .insert(index.to_string(), Value::Object(meta));
        }
    }

    let source = obj
        .get("source")
        .ok_or_else(|| {
            LlmError::ParseError("Anthropic document block is missing source".to_string())
        })
        .and_then(|value| expect_object(value, "Anthropic document block.source"))?;
    let source_type = required_string(source, "type", "Anthropic document block.source")?;
    let title = optional_string(obj, "title");

    let (media_source, media_type) = match source_type.as_str() {
        "url" => {
            let url = required_string(source, "url", "Anthropic document block.source")?;
            let media_type = infer_document_media_type(title.as_deref(), Some(url.as_str()));
            (MediaSource::Url { url }, media_type)
        }
        "base64" => (
            MediaSource::Base64 {
                data: required_string(source, "data", "Anthropic document block.source")?,
            },
            optional_string(source, "media_type").unwrap_or_else(|| "application/pdf".to_string()),
        ),
        "text" => (
            MediaSource::Binary {
                data: optional_string(source, "data")
                    .unwrap_or_default()
                    .into_bytes(),
            },
            optional_string(source, "media_type").unwrap_or_else(|| "text/plain".to_string()),
        ),
        other => {
            return Err(LlmError::ParseError(format!(
                "unsupported Anthropic document source `{other}`"
            )));
        }
    };

    Ok(ContentPart::File {
        source: media_source,
        media_type,
        filename: title,
        provider_metadata: None,
    })
}

fn parse_anthropic_tool_result_part(obj: &Map<String, Value>) -> Result<ContentPart, LlmError> {
    let tool_call_id = required_string(obj, "tool_use_id", "Anthropic tool_result block")?;
    let default_output = Value::String(String::new());
    let output_value = obj.get("content").unwrap_or(&default_output);
    let is_error = obj
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let output = parse_anthropic_tool_result_output(output_value, is_error)?;

    Ok(ContentPart::ToolResult {
        tool_call_id,
        tool_name: String::new(),
        output,
        provider_executed: None,
        provider_metadata: None,
    })
}

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
                out.push(ToolResultContentPart::Image {
                    source: media_source,
                    detail: None,
                });
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

fn parse_anthropic_tools(value: &Value) -> Result<Vec<Tool>, LlmError> {
    let mut tools = Vec::new();
    for value in expect_array(value, "Anthropic request.tools")? {
        tools.push(parse_anthropic_tool(value)?);
    }
    Ok(tools)
}

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
            provider_tool.args = collect_remaining_object_fields(obj, &["type", "name"]);
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

fn strip_developer_prefix(text: &str) -> (MessageRole, String) {
    let prefix = "Developer instructions: ";
    if let Some(stripped) = text.strip_prefix(prefix) {
        (MessageRole::Developer, stripped.to_string())
    } else {
        (MessageRole::System, text.to_string())
    }
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
    let owner = obj
        .get("json_schema")
        .and_then(Value::as_object)
        .unwrap_or(obj);

    if obj.get("type").and_then(Value::as_str) != Some("json_schema")
        && !owner.contains_key("schema")
    {
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
        vec![ContentPart::reasoning(reasoning)]
    } else {
        vec![ContentPart::text(text)]
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
                provider_metadata: None,
            } if !matches!(role, MessageRole::Tool) => MessageContent::Text(text),
            part => MessageContent::MultiModal(vec![part]),
        }
    } else {
        MessageContent::MultiModal(parts)
    };

    ChatMessage {
        role,
        content,
        metadata: Default::default(),
    }
}

fn text_message(role: MessageRole, text: impl Into<String>) -> ChatMessage {
    ChatMessage {
        role,
        content: MessageContent::Text(text.into()),
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
        MessageContent::Json(value) => vec![ContentPart::text(
            serde_json::to_string(&value).unwrap_or_default(),
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

fn openai_provider_tool_id_from_wire_type(wire_type: &str) -> String {
    match wire_type {
        "computer_use_preview" => siumai_core::tools::openai::COMPUTER_USE_ID.to_string(),
        "web_search" => siumai_core::tools::openai::WEB_SEARCH_ID.to_string(),
        "web_search_preview" => siumai_core::tools::openai::WEB_SEARCH_PREVIEW_ID.to_string(),
        "file_search" => siumai_core::tools::openai::FILE_SEARCH_ID.to_string(),
        "code_interpreter" => siumai_core::tools::openai::CODE_INTERPRETER_ID.to_string(),
        "image_generation" => siumai_core::tools::openai::IMAGE_GENERATION_ID.to_string(),
        "local_shell" => siumai_core::tools::openai::LOCAL_SHELL_ID.to_string(),
        "shell" => siumai_core::tools::openai::SHELL_ID.to_string(),
        "mcp" => siumai_core::tools::openai::MCP_ID.to_string(),
        "apply_patch" => siumai_core::tools::openai::APPLY_PATCH_ID.to_string(),
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

fn anthropic_provider_tool_id_from_wire_type(wire_type: &str) -> Option<String> {
    siumai_core::tools::anthropic::SERVER_TOOL_SPECS
        .iter()
        .find(|spec| spec.tool_type == wire_type)
        .map(|spec| spec.id.to_string())
}

fn default_anthropic_tool_name(wire_type: &str) -> String {
    siumai_core::tools::anthropic::SERVER_TOOL_SPECS
        .iter()
        .find(|spec| spec.tool_type == wire_type)
        .map(|spec| spec.tool_name.to_string())
        .unwrap_or_else(|| wire_type.to_string())
}

fn default_provider_defined_tool(provider_id: &str) -> Option<Tool> {
    crate::tools::provider_defined_tool(provider_id)
}

fn record_openai_file_id_prefix(prefixes: &mut BTreeSet<String>, value: &str) {
    for prefix in ["file_", "file-"] {
        if value.starts_with(prefix) {
            prefixes.insert(prefix.to_string());
        }
    }
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

fn openai_item_id_metadata(obj: &Map<String, Value>) -> Option<HashMap<String, Value>> {
    let item_id = obj.get("id")?.as_str()?.to_string();
    let mut provider_metadata = HashMap::new();
    provider_metadata.insert(
        "openai".to_string(),
        json!({
            "itemId": item_id,
        }),
    );
    Some(provider_metadata)
}
