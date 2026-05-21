//! Gemini GenerateContent request bridge adapter.
//!
//! This module owns the Gemini-specific JSON -> `ChatRequest` policy. The
//! bridge crate keeps bridge reports, loss policy, hooks, and public wrappers,
//! while protocol-specific wire parsing lives with the Gemini protocol types.

use std::collections::{HashMap, VecDeque};

use serde_json::{Map, Value, json};
use siumai_core::error::LlmError;
use siumai_core::types::chat::{
    FilePartSource, ImageDetail, MediaSource, MessageContent, ResponseFormat,
};
use siumai_core::types::{
    ChatMessage, ChatRequest, ContentPart, MessageMetadata, MessageRole, ProviderOptionsMap, Tool,
    ToolChoice, ToolResultOutput,
};
use uuid::Uuid;

use super::types::{
    CodeExecutionOutcome as GeminiCodeExecutionOutcome, Content as GeminiContent,
    FunctionCallingMode as GeminiFunctionCallingMode, GeminiTool as GeminiRequestTool,
    GenerateContentRequest as GeminiGenerateContentRequest,
    GenerationConfig as GeminiGenerationConfig, Part as GeminiPart, ToolConfig as GeminiToolConfig,
};

/// Parse a Gemini GenerateContent JSON request into Siumai's normalized chat request.
pub fn parse_json_to_chat_request(value: &Value) -> Result<ChatRequest, LlmError> {
    let obj = expect_object(value, "Gemini GenerateContent request")?;
    let typed: GeminiGenerateContentRequest =
        serde_json::from_value(value.clone()).map_err(|err| {
            LlmError::ParseError(format!("invalid Gemini GenerateContent request: {err}"))
        })?;

    let mut request = ChatRequest::new(Vec::new());
    let mut google_options = Map::new();
    let mut pending_tool_call_ids = HashMap::new();

    request.common_params.model = required_string(obj, "model", "Gemini GenerateContent request")?;

    if let Some(system_instruction) = typed.system_instruction.as_ref()
        && let Some(message) = parse_gemini_system_instruction(system_instruction)?
    {
        request.messages.push(message);
    }

    for content in &typed.contents {
        if let Some(message) = parse_gemini_content(content, &mut pending_tool_call_ids)? {
            request.messages.push(message);
        }
    }
    request.messages = compact_adjacent_messages(std::mem::take(&mut request.messages));

    if let Some(tools) = typed.tools.as_ref() {
        let parsed = parse_gemini_tools(tools)?;
        if !parsed.is_empty() {
            request.tools = Some(parsed);
        }
    }

    if let Some(generation_config) = typed.generation_config.as_ref() {
        parse_gemini_generation_config(
            generation_config,
            obj.get("generationConfig").and_then(Value::as_object),
            &mut request,
            &mut google_options,
        )?;
    }

    if let Some(tool_config) = typed.tool_config.as_ref() {
        let parsed = parse_gemini_tool_config(
            tool_config,
            obj.get("toolConfig").and_then(Value::as_object),
            &mut request,
            &mut google_options,
        )?;
        if let Some(allowed_names) = parsed.allowed_function_names
            && let Some(tools) = &mut request.tools
        {
            tools.retain(|tool| match tool {
                Tool::Function { function } => {
                    allowed_names.iter().any(|name| name == &function.name)
                }
                _ => true,
            });
            if tools.is_empty() {
                request.tools = None;
            }
        }
    }

    if let Some(cached_content) = typed.cached_content.as_ref() {
        google_options.insert(
            "cachedContent".to_string(),
            Value::String(cached_content.clone()),
        );
    }
    if let Some(safety_settings) = obj.get("safetySettings").filter(|value| value.is_array()) {
        google_options.insert("safetySettings".to_string(), safety_settings.clone());
    } else if let Some(safety_settings) = typed.safety_settings.as_ref() {
        google_options.insert(
            "safetySettings".to_string(),
            serde_json::to_value(safety_settings).map_err(|err| {
                LlmError::ParseError(format!("failed to serialize Gemini safetySettings: {err}"))
            })?,
        );
    }
    if let Some(labels) = obj.get("labels").filter(|value| value.is_object()) {
        google_options.insert("labels".to_string(), labels.clone());
    }

    if !google_options.is_empty() {
        request
            .provider_options_map
            .insert("google", Value::Object(google_options));
    }

    Ok(request)
}

#[derive(Debug, Default)]
struct GeminiToolConfigParse {
    allowed_function_names: Option<Vec<String>>,
}

fn parse_gemini_system_instruction(
    content: &GeminiContent,
) -> Result<Option<ChatMessage>, LlmError> {
    let mut pending_tool_call_ids = HashMap::new();
    let mut parts = Vec::new();
    for part in &content.parts {
        if let Some(parsed) = parse_gemini_part(part, &mut pending_tool_call_ids)? {
            parts.push(parsed);
        }
    }

    if parts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(message_from_parts(MessageRole::System, parts)))
    }
}

fn parse_gemini_content(
    content: &GeminiContent,
    pending_tool_call_ids: &mut HashMap<String, VecDeque<String>>,
) -> Result<Option<ChatMessage>, LlmError> {
    let mut parts = Vec::new();
    for part in &content.parts {
        if let Some(parsed) = parse_gemini_part(part, pending_tool_call_ids)? {
            parts.push(parsed);
        }
    }

    if parts.is_empty() {
        return Ok(None);
    }

    let role = parse_gemini_message_role(content.role.as_deref(), &parts)?;
    Ok(Some(message_from_parts(role, parts)))
}

fn parse_gemini_message_role(
    role: Option<&str>,
    parts: &[ContentPart],
) -> Result<MessageRole, LlmError> {
    let has_tool_calls = parts
        .iter()
        .any(|part| matches!(part, ContentPart::ToolCall { .. }));
    let has_reasoning = parts
        .iter()
        .any(|part| matches!(part, ContentPart::Reasoning { .. }));
    let only_tool_results = parts
        .iter()
        .all(|part| matches!(part, ContentPart::ToolResult { .. }));

    match role {
        Some("model") => Ok(MessageRole::Assistant),
        Some("user") => {
            if only_tool_results {
                Ok(MessageRole::Tool)
            } else {
                Ok(MessageRole::User)
            }
        }
        None => {
            if has_tool_calls || has_reasoning {
                Ok(MessageRole::Assistant)
            } else if only_tool_results {
                Ok(MessageRole::Tool)
            } else {
                Ok(MessageRole::User)
            }
        }
        Some(other) => Err(LlmError::ParseError(format!(
            "unsupported Gemini content role `{other}`"
        ))),
    }
}

fn parse_gemini_part(
    part: &GeminiPart,
    pending_tool_call_ids: &mut HashMap<String, VecDeque<String>>,
) -> Result<Option<ContentPart>, LlmError> {
    Ok(match part {
        GeminiPart::Text {
            text,
            thought,
            thought_signature,
        } => {
            let provider_options =
                gemini_thought_signature_provider_options(thought_signature.as_deref());
            if thought.unwrap_or(false) {
                Some(request_reasoning_part(text.clone(), provider_options))
            } else {
                Some(request_text_part(text.clone(), provider_options))
            }
        }
        GeminiPart::InlineData {
            inline_data,
            thought_signature,
            ..
        } => Some(parse_gemini_inline_data_part(
            &inline_data.mime_type,
            &inline_data.data,
            thought_signature.as_deref(),
        )),
        GeminiPart::FileData {
            file_data,
            thought_signature,
            ..
        } => Some(parse_gemini_file_data_part(
            &file_data.file_uri,
            file_data.mime_type.as_deref(),
            thought_signature.as_deref(),
        )),
        GeminiPart::FunctionCall {
            function_call,
            thought_signature,
        } => {
            let tool_call_id =
                push_pending_gemini_tool_call_id(pending_tool_call_ids, &function_call.name);
            Some(request_tool_call_part(
                tool_call_id,
                function_call.name.clone(),
                function_call
                    .args
                    .clone()
                    .unwrap_or_else(|| Value::Object(Map::new())),
                None,
                None,
                gemini_thought_signature_provider_options(thought_signature.as_deref()),
            ))
        }
        GeminiPart::FunctionResponse {
            function_response,
            thought_signature,
        } => {
            let tool_call_id =
                take_pending_gemini_tool_call_id(pending_tool_call_ids, &function_response.name);
            Some(request_tool_result_part(
                tool_call_id,
                function_response.name.clone(),
                parse_gemini_function_response_output(&function_response.response),
                None,
                None,
                gemini_thought_signature_provider_options(thought_signature.as_deref()),
            ))
        }
        GeminiPart::ExecutableCode {
            executable_code,
            thought_signature,
        } => {
            let tool_call_id =
                push_pending_gemini_tool_call_id(pending_tool_call_ids, "code_execution");
            let language = match executable_code.language {
                super::types::CodeLanguage::Python => "PYTHON",
                super::types::CodeLanguage::Unspecified => "LANGUAGE_UNSPECIFIED",
            };
            Some(request_tool_call_part(
                tool_call_id,
                "code_execution",
                json!({
                    "language": language,
                    "code": executable_code.code.clone(),
                }),
                Some(true),
                Some(true),
                gemini_thought_signature_provider_options(thought_signature.as_deref()),
            ))
        }
        GeminiPart::CodeExecutionResult {
            code_execution_result,
            thought_signature,
        } => {
            let tool_call_id =
                take_pending_gemini_tool_call_id(pending_tool_call_ids, "code_execution");
            let outcome = match code_execution_result.outcome {
                GeminiCodeExecutionOutcome::Ok => "OUTCOME_OK",
                GeminiCodeExecutionOutcome::Failed => "OUTCOME_FAILED",
                GeminiCodeExecutionOutcome::DeadlineExceeded => "OUTCOME_DEADLINE_EXCEEDED",
                GeminiCodeExecutionOutcome::Unspecified => "OUTCOME_UNSPECIFIED",
            };
            Some(request_tool_result_part(
                tool_call_id,
                "code_execution",
                ToolResultOutput::json(json!({
                    "outcome": outcome,
                    "output": code_execution_result.output.clone(),
                })),
                Some(true),
                Some(true),
                gemini_thought_signature_provider_options(thought_signature.as_deref()),
            ))
        }
    })
}

fn parse_gemini_inline_data_part(
    mime_type: &str,
    data: &str,
    thought_signature: Option<&str>,
) -> ContentPart {
    let provider_options = gemini_thought_signature_provider_options(thought_signature);
    if mime_type.starts_with("image/") {
        request_image_part(
            FilePartSource::base64(data),
            Some(mime_type.to_string()),
            None,
            provider_options,
        )
    } else if mime_type.starts_with("audio/") {
        request_audio_part(
            MediaSource::Base64 {
                data: data.to_string(),
            },
            Some(mime_type.to_string()),
            provider_options,
        )
    } else {
        request_file_part(
            FilePartSource::base64(data),
            mime_type.to_string(),
            None,
            provider_options,
        )
    }
}

fn parse_gemini_file_data_part(
    file_uri: &str,
    mime_type: Option<&str>,
    thought_signature: Option<&str>,
) -> ContentPart {
    let provider_options = gemini_thought_signature_provider_options(thought_signature);
    match mime_type {
        Some(mime) if mime.starts_with("image/") => request_image_part(
            FilePartSource::url(file_uri),
            Some(mime.to_string()),
            None,
            provider_options,
        ),
        Some(mime) if mime.starts_with("audio/") => request_audio_part(
            MediaSource::Url {
                url: file_uri.to_string(),
            },
            Some(mime.to_string()),
            provider_options,
        ),
        Some(mime) => request_file_part(
            FilePartSource::url(file_uri),
            mime.to_string(),
            None,
            provider_options,
        ),
        None => request_file_part(
            FilePartSource::url(file_uri),
            "application/octet-stream",
            None,
            provider_options,
        ),
    }
}

fn parse_gemini_function_response_output(value: &Value) -> ToolResultOutput {
    let payload = value
        .as_object()
        .and_then(|obj| obj.get("content"))
        .cloned()
        .unwrap_or_else(|| value.clone());

    match payload {
        Value::String(text) => ToolResultOutput::text(text),
        other => ToolResultOutput::json(other),
    }
}

fn gemini_thought_signature_provider_options(
    thought_signature: Option<&str>,
) -> ProviderOptionsMap {
    let mut provider_options = ProviderOptionsMap::default();
    let Some(thought_signature) = thought_signature
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return provider_options;
    };
    provider_options.insert("google", json!({ "thoughtSignature": thought_signature }));
    provider_options
}

fn push_pending_gemini_tool_call_id(
    pending_tool_call_ids: &mut HashMap<String, VecDeque<String>>,
    tool_name: &str,
) -> String {
    let tool_call_id = format!("call_{}", Uuid::new_v4().simple());
    pending_tool_call_ids
        .entry(tool_name.to_string())
        .or_default()
        .push_back(tool_call_id.clone());
    tool_call_id
}

fn take_pending_gemini_tool_call_id(
    pending_tool_call_ids: &mut HashMap<String, VecDeque<String>>,
    tool_name: &str,
) -> String {
    let (tool_call_id, remove_entry) = match pending_tool_call_ids.get_mut(tool_name) {
        Some(ids) => {
            let value = ids.pop_front();
            (value, ids.is_empty())
        }
        None => (None, false),
    };

    if remove_entry {
        pending_tool_call_ids.remove(tool_name);
    }

    tool_call_id.unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()))
}

fn parse_gemini_tools(tools: &[GeminiRequestTool]) -> Result<Vec<Tool>, LlmError> {
    let mut parsed = Vec::new();

    for tool in tools {
        match tool {
            GeminiRequestTool::FunctionDeclarations {
                function_declarations,
            } => {
                for function in function_declarations {
                    parsed.push(Tool::function(
                        function.name.clone(),
                        function.description.clone(),
                        function
                            .parameters
                            .clone()
                            .unwrap_or_else(|| Value::Object(Map::new())),
                    ));
                }
            }
            GeminiRequestTool::CodeExecution { .. } => {
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.code_execution",
                    "code_execution",
                    Value::Object(Map::new()),
                ))
            }
            GeminiRequestTool::GoogleSearch { .. } => {
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.google_search",
                    "google_search",
                    Value::Object(Map::new()),
                ))
            }
            GeminiRequestTool::GoogleMaps { .. } => {
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.google_maps",
                    "google_maps",
                    Value::Object(Map::new()),
                ))
            }
            GeminiRequestTool::GoogleSearchRetrieval {
                google_search_retrieval,
            } => {
                let mut args = Map::new();
                if let Some(config) = &google_search_retrieval.dynamic_retrieval_config {
                    let mode = match config.mode {
                        super::types::DynamicRetrievalMode::Dynamic => "MODE_DYNAMIC",
                        super::types::DynamicRetrievalMode::Unspecified => "MODE_UNSPECIFIED",
                    };
                    args.insert("mode".to_string(), Value::String(mode.to_string()));
                    if let Some(threshold) = &config.dynamic_threshold {
                        args.insert(
                            "dynamicThreshold".to_string(),
                            Value::Number(threshold.clone()),
                        );
                    }
                }
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.google_search",
                    "google_search",
                    Value::Object(args),
                ));
            }
            GeminiRequestTool::UrlContext { .. } => {
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.url_context",
                    "url_context",
                    Value::Object(Map::new()),
                ))
            }
            GeminiRequestTool::FileSearch { file_search } => {
                let mut args = Map::new();
                if let Some(names) = &file_search.file_search_store_names {
                    args.insert("fileSearchStoreNames".to_string(), json!(names));
                }
                if let Some(top_k) = file_search.top_k {
                    args.insert("topK".to_string(), json!(top_k));
                }
                if let Some(filter) = &file_search.metadata_filter {
                    args.insert("metadataFilter".to_string(), json!(filter));
                }
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.file_search",
                    "file_search",
                    Value::Object(args),
                ));
            }
            GeminiRequestTool::EnterpriseWebSearch { .. } => {
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.enterprise_web_search",
                    "enterprise_web_search",
                    Value::Object(Map::new()),
                ))
            }
            GeminiRequestTool::Retrieval { retrieval } => {
                let mut args = Map::new();
                args.insert(
                    "ragCorpus".to_string(),
                    Value::String(retrieval.vertex_rag_store.rag_resources.rag_corpus.clone()),
                );
                if let Some(top_k) = retrieval.vertex_rag_store.similarity_top_k {
                    args.insert("topK".to_string(), json!(top_k));
                }
                parsed.push(parse_gemini_provider_defined_tool(
                    "google.vertex_rag_store",
                    "vertex_rag_store",
                    Value::Object(args),
                ));
            }
        }
    }

    Ok(parsed)
}

fn parse_gemini_provider_defined_tool(provider_id: &str, default_name: &str, args: Value) -> Tool {
    let mut tool =
        crate::tool_catalog::google::provider_defined_tool(provider_id).unwrap_or_else(|| {
            Tool::provider_defined(provider_id.to_string(), default_name.to_string())
        });

    if let Tool::ProviderDefined(provider_tool) = &mut tool {
        provider_tool.name = default_name.to_string();
        provider_tool.args = args;
    }

    tool
}

fn parse_gemini_tool_config(
    tool_config: &GeminiToolConfig,
    raw_tool_config: Option<&Map<String, Value>>,
    request: &mut ChatRequest,
    google_options: &mut Map<String, Value>,
) -> Result<GeminiToolConfigParse, LlmError> {
    let mut parsed = GeminiToolConfigParse::default();

    let raw_function_calling_config = raw_tool_config
        .and_then(|obj| obj.get("functionCallingConfig"))
        .and_then(Value::as_object);

    let allowed_function_names = raw_function_calling_config
        .and_then(|obj| obj.get("allowedFunctionNames"))
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|names| !names.is_empty())
        .or_else(|| {
            tool_config
                .function_calling_config
                .as_ref()
                .and_then(|cfg| cfg.allowed_function_names.clone())
                .filter(|names| !names.is_empty())
        });

    let mode = raw_function_calling_config
        .and_then(|obj| obj.get("mode"))
        .and_then(Value::as_str)
        .map(|value| match value {
            "NONE" => GeminiFunctionCallingMode::None,
            "ANY" => GeminiFunctionCallingMode::Any,
            "AUTO" => GeminiFunctionCallingMode::Auto,
            _ => GeminiFunctionCallingMode::Unspecified,
        })
        .or_else(|| {
            tool_config
                .function_calling_config
                .as_ref()
                .and_then(|cfg| cfg.mode.clone())
        });

    if raw_function_calling_config.is_some() || tool_config.function_calling_config.is_some() {
        request.tool_choice = match mode.as_ref() {
            Some(GeminiFunctionCallingMode::None) => Some(ToolChoice::None),
            Some(GeminiFunctionCallingMode::Any) => {
                if allowed_function_names
                    .as_ref()
                    .is_some_and(|names| names.len() == 1)
                {
                    Some(ToolChoice::tool(
                        allowed_function_names
                            .as_ref()
                            .expect("checked single item")[0]
                            .clone(),
                    ))
                } else {
                    Some(ToolChoice::Required)
                }
            }
            Some(GeminiFunctionCallingMode::Auto)
            | Some(GeminiFunctionCallingMode::Unspecified)
            | None => Some(ToolChoice::Auto),
        };

        parsed.allowed_function_names = allowed_function_names;
    }

    if let Some(retrieval_config) = raw_tool_config
        .and_then(|obj| obj.get("retrievalConfig"))
        .filter(|value| value.is_object())
    {
        google_options.insert("retrievalConfig".to_string(), retrieval_config.clone());
    } else if let Some(retrieval_config) = &tool_config.retrieval_config {
        google_options.insert(
            "retrievalConfig".to_string(),
            serde_json::to_value(retrieval_config).map_err(|err| {
                LlmError::ParseError(format!("failed to serialize Gemini retrievalConfig: {err}"))
            })?,
        );
    }

    Ok(parsed)
}

fn parse_gemini_generation_config(
    generation_config: &GeminiGenerationConfig,
    raw_generation_config: Option<&Map<String, Value>>,
    request: &mut ChatRequest,
    google_options: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    request.common_params.temperature = generation_config.temperature;
    request.common_params.top_p = generation_config.top_p;
    request.common_params.frequency_penalty = generation_config.frequency_penalty;
    request.common_params.presence_penalty = generation_config.presence_penalty;
    request.common_params.stop_sequences = generation_config
        .stop_sequences
        .clone()
        .filter(|sequences| !sequences.is_empty());

    if let Some(max_output_tokens) = generation_config.max_output_tokens {
        request.common_params.max_tokens =
            Some(u32::try_from(max_output_tokens).map_err(|_| {
                LlmError::ParseError(
                    "Gemini generationConfig.maxOutputTokens must be >= 0".to_string(),
                )
            })?);
    }
    if let Some(top_k) = generation_config.top_k {
        request.common_params.top_k = Some(f64::from(top_k));
    }
    if let Some(seed) = generation_config.seed {
        request.common_params.seed = Some(u64::try_from(seed).map_err(|_| {
            LlmError::ParseError("Gemini generationConfig.seed must be >= 0".to_string())
        })?);
    }

    if let Some(raw) = raw_generation_config {
        if let Some(schema) = raw.get("responseJsonSchema") {
            request.response_format = Some(ResponseFormat::json_schema(schema.clone()));
            google_options.insert("responseJsonSchema".to_string(), schema.clone());
            google_options.insert("structuredOutputs".to_string(), Value::Bool(false));
        } else if let Some(schema) = raw.get("responseSchema") {
            request.response_format = Some(ResponseFormat::json_schema(schema.clone()));
        }

        let preserve_keys = [
            "responseMimeType",
            "responseModalities",
            "thinkingConfig",
            "audioTimestamp",
            "mediaResolution",
            "imageConfig",
            "responseLogprobs",
            "logprobs",
        ];
        for key in preserve_keys {
            if let Some(value) = raw.get(key) {
                let should_skip = key == "responseMimeType"
                    && value.as_str() == Some("application/json")
                    && request.response_format.is_some();
                if !should_skip {
                    google_options.insert(key.to_string(), value.clone());
                }
            }
        }
    }

    Ok(())
}

fn request_text_part(text: impl Into<String>, provider_options: ProviderOptionsMap) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options,
        provider_metadata: None,
    }
}

fn request_reasoning_part(
    text: impl Into<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Reasoning {
        text: text.into(),
        provider_options,
        provider_metadata: None,
    }
}

fn request_image_part(
    source: FilePartSource,
    media_type: Option<String>,
    detail: Option<ImageDetail>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Image {
        source,
        media_type,
        detail,
        provider_options,
        provider_metadata: None,
    }
}

fn request_audio_part(
    source: MediaSource,
    media_type: Option<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Audio {
        source,
        media_type,
        provider_options,
        provider_metadata: None,
    }
}

fn request_file_part(
    source: FilePartSource,
    media_type: impl Into<String>,
    filename: Option<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::File {
        source,
        media_type: media_type.into(),
        filename,
        provider_options,
        provider_metadata: None,
    }
}

fn request_tool_call_part(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    arguments: serde_json::Value,
    provider_executed: Option<bool>,
    dynamic: Option<bool>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::ToolCall {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        arguments,
        provider_executed,
        dynamic,
        invalid: None,
        error: None,
        title: None,
        provider_options,
        provider_metadata: None,
    }
}

fn request_tool_result_part(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    output: ToolResultOutput,
    provider_executed: Option<bool>,
    dynamic: Option<bool>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::ToolResult {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        output,
        input: None,
        provider_executed,
        dynamic,
        preliminary: None,
        title: None,
        provider_options,
        provider_metadata: None,
    }
}

fn expect_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, LlmError> {
    value
        .as_object()
        .ok_or_else(|| LlmError::ParseError(format!("{label} must be a JSON object")))
}

fn required_string(obj: &Map<String, Value>, key: &str, label: &str) -> Result<String, LlmError> {
    obj.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| LlmError::ParseError(format!("{label}.{key} must be a non-empty string")))
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
        metadata: MessageMetadata::default(),
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
        MessageContent::Json(value) => vec![request_text_part(
            serde_json::to_string(&value).unwrap_or_default(),
            ProviderOptionsMap::default(),
        )],
    }
}

fn parse_text_like_content_parts(text: &str) -> Vec<ContentPart> {
    if let Some(reasoning) = strip_wrapped_thinking(text) {
        vec![request_reasoning_part(
            reasoning,
            ProviderOptionsMap::default(),
        )]
    } else {
        vec![request_text_part(text, ProviderOptionsMap::default())]
    }
}

fn strip_wrapped_thinking(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let inner = trimmed
        .strip_prefix("<thinking>")
        .and_then(|value| value.strip_suffix("</thinking>"))?;
    Some(inner.to_string())
}
