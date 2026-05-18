use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::HashSet;

use crate::LlmError;
use crate::provider_options::gemini::{
    GoogleInteractionsResponseFormatEntry, GoogleLanguageModelInteractionsOptions,
};
use crate::types::{
    ChatMessage, ChatRequest, ContentPart, MediaSource, MessageContent, MessageRole,
    ProviderMetadataMap, ProviderOptionsMap, ResponseFormat, Tool, ToolChoice, ToolFunction,
    ToolResultContentPart, ToolResultFileId, ToolResultOutput, Warning,
};

use super::GoogleInteractionsModelInput;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct GoogleInteractionsPreparedRequest {
    pub(crate) body: GoogleInteractionsRequestBody,
    pub(crate) warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct GoogleInteractionsRequestBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent: Option<String>,
    input: Vec<GoogleInteractionsStep>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "system_instruction")]
    system_instruction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "response_format")]
    response_format: Option<Vec<GoogleInteractionsResponseFormatWireEntry>>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "response_modalities"
    )]
    response_modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generation_config")]
    generation_config: Option<GoogleInteractionsGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GoogleInteractionsTool>>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "previous_interaction_id"
    )]
    previous_interaction_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "service_tier")]
    service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<bool>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GoogleInteractionsStep {
    UserInput {
        content: Vec<GoogleInteractionsContentBlock>,
    },
    ModelOutput {
        content: Vec<GoogleInteractionsContentBlock>,
    },
    Thought {
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<Vec<GoogleInteractionsContentBlock>>,
    },
    FunctionCall {
        id: String,
        name: String,
        arguments: Map<String, Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GoogleInteractionsContentBlock {
    Text {
        text: String,
    },
    Image {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "mime_type")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        resolution: Option<String>,
    },
    Audio {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "mime_type")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
    },
    Document {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "mime_type")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
    },
    Video {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "mime_type")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        resolution: Option<String>,
    },
    FunctionResult {
        #[serde(rename = "call_id")]
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        result: Value,
        #[serde(skip_serializing_if = "Option::is_none", rename = "is_error")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct GoogleInteractionsGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "top_p")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "stop_sequences")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "max_output_tokens")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinking_level")]
    thinking_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinking_summaries")]
    thinking_summaries: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_choice")]
    tool_choice: Option<GoogleInteractionsToolChoice>,
}

impl GoogleInteractionsGenerationConfig {
    fn from_request(
        req: &ChatRequest,
        opts: &GoogleLanguageModelInteractionsOptions,
        tool_choice: Option<GoogleInteractionsToolChoice>,
    ) -> Option<Self> {
        let config = Self {
            temperature: req.common_params.temperature,
            top_p: req.common_params.top_p,
            seed: req.common_params.seed,
            stop_sequences: req.common_params.stop_sequences.clone(),
            max_output_tokens: req
                .common_params
                .max_completion_tokens
                .or(req.common_params.max_tokens),
            thinking_level: opts.thinking_level.clone(),
            thinking_summaries: opts.thinking_summaries.clone(),
            tool_choice,
        };

        if config == Self::default() {
            None
        } else {
            Some(config)
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GoogleInteractionsTool {
    Function {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<Value>,
    },
    CodeExecution,
    UrlContext,
    ComputerUse {
        #[serde(skip_serializing_if = "Option::is_none")]
        environment: Option<String>,
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "excludedPredefinedFunctions"
        )]
        excluded_predefined_functions: Option<Vec<String>>,
    },
    McpServer {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        headers: Option<Map<String, Value>>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "allowed_tools")]
        allowed_tools: Option<Vec<Value>>,
    },
    GoogleSearch {
        #[serde(skip_serializing_if = "Option::is_none", rename = "search_types")]
        search_types: Option<Vec<String>>,
    },
    FileSearch {
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "file_search_store_names"
        )]
        file_search_store_names: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "metadata_filter")]
        metadata_filter: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "top_k")]
        top_k: Option<u64>,
    },
    GoogleMaps {
        #[serde(skip_serializing_if = "Option::is_none", rename = "enable_widget")]
        enable_widget: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        latitude: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        longitude: Option<f64>,
    },
    Retrieval {
        #[serde(skip_serializing_if = "Option::is_none", rename = "retrieval_types")]
        retrieval_types: Option<Vec<String>>,
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "vertex_ai_search_config"
        )]
        vertex_ai_search_config: Option<Value>,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
enum GoogleInteractionsToolChoice {
    Mode(String),
    AllowedTools {
        #[serde(rename = "allowed_tools")]
        allowed_tools: GoogleInteractionsAllowedToolsConfig,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct GoogleInteractionsAllowedToolsConfig {
    mode: String,
    tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
enum GoogleInteractionsResponseFormatWireEntry {
    Text {
        #[serde(skip_serializing_if = "Option::is_none", rename = "mime_type")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<Value>,
    },
    Image {
        #[serde(skip_serializing_if = "Option::is_none", rename = "mime_type")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "aspect_ratio")]
        aspect_ratio: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "image_size")]
        image_size: Option<String>,
    },
    Audio {
        #[serde(skip_serializing_if = "Option::is_none", rename = "mime_type")]
        mime_type: Option<String>,
    },
}

fn parse_interactions_options(
    request: &ChatRequest,
) -> Result<GoogleLanguageModelInteractionsOptions, LlmError> {
    let Some(value) = request.provider_option("google") else {
        return Ok(GoogleLanguageModelInteractionsOptions::default());
    };
    if value.is_null() {
        return Ok(GoogleLanguageModelInteractionsOptions::default());
    }
    serde_json::from_value(value.clone()).map_err(|error| {
        LlmError::InvalidParameter(format!(
            "invalid google.interactions provider options: {error}"
        ))
    })
}

pub(crate) fn build_interactions_request_body(
    model_input: &GoogleInteractionsModelInput,
    request: &ChatRequest,
    stream: bool,
) -> Result<GoogleInteractionsPreparedRequest, LlmError> {
    if model_input.is_agent() {
        return Err(LlmError::UnsupportedOperation(
            "google.interactions agent request conversion is tracked by GIR-030".to_string(),
        ));
    }

    let options = parse_interactions_options(request)?;
    let mut warnings = Vec::new();

    let mut input = Vec::new();
    let mut system_texts = Vec::new();
    let mut dropped_tool_call_ids = HashSet::new();
    let previous_interaction_id = options.previous_interaction_id.as_deref();
    let should_compact = previous_interaction_id.is_some() && options.store != Some(false);

    if previous_interaction_id.is_some() && options.store == Some(false) {
        warnings.push(Warning::other(
            "google.interactions: previousInteractionId was set together with store: false; full history will be sent.",
        ));
    }

    for message in &request.messages {
        if should_compact
            && message.role == MessageRole::Assistant
            && message_matches_interaction(message, previous_interaction_id.expect("checked"))
        {
            collect_tool_call_ids(message, &mut dropped_tool_call_ids);
            continue;
        }

        match message.role {
            MessageRole::System | MessageRole::Developer => {
                let text = message.content.all_text();
                if !text.is_empty() {
                    system_texts.push(text);
                }
            }
            MessageRole::User => {
                let content = convert_user_content(
                    &message.content,
                    options.media_resolution.as_deref(),
                    &mut warnings,
                );
                if !content.is_empty() {
                    input.push(GoogleInteractionsStep::UserInput { content });
                }
            }
            MessageRole::Assistant => {
                convert_assistant_content(
                    &message.content,
                    options.media_resolution.as_deref(),
                    &mut input,
                    &mut warnings,
                );
            }
            MessageRole::Tool => {
                let content =
                    convert_tool_message(&message.content, &dropped_tool_call_ids, &mut warnings);
                if !content.is_empty() {
                    input.push(GoogleInteractionsStep::UserInput { content });
                }
            }
        }
    }

    let converted_system_instruction = if system_texts.is_empty() {
        None
    } else {
        Some(system_texts.join("\n\n"))
    };
    let system_instruction = match (
        converted_system_instruction,
        options.system_instruction.clone(),
    ) {
        (Some(from_prompt), Some(_from_options)) => {
            warnings.push(Warning::other(
                    "google.interactions: both system messages and providerOptions.google.systemInstruction were set; using system messages.",
                ));
            Some(from_prompt)
        }
        (Some(from_prompt), None) => Some(from_prompt),
        (None, Some(from_options)) => Some(from_options),
        (None, None) => None,
    };

    let response_format = build_response_format(request, &options, &mut warnings)?;
    let prepared_tools = prepare_interactions_tools(
        request.tools.as_deref(),
        request.tool_choice.as_ref(),
        &mut warnings,
    );
    let generation_config = GoogleInteractionsGenerationConfig::from_request(
        request,
        &options,
        prepared_tools.tool_choice,
    );

    let body = GoogleInteractionsRequestBody {
        model: Some(model_input.id().to_string()),
        agent: None,
        input,
        system_instruction,
        response_format,
        response_modalities: options.response_modalities.clone(),
        generation_config,
        tools: prepared_tools.tools,
        previous_interaction_id: options.previous_interaction_id.clone(),
        service_tier: options.service_tier.clone(),
        store: options.store,
        stream: stream.then_some(true),
        background: None,
    };

    Ok(GoogleInteractionsPreparedRequest { body, warnings })
}

#[derive(Default)]
struct PreparedInteractionsTools {
    tools: Option<Vec<GoogleInteractionsTool>>,
    tool_choice: Option<GoogleInteractionsToolChoice>,
}

fn prepare_interactions_tools(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
    warnings: &mut Vec<Warning>,
) -> PreparedInteractionsTools {
    let Some(tools) = tools.filter(|tools| !tools.is_empty()) else {
        return PreparedInteractionsTools::default();
    };

    let mut interactions_tools = Vec::new();
    let mut has_function_tool = false;

    for tool in tools {
        match tool {
            Tool::Function { function } => {
                has_function_tool = true;
                interactions_tools.push(function_tool_to_interactions_tool(function));
            }
            Tool::ProviderDefined(provider_tool) => match provider_tool.id.as_str() {
                "google.google_search" => {
                    interactions_tools.push(google_search_tool(provider_tool.args.as_object()));
                }
                "google.code_execution" => {
                    interactions_tools.push(GoogleInteractionsTool::CodeExecution);
                }
                "google.url_context" => {
                    interactions_tools.push(GoogleInteractionsTool::UrlContext);
                }
                "google.file_search" => {
                    interactions_tools.push(file_search_tool(provider_tool.args.as_object()));
                }
                "google.google_maps" => {
                    interactions_tools.push(google_maps_tool(provider_tool.args.as_object()));
                }
                "google.computer_use" => {
                    interactions_tools.push(computer_use_tool(provider_tool.args.as_object()));
                }
                "google.mcp_server" => {
                    interactions_tools.push(mcp_server_tool(provider_tool.args.as_object()));
                }
                "google.retrieval" | "google.google_search_retrieval" | "google.vertex_rag_store" => {
                    interactions_tools.push(retrieval_tool(provider_tool.args.as_object()));
                }
                other => warnings.push(Warning::unsupported(
                    format!("provider-defined tool {other}"),
                    Some(format!(
                        "provider-defined tool {other} is not supported by google.interactions; tool dropped."
                    )),
                )),
            },
        }
    }

    let tool_choice = if has_function_tool {
        map_tool_choice(tool_choice)
    } else {
        None
    };

    PreparedInteractionsTools {
        tools: (!interactions_tools.is_empty()).then_some(interactions_tools),
        tool_choice,
    }
}

fn function_tool_to_interactions_tool(function: &ToolFunction) -> GoogleInteractionsTool {
    GoogleInteractionsTool::Function {
        name: Some(function.name.clone()),
        description: (!function.description.is_empty()).then(|| function.description.clone()),
        parameters: Some(function.parameters.clone()),
    }
}

fn google_search_tool(args: Option<&Map<String, Value>>) -> GoogleInteractionsTool {
    let search_types = args
        .and_then(|args| {
            object_field(args, "searchTypes").or_else(|| object_field(args, "search_types"))
        })
        .map(|search_types| {
            let mut values = Vec::new();
            if search_types
                .get("webSearch")
                .or_else(|| search_types.get("web_search"))
                .is_some()
            {
                values.push("web_search".to_string());
            }
            if search_types
                .get("imageSearch")
                .or_else(|| search_types.get("image_search"))
                .is_some()
            {
                values.push("image_search".to_string());
            }
            if search_types
                .get("enterpriseWebSearch")
                .or_else(|| search_types.get("enterprise_web_search"))
                .is_some()
            {
                values.push("enterprise_web_search".to_string());
            }
            values
        })
        .filter(|values| !values.is_empty());

    GoogleInteractionsTool::GoogleSearch { search_types }
}

fn file_search_tool(args: Option<&Map<String, Value>>) -> GoogleInteractionsTool {
    GoogleInteractionsTool::FileSearch {
        file_search_store_names: args.and_then(|args| {
            string_array_field(args, "fileSearchStoreNames")
                .or_else(|| string_array_field(args, "file_search_store_names"))
        }),
        metadata_filter: args.and_then(|args| {
            string_field(args, "metadataFilter").or_else(|| string_field(args, "metadata_filter"))
        }),
        top_k: args.and_then(|args| u64_field(args, "topK").or_else(|| u64_field(args, "top_k"))),
    }
}

fn google_maps_tool(args: Option<&Map<String, Value>>) -> GoogleInteractionsTool {
    GoogleInteractionsTool::GoogleMaps {
        enable_widget: args.and_then(|args| {
            bool_field(args, "enableWidget").or_else(|| bool_field(args, "enable_widget"))
        }),
        latitude: args.and_then(|args| f64_field(args, "latitude")),
        longitude: args.and_then(|args| f64_field(args, "longitude")),
    }
}

fn computer_use_tool(args: Option<&Map<String, Value>>) -> GoogleInteractionsTool {
    GoogleInteractionsTool::ComputerUse {
        environment: args
            .and_then(|args| string_field(args, "environment"))
            .or_else(|| Some("browser".to_string())),
        excluded_predefined_functions: args.and_then(|args| {
            string_array_field(args, "excludedPredefinedFunctions")
                .or_else(|| string_array_field(args, "excluded_predefined_functions"))
        }),
    }
}

fn mcp_server_tool(args: Option<&Map<String, Value>>) -> GoogleInteractionsTool {
    GoogleInteractionsTool::McpServer {
        name: args.and_then(|args| string_field(args, "name")),
        url: args.and_then(|args| string_field(args, "url")),
        headers: args.and_then(|args| object_field(args, "headers").cloned()),
        allowed_tools: args.and_then(|args| {
            value_array_field(args, "allowedTools")
                .or_else(|| value_array_field(args, "allowed_tools"))
        }),
    }
}

fn retrieval_tool(args: Option<&Map<String, Value>>) -> GoogleInteractionsTool {
    GoogleInteractionsTool::Retrieval {
        retrieval_types: args
            .and_then(|args| {
                string_array_field(args, "retrievalTypes")
                    .or_else(|| string_array_field(args, "retrieval_types"))
            })
            .or_else(|| Some(vec!["vertex_ai_search".to_string()])),
        vertex_ai_search_config: args.and_then(|args| {
            args.get("vertexAiSearchConfig")
                .or_else(|| args.get("vertex_ai_search_config"))
                .cloned()
                .or_else(|| {
                    string_field(args, "ragCorpus")
                        .or_else(|| string_field(args, "rag_corpus"))
                        .map(|rag_corpus| {
                            serde_json::json!({
                                "datastores": [rag_corpus]
                            })
                        })
                })
        }),
    }
}

fn map_tool_choice(tool_choice: Option<&ToolChoice>) -> Option<GoogleInteractionsToolChoice> {
    match tool_choice? {
        ToolChoice::Auto => Some(GoogleInteractionsToolChoice::Mode("auto".to_string())),
        ToolChoice::Required => Some(GoogleInteractionsToolChoice::Mode("any".to_string())),
        ToolChoice::None => Some(GoogleInteractionsToolChoice::Mode("none".to_string())),
        ToolChoice::Tool { name } => Some(GoogleInteractionsToolChoice::AllowedTools {
            allowed_tools: GoogleInteractionsAllowedToolsConfig {
                mode: "validated".to_string(),
                tools: vec![name.clone()],
            },
        }),
    }
}

fn build_response_format(
    request: &ChatRequest,
    options: &GoogleLanguageModelInteractionsOptions,
    warnings: &mut Vec<Warning>,
) -> Result<Option<Vec<GoogleInteractionsResponseFormatWireEntry>>, LlmError> {
    let mut entries = Vec::new();

    if let Some(format) = &request.response_format {
        match format {
            ResponseFormat::JsonObject { .. } => {
                entries.push(GoogleInteractionsResponseFormatWireEntry::Text {
                    mime_type: Some("application/json".to_string()),
                    schema: None,
                });
            }
            ResponseFormat::Json { schema, .. } => {
                entries.push(GoogleInteractionsResponseFormatWireEntry::Text {
                    mime_type: Some("application/json".to_string()),
                    schema: Some(schema.clone()),
                });
            }
        }
    }

    if let Some(option_entries) = &options.response_format {
        for entry in option_entries {
            entries.push(response_format_entry_to_wire(entry)?);
        }
    }

    if let Some(image_config) = &options.image_config {
        let already_has_image_entry = entries.iter().any(|entry| {
            matches!(
                entry,
                GoogleInteractionsResponseFormatWireEntry::Image { .. }
            )
        });
        warnings.push(Warning::deprecated(
            "providerOptions.google.imageConfig",
            if already_has_image_entry {
                "Use providerOptions.google.responseFormat instead; imageConfig was ignored because responseFormat already supplies an image entry."
            } else {
                "Use providerOptions.google.responseFormat with a { type: \"image\", ... } entry instead."
            },
        ));
        if !already_has_image_entry {
            entries.push(GoogleInteractionsResponseFormatWireEntry::Image {
                mime_type: Some("image/png".to_string()),
                aspect_ratio: image_config.aspect_ratio.clone(),
                image_size: image_config.image_size.clone(),
            });
        }
    }

    if entries.is_empty() {
        Ok(None)
    } else {
        Ok(Some(entries))
    }
}

fn response_format_entry_to_wire(
    entry: &GoogleInteractionsResponseFormatEntry,
) -> Result<GoogleInteractionsResponseFormatWireEntry, LlmError> {
    let value = serde_json::to_value(entry).map_err(|error| {
        LlmError::InvalidParameter(format!(
            "invalid google.interactions responseFormat entry: {error}"
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        LlmError::InvalidParameter(
            "google.interactions responseFormat entry must serialize as an object".to_string(),
        )
    })?;
    match object.get("type").and_then(Value::as_str) {
        Some("text") => Ok(GoogleInteractionsResponseFormatWireEntry::Text {
            mime_type: string_field(object, "mimeType")
                .or_else(|| string_field(object, "mime_type")),
            schema: object.get("schema").cloned(),
        }),
        Some("image") => Ok(GoogleInteractionsResponseFormatWireEntry::Image {
            mime_type: string_field(object, "mimeType")
                .or_else(|| string_field(object, "mime_type")),
            aspect_ratio: string_field(object, "aspectRatio")
                .or_else(|| string_field(object, "aspect_ratio")),
            image_size: string_field(object, "imageSize")
                .or_else(|| string_field(object, "image_size")),
        }),
        Some("audio") => Ok(GoogleInteractionsResponseFormatWireEntry::Audio {
            mime_type: string_field(object, "mimeType")
                .or_else(|| string_field(object, "mime_type")),
        }),
        other => Err(LlmError::InvalidParameter(format!(
            "unsupported google.interactions responseFormat entry type {other:?}"
        ))),
    }
}

fn string_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    object.get(key)?.as_str().map(ToOwned::to_owned)
}

fn object_field<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a Map<String, Value>> {
    object.get(key)?.as_object()
}

fn value_array_field(object: &Map<String, Value>, key: &str) -> Option<Vec<Value>> {
    Some(object.get(key)?.as_array()?.clone())
}

fn string_array_field(object: &Map<String, Value>, key: &str) -> Option<Vec<String>> {
    let values = object
        .get(key)?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(values)
}

fn bool_field(object: &Map<String, Value>, key: &str) -> Option<bool> {
    object.get(key)?.as_bool()
}

fn u64_field(object: &Map<String, Value>, key: &str) -> Option<u64> {
    object.get(key)?.as_u64()
}

fn f64_field(object: &Map<String, Value>, key: &str) -> Option<f64> {
    object.get(key)?.as_f64()
}

fn convert_user_content(
    content: &MessageContent,
    media_resolution: Option<&str>,
    warnings: &mut Vec<Warning>,
) -> Vec<GoogleInteractionsContentBlock> {
    match content {
        MessageContent::Text(text) => {
            vec![GoogleInteractionsContentBlock::Text { text: text.clone() }]
        }
        MessageContent::MultiModal(parts) => merge_adjacent_text_content(
            parts
                .iter()
                .filter_map(|part| content_part_to_block(part, media_resolution, warnings))
                .collect(),
        ),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            vec![GoogleInteractionsContentBlock::Text {
                text: serde_json::to_string(value).unwrap_or_default(),
            }]
        }
    }
}

fn convert_assistant_content(
    content: &MessageContent,
    media_resolution: Option<&str>,
    input: &mut Vec<GoogleInteractionsStep>,
    warnings: &mut Vec<Warning>,
) {
    let mut pending_model_output = Vec::new();
    let flush = |input: &mut Vec<GoogleInteractionsStep>,
                 pending_model_output: &mut Vec<GoogleInteractionsContentBlock>| {
        if !pending_model_output.is_empty() {
            input.push(GoogleInteractionsStep::ModelOutput {
                content: std::mem::take(pending_model_output),
            });
        }
    };

    match content {
        MessageContent::Text(text) => {
            if !text.is_empty() {
                pending_model_output
                    .push(GoogleInteractionsContentBlock::Text { text: text.clone() });
            }
        }
        MessageContent::MultiModal(parts) => {
            for part in parts {
                match part {
                    ContentPart::Reasoning {
                        text,
                        provider_options,
                        provider_metadata,
                        ..
                    } => {
                        flush(input, &mut pending_model_output);
                        let summary = (!text.is_empty()).then(|| {
                            vec![GoogleInteractionsContentBlock::Text { text: text.clone() }]
                        });
                        input.push(GoogleInteractionsStep::Thought {
                            signature: signature_from_provider_fields(
                                Some(provider_options),
                                provider_metadata.as_ref(),
                            ),
                            summary,
                        });
                    }
                    ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        provider_options,
                        provider_metadata,
                        ..
                    } => {
                        flush(input, &mut pending_model_output);
                        input.push(GoogleInteractionsStep::FunctionCall {
                            id: tool_call_id.clone(),
                            name: tool_name.clone(),
                            arguments: tool_arguments_to_object(arguments),
                            signature: signature_from_provider_fields(
                                Some(provider_options),
                                provider_metadata.as_ref(),
                            ),
                        });
                    }
                    _ => {
                        if let Some(block) = content_part_to_block(part, media_resolution, warnings)
                        {
                            pending_model_output.push(block);
                        }
                    }
                }
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            pending_model_output.push(GoogleInteractionsContentBlock::Text {
                text: serde_json::to_string(value).unwrap_or_default(),
            });
        }
    }
    flush(input, &mut pending_model_output);
}

fn convert_tool_message(
    content: &MessageContent,
    dropped_tool_call_ids: &HashSet<String>,
    warnings: &mut Vec<Warning>,
) -> Vec<GoogleInteractionsContentBlock> {
    match content {
        MessageContent::Text(text) => {
            warnings.push(Warning::other(
                "google.interactions: legacy text tool message was converted without a tool call id.",
            ));
            vec![GoogleInteractionsContentBlock::FunctionResult {
                call_id: String::new(),
                name: None,
                result: Value::String(text.clone()),
                is_error: None,
                signature: None,
            }]
        }
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolResult {
                    tool_call_id,
                    tool_name,
                    output,
                    provider_options,
                    provider_metadata,
                    ..
                } => {
                    if dropped_tool_call_ids.contains(tool_call_id) {
                        return None;
                    }
                    Some(tool_result_to_block(
                        tool_call_id,
                        tool_name,
                        output,
                        Some(provider_options),
                        provider_metadata.as_ref(),
                        warnings,
                    ))
                }
                other => {
                    warnings.push(Warning::other(format!(
                        "google.interactions: unsupported tool message part type {:?}; part dropped.",
                        part_kind(other)
                    )));
                    None
                }
            })
            .collect(),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            vec![GoogleInteractionsContentBlock::FunctionResult {
                call_id: String::new(),
                name: None,
                result: Value::String(serde_json::to_string(value).unwrap_or_default()),
                is_error: None,
                signature: None,
            }]
        }
    }
}

fn content_part_to_block(
    part: &ContentPart,
    media_resolution: Option<&str>,
    warnings: &mut Vec<Warning>,
) -> Option<GoogleInteractionsContentBlock> {
    match part {
        ContentPart::Text { text, .. } => {
            Some(GoogleInteractionsContentBlock::Text { text: text.clone() })
        }
        ContentPart::Image {
            source, media_type, ..
        } => media_source_to_block(
            "image",
            source.as_media_source(),
            source.as_provider_reference(),
            media_type.as_deref(),
            media_resolution,
            warnings,
        ),
        ContentPart::Audio {
            source, media_type, ..
        } => media_source_to_block(
            "audio",
            Some(source),
            None,
            media_type.as_deref(),
            None,
            warnings,
        ),
        ContentPart::File {
            source, media_type, ..
        } => media_source_to_block(
            media_kind_for_type(media_type),
            source.as_media_source(),
            source.as_provider_reference(),
            Some(media_type.as_str()),
            media_resolution,
            warnings,
        ),
        ContentPart::ReasoningFile {
            source, media_type, ..
        } => media_source_to_block(
            media_kind_for_type(media_type),
            Some(source),
            None,
            Some(media_type.as_str()),
            media_resolution,
            warnings,
        ),
        other => {
            warnings.push(Warning::other(format!(
                "google.interactions: unsupported content part type {:?}; part dropped.",
                part_kind(other)
            )));
            None
        }
    }
}

fn media_source_to_block(
    kind: &str,
    media_source: Option<&MediaSource>,
    provider_reference: Option<&crate::types::ProviderReference>,
    media_type: Option<&str>,
    media_resolution: Option<&str>,
    warnings: &mut Vec<Warning>,
) -> Option<GoogleInteractionsContentBlock> {
    let (data, uri) = if let Some(source) = media_source {
        (source.as_base64(), source.as_url().map(ToOwned::to_owned))
    } else if let Some(reference) = provider_reference {
        let Some(value) = reference
            .preferred_value(&["google", "gemini", "vertex", "google-vertex"])
            .or_else(|| reference.0.values().next().map(String::as_str))
        else {
            warnings.push(Warning::other(
                "google.interactions: provider reference did not contain a usable file id; part dropped.",
            ));
            return None;
        };
        (None, Some(value.to_string()))
    } else {
        (None, None)
    };

    let resolution = match kind {
        "image" | "video" => media_resolution.map(ToOwned::to_owned),
        _ => None,
    };
    let mime_type = media_type
        .filter(|value| value.contains('/'))
        .map(str::to_string);

    Some(match kind {
        "image" => GoogleInteractionsContentBlock::Image {
            data,
            mime_type,
            uri,
            resolution,
        },
        "audio" => GoogleInteractionsContentBlock::Audio {
            data,
            mime_type,
            uri,
        },
        "video" => GoogleInteractionsContentBlock::Video {
            data,
            mime_type,
            uri,
            resolution,
        },
        "document" => GoogleInteractionsContentBlock::Document {
            data,
            mime_type,
            uri,
        },
        _ => {
            warnings.push(Warning::other(format!(
                "google.interactions: unsupported file media type {:?}; part dropped.",
                media_type
            )));
            return None;
        }
    })
}

fn media_kind_for_type(media_type: &str) -> &str {
    match media_type.split('/').next().unwrap_or_default() {
        "image" => "image",
        "audio" => "audio",
        "video" => "video",
        "application" | "text" => "document",
        _ => "unsupported",
    }
}

fn tool_result_to_block(
    tool_call_id: &str,
    tool_name: &str,
    output: &ToolResultOutput,
    provider_options: Option<&ProviderOptionsMap>,
    provider_metadata: Option<&ProviderMetadataMap>,
    warnings: &mut Vec<Warning>,
) -> GoogleInteractionsContentBlock {
    let (result, is_error) = match output {
        ToolResultOutput::Text { value, .. } => (Value::String(value.clone()), None),
        ToolResultOutput::Json { value, .. } => (
            Value::String(serde_json::to_string(value).unwrap_or_default()),
            None,
        ),
        ToolResultOutput::ExecutionDenied { reason, .. } => (
            Value::String(
                reason
                    .clone()
                    .unwrap_or_else(|| "Tool execution denied by user.".to_string()),
            ),
            Some(true),
        ),
        ToolResultOutput::ErrorText { value, .. } => (Value::String(value.clone()), Some(true)),
        ToolResultOutput::ErrorJson { value, .. } => (
            Value::String(serde_json::to_string(value).unwrap_or_default()),
            Some(true),
        ),
        ToolResultOutput::Content { value, .. } => {
            let blocks: Vec<_> = value
                .iter()
                .filter_map(|part| tool_result_content_part_to_block(part, warnings))
                .collect();
            (serde_json::to_value(blocks).unwrap_or(Value::Null), None)
        }
    };

    GoogleInteractionsContentBlock::FunctionResult {
        call_id: tool_call_id.to_string(),
        name: (!tool_name.is_empty()).then(|| tool_name.to_string()),
        result,
        is_error,
        signature: signature_from_provider_fields(provider_options, provider_metadata),
    }
}

fn tool_result_content_part_to_block(
    part: &ToolResultContentPart,
    warnings: &mut Vec<Warning>,
) -> Option<GoogleInteractionsContentBlock> {
    match part {
        ToolResultContentPart::Text { text, .. } => {
            Some(GoogleInteractionsContentBlock::Text { text: text.clone() })
        }
        ToolResultContentPart::ImageData {
            data, media_type, ..
        } => Some(GoogleInteractionsContentBlock::Image {
            data: Some(data.clone()),
            mime_type: Some(media_type.clone()),
            uri: None,
            resolution: None,
        }),
        ToolResultContentPart::ImageUrl { url, .. } => {
            Some(GoogleInteractionsContentBlock::Image {
                data: None,
                mime_type: None,
                uri: Some(url.clone()),
                resolution: None,
            })
        }
        ToolResultContentPart::ImageFileId { file_id, .. } => file_id_to_image_block(file_id),
        ToolResultContentPart::ImageFileReference {
            provider_reference, ..
        } => provider_reference_to_image_block(provider_reference),
        other => {
            warnings.push(Warning::other(format!(
                "google.interactions: tool-result content part type {:?} is not supported in function_result.result; part dropped.",
                tool_result_content_part_kind(other)
            )));
            None
        }
    }
}

fn file_id_to_image_block(file_id: &ToolResultFileId) -> Option<GoogleInteractionsContentBlock> {
    let uri = match file_id {
        ToolResultFileId::Single(value) => value.as_str(),
        ToolResultFileId::PerProvider(values) => values
            .get("google")
            .or_else(|| values.get("gemini"))
            .or_else(|| values.values().next())?
            .as_str(),
    };
    Some(GoogleInteractionsContentBlock::Image {
        data: None,
        mime_type: None,
        uri: Some(uri.to_string()),
        resolution: None,
    })
}

fn provider_reference_to_image_block(
    reference: &crate::types::ProviderReference,
) -> Option<GoogleInteractionsContentBlock> {
    let uri = reference
        .preferred_value(&["google", "gemini", "vertex", "google-vertex"])
        .or_else(|| reference.0.values().next().map(String::as_str))?;
    Some(GoogleInteractionsContentBlock::Image {
        data: None,
        mime_type: None,
        uri: Some(uri.to_string()),
        resolution: None,
    })
}

fn merge_adjacent_text_content(
    content: Vec<GoogleInteractionsContentBlock>,
) -> Vec<GoogleInteractionsContentBlock> {
    let mut result: Vec<GoogleInteractionsContentBlock> = Vec::new();
    for block in content {
        match (result.last_mut(), block) {
            (
                Some(GoogleInteractionsContentBlock::Text { text: current }),
                GoogleInteractionsContentBlock::Text { text: next },
            ) => {
                current.push_str("\n\n");
                current.push_str(&next);
            }
            (_, block) => result.push(block),
        }
    }
    result
}

fn message_matches_interaction(message: &ChatMessage, previous_interaction_id: &str) -> bool {
    content_parts(&message.content).iter().any(|part| {
        part_provider_fields(part)
            .and_then(|(provider_options, provider_metadata)| {
                interaction_id_from_provider_fields(provider_options, provider_metadata)
            })
            .is_some_and(|value| value == previous_interaction_id)
    })
}

fn collect_tool_call_ids(message: &ChatMessage, out: &mut HashSet<String>) {
    for part in content_parts(&message.content) {
        if let ContentPart::ToolCall { tool_call_id, .. } = part {
            out.insert(tool_call_id.clone());
        }
    }
}

fn content_parts(content: &MessageContent) -> Vec<&ContentPart> {
    match content {
        MessageContent::MultiModal(parts) => parts.iter().collect(),
        _ => Vec::new(),
    }
}

fn part_provider_fields(
    part: &ContentPart,
) -> Option<(Option<&ProviderOptionsMap>, Option<&ProviderMetadataMap>)> {
    match part {
        ContentPart::Text {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::Image {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::Audio {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::File {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::ReasoningFile {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::Custom {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::ToolCall {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::ToolApprovalRequest {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::ToolResult {
            provider_options,
            provider_metadata,
            ..
        }
        | ContentPart::Reasoning {
            provider_options,
            provider_metadata,
            ..
        } => Some((Some(provider_options), provider_metadata.as_ref())),
        ContentPart::Source {
            provider_metadata, ..
        } => Some((None, provider_metadata.as_ref())),
        ContentPart::ToolApprovalResponse { .. } => None,
    }
}

fn metadata_object<'a>(
    metadata: &'a ProviderMetadataMap,
    provider_id: &str,
) -> Option<&'a Map<String, Value>> {
    metadata.get(provider_id)?.as_object()
}

fn options_object<'a>(
    options: &'a ProviderOptionsMap,
    provider_id: &str,
) -> Option<&'a Map<String, Value>> {
    options.get(provider_id)?.as_object()
}

fn provider_fields_string_any(
    provider_options: Option<&ProviderOptionsMap>,
    metadata: Option<&ProviderMetadataMap>,
    keys: &[&str],
) -> Option<String> {
    for provider_id in ["google", "gemini", "vertex", "google-vertex"] {
        if let Some(options) = provider_options
            && let Some(object) = options_object(options, provider_id)
        {
            for key in keys {
                if let Some(value) = object.get(*key).and_then(Value::as_str) {
                    return Some(value.to_string());
                }
            }
        }
        if let Some(metadata) = metadata
            && let Some(object) = metadata_object(metadata, provider_id)
        {
            for key in keys {
                if let Some(value) = object.get(*key).and_then(Value::as_str) {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

fn signature_from_provider_fields(
    provider_options: Option<&ProviderOptionsMap>,
    metadata: Option<&ProviderMetadataMap>,
) -> Option<String> {
    provider_fields_string_any(
        provider_options,
        metadata,
        &["signature", "thoughtSignature", "thought_signature"],
    )
}

fn interaction_id_from_provider_fields(
    provider_options: Option<&ProviderOptionsMap>,
    metadata: Option<&ProviderMetadataMap>,
) -> Option<String> {
    provider_fields_string_any(
        provider_options,
        metadata,
        &["interactionId", "interaction_id"],
    )
}

fn tool_arguments_to_object(arguments: &Value) -> Map<String, Value> {
    match arguments {
        Value::Object(object) => object.clone(),
        Value::String(value) => match serde_json::from_str::<Value>(value) {
            Ok(Value::Object(object)) => object,
            Ok(other) => Map::from_iter([("value".to_string(), other)]),
            Err(_) => Map::from_iter([("value".to_string(), Value::String(value.clone()))]),
        },
        other => Map::from_iter([("value".to_string(), other.clone())]),
    }
}

fn part_kind(part: &ContentPart) -> &'static str {
    match part {
        ContentPart::Text { .. } => "text",
        ContentPart::Image { .. } => "image",
        ContentPart::Audio { .. } => "audio",
        ContentPart::File { .. } => "file",
        ContentPart::ReasoningFile { .. } => "reasoning-file",
        ContentPart::Custom { .. } => "custom",
        ContentPart::Source { .. } => "source",
        ContentPart::ToolCall { .. } => "tool-call",
        ContentPart::ToolApprovalResponse { .. } => "tool-approval-response",
        ContentPart::ToolApprovalRequest { .. } => "tool-approval-request",
        ContentPart::ToolResult { .. } => "tool-result",
        ContentPart::Reasoning { .. } => "reasoning",
    }
}

fn tool_result_content_part_kind(part: &ToolResultContentPart) -> &'static str {
    match part {
        ToolResultContentPart::Text { .. } => "text",
        ToolResultContentPart::FileData { .. } => "file-data",
        ToolResultContentPart::FileUrl { .. } => "file-url",
        ToolResultContentPart::FileId { .. } => "file-id",
        ToolResultContentPart::FileReference { .. } => "file-reference",
        ToolResultContentPart::ImageData { .. } => "image-data",
        ToolResultContentPart::ImageUrl { .. } => "image-url",
        ToolResultContentPart::ImageFileId { .. } => "image-file-id",
        ToolResultContentPart::ImageFileReference { .. } => "image-file-reference",
        ToolResultContentPart::Custom { .. } => "custom",
    }
}
