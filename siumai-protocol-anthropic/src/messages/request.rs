use std::collections::{BTreeMap, BTreeSet};

use base64::Engine as _;
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, ExecutionOwner, LanguageRequest, MediaData, MediaPart, Message, MessageRole,
    ModelId, OpaqueProviderItem, ProviderScope, ToolChoice, ToolOutcome, ToolResult, ToolSpec,
};

use super::annotations::{
    AnthropicToolReferenceKind, CacheControl, CacheTtl, MessagesAnnotationResolver,
    MessagesFileBlock, MidConversationToolChange, MidConversationToolChangeKind,
    NoMessagesAnnotations, file_scope_is_complete,
};
use super::native_content::{
    AnthropicHostedToolBlockRef, caller_is_replayable, inspect_hosted_tool_value,
    is_maintained_hosted_kind, validate_caller,
};
use super::options::{
    AnthropicTool, McpServer, McpToolConfig, MessagesContainer, MessagesRequestOptions,
    MessagesTokenCountOptions, ServerFallback, ServerFallbacks, ThinkingConfig, TokenTaskBudget,
    UserLocation, compact_field, validate_tool_node_options,
};
use super::rules::{CacheControlWireStyle, MessagesEncodingRules, MidConversationSystemEncoding};
use super::{MessagesCodecError, OPAQUE_CONTENT_BLOCK_KIND};

const MAX_CACHE_BREAKPOINTS: usize = 4;

/// Return the sentinel JSON Schema used for Anthropic-defined tool anchors.
///
/// The schema is deliberately impossible for a local function tool. A provider
/// annotation resolver must opt the tool into a typed Anthropic tool projection
/// before the codec will accept this sentinel.
pub const fn anthropic_tool_anchor_schema() -> Value {
    Value::Bool(false)
}

/// Return whether a top-level raw body option would override canonical request
/// or transport authority.
///
/// Matching is exact after ASCII case and separator normalization. Nested
/// provider-body data is intentionally outside this policy.
pub fn is_protected_option_field(name: &str) -> bool {
    matches!(
        compact_field(name).as_str(),
        "model"
            | "messages"
            | "system"
            | "maxtokens"
            | "maxoutputtokens"
            | "stream"
            | "tools"
            | "toolchoice"
            | "temperature"
            | "topp"
            | "stopsequences"
            | "stopsequence"
            | "diagnostics"
            | "method"
            | "target"
            | "apikey"
            | "xapikey"
            | "authorization"
            | "authorizationtoken"
            | "auth"
            | "token"
            | "bearer"
            | "credential"
            | "credentials"
            | "endpoint"
            | "baseurl"
            | "url"
            | "host"
            | "headers"
            | "header"
            | "anthropicversion"
            | "anthropicbeta"
            | "proxy"
            | "tls"
            | "audience"
            | "retry"
            | "retrypolicy"
            | "timeout"
            | "connecttimeout"
            | "readtimeout"
            | "calltimeout"
    )
}

/// Encode one canonical language request as an Anthropic Messages JSON body.
pub fn encode_request(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
) -> Result<Value, MessagesCodecError> {
    encode_request_with_resolver_and_rules(
        model,
        request,
        options,
        &NoMessagesAnnotations,
        &MessagesEncodingRules::native(),
    )
}

/// Encode a request with an exact configured scope for provider-native replay.
pub fn encode_request_for_scope(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
) -> Result<Value, MessagesCodecError> {
    encode_request_for_scope_with_resolver_and_rules(
        scope,
        model,
        request,
        options,
        &NoMessagesAnnotations,
        &MessagesEncodingRules::native(),
    )
}

/// Encode a request using explicit compatible-dialect rules and no provider annotations.
pub fn encode_request_with_rules(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
    rules: &MessagesEncodingRules,
) -> Result<Value, MessagesCodecError> {
    encode_request_with_resolver_and_rules(model, request, options, &NoMessagesAnnotations, rules)
}

/// Encode a request after a provider-owned resolver projects durable
/// annotations into bounded Messages wire controls.
pub fn encode_request_with_resolver(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
    resolver: &dyn MessagesAnnotationResolver,
) -> Result<Value, MessagesCodecError> {
    encode_request_with_resolver_and_rules(
        model,
        request,
        options,
        resolver,
        &MessagesEncodingRules::native(),
    )
}

/// Encode with exact replay scope after resolving provider-owned annotations.
pub fn encode_request_for_scope_with_resolver(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
    resolver: &dyn MessagesAnnotationResolver,
) -> Result<Value, MessagesCodecError> {
    encode_request_for_scope_with_resolver_and_rules(
        scope,
        model,
        request,
        options,
        resolver,
        &MessagesEncodingRules::native(),
    )
}

/// Encode a request with both provider annotations and explicit dialect rules.
pub fn encode_request_with_resolver_and_rules(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
    resolver: &dyn MessagesAnnotationResolver,
    rules: &MessagesEncodingRules,
) -> Result<Value, MessagesCodecError> {
    encode_request_with_optional_scope(None, model, request, options, resolver, rules)
}

/// Encode with both exact replay scope and provider-owned annotation rules.
pub fn encode_request_for_scope_with_resolver_and_rules(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
    resolver: &dyn MessagesAnnotationResolver,
    rules: &MessagesEncodingRules,
) -> Result<Value, MessagesCodecError> {
    encode_request_with_optional_scope(Some(scope), model, request, options, resolver, rules)
}

/// Encode one canonical language request for Anthropic's token-count operation.
pub fn encode_count_tokens_request(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesTokenCountOptions,
) -> Result<Value, MessagesCodecError> {
    encode_count_tokens_request_with_optional_scope(
        None,
        model,
        request,
        options,
        &NoMessagesAnnotations,
        &MessagesEncodingRules::native(),
    )
}

/// Encode a token-count request with provider annotations and explicit dialect rules.
pub fn encode_count_tokens_request_with_resolver_and_rules(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesTokenCountOptions,
    resolver: &dyn MessagesAnnotationResolver,
    rules: &MessagesEncodingRules,
) -> Result<Value, MessagesCodecError> {
    encode_count_tokens_request_with_optional_scope(None, model, request, options, resolver, rules)
}

/// Encode a token-count request with exact replay scope, annotations, and dialect rules.
pub fn encode_count_tokens_request_for_scope_with_resolver_and_rules(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesTokenCountOptions,
    resolver: &dyn MessagesAnnotationResolver,
    rules: &MessagesEncodingRules,
) -> Result<Value, MessagesCodecError> {
    encode_count_tokens_request_with_optional_scope(
        Some(scope),
        model,
        request,
        options,
        resolver,
        rules,
    )
}

fn encode_count_tokens_request_with_optional_scope(
    scope: Option<&ProviderScope>,
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesTokenCountOptions,
    resolver: &dyn MessagesAnnotationResolver,
    rules: &MessagesEncodingRules,
) -> Result<Value, MessagesCodecError> {
    request
        .validate()
        .map_err(MessagesCodecError::InvalidLanguageRequest)?;
    validate_count_tokens_generation_fields(request)?;
    options.validate(request)?;

    let cache_style = rules.cache_control();
    let (system, messages) = encode_prompt(
        scope,
        &request.messages,
        resolver,
        cache_style,
        rules.video_input(),
        rules.mid_conversation_system(),
    )?;
    if messages.is_empty() {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages",
            reason: "must contain at least one user or assistant message",
        });
    }
    let tools = encode_tools(&request.tools, resolver, cache_style)?;
    validate_tool_change_references(&messages, &tools)?;
    validate_mcp_and_skills_bindings(&tools, &messages, None, options.mcp_servers.as_deref())?;
    if tools.is_empty()
        && request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| !matches!(choice, ToolChoice::None))
    {
        return Err(MessagesCodecError::InvalidOption {
            field: "tool_choice",
            reason: "requires at least one model-visible tool",
        });
    }
    validate_cache_breakpoints(
        &tools,
        &system,
        &messages,
        options.cache_control,
        cache_style,
    )?;

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));
    body.insert("messages".to_string(), Value::Array(messages));
    if !system.is_empty() {
        body.insert("system".to_string(), Value::Array(system));
    }
    if !tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(tools));
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert("tool_choice".to_string(), encode_tool_choice(tool_choice)?);
    }
    if let Some(output_config) = encode_output_config(
        options.output_effort,
        request.structured_output.as_ref(),
        options.task_budget,
    )? {
        body.insert("output_config".to_string(), output_config);
    }
    if let Some(cache_control) = options.cache_control {
        body.insert(
            "cache_control".to_string(),
            encode_cache_control(cache_control, cache_style)?,
        );
    }
    if let Some(thinking) = options.thinking {
        body.insert("thinking".to_string(), encode_thinking(thinking));
    }
    if let Some(speed) = options.speed {
        body.insert(
            "speed".to_string(),
            Value::String(speed.as_wire_str().to_string()),
        );
    }
    if let Some(context_management) = &options.context_management {
        body.insert(
            "context_management".to_string(),
            serde_json::to_value(context_management).map_err(MessagesCodecError::JsonEncode)?,
        );
    }
    if let Some(mcp_servers) = &options.mcp_servers {
        body.insert(
            "mcp_servers".to_string(),
            serde_json::to_value(mcp_servers).map_err(MessagesCodecError::JsonEncode)?,
        );
    }
    Ok(Value::Object(body))
}

fn validate_count_tokens_generation_fields(
    request: &LanguageRequest,
) -> Result<(), MessagesCodecError> {
    if request.generation.max_output_tokens.is_some() {
        return Err(MessagesCodecError::Unsupported {
            feature: "max_output_tokens in token-count requests",
        });
    }
    if request.generation.temperature.is_some() {
        return Err(MessagesCodecError::Unsupported {
            feature: "temperature in token-count requests",
        });
    }
    if request.generation.top_p.is_some() {
        return Err(MessagesCodecError::Unsupported {
            feature: "top_p in token-count requests",
        });
    }
    if !request.generation.stop_sequences.is_empty() {
        return Err(MessagesCodecError::Unsupported {
            feature: "stop_sequences in token-count requests",
        });
    }
    if request.generation.seed.is_some() {
        return Err(MessagesCodecError::Unsupported {
            feature: "seed in token-count requests",
        });
    }
    Ok(())
}

fn encode_request_with_optional_scope(
    scope: Option<&ProviderScope>,
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesRequestOptions,
    resolver: &dyn MessagesAnnotationResolver,
    rules: &MessagesEncodingRules,
) -> Result<Value, MessagesCodecError> {
    request
        .validate()
        .map_err(MessagesCodecError::InvalidLanguageRequest)?;
    options.validate(request)?;
    if let Some(fallbacks) = &options.fallbacks {
        fallbacks.validate_primary_model(model)?;
    }

    let max_tokens =
        request
            .generation
            .max_output_tokens
            .ok_or(MessagesCodecError::InvalidOption {
                field: "max_output_tokens",
                reason: "is required by Anthropic Messages",
            })?;
    if request.generation.seed.is_some() {
        return Err(MessagesCodecError::Unsupported {
            feature: "deterministic seed control",
        });
    }
    let temperature_maximum = rules.temperature().maximum();
    if request
        .generation
        .temperature
        .is_some_and(|temperature| temperature > temperature_maximum)
    {
        return Err(MessagesCodecError::InvalidOption {
            field: "temperature",
            reason: if temperature_maximum == 1.0 {
                "must be between 0 and 1 for Anthropic Messages"
            } else {
                "exceeds the configured Messages dialect maximum"
            },
        });
    }

    let cache_style = rules.cache_control();
    let (system, messages) = encode_prompt(
        scope,
        &request.messages,
        resolver,
        cache_style,
        rules.video_input(),
        rules.mid_conversation_system(),
    )?;
    if messages.is_empty() {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages",
            reason: "must contain at least one user or assistant message",
        });
    }
    let tools = encode_tools(&request.tools, resolver, cache_style)?;
    validate_tool_change_references(&messages, &tools)?;
    validate_mcp_and_skills_bindings(
        &tools,
        &messages,
        options.container.as_ref(),
        options.mcp_servers.as_deref(),
    )?;
    if tools.is_empty()
        && request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| !matches!(choice, ToolChoice::None))
    {
        return Err(MessagesCodecError::InvalidOption {
            field: "tool_choice",
            reason: "requires at least one model-visible tool",
        });
    }

    validate_cache_breakpoints(
        &tools,
        &system,
        &messages,
        options.cache_control,
        cache_style,
    )?;

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));
    body.insert("max_tokens".to_string(), Value::from(max_tokens));
    body.insert("messages".to_string(), Value::Array(messages));
    body.insert("stream".to_string(), Value::Bool(options.stream));
    if !system.is_empty() {
        body.insert("system".to_string(), Value::Array(system));
    }
    if let Some(temperature) = request.generation.temperature {
        body.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.generation.top_p {
        body.insert("top_p".to_string(), Value::from(top_p));
    }
    if let Some(top_k) = options.top_k {
        body.insert("top_k".to_string(), Value::from(top_k));
    }
    if !request.generation.stop_sequences.is_empty() {
        body.insert(
            "stop_sequences".to_string(),
            serde_json::to_value(&request.generation.stop_sequences)
                .map_err(MessagesCodecError::JsonEncode)?,
        );
    }
    if !tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(tools));
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert("tool_choice".to_string(), encode_tool_choice(tool_choice)?);
    }
    if let Some(output_config) = encode_output_config(
        options.output_effort,
        request.structured_output.as_ref(),
        options.task_budget,
    )? {
        body.insert("output_config".to_string(), output_config);
    }
    if let Some(metadata) = &options.metadata {
        body.insert(
            "metadata".to_string(),
            json!({ "user_id": metadata.user_id() }),
        );
    }
    if let Some(thinking) = options.thinking {
        body.insert("thinking".to_string(), encode_thinking(thinking));
    }
    if let Some(fallbacks) = &options.fallbacks {
        body.insert("fallbacks".to_string(), encode_fallbacks(fallbacks)?);
    }
    if let Some(service_tier) = options.service_tier {
        body.insert(
            "service_tier".to_string(),
            Value::String(service_tier.as_wire_str().to_string()),
        );
    }
    if let Some(cache_control) = options.cache_control {
        body.insert(
            "cache_control".to_string(),
            encode_cache_control(cache_control, cache_style)?,
        );
    }
    if let Some(speed) = options.speed {
        body.insert(
            "speed".to_string(),
            Value::String(speed.as_wire_str().to_string()),
        );
    }
    if let Some(inference_geo) = &options.inference_geo {
        body.insert(
            "inference_geo".to_string(),
            Value::String(inference_geo.as_str().to_string()),
        );
    }
    if let Some(container) = &options.container {
        body.insert("container".to_string(), encode_container(container)?);
    }
    if let Some(context_management) = &options.context_management {
        body.insert(
            "context_management".to_string(),
            serde_json::to_value(context_management).map_err(MessagesCodecError::JsonEncode)?,
        );
    }
    if let Some(mcp_servers) = &options.mcp_servers {
        body.insert(
            "mcp_servers".to_string(),
            serde_json::to_value(mcp_servers).map_err(MessagesCodecError::JsonEncode)?,
        );
    }
    body.extend(options.extra.clone());
    Ok(Value::Object(body))
}

fn encode_prompt(
    scope: Option<&ProviderScope>,
    messages: &[Message],
    resolver: &dyn MessagesAnnotationResolver,
    cache_style: CacheControlWireStyle,
    video_input: bool,
    mid_conversation_system: MidConversationSystemEncoding,
) -> Result<(Vec<Value>, Vec<Value>), MessagesCodecError> {
    let mut system = Vec::new();
    let mut encoded_messages = Vec::new();
    let mut conversation_started = false;

    for message in messages {
        match message.role() {
            MessageRole::System => {
                let placement = if conversation_started {
                    SystemMessagePlacement::Conversation
                } else {
                    SystemMessagePlacement::Preamble
                };
                if placement == SystemMessagePlacement::Conversation
                    && mid_conversation_system == MidConversationSystemEncoding::Unsupported
                {
                    return Err(MessagesCodecError::Unsupported {
                        feature: "mid-conversation system messages",
                    });
                }
                let blocks = encode_system_message(message, resolver, placement, cache_style)?;
                if placement == SystemMessagePlacement::Conversation {
                    encoded_messages.push(json!({ "role": "system", "content": blocks }));
                } else {
                    system.extend(blocks);
                }
            }
            MessageRole::Developer => {
                return Err(MessagesCodecError::Unsupported {
                    feature: "developer-role messages",
                });
            }
            MessageRole::User | MessageRole::Assistant | MessageRole::Tool => {
                conversation_started = true;
                let (role, content) = encode_conversation_message(
                    scope,
                    message,
                    resolver,
                    cache_style,
                    video_input,
                )?;
                merge_wire_message(&mut encoded_messages, role, content)?;
            }
            _ => {
                return Err(MessagesCodecError::Unsupported {
                    feature: "this message role",
                });
            }
        }
    }

    Ok((system, encoded_messages))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SystemMessagePlacement {
    Preamble,
    Conversation,
}

fn encode_system_message(
    message: &Message,
    resolver: &dyn MessagesAnnotationResolver,
    placement: SystemMessagePlacement,
    cache_style: CacheControlWireStyle,
) -> Result<Vec<Value>, MessagesCodecError> {
    let mut blocks = Vec::with_capacity(message.content().len());
    for part in message.content() {
        let content_options = resolver.resolve_content(part.annotations())?;
        if content_options.file().is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.file",
                reason: "must appear in a user message",
            });
        }
        if let Some(tool_change) = content_options.tool_change() {
            if placement != SystemMessagePlacement::Conversation {
                return Err(MessagesCodecError::InvalidOption {
                    field: "messages.tool_change",
                    reason: "must appear after at least one conversational message",
                });
            }
            if content_options.cache_control().is_some() {
                return Err(MessagesCodecError::ConflictingCacheAnnotation);
            }
            if !matches!(part.content(), ContentPart::Text { text } if text.is_empty()) {
                return Err(MessagesCodecError::InvalidOption {
                    field: "messages.tool_change.anchor",
                    reason: "must use the empty text anchor created by the provider helper",
                });
            }
            blocks.push(encode_mid_conversation_tool_change(tool_change)?);
            continue;
        }
        let ContentPart::Text { text } = part.content() else {
            return Err(MessagesCodecError::Unsupported {
                feature: "non-text system content",
            });
        };
        let mut block = json!({ "type": "text", "text": text });
        if let Some(cache_control) = content_options.cache_control() {
            apply_cache_control(&mut block, cache_control, cache_style)?;
        }
        blocks.push(block);
    }
    apply_message_options(
        resolver.resolve_message(message.annotations())?,
        &mut blocks,
        cache_style,
    )?;
    if blocks.is_empty() {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages",
            reason: "system messages must not be empty",
        });
    }
    Ok(blocks)
}

fn encode_mid_conversation_tool_change(
    change: &MidConversationToolChange,
) -> Result<Value, MessagesCodecError> {
    change.validate()?;
    let (kind, reference) = match change.kind() {
        MidConversationToolChangeKind::ToolAddition { tool } => ("tool_addition", tool),
        MidConversationToolChangeKind::ToolRemoval { tool } => ("tool_removal", tool),
    };
    Ok(json!({
        "type": kind,
        "tool": encode_anthropic_tool_reference(reference),
    }))
}

fn encode_anthropic_tool_reference(reference: &super::AnthropicToolReference) -> Value {
    match reference.kind() {
        AnthropicToolReferenceKind::Tool { name } => {
            json!({ "type": "tool_reference", "name": name })
        }
        AnthropicToolReferenceKind::McpTool { server_name, name } => json!({
            "type": "mcp_tool_reference",
            "server_name": server_name,
            "name": name,
        }),
        AnthropicToolReferenceKind::McpToolset { server_name } => json!({
            "type": "mcp_toolset_reference",
            "server_name": server_name,
        }),
    }
}

fn validate_tool_change_references(
    messages: &[Value],
    tools: &[Value],
) -> Result<(), MessagesCodecError> {
    for block in messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|block| {
            block
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| matches!(kind, "tool_addition" | "tool_removal"))
        })
    {
        let reference = block.get("tool").and_then(Value::as_object).ok_or(
            MessagesCodecError::ProtocolViolation {
                reason: "internal tool-change projection omitted its reference",
            },
        )?;
        let reference_type = reference.get("type").and_then(Value::as_str).ok_or(
            MessagesCodecError::ProtocolViolation {
                reason: "internal tool-change reference omitted its type",
            },
        )?;
        let declared = match reference_type {
            "tool_reference" => reference
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| {
                    tools
                        .iter()
                        .any(|tool| tool.get("name").and_then(Value::as_str) == Some(name))
                }),
            "mcp_tool_reference" | "mcp_toolset_reference" => reference
                .get("server_name")
                .and_then(Value::as_str)
                .is_some_and(|server_name| {
                    tools.iter().any(|tool| {
                        tool.get("mcp_server_name").and_then(Value::as_str) == Some(server_name)
                    })
                }),
            _ => false,
        };
        if !declared {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.tool_change.tool",
                reason: "must reference a tool or MCP toolset declared in the request",
            });
        }
    }
    Ok(())
}

fn encode_conversation_message(
    scope: Option<&ProviderScope>,
    message: &Message,
    resolver: &dyn MessagesAnnotationResolver,
    cache_style: CacheControlWireStyle,
    video_input: bool,
) -> Result<(&'static str, Vec<Value>), MessagesCodecError> {
    let role = match message.role() {
        MessageRole::User | MessageRole::Tool => "user",
        MessageRole::Assistant => "assistant",
        _ => {
            return Err(MessagesCodecError::Unsupported {
                feature: "this conversational message role",
            });
        }
    };
    let suppress_native_reasoning = validate_native_reasoning_projection(message, scope)?;
    let suppress_native_tool_calls = validate_native_tool_call_projection(message, scope)?;
    let mut blocks = Vec::with_capacity(message.content().len());
    for (content_index, part) in message.content().iter().enumerate() {
        let content_options = resolver.resolve_content(part.annotations())?;
        if content_options.tool_change().is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.tool_change",
                reason: "must be attached to a mid-conversation system message",
            });
        }
        if let Some(file) = content_options.file() {
            if message.role() != MessageRole::User {
                return Err(MessagesCodecError::InvalidOption {
                    field: "messages.file",
                    reason: "must appear in a user message",
                });
            }
            if !matches!(part.content(), ContentPart::Text { text } if text.is_empty()) {
                return Err(MessagesCodecError::InvalidOption {
                    field: "messages.file.anchor",
                    reason: "must use the empty text anchor created by the provider helper",
                });
            }
            let mut block = encode_file_block(file, scope)?;
            if let Some(cache_control) = content_options.cache_control() {
                apply_cache_control(&mut block, cache_control, cache_style)?;
            }
            blocks.push(block);
            continue;
        }
        if suppress_native_tool_calls.contains(&content_index) {
            if content_options.cache_control().is_some() {
                return Err(MessagesCodecError::Unsupported {
                    feature: "cache annotation on a suppressed native tool projection",
                });
            }
            continue;
        }
        let Some(mut block) = encode_content_part(
            message.role(),
            part.content(),
            suppress_native_reasoning,
            video_input,
            scope,
        )?
        else {
            if content_options.cache_control().is_some() {
                return Err(MessagesCodecError::Unsupported {
                    feature: "cache annotation on a suppressed reasoning projection",
                });
            }
            continue;
        };
        if let Some(cache_control) = content_options.cache_control() {
            if matches!(part.content(), ContentPart::ProviderOpaque(_)) {
                return Err(MessagesCodecError::Unsupported {
                    feature: "cache annotations on native opaque content",
                });
            }
            apply_cache_control(&mut block, cache_control, cache_style)?;
        }
        blocks.push(block);
    }
    apply_message_options(
        resolver.resolve_message(message.annotations())?,
        &mut blocks,
        cache_style,
    )?;
    if blocks.is_empty() {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages",
            reason: "conversational messages must contain encodable content",
        });
    }
    Ok((role, blocks))
}

fn encode_content_part(
    role: MessageRole,
    part: &ContentPart,
    suppress_native_reasoning: bool,
    video_input: bool,
    scope: Option<&ProviderScope>,
) -> Result<Option<Value>, MessagesCodecError> {
    match part {
        ContentPart::Text { text } if role != MessageRole::Tool => {
            Ok(Some(json!({ "type": "text", "text": text })))
        }
        ContentPart::Text { .. } => Err(MessagesCodecError::Unsupported {
            feature: "plain text in a tool-role message",
        }),
        ContentPart::Media(media) if role == MessageRole::User => {
            Ok(Some(encode_media(media, video_input)?))
        }
        ContentPart::Media(_) => Err(MessagesCodecError::Unsupported {
            feature: "media outside a user message",
        }),
        ContentPart::ToolCall(call) if role == MessageRole::Assistant => {
            if !matches!(call.owner(), ExecutionOwner::Local) {
                return Err(MessagesCodecError::Unsupported {
                    feature: "provider-owned tool calls without native Anthropic content",
                });
            }
            if call.id().trim().is_empty() || call.name().trim().is_empty() {
                return Err(MessagesCodecError::InvalidOption {
                    field: "messages.tool_use",
                    reason: "tool call ID and name must not be empty",
                });
            }
            Ok(Some(json!({
                "type": "tool_use",
                "id": call.id(),
                "name": call.name(),
                "input": call.arguments(),
            })))
        }
        ContentPart::ToolCall(_) => Err(MessagesCodecError::Unsupported {
            feature: "tool calls outside assistant messages",
        }),
        ContentPart::ToolResult(result)
            if matches!(role, MessageRole::User | MessageRole::Tool) =>
        {
            Ok(Some(encode_tool_result(result)?))
        }
        ContentPart::ToolResult(_) => Err(MessagesCodecError::Unsupported {
            feature: "tool results outside user or tool messages",
        }),
        ContentPart::Reasoning { .. } if suppress_native_reasoning => Ok(None),
        ContentPart::Reasoning { .. } => Err(MessagesCodecError::Unsupported {
            feature: "reasoning history without its native signed Anthropic block",
        }),
        ContentPart::ProviderOpaque(item) if role == MessageRole::Assistant => {
            Ok(Some(encode_opaque_item(item, scope)?))
        }
        ContentPart::ProviderOpaque(_) => Err(MessagesCodecError::Unsupported {
            feature: "native Anthropic content outside assistant messages",
        }),
        ContentPart::Refusal { .. } => Err(MessagesCodecError::Unsupported {
            feature: "portable refusal history",
        }),
        ContentPart::Citation(_) => Err(MessagesCodecError::Unsupported {
            feature: "standalone citation history",
        }),
        _ => Err(MessagesCodecError::Unsupported {
            feature: "this content part",
        }),
    }
}

fn validate_native_reasoning_projection(
    message: &Message,
    scope: Option<&ProviderScope>,
) -> Result<bool, MessagesCodecError> {
    let mut has_native_reasoning = false;
    let mut native_reasoning = Vec::new();
    let mut portable_reasoning = Vec::new();

    for part in message.content() {
        match part.content() {
            ContentPart::Reasoning { text } => portable_reasoning.push(text.as_str()),
            ContentPart::ProviderOpaque(item) if is_native_reasoning_block(item, scope) => {
                has_native_reasoning = true;
                if item.data().get("type").and_then(Value::as_str) == Some("thinking") {
                    let text = item
                        .data()
                        .get("thinking")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                        .ok_or(MessagesCodecError::ProtocolViolation {
                            reason: "native reasoning block omitted required replay state",
                        })?;
                    native_reasoning.push(text);
                }
            }
            _ => {}
        }
    }

    if has_native_reasoning
        && !portable_reasoning.is_empty()
        && portable_reasoning != native_reasoning
    {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages.reasoning",
            reason: "portable reasoning does not match its native signed replay block",
        });
    }

    Ok(has_native_reasoning)
}

fn encode_media(media: &MediaPart, video_input: bool) -> Result<Value, MessagesCodecError> {
    let block_type = if media.media_type.starts_with("image/") {
        "image"
    } else if media.media_type == "application/pdf" {
        "document"
    } else if media.media_type.starts_with("video/") && video_input {
        "video"
    } else {
        return Err(MessagesCodecError::Unsupported {
            feature: if media.media_type.starts_with("video/") {
                "video input is not enabled for this Messages dialect"
            } else {
                "media other than images, video, or PDF documents"
            },
        });
    };
    let source = match &media.data {
        MediaData::Bytes(bytes) => json!({
            "type": "base64",
            "media_type": media.media_type,
            "data": base64::engine::general_purpose::STANDARD.encode(bytes),
        }),
        MediaData::Url(url) => json!({ "type": "url", "url": url }),
        _ => {
            return Err(MessagesCodecError::Unsupported {
                feature: "this media source",
            });
        }
    };
    let mut block = Map::new();
    block.insert("type".to_string(), Value::String(block_type.to_string()));
    block.insert("source".to_string(), source);
    if block_type == "document"
        && let Some(name) = &media.name
    {
        block.insert("title".to_string(), Value::String(name.clone()));
    }
    Ok(Value::Object(block))
}

fn encode_file_block(
    file: &MessagesFileBlock,
    selected_scope: Option<&ProviderScope>,
) -> Result<Value, MessagesCodecError> {
    let selected_scope = selected_scope.ok_or(MessagesCodecError::Unsupported {
        feature: "Anthropic file references without an exact provider scope",
    })?;
    if !file_scope_is_complete(selected_scope) {
        return Err(MessagesCodecError::Unsupported {
            feature: "Anthropic file references without a complete replay scope",
        });
    }
    if !file
        .reference()
        .scope()
        .shares_replay_domain(selected_scope)
    {
        return Err(MessagesCodecError::Unsupported {
            feature: "Anthropic file references outside the current replay scope",
        });
    }
    validate_file_id(file.reference().file_id())?;

    let file_id = file.reference().file_id();
    Ok(match file {
        MessagesFileBlock::Image(_) => json!({
            "type": "image",
            "source": {"type": "file", "file_id": file_id},
        }),
        MessagesFileBlock::Document {
            title,
            context,
            citations,
            ..
        } => {
            let mut block = Map::new();
            block.insert("type".to_string(), Value::String("document".to_string()));
            block.insert(
                "source".to_string(),
                json!({"type": "file", "file_id": file_id}),
            );
            if let Some(title) = title {
                block.insert("title".to_string(), Value::String(title.clone()));
            }
            if let Some(context) = context {
                block.insert("context".to_string(), Value::String(context.clone()));
            }
            if let Some(enabled) = citations {
                block.insert("citations".to_string(), json!({"enabled": enabled}));
            }
            Value::Object(block)
        }
        MessagesFileBlock::ContainerUpload(_) => {
            json!({"type": "container_upload", "file_id": file_id})
        }
    })
}

fn validate_file_id(file_id: &str) -> Result<(), MessagesCodecError> {
    if file_id.is_empty() || file_id.len() > 512 || file_id.chars().any(char::is_control) {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages.file.file_id",
            reason: "must be non-empty, bounded, and free of control characters",
        });
    }
    Ok(())
}

fn encode_tool_result(result: &ToolResult) -> Result<Value, MessagesCodecError> {
    if result.call_id.trim().is_empty() {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages.tool_result.tool_use_id",
            reason: "must not be empty",
        });
    }
    let (content, is_error) = match &result.outcome {
        ToolOutcome::Success { value } => {
            let content = value.as_str().map_or_else(
                || serde_json::to_string(value).map_err(MessagesCodecError::JsonEncode),
                |value| Ok(value.to_string()),
            )?;
            (content, false)
        }
        ToolOutcome::Denied { reason } => (reason.clone(), true),
        ToolOutcome::ExecutionFailed { message, .. } => (message.clone(), true),
        ToolOutcome::Cancelled { reason } => (reason.clone(), true),
        _ => {
            return Err(MessagesCodecError::Unsupported {
                feature: "this tool outcome",
            });
        }
    };
    let mut block = Map::new();
    block.insert("type".to_string(), Value::String("tool_result".to_string()));
    block.insert(
        "tool_use_id".to_string(),
        Value::String(result.call_id.clone()),
    );
    block.insert("content".to_string(), Value::String(content));
    if is_error {
        block.insert("is_error".to_string(), Value::Bool(true));
    }
    Ok(Value::Object(block))
}

fn encode_opaque_item(
    item: &OpaqueProviderItem,
    scope: Option<&ProviderScope>,
) -> Result<Value, MessagesCodecError> {
    if scope.is_none_or(|scope| !item.provenance().matches_replay_target(scope))
        || item.kind() != OPAQUE_CONTENT_BLOCK_KIND
    {
        return Err(MessagesCodecError::Unsupported {
            feature: "opaque content outside the current Anthropic replay scope",
        });
    }
    let block = item.data();
    let kind =
        block
            .get("type")
            .and_then(Value::as_str)
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "native content block omitted its type",
            })?;
    if !matches!(
        kind,
        "thinking" | "redacted_thinking" | "compaction" | "tool_use"
    ) && !is_maintained_hosted_kind(kind)
    {
        return Err(MessagesCodecError::Unsupported {
            feature: "replay of this native Anthropic content block",
        });
    }
    if !block.is_object() {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "native content block was not an object",
        });
    }
    match kind {
        "thinking" | "redacted_thinking" => {
            let required_fields: &[&str] = match kind {
                "thinking" => &["thinking", "signature"],
                "redacted_thinking" => &["data"],
                _ => unreachable!("matched native reasoning block"),
            };
            if required_fields.iter().any(|field| {
                block
                    .get(*field)
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            }) {
                return Err(MessagesCodecError::ProtocolViolation {
                    reason: "native content block omitted required replay state",
                });
            }
        }
        "compaction" => {
            match block.get("content") {
                Some(Value::Null) => {}
                Some(Value::String(content)) if !content.is_empty() => {}
                _ => {
                    return Err(MessagesCodecError::ProtocolViolation {
                        reason: "native compaction block omitted valid replay content",
                    });
                }
            }
            if block
                .get("encrypted_content")
                .is_some_and(|value| !value.is_null() && !value.is_string())
            {
                return Err(MessagesCodecError::ProtocolViolation {
                    reason: "native compaction block contained invalid encrypted replay state",
                });
            }
        }
        "tool_use" => {
            let object = block
                .as_object()
                .expect("native content block was validated as an object");
            let caller = validate_caller(object)?;
            if caller.is_none() {
                return Err(MessagesCodecError::Unsupported {
                    feature: "replay of a native tool_use block without caller metadata",
                });
            }
            if !caller_is_replayable(caller) {
                return Err(MessagesCodecError::Unsupported {
                    feature: "replay of a native tool_use block with an unknown caller",
                });
            }
            validate_native_caller_tool_use(object)?;
            validate_complete_native_replay_scope(item, scope)?;
        }
        kind if is_maintained_hosted_kind(kind) => {
            let hosted =
                inspect_hosted_tool_value(block)?.ok_or(MessagesCodecError::ProtocolViolation {
                    reason: "maintained hosted-tool block could not be inspected",
                })?;
            if !caller_is_replayable(hosted.caller()) {
                return Err(MessagesCodecError::Unsupported {
                    feature: "replay of a native hosted-tool block with an unknown caller",
                });
            }
            validate_maintained_hosted_identity(item, hosted)?;
            validate_complete_native_replay_scope(item, scope)?;
        }
        _ => {
            return Err(MessagesCodecError::Unsupported {
                feature: "replay of this native Anthropic content block",
            });
        }
    }
    Ok(block.clone())
}

fn validate_native_tool_call_projection(
    message: &Message,
    scope: Option<&ProviderScope>,
) -> Result<BTreeSet<usize>, MessagesCodecError> {
    if message.role() != MessageRole::Assistant {
        return Ok(BTreeSet::new());
    }

    let local_calls = message
        .content()
        .iter()
        .enumerate()
        .filter_map(|(index, part)| match part.content() {
            ContentPart::ToolCall(call) if matches!(call.owner(), ExecutionOwner::Local) => {
                Some((index, call))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut suppressed = BTreeSet::new();

    for part in message.content() {
        let ContentPart::ProviderOpaque(item) = part.content() else {
            continue;
        };
        if item.kind() != OPAQUE_CONTENT_BLOCK_KIND
            || item.data().get("type").and_then(Value::as_str) != Some("tool_use")
        {
            continue;
        }
        let object = item
            .data()
            .as_object()
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "native tool_use replay block was not an object",
            })?;
        let caller = validate_caller(object)?;
        if caller.is_none() {
            continue;
        }
        if !caller_is_replayable(caller) {
            return Err(MessagesCodecError::Unsupported {
                feature: "replay of a native tool_use block with an unknown caller",
            });
        }
        validate_complete_native_replay_scope(item, scope)?;
        let (id, name, input) = validate_native_caller_tool_use(object)?;
        if item.item_id() != Some(id) {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.tool_use.id",
                reason: "native tool identity does not match its retained item ID",
            });
        }
        validate_caller_relation(item, caller)?;
        validate_relation_kinds(item, &["caller"])?;

        let matching = local_calls
            .iter()
            .filter(|(_, call)| call.id() == id)
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.tool_use",
                reason: "native caller tool replay requires exactly one matching local tool call",
            });
        }
        let &(index, call) = matching[0];
        if call.name() != name || call.arguments() != input {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.tool_use",
                reason: "native caller tool replay does not match the local tool call semantics",
            });
        }
        if !suppressed.insert(index) {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.tool_use",
                reason: "more than one native replay block referenced the same local tool call",
            });
        }
    }
    Ok(suppressed)
}

fn validate_maintained_hosted_identity(
    item: &OpaqueProviderItem,
    hosted: AnthropicHostedToolBlockRef<'_>,
) -> Result<(), MessagesCodecError> {
    match hosted {
        AnthropicHostedToolBlockRef::ServerToolUse(value) => {
            validate_retained_item_id(item, value.id())?;
            validate_caller_relation(item, value.caller())?;
            validate_relation_kinds(item, &["caller"])
        }
        AnthropicHostedToolBlockRef::McpToolUse(value) => {
            validate_retained_item_id(item, value.id())?;
            validate_caller_relation(item, value.caller())?;
            validate_relation_kinds(item, &["caller"])
        }
        AnthropicHostedToolBlockRef::Result(value) => {
            validate_single_relation(item, "related_item", value.tool_use_id())?;
            validate_caller_relation(item, value.caller())?;
            validate_relation_kinds(item, &["related_item", "caller"])
        }
    }
}

fn validate_retained_item_id(
    item: &OpaqueProviderItem,
    expected: &str,
) -> Result<(), MessagesCodecError> {
    if item.item_id() != Some(expected) {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages.native.item_id",
            reason: "native hosted-tool identity does not match its retained item ID",
        });
    }
    Ok(())
}

fn validate_caller_relation(
    item: &OpaqueProviderItem,
    caller: Option<&Value>,
) -> Result<(), MessagesCodecError> {
    let caller_id = caller
        .and_then(Value::as_object)
        .and_then(|object| object.get("tool_id"))
        .and_then(Value::as_str);
    match caller_id {
        Some(caller_id) => validate_single_relation(item, "caller", caller_id),
        None if item
            .relations()
            .iter()
            .any(|relation| relation.kind() == "caller") =>
        {
            Err(MessagesCodecError::InvalidOption {
                field: "messages.native.relations",
                reason: "native hosted-tool caller relation was absent from the wire block",
            })
        }
        None => Ok(()),
    }
}

fn validate_relation_kinds(
    item: &OpaqueProviderItem,
    allowed: &[&str],
) -> Result<(), MessagesCodecError> {
    if item
        .relations()
        .iter()
        .any(|relation| !allowed.contains(&relation.kind()))
    {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages.native.relations",
            reason: "native hosted-tool item contained an unsupported identity relation",
        });
    }
    Ok(())
}

fn validate_single_relation(
    item: &OpaqueProviderItem,
    kind: &str,
    expected: &str,
) -> Result<(), MessagesCodecError> {
    let matching = item
        .relations()
        .iter()
        .filter(|relation| relation.kind() == kind)
        .collect::<Vec<_>>();
    if matching.len() != 1 || matching[0].target_id() != expected {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages.native.relations",
            reason: "native hosted-tool relation does not match its retained wire identity",
        });
    }
    Ok(())
}

fn validate_native_caller_tool_use(
    object: &Map<String, Value>,
) -> Result<(&str, &str, &Value), MessagesCodecError> {
    let id = required_native_identifier(object, "id")?;
    let name = required_native_identifier(object, "name")?;
    let input = object
        .get("input")
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "native caller tool_use omitted its input",
        })?;
    Ok((id, name, input))
}

fn validate_complete_native_replay_scope(
    item: &OpaqueProviderItem,
    selected_scope: Option<&ProviderScope>,
) -> Result<(), MessagesCodecError> {
    let selected_scope = selected_scope.ok_or(MessagesCodecError::Unsupported {
        feature: "native Anthropic hosted-tool replay without an exact provider scope",
    })?;
    let item_scope = item.provenance().scope();
    let complete = |scope: &ProviderScope| {
        scope.platform().is_some()
            && scope.protocol().is_some()
            && scope.api_mode().is_some()
            && scope
                .replay_domain()
                .and_then(|domain| domain.caller_scope())
                .is_some()
    };
    if !complete(selected_scope)
        || !complete(item_scope)
        || !item.provenance().matches_replay_target(selected_scope)
    {
        return Err(MessagesCodecError::Unsupported {
            feature: "native Anthropic hosted-tool content outside the complete replay scope",
        });
    }
    Ok(())
}

fn required_native_identifier<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, MessagesCodecError> {
    let value =
        object
            .get(field)
            .and_then(Value::as_str)
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "native caller tool_use omitted a required identifier",
            })?;
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "native caller tool_use contained an invalid identifier",
        });
    }
    Ok(value)
}

fn is_native_reasoning_block(item: &OpaqueProviderItem, scope: Option<&ProviderScope>) -> bool {
    scope.is_some_and(|scope| item.provenance().matches_replay_target(scope))
        && item.kind() == OPAQUE_CONTENT_BLOCK_KIND
        && item
            .data()
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, "thinking" | "redacted_thinking"))
}

fn merge_wire_message(
    messages: &mut Vec<Value>,
    role: &'static str,
    content: Vec<Value>,
) -> Result<(), MessagesCodecError> {
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
    {
        let existing = last
            .get_mut("content")
            .and_then(Value::as_array_mut)
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "internal message projection produced invalid content",
            })?;
        existing.extend(content);
        return Ok(());
    }
    messages.push(json!({ "role": role, "content": content }));
    Ok(())
}

fn encode_tools(
    tools: &[ToolSpec],
    resolver: &dyn MessagesAnnotationResolver,
    cache_style: CacheControlWireStyle,
) -> Result<Vec<Value>, MessagesCodecError> {
    tools
        .iter()
        .map(|tool| {
            let tool_options = resolver.resolve_tool(tool.annotations())?;
            let anthropic_tool = tool_options.anthropic_tool();
            validate_tool_node_options(&tool_options, anthropic_tool)?;
            if let Some(anthropic_tool) = anthropic_tool {
                validate_anthropic_tool_anchor(tool, anthropic_tool)?;
                return encode_anthropic_tool(anthropic_tool, &tool_options, cache_style);
            }

            encode_function_tool(tool, &tool_options, cache_style)
        })
        .collect()
}

fn validate_mcp_and_skills_bindings(
    tools: &[Value],
    messages: &[Value],
    container: Option<&MessagesContainer>,
    mcp_servers: Option<&[McpServer]>,
) -> Result<(), MessagesCodecError> {
    let mut server_names = BTreeSet::new();
    for server in mcp_servers.unwrap_or_default() {
        if !server_names.insert(server.name()) {
            return Err(MessagesCodecError::InvalidOption {
                field: "mcp_servers[].name",
                reason: "must be unique within one request",
            });
        }
    }

    let mut toolsets = BTreeMap::<&str, usize>::new();
    let mut has_code_execution = false;
    for tool in tools {
        let Some(kind) = tool.get("type").and_then(Value::as_str) else {
            continue;
        };
        has_code_execution |= kind.starts_with("code_execution_");
        if kind != "mcp_toolset" {
            continue;
        }
        let server_name = tool.get("mcp_server_name").and_then(Value::as_str).ok_or(
            MessagesCodecError::ProtocolViolation {
                reason: "internal MCP toolset projection omitted its server name",
            },
        )?;
        *toolsets.entry(server_name).or_default() += 1;
    }

    if toolsets.values().any(|count| *count != 1) {
        return Err(MessagesCodecError::InvalidOption {
            field: "tools.mcp_toolset.mcp_server_name",
            reason: "each MCP server must have exactly one MCP toolset",
        });
    }
    if server_names.len() != toolsets.len()
        || server_names
            .iter()
            .any(|server_name| !toolsets.contains_key(*server_name))
    {
        return Err(MessagesCodecError::InvalidOption {
            field: "mcp_servers",
            reason: "servers and MCP toolsets must reference each other exactly once",
        });
    }
    if container.is_some_and(MessagesContainer::has_skills) && !has_code_execution {
        return Err(MessagesCodecError::InvalidOption {
            field: "container.skills",
            reason: "requires an Anthropic code-execution tool",
        });
    }
    let has_container_upload = messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("container_upload"));
    if has_container_upload && !has_code_execution {
        return Err(MessagesCodecError::InvalidOption {
            field: "messages.container_upload",
            reason: "requires an Anthropic code-execution tool",
        });
    }
    Ok(())
}

fn encode_function_tool(
    tool: &ToolSpec,
    options: &super::ToolNodeOptions,
    cache_style: CacheControlWireStyle,
) -> Result<Value, MessagesCodecError> {
    if !tool.input_schema().is_object() {
        return Err(MessagesCodecError::Unsupported {
            feature: "boolean function-tool JSON Schemas",
        });
    }
    let mut encoded = Map::new();
    encoded.insert("name".to_string(), Value::String(tool.name().to_string()));
    if let Some(description) = tool.description() {
        encoded.insert(
            "description".to_string(),
            Value::String(description.to_string()),
        );
    }
    encoded.insert("input_schema".to_string(), tool.input_schema().clone());
    apply_common_tool_options(&mut encoded, options, cache_style)?;
    Ok(Value::Object(encoded))
}

fn validate_anthropic_tool_anchor(
    tool: &ToolSpec,
    anthropic_tool: &AnthropicTool,
) -> Result<(), MessagesCodecError> {
    if tool.name() != anthropic_tool.canonical_name() {
        return Err(MessagesCodecError::InvalidOption {
            field: "tools.anthropic_tool.anchor.name",
            reason: "must match the Anthropic tool's canonical name",
        });
    }
    if tool.description().is_some() {
        return Err(MessagesCodecError::InvalidOption {
            field: "tools.anthropic_tool.anchor.description",
            reason: "must be omitted for an Anthropic-defined tool anchor",
        });
    }
    if tool.input_schema() != &anthropic_tool_anchor_schema() {
        return Err(MessagesCodecError::InvalidOption {
            field: "tools.anthropic_tool.anchor.input_schema",
            reason: "must use anthropic_tool_anchor_schema()",
        });
    }
    Ok(())
}

fn encode_anthropic_tool(
    anthropic_tool: &AnthropicTool,
    options: &super::ToolNodeOptions,
    cache_style: CacheControlWireStyle,
) -> Result<Value, MessagesCodecError> {
    let mut encoded = Map::new();
    encoded.insert(
        "type".to_string(),
        Value::String(anthropic_tool.wire_type().to_string()),
    );

    match anthropic_tool {
        AnthropicTool::McpToolset(mcp) => {
            encoded.insert(
                "mcp_server_name".to_string(),
                Value::String(mcp.server_name().to_string()),
            );
            if !mcp.configs().is_empty() {
                let configs = mcp
                    .configs()
                    .iter()
                    .map(|(name, config)| {
                        (name.clone(), Value::Object(encode_mcp_tool_config(*config)))
                    })
                    .collect();
                encoded.insert("configs".to_string(), Value::Object(configs));
            }
            if let Some(default_config) = mcp.default_config() {
                encoded.insert(
                    "default_config".to_string(),
                    Value::Object(encode_mcp_tool_config(default_config)),
                );
            }
        }
        AnthropicTool::WebSearch20260318(web_search) => {
            insert_anthropic_tool_name(&mut encoded, anthropic_tool);
            insert_string_list(
                &mut encoded,
                "allowed_domains",
                web_search.allowed_domains(),
            );
            insert_string_list(
                &mut encoded,
                "blocked_domains",
                web_search.blocked_domains(),
            );
            insert_u32(&mut encoded, "max_uses", web_search.max_uses());
            if let Some(response_inclusion) = web_search.response_inclusion() {
                encoded.insert(
                    "response_inclusion".to_string(),
                    Value::String(response_inclusion.as_wire_str().to_string()),
                );
            }
            if let Some(user_location) = web_search.user_location() {
                encoded.insert(
                    "user_location".to_string(),
                    Value::Object(encode_user_location(user_location)),
                );
            }
        }
        AnthropicTool::WebFetch20260318(web_fetch) => {
            insert_anthropic_tool_name(&mut encoded, anthropic_tool);
            insert_string_list(&mut encoded, "allowed_domains", web_fetch.allowed_domains());
            insert_string_list(&mut encoded, "blocked_domains", web_fetch.blocked_domains());
            if let Some(citations) = web_fetch.citations() {
                encoded.insert("citations".to_string(), json!({ "enabled": citations }));
            }
            insert_u32(
                &mut encoded,
                "max_content_tokens",
                web_fetch.max_content_tokens(),
            );
            insert_u32(&mut encoded, "max_uses", web_fetch.max_uses());
            if let Some(response_inclusion) = web_fetch.response_inclusion() {
                encoded.insert(
                    "response_inclusion".to_string(),
                    Value::String(response_inclusion.as_wire_str().to_string()),
                );
            }
            if let Some(use_cache) = web_fetch.use_cache() {
                encoded.insert("use_cache".to_string(), Value::Bool(use_cache));
            }
        }
        AnthropicTool::Advisor20260301(advisor) => {
            insert_anthropic_tool_name(&mut encoded, anthropic_tool);
            encoded.insert(
                "model".to_string(),
                Value::String(advisor.model_name().to_string()),
            );
            if let Some(max_tokens) = advisor.max_tokens() {
                encoded.insert("max_tokens".to_string(), Value::from(max_tokens));
            }
            insert_u32(&mut encoded, "max_uses", advisor.max_uses());
            if let Some(caching) = advisor.caching() {
                encoded.insert(
                    "caching".to_string(),
                    encode_cache_control(caching, cache_style)?,
                );
            }
        }
        AnthropicTool::TextEditor20250728(text_editor) => {
            insert_anthropic_tool_name(&mut encoded, anthropic_tool);
            insert_u32(&mut encoded, "max_characters", text_editor.max_characters());
        }
        AnthropicTool::Computer20251124(computer) => {
            insert_anthropic_tool_name(&mut encoded, anthropic_tool);
            encoded.insert(
                "display_width_px".to_string(),
                Value::from(computer.display_width_px()),
            );
            encoded.insert(
                "display_height_px".to_string(),
                Value::from(computer.display_height_px()),
            );
            insert_u32(&mut encoded, "display_number", computer.display_number());
            encoded.insert(
                "enable_zoom".to_string(),
                Value::Bool(computer.enable_zoom()),
            );
        }
        AnthropicTool::CodeExecution20260521
        | AnthropicTool::ToolSearchRegex20251119
        | AnthropicTool::ToolSearchBm25V20251119
        | AnthropicTool::Memory20250818
        | AnthropicTool::Bash20250124 => insert_anthropic_tool_name(&mut encoded, anthropic_tool),
    }

    apply_common_tool_options(&mut encoded, options, cache_style)?;
    Ok(Value::Object(encoded))
}

fn insert_anthropic_tool_name(encoded: &mut Map<String, Value>, anthropic_tool: &AnthropicTool) {
    encoded.insert(
        "name".to_string(),
        Value::String(anthropic_tool.canonical_name().to_string()),
    );
}

fn insert_string_list(encoded: &mut Map<String, Value>, field: &str, values: Option<&[String]>) {
    if let Some(values) = values {
        encoded.insert(
            field.to_string(),
            Value::Array(values.iter().cloned().map(Value::String).collect()),
        );
    }
}

fn insert_u32(encoded: &mut Map<String, Value>, field: &str, value: Option<u32>) {
    if let Some(value) = value {
        encoded.insert(field.to_string(), Value::from(value));
    }
}

fn encode_user_location(location: &UserLocation) -> Map<String, Value> {
    let mut encoded = Map::new();
    encoded.insert("type".to_string(), Value::String("approximate".to_string()));
    for (field, value) in [
        ("city", location.city()),
        ("country", location.country()),
        ("region", location.region()),
        ("timezone", location.timezone()),
    ] {
        if let Some(value) = value {
            encoded.insert(field.to_string(), Value::String(value.to_string()));
        }
    }
    encoded
}

fn encode_mcp_tool_config(config: McpToolConfig) -> Map<String, Value> {
    let mut encoded = Map::new();
    if let Some(enabled) = config.enabled() {
        encoded.insert("enabled".to_string(), Value::Bool(enabled));
    }
    if let Some(defer_loading) = config.defer_loading() {
        encoded.insert("defer_loading".to_string(), Value::Bool(defer_loading));
    }
    encoded
}

fn apply_common_tool_options(
    encoded: &mut Map<String, Value>,
    options: &super::ToolNodeOptions,
    cache_style: CacheControlWireStyle,
) -> Result<(), MessagesCodecError> {
    if !options.allowed_callers().is_empty() {
        encoded.insert(
            "allowed_callers".to_string(),
            Value::Array(
                options
                    .allowed_callers()
                    .iter()
                    .map(|caller| Value::String(caller.as_wire_str().to_string()))
                    .collect(),
            ),
        );
    }
    if let Some(strict) = options.strict() {
        encoded.insert("strict".to_string(), Value::Bool(strict));
    }
    if let Some(defer_loading) = options.defer_loading() {
        encoded.insert("defer_loading".to_string(), Value::Bool(defer_loading));
    }
    if let Some(cache_control) = options.cache_control() {
        if encoded.contains_key("cache_control") {
            return Err(MessagesCodecError::ConflictingCacheAnnotation);
        }
        encoded.insert(
            "cache_control".to_string(),
            encode_cache_control(cache_control, cache_style)?,
        );
    }
    Ok(())
}

fn encode_tool_choice(choice: &ToolChoice) -> Result<Value, MessagesCodecError> {
    match choice {
        ToolChoice::Auto => Ok(json!({ "type": "auto" })),
        ToolChoice::None => Ok(json!({ "type": "none" })),
        ToolChoice::Required => Ok(json!({ "type": "any" })),
        ToolChoice::Named { name } => Ok(json!({ "type": "tool", "name": name })),
        _ => Err(MessagesCodecError::Unsupported {
            feature: "this tool-choice mode",
        }),
    }
}

fn encode_thinking(thinking: ThinkingConfig) -> Value {
    let mut encoded = match thinking {
        ThinkingConfig::Disabled => return json!({ "type": "disabled" }),
        ThinkingConfig::Enabled { budget_tokens, .. } => {
            let mut encoded = Map::new();
            encoded.insert("type".to_string(), Value::String("enabled".to_string()));
            encoded.insert("budget_tokens".to_string(), Value::from(budget_tokens));
            encoded
        }
        ThinkingConfig::Adaptive { .. } => {
            let mut encoded = Map::new();
            encoded.insert("type".to_string(), Value::String("adaptive".to_string()));
            encoded
        }
    };
    if let Some(display) = thinking.display() {
        encoded.insert(
            "display".to_string(),
            Value::String(display.as_wire_str().to_string()),
        );
    }
    Value::Object(encoded)
}

fn encode_output_config(
    effort: Option<super::OutputEffort>,
    format: Option<&siumai_core::StructuredOutputSpec>,
    task_budget: Option<TokenTaskBudget>,
) -> Result<Option<Value>, MessagesCodecError> {
    let mut encoded = Map::new();
    if let Some(effort) = effort {
        encoded.insert(
            "effort".to_string(),
            Value::String(effort.as_wire_str().to_string()),
        );
    }
    if let Some(format) = format {
        encoded.insert("format".to_string(), encode_structured_output(format)?);
    }
    if let Some(task_budget) = task_budget {
        let mut budget = Map::new();
        budget.insert("type".to_string(), Value::String("tokens".to_string()));
        budget.insert("total".to_string(), Value::from(task_budget.total()));
        if let Some(remaining) = task_budget.remaining() {
            budget.insert("remaining".to_string(), Value::from(remaining));
        }
        encoded.insert("task_budget".to_string(), Value::Object(budget));
    }
    Ok((!encoded.is_empty()).then_some(Value::Object(encoded)))
}

fn encode_container(container: &MessagesContainer) -> Result<Value, MessagesCodecError> {
    serde_json::to_value(container).map_err(MessagesCodecError::JsonEncode)
}

fn encode_structured_output(
    output: &siumai_core::StructuredOutputSpec,
) -> Result<Value, MessagesCodecError> {
    if !output.strict {
        return Err(MessagesCodecError::Unsupported {
            feature: "non-strict structured output",
        });
    }
    if !output.schema.is_object() {
        return Err(MessagesCodecError::Unsupported {
            feature: "boolean structured-output schemas",
        });
    }
    Ok(json!({
        "type": "json_schema",
        "schema": output.schema,
    }))
}

fn encode_fallbacks(fallbacks: &ServerFallbacks) -> Result<Value, MessagesCodecError> {
    match fallbacks {
        ServerFallbacks::Default => Ok(Value::String("default".to_string())),
        ServerFallbacks::Explicit(fallbacks) => fallbacks
            .iter()
            .map(encode_fallback)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
    }
}

fn encode_fallback(fallback: &ServerFallback) -> Result<Value, MessagesCodecError> {
    let mut encoded = Map::new();
    encoded.insert(
        "model".to_string(),
        Value::String(fallback.model_name().to_string()),
    );
    if let Some(max_tokens) = fallback.max_tokens() {
        encoded.insert("max_tokens".to_string(), Value::from(max_tokens));
    }
    if let Some(thinking) = fallback.thinking() {
        encoded.insert("thinking".to_string(), encode_thinking(thinking));
    }
    if let Some(output_config) = fallback.output_config()
        && let Some(output_config) =
            encode_output_config(output_config.effort(), output_config.format(), None)?
    {
        encoded.insert("output_config".to_string(), output_config);
    }
    if let Some(speed) = fallback.speed() {
        encoded.insert(
            "speed".to_string(),
            Value::String(speed.as_wire_str().to_string()),
        );
    }
    Ok(Value::Object(encoded))
}

fn apply_message_options(
    options: super::MessageNodeOptions,
    blocks: &mut [Value],
    cache_style: CacheControlWireStyle,
) -> Result<(), MessagesCodecError> {
    let Some(cache_control) = options.cache_control() else {
        return Ok(());
    };
    let last = blocks.last_mut().ok_or(MessagesCodecError::InvalidOption {
        field: "messages",
        reason: "a cache-annotated message must contain an encodable block",
    })?;
    if last
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            matches!(
                kind,
                "thinking" | "redacted_thinking" | "tool_addition" | "tool_removal"
            )
        })
    {
        return Err(MessagesCodecError::Unsupported {
            feature: "message-level caching after a native thinking or tool-change block",
        });
    }
    apply_cache_control(last, cache_control, cache_style)
}

fn apply_cache_control(
    block: &mut Value,
    cache_control: CacheControl,
    cache_style: CacheControlWireStyle,
) -> Result<(), MessagesCodecError> {
    let object = block
        .as_object_mut()
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "cacheable wire block was not an object",
        })?;
    if object.contains_key("cache_control") {
        return Err(MessagesCodecError::ConflictingCacheAnnotation);
    }
    object.insert(
        "cache_control".to_string(),
        encode_cache_control(cache_control, cache_style)?,
    );
    Ok(())
}

fn encode_cache_control(
    cache_control: CacheControl,
    cache_style: CacheControlWireStyle,
) -> Result<Value, MessagesCodecError> {
    match (cache_style, cache_control.ttl()) {
        (CacheControlWireStyle::ExplicitTtl, ttl) => Ok(json!({
            "type": "ephemeral",
            "ttl": ttl.as_wire_str(),
        })),
        (CacheControlWireStyle::FiveMinutesImplicit, CacheTtl::FiveMinutes) => {
            Ok(json!({ "type": "ephemeral" }))
        }
        (CacheControlWireStyle::FiveMinutesImplicit, CacheTtl::OneHour) => {
            Err(MessagesCodecError::Unsupported {
                feature: "one-hour prompt caching when the Messages dialect omits cache TTL",
            })
        }
    }
}

fn validate_cache_breakpoints(
    tools: &[Value],
    system: &[Value],
    messages: &[Value],
    automatic_cache: Option<CacheControl>,
    cache_style: CacheControlWireStyle,
) -> Result<(), MessagesCodecError> {
    let mut ttls = Vec::new();
    collect_cache_ttls(tools.iter(), &mut ttls, cache_style)?;
    collect_cache_ttls(system.iter(), &mut ttls, cache_style)?;
    for message in messages {
        let content = message.get("content").and_then(Value::as_array).ok_or(
            MessagesCodecError::ProtocolViolation {
                reason: "internal message projection produced invalid content",
            },
        )?;
        collect_cache_ttls(content.iter(), &mut ttls, cache_style)?;
    }
    if let Some(automatic_cache) = automatic_cache
        && let Some(target) = last_cacheable_block(tools, system, messages)
    {
        let explicit_ttl = target
            .get("cache_control")
            .map(|value| decode_cache_ttl(value, cache_style))
            .transpose()?;
        match explicit_ttl {
            Some(explicit_ttl) if explicit_ttl == automatic_cache.ttl() => {}
            Some(_) => return Err(MessagesCodecError::ConflictingCacheAnnotation),
            None => ttls.push(automatic_cache.ttl()),
        }
    }
    if ttls.len() > MAX_CACHE_BREAKPOINTS {
        return Err(MessagesCodecError::TooManyCacheBreakpoints {
            actual: ttls.len(),
            maximum: MAX_CACHE_BREAKPOINTS,
        });
    }
    let mut saw_five_minutes = false;
    for ttl in ttls {
        match ttl {
            CacheTtl::FiveMinutes => saw_five_minutes = true,
            CacheTtl::OneHour if saw_five_minutes => {
                return Err(MessagesCodecError::InvalidCacheTtlOrder);
            }
            CacheTtl::OneHour => {}
        }
    }
    Ok(())
}

fn last_cacheable_block<'a>(
    tools: &'a [Value],
    system: &'a [Value],
    messages: &'a [Value],
) -> Option<&'a Value> {
    for message in messages.iter().rev() {
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        if let Some(block) = content
            .iter()
            .rev()
            .find(|block| is_automatic_cache_target(block))
        {
            return Some(block);
        }
    }
    system
        .iter()
        .rev()
        .find(|block| is_automatic_cache_target(block))
        .or_else(|| {
            tools
                .iter()
                .rev()
                .find(|block| is_automatic_cache_target(block))
        })
}

fn is_automatic_cache_target(block: &Value) -> bool {
    block.as_object().is_some()
        && !block
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| {
                matches!(
                    kind,
                    "thinking" | "redacted_thinking" | "tool_addition" | "tool_removal"
                )
            })
}

fn collect_cache_ttls<'a>(
    blocks: impl IntoIterator<Item = &'a Value>,
    ttls: &mut Vec<CacheTtl>,
    cache_style: CacheControlWireStyle,
) -> Result<(), MessagesCodecError> {
    for block in blocks {
        let Some(cache_control) = block.get("cache_control") else {
            continue;
        };
        ttls.push(decode_cache_ttl(cache_control, cache_style)?);
    }
    Ok(())
}

fn decode_cache_ttl(
    cache_control: &Value,
    cache_style: CacheControlWireStyle,
) -> Result<CacheTtl, MessagesCodecError> {
    let ttl = cache_control.get("ttl").and_then(Value::as_str);
    Ok(match ttl {
        Some("5m") => CacheTtl::FiveMinutes,
        Some("1h") => CacheTtl::OneHour,
        None if cache_style == CacheControlWireStyle::FiveMinutesImplicit => CacheTtl::FiveMinutes,
        None => {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "cache-control block omitted its TTL",
            });
        }
        _ => {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "cache-control block used an unknown TTL",
            });
        }
    })
}
