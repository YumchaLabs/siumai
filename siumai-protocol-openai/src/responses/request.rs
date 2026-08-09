//! Responses request encoding with strict native-history replay.

use std::collections::{BTreeMap, BTreeSet};

use base64::Engine as _;
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, DEFAULT_TOOL_INPUT_BYTE_LIMIT, Error, ErrorKind, LanguageRequest, MediaData,
    Message, MessageRole, ModelId, OpaqueProviderItem, ProviderScope, StructuredOutputSpec,
    ToolChoice, ToolOutcome, ToolResult,
};

use crate::{NoPromptCacheAnnotations, PromptCacheAnnotationResolver};

use super::wire::{OutputContentPart, OutputItem};
use super::{API_MODE_ID, OPENAI_RESPONSES_OPAQUE_KIND, OPENAI_RESPONSES_PROTOCOL};

/// Internal merge key accepted from provider-owned typed options.
pub const TEXT_VERBOSITY_OPTION: &str = "text_verbosity";

/// Media kinds accepted by one Responses-compatible wire dialect.
///
/// Native OpenAI Responses accepts images and generic files. Compatible
/// providers may opt into video input or disable generic file input without
/// gaining authority over endpoint, authentication, or transport behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponsesMediaDialect {
    image_input: bool,
    video_input: bool,
    file_input: bool,
}

impl ResponsesMediaDialect {
    /// Native OpenAI Responses media behavior.
    pub const fn native() -> Self {
        Self {
            image_input: true,
            video_input: false,
            file_input: true,
        }
    }

    pub const fn with_image_input(mut self, enabled: bool) -> Self {
        self.image_input = enabled;
        self
    }

    pub const fn with_video_input(mut self, enabled: bool) -> Self {
        self.video_input = enabled;
        self
    }

    pub const fn with_file_input(mut self, enabled: bool) -> Self {
        self.file_input = enabled;
        self
    }
}

impl Default for ResponsesMediaDialect {
    fn default() -> Self {
        Self::native()
    }
}

/// OpenAI execution paths allowed to invoke one function tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum FunctionToolCaller {
    Direct,
    Programmatic,
}

impl FunctionToolCaller {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Programmatic => "programmatic",
        }
    }
}

/// Responses-only wire controls for one portable function tool.
#[derive(Debug, Clone, Default)]
pub struct FunctionToolEncodingOptions {
    strict: Option<bool>,
    defer_loading: Option<bool>,
    allowed_callers: BTreeSet<FunctionToolCaller>,
    output_schema: Option<Value>,
}

impl FunctionToolEncodingOptions {
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = Some(defer_loading);
        self
    }

    pub fn with_allowed_caller(mut self, caller: FunctionToolCaller) -> Self {
        self.allowed_callers.insert(caller);
        self
    }

    pub fn with_output_schema(mut self, output_schema: Value) -> Self {
        self.output_schema = Some(output_schema);
        self
    }

    fn validate(&self) -> Result<(), Error> {
        if self
            .output_schema
            .as_ref()
            .is_some_and(|schema| !schema.is_object() && !schema.is_boolean())
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI function output schema must be a JSON Schema object or boolean",
            ));
        }
        Ok(())
    }
}

/// Protocol-owned request shaping. Provider crates keep typed user options and
/// translate them into this wire-focused structure after applying precedence.
#[derive(Debug, Clone, Default)]
pub struct RequestEncodingOptions {
    stream: bool,
    extra: BTreeMap<String, Value>,
    native_tools: Vec<Value>,
    function_tools: BTreeMap<String, FunctionToolEncodingOptions>,
    media_dialect: ResponsesMediaDialect,
}

impl RequestEncodingOptions {
    pub fn new(stream: bool) -> Self {
        Self {
            stream,
            ..Self::default()
        }
    }

    pub fn stream(&self) -> bool {
        self.stream
    }

    pub fn with_extra(mut self, extra: BTreeMap<String, Value>) -> Self {
        self.extra = extra;
        self
    }

    pub fn with_native_tool(mut self, tool: Value) -> Self {
        self.native_tools.push(tool);
        self
    }

    pub fn with_function_tool_options(
        mut self,
        name: impl Into<String>,
        options: FunctionToolEncodingOptions,
    ) -> Self {
        self.function_tools.insert(name.into(), options);
        self
    }

    pub const fn with_media_dialect(mut self, dialect: ResponsesMediaDialect) -> Self {
        self.media_dialect = dialect;
        self
    }
}

/// Encode a request with only request-wide Responses options.
pub fn encode_request(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    stream: bool,
    extra: &BTreeMap<String, Value>,
) -> Result<Value, Error> {
    encode_request_with_options(
        scope,
        model,
        request,
        &RequestEncodingOptions::new(stream).with_extra(extra.clone()),
    )
}

/// Encode a request while preserving same-protocol opaque history verbatim.
pub fn encode_request_with_options(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    options: &RequestEncodingOptions,
) -> Result<Value, Error> {
    encode_request_with_options_and_resolver(
        scope,
        model,
        request,
        options,
        &NoPromptCacheAnnotations,
    )
}

/// Encode a Responses request after resolving provider-owned content
/// annotations.
pub fn encode_request_with_options_and_resolver(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    options: &RequestEncodingOptions,
    resolver: &dyn PromptCacheAnnotationResolver,
) -> Result<Value, Error> {
    request.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "invalid language request").with_source(source)
    })?;
    if request.generation.seed.is_some() || !request.generation.stop_sequences.is_empty() {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI Responses does not encode neutral seed or stop-sequence controls",
        ));
    }
    for key in options.extra.keys() {
        if key != TEXT_VERBOSITY_OPTION && is_protected_option_field(key) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI Responses options attempted to override a protected request field",
            ));
        }
        if key.trim().is_empty() || key.chars().any(char::is_control) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI Responses option name is empty or contains control characters",
            ));
        }
    }
    let request_tool_names = request
        .tools
        .iter()
        .map(|tool| tool.name().to_string())
        .collect::<BTreeSet<_>>();
    for (name, tool_options) in &options.function_tools {
        if !request_tool_names.contains(name) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI function-tool options referenced a tool absent from the language request",
            ));
        }
        tool_options.validate()?;
    }

    validate_target_scope(scope)?;
    let call_contexts = collect_native_call_contexts(&request.messages, scope)?;
    let mut input = Vec::new();
    for message in &request.messages {
        encode_message(
            message,
            scope,
            options,
            &call_contexts,
            resolver,
            &mut input,
        )?;
    }

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));
    body.insert("input".to_string(), Value::Array(input));
    body.insert("stream".to_string(), Value::Bool(options.stream));
    insert_optional_u64(
        &mut body,
        "max_output_tokens",
        request.generation.max_output_tokens,
    );
    insert_optional_f64(&mut body, "temperature", request.generation.temperature)?;
    insert_optional_f64(&mut body, "top_p", request.generation.top_p)?;

    let mut tools = request
        .tools
        .iter()
        .map(|tool| {
            let name = tool.name();
            let tool_options = options.function_tools.get(name);
            let mut object = Map::new();
            object.insert("type".to_string(), Value::String("function".to_string()));
            object.insert("name".to_string(), Value::String(name.to_string()));
            if let Some(description) = tool.description() {
                object.insert(
                    "description".to_string(),
                    Value::String(description.to_string()),
                );
            }
            object.insert("parameters".to_string(), tool.input_schema().clone());
            if let Some(tool_options) = tool_options {
                if let Some(strict) = tool_options.strict {
                    object.insert("strict".to_string(), Value::Bool(strict));
                }
                if let Some(defer_loading) = tool_options.defer_loading {
                    object.insert("defer_loading".to_string(), Value::Bool(defer_loading));
                }
                if !tool_options.allowed_callers.is_empty() {
                    object.insert(
                        "allowed_callers".to_string(),
                        Value::Array(
                            tool_options
                                .allowed_callers
                                .iter()
                                .map(|caller| Value::String(caller.as_str().to_string()))
                                .collect(),
                        ),
                    );
                }
                if let Some(output_schema) = &tool_options.output_schema {
                    object.insert("output_schema".to_string(), output_schema.clone());
                }
            }
            Value::Object(object)
        })
        .collect::<Vec<_>>();
    for tool in &options.native_tools {
        if !tool.is_object() || tool.get("type").and_then(Value::as_str).is_none() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI native tool must be a JSON object with a string `type`",
            ));
        }
        tools.push(tool.clone());
    }
    if !tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(tools));
    }
    if let Some(choice) = &request.tool_choice {
        body.insert("tool_choice".to_string(), encode_tool_choice(choice)?);
    }
    let text_verbosity = options.extra.get(TEXT_VERBOSITY_OPTION);
    if let Some(verbosity) = text_verbosity
        && verbosity.as_str().is_none()
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI Responses text verbosity must be a string",
        ));
    }
    if request.structured_output.is_some() || text_verbosity.is_some() {
        let mut text = request
            .structured_output
            .as_ref()
            .map(encode_structured_output)
            .unwrap_or_else(|| Value::Object(Map::new()));
        if let Some(verbosity) = text_verbosity {
            text.as_object_mut()
                .expect("text request encoding always creates an object")
                .insert("verbosity".to_string(), verbosity.clone());
        }
        body.insert("text".to_string(), text);
    }
    for (key, value) in &options.extra {
        if key == TEXT_VERBOSITY_OPTION {
            continue;
        }
        body.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(body))
}

/// Fields owned by the neutral request encoder and unavailable to raw or typed
/// request-wide option maps.
pub fn is_protected_option_field(field: &str) -> bool {
    matches!(
        field,
        "model"
            | "input"
            | "stream"
            | "background"
            | "max_output_tokens"
            | "temperature"
            | "top_p"
            | "stop"
            | "seed"
            | "tools"
            | "tool_choice"
            | "text"
    )
}

#[derive(Debug, Clone)]
struct NativeCallContext {
    kind: NativeCallKind,
    caller: Option<Value>,
}

#[derive(Debug, Clone, Copy)]
enum NativeCallKind {
    Function,
    Custom,
}

fn collect_native_call_contexts(
    messages: &[Message],
    scope: &ProviderScope,
) -> Result<BTreeMap<String, NativeCallContext>, Error> {
    let mut contexts = BTreeMap::new();
    for message in messages {
        for part in message.content() {
            let ContentPart::ProviderOpaque(opaque) = part.content() else {
                continue;
            };
            let item = replayable_item(opaque, scope)?;
            let context = match item {
                OutputItem::FunctionCall(call) => Some((
                    call.call_id,
                    NativeCallContext {
                        kind: NativeCallKind::Function,
                        caller: call
                            .caller
                            .map(|caller| serde_json::to_value(caller).unwrap_or(Value::Null)),
                    },
                )),
                OutputItem::CustomToolCall(call) => Some((
                    call.call_id,
                    NativeCallContext {
                        kind: NativeCallKind::Custom,
                        caller: call
                            .caller
                            .map(|caller| serde_json::to_value(caller).unwrap_or(Value::Null)),
                    },
                )),
                _ => None,
            };
            if let Some((call_id, context)) = context
                && contexts.insert(call_id, context).is_some()
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "OpenAI Responses history reused a tool call ID",
                ));
            }
        }
    }
    Ok(contexts)
}

fn encode_message(
    message: &Message,
    scope: &ProviderScope,
    options: &RequestEncodingOptions,
    call_contexts: &BTreeMap<String, NativeCallContext>,
    resolver: &dyn PromptCacheAnnotationResolver,
    input: &mut Vec<Value>,
) -> Result<(), Error> {
    let native_items = message
        .content()
        .iter()
        .filter_map(|part| match part.content() {
            ContentPart::ProviderOpaque(opaque) => Some(replayable_item(opaque, scope)),
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    let suppression = NativeProjectionSuppression::from_message(message, &native_items)?;

    for item in native_items {
        input.push(item.to_value().map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "failed to replay a native OpenAI Responses item",
            )
            .with_source(source)
        })?);
    }

    let message_role = message.role();
    let mut message_content = Vec::new();
    for (content_index, part) in message.content().iter().enumerate() {
        let explicit_cache = resolver
            .resolve_content(part.annotations())?
            .explicit_breakpoint();
        match part.content() {
            ContentPart::Text { text } if !suppression.message => {
                let mut block = match message_role {
                    MessageRole::Assistant => json!({"type": "output_text", "text": text}),
                    MessageRole::System
                    | MessageRole::Developer
                    | MessageRole::User
                    | MessageRole::Tool => json!({"type": "input_text", "text": text}),
                    _ => {
                        return Err(Error::new(
                            ErrorKind::Unsupported,
                            "OpenAI Responses cannot encode this message role",
                        ));
                    }
                };
                apply_cache_breakpoint(&mut block, explicit_cache)?;
                message_content.push(block);
            }
            ContentPart::Text { .. } if suppression.message => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
            }
            ContentPart::Reasoning { .. } if suppression.reasoning => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
            }
            ContentPart::Reasoning { .. } => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "reasoning history requires its native OpenAI Responses item",
                ));
            }
            ContentPart::Media(media) if !suppression.message => {
                let mut block = encode_media(media, options.media_dialect)?;
                apply_cache_breakpoint(&mut block, explicit_cache)?;
                message_content.push(block);
            }
            ContentPart::Media(_) if suppression.message => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
            }
            ContentPart::Citation(_) | ContentPart::Refusal { .. } if suppression.message => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
            }
            ContentPart::Citation(_) | ContentPart::Refusal { .. } => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "response-only content requires its native OpenAI Responses message item",
                ));
            }
            ContentPart::ToolCall(_) if suppression.tool_calls.contains(&content_index) => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
            }
            ContentPart::ToolCall(call) => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
                input.push(json!({
                    "type": "function_call",
                    "call_id": call.id(),
                    "name": call.name(),
                    "arguments": serde_json::to_string(call.arguments()).map_err(|source| {
                        Error::new(ErrorKind::InvalidInput, "failed to encode tool arguments")
                            .with_source(source)
                    })?,
                }));
            }
            ContentPart::ToolResult(result) => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
                input.push(encode_tool_result(result, call_contexts)?);
            }
            ContentPart::ProviderOpaque(_) => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
            }
            _ => {
                reject_prompt_cache_breakpoint(explicit_cache)?;
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "OpenAI Responses cannot encode this neutral content part",
                ));
            }
        }
    }

    if !message_content.is_empty() {
        input.push(json!({
            "role": encode_role(message_role)?,
            "content": message_content,
        }));
    }
    Ok(())
}

fn replayable_item(
    opaque: &OpaqueProviderItem,
    scope: &ProviderScope,
) -> Result<OutputItem, Error> {
    let provenance = opaque.provenance();
    if !provenance.matches_replay_target(scope) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "foreign replay-domain history requires explicit projection before OpenAI Responses encoding",
        ));
    }
    if opaque.kind() != OPENAI_RESPONSES_OPAQUE_KIND {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI Responses history contained a native item that cannot be replayed",
        ));
    }
    serde_json::from_value::<OutputItem>(opaque.data().clone()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "stored OpenAI Responses history item is malformed",
        )
        .with_source(source)
    })
}

fn validate_target_scope(scope: &ProviderScope) -> Result<(), Error> {
    if scope
        .protocol()
        .is_some_and(|protocol| protocol.as_str() != OPENAI_RESPONSES_PROTOCOL)
        || scope
            .api_mode()
            .is_some_and(|api_mode| api_mode.as_str() != API_MODE_ID)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI Responses encoder received a scope for another protocol or API mode",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct NativeProjectionSuppression {
    message: bool,
    reasoning: bool,
    tool_calls: BTreeSet<usize>,
}

impl NativeProjectionSuppression {
    fn from_message(message: &Message, items: &[OutputItem]) -> Result<Self, Error> {
        Ok(Self {
            message: validate_native_message_projection(message, items)?,
            reasoning: validate_native_reasoning_projection(message, items)?,
            tool_calls: matched_native_tool_call_indices(message, items)?,
        })
    }
}

fn validate_native_message_projection(
    message: &Message,
    items: &[OutputItem],
) -> Result<bool, Error> {
    let native_messages = items
        .iter()
        .filter_map(|item| match item {
            OutputItem::Message(message) => Some(message),
            _ => None,
        })
        .collect::<Vec<_>>();
    if native_messages.is_empty() {
        return Ok(false);
    }

    let expected_role = encode_role(message.role())?;
    if native_messages
        .iter()
        .any(|message| message.role != expected_role)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "portable OpenAI Responses message role disagreed with its native replay item",
        ));
    }

    let native_projection = native_messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            OutputContentPart::Text(text) => Some(ContentPart::Text {
                text: text.text.clone(),
            }),
            OutputContentPart::Refusal(_) | OutputContentPart::Unknown(_) => None,
        })
        .collect::<Vec<_>>();
    let portable_projection = message
        .content()
        .iter()
        .filter_map(|part| match part.content() {
            ContentPart::Text { .. } | ContentPart::Media(_) => Some(part.content().clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !portable_projection.is_empty() && portable_projection != native_projection {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "portable OpenAI Responses message content disagreed with its native replay item",
        ));
    }
    Ok(true)
}

fn validate_native_reasoning_projection(
    message: &Message,
    items: &[OutputItem],
) -> Result<bool, Error> {
    let native_reasoning = items
        .iter()
        .filter_map(|item| match item {
            OutputItem::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect::<Vec<_>>();
    if native_reasoning.is_empty() {
        return Ok(false);
    }

    let native_projection = native_reasoning
        .iter()
        .flat_map(|reasoning| reasoning.summary.iter().chain(reasoning.content.iter()))
        .map(|part| part.text.as_str())
        .collect::<Vec<_>>();
    let portable_projection = message
        .content()
        .iter()
        .filter_map(|part| match part.content() {
            ContentPart::Reasoning { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !portable_projection.is_empty() && portable_projection != native_projection {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "portable OpenAI Responses reasoning disagreed with its native replay item",
        ));
    }
    Ok(true)
}

fn matched_native_tool_call_indices(
    message: &Message,
    items: &[OutputItem],
) -> Result<BTreeSet<usize>, Error> {
    let mut matched = BTreeSet::new();
    let mut portable_call_counts = BTreeMap::new();
    for part in message.content() {
        if let ContentPart::ToolCall(call) = part.content() {
            *portable_call_counts.entry(call.id()).or_insert(0usize) += 1;
        }
    }
    for (content_index, part) in message.content().iter().enumerate() {
        let ContentPart::ToolCall(portable) = part.content() else {
            continue;
        };
        let native = items
            .iter()
            .filter(|item| match item {
                OutputItem::FunctionCall(call) => call.call_id == portable.id(),
                OutputItem::CustomToolCall(call) => call.call_id == portable.id(),
                OutputItem::Program(program) => program.call_id == portable.id(),
                _ => false,
            })
            .collect::<Vec<_>>();
        if native.is_empty() {
            continue;
        }
        if native.len() != 1
            || portable_call_counts
                .get(portable.id())
                .copied()
                .unwrap_or_default()
                != 1
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "native and portable OpenAI tool calls could not be paired one-to-one",
            ));
        }

        let OutputItem::FunctionCall(native) = native[0] else {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "a local OpenAI tool call collided with provider-owned native tool activity",
            ));
        };
        if native.arguments.len() > DEFAULT_TOOL_INPUT_BYTE_LIMIT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "native OpenAI function call arguments exceeded the semantic tool-input limit",
            ));
        }
        let native_arguments =
            serde_json::from_str::<Value>(&native.arguments).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "native OpenAI function call arguments were not valid JSON",
                )
                .with_source(source)
            })?;
        if native.name.as_str() != portable.name() || &native_arguments != portable.arguments() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "portable OpenAI function call disagreed with its native replay item",
            ));
        }
        matched.insert(content_index);
    }
    Ok(matched)
}

fn encode_tool_result(
    result: &ToolResult,
    contexts: &BTreeMap<String, NativeCallContext>,
) -> Result<Value, Error> {
    let output = match &result.outcome {
        ToolOutcome::Success { value } => match value {
            Value::String(value) => value.clone(),
            value => serde_json::to_string(value).map_err(|source| {
                Error::new(ErrorKind::InvalidInput, "failed to encode tool result")
                    .with_source(source)
            })?,
        },
        outcome => serde_json::to_string(outcome).map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "failed to encode tool outcome").with_source(source)
        })?,
    };
    let context = contexts.get(&result.call_id);
    let kind = match context.map(|context| context.kind) {
        Some(NativeCallKind::Custom) => "custom_tool_call_output",
        Some(NativeCallKind::Function) | None => "function_call_output",
    };
    let mut item = Map::new();
    item.insert("type".to_string(), Value::String(kind.to_string()));
    item.insert("call_id".to_string(), Value::String(result.call_id.clone()));
    item.insert("output".to_string(), Value::String(output));
    if let Some(context) = context
        && matches!(context.kind, NativeCallKind::Function)
        && let Some(caller) = &context.caller
    {
        item.insert("caller".to_string(), caller.clone());
    }
    Ok(Value::Object(item))
}

fn encode_media(
    media: &siumai_core::MediaPart,
    dialect: ResponsesMediaDialect,
) -> Result<Value, Error> {
    let is_image = media.media_type.starts_with("image/");
    let is_video = media.media_type.starts_with("video/");
    match (&media.data, is_image, is_video) {
        (_, true, _) if !dialect.image_input => Err(Error::new(
            ErrorKind::Unsupported,
            "this Responses dialect does not support image input",
        )),
        (_, _, true) if !dialect.video_input => Err(Error::new(
            ErrorKind::Unsupported,
            "this Responses dialect does not support video input",
        )),
        (_, false, false) if !dialect.file_input => Err(Error::new(
            ErrorKind::Unsupported,
            "this Responses dialect does not support generic file input",
        )),
        (MediaData::Url(url), true, _) => Ok(json!({
            "type": "input_image",
            "image_url": url,
        })),
        (MediaData::Url(url), _, true) => Ok(json!({
            "type": "input_video",
            "video_url": url,
        })),
        (MediaData::Url(url), false, false) => Ok(json!({
            "type": "input_file",
            "file_url": url,
        })),
        (MediaData::Bytes(bytes), true, _) => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes.as_ref());
            Ok(json!({
                "type": "input_image",
                "image_url": format!("data:{};base64,{encoded}", media.media_type),
            }))
        }
        (MediaData::Bytes(bytes), _, true) => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes.as_ref());
            Ok(json!({
                "type": "input_video",
                "video_url": format!("data:{};base64,{encoded}", media.media_type),
            }))
        }
        (MediaData::Bytes(bytes), false, false) => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes.as_ref());
            Ok(json!({
                "type": "input_file",
                "filename": media.name.as_deref().unwrap_or("input"),
                "file_data": format!("data:{};base64,{encoded}", media.media_type),
            }))
        }
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI Responses does not support this media source",
        )),
    }
}

fn apply_cache_breakpoint(block: &mut Value, explicit: bool) -> Result<(), Error> {
    if !explicit {
        return Ok(());
    }
    let object = block.as_object_mut().ok_or_else(|| {
        Error::new(
            ErrorKind::Internal,
            "OpenAI input block encoder produced a non-object",
        )
    })?;
    object.insert(
        "prompt_cache_breakpoint".to_string(),
        json!({"mode": "explicit"}),
    );
    Ok(())
}

fn reject_prompt_cache_breakpoint(enabled: bool) -> Result<(), Error> {
    if enabled {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI prompt-cache breakpoint targeted a non-encodable content node",
        ));
    }
    Ok(())
}

fn encode_role(role: MessageRole) -> Result<&'static str, Error> {
    match role {
        MessageRole::System => Ok("system"),
        MessageRole::Developer => Ok("developer"),
        MessageRole::User => Ok("user"),
        MessageRole::Assistant => Ok("assistant"),
        MessageRole::Tool => Ok("tool"),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI Responses cannot encode this message role",
        )),
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Result<Value, Error> {
    match choice {
        ToolChoice::Auto => Ok(Value::String("auto".to_string())),
        ToolChoice::None => Ok(Value::String("none".to_string())),
        ToolChoice::Required => Ok(Value::String("required".to_string())),
        ToolChoice::Named { name } => Ok(json!({"type": "function", "name": name})),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI Responses cannot encode this tool choice",
        )),
    }
}

fn encode_structured_output(output: &StructuredOutputSpec) -> Value {
    json!({
        "format": {
            "type": "json_schema",
            "name": output.name,
            "description": output.description,
            "schema": output.schema,
            "strict": output.strict,
        }
    })
}

fn insert_optional_u64(object: &mut Map<String, Value>, key: &str, value: Option<u64>) {
    if let Some(value) = value {
        object.insert(key.to_string(), Value::from(value));
    }
}

fn insert_optional_f64(
    object: &mut Map<String, Value>,
    key: &str,
    value: Option<f64>,
) -> Result<(), Error> {
    if let Some(value) = value {
        let number = serde_json::Number::from_f64(value).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "OpenAI numeric request option must be finite",
            )
        })?;
        object.insert(key.to_string(), Value::Number(number));
    }
    Ok(())
}
