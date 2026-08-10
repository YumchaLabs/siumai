use std::collections::{BTreeMap, BTreeSet};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, Error, ErrorKind, LanguageCallError, LanguageCompletionReason,
    LanguageIncompleteReason, LanguageRequest, LanguageResponse, LanguageTermination, MediaData,
    MediaPart, Message, MessageRole, ModelId, OpaqueProviderItem, ProviderItemRelation,
    ProviderProvenance, ProviderScope, ToolCall, ToolChoice, ToolOutcome, ToolResult, Usage,
    Warning, WarningKind,
};

/// Stable Gemini Interactions create target relative to the official API origin.
pub const STABLE_V1_LANGUAGE_TARGET: &str = "v1/interactions";

/// Provider-native replay kind used for one Gemini Interactions step.
pub const INTERACTIONS_STEP_KIND: &str = "google.interactions.step";
/// Provider-native replay kind used for one non-portable model-output content block.
pub const INTERACTIONS_CONTENT_KIND: &str = "google.interactions.content";

const MAX_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;
const MAX_MEDIA_ENCODED_BYTES: usize = (MAX_MEDIA_BYTES / 3 + 1) * 4;
const MAX_STEPS: usize = 256;
const MAX_CONTENT_PARTS_PER_STEP: usize = 256;
const MAX_RESPONSE_ID_BYTES: usize = 4 * 1024;
const MAX_METADATA_TEXT_BYTES: usize = 4 * 1024;
const MAX_MODALITY_ENTRIES: usize = 32;
const MAX_MODALITY_BYTES: usize = 64;

/// Whether the Interactions service may retain the request and response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InteractionStorage {
    #[default]
    Disabled,
    Enabled,
}

impl InteractionStorage {
    const fn as_bool(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Thinking depth accepted by the stable Interactions generation config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InteractionThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

impl InteractionThinkingLevel {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Whether stable Interactions returns provider thought summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InteractionThinkingSummaries {
    Auto,
    None,
}

impl InteractionThinkingSummaries {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
        }
    }
}

/// Checked protocol-level settings for one portable language request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionLanguageConfig {
    storage: InteractionStorage,
    thinking_level: Option<InteractionThinkingLevel>,
    thinking_summaries: Option<InteractionThinkingSummaries>,
}

impl Default for InteractionLanguageConfig {
    fn default() -> Self {
        Self {
            storage: InteractionStorage::Disabled,
            thinking_level: None,
            thinking_summaries: None,
        }
    }
}

impl InteractionLanguageConfig {
    pub const fn new() -> Self {
        Self {
            storage: InteractionStorage::Disabled,
            thinking_level: None,
            thinking_summaries: None,
        }
    }

    pub const fn with_storage(mut self, storage: InteractionStorage) -> Self {
        self.storage = storage;
        self
    }

    pub const fn with_thinking_level(mut self, level: InteractionThinkingLevel) -> Self {
        self.thinking_level = Some(level);
        self
    }

    pub const fn with_thinking_summaries(
        mut self,
        summaries: InteractionThinkingSummaries,
    ) -> Self {
        self.thinking_summaries = Some(summaries);
        self
    }

    pub const fn storage(&self) -> InteractionStorage {
        self.storage
    }

    pub const fn thinking_level(&self) -> Option<InteractionThinkingLevel> {
        self.thinking_level
    }

    pub const fn thinking_summaries(&self) -> Option<InteractionThinkingSummaries> {
        self.thinking_summaries
    }
}

/// Lossless direct Interactions resource plus its portable terminal projection.
pub struct DecodedInteraction {
    native: Value,
    portable: Result<LanguageResponse, LanguageCallError>,
}

impl std::fmt::Debug for DecodedInteraction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodedInteraction")
            .field("native", &"<redacted>")
            .field("portable_is_ok", &self.portable.is_ok())
            .finish_non_exhaustive()
    }
}

impl DecodedInteraction {
    pub fn native(&self) -> &Value {
        &self.native
    }

    pub fn portable(&self) -> Result<&LanguageResponse, &LanguageCallError> {
        self.portable.as_ref()
    }

    pub fn into_parts(self) -> (Value, Result<LanguageResponse, LanguageCallError>) {
        (self.native, self.portable)
    }

    pub fn map_canonical(mut self, map: impl FnOnce(LanguageResponse) -> LanguageResponse) -> Self {
        self.portable = self.portable.map(map);
        self
    }

    pub fn into_result(self) -> Result<LanguageResponse, LanguageCallError> {
        self.portable
    }
}

/// Encode one stable v1 Interactions language request.
pub fn encode_language_request(
    request: &LanguageRequest,
    model: &ModelId,
    scope: &ProviderScope,
    config: &InteractionLanguageConfig,
    stream: bool,
) -> Result<Value, Error> {
    request.validate().map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Gemini Interactions language request is invalid",
        )
        .with_source(source)
    })?;

    if request.generation.temperature.is_some() || request.generation.top_p.is_some() {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "stable Gemini Interactions does not expose portable temperature or top-p fields",
        ));
    }

    let (system_instruction, input) = encode_messages(&request.messages, scope, model)?;
    if input.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Interactions requires at least one non-system input step",
        ));
    }

    let mut body = Map::new();
    body.insert(
        "model".to_string(),
        Value::String(model.as_str().to_string()),
    );
    body.insert("input".to_string(), Value::Array(input));
    body.insert("store".to_string(), Value::Bool(config.storage.as_bool()));
    if stream {
        body.insert("stream".to_string(), Value::Bool(true));
    }
    if let Some(system_instruction) = system_instruction {
        body.insert(
            "system_instruction".to_string(),
            Value::String(system_instruction),
        );
    }

    let generation_config = encode_generation_config(request, config)?;
    if !generation_config.is_empty() {
        body.insert(
            "generation_config".to_string(),
            Value::Object(generation_config),
        );
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "name": tool.name(),
                            "description": tool.description(),
                            "parameters": tool.input_schema(),
                        })
                    })
                    .collect(),
            ),
        );
    }
    if let Some(structured) = &request.structured_output {
        if !structured.strict {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Interactions cannot preserve non-strict structured-output semantics",
            ));
        }
        body.insert(
            "response_format".to_string(),
            json!({
                "type": "text",
                "mime_type": "application/json",
                "schema": structured.schema,
            }),
        );
    }

    Ok(Value::Object(body))
}

/// Decode one complete stable v1 Interactions language resource.
pub fn decode_language_response(
    body: &[u8],
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<DecodedInteraction, Error> {
    if body.len() > MAX_RESPONSE_BODY_BYTES {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Interactions response exceeded the protocol body limit",
        ));
    }
    let native = serde_json::from_slice::<Value>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "provider returned malformed Gemini Interactions JSON",
        )
        .with_source(source)
    })?;
    let wire = serde_json::from_value::<InteractionWire>(native.clone()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "provider returned an invalid Gemini Interactions resource",
        )
        .with_source(source)
    })?;
    let portable = project_interaction(wire, scope, requested_model)?;
    Ok(DecodedInteraction { native, portable })
}

/// Decode and validate a provider-native Interactions resource without
/// requiring it to be terminal or portable.
pub fn decode_interaction_resource(body: &[u8]) -> Result<Value, Error> {
    if body.len() > MAX_RESPONSE_BODY_BYTES {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Interactions response exceeded the protocol body limit",
        ));
    }
    let native = serde_json::from_slice::<Value>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "provider returned malformed Gemini Interactions JSON",
        )
        .with_source(source)
    })?;
    serde_json::from_value::<InteractionWire>(native.clone()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "provider returned an invalid Gemini Interactions resource",
        )
        .with_source(source)
    })?;
    Ok(native)
}

pub(crate) fn project_interaction(
    wire: InteractionWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<Result<LanguageResponse, LanguageCallError>, Error> {
    if wire.steps.len() > MAX_STEPS {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Interactions response exceeded the step limit",
        ));
    }
    let response_id = checked_optional_text(wire.id, MAX_RESPONSE_ID_BYTES, "interaction ID")?;
    let response_model = checked_optional_text(wire.model, MAX_METADATA_TEXT_BYTES, "model ID")?
        .map(ModelId::new)
        .transpose()
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "Gemini Interactions returned an invalid model identifier",
            )
            .with_source(source)
        })?
        .unwrap_or_else(|| requested_model.clone());
    let terminal = terminal_mapping(wire.status, &wire.steps)?;
    let mut content = Vec::new();
    let mut warnings = Vec::new();
    for step in &wire.steps {
        project_step(step, scope, &response_model, &mut content, &mut warnings)?;
    }
    let usage = decode_usage(wire.usage)?;
    let mut provider = BTreeMap::new();
    if let Some(value) =
        checked_optional_text(wire.service_tier, MAX_METADATA_TEXT_BYTES, "service tier")?
    {
        provider.insert("google.service_tier".to_string(), Value::String(value));
    }
    if let Some(value) = checked_optional_text(wire.created, MAX_METADATA_TEXT_BYTES, "created")? {
        provider.insert("google.created".to_string(), Value::String(value));
    }
    if let Some(value) = checked_optional_text(wire.updated, MAX_METADATA_TEXT_BYTES, "updated")? {
        provider.insert("google.updated".to_string(), Value::String(value));
    }
    let termination = match terminal {
        InteractionTerminal::Success(termination) => termination,
        terminal => {
            let error = match terminal {
                InteractionTerminal::Failed => {
                    Error::new(ErrorKind::Provider, "Gemini interaction failed")
                }
                InteractionTerminal::Cancelled => {
                    Error::new(ErrorKind::Cancelled, "Gemini interaction was cancelled")
                }
                InteractionTerminal::Success(_) => unreachable!(),
            };
            return Ok(Err(LanguageCallError::new(
                error,
                crate::generate_content::partial_output(&content, &usage),
            )));
        }
    };
    let mut response = LanguageResponse::new(termination, content, usage)
        .map_err(|source| {
            Error::protocol_violation("invalid Gemini terminal response").with_source(source)
        })?
        .with_model(response_model)
        .with_warnings(warnings)
        .with_provider_metadata(provider);
    if let Some(response_id) = response_id {
        response = response.with_id(response_id);
    }
    Ok(Ok(response))
}

fn encode_generation_config(
    request: &LanguageRequest,
    config: &InteractionLanguageConfig,
) -> Result<Map<String, Value>, Error> {
    let mut generation = Map::new();
    if let Some(maximum) = request.generation.max_output_tokens {
        let maximum = u32::try_from(maximum).map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                "Gemini max output tokens exceeds the stable wire range",
            )
        })?;
        generation.insert("max_output_tokens".to_string(), Value::from(maximum));
    }
    if let Some(seed) = request.generation.seed {
        let seed = i32::try_from(seed).map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                "Gemini seed exceeds the stable Interactions wire range",
            )
        })?;
        generation.insert("seed".to_string(), Value::from(seed));
    }
    if !request.generation.stop_sequences.is_empty() {
        generation.insert(
            "stop_sequences".to_string(),
            serde_json::to_value(&request.generation.stop_sequences).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini stop sequences could not be encoded",
                )
                .with_source(source)
            })?,
        );
    }
    if let Some(level) = config.thinking_level {
        generation.insert(
            "thinking_level".to_string(),
            Value::String(level.as_wire().to_string()),
        );
    }
    if let Some(summaries) = config.thinking_summaries {
        generation.insert(
            "thinking_summaries".to_string(),
            Value::String(summaries.as_wire().to_string()),
        );
    }
    if let Some(choice) = &request.tool_choice {
        let value = match choice {
            ToolChoice::Auto => Value::String("auto".to_string()),
            ToolChoice::None => Value::String("none".to_string()),
            ToolChoice::Required => Value::String("any".to_string()),
            ToolChoice::Named { name } => json!({
                "allowed_tools": {
                    "mode": "any",
                    "tools": [name],
                }
            }),
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "Gemini Interactions does not support this portable tool choice",
                ));
            }
        };
        generation.insert("tool_choice".to_string(), value);
    }
    Ok(generation)
}

fn encode_messages(
    messages: &[Message],
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<(Option<String>, Vec<Value>), Error> {
    let mut system = Vec::new();
    let mut steps = Vec::new();
    let mut conversation_started = false;
    for message in messages {
        match message.role() {
            MessageRole::System => {
                if conversation_started {
                    return Err(Error::new(
                        ErrorKind::Unsupported,
                        "Gemini Interactions requires system instructions before conversation input",
                    ));
                }
                for part in message.content() {
                    let ContentPart::Text { text } = part.content() else {
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "Gemini system and developer messages support text only",
                        ));
                    };
                    system.push(text.clone());
                }
            }
            MessageRole::Developer => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "Gemini Interactions does not expose a distinct developer-message role",
                ));
            }
            MessageRole::User => {
                conversation_started = true;
                let mut content = Vec::new();
                for part in message.content() {
                    match part.content() {
                        ContentPart::Text { text } => {
                            content.push(json!({"type": "text", "text": text}));
                        }
                        ContentPart::Media(media) => content.push(encode_media(media)?),
                        _ => {
                            return Err(Error::new(
                                ErrorKind::InvalidInput,
                                "Gemini user messages contain an unsupported content part",
                            ));
                        }
                    }
                }
                if !content.is_empty() {
                    steps.push(json!({"type": "user_input", "content": content}));
                }
            }
            MessageRole::Assistant => {
                conversation_started = true;
                encode_assistant_message(message, scope, model, &mut steps)?;
            }
            MessageRole::Tool => {
                conversation_started = true;
                for part in message.content() {
                    let ContentPart::ToolResult(result) = part.content() else {
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "Gemini tool messages may contain tool results only",
                        ));
                    };
                    steps.push(encode_tool_result(result)?);
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "Gemini Interactions does not support this message role",
                ));
            }
        }
    }
    let system = (!system.is_empty()).then(|| system.join("\n\n"));
    Ok((system, steps))
}

fn encode_assistant_message(
    message: &Message,
    scope: &ProviderScope,
    model: &ModelId,
    steps: &mut Vec<Value>,
) -> Result<(), Error> {
    let mut suppressed = BTreeSet::new();
    for (opaque_index, part) in message.content().iter().enumerate() {
        let ContentPart::ProviderOpaque(opaque) = part.content() else {
            continue;
        };
        validate_replay_item(opaque, scope, model)?;
        if opaque.kind() == INTERACTIONS_CONTENT_KIND {
            continue;
        }
        match step_type(opaque.data())? {
            "thought" => {
                let expected = thought_text(opaque.data())?;
                if let Some((index, _)) = message
                    .content()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != opaque_index && !suppressed.contains(index))
                    .find(|(_, candidate)| {
                        matches!(
                            candidate.content(),
                            ContentPart::Reasoning { text } if text == &expected
                        )
                    })
                {
                    suppressed.insert(index);
                }
            }
            "function_call" => {
                let native = function_call_fields(opaque.data())?;
                if let Some((index, call)) = message
                    .content()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, candidate)| match candidate.content() {
                        ContentPart::ToolCall(call) if call.id() == native.id => {
                            Some((index, call))
                        }
                        _ => None,
                    })
                    .next()
                {
                    if call.name() != native.name || call.arguments() != native.arguments {
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "Gemini native function-call replay disagrees with its portable sibling",
                        ));
                    }
                    suppressed.insert(index);
                }
            }
            _ => {}
        }
    }

    let mut pending_model_output = Vec::new();
    let flush_model_output = |pending: &mut Vec<Value>, steps: &mut Vec<Value>| {
        if !pending.is_empty() {
            steps.push(json!({"type": "model_output", "content": std::mem::take(pending)}));
        }
    };
    for (index, part) in message.content().iter().enumerate() {
        if suppressed.contains(&index) {
            continue;
        }
        match part.content() {
            ContentPart::Text { text } => {
                pending_model_output.push(json!({"type": "text", "text": text}));
            }
            ContentPart::Media(media) => pending_model_output.push(encode_media(media)?),
            ContentPart::Reasoning { text } => {
                flush_model_output(&mut pending_model_output, steps);
                steps.push(json!({
                    "type": "thought",
                    "summary": [{"type": "text", "text": text}],
                }));
            }
            ContentPart::ToolCall(call) => {
                flush_model_output(&mut pending_model_output, steps);
                steps.push(json!({
                    "type": "function_call",
                    "id": call.id(),
                    "name": call.name(),
                    "arguments": call.arguments(),
                }));
            }
            ContentPart::ProviderOpaque(opaque) => {
                if opaque.kind() == INTERACTIONS_CONTENT_KIND {
                    pending_model_output.push(opaque.data().clone());
                } else {
                    flush_model_output(&mut pending_model_output, steps);
                    steps.push(opaque.data().clone());
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini assistant history contains an unsupported content part",
                ));
            }
        }
    }
    flush_model_output(&mut pending_model_output, steps);
    Ok(())
}

fn validate_replay_item(
    item: &OpaqueProviderItem,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<(), Error> {
    if !matches!(
        item.kind(),
        INTERACTIONS_STEP_KIND | INTERACTIONS_CONTENT_KIND
    ) || !item.provenance().matches_replay_target(scope)
        || item.provenance().model() != model
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini provider-native replay item does not match the target scope and model",
        ));
    }
    Ok(())
}

fn encode_media(media: &MediaPart) -> Result<Value, Error> {
    let top_level = media.media_type.split('/').next().unwrap_or_default();
    let kind = match (top_level, media.media_type.as_str()) {
        ("image", _) => "image",
        ("audio", _) => "audio",
        (_, "application/pdf" | "text/csv") => "document",
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "stable Gemini Interactions cannot encode this portable media type",
            ));
        }
    };
    let mut value = Map::new();
    value.insert("type".to_string(), Value::String(kind.to_string()));
    value.insert(
        "mime_type".to_string(),
        Value::String(media.media_type.clone()),
    );
    match &media.data {
        MediaData::Bytes(bytes) => {
            if bytes.is_empty() || bytes.len() > MAX_MEDIA_BYTES {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini media input violates the protocol byte limit",
                ));
            }
            value.insert(
                "data".to_string(),
                Value::String(base64::engine::general_purpose::STANDARD.encode(bytes)),
            );
        }
        MediaData::Url(uri) => {
            if !valid_bounded_text(uri, MAX_METADATA_TEXT_BYTES * 4) {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini media URI is invalid or too large",
                ));
            }
            value.insert("uri".to_string(), Value::String(uri.clone()));
        }
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Interactions does not support this portable media carrier",
            ));
        }
    }
    Ok(Value::Object(value))
}

fn encode_tool_result(result: &ToolResult) -> Result<Value, Error> {
    let (is_error, value) = match &result.outcome {
        ToolOutcome::Success { value } => (false, supported_tool_result(value)?),
        ToolOutcome::Denied { reason } => (true, Value::String(reason.clone())),
        ToolOutcome::ExecutionFailed { message, .. } => (true, Value::String(message.clone())),
        ToolOutcome::Cancelled { reason } => (true, Value::String(reason.clone())),
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Interactions cannot encode this portable tool outcome",
            ));
        }
    };
    Ok(json!({
        "type": "function_result",
        "call_id": result.call_id,
        "name": result.name,
        "is_error": is_error,
        "result": value,
    }))
}

fn supported_tool_result(value: &Value) -> Result<Value, Error> {
    match value {
        Value::Object(_) | Value::String(_) => Ok(value.clone()),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "Gemini Interactions tool results preserve only JSON objects or strings",
        )),
    }
}

pub(crate) fn project_step(
    step: &Value,
    scope: &ProviderScope,
    model: &ModelId,
    content: &mut Vec<ContentPart>,
    warnings: &mut Vec<Warning>,
) -> Result<(), Error> {
    match step_type(step)? {
        "model_output" => project_model_output(step, scope, model, content, warnings),
        "thought" => {
            let text = thought_text(step)?;
            content.push(ContentPart::Reasoning { text });
            content.push(ContentPart::ProviderOpaque(opaque_step(
                step, scope, model,
            )?));
            Ok(())
        }
        "function_call" => {
            let fields = function_call_fields(step)?;
            if !fields.arguments.is_object() {
                return Err(Error::protocol_violation(
                    "Gemini function-call arguments must be a JSON object",
                ));
            }
            let call = ToolCall::local(fields.id, fields.name, fields.arguments.clone()).map_err(
                |source| {
                    Error::protocol_violation(
                        "Gemini returned an invalid caller-executed function call",
                    )
                    .with_source(source)
                },
            )?;
            content.push(ContentPart::ToolCall(call));
            content.push(ContentPart::ProviderOpaque(opaque_step(
                step, scope, model,
            )?));
            Ok(())
        }
        _ => {
            content.push(ContentPart::ProviderOpaque(opaque_step(
                step, scope, model,
            )?));
            Ok(())
        }
    }
}

fn project_model_output(
    step: &Value,
    scope: &ProviderScope,
    model: &ModelId,
    content: &mut Vec<ContentPart>,
    warnings: &mut Vec<Warning>,
) -> Result<(), Error> {
    let blocks = step
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::protocol_violation("Gemini model output omitted its content"))?;
    if blocks.len() > MAX_CONTENT_PARTS_PER_STEP {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini model output exceeded the content-part limit",
        ));
    }
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block.get("text").and_then(Value::as_str).ok_or_else(|| {
                    Error::protocol_violation("Gemini text output omitted its text")
                })?;
                content.push(ContentPart::Text {
                    text: text.to_string(),
                });
            }
            Some("image" | "audio" | "document") => {
                content.push(ContentPart::Media(decode_media(block)?));
            }
            Some(_) | None => {
                content.push(ContentPart::ProviderOpaque(opaque_content(
                    block, scope, model,
                )?));
                warnings.push(Warning::new(
                    WarningKind::UnsupportedOption,
                    "Gemini returned a non-portable model-output content block retained as provider-native replay data",
                ));
            }
        }
    }
    Ok(())
}

fn decode_media(block: &Value) -> Result<MediaPart, Error> {
    let media_type = block
        .get("mime_type")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::protocol_violation("Gemini media output omitted its MIME type"))?;
    if !valid_bounded_text(media_type, 256) || !media_type.contains('/') {
        return Err(Error::protocol_violation(
            "Gemini returned an invalid media MIME type",
        ));
    }
    let data = match (
        block.get("data").and_then(Value::as_str),
        block.get("uri").and_then(Value::as_str),
    ) {
        (Some(encoded), _) => {
            if encoded.len() > MAX_MEDIA_ENCODED_BYTES {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini inline media exceeded the encoded media limit",
                ));
            }
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|source| {
                    Error::protocol_violation("Gemini returned invalid base64 media")
                        .with_source(source)
                })?;
            if decoded.is_empty() || decoded.len() > MAX_MEDIA_BYTES {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini inline media violated the decoded media limit",
                ));
            }
            MediaData::Bytes(decoded.into())
        }
        (None, Some(uri)) if valid_bounded_text(uri, MAX_METADATA_TEXT_BYTES * 4) => {
            MediaData::Url(uri.to_string())
        }
        _ => {
            return Err(Error::protocol_violation(
                "Gemini media output omitted valid data or URI",
            ));
        }
    };
    Ok(MediaPart {
        media_type: media_type.to_string(),
        data,
        name: None,
    })
}

pub(crate) fn opaque_step(
    step: &Value,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<OpaqueProviderItem, Error> {
    let provenance = ProviderProvenance::from_scope(scope, model.clone()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Gemini Interactions replay requires an explicit replay domain",
        )
        .with_source(source)
    })?;
    let mut builder = OpaqueProviderItem::builder(provenance, INTERACTIONS_STEP_KIND, step.clone());
    if let Some(id) = step
        .get("id")
        .or_else(|| step.get("call_id"))
        .and_then(Value::as_str)
    {
        builder = builder.item_id(id);
        if step.get("type").and_then(Value::as_str) == Some("function_call") {
            let relation = ProviderItemRelation::call(id).map_err(|source| {
                Error::protocol_violation("Gemini function-call relation is invalid")
                    .with_source(source)
            })?;
            builder = builder.relations([relation]);
        }
    }
    builder.build().map_err(|source| {
        Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Interactions step exceeded the opaque replay limit",
        )
        .with_source(source)
    })
}

fn opaque_content(
    block: &Value,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<OpaqueProviderItem, Error> {
    let provenance = ProviderProvenance::from_scope(scope, model.clone()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Gemini Interactions replay requires an explicit replay domain",
        )
        .with_source(source)
    })?;
    OpaqueProviderItem::builder(provenance, INTERACTIONS_CONTENT_KIND, block.clone())
        .build()
        .map_err(|source| {
            Error::new(
                ErrorKind::ResponseLimit,
                "Gemini Interactions content exceeded the opaque replay limit",
            )
            .with_source(source)
        })
}

pub(crate) fn step_type(step: &Value) -> Result<&str, Error> {
    step.get("type")
        .and_then(Value::as_str)
        .filter(|value| valid_bounded_text(value, 128))
        .ok_or_else(|| Error::protocol_violation("Gemini Interactions step has an invalid type"))
}

pub(crate) fn thought_text(step: &Value) -> Result<String, Error> {
    let Some(summary) = step.get("summary") else {
        return Ok(String::new());
    };
    let summary = summary
        .as_array()
        .ok_or_else(|| Error::protocol_violation("Gemini thought summary must be an array"))?;
    if summary.len() > MAX_CONTENT_PARTS_PER_STEP {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini thought summary exceeded the content-part limit",
        ));
    }
    let mut texts = Vec::new();
    for item in summary {
        if item.get("type").and_then(Value::as_str) == Some("text") {
            let text = item.get("text").and_then(Value::as_str).ok_or_else(|| {
                Error::protocol_violation("Gemini thought text omitted its value")
            })?;
            texts.push(text);
        }
    }
    Ok(texts.join("\n"))
}

pub(crate) struct FunctionCallFields<'a> {
    pub(crate) id: &'a str,
    pub(crate) name: &'a str,
    pub(crate) arguments: &'a Value,
}

pub(crate) fn function_call_fields(step: &Value) -> Result<FunctionCallFields<'_>, Error> {
    let id = step
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::protocol_violation("Gemini function call omitted its ID"))?;
    let name = step
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::protocol_violation("Gemini function call omitted its name"))?;
    let arguments = step
        .get("arguments")
        .ok_or_else(|| Error::protocol_violation("Gemini function call omitted its arguments"))?;
    Ok(FunctionCallFields {
        id,
        name,
        arguments,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InteractionTerminal {
    Success(LanguageTermination),
    Failed,
    Cancelled,
}

pub(crate) fn terminal_mapping(
    status: InteractionStatus,
    steps: &[Value],
) -> Result<InteractionTerminal, Error> {
    let has_function_call = steps
        .iter()
        .any(|step| step.get("type").and_then(Value::as_str) == Some("function_call"));
    match status {
        InteractionStatus::Completed => Ok(InteractionTerminal::Success(
            LanguageTermination::Completed(if has_function_call {
                LanguageCompletionReason::ToolCalls
            } else {
                LanguageCompletionReason::Stop
            }),
        )),
        InteractionStatus::RequiresAction if has_function_call => Ok(InteractionTerminal::Success(
            LanguageTermination::Completed(LanguageCompletionReason::ToolCalls),
        )),
        InteractionStatus::RequiresAction => Err(Error::protocol_violation(
            "Gemini requires_action response omitted a caller-executed function call",
        )),
        InteractionStatus::Incomplete => Ok(InteractionTerminal::Success(
            LanguageTermination::Incomplete(LanguageIncompleteReason::Other(
                "provider_incomplete".to_string(),
            )),
        )),
        InteractionStatus::Failed => Ok(InteractionTerminal::Failed),
        InteractionStatus::Cancelled => Ok(InteractionTerminal::Cancelled),
        InteractionStatus::InProgress => Err(Error::protocol_violation(
            "Gemini returned a non-terminal interaction to a terminal call",
        )),
    }
}

pub(crate) fn decode_usage(wire: Option<InteractionUsageWire>) -> Result<Usage, Error> {
    let Some(wire) = wire else {
        return Ok(Usage::default());
    };
    let mut usage = Usage::default()
        .with_input_tokens(wire.total_input_tokens)
        .with_output_tokens(wire.total_output_tokens)
        .with_total_tokens(wire.total_tokens)
        .with_reasoning_tokens(wire.total_thought_tokens)
        .with_cache_read_tokens(wire.total_cached_tokens)
        .with_orchestration_tokens(wire.total_tool_use_tokens);
    for (name, values) in [
        (
            "google.input_tokens_by_modality",
            wire.input_tokens_by_modality,
        ),
        (
            "google.output_tokens_by_modality",
            wire.output_tokens_by_modality,
        ),
        (
            "google.cached_tokens_by_modality",
            wire.cached_tokens_by_modality,
        ),
        (
            "google.tool_use_tokens_by_modality",
            wire.tool_use_tokens_by_modality,
        ),
    ] {
        if let Some(values) = values {
            if values.len() > MAX_MODALITY_ENTRIES
                || values.iter().any(|entry| {
                    !valid_bounded_text(&entry.modality, MAX_MODALITY_BYTES)
                        || !entry.modality.is_ascii()
                })
            {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini usage modalities exceeded the protocol limit",
                ));
            }
            usage = usage.with_provider_value(
                name,
                serde_json::to_value(values).map_err(|source| {
                    Error::new(
                        ErrorKind::Internal,
                        "Gemini usage modalities could not be retained",
                    )
                    .with_source(source)
                })?,
            );
        }
    }
    Ok(usage)
}

fn checked_optional_text(
    value: Option<String>,
    maximum: usize,
    field: &'static str,
) -> Result<Option<String>, Error> {
    value
        .map(|value| {
            if valid_bounded_text(&value, maximum) {
                Ok(value)
            } else {
                let _ = field;
                Err(Error::protocol_violation(
                    "Gemini returned invalid bounded response metadata",
                ))
            }
        })
        .transpose()
}

pub(crate) fn valid_bounded_text(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InteractionStatus {
    InProgress,
    RequiresAction,
    Completed,
    Failed,
    Cancelled,
    Incomplete,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct InteractionWire {
    #[serde(default)]
    pub(crate) id: Option<String>,
    pub(crate) status: InteractionStatus,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) steps: Vec<Value>,
    #[serde(default)]
    pub(crate) usage: Option<InteractionUsageWire>,
    #[serde(default)]
    pub(crate) service_tier: Option<String>,
    #[serde(default)]
    pub(crate) created: Option<String>,
    #[serde(default)]
    pub(crate) updated: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct InteractionUsageWire {
    #[serde(default)]
    pub(crate) total_input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_thought_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_cached_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_tool_use_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) input_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
    #[serde(default)]
    pub(crate) output_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
    #[serde(default)]
    pub(crate) cached_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
    #[serde(default)]
    pub(crate) tool_use_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ModalityTokensWire {
    pub(crate) modality: String,
    pub(crate) tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{
        ApiModeId, PlatformId, ProtocolId, ProviderId, ReplayDomain, ReplayDomainId, ToolSpec,
    };

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new("google").unwrap())
            .with_platform(PlatformId::new("gemini-api").unwrap())
            .with_protocol(ProtocolId::new("gemini-interactions").unwrap())
            .with_api_mode(ApiModeId::new("interactions").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("google-gemini-api").unwrap(),
            ))
    }

    #[test]
    fn request_uses_current_response_format_and_canonical_tool_arguments() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let tool = ToolSpec::new(
            "lookup",
            Some("Lookup a value".to_string()),
            json!({"type": "object", "properties": {"key": {"type": "string"}}}),
        )
        .unwrap();
        let mut request = LanguageRequest::new(vec![Message::user("lookup x")]);
        request.tools.push(tool);
        request.structured_output = Some(siumai_core::StructuredOutputSpec {
            name: "answer".to_string(),
            description: None,
            schema: json!({"type": "object"}),
            strict: true,
        });

        let body = encode_language_request(
            &request,
            &model,
            &scope(),
            &InteractionLanguageConfig::default(),
            false,
        )
        .unwrap();
        let value = body;

        assert_eq!(value["store"], false);
        assert_eq!(value["response_format"]["type"], "text");
        assert_eq!(value["response_format"]["mime_type"], "application/json");
        assert!(value.get("outputs").is_none());
        assert!(value.get("response_mime_type").is_none());
        assert!(value.get("response_modalities").is_none());
    }

    #[test]
    fn lossy_request_semantics_fail_before_wire_encoding() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let config = InteractionLanguageConfig::default();

        let developer = LanguageRequest::new(vec![Message::developer("policy")]);
        assert!(matches!(
            encode_language_request(&developer, &model, &scope(), &config, false),
            Err(error) if error.kind() == ErrorKind::Unsupported
        ));

        let late_system =
            LanguageRequest::new(vec![Message::user("hello"), Message::system("late policy")]);
        assert!(matches!(
            encode_language_request(&late_system, &model, &scope(), &config, false),
            Err(error) if error.kind() == ErrorKind::Unsupported
        ));

        let mut non_strict = LanguageRequest::new(vec![Message::user("json")]);
        non_strict.structured_output = Some(siumai_core::StructuredOutputSpec {
            name: "answer".to_string(),
            description: None,
            schema: json!({"type": "object"}),
            strict: false,
        });
        assert!(matches!(
            encode_language_request(&non_strict, &model, &scope(), &config, false),
            Err(error) if error.kind() == ErrorKind::Unsupported
        ));

        let scalar_result = LanguageRequest::new(vec![Message::tool_result(ToolResult {
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
            outcome: ToolOutcome::Success { value: json!(42) },
        })]);
        assert!(matches!(
            encode_language_request(&scalar_result, &model, &scope(), &config, false),
            Err(error) if error.kind() == ErrorKind::Unsupported
        ));
    }

    #[test]
    fn direct_response_preserves_thought_and_function_replay() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let body = serde_json::to_vec(&json!({
            "id": "interaction-1",
            "status": "requires_action",
            "model": model.as_str(),
            "usage": {
                "total_input_tokens": 5,
                "total_output_tokens": 2,
                "total_thought_tokens": 1,
                "total_tokens": 8
            },
            "steps": [
                {
                    "type": "thought",
                    "signature": "signed",
                    "summary": [{"type": "text", "text": "reason"}]
                },
                {
                    "type": "function_call",
                    "id": "call/1",
                    "name": "lookup",
                    "arguments": {"key": "x"}
                }
            ]
        }))
        .unwrap();

        let decoded = decode_language_response(&body, &scope(), &model).unwrap();
        let response = decoded.portable().unwrap();

        assert_eq!(
            response.termination(),
            &LanguageTermination::Completed(LanguageCompletionReason::ToolCalls)
        );
        assert_eq!(response.content().len(), 4);
        assert!(matches!(
            &response.content()[0],
            ContentPart::Reasoning { text } if text == "reason"
        ));
        assert!(matches!(
            &response.content()[2],
            ContentPart::ToolCall(call) if call.arguments() == &json!({"key": "x"})
        ));
        assert_eq!(
            response.usage().input_tokens,
            siumai_core::UsageValue::Known(5)
        );
    }

    #[test]
    fn unknown_output_content_remains_replayable() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let body = serde_json::to_vec(&json!({
            "id": "interaction-opaque",
            "status": "completed",
            "model": model.as_str(),
            "steps": [{
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "answer"},
                    {"type": "future_content", "value": {"id": "native-1"}}
                ]
            }]
        }))
        .unwrap();
        let response = decode_language_response(&body, &scope(), &model)
            .unwrap()
            .into_result()
            .unwrap();
        assert!(response.content().iter().any(|part| matches!(
            part,
            ContentPart::ProviderOpaque(item) if item.kind() == INTERACTIONS_CONTENT_KIND
        )));

        let history = response.project_assistant_history().into_message().unwrap();
        let replay = LanguageRequest::new(vec![history, Message::user("continue")]);
        let encoded = encode_language_request(
            &replay,
            &model,
            &scope(),
            &InteractionLanguageConfig::default(),
            false,
        )
        .unwrap();
        assert_eq!(encoded["input"][0]["content"][1]["type"], "future_content");
    }
}
