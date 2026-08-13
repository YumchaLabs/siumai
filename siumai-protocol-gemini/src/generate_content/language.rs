use std::collections::{BTreeMap, BTreeSet};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, Error, ErrorKind, LanguageCallError, LanguageCompletionReason,
    LanguageIncompleteReason, LanguageRequest, LanguageResponse, LanguageTermination, MediaData,
    MediaPart, Message, MessageRole, ModelId, OpaqueProviderItem, PartialLanguageOutput,
    PartialLanguageOutputPart, ProviderItemRelation, ProviderProvenance, ProviderScope, ToolCall,
    ToolChoice, ToolOutcome, ToolResult, Usage, UsageValue, Warning, WarningKind,
};

/// Stable v1 target template for Google's Legacy Generate Content product mode.
pub const LEGACY_STABLE_V1_GENERATE_CONTENT_TARGET: &str = "v1/models/{model}:generateContent";
/// Stable v1 SSE target template for Google's Legacy Generate Content product mode.
pub const LEGACY_STABLE_V1_STREAM_GENERATE_CONTENT_TARGET: &str =
    "v1/models/{model}:streamGenerateContent?alt=sse";

/// Provider-native replay kind used for one Generate Content response part.
pub const GENERATE_CONTENT_PART_KIND: &str = "google.generate_content.part";

const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;
const MAX_MEDIA_ENCODED_BYTES: usize = (MAX_MEDIA_BYTES / 3 + 1) * 4;
const MAX_CONTENTS: usize = 512;
const MAX_PARTS_PER_CONTENT: usize = 256;
const MAX_CANDIDATES: usize = 1;
const MAX_RESPONSE_ID_BYTES: usize = 4 * 1024;
const MAX_METADATA_TEXT_BYTES: usize = 4 * 1024;
const MAX_PROVIDER_METADATA_BYTES: usize = 1024 * 1024;
const MAX_MODALITY_ENTRIES: usize = 32;
const MAX_MODALITY_BYTES: usize = 64;
const MAX_STOP_SEQUENCES: usize = 5;

/// Service tier accepted by stable v1 Generate Content requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GenerateContentServiceTier {
    Standard,
    Flex,
    Priority,
}

impl GenerateContentServiceTier {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Flex => "flex",
            Self::Priority => "priority",
        }
    }
}

/// Thinking level accepted by stable v1 Generate Content generation config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GenerateContentThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

impl GenerateContentThinkingLevel {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Minimal => "MINIMAL",
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
        }
    }
}

/// Checked provider-owned thinking controls for Generate Content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenerateContentThinkingConfig {
    include_thoughts: Option<bool>,
    budget: Option<i32>,
    level: Option<GenerateContentThinkingLevel>,
}

impl GenerateContentThinkingConfig {
    pub const fn new() -> Self {
        Self {
            include_thoughts: None,
            budget: None,
            level: None,
        }
    }

    pub const fn with_include_thoughts(mut self, include: bool) -> Self {
        self.include_thoughts = Some(include);
        self
    }

    pub const fn with_budget(mut self, budget: i32) -> Self {
        self.budget = Some(budget);
        self
    }

    pub const fn with_level(mut self, level: GenerateContentThinkingLevel) -> Self {
        self.level = Some(level);
        self
    }

    pub const fn include_thoughts(&self) -> Option<bool> {
        self.include_thoughts
    }

    pub const fn budget(&self) -> Option<i32> {
        self.budget
    }

    pub const fn level(&self) -> Option<GenerateContentThinkingLevel> {
        self.level
    }
}

/// Checked protocol-level settings for one portable Generate Content request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenerateContentLanguageConfig {
    service_tier: Option<GenerateContentServiceTier>,
    store: Option<bool>,
    top_k: Option<u32>,
    thinking: Option<GenerateContentThinkingConfig>,
}

impl GenerateContentLanguageConfig {
    pub const fn new() -> Self {
        Self {
            service_tier: None,
            store: None,
            top_k: None,
            thinking: None,
        }
    }

    pub const fn with_service_tier(mut self, tier: GenerateContentServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    pub const fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    pub const fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub const fn with_thinking(mut self, thinking: GenerateContentThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn service_tier(&self) -> Option<GenerateContentServiceTier> {
        self.service_tier
    }

    pub const fn store(&self) -> Option<bool> {
        self.store
    }

    pub const fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    pub const fn thinking(&self) -> Option<&GenerateContentThinkingConfig> {
        self.thinking.as_ref()
    }
}

/// Lossless direct Generate Content response plus its portable projection.
pub struct DecodedGenerateContent {
    native: Value,
    portable: Result<LanguageResponse, LanguageCallError>,
}

impl std::fmt::Debug for DecodedGenerateContent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodedGenerateContent")
            .field("native", &"<redacted>")
            .field("portable_is_ok", &self.portable.is_ok())
            .finish_non_exhaustive()
    }
}

impl DecodedGenerateContent {
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

/// Encode one stable v1 Generate Content language request.
pub fn encode_language_request(
    request: &LanguageRequest,
    model: &ModelId,
    scope: &ProviderScope,
    config: &GenerateContentLanguageConfig,
) -> Result<Value, Error> {
    request.validate().map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Gemini Generate Content language request is invalid",
        )
        .with_source(source)
    })?;

    let (system_instruction, contents) = encode_messages(&request.messages, scope, model)?;
    if contents.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Generate Content requires at least one non-system content turn",
        ));
    }
    if contents.len() > MAX_CONTENTS {
        return Err(Error::new(
            ErrorKind::LimitExceeded,
            "Gemini Generate Content request exceeded the content-turn limit",
        ));
    }

    let mut body = Map::new();
    body.insert("contents".to_string(), Value::Array(contents));
    if let Some(system_instruction) = system_instruction {
        body.insert(
            "systemInstruction".to_string(),
            json!({"parts": system_instruction}),
        );
    }
    if !request.tools.is_empty() {
        body.insert("tools".to_string(), encode_tools(request)?);
    }
    if let Some(tool_config) = encode_tool_choice(request)? {
        body.insert("toolConfig".to_string(), tool_config);
    }

    let generation_config = encode_generation_config(request, config)?;
    if !generation_config.is_empty() {
        body.insert(
            "generationConfig".to_string(),
            Value::Object(generation_config),
        );
    }
    if let Some(tier) = config.service_tier {
        body.insert(
            "serviceTier".to_string(),
            Value::String(tier.as_wire().to_string()),
        );
    }
    if let Some(store) = config.store {
        body.insert("store".to_string(), Value::Bool(store));
    }

    let body = Value::Object(body);
    ensure_encoded_value_limit(
        &body,
        MAX_REQUEST_BODY_BYTES,
        "Gemini Generate Content request exceeded the protocol body limit",
    )?;
    Ok(body)
}

/// Decode one complete stable v1 Generate Content language response.
pub fn decode_language_response(
    body: &[u8],
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<DecodedGenerateContent, Error> {
    if body.len() > MAX_RESPONSE_BODY_BYTES {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Generate Content response exceeded the protocol body limit",
        ));
    }
    let native = serde_json::from_slice::<Value>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "provider returned malformed Gemini Generate Content JSON",
        )
        .with_source(source)
    })?;
    let portable = project_response_value(&native, scope, requested_model)?;
    Ok(DecodedGenerateContent { native, portable })
}

fn encode_generation_config(
    request: &LanguageRequest,
    config: &GenerateContentLanguageConfig,
) -> Result<Map<String, Value>, Error> {
    let generation = &request.generation;
    let mut wire = Map::new();
    if let Some(maximum) = generation.max_output_tokens {
        if maximum > i32::MAX as u64 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini max output tokens exceed the stable v1 integer range",
            ));
        }
        wire.insert("maxOutputTokens".to_string(), Value::from(maximum));
    }
    if let Some(temperature) = generation.temperature {
        if temperature > 2.0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini temperature must be between zero and two",
            ));
        }
        wire.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = generation.top_p {
        wire.insert("topP".to_string(), Value::from(top_p));
    }
    if !generation.stop_sequences.is_empty() {
        if generation.stop_sequences.len() > MAX_STOP_SEQUENCES {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini accepts at most five stop sequences",
            ));
        }
        wire.insert(
            "stopSequences".to_string(),
            serde_json::to_value(&generation.stop_sequences).map_err(|source| {
                Error::new(
                    ErrorKind::Internal,
                    "Gemini stop sequences could not be encoded",
                )
                .with_source(source)
            })?,
        );
    }
    if let Some(seed) = generation.seed {
        if seed > i32::MAX as u64 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini seed exceeds the stable v1 integer range",
            ));
        }
        wire.insert("seed".to_string(), Value::from(seed));
    }
    if let Some(top_k) = config.top_k {
        if top_k > i32::MAX as u32 {
            return Err(Error::new(
                ErrorKind::Configuration,
                "Gemini top-k exceeds the stable v1 integer range",
            ));
        }
        wire.insert("topK".to_string(), Value::from(top_k));
    }
    if let Some(thinking) = &config.thinking {
        if thinking.budget.is_some() && thinking.level.is_some() {
            return Err(Error::new(
                ErrorKind::Configuration,
                "Gemini thinking budget and thinking level are mutually exclusive",
            ));
        }
        let mut value = Map::new();
        if let Some(include) = thinking.include_thoughts {
            value.insert("includeThoughts".to_string(), Value::Bool(include));
        }
        if let Some(budget) = thinking.budget {
            value.insert("thinkingBudget".to_string(), Value::from(budget));
        }
        if let Some(level) = thinking.level {
            value.insert(
                "thinkingLevel".to_string(),
                Value::String(level.as_wire().to_string()),
            );
        }
        if !value.is_empty() {
            wire.insert("thinkingConfig".to_string(), Value::Object(value));
        }
    }
    if let Some(structured) = &request.structured_output {
        if !structured.strict {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Generate Content cannot preserve non-strict structured-output semantics",
            ));
        }
        wire.insert(
            "responseFormat".to_string(),
            json!({
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": structured.schema,
                }
            }),
        );
    }
    Ok(wire)
}

fn encode_messages(
    messages: &[Message],
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<(Option<Vec<Value>>, Vec<Value>), Error> {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut ordinary_turn_seen = false;

    for message in messages {
        match message.role() {
            MessageRole::System | MessageRole::Developer => {
                if ordinary_turn_seen {
                    return Err(Error::new(
                        ErrorKind::Unsupported,
                        "Gemini cannot preserve a system or developer instruction after conversation content",
                    ));
                }
                for part in message.content() {
                    let ContentPart::Text { text } = part.content() else {
                        return Err(Error::new(
                            ErrorKind::Unsupported,
                            "Gemini system and developer instructions support text only",
                        ));
                    };
                    system_parts.push(json!({"text": text}));
                }
            }
            MessageRole::User => {
                ordinary_turn_seen = true;
                let parts = message
                    .content()
                    .iter()
                    .map(|part| match part.content() {
                        ContentPart::Text { text } => Ok(json!({"text": text})),
                        ContentPart::Media(media) => encode_media(media),
                        _ => Err(Error::new(
                            ErrorKind::InvalidInput,
                            "Gemini user content contains an unsupported part",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                push_content(&mut contents, "user", parts)?;
            }
            MessageRole::Assistant => {
                ordinary_turn_seen = true;
                let parts = encode_assistant_message(message, scope, model)?;
                push_content(&mut contents, "model", parts)?;
            }
            MessageRole::Tool => {
                ordinary_turn_seen = true;
                let parts = message
                    .content()
                    .iter()
                    .map(|part| match part.content() {
                        ContentPart::ToolResult(result) => encode_tool_result(result),
                        _ => Err(Error::new(
                            ErrorKind::InvalidInput,
                            "Gemini tool message contains an unsupported part",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                push_content(&mut contents, "user", parts)?;
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "Gemini Generate Content does not support this message role",
                ));
            }
        }
    }

    Ok(((!system_parts.is_empty()).then_some(system_parts), contents))
}

fn push_content(contents: &mut Vec<Value>, role: &str, parts: Vec<Value>) -> Result<(), Error> {
    if parts.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini content turns must not be empty",
        ));
    }
    if parts.len() > MAX_PARTS_PER_CONTENT {
        return Err(Error::new(
            ErrorKind::LimitExceeded,
            "Gemini content turn exceeded the part limit",
        ));
    }
    contents.push(json!({"role": role, "parts": parts}));
    Ok(())
}

fn encode_assistant_message(
    message: &Message,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<Vec<Value>, Error> {
    let mut suppressed = BTreeSet::new();
    for (opaque_index, part) in message.content().iter().enumerate() {
        let ContentPart::ProviderOpaque(opaque) = part.content() else {
            continue;
        };
        validate_replay_item(opaque, scope, model)?;
        let raw = opaque.data();
        if let Some(call) = raw.get("functionCall") {
            let name = required_text(call, "name", "Gemini replay function call omitted its name")?;
            let arguments = call.get("args").cloned().unwrap_or_else(|| json!({}));
            if !arguments.is_object() {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini replay function-call arguments must be a JSON object",
                ));
            }
            let caller_id = opaque
                .relations()
                .iter()
                .find(|relation| relation.kind() == "call")
                .map(ProviderItemRelation::target_id)
                .or_else(|| call.get("id").and_then(Value::as_str));
            if let Some(caller_id) = caller_id
                && let Some((index, sibling)) = message
                    .content()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, candidate)| match candidate.content() {
                        ContentPart::ToolCall(call) if call.id() == caller_id => {
                            Some((index, call))
                        }
                        _ => None,
                    })
                    .next()
            {
                if sibling.name() != name || sibling.arguments() != &arguments {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "Gemini native function-call replay disagrees with its portable sibling",
                    ));
                }
                suppressed.insert(index);
            }
        } else if raw.get("thought").and_then(Value::as_bool) == Some(true)
            && let Some(text) = raw.get("text").and_then(Value::as_str)
            && let Some((index, _)) = message
                .content()
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != opaque_index && !suppressed.contains(index))
                .find(|(_, candidate)| {
                    matches!(candidate.content(), ContentPart::Reasoning { text: sibling } if sibling == text)
                })
        {
            suppressed.insert(index);
        } else if let Some(text) = raw.get("text").and_then(Value::as_str)
            && let Some((index, _)) = message
                .content()
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != opaque_index && !suppressed.contains(index))
                .find(|(_, candidate)| {
                    matches!(candidate.content(), ContentPart::Text { text: sibling } if sibling == text)
                })
        {
            suppressed.insert(index);
        }
    }

    let mut parts = Vec::new();
    for (index, part) in message.content().iter().enumerate() {
        if suppressed.contains(&index) {
            continue;
        }
        match part.content() {
            ContentPart::Text { text } => parts.push(json!({"text": text})),
            ContentPart::Reasoning { text } => {
                parts.push(json!({"text": text, "thought": true}));
            }
            ContentPart::Media(media) => parts.push(encode_media(media)?),
            ContentPart::ToolCall(call) => {
                if !call.arguments().is_object() {
                    return Err(Error::new(
                        ErrorKind::Unsupported,
                        "Gemini function-call arguments must be a JSON object",
                    ));
                }
                parts.push(json!({
                    "functionCall": {
                        "id": call.id(),
                        "name": call.name(),
                        "args": call.arguments(),
                    }
                }));
            }
            ContentPart::ProviderOpaque(opaque) => parts.push(opaque.data().clone()),
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini assistant history contains an unsupported content part",
                ));
            }
        }
    }
    Ok(parts)
}

fn validate_replay_item(
    item: &OpaqueProviderItem,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<(), Error> {
    if item.kind() != GENERATE_CONTENT_PART_KIND
        || !item.provenance().matches_replay_target(scope)
        || item.provenance().model() != model
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Generate Content replay state does not match the target execution scope",
        ));
    }
    if !item.data().is_object() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Generate Content replay state is malformed",
        ));
    }
    Ok(())
}

fn encode_media(media: &MediaPart) -> Result<Value, Error> {
    if media.name.is_some() {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "Gemini Generate Content cannot preserve portable media names",
        ));
    }
    if !valid_media_type(&media.media_type) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini media input has an invalid MIME type",
        ));
    }
    match &media.data {
        MediaData::Bytes(bytes) => {
            if bytes.is_empty() || bytes.len() > MAX_MEDIA_BYTES {
                return Err(Error::new(
                    ErrorKind::LimitExceeded,
                    "Gemini inline media violated the protocol media limit",
                ));
            }
            Ok(json!({
                "inlineData": {
                    "mimeType": media.media_type,
                    "data": base64::engine::general_purpose::STANDARD.encode(bytes),
                }
            }))
        }
        MediaData::Url(uri) => {
            if !valid_bounded_text(uri, MAX_METADATA_TEXT_BYTES * 4) {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini file media URI is invalid",
                ));
            }
            Ok(json!({
                "fileData": {
                    "mimeType": media.media_type,
                    "fileUri": uri,
                }
            }))
        }
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "Gemini Generate Content does not support this media carrier",
        )),
    }
}

fn encode_tool_result(result: &ToolResult) -> Result<Value, Error> {
    if !valid_bounded_text(&result.call_id, MAX_METADATA_TEXT_BYTES)
        || !valid_bounded_text(&result.name, 128)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini tool result contains an invalid identity",
        ));
    }
    let response = match &result.outcome {
        ToolOutcome::Success { value } => match value {
            Value::Object(_) => value.clone(),
            value => json!({"output": value}),
        },
        ToolOutcome::Denied { reason } => json!({
            "error": {"type": "denied", "message": reason}
        }),
        ToolOutcome::ExecutionFailed {
            message,
            retryable,
            details,
        } => json!({
            "error": {
                "type": "execution_failed",
                "message": message,
                "retryable": retryable,
                "details": details,
            }
        }),
        ToolOutcome::Cancelled { reason } => json!({
            "error": {"type": "cancelled", "message": reason}
        }),
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini does not support this tool-result outcome",
            ));
        }
    };
    ensure_encoded_value_limit(
        &response,
        MAX_PROVIDER_METADATA_BYTES,
        "Gemini tool result exceeded the protocol value limit",
    )?;
    Ok(json!({
        "functionResponse": {
            "id": result.call_id,
            "name": result.name,
            "response": response,
        }
    }))
}

fn encode_tools(request: &LanguageRequest) -> Result<Value, Error> {
    let declarations = request
        .tools
        .iter()
        .map(|tool| {
            if !tool.input_schema().is_object() {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "Gemini function declarations require an object schema",
                ));
            }
            let description = tool.description().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini function declarations require a description",
                )
            })?;
            Ok(json!({
                "name": tool.name(),
                "description": description,
                "parameters": tool.input_schema(),
            }))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(json!([{"functionDeclarations": declarations}]))
}

fn encode_tool_choice(request: &LanguageRequest) -> Result<Option<Value>, Error> {
    let Some(choice) = &request.tool_choice else {
        return Ok(None);
    };
    if request.tools.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini tool choice requires at least one function declaration",
        ));
    }
    let config = match choice {
        ToolChoice::Auto => json!({"mode": "AUTO"}),
        ToolChoice::None => json!({"mode": "NONE"}),
        ToolChoice::Required => json!({"mode": "ANY"}),
        ToolChoice::Named { name } => {
            json!({"mode": "ANY", "allowedFunctionNames": [name]})
        }
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini does not support this portable tool choice",
            ));
        }
    };
    Ok(Some(json!({"functionCallingConfig": config})))
}

pub(crate) fn project_response_value(
    native: &Value,
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<Result<LanguageResponse, LanguageCallError>, Error> {
    let wire = serde_json::from_value::<GenerateContentResponseWire>(native.clone()).map_err(
        |source| {
            Error::protocol_violation("Gemini Generate Content response is malformed")
                .with_source(source)
        },
    )?;
    if wire.candidates.len() > MAX_CANDIDATES {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "portable Gemini Generate Content responses support exactly one candidate",
        ));
    }

    let response_id = checked_optional_text(
        wire.response_id,
        MAX_RESPONSE_ID_BYTES,
        "response identifier",
    )?;
    let response_model = wire
        .model_version
        .map(ModelId::new)
        .transpose()
        .map_err(|source| {
            Error::protocol_violation("Gemini returned an invalid response model")
                .with_source(source)
        })?
        .unwrap_or_else(|| requested_model.clone());
    let usage = decode_usage(wire.usage_metadata)?;
    let mut provider = BTreeMap::new();
    retain_provider_metadata(
        &mut provider,
        "google.generate_content.prompt_feedback",
        wire.prompt_feedback.as_ref(),
    )?;
    retain_provider_metadata(
        &mut provider,
        "google.generate_content.model_status",
        wire.model_status.as_ref(),
    )?;

    let (terminal, content, warnings) = if let Some(candidate) = wire.candidates.first() {
        if candidate.index.unwrap_or(0) != 0 {
            return Err(Error::protocol_violation(
                "Gemini portable response candidate had a non-zero index",
            ));
        }
        let content_wire = candidate.content.clone().unwrap_or_default();
        if content_wire
            .role
            .as_deref()
            .is_some_and(|role| role != "model")
        {
            return Err(Error::protocol_violation(
                "Gemini response candidate used an invalid content role",
            ));
        }
        if content_wire.parts.len() > MAX_PARTS_PER_CONTENT {
            return Err(Error::new(
                ErrorKind::ResponseLimit,
                "Gemini response candidate exceeded the content-part limit",
            ));
        }
        let parts = normalize_response_parts(content_wire.parts)?;
        let mut projected = Vec::new();
        let mut warnings = Vec::new();
        for (part_index, part) in parts.iter().enumerate() {
            project_response_part(
                part,
                scope,
                requested_model,
                response_id.as_deref(),
                0,
                part_index,
                &mut projected,
                &mut warnings,
            )?;
        }
        let terminal = map_finish_reason(
            candidate.finish_reason.as_deref(),
            projected
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall(_))),
        )?;
        let metadata = candidate.metadata_value();
        retain_provider_metadata(
            &mut provider,
            "google.generate_content.candidate",
            Some(&metadata),
        )?;
        (terminal, projected, warnings)
    } else if prompt_block_reason(wire.prompt_feedback.as_ref())?.is_some() {
        (
            GenerateContentTerminal::Success(LanguageTermination::Incomplete(
                LanguageIncompleteReason::ContentFilter,
            )),
            vec![ContentPart::Refusal { reason: None }],
            Vec::new(),
        )
    } else {
        return Err(Error::protocol_violation(
            "Gemini Generate Content response omitted both candidates and prompt-block feedback",
        ));
    };

    let termination = match terminal {
        GenerateContentTerminal::Success(termination) => termination,
        GenerateContentTerminal::Failed => {
            let error = Error::new(
                ErrorKind::Provider,
                "Gemini Generate Content failed to produce a valid result",
            );
            return Ok(Err(LanguageCallError::new(
                error,
                partial_output(&content, &usage),
            )));
        }
    };

    let mut response = LanguageResponse::new(termination, content, usage)
        .map_err(|source| {
            Error::protocol_violation(
                "Gemini Generate Content produced an invalid portable response",
            )
            .with_source(source)
        })?
        .with_model(response_model)
        .with_warnings(warnings)
        .with_provider_metadata(provider);
    if let Some(response_id) = response_id {
        response = response.with_id(response_id);
    }
    Ok(Ok(response))
}

pub(crate) fn partial_output(
    content: &[ContentPart],
    usage: &Usage,
) -> Option<PartialLanguageOutput> {
    let content = content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => {
                Some(PartialLanguageOutputPart::Text { text: text.clone() })
            }
            ContentPart::Reasoning { text } => {
                Some(PartialLanguageOutputPart::Reasoning { text: text.clone() })
            }
            ContentPart::Refusal { reason } => Some(PartialLanguageOutputPart::Refusal {
                reason: reason.clone(),
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    let observed_usage = [
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.reasoning_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.audio_input_tokens,
        usage.audio_output_tokens,
        usage.orchestration_tokens,
    ]
    .into_iter()
    .any(|value| matches!(value, UsageValue::Known(_)));
    if content.is_empty() && !observed_usage {
        return None;
    }
    PartialLanguageOutput::new(content, usage.clone()).ok()
}

pub(crate) fn normalize_response_parts(parts: Vec<Value>) -> Result<Vec<Value>, Error> {
    let mut normalized: Vec<Value> = Vec::with_capacity(parts.len());
    for part in parts {
        ensure_encoded_value_limit(
            &part,
            MAX_PROVIDER_METADATA_BYTES,
            "Gemini response part exceeded the protocol value limit",
        )?;
        let merge_text = part.get("text").and_then(Value::as_str).map(|text| {
            (
                text,
                part.get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            )
        });
        if let Some((text, thought)) = merge_text
            && let Some(previous) = normalized.last_mut()
            && previous.get("text").and_then(Value::as_str).is_some()
            && previous
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                == thought
            && compatible_text_part(previous)
            && compatible_text_part(&part)
            && compatible_text_signatures(previous, &part)
        {
            let previous_text = previous
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            previous["text"] = Value::String(previous_text + text);
            if let Some(signature) = part.get("thoughtSignature") {
                previous["thoughtSignature"] = signature.clone();
            }
            continue;
        }
        normalized.push(part);
    }
    Ok(normalized)
}

fn compatible_text_part(part: &Value) -> bool {
    part.as_object().is_some_and(|object| {
        object
            .keys()
            .all(|key| matches!(key.as_str(), "text" | "thought" | "thoughtSignature"))
    })
}

fn compatible_text_signatures(left: &Value, right: &Value) -> bool {
    match (left.get("thoughtSignature"), right.get("thoughtSignature")) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn project_response_part(
    part: &Value,
    scope: &ProviderScope,
    model: &ModelId,
    response_id: Option<&str>,
    candidate_index: usize,
    part_index: usize,
    content: &mut Vec<ContentPart>,
    warnings: &mut Vec<Warning>,
) -> Result<(), Error> {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            content.push(ContentPart::Reasoning {
                text: text.to_string(),
            });
            content.push(ContentPart::ProviderOpaque(opaque_part(
                part, scope, model, None, None,
            )?));
        } else {
            content.push(ContentPart::Text {
                text: text.to_string(),
            });
            if part.get("thoughtSignature").is_some() || part.get("partMetadata").is_some() {
                content.push(ContentPart::ProviderOpaque(opaque_part(
                    part, scope, model, None, None,
                )?));
            }
        }
        return Ok(());
    }
    if part.get("inlineData").is_some() || part.get("fileData").is_some() {
        content.push(ContentPart::Media(decode_media(part)?));
        return Ok(());
    }
    if let Some(call) = part.get("functionCall") {
        let name = required_text(call, "name", "Gemini function call omitted its name")?;
        let arguments = call.get("args").cloned().unwrap_or_else(|| json!({}));
        if !arguments.is_object() {
            return Err(Error::protocol_violation(
                "Gemini function-call arguments were not a JSON object",
            ));
        }
        let wire_id = checked_optional_text(
            call.get("id").and_then(Value::as_str).map(str::to_string),
            MAX_METADATA_TEXT_BYTES,
            "function call identifier",
        )?;
        let public_id = match wire_id.clone() {
            Some(wire_id) => wire_id,
            None => synthetic_call_id(response_id, candidate_index, part_index)?,
        };
        let portable = ToolCall::local(public_id.clone(), name, arguments).map_err(|source| {
            Error::protocol_violation("Gemini returned an invalid caller-executed function call")
                .with_source(source)
        })?;
        content.push(ContentPart::ToolCall(portable));
        content.push(ContentPart::ProviderOpaque(opaque_part(
            part,
            scope,
            model,
            wire_id.as_deref(),
            Some(public_id.as_str()),
        )?));
        return Ok(());
    }

    content.push(ContentPart::ProviderOpaque(opaque_part(
        part, scope, model, None, None,
    )?));
    warnings.push(Warning::new(
        WarningKind::UnsupportedOption,
        "Gemini returned a provider-native Generate Content part retained as replay data",
    ));
    Ok(())
}

fn decode_media(part: &Value) -> Result<MediaPart, Error> {
    if let Some(inline) = part.get("inlineData") {
        let media_type = required_text(
            inline,
            "mimeType",
            "Gemini inline media omitted its MIME type",
        )?;
        if !valid_media_type(media_type) {
            return Err(Error::protocol_violation(
                "Gemini returned an invalid inline-media MIME type",
            ));
        }
        let encoded = required_text(
            inline,
            "data",
            "Gemini inline media omitted its base64 data",
        )?;
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
        return Ok(MediaPart {
            media_type: media_type.to_string(),
            data: MediaData::Bytes(decoded.into()),
            name: None,
        });
    }
    if let Some(file) = part.get("fileData") {
        let media_type =
            required_text(file, "mimeType", "Gemini file media omitted its MIME type")?;
        let uri = required_text(file, "fileUri", "Gemini file media omitted its URI")?;
        if !valid_media_type(media_type) || !valid_bounded_text(uri, MAX_METADATA_TEXT_BYTES * 4) {
            return Err(Error::protocol_violation(
                "Gemini returned invalid file media",
            ));
        }
        return Ok(MediaPart {
            media_type: media_type.to_string(),
            data: MediaData::Url(uri.to_string()),
            name: None,
        });
    }
    Err(Error::protocol_violation(
        "Gemini media part omitted inline or file data",
    ))
}

fn opaque_part(
    part: &Value,
    scope: &ProviderScope,
    model: &ModelId,
    wire_id: Option<&str>,
    public_call_id: Option<&str>,
) -> Result<OpaqueProviderItem, Error> {
    let provenance = ProviderProvenance::from_scope(scope, model.clone()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Gemini Generate Content replay requires an explicit replay domain",
        )
        .with_source(source)
    })?;
    let mut builder =
        OpaqueProviderItem::builder(provenance, GENERATE_CONTENT_PART_KIND, part.clone());
    if let Some(wire_id) = wire_id {
        builder = builder.item_id(wire_id);
    }
    if let Some(public_call_id) = public_call_id {
        let relation = ProviderItemRelation::call(public_call_id).map_err(|source| {
            Error::protocol_violation("Gemini function-call relation is invalid")
                .with_source(source)
        })?;
        builder = builder.relations([relation]);
    }
    builder.build().map_err(|source| {
        Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Generate Content part exceeded the opaque replay limit",
        )
        .with_source(source)
    })
}

fn synthetic_call_id(
    response_id: Option<&str>,
    candidate_index: usize,
    part_index: usize,
) -> Result<String, Error> {
    let response_id = response_id.ok_or_else(|| {
        Error::protocol_violation(
            "Gemini function call omitted both its call ID and the enclosing response ID",
        )
    })?;
    Ok(format!(
        "gemini-call:{}:{response_id}:{candidate_index}:{part_index}",
        response_id.len()
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GenerateContentTerminal {
    Success(LanguageTermination),
    Failed,
}

fn map_finish_reason(
    reason: Option<&str>,
    has_tool_call: bool,
) -> Result<GenerateContentTerminal, Error> {
    let reason = reason.ok_or_else(|| {
        Error::protocol_violation("Gemini terminal response omitted its finish reason")
    })?;
    let value = match reason {
        "STOP" if has_tool_call => GenerateContentTerminal::Success(
            LanguageTermination::Completed(LanguageCompletionReason::ToolCalls),
        ),
        "STOP" => GenerateContentTerminal::Success(LanguageTermination::Completed(
            LanguageCompletionReason::Stop,
        )),
        "MAX_TOKENS" => GenerateContentTerminal::Success(LanguageTermination::Incomplete(
            LanguageIncompleteReason::MaxOutputTokens,
        )),
        "SAFETY"
        | "RECITATION"
        | "LANGUAGE"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "IMAGE_RECITATION"
        | "ESCALATION" => GenerateContentTerminal::Success(LanguageTermination::Incomplete(
            LanguageIncompleteReason::ContentFilter,
        )),
        "MALFORMED_FUNCTION_CALL"
        | "MALFORMED_RESPONSE"
        | "UNEXPECTED_TOOL_CALL"
        | "TOO_MANY_TOOL_CALLS"
        | "MISSING_THOUGHT_SIGNATURE"
        | "NO_IMAGE"
        | "IMAGE_OTHER" => GenerateContentTerminal::Failed,
        "FINISH_REASON_UNSPECIFIED" => {
            return Err(Error::protocol_violation(
                "Gemini terminal response used an unspecified finish reason",
            ));
        }
        other if valid_bounded_text(other, 128) => GenerateContentTerminal::Success(
            LanguageTermination::Incomplete(LanguageIncompleteReason::Other(other.to_string())),
        ),
        _ => {
            return Err(Error::protocol_violation(
                "Gemini returned an invalid finish reason",
            ));
        }
    };
    Ok(value)
}

pub(crate) fn decode_usage(wire: Option<UsageMetadataWire>) -> Result<Usage, Error> {
    let Some(wire) = wire else {
        return Ok(Usage::default());
    };
    validate_modality_entries(wire.prompt_tokens_details.as_deref())?;
    validate_modality_entries(wire.candidates_tokens_details.as_deref())?;
    validate_modality_entries(wire.cache_tokens_details.as_deref())?;
    validate_modality_entries(wire.tool_use_prompt_tokens_details.as_deref())?;

    let audio_input = sum_modality(wire.prompt_tokens_details.as_deref(), "AUDIO")?;
    let audio_output = sum_modality(wire.candidates_tokens_details.as_deref(), "AUDIO")?;
    let cached = sum_all_modalities(wire.cache_tokens_details.as_deref())?;
    let mut usage = Usage::default()
        .with_input_tokens(wire.prompt_token_count)
        .with_output_tokens(wire.candidates_token_count)
        .with_total_tokens(wire.total_token_count)
        .with_reasoning_tokens(wire.thoughts_token_count)
        .with_cache_read_tokens(cached)
        .with_audio_input_tokens(audio_input)
        .with_audio_output_tokens(audio_output)
        .with_orchestration_tokens(wire.tool_use_prompt_token_count);

    for (name, value) in [
        (
            "google.prompt_tokens_by_modality",
            wire.prompt_tokens_details.as_ref(),
        ),
        (
            "google.output_tokens_by_modality",
            wire.candidates_tokens_details.as_ref(),
        ),
        (
            "google.cache_tokens_by_modality",
            wire.cache_tokens_details.as_ref(),
        ),
        (
            "google.tool_use_tokens_by_modality",
            wire.tool_use_prompt_tokens_details.as_ref(),
        ),
    ] {
        if let Some(value) = value {
            usage = usage.with_provider_value(
                name,
                serde_json::to_value(value).map_err(|source| {
                    Error::new(
                        ErrorKind::Internal,
                        "Gemini usage modalities could not be retained",
                    )
                    .with_source(source)
                })?,
            );
        }
    }
    if let Some(service_tier) = wire.service_tier {
        if !valid_bounded_text(&service_tier, 64) {
            return Err(Error::protocol_violation(
                "Gemini usage contained an invalid service tier",
            ));
        }
        usage = usage.with_provider_value("google.service_tier", service_tier);
    }
    Ok(usage)
}

fn validate_modality_entries(entries: Option<&[ModalityTokenCountWire]>) -> Result<(), Error> {
    if entries.is_some_and(|entries| {
        entries.len() > MAX_MODALITY_ENTRIES
            || entries.iter().any(|entry| {
                !valid_bounded_text(&entry.modality, MAX_MODALITY_BYTES)
                    || !entry.modality.is_ascii()
            })
    }) {
        Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini usage modalities exceeded the protocol limit",
        ))
    } else {
        Ok(())
    }
}

fn sum_modality(
    entries: Option<&[ModalityTokenCountWire]>,
    modality: &str,
) -> Result<Option<u64>, Error> {
    let Some(entries) = entries else {
        return Ok(None);
    };
    entries
        .iter()
        .filter(|entry| entry.modality == modality)
        .try_fold(None::<u64>, |total, entry| {
            total
                .unwrap_or(0)
                .checked_add(entry.token_count)
                .map(Some)
                .ok_or_else(|| Error::protocol_violation("Gemini modality token counts overflowed"))
        })
}

fn sum_all_modalities(entries: Option<&[ModalityTokenCountWire]>) -> Result<Option<u64>, Error> {
    let Some(entries) = entries else {
        return Ok(None);
    };
    entries
        .iter()
        .try_fold(0_u64, |total, entry| {
            total
                .checked_add(entry.token_count)
                .ok_or_else(|| Error::protocol_violation("Gemini modality token counts overflowed"))
        })
        .map(Some)
}

fn retain_provider_metadata(
    provider: &mut BTreeMap<String, Value>,
    key: &str,
    value: Option<&Value>,
) -> Result<(), Error> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() || value.as_object().is_some_and(Map::is_empty) {
        return Ok(());
    }
    ensure_encoded_value_limit(
        value,
        MAX_PROVIDER_METADATA_BYTES,
        "Gemini provider metadata exceeded the protocol limit",
    )?;
    provider.insert(key.to_string(), value.clone());
    Ok(())
}

fn prompt_block_reason(prompt_feedback: Option<&Value>) -> Result<Option<&str>, Error> {
    let Some(prompt_feedback) = prompt_feedback else {
        return Ok(None);
    };
    let reason = prompt_feedback.get("blockReason").and_then(Value::as_str);
    if reason.is_some_and(|reason| !valid_bounded_text(reason, 128)) {
        return Err(Error::protocol_violation(
            "Gemini prompt feedback contained an invalid block reason",
        ));
    }
    Ok(reason.filter(|reason| *reason != "BLOCK_REASON_UNSPECIFIED"))
}

fn required_text<'a>(
    value: &'a Value,
    field: &str,
    message: &'static str,
) -> Result<&'a str, Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| valid_bounded_text(value, MAX_METADATA_TEXT_BYTES))
        .ok_or_else(|| Error::protocol_violation(message))
}

fn checked_optional_text(
    value: Option<String>,
    maximum: usize,
    _field: &'static str,
) -> Result<Option<String>, Error> {
    value
        .map(|value| {
            if valid_bounded_text(&value, maximum) {
                Ok(value)
            } else {
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

fn valid_media_type(value: &str) -> bool {
    valid_bounded_text(value, 256)
        && value.contains('/')
        && value.is_ascii()
        && !value.chars().any(char::is_whitespace)
}

fn ensure_encoded_value_limit(
    value: &Value,
    maximum: usize,
    message: &'static str,
) -> Result<(), Error> {
    let length = serde_json::to_vec(value)
        .map_err(|source| {
            Error::new(
                ErrorKind::Internal,
                "Gemini protocol value could not be measured",
            )
            .with_source(source)
        })?
        .len();
    if length > maximum {
        Err(Error::new(ErrorKind::ResponseLimit, message))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ContentWire {
    #[serde(default)]
    pub(crate) role: Option<String>,
    #[serde(default)]
    pub(crate) parts: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentResponseWire {
    #[serde(default)]
    candidates: Vec<CandidateWire>,
    #[serde(default)]
    prompt_feedback: Option<Value>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadataWire>,
    #[serde(default)]
    model_version: Option<String>,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    model_status: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateWire {
    #[serde(default)]
    content: Option<ContentWire>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    safety_ratings: Option<Value>,
    #[serde(default)]
    citation_metadata: Option<Value>,
    #[serde(default)]
    grounding_attributions: Option<Value>,
    #[serde(default)]
    grounding_metadata: Option<Value>,
    #[serde(default)]
    url_context_metadata: Option<Value>,
    #[serde(default)]
    finish_message: Option<String>,
    #[serde(default)]
    token_count: Option<u64>,
    #[serde(default)]
    avg_logprobs: Option<f64>,
    #[serde(default)]
    logprobs_result: Option<Value>,
}

impl CandidateWire {
    fn metadata_value(&self) -> Value {
        let mut metadata = Map::new();
        if let Some(index) = self.index {
            metadata.insert("index".to_string(), Value::from(index));
        }
        if let Some(reason) = &self.finish_reason {
            metadata.insert("finishReason".to_string(), Value::String(reason.clone()));
        }
        for (key, value) in [
            ("safetyRatings", self.safety_ratings.as_ref()),
            ("citationMetadata", self.citation_metadata.as_ref()),
            (
                "groundingAttributions",
                self.grounding_attributions.as_ref(),
            ),
            ("groundingMetadata", self.grounding_metadata.as_ref()),
            ("urlContextMetadata", self.url_context_metadata.as_ref()),
            ("logprobsResult", self.logprobs_result.as_ref()),
        ] {
            if let Some(value) = value {
                metadata.insert(key.to_string(), value.clone());
            }
        }
        if let Some(message) = &self.finish_message {
            metadata.insert("finishMessage".to_string(), Value::String(message.clone()));
        }
        if let Some(count) = self.token_count {
            metadata.insert("tokenCount".to_string(), Value::from(count));
        }
        if let Some(logprobs) = self.avg_logprobs {
            metadata.insert("avgLogprobs".to_string(), Value::from(logprobs));
        }
        Value::Object(metadata)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UsageMetadataWire {
    #[serde(default)]
    pub(crate) prompt_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) candidates_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) total_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) thoughts_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) tool_use_prompt_token_count: Option<u64>,
    #[serde(default)]
    pub(crate) prompt_tokens_details: Option<Vec<ModalityTokenCountWire>>,
    #[serde(default)]
    pub(crate) candidates_tokens_details: Option<Vec<ModalityTokenCountWire>>,
    #[serde(default)]
    pub(crate) cache_tokens_details: Option<Vec<ModalityTokenCountWire>>,
    #[serde(default)]
    pub(crate) tool_use_prompt_tokens_details: Option<Vec<ModalityTokenCountWire>>,
    #[serde(default)]
    pub(crate) service_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModalityTokenCountWire {
    pub(crate) modality: String,
    pub(crate) token_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{
        ApiModeId, MessagePart, PlatformId, ProtocolId, ProviderId, ReplayDomain, ReplayDomainId,
        StructuredOutputSpec, ToolSpec,
    };

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new("google").unwrap())
            .with_platform(PlatformId::new("gemini-api").unwrap())
            .with_protocol(ProtocolId::new("gemini-generate-content").unwrap())
            .with_api_mode(ApiModeId::new("generate-content").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("google-gemini-api").unwrap(),
            ))
    }

    #[test]
    fn request_preserves_instruction_order_and_uses_current_response_format() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let tool = ToolSpec::new(
            "lookup",
            Some("Lookup a value".to_string()),
            json!({"type": "object", "properties": {"key": {"type": "string"}}}),
        )
        .unwrap();
        let user = Message::user_parts([
            MessagePart::text("lookup x"),
            MessagePart::new(ContentPart::Media(MediaPart {
                media_type: "image/png".to_string(),
                data: MediaData::Bytes(vec![1, 2, 3].into()),
                name: None,
            })),
        ])
        .unwrap();
        let mut request = LanguageRequest::new(vec![
            Message::system("policy-a"),
            Message::developer("policy-b"),
            user,
        ]);
        request.generation.max_output_tokens = Some(128);
        request.generation.temperature = Some(0.7);
        request.generation.top_p = Some(0.9);
        request.generation.stop_sequences = vec!["END".to_string()];
        request.generation.seed = Some(7);
        request.tools.push(tool);
        request.tool_choice = Some(ToolChoice::Named {
            name: "lookup".to_string(),
        });
        request.structured_output = Some(StructuredOutputSpec {
            name: "answer".to_string(),
            description: None,
            schema: json!({"type": "object"}),
            strict: true,
        });

        let body = encode_language_request(
            &request,
            &model,
            &scope(),
            &GenerateContentLanguageConfig::default(),
        )
        .unwrap();

        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "policy-a");
        assert_eq!(body["systemInstruction"]["parts"][1]["text"], "policy-b");
        assert_eq!(
            body["generationConfig"]["responseFormat"]["text"]["mimeType"],
            "APPLICATION_JSON"
        );
        assert!(body["generationConfig"].get("responseMimeType").is_none());
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            "lookup"
        );
    }

    #[test]
    fn direct_response_preserves_synthetic_call_identity_and_raw_replay() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let body = serde_json::to_vec(&json!({
            "responseId": "response-1",
            "modelVersion": model.as_str(),
            "candidates": [{
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "reason", "thought": true, "thoughtSignature": "signed"},
                        {"functionCall": {"name": "lookup", "args": {"key": "x"}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 2,
                "totalTokenCount": 6
            }
        }))
        .unwrap();

        let decoded = decode_language_response(&body, &scope(), &model).unwrap();
        let response = decoded.portable().unwrap();
        let call = response
            .content()
            .iter()
            .find_map(|part| match part {
                ContentPart::ToolCall(call) => Some(call),
                _ => None,
            })
            .unwrap();
        assert_eq!(call.id(), "gemini-call:10:response-1:0:1");
        let opaque = response
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::ProviderOpaque(item) if item.data().get("functionCall").is_some() => {
                    Some(item)
                }
                _ => None,
            })
            .next()
            .unwrap();
        assert!(opaque.item_id().is_none());
        assert!(opaque.data()["functionCall"].get("id").is_none());
        assert_eq!(opaque.relations()[0].target_id(), call.id());

        let history = response.project_assistant_history().into_message().unwrap();
        let replay = LanguageRequest::new(vec![history, Message::user("continue")]);
        let encoded = encode_language_request(
            &replay,
            &model,
            &scope(),
            &GenerateContentLanguageConfig::default(),
        )
        .unwrap();
        assert!(
            encoded["contents"][0]["parts"][1]["functionCall"]
                .get("id")
                .is_none()
        );
    }

    #[test]
    fn synthetic_call_ids_are_response_scoped_and_fail_without_response_identity() {
        let first = synthetic_call_id(Some("response-a"), 0, 0).unwrap();
        let second = synthetic_call_id(Some("response-b"), 0, 0).unwrap();
        assert_ne!(first, second);
        assert_eq!(first, "gemini-call:10:response-a:0:0");
        assert!(matches!(
            synthetic_call_id(None, 0, 0),
            Err(error) if error.kind() == ErrorKind::ProtocolViolation
        ));
    }

    #[test]
    fn tool_results_use_explicit_object_envelopes() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let request = LanguageRequest::new(vec![Message::tool_result(ToolResult {
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
            outcome: ToolOutcome::Success { value: json!(42) },
        })]);
        let body = encode_language_request(
            &request,
            &model,
            &scope(),
            &GenerateContentLanguageConfig::default(),
        )
        .unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0]["functionResponse"]["response"],
            json!({"output": 42})
        );
    }

    #[test]
    fn blocked_prompt_is_a_typed_incomplete_response() {
        let model = ModelId::new("gemini-3.6-flash").unwrap();
        let body = serde_json::to_vec(&json!({
            "responseId": "blocked-1",
            "modelVersion": model.as_str(),
            "promptFeedback": {"blockReason": "SAFETY", "safetyRatings": []},
            "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3}
        }))
        .unwrap();
        let response = decode_language_response(&body, &scope(), &model)
            .unwrap()
            .into_result()
            .unwrap();
        assert_eq!(
            response.termination(),
            &LanguageTermination::Incomplete(LanguageIncompleteReason::ContentFilter)
        );
    }
}
