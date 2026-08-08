use std::collections::{BTreeMap, BTreeSet};

use base64::Engine;
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, Error, ErrorKind, LanguageRequest, MediaData, Message, MessageRole, ModelId,
    ProviderScope, ToolChoice, ToolOutcome,
};

use super::ChatCompletionsDialect;
use super::reasoning::replay_reasoning_details;

pub const CHAT_COMPLETIONS_TARGET: &str = "chat/completions";
const MAX_PROMPT_CACHE_BREAKPOINTS: usize = 50;

const PROTECTED_FIELDS: &[&str] = &[
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "response_format",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "n",
];

pub fn is_protected_option_field(name: &str) -> bool {
    PROTECTED_FIELDS.contains(&name)
}

/// One canonical content block that should carry an explicit cache breakpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChatPromptCacheBlock {
    pub message_index: usize,
    pub content_index: usize,
}

impl ChatPromptCacheBlock {
    pub const fn new(message_index: usize, content_index: usize) -> Self {
        Self {
            message_index,
            content_index,
        }
    }
}

/// Protocol-owned Chat Completions request shaping.
#[derive(Debug, Clone, Default)]
pub struct ChatRequestEncodingOptions {
    stream: bool,
    extra: BTreeMap<String, Value>,
    prompt_cache_breakpoints: BTreeSet<ChatPromptCacheBlock>,
}

impl ChatRequestEncodingOptions {
    pub fn new(stream: bool) -> Self {
        Self {
            stream,
            ..Self::default()
        }
    }

    pub fn with_extra(mut self, extra: BTreeMap<String, Value>) -> Self {
        self.extra = extra;
        self
    }

    pub fn with_prompt_cache_breakpoint(mut self, block: ChatPromptCacheBlock) -> Self {
        self.prompt_cache_breakpoints.insert(block);
        self
    }
}

pub fn encode_request(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    stream: bool,
    dialect: &ChatCompletionsDialect,
    extra: &BTreeMap<String, Value>,
) -> Result<Value, Error> {
    encode_request_with_options(
        scope,
        model,
        request,
        dialect,
        &ChatRequestEncodingOptions::new(stream).with_extra(extra.clone()),
    )
}

pub fn encode_request_with_options(
    scope: &ProviderScope,
    model: &ModelId,
    request: &LanguageRequest,
    dialect: &ChatCompletionsDialect,
    options: &ChatRequestEncodingOptions,
) -> Result<Value, Error> {
    dialect
        .validate_reasoning_configuration()
        .map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "Chat Completions reasoning dialect configuration is invalid",
            )
            .with_source(source)
        })?;
    request.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "language request is invalid").with_source(source)
    })?;
    if options.prompt_cache_breakpoints.len() > MAX_PROMPT_CACHE_BREAKPOINTS {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI Chat Completions accepts at most 50 prompt-cache breakpoints per request",
        ));
    }

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));
    body.insert(
        "messages".to_string(),
        Value::Array(encode_messages(
            scope,
            model,
            &request.messages,
            dialect,
            options,
        )?),
    );
    body.insert("stream".to_string(), Value::Bool(options.stream));

    if options.stream && dialect.supports_stream_usage() {
        body.insert(
            "stream_options".to_string(),
            json!({ "include_usage": true }),
        );
    }
    if let Some(value) = request.generation.max_output_tokens {
        body.insert(
            dialect.max_output_tokens_field().as_str().to_string(),
            Value::from(value),
        );
    }
    if let Some(value) = request.generation.temperature {
        body.insert("temperature".to_string(), Value::from(value));
    }
    if let Some(value) = request.generation.top_p {
        body.insert("top_p".to_string(), Value::from(value));
    }
    if !request.generation.stop_sequences.is_empty() {
        body.insert(
            "stop".to_string(),
            serde_json::to_value(&request.generation.stop_sequences).map_err(json_encode_error)?,
        );
    }
    if let Some(value) = request.generation.seed {
        body.insert("seed".to_string(), Value::from(value));
    }

    if !request.tools.is_empty() {
        let tools = request
            .tools
            .iter()
            .map(|tool| {
                let mut function = Map::new();
                function.insert("name".to_string(), Value::String(tool.name().to_string()));
                if let Some(description) = tool.description() {
                    function.insert(
                        "description".to_string(),
                        Value::String(description.to_string()),
                    );
                }
                function.insert("parameters".to_string(), tool.input_schema().clone());
                if let Some(strict) = dialect.function_tool_strict() {
                    function.insert("strict".to_string(), Value::Bool(strict));
                }
                json!({
                    "type": "function",
                    "function": function,
                })
            })
            .collect();
        body.insert("tools".to_string(), Value::Array(tools));
    }
    if let Some(choice) = &request.tool_choice {
        body.insert("tool_choice".to_string(), encode_tool_choice(choice)?);
    }
    if let Some(output) = &request.structured_output {
        body.insert(
            "response_format".to_string(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": output.name,
                    "description": output.description,
                    "schema": output.schema,
                    "strict": output.strict,
                }
            }),
        );
    }

    for (name, value) in &options.extra {
        if is_protected_option_field(name) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "provider options attempted to override a canonical Chat Completions field",
            ));
        }
        body.insert(name.clone(), value.clone());
    }
    Ok(Value::Object(body))
}

fn encode_messages(
    scope: &ProviderScope,
    model: &ModelId,
    messages: &[Message],
    dialect: &ChatCompletionsDialect,
    options: &ChatRequestEncodingOptions,
) -> Result<Vec<Value>, Error> {
    let mut encoded = Vec::with_capacity(messages.len());
    let mut seen_breakpoints = BTreeSet::new();
    for (message_index, message) in messages.iter().enumerate() {
        if message.role() == MessageRole::Tool {
            encode_tool_results(message, &mut encoded)?;
        } else {
            encoded.push(encode_message(
                scope,
                model,
                message,
                message_index,
                dialect,
                options,
                &mut seen_breakpoints,
            )?);
        }
    }
    if seen_breakpoints != options.prompt_cache_breakpoints {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "an OpenAI Chat Completions prompt-cache breakpoint did not identify an encodable content block",
        ));
    }
    Ok(encoded)
}

fn encode_message(
    scope: &ProviderScope,
    model: &ModelId,
    message: &Message,
    message_index: usize,
    dialect: &ChatCompletionsDialect,
    options: &ChatRequestEncodingOptions,
    seen_breakpoints: &mut BTreeSet<ChatPromptCacheBlock>,
) -> Result<Value, Error> {
    let message_role = message.role();
    let role = match message_role {
        MessageRole::System => "system",
        MessageRole::Developer if dialect.supports_developer_role() => "developer",
        MessageRole::Developer => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "selected Chat Completions dialect does not support developer messages",
            ));
        }
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => unreachable!("tool messages are expanded separately"),
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "selected Chat Completions dialect does not support this message role",
            ));
        }
    };

    let mut text = Vec::new();
    let mut content_blocks = Vec::new();
    let mut has_media = false;
    let mut reasoning = Vec::new();
    let mut reasoning_details = None;
    let mut tool_calls = Vec::new();
    for (content_index, part) in message.content().iter().enumerate() {
        match part.content() {
            ContentPart::Text { text: value } => {
                text.push(value.as_str());
                content_blocks.push((content_index, json!({ "type": "text", "text": value })));
            }
            ContentPart::Media(value)
                if message_role == MessageRole::User
                    && (value.media_type.starts_with("image/")
                        || (dialect.supports_video_input()
                            && value.media_type.starts_with("video/"))) =>
            {
                has_media = true;
                content_blocks.push((content_index, encode_media(value)?));
            }
            ContentPart::Reasoning { text: value }
                if message_role == MessageRole::Assistant
                    && dialect.reasoning_input_field().is_some() =>
            {
                reasoning.push(value.as_str());
            }
            ContentPart::ProviderOpaque(item) if message_role == MessageRole::Assistant => {
                if dialect.reasoning_details_field().is_none() {
                    return Err(Error::new(
                        ErrorKind::Unsupported,
                        "provider-native Chat Completions history is not replayable in this dialect",
                    ));
                }
                let value = replay_reasoning_details(scope, model, item)?;
                if reasoning_details.replace(value).is_some() {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "assistant history contained duplicate Chat Completions reasoning_details",
                    ));
                }
            }
            ContentPart::ToolCall(call) if message_role == MessageRole::Assistant => {
                tool_calls.push(json!({
                    "id": call.id(),
                    "type": "function",
                    "function": {
                        "name": call.name(),
                        "arguments": serde_json::to_string(call.arguments())
                            .map_err(json_encode_error)?,
                    }
                }));
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "message content cannot be projected to the selected Chat Completions dialect",
                ));
            }
        }
    }

    let mut object = Map::new();
    object.insert("role".to_string(), Value::String(role.to_string()));
    let has_breakpoints = options
        .prompt_cache_breakpoints
        .iter()
        .any(|block| block.message_index == message_index);
    if !has_media && !has_breakpoints {
        object.insert("content".to_string(), Value::String(text.concat()));
    } else {
        let mut content = Vec::with_capacity(content_blocks.len());
        for (content_index, mut block) in content_blocks {
            let coordinate = ChatPromptCacheBlock::new(message_index, content_index);
            if options.prompt_cache_breakpoints.contains(&coordinate) {
                let Some(object) = block.as_object_mut() else {
                    return Err(Error::new(
                        ErrorKind::Internal,
                        "Chat Completions encoder produced a non-object content block",
                    ));
                };
                object.insert(
                    "prompt_cache_breakpoint".to_string(),
                    json!({ "mode": "explicit" }),
                );
                seen_breakpoints.insert(coordinate);
            }
            content.push(block);
        }
        object.insert("content".to_string(), Value::Array(content));
    }
    if let Some(field) = dialect.reasoning_input_field()
        && !reasoning.is_empty()
    {
        object.insert(field.to_string(), Value::String(reasoning.concat()));
    }
    if let Some(value) = reasoning_details {
        let field = dialect
            .reasoning_details_field()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "reasoning replay field was missing"))?;
        object.insert(field.to_string(), value);
    }
    if !tool_calls.is_empty() {
        object.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    Ok(Value::Object(object))
}

fn encode_media(media: &siumai_core::MediaPart) -> Result<Value, Error> {
    let url = match &media.data {
        MediaData::Url(url) => url.clone(),
        MediaData::Bytes(bytes) => format!(
            "data:{};base64,{}",
            media.media_type,
            base64::engine::general_purpose::STANDARD.encode(bytes)
        ),
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "selected Chat Completions dialect does not support this media source",
            ));
        }
    };
    let kind = if media.media_type.starts_with("video/") {
        "video_url"
    } else {
        "image_url"
    };
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String(kind.to_string()));
    object.insert(kind.to_string(), json!({ "url": url }));
    Ok(Value::Object(object))
}

fn encode_tool_results(message: &Message, output: &mut Vec<Value>) -> Result<(), Error> {
    if message.content().is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "tool message requires at least one tool result",
        ));
    }
    for part in message.content() {
        let ContentPart::ToolResult(result) = part.content() else {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "tool messages may contain only tool results",
            ));
        };
        let content = match &result.outcome {
            ToolOutcome::Success { value } => serde_json::to_string(value),
            outcome => serde_json::to_string(outcome),
        }
        .map_err(json_encode_error)?;
        output.push(json!({
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.name,
            "content": content,
        }));
    }
    Ok(())
}

fn encode_tool_choice(choice: &ToolChoice) -> Result<Value, Error> {
    Ok(match choice {
        ToolChoice::Auto => Value::String("auto".to_string()),
        ToolChoice::None => Value::String("none".to_string()),
        ToolChoice::Required => Value::String("required".to_string()),
        ToolChoice::Named { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "selected Chat Completions dialect does not support this tool choice",
            ));
        }
    })
}

fn json_encode_error(source: serde_json::Error) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "language request contains a value that cannot be encoded as JSON",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use siumai_core::{
        ApiModeId, ContentPart, GenerationConfig, LanguageRequest, MediaData, MediaPart, Message,
        MessageRole, ModelId, ProtocolId, ProviderId, ProviderScope, ReplayDomain, ReplayDomainId,
    };

    use super::*;

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new("test-provider").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("chat-request-test").unwrap(),
            ))
    }

    #[test]
    fn encodes_omitted_controls_tools_images_and_structured_output() {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            [
                ContentPart::Text {
                    text: "inspect".to_string(),
                },
                ContentPart::Media(MediaPart {
                    media_type: "image/png".to_string(),
                    data: MediaData::Bytes(vec![1, 2, 3].into()),
                    name: None,
                }),
            ],
        )])
        .with_generation(GenerationConfig {
            max_output_tokens: Some(100),
            ..GenerationConfig::default()
        });
        let body = encode_request(
            &scope(),
            &ModelId::new("future:model").unwrap(),
            &request,
            true,
            &ChatCompletionsDialect::generic(),
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(body["model"], "future:model");
        assert_eq!(body["max_tokens"], 100);
        assert!(body.get("temperature").is_none());
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert!(
            body["messages"][0]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn video_input_requires_an_explicit_verified_dialect() {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            [ContentPart::Media(MediaPart {
                media_type: "video/mp4".to_string(),
                data: MediaData::Bytes(vec![1, 2, 3].into()),
                name: None,
            })],
        )]);
        assert!(
            encode_request(
                &scope(),
                &ModelId::new("model").unwrap(),
                &request,
                false,
                &ChatCompletionsDialect::generic(),
                &BTreeMap::new(),
            )
            .is_err()
        );

        let body = encode_request(
            &scope(),
            &ModelId::new("model").unwrap(),
            &request,
            false,
            &ChatCompletionsDialect::generic().with_video_input(true),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(
            body["messages"][0]["content"][0]["video_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:video/mp4;base64,")
        );
    }

    #[test]
    fn rejects_lossy_history_projection_and_protected_overrides() {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            [ContentPart::Reasoning {
                text: "private state".to_string(),
            }],
        )]);
        assert!(
            encode_request(
                &scope(),
                &ModelId::new("model").unwrap(),
                &request,
                false,
                &ChatCompletionsDialect::generic(),
                &BTreeMap::new(),
            )
            .is_err()
        );

        let mut extra = BTreeMap::new();
        extra.insert("model".to_string(), json!("rewritten"));
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
        assert!(
            encode_request(
                &scope(),
                &ModelId::new("model").unwrap(),
                &request,
                false,
                &ChatCompletionsDialect::generic(),
                &extra,
            )
            .is_err()
        );
    }

    #[test]
    fn explicit_prompt_cache_breakpoints_preserve_content_coordinates() {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            [
                ContentPart::Text {
                    text: "stable prefix".to_string(),
                },
                ContentPart::Text {
                    text: "dynamic suffix".to_string(),
                },
            ],
        )]);
        let options = ChatRequestEncodingOptions::new(false)
            .with_extra(BTreeMap::from([(
                "prompt_cache_options".to_string(),
                json!({ "mode": "explicit", "ttl": "30m" }),
            )]))
            .with_prompt_cache_breakpoint(ChatPromptCacheBlock::new(0, 0));
        let body = encode_request_with_options(
            &scope(),
            &ModelId::new("gpt-5.6").unwrap(),
            &request,
            &ChatCompletionsDialect::generic(),
            &options,
        )
        .unwrap();

        assert_eq!(
            body["messages"][0]["content"][0]["prompt_cache_breakpoint"],
            json!({ "mode": "explicit" })
        );
        assert_eq!(body["messages"][0]["content"][1]["text"], "dynamic suffix");
        assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    }

    #[test]
    fn rejects_unencodable_or_excessive_prompt_cache_breakpoints() {
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
        let missing = ChatRequestEncodingOptions::new(false)
            .with_prompt_cache_breakpoint(ChatPromptCacheBlock::new(0, 1));
        assert!(
            encode_request_with_options(
                &scope(),
                &ModelId::new("gpt-5.6").unwrap(),
                &request,
                &ChatCompletionsDialect::generic(),
                &missing,
            )
            .is_err()
        );

        let mut excessive = ChatRequestEncodingOptions::new(false);
        for content_index in 0..51 {
            excessive =
                excessive.with_prompt_cache_breakpoint(ChatPromptCacheBlock::new(0, content_index));
        }
        assert!(
            encode_request_with_options(
                &scope(),
                &ModelId::new("gpt-5.6").unwrap(),
                &request,
                &ChatCompletionsDialect::generic(),
                &excessive,
            )
            .is_err()
        );
    }
}
