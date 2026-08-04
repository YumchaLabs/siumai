use std::collections::BTreeMap;

use base64::Engine;
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, Error, ErrorKind, LanguageRequest, MediaData, Message, MessageRole, ModelId,
    ToolChoice, ToolOutcome,
};

use super::ChatCompletionsDialect;

pub const CHAT_COMPLETIONS_TARGET: &str = "chat/completions";

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

pub fn encode_request(
    model: &ModelId,
    request: &LanguageRequest,
    stream: bool,
    dialect: &ChatCompletionsDialect,
    extra: &BTreeMap<String, Value>,
) -> Result<Value, Error> {
    request.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "language request is invalid").with_source(source)
    })?;

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));
    body.insert(
        "messages".to_string(),
        Value::Array(encode_messages(&request.messages, dialect)?),
    );
    body.insert("stream".to_string(), Value::Bool(stream));

    if stream && dialect.supports_stream_usage() {
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
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.input_schema(),
                    }
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

    for (name, value) in extra {
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
    messages: &[Message],
    dialect: &ChatCompletionsDialect,
) -> Result<Vec<Value>, Error> {
    let mut encoded = Vec::with_capacity(messages.len());
    for message in messages {
        if message.role == MessageRole::Tool {
            encode_tool_results(message, &mut encoded)?;
        } else {
            encoded.push(encode_message(message, dialect)?);
        }
    }
    Ok(encoded)
}

fn encode_message(message: &Message, dialect: &ChatCompletionsDialect) -> Result<Value, Error> {
    let role = match message.role {
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
    let mut media = Vec::new();
    let mut reasoning = Vec::new();
    let mut tool_calls = Vec::new();
    for part in &message.content {
        match part {
            ContentPart::Text { text: value } => text.push(value.as_str()),
            ContentPart::Media(value)
                if message.role == MessageRole::User && value.media_type.starts_with("image/") =>
            {
                media.push(encode_image(value)?);
            }
            ContentPart::Reasoning { text: value }
                if message.role == MessageRole::Assistant
                    && dialect.reasoning_input_field().is_some() =>
            {
                reasoning.push(value.as_str());
            }
            ContentPart::ToolCall(call) if message.role == MessageRole::Assistant => {
                tool_calls.push(json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": serde_json::to_string(&call.arguments)
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
    if media.is_empty() {
        object.insert("content".to_string(), Value::String(text.concat()));
    } else {
        let mut content = text
            .into_iter()
            .map(|text| json!({ "type": "text", "text": text }))
            .collect::<Vec<_>>();
        content.extend(media);
        object.insert("content".to_string(), Value::Array(content));
    }
    if let Some(field) = dialect.reasoning_input_field()
        && !reasoning.is_empty()
    {
        object.insert(field.to_string(), Value::String(reasoning.concat()));
    }
    if !tool_calls.is_empty() {
        object.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    Ok(Value::Object(object))
}

fn encode_image(media: &siumai_core::MediaPart) -> Result<Value, Error> {
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
    Ok(json!({
        "type": "image_url",
        "image_url": { "url": url }
    }))
}

fn encode_tool_results(message: &Message, output: &mut Vec<Value>) -> Result<(), Error> {
    if message.content.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "tool message requires at least one tool result",
        ));
    }
    for part in &message.content {
        let ContentPart::ToolResult(result) = part else {
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
        ContentPart, GenerationConfig, LanguageRequest, MediaData, MediaPart, Message, MessageRole,
        ModelId,
    };

    use super::*;

    #[test]
    fn encodes_omitted_controls_tools_images_and_structured_output() {
        let request = LanguageRequest::new(vec![Message {
            role: MessageRole::User,
            content: vec![
                ContentPart::Text {
                    text: "inspect".to_string(),
                },
                ContentPart::Media(MediaPart {
                    media_type: "image/png".to_string(),
                    data: MediaData::Bytes(vec![1, 2, 3].into()),
                    name: None,
                }),
            ],
        }])
        .with_generation(GenerationConfig {
            max_output_tokens: Some(100),
            ..GenerationConfig::default()
        });
        let body = encode_request(
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
    fn rejects_lossy_history_projection_and_protected_overrides() {
        let request = LanguageRequest::new(vec![Message {
            role: MessageRole::Assistant,
            content: vec![ContentPart::Reasoning {
                text: "private state".to_string(),
            }],
        }]);
        assert!(
            encode_request(
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
                &ModelId::new("model").unwrap(),
                &request,
                false,
                &ChatCompletionsDialect::generic(),
                &extra,
            )
            .is_err()
        );
    }
}
