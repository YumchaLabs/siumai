use std::collections::BTreeMap;

use serde_json::{Map, Value};
use siumai_core::{
    ContentPart, DEFAULT_TOOL_INPUT_BYTE_LIMIT, Error, ErrorKind, FinishReason, LanguageResponse,
    LanguageResponseStatus, ModelId, OpaqueProviderItem, ProviderProvenance, ProviderScope,
    ToolCall, Usage, UsageValue,
};

use super::ChatCompletionsDialect;
use super::reasoning::preserve_reasoning_details;
use super::wire::{AssistantMessageWire, ChatResponseWire, ToolCallWire, UsageWire};

pub fn decode_response(
    scope: &ProviderScope,
    requested_model: &ModelId,
    body: &[u8],
    dialect: &ChatCompletionsDialect,
) -> Result<LanguageResponse, Error> {
    dialect
        .validate_reasoning_configuration()
        .map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "Chat Completions reasoning dialect configuration is invalid",
            )
            .with_source(source)
        })?;
    let wire = serde_json::from_slice::<ChatResponseWire>(body).map_err(json_decode_error)?;
    if wire.choices.len() != 1 || wire.choices[0].index != 0 {
        return Err(protocol_error(
            "Chat Completions response must contain exactly choice index zero",
        ));
    }
    let choice = wire.choices.into_iter().next().expect("length was checked");
    let model = parse_model(wire.model.as_deref(), requested_model)?;
    let content = decode_message(scope, &model, choice.message, dialect)?;
    let finish_reason = choice
        .finish_reason
        .as_deref()
        .map(decode_finish_reason)
        .ok_or_else(|| protocol_error("Chat Completions response omitted its finish reason"))?;

    build_response(
        wire.id,
        model,
        content,
        finish_reason,
        wire.usage
            .map(|usage| decode_usage(usage, dialect))
            .unwrap_or_default(),
        selected_response_metadata(&wire.extra),
    )
}

pub(crate) fn build_response(
    id: Option<String>,
    model: ModelId,
    content: Vec<ContentPart>,
    finish_reason: FinishReason,
    usage: Usage,
    provider: BTreeMap<String, Value>,
) -> Result<LanguageResponse, Error> {
    let status = match &finish_reason {
        FinishReason::Length => LanguageResponseStatus::Incomplete {
            reason: Some(siumai_core::LanguageIncompleteReason::MaxOutputTokens),
        },
        FinishReason::ContentFilter => LanguageResponseStatus::Incomplete {
            reason: Some(siumai_core::LanguageIncompleteReason::ContentFilter),
        },
        FinishReason::Error => LanguageResponseStatus::Failed,
        FinishReason::Cancelled => LanguageResponseStatus::Cancelled,
        _ => LanguageResponseStatus::Completed,
    };
    let mut response = LanguageResponse::new(status, content, finish_reason, usage)
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "Chat Completions response produced an inconsistent terminal state",
            )
            .with_source(source)
        })?
        .with_model(model)
        .with_provider_metadata(provider);
    if let Some(id) = id {
        response = response.with_id(id);
    }
    Ok(response)
}

pub(crate) fn decode_message(
    scope: &ProviderScope,
    model: &ModelId,
    message: AssistantMessageWire,
    dialect: &ChatCompletionsDialect,
) -> Result<Vec<ContentPart>, Error> {
    let mut content = Vec::new();
    if let Some(value) = message.content {
        decode_content(scope, model, value, &mut content)?;
    }
    if let Some(field) = dialect.reasoning_output_field()
        && let Some(value) = message.extra.get(field)
        && !value.is_null()
    {
        let text = value
            .as_str()
            .ok_or_else(|| protocol_error("Chat Completions reasoning field was not a string"))?;
        content.push(ContentPart::Reasoning {
            text: text.to_string(),
        });
    }
    if let Some(field) = dialect.reasoning_details_field()
        && let Some(value) = message.extra.get(field)
        && !value.is_null()
    {
        content.push(ContentPart::ProviderOpaque(preserve_reasoning_details(
            scope, model, value,
        )?));
    }
    if let Some(reason) = message.refusal {
        content.push(ContentPart::Refusal {
            reason: Some(reason),
        });
    }
    for call in message.tool_calls {
        content.push(ContentPart::ToolCall(decode_tool_call(call)?));
    }
    Ok(content)
}

fn decode_content(
    scope: &ProviderScope,
    model: &ModelId,
    value: Value,
    output: &mut Vec<ContentPart>,
) -> Result<(), Error> {
    match value {
        Value::Null => Ok(()),
        Value::String(text) => {
            output.push(ContentPart::Text { text });
            Ok(())
        }
        Value::Array(parts) => {
            for part in parts {
                let object = part.as_object().ok_or_else(|| {
                    protocol_error("Chat Completions content part was not an object")
                })?;
                match object.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = required_string(object, "text")?;
                        output.push(ContentPart::Text {
                            text: text.to_string(),
                        });
                    }
                    Some("refusal") => {
                        let reason = required_string(object, "refusal")?;
                        output.push(ContentPart::Refusal {
                            reason: Some(reason.to_string()),
                        });
                    }
                    Some(kind) => output.push(ContentPart::ProviderOpaque(opaque_content(
                        scope,
                        model,
                        kind,
                        part.clone(),
                    )?)),
                    None => {
                        return Err(protocol_error(
                            "Chat Completions content part omitted its type",
                        ));
                    }
                }
            }
            Ok(())
        }
        _ => Err(protocol_error(
            "Chat Completions message content had an unsupported JSON shape",
        )),
    }
}

fn opaque_content(
    scope: &ProviderScope,
    model: &ModelId,
    kind: &str,
    value: Value,
) -> Result<OpaqueProviderItem, Error> {
    let provenance = ProviderProvenance::from_scope(scope, model.clone()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Chat Completions replay requires an explicit provider replay domain",
        )
        .with_source(source)
    })?;
    OpaqueProviderItem::new(provenance, format!("chat.content.{kind}"), value).map_err(|source| {
        Error::new(
            ErrorKind::ResponseLimit,
            "provider-native Chat Completions content exceeded its preservation limit",
        )
        .with_source(source)
    })
}

pub(crate) fn decode_tool_call(call: ToolCallWire) -> Result<ToolCall, Error> {
    if call.kind != "function" || call.id.is_empty() || call.function.name.is_empty() {
        return Err(protocol_error(
            "Chat Completions tool call identity or type was invalid",
        ));
    }
    if call.function.arguments.len() > DEFAULT_TOOL_INPUT_BYTE_LIMIT {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Chat Completions tool arguments exceeded the byte limit",
        ));
    }
    let arguments = serde_json::from_str(&call.function.arguments).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Chat Completions tool arguments were not complete JSON",
        )
        .with_source(source)
    })?;
    ToolCall::local(call.id, call.function.name, arguments).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Chat Completions tool call violated the canonical tool contract",
        )
        .with_source(source)
    })
}

pub(crate) fn decode_finish_reason(value: &str) -> FinishReason {
    match value {
        "stop" => FinishReason::Stop,
        "length" | "max_tokens" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        "refusal" => FinishReason::Refusal,
        other => FinishReason::Other(other.to_string()),
    }
}

pub(crate) fn decode_usage(wire: UsageWire, dialect: &ChatCompletionsDialect) -> Usage {
    let mut provider = BTreeMap::new();
    for key in [
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
        "cached_tokens",
    ] {
        if let Some(value) = wire.extra.get(key).filter(|value| value.is_number()) {
            provider.insert(key.to_string(), value.clone());
        }
    }
    let mut usage = Usage::default();
    usage.input_tokens = usage_value(wire.prompt_tokens);
    usage.output_tokens = usage_value(wire.completion_tokens);
    usage.total_tokens = usage_value(wire.total_tokens);
    usage.reasoning_tokens = usage_value(
        wire.completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
    );
    let standard_cache_read = wire
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens);
    let dialect_cache_read = dialect
        .cache_read_tokens_field()
        .and_then(|field| wire.extra.get(field))
        .and_then(Value::as_u64);
    usage.cache_read_tokens = usage_value(standard_cache_read.or(dialect_cache_read));
    usage.cache_write_tokens = usage_value(
        dialect
            .cache_write_tokens_field()
            .and_then(|field| wire.extra.get(field))
            .and_then(Value::as_u64),
    );
    usage.audio_output_tokens = usage_value(
        wire.completion_tokens_details
            .and_then(|details| details.audio_tokens),
    );
    usage.provider = provider;
    usage
}

fn usage_value(value: Option<u64>) -> UsageValue {
    value.map_or(UsageValue::Unknown, UsageValue::Known)
}

pub(crate) fn parse_model(
    response_model: Option<&str>,
    requested_model: &ModelId,
) -> Result<ModelId, Error> {
    match response_model.filter(|value| !value.is_empty()) {
        Some(value) => ModelId::new(value).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "Chat Completions response contained an invalid model ID",
            )
            .with_source(source)
        }),
        None => Ok(requested_model.clone()),
    }
}

fn selected_response_metadata(extra: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut selected = Map::new();
    for key in ["service_tier", "system_fingerprint"] {
        if let Some(value) = extra.get(key) {
            selected.insert(key.to_string(), value.clone());
        }
    }
    if selected.is_empty() {
        BTreeMap::new()
    } else {
        BTreeMap::from([("openai".to_string(), Value::Object(selected))])
    }
}

fn required_string<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a str, Error> {
    object
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| protocol_error("Chat Completions content part omitted required text"))
}

fn json_decode_error(source: serde_json::Error) -> Error {
    Error::new(
        ErrorKind::Protocol,
        "provider returned malformed Chat Completions JSON",
    )
    .with_source(source)
}

pub(crate) fn protocol_error(message: &'static str) -> Error {
    Error::new(ErrorKind::Protocol, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{
        ApiModeId, LanguageRequest, Message, MessageRole, OpaqueProviderItem, PlatformId,
        ProtocolId, ProviderId, ProviderProvenance, ProviderScope, ReplayDomain, ReplayDomainId,
    };

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new("deepseek").unwrap())
            .with_platform(PlatformId::new("public-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("chat-response-test").unwrap(),
            ))
    }

    fn minimax_scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new("minimax").unwrap())
            .with_platform(PlatformId::new("minimax-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("minimax-test").unwrap(),
            ))
    }

    fn minimax_dialect() -> ChatCompletionsDialect {
        let reasoning = super::super::WireFieldName::new("reasoning_content").unwrap();
        ChatCompletionsDialect::generic()
            .with_reasoning_input_field(reasoning.clone())
            .with_reasoning_output_field(reasoning)
            .with_replayable_reasoning_details_field(
                super::super::WireFieldName::new("reasoning_details").unwrap(),
            )
            .unwrap()
    }

    fn reasoning_details() -> Value {
        serde_json::json!([{
            "type": "reasoning.text",
            "id": "reasoning-text-1",
            "format": "MiniMax-response-v1",
            "index": 0,
            "text": "inspect the weather tool"
        }])
    }

    #[test]
    fn decodes_text_reasoning_tools_usage_and_unknown_parts_losslessly() {
        let body = serde_json::to_vec(&serde_json::json!({
            "id": "chat-1",
            "model": "future:model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "vendor_state", "token": "opaque"}
                    ],
                    "reasoning_content": "thought",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"q\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 3,
                "total_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        }))
        .unwrap();
        let dialect = ChatCompletionsDialect::generic().with_reasoning_output_field(
            super::super::WireFieldName::new("reasoning_content").unwrap(),
        );
        let response = decode_response(
            &scope(),
            &ModelId::new("fallback").unwrap(),
            &body,
            &dialect,
        )
        .unwrap();

        assert_eq!(response.model().unwrap().as_str(), "future:model");
        assert_eq!(response.usage().input_tokens, UsageValue::Known(0));
        assert_eq!(response.usage().reasoning_tokens, UsageValue::Known(2));
        assert!(
            response
                .content()
                .iter()
                .any(|part| matches!(part, ContentPart::ProviderOpaque(_)))
        );
        assert!(
            response
                .content()
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall(call) if call.name() == "lookup"))
        );
    }

    #[test]
    fn rejects_multiple_choices_and_partial_tool_json() {
        let multiple = br#"{"choices": [{"index":0,"message":{},"finish_reason":"stop"},{"index":1,"message":{},"finish_reason":"stop"}]}"#;
        assert!(
            decode_response(
                &scope(),
                &ModelId::new("model").unwrap(),
                multiple,
                &ChatCompletionsDialect::generic(),
            )
            .is_err()
        );

        let partial = br#"{"choices":[{"index":0,"message":{"tool_calls":[{"id":"x","type":"function","function":{"name":"f","arguments":"{"}}]},"finish_reason":"tool_calls"}]}"#;
        assert!(
            decode_response(
                &scope(),
                &ModelId::new("model").unwrap(),
                partial,
                &ChatCompletionsDialect::generic(),
            )
            .is_err()
        );

        let oversized = serde_json::to_vec(&serde_json::json!({
            "choices": [{
                "index": 0,
                "message": {
                    "tool_calls": [{
                        "id": "call-oversized",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": format!(
                                "{}{{}}",
                                " ".repeat(DEFAULT_TOOL_INPUT_BYTE_LIMIT)
                            )
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();
        let error = decode_response(
            &scope(),
            &ModelId::new("model").unwrap(),
            &oversized,
            &ChatCompletionsDialect::generic(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
    }

    #[test]
    fn preserves_and_replays_structured_reasoning_only_for_exact_provenance() {
        let scope = minimax_scope();
        let model = ModelId::new("MiniMax-M3").unwrap();
        let details = reasoning_details();
        let body = serde_json::to_vec(&serde_json::json!({
            "id": "chat-minimax-1",
            "model": model.as_str(),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "calling the tool",
                    "reasoning_content": "inspect the weather tool",
                    "reasoning_details": details,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{\"city\":\"SF\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();
        let dialect = minimax_dialect();
        let response = decode_response(&scope, &model, &body, &dialect).unwrap();
        let native = response
            .content()
            .iter()
            .find_map(|part| match part {
                ContentPart::ProviderOpaque(item) => Some(item.clone()),
                _ => None,
            })
            .expect("reasoning details");
        assert_eq!(native.kind(), super::super::REASONING_DETAILS_OPAQUE_KIND);
        assert_eq!(native.provenance().provider().as_str(), "minimax");
        assert_eq!(
            native.provenance().platform().map(PlatformId::as_str),
            Some("minimax-api")
        );
        assert_eq!(native.provenance().protocol().as_str(), "openai");
        assert_eq!(native.provenance().model().as_str(), "MiniMax-M3");
        assert_eq!(native.data(), &reasoning_details());

        let history = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            response.content().iter().cloned(),
        )]);
        let encoded = super::super::encode_request(
            &scope,
            &model,
            &history,
            false,
            &dialect,
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(
            encoded["messages"][0]["reasoning_content"],
            "inspect the weather tool"
        );
        assert_eq!(
            encoded["messages"][0]["reasoning_details"],
            reasoning_details()
        );

        let foreign_scope = ProviderScope::new(ProviderId::new("foreign").unwrap())
            .with_platform(PlatformId::new("minimax-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("minimax-test").unwrap(),
            ));
        let foreign = OpaqueProviderItem::new(
            ProviderProvenance::from_scope(&foreign_scope, model.clone()).unwrap(),
            super::super::REASONING_DETAILS_OPAQUE_KIND,
            reasoning_details(),
        )
        .unwrap();
        let foreign_history = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            [ContentPart::ProviderOpaque(foreign)],
        )]);
        let foreign_error = super::super::encode_request(
            &scope,
            &model,
            &foreign_history,
            false,
            &dialect,
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert_eq!(foreign_error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn root_cache_usage_requires_explicit_dialect_mappings() {
        let body = br#"{
            "choices":[{"index":0,"message":{"content":"ok"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cached_tokens":7,"created_cache_tokens":3}
        }"#;
        let model = ModelId::new("model").unwrap();

        let generic =
            decode_response(&scope(), &model, body, &ChatCompletionsDialect::generic()).unwrap();
        assert_eq!(generic.usage().cache_read_tokens, UsageValue::Unknown);
        assert_eq!(generic.usage().cache_write_tokens, UsageValue::Unknown);

        let mapped = decode_response(
            &scope(),
            &model,
            body,
            &ChatCompletionsDialect::generic()
                .with_cache_read_tokens_field(
                    super::super::WireFieldName::new("cached_tokens").unwrap(),
                )
                .with_cache_write_tokens_field(
                    super::super::WireFieldName::new("created_cache_tokens").unwrap(),
                ),
        )
        .unwrap();
        assert_eq!(mapped.usage().cache_read_tokens, UsageValue::Known(7));
        assert_eq!(mapped.usage().cache_write_tokens, UsageValue::Known(3));
    }
}
