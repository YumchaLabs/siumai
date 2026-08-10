use std::collections::BTreeMap;

use base64::Engine as _;
use serde_json::{Map, Value};
use siumai_core::{
    Citation, ContentPart, LanguageCompletionReason, LanguageIncompleteReason, LanguageResponse,
    LanguageTermination, MediaData, MediaPart, ModelId, OpaqueProviderItem, ProviderProvenance,
    ProviderScope, ToolCall, Usage, UsageValue,
};

use super::wire::{MessageResponseWire, UsageWire};
use super::{MessagesCodecError, OPAQUE_CONTENT_BLOCK_KIND, PROTOCOL_ID};

/// Decode one terminal Anthropic Messages response into the canonical response.
pub fn decode_response(
    body: &[u8],
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<LanguageResponse, MessagesCodecError> {
    let wire = serde_json::from_slice::<MessageResponseWire>(body)
        .map_err(MessagesCodecError::JsonDecode)?;
    decode_response_wire(wire, scope, requested_model)
}

pub(crate) fn decode_response_wire(
    wire: MessageResponseWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<LanguageResponse, MessagesCodecError> {
    if wire.kind != "message" || wire.role != "assistant" {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "terminal response was not an assistant message",
        });
    }
    if wire.id.trim().is_empty() {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "terminal response omitted its message ID",
        });
    }
    let model = if wire.model.trim().is_empty() {
        requested_model.clone()
    } else {
        ModelId::new(wire.model.clone()).map_err(MessagesCodecError::InvalidModelId)?
    };
    let stop = map_stop_reason(wire.stop_reason.as_deref())?;
    let refusal_reason = decode_refusal_reason(wire.stop_details.as_ref())?;
    // Anthropic marks any partial output preceding `refusal` as invalid. Keep
    // the raw stop details as provider metadata, but expose only the canonical
    // refusal in the terminal response.
    let content = if matches!(
        &stop,
        LanguageTermination::Completed(LanguageCompletionReason::Refusal)
    ) {
        refusal_content(refusal_reason)
    } else {
        let mut content = Vec::new();
        for block in &wire.content {
            content.extend(decode_content_block(block, scope, &model)?);
        }
        content
    };
    let usage = decode_usage(&wire.usage);
    let provider = response_metadata(
        wire.extra,
        wire.stop_sequence,
        wire.stop_details,
        &wire.usage,
    );
    let response = LanguageResponse::new(stop, content, usage)
        .map_err(MessagesCodecError::InvalidCanonicalResponse)?
        .with_id(wire.id)
        .with_model(model)
        .with_provider_metadata(single_namespace_metadata(provider));
    Ok(response)
}

pub(crate) struct StreamResponseParts<'a> {
    pub(crate) id: String,
    pub(crate) model: ModelId,
    pub(crate) stop_reason: Option<&'a str>,
    pub(crate) stop_sequence: Option<String>,
    pub(crate) stop_details: Option<Value>,
    pub(crate) content: Vec<ContentPart>,
    pub(crate) usage_wire: &'a UsageWire,
    pub(crate) provider: BTreeMap<String, Value>,
}

pub(crate) fn build_stream_response(
    parts: StreamResponseParts<'_>,
) -> Result<LanguageResponse, MessagesCodecError> {
    let stop = map_stop_reason(parts.stop_reason)?;
    let refusal_reason = decode_refusal_reason(parts.stop_details.as_ref())?;
    // Stream deltas are provisional. The terminal response must not promote
    // content that Anthropic invalidated with a refusal stop reason.
    let content = if matches!(
        &stop,
        LanguageTermination::Completed(LanguageCompletionReason::Refusal)
    ) {
        refusal_content(refusal_reason)
    } else {
        parts.content
    };
    let usage = decode_usage(parts.usage_wire);
    let provider = response_metadata(
        parts.provider,
        parts.stop_sequence,
        parts.stop_details,
        parts.usage_wire,
    );
    LanguageResponse::new(stop, content, usage)
        .map_err(MessagesCodecError::InvalidCanonicalResponse)
        .map(|response| {
            response
                .with_id(parts.id)
                .with_model(parts.model)
                .with_provider_metadata(single_namespace_metadata(provider))
        })
}

pub(crate) fn decode_content_block(
    block: &Value,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<Vec<ContentPart>, MessagesCodecError> {
    let object = block
        .as_object()
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "response content block was not an object",
        })?;
    let kind = object.get("type").and_then(Value::as_str).ok_or(
        MessagesCodecError::ProtocolViolation {
            reason: "response content block omitted its type",
        },
    )?;
    match kind {
        "text" => decode_text_block(object),
        "thinking" => {
            let thinking = required_non_empty_string(object, "thinking")?;
            required_non_empty_string(object, "signature")?;
            Ok(vec![
                ContentPart::Reasoning {
                    text: thinking.to_string(),
                },
                ContentPart::ProviderOpaque(retain_native_block(block, scope, model)?),
            ])
        }
        "redacted_thinking" => {
            required_non_empty_string(object, "data")?;
            Ok(vec![ContentPart::ProviderOpaque(retain_native_block(
                block, scope, model,
            )?)])
        }
        "tool_use" => {
            let id = required_non_empty_string(object, "id")?.to_string();
            let name = required_non_empty_string(object, "name")?.to_string();
            let input =
                object
                    .get("input")
                    .cloned()
                    .ok_or(MessagesCodecError::ProtocolViolation {
                        reason: "tool_use block omitted its input",
                    })?;
            if !input.is_object() {
                return Err(MessagesCodecError::ProtocolViolation {
                    reason: "tool_use input must be a JSON object",
                });
            }
            Ok(vec![ContentPart::ToolCall(
                ToolCall::local(id, name, input).map_err(MessagesCodecError::InvalidToolCall)?,
            )])
        }
        "refusal" => Ok(vec![ContentPart::Refusal {
            reason: object
                .get("refusal")
                .or_else(|| object.get("reason"))
                .and_then(Value::as_str)
                .map(ToString::to_string),
        }]),
        "image" | "document" => Ok(vec![ContentPart::Media(decode_media_block(kind, object)?)]),
        _ => Ok(vec![ContentPart::ProviderOpaque(retain_native_block(
            block, scope, model,
        )?)]),
    }
}

fn decode_text_block(object: &Map<String, Value>) -> Result<Vec<ContentPart>, MessagesCodecError> {
    let text = required_string(object, "text")?;
    let mut parts = vec![ContentPart::Text {
        text: text.to_string(),
    }];
    if let Some(citations) = object.get("citations") {
        let citations = citations
            .as_array()
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "text citations were not an array",
            })?;
        for (index, citation) in citations.iter().enumerate() {
            parts.push(ContentPart::Citation(decode_citation(citation, index)?));
        }
    }
    Ok(parts)
}

fn decode_citation(value: &Value, index: usize) -> Result<Citation, MessagesCodecError> {
    let object = value
        .as_object()
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "citation entry was not an object",
        })?;
    let url = object
        .get("url")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let source_id = url.clone().or_else(|| {
        object
            .get("encrypted_index")
            .and_then(Value::as_str)
            .map(ToString::to_string)
    });
    let source_id = source_id.unwrap_or_else(|| {
        object
            .get("document_index")
            .and_then(Value::as_u64)
            .map_or_else(
                || format!("citation-{index}"),
                |value| format!("document-{value}"),
            )
    });
    let title = object
        .get("title")
        .or_else(|| object.get("document_title"))
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let start = object
        .get("start_char_index")
        .or_else(|| object.get("start_page_number"))
        .or_else(|| object.get("start_block_index"))
        .and_then(Value::as_u64);
    let end = object
        .get("end_char_index")
        .or_else(|| object.get("end_page_number"))
        .or_else(|| object.get("end_block_index"))
        .and_then(Value::as_u64);
    Ok(Citation {
        source_id,
        title,
        url,
        start,
        end,
        provider: single_namespace_metadata(object.clone().into_iter().collect()),
    })
}

fn decode_media_block(
    block_type: &str,
    object: &Map<String, Value>,
) -> Result<MediaPart, MessagesCodecError> {
    let source = object.get("source").and_then(Value::as_object).ok_or(
        MessagesCodecError::ProtocolViolation {
            reason: "media content block omitted its source",
        },
    )?;
    let source_type = required_non_empty_string(source, "type")?;
    let (media_type, data) = match source_type {
        "base64" => {
            let media_type = required_non_empty_string(source, "media_type")?.to_string();
            let encoded = required_non_empty_string(source, "data")?;
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|_| MessagesCodecError::ProtocolViolation {
                    reason: "media content block contained invalid base64",
                })?;
            (media_type, MediaData::Bytes(decoded.into()))
        }
        "url" => {
            let url = required_non_empty_string(source, "url")?.to_string();
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    if block_type == "document" {
                        "application/pdf".to_string()
                    } else {
                        "application/octet-stream".to_string()
                    }
                });
            (media_type, MediaData::Url(url))
        }
        _ => {
            return Err(MessagesCodecError::Unsupported {
                feature: "this response media source",
            });
        }
    };
    Ok(MediaPart {
        media_type,
        data,
        name: object
            .get("title")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    })
}

fn retain_native_block(
    block: &Value,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<OpaqueProviderItem, MessagesCodecError> {
    let provenance = ProviderProvenance::from_scope(scope, model.clone())
        .map_err(MessagesCodecError::InvalidProvenance)?;
    let mut builder =
        OpaqueProviderItem::builder(provenance, OPAQUE_CONTENT_BLOCK_KIND, block.clone());
    if let Some(id) = block.get("id").and_then(Value::as_str) {
        builder = builder.item_id(id);
    }
    builder
        .build()
        .map_err(MessagesCodecError::InvalidOpaqueItem)
}

pub(crate) fn map_stop_reason(
    stop_reason: Option<&str>,
) -> Result<LanguageTermination, MessagesCodecError> {
    let stop_reason = stop_reason.ok_or(MessagesCodecError::ProtocolViolation {
        reason: "terminal response omitted its stop reason",
    })?;
    let mapping = match stop_reason {
        "end_turn" | "stop_sequence" => {
            LanguageTermination::Completed(LanguageCompletionReason::Stop)
        }
        "tool_use" => LanguageTermination::Completed(LanguageCompletionReason::ToolCalls),
        "max_tokens" => LanguageTermination::Incomplete(LanguageIncompleteReason::MaxOutputTokens),
        "refusal" => LanguageTermination::Completed(LanguageCompletionReason::Refusal),
        "pause_turn" | "model_context_window_exceeded" => LanguageTermination::Incomplete(
            LanguageIncompleteReason::Other(stop_reason.to_string()),
        ),
        other => LanguageTermination::Completed(LanguageCompletionReason::Other(other.to_string())),
    };
    Ok(mapping)
}

pub(crate) fn decode_usage(wire: &UsageWire) -> Usage {
    let total = match (wire.input_tokens, wire.output_tokens) {
        (Some(input), Some(output)) => input
            .checked_add(output)
            .map_or(UsageValue::Unknown, UsageValue::Known),
        _ => UsageValue::Unknown,
    };
    let details = wire.provider_details();
    let mut usage = Usage::default()
        .with_input_tokens(wire.input_tokens)
        .with_output_tokens(wire.output_tokens)
        .with_total_tokens(total)
        .with_reasoning_tokens(
            wire.output_tokens_details
                .as_ref()
                .and_then(|details| details.thinking_tokens),
        )
        .with_cache_read_tokens(wire.cache_read_input_tokens)
        .with_cache_write_tokens(wire.cache_creation_input_tokens);
    if !details.is_empty() {
        usage = usage.with_provider_value(PROTOCOL_ID, Value::Object(details));
    }
    usage
}

pub(crate) fn decode_refusal_reason(
    stop_details: Option<&Value>,
) -> Result<Option<String>, MessagesCodecError> {
    let Some(stop_details) = stop_details else {
        return Ok(None);
    };
    let object = stop_details
        .as_object()
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "stop_details was not an object",
        })?;
    for field in ["type", "category", "explanation", "recommended_model"] {
        if object
            .get(field)
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "stop_details contained a non-string known field",
            });
        }
    }
    Ok(object
        .get("explanation")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string))
}

fn refusal_content(reason: Option<String>) -> Vec<ContentPart> {
    vec![ContentPart::Refusal { reason }]
}

fn response_metadata(
    mut values: BTreeMap<String, Value>,
    stop_sequence: Option<String>,
    stop_details: Option<Value>,
    usage: &UsageWire,
) -> BTreeMap<String, Value> {
    if let Some(stop_sequence) = stop_sequence {
        values.insert("stop_sequence".to_string(), Value::String(stop_sequence));
    }
    if let Some(stop_details) = stop_details {
        values.insert("stop_details".to_string(), stop_details);
    }
    let raw_usage = usage.provider_details();
    if !raw_usage.is_empty() {
        values.insert("usage".to_string(), Value::Object(raw_usage));
    }
    if let Some(iterations) = usage.iterations() {
        values.insert("iterations".to_string(), iterations.clone());
    }
    values
}

fn single_namespace_metadata(values: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    if values.is_empty() {
        BTreeMap::new()
    } else {
        BTreeMap::from([(
            PROTOCOL_ID.to_string(),
            Value::Object(values.into_iter().collect()),
        )])
    }
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, MessagesCodecError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "content block omitted a required string field",
        })
}

fn required_non_empty_string<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, MessagesCodecError> {
    required_string(object, field).and_then(|value| {
        if value.is_empty() {
            Err(MessagesCodecError::ProtocolViolation {
                reason: "content block contained an empty required string field",
            })
        } else {
            Ok(value)
        }
    })
}
