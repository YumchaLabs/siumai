//! Provider-agnostic structured output helpers.
//!
//! This module focuses on best-effort JSON extraction from model output. Provider-specific
//! schema enforcement lives at the protocol/provider layers (e.g. OpenAI `response_format`,
//! Anthropic `output_format`, Gemini `responseSchema`, or reserved tool strategies).

use crate::error::LlmError;
use crate::streaming::{ChatStream, StreamProcessor};
use crate::types::{ChatResponse, ChatStreamEvent, ContentPart, FinishReason, MessageContent};
use futures::StreamExt;

fn extract_first_markdown_fenced_block(text: &str) -> Option<&str> {
    let start = text.find("```")?;
    let after_start = start + 3;
    let rest = text.get(after_start..)?;

    // Skip optional language tag until newline (```json\n...).
    let newline = rest.find('\n')?;
    let content_start = after_start + newline + 1;

    let remaining = text.get(content_start..)?;
    let end_rel = remaining.find("```")?;
    let content_end = content_start + end_rel;

    text.get(content_start..content_end).map(str::trim)
}

fn extract_braced_slice(text: &str) -> Option<&str> {
    let obj_start = text.find('{');
    let obj_end = text.rfind('}');
    let arr_start = text.find('[');
    let arr_end = text.rfind(']');

    match (obj_start, obj_end, arr_start, arr_end) {
        (Some(os), Some(oe), Some(as_), Some(ae)) => {
            let obj_len = oe.saturating_sub(os);
            let arr_len = ae.saturating_sub(as_);
            if obj_len >= arr_len {
                text.get(os..=oe)
            } else {
                text.get(as_..=ae)
            }
        }
        (Some(os), Some(oe), _, _) => text.get(os..=oe),
        (_, _, Some(as_), Some(ae)) => text.get(as_..=ae),
        _ => None,
    }
}

fn parse_json_value_candidate(candidate: &str) -> Result<serde_json::Value, LlmError> {
    crate::streaming::parse_json_with_repair::<serde_json::Value>(candidate)
        .map_err(|e| LlmError::ParseError(format!("Failed to parse JSON response: {e}")))
}

fn parse_json_value_candidate_strict(candidate: &str) -> Result<serde_json::Value, LlmError> {
    serde_json::from_str::<serde_json::Value>(candidate)
        .map_err(|e| LlmError::ParseError(format!("Failed to parse JSON response: {e}")))
}

fn extract_reserved_json_tool_arguments(response: &ChatResponse) -> Option<serde_json::Value> {
    let MessageContent::MultiModal(parts) = &response.content else {
        return None;
    };

    for part in parts {
        let ContentPart::ToolCall {
            tool_name,
            arguments,
            ..
        } = part
        else {
            continue;
        };

        if tool_name == "json" {
            return Some(arguments.clone());
        }
    }

    None
}

fn normalize_tool_json_arguments(args: serde_json::Value) -> Result<serde_json::Value, LlmError> {
    match args {
        serde_json::Value::String(s) => extract_json_value(&s),
        other => Ok(other),
    }
}

fn normalize_tool_json_arguments_strict(
    args: serde_json::Value,
) -> Result<serde_json::Value, LlmError> {
    match args {
        serde_json::Value::String(s) => extract_json_value_strict(&s),
        other => Ok(other),
    }
}

fn looks_like_json_candidate(candidate: &str) -> bool {
    let trimmed = candidate.trim_start();
    matches!(
        trimmed.chars().next(),
        Some('{')
            | Some('[')
            | Some('"')
            | Some('-')
            | Some('0'..='9')
            | Some('t')
            | Some('f')
            | Some('n')
    )
}

/// Best-effort parse of a JSON value from a model output string.
///
/// This attempts candidates in order:
/// 1) Markdown fenced code block content (```json ... ```).
/// 2) The full trimmed string.
/// 3) A `{ ... }` / `[ ... ]` slice extracted from the string.
pub fn extract_json_value(text: &str) -> Result<serde_json::Value, LlmError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(LlmError::ParseError(
            "Failed to parse JSON response: empty output".to_string(),
        ));
    }

    if let Some(fenced) = extract_first_markdown_fenced_block(trimmed) {
        if let Ok(v) = parse_json_value_candidate_strict(fenced) {
            return Ok(v);
        }
        if looks_like_json_candidate(fenced)
            && let Ok(v) = parse_json_value_candidate(fenced)
        {
            return Ok(v);
        }
    }

    if let Ok(v) = parse_json_value_candidate_strict(trimmed) {
        return Ok(v);
    }
    if looks_like_json_candidate(trimmed)
        && let Ok(v) = parse_json_value_candidate(trimmed)
    {
        return Ok(v);
    }

    if let Some(slice) = extract_braced_slice(trimmed) {
        if let Ok(v) = parse_json_value_candidate_strict(slice) {
            return Ok(v);
        }
        if let Ok(v) = parse_json_value_candidate(slice) {
            return Ok(v);
        }
    }

    Err(LlmError::ParseError(
        "Failed to parse JSON response: no valid JSON candidate found".to_string(),
    ))
}

fn extract_json_value_strict(text: &str) -> Result<serde_json::Value, LlmError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(LlmError::ParseError(
            "Failed to parse JSON response: empty output".to_string(),
        ));
    }

    if let Some(fenced) = extract_first_markdown_fenced_block(trimmed)
        && let Ok(v) = parse_json_value_candidate_strict(fenced)
    {
        return Ok(v);
    }

    if let Ok(v) = parse_json_value_candidate_strict(trimmed) {
        return Ok(v);
    }

    if let Some(slice) = extract_braced_slice(trimmed)
        && let Ok(v) = parse_json_value_candidate_strict(slice)
    {
        return Ok(v);
    }

    Err(LlmError::ParseError(
        "Failed to parse JSON response: no valid complete JSON candidate found".to_string(),
    ))
}

fn content_filter_json_error() -> LlmError {
    LlmError::ParseError(
        "Failed to parse JSON response: response finished with content filtering/refusal before valid JSON was produced"
            .to_string(),
    )
}

fn incomplete_stream_json_error() -> LlmError {
    LlmError::ParseError(
        "Failed to parse JSON response: stream ended before a complete JSON value was produced"
            .to_string(),
    )
}

fn response_has_structured_output_candidate(response: &ChatResponse) -> bool {
    !response.text().unwrap_or_default().trim().is_empty()
        || extract_reserved_json_tool_arguments(response).is_some()
}

fn merge_stream_end_response_with_accumulated(
    processor: &StreamProcessor,
    response: ChatResponse,
) -> ChatResponse {
    let response_has_candidate = response_has_structured_output_candidate(&response);
    let ChatResponse {
        id,
        content,
        model,
        usage,
        finish_reason,
        audio,
        system_fingerprint,
        service_tier,
        warnings,
        provider_metadata,
    } = response;

    let accumulated = processor.build_final_response_with_finish_reason(finish_reason.clone());

    ChatResponse {
        id: id.or(accumulated.id),
        content: if response_has_candidate {
            content
        } else {
            accumulated.content
        },
        model: model.or(accumulated.model),
        usage: accumulated.usage.or(usage),
        finish_reason: finish_reason.or(accumulated.finish_reason),
        audio: audio.or(accumulated.audio),
        system_fingerprint: system_fingerprint.or(accumulated.system_fingerprint),
        service_tier: service_tier.or(accumulated.service_tier),
        warnings: warnings.or(accumulated.warnings),
        provider_metadata: accumulated.provider_metadata.or(provider_metadata),
    }
}

fn deserialize_structured_output<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
) -> Result<T, LlmError> {
    serde_json::from_value(value).map_err(|e| {
        LlmError::ParseError(format!(
            "Failed to deserialize structured output JSON into target type: {e}"
        ))
    })
}

/// Best-effort parse of a JSON value from a unified chat response.
pub fn extract_json_value_from_response(
    response: &ChatResponse,
) -> Result<serde_json::Value, LlmError> {
    let text = response.text().unwrap_or_default();
    let reserved_json_tool_arguments = extract_reserved_json_tool_arguments(response);

    if response.finish_reason == Some(FinishReason::ContentFilter) {
        if let Some(args) = reserved_json_tool_arguments.clone() {
            return normalize_tool_json_arguments_strict(args)
                .map_err(|_| content_filter_json_error());
        }

        if let Ok(v) = extract_json_value_strict(&text) {
            return Ok(v);
        }
        return Err(content_filter_json_error());
    }

    if let Ok(v) = extract_json_value(&text) {
        return Ok(v);
    }

    if let Some(args) = reserved_json_tool_arguments {
        return normalize_tool_json_arguments(args);
    }

    extract_json_value(&text)
}

fn extract_json_value_from_incomplete_stream_response(
    response: &ChatResponse,
) -> Result<serde_json::Value, LlmError> {
    let text = response.text().unwrap_or_default();
    if let Ok(v) = extract_json_value_strict(&text) {
        return Ok(v);
    }

    if let Some(args) = extract_reserved_json_tool_arguments(response) {
        return normalize_tool_json_arguments_strict(args)
            .map_err(|_| incomplete_stream_json_error());
    }

    Err(incomplete_stream_json_error())
}

/// Best-effort parse of a JSON value from a streaming response.
///
/// This consumes the stream and returns once either:
/// - A `StreamEnd { response }` event is observed, or
/// - The stream terminates (then we build a best-effort final response from the deltas).
pub async fn extract_json_value_from_stream(
    mut stream: ChatStream,
) -> Result<serde_json::Value, LlmError> {
    let mut processor = StreamProcessor::new();

    while let Some(item) = stream.next().await {
        let ev = item?;
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let _ = processor.process_event(ChatStreamEvent::StreamEnd {
                    response: response.clone(),
                });
                let final_response =
                    merge_stream_end_response_with_accumulated(&processor, response);

                if final_response.finish_reason == Some(FinishReason::Unknown) {
                    return extract_json_value_from_incomplete_stream_response(&final_response);
                }

                return extract_json_value_from_response(&final_response);
            }
            ChatStreamEvent::Error { error } => {
                return Err(LlmError::StreamError(error));
            }
            other => {
                let _ = processor.process_event(other);
            }
        }
    }

    let resp = processor.build_final_response();
    extract_json_value_from_incomplete_stream_response(&resp)
}

/// Best-effort parse + deserialize of a JSON value from a streaming response.
pub async fn extract_json_from_stream<T: serde::de::DeserializeOwned>(
    stream: ChatStream,
) -> Result<T, LlmError> {
    let v = extract_json_value_from_stream(stream).await?;
    deserialize_structured_output(v)
}

/// Best-effort parse + deserialize of a JSON value from a model output string.
pub fn extract_json<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, LlmError> {
    let v = extract_json_value(text)?;
    deserialize_structured_output(v)
}

/// Best-effort parse + deserialize of a JSON value from a unified chat response.
pub fn extract_json_from_response<T: serde::de::DeserializeOwned>(
    response: &ChatResponse,
) -> Result<T, LlmError> {
    let v = extract_json_value_from_response(response)?;
    deserialize_structured_output(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatResponse, MessageContent};
    use serde::Deserialize;

    #[test]
    fn extracts_json_from_plain_text() {
        let v = extract_json_value(r#"{"value":"ok"}"#).expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[test]
    fn extracts_json_from_markdown_fence() {
        let v = extract_json_value("```json\n{\"value\":\"ok\"}\n```").expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[test]
    fn extracts_json_from_wrapped_text() {
        let v = extract_json_value("Sure!\n{\"value\":\"ok\"}\nThanks.").expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[test]
    fn extracts_json_from_response_object() {
        let resp = ChatResponse::new(MessageContent::Text("{\"value\":\"ok\"}".to_string()));
        let v = extract_json_value_from_response(&resp).expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[test]
    fn extracts_json_from_reserved_tool_call_when_text_is_empty() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![ContentPart::tool_call(
            "call_1",
            "json",
            serde_json::json!({ "value": "ok" }),
            None,
        )]));
        let v = extract_json_value_from_response(&resp).expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[tokio::test]
    async fn extracts_json_from_stream_content_deltas() {
        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "{\"value\":\"ok\"}".to_string(),
                index: Some(0),
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse::new(MessageContent::Text("{\"value\":\"ok\"}".to_string())),
            }),
        ];
        let stream = Box::pin(futures::stream::iter(events));
        let v = extract_json_value_from_stream(stream).await.expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[tokio::test]
    async fn extracts_json_from_unknown_stream_end_using_accumulated_deltas() {
        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "{\"value\":\"ok\"}".to_string(),
                index: Some(0),
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse::empty_with_finish_reason(FinishReason::Unknown),
            }),
        ];
        let stream = Box::pin(futures::stream::iter(events));

        let v = extract_json_value_from_stream(stream).await.expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[tokio::test]
    async fn extracts_json_from_stream_tool_call_deltas_without_stream_end() {
        let events = vec![Ok(ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("json".to_string()),
            arguments_delta: Some("{\"value\":\"ok\"}".to_string()),
            index: Some(0),
        })];
        let stream = Box::pin(futures::stream::iter(events));
        let v = extract_json_value_from_stream(stream).await.expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[tokio::test]
    async fn rejects_truncated_json_tool_call_deltas_without_stream_end() {
        let events = vec![Ok(ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("json".to_string()),
            arguments_delta: Some("{\"value\":".to_string()),
            index: Some(0),
        })];
        let stream = Box::pin(futures::stream::iter(events));

        let err = extract_json_value_from_stream(stream)
            .await
            .expect_err("truncated tool stream should fail");
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn returns_content_filter_error_when_no_json_was_produced() {
        let mut resp = ChatResponse::new(MessageContent::Text(
            "I cannot comply with that request.".to_string(),
        ));
        resp.finish_reason = Some(FinishReason::ContentFilter);

        let err = extract_json_value_from_response(&resp).expect_err("content filter error");
        match err {
            LlmError::ParseError(message) => assert!(message.contains("content filtering/refusal")),
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn preserves_complete_json_when_content_filter_finish_reason_is_set() {
        let mut resp = ChatResponse::new(MessageContent::Text("{\"value\":\"ok\"}".to_string()));
        resp.finish_reason = Some(FinishReason::ContentFilter);

        let value = extract_json_value_from_response(&resp).expect("strict json should parse");
        assert_eq!(value["value"], "ok");
    }

    #[tokio::test]
    async fn rejects_truncated_json_when_stream_ends_without_stream_end_event() {
        let events = vec![Ok(ChatStreamEvent::ContentDelta {
            delta: "{\"value\":".to_string(),
            index: Some(0),
        })];
        let stream = Box::pin(futures::stream::iter(events));

        let err = extract_json_value_from_stream(stream)
            .await
            .expect_err("truncated stream should fail");
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_truncated_json_when_unknown_stream_end_uses_accumulated_deltas() {
        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "{\"value\":".to_string(),
                index: Some(0),
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse::empty_with_finish_reason(FinishReason::Unknown),
            }),
        ];
        let stream = Box::pin(futures::stream::iter(events));

        let err = extract_json_value_from_stream(stream)
            .await
            .expect_err("truncated stream should fail");
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn accepts_complete_json_when_stream_ends_without_stream_end_event() {
        let events = vec![Ok(ChatStreamEvent::ContentDelta {
            delta: "{\"value\":\"ok\"}".to_string(),
            index: Some(0),
        })];
        let stream = Box::pin(futures::stream::iter(events));

        let v = extract_json_value_from_stream(stream).await.expect("parse");
        assert_eq!(v["value"], "ok");
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct TypedStructuredOutput {
        value: String,
    }

    #[derive(Debug, Deserialize)]
    struct WrongTypedStructuredOutput {
        #[serde(rename = "value")]
        _value: u32,
    }

    #[test]
    fn extracts_typed_json_from_response() {
        let resp = ChatResponse::new(MessageContent::Text("{\"value\":\"ok\"}".to_string()));
        let typed: TypedStructuredOutput = extract_json_from_response(&resp).expect("typed parse");
        assert_eq!(
            typed,
            TypedStructuredOutput {
                value: "ok".to_string()
            }
        );
    }

    #[test]
    fn typed_extraction_reports_target_type_mismatch() {
        let resp = ChatResponse::new(MessageContent::Text("{\"value\":\"ok\"}".to_string()));
        let err = extract_json_from_response::<WrongTypedStructuredOutput>(&resp)
            .expect_err("typed mismatch");
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("deserialize structured output JSON into target type"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }
}
