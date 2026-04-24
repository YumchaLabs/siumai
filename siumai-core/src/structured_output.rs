//! Provider-agnostic structured output helpers.
//!
//! This module focuses on best-effort JSON extraction from model output. Provider-specific
//! schema enforcement lives at the protocol/provider layers (e.g. OpenAI `response_format`,
//! Anthropic `output_format`, Gemini `responseSchema`, or reserved tool strategies).

use crate::error::LlmError;
use crate::streaming::{ChatStream, StreamProcessor};
use crate::types::{ChatResponse, ChatStreamEvent, ContentPart, FinishReason, MessageContent};
use futures::StreamExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonRepairState {
    Root,
    Finish,
    InsideString,
    InsideStringEscape,
    InsideLiteral,
    InsideNumber,
    InsideObjectStart,
    InsideObjectKey,
    InsideObjectAfterKey,
    InsideObjectBeforeValue,
    InsideObjectAfterValue,
    InsideObjectAfterComma,
    InsideArrayStart,
    InsideArrayAfterValue,
    InsideArrayAfterComma,
}

/// State returned by `parse_partial_json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialJsonParseState {
    /// No input was provided.
    UndefinedInput,
    /// The input parsed successfully without repair.
    SuccessfulParse,
    /// The input needed partial-JSON repair before it parsed successfully.
    RepairedParse,
    /// The input could not be parsed even after repair.
    FailedParse,
}

/// Result returned by `parse_partial_json`.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialJsonParseResult {
    /// Parsed JSON value when parsing succeeded.
    pub value: Option<serde_json::Value>,
    /// Parse state describing whether repair was needed.
    pub state: PartialJsonParseState,
}

fn process_partial_value_start(
    stack: &mut Vec<JsonRepairState>,
    ch: char,
    byte_start: usize,
    byte_end: usize,
    last_valid_end: &mut Option<usize>,
    literal_start: &mut Option<usize>,
    swap_state: JsonRepairState,
) {
    match ch {
        '"' => {
            *last_valid_end = Some(byte_end);
            stack.pop();
            stack.push(swap_state);
            stack.push(JsonRepairState::InsideString);
        }
        'f' | 't' | 'n' => {
            *last_valid_end = Some(byte_end);
            *literal_start = Some(byte_start);
            stack.pop();
            stack.push(swap_state);
            stack.push(JsonRepairState::InsideLiteral);
        }
        '-' => {
            stack.pop();
            stack.push(swap_state);
            stack.push(JsonRepairState::InsideNumber);
        }
        '0'..='9' => {
            *last_valid_end = Some(byte_end);
            stack.pop();
            stack.push(swap_state);
            stack.push(JsonRepairState::InsideNumber);
        }
        '{' => {
            *last_valid_end = Some(byte_end);
            stack.pop();
            stack.push(swap_state);
            stack.push(JsonRepairState::InsideObjectStart);
        }
        '[' => {
            *last_valid_end = Some(byte_end);
            stack.pop();
            stack.push(swap_state);
            stack.push(JsonRepairState::InsideArrayStart);
        }
        _ => {}
    }
}

fn process_after_object_value(
    stack: &mut Vec<JsonRepairState>,
    ch: char,
    byte_end: usize,
    last_valid_end: &mut Option<usize>,
) {
    match ch {
        ',' => {
            stack.pop();
            stack.push(JsonRepairState::InsideObjectAfterComma);
        }
        '}' => {
            *last_valid_end = Some(byte_end);
            stack.pop();
        }
        _ => {}
    }
}

fn process_after_array_value(
    stack: &mut Vec<JsonRepairState>,
    ch: char,
    byte_end: usize,
    last_valid_end: &mut Option<usize>,
) {
    match ch {
        ',' => {
            stack.pop();
            stack.push(JsonRepairState::InsideArrayAfterComma);
        }
        ']' => {
            *last_valid_end = Some(byte_end);
            stack.pop();
        }
        _ => {}
    }
}

/// Repair incomplete JSON using the same single-pass strategy as AI SDK `fixJson`.
///
/// This helper is intentionally conservative: it repairs incomplete JSON by truncating to the last
/// valid token and appending required delimiters. Fully invalid JSON is still rejected by the
/// subsequent JSON parser.
pub fn fix_partial_json(input: &str) -> String {
    let mut stack = vec![JsonRepairState::Root];
    let mut last_valid_end: Option<usize> = None;
    let mut literal_start: Option<usize> = None;

    for (byte_start, ch) in input.char_indices() {
        let byte_end = byte_start + ch.len_utf8();
        let current_state = *stack.last().unwrap_or(&JsonRepairState::Finish);

        match current_state {
            JsonRepairState::Root => {
                process_partial_value_start(
                    &mut stack,
                    ch,
                    byte_start,
                    byte_end,
                    &mut last_valid_end,
                    &mut literal_start,
                    JsonRepairState::Finish,
                );
            }
            JsonRepairState::InsideObjectStart => match ch {
                '"' => {
                    stack.pop();
                    stack.push(JsonRepairState::InsideObjectKey);
                }
                '}' => {
                    last_valid_end = Some(byte_end);
                    stack.pop();
                }
                _ => {}
            },
            JsonRepairState::InsideObjectAfterComma => {
                if ch == '"' {
                    stack.pop();
                    stack.push(JsonRepairState::InsideObjectKey);
                }
            }
            JsonRepairState::InsideObjectKey => {
                if ch == '"' {
                    stack.pop();
                    stack.push(JsonRepairState::InsideObjectAfterKey);
                }
            }
            JsonRepairState::InsideObjectAfterKey => {
                if ch == ':' {
                    stack.pop();
                    stack.push(JsonRepairState::InsideObjectBeforeValue);
                }
            }
            JsonRepairState::InsideObjectBeforeValue => {
                process_partial_value_start(
                    &mut stack,
                    ch,
                    byte_start,
                    byte_end,
                    &mut last_valid_end,
                    &mut literal_start,
                    JsonRepairState::InsideObjectAfterValue,
                );
            }
            JsonRepairState::InsideObjectAfterValue => {
                process_after_object_value(&mut stack, ch, byte_end, &mut last_valid_end);
            }
            JsonRepairState::InsideString => match ch {
                '"' => {
                    stack.pop();
                    last_valid_end = Some(byte_end);
                }
                '\\' => stack.push(JsonRepairState::InsideStringEscape),
                _ => last_valid_end = Some(byte_end),
            },
            JsonRepairState::InsideArrayStart => match ch {
                ']' => {
                    last_valid_end = Some(byte_end);
                    stack.pop();
                }
                _ => {
                    last_valid_end = Some(byte_end);
                    process_partial_value_start(
                        &mut stack,
                        ch,
                        byte_start,
                        byte_end,
                        &mut last_valid_end,
                        &mut literal_start,
                        JsonRepairState::InsideArrayAfterValue,
                    );
                }
            },
            JsonRepairState::InsideArrayAfterValue => match ch {
                ',' => {
                    stack.pop();
                    stack.push(JsonRepairState::InsideArrayAfterComma);
                }
                ']' => {
                    last_valid_end = Some(byte_end);
                    stack.pop();
                }
                _ => last_valid_end = Some(byte_end),
            },
            JsonRepairState::InsideArrayAfterComma => {
                process_partial_value_start(
                    &mut stack,
                    ch,
                    byte_start,
                    byte_end,
                    &mut last_valid_end,
                    &mut literal_start,
                    JsonRepairState::InsideArrayAfterValue,
                );
            }
            JsonRepairState::InsideStringEscape => {
                stack.pop();
                last_valid_end = Some(byte_end);
            }
            JsonRepairState::InsideNumber => match ch {
                '0'..='9' => last_valid_end = Some(byte_end),
                'e' | 'E' | '-' | '.' => {}
                ',' => {
                    stack.pop();
                    if stack.last() == Some(&JsonRepairState::InsideArrayAfterValue) {
                        process_after_array_value(&mut stack, ch, byte_end, &mut last_valid_end);
                    }
                    if stack.last() == Some(&JsonRepairState::InsideObjectAfterValue) {
                        process_after_object_value(&mut stack, ch, byte_end, &mut last_valid_end);
                    }
                }
                '}' => {
                    stack.pop();
                    if stack.last() == Some(&JsonRepairState::InsideObjectAfterValue) {
                        process_after_object_value(&mut stack, ch, byte_end, &mut last_valid_end);
                    }
                }
                ']' => {
                    stack.pop();
                    if stack.last() == Some(&JsonRepairState::InsideArrayAfterValue) {
                        process_after_array_value(&mut stack, ch, byte_end, &mut last_valid_end);
                    }
                }
                _ => {
                    stack.pop();
                }
            },
            JsonRepairState::InsideLiteral => {
                let Some(start) = literal_start else {
                    continue;
                };
                let partial_literal = &input[start..byte_end];

                if !("false".starts_with(partial_literal)
                    || "true".starts_with(partial_literal)
                    || "null".starts_with(partial_literal))
                {
                    stack.pop();

                    if stack.last() == Some(&JsonRepairState::InsideObjectAfterValue) {
                        process_after_object_value(&mut stack, ch, byte_end, &mut last_valid_end);
                    } else if stack.last() == Some(&JsonRepairState::InsideArrayAfterValue) {
                        process_after_array_value(&mut stack, ch, byte_end, &mut last_valid_end);
                    }
                } else {
                    last_valid_end = Some(byte_end);
                }
            }
            JsonRepairState::Finish => {}
        }
    }

    let mut result = input[..last_valid_end.unwrap_or(0)].to_string();

    for state in stack.iter().rev() {
        match state {
            JsonRepairState::InsideString => result.push('"'),
            JsonRepairState::InsideObjectKey
            | JsonRepairState::InsideObjectAfterKey
            | JsonRepairState::InsideObjectAfterComma
            | JsonRepairState::InsideObjectStart
            | JsonRepairState::InsideObjectBeforeValue
            | JsonRepairState::InsideObjectAfterValue => result.push('}'),
            JsonRepairState::InsideArrayStart
            | JsonRepairState::InsideArrayAfterComma
            | JsonRepairState::InsideArrayAfterValue => result.push(']'),
            JsonRepairState::InsideLiteral => {
                let Some(start) = literal_start else {
                    continue;
                };
                let partial_literal = &input[start..];
                if "true".starts_with(partial_literal) {
                    result.push_str(&"true"[partial_literal.len()..]);
                } else if "false".starts_with(partial_literal) {
                    result.push_str(&"false"[partial_literal.len()..]);
                } else if "null".starts_with(partial_literal) {
                    result.push_str(&"null"[partial_literal.len()..]);
                }
            }
            JsonRepairState::Root
            | JsonRepairState::Finish
            | JsonRepairState::InsideStringEscape
            | JsonRepairState::InsideNumber => {}
        }
    }

    result
}

/// Parse a partial JSON string using AI SDK `parsePartialJson` semantics.
pub fn parse_partial_json(json_text: Option<&str>) -> PartialJsonParseResult {
    let Some(json_text) = json_text else {
        return PartialJsonParseResult {
            value: None,
            state: PartialJsonParseState::UndefinedInput,
        };
    };

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_text) {
        return PartialJsonParseResult {
            value: Some(value),
            state: PartialJsonParseState::SuccessfulParse,
        };
    }

    let repaired = fix_partial_json(json_text);
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
        return PartialJsonParseResult {
            value: Some(value),
            state: PartialJsonParseState::RepairedParse,
        };
    }

    PartialJsonParseResult {
        value: None,
        state: PartialJsonParseState::FailedParse,
    }
}

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
        raw_finish_reason,
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
        raw_finish_reason: raw_finish_reason.or(accumulated.raw_finish_reason),
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
    use crate::types::{ChatResponse, MessageContent, Usage};
    use serde::Deserialize;

    #[test]
    fn fix_partial_json_matches_ai_sdk_scalar_repairs() {
        assert_eq!(fix_partial_json(""), "");
        assert_eq!(fix_partial_json("nul"), "null");
        assert_eq!(fix_partial_json("t"), "true");
        assert_eq!(fix_partial_json("fals"), "false");
        assert_eq!(fix_partial_json("12."), "12");
        assert_eq!(fix_partial_json("-"), "");
        assert_eq!(fix_partial_json("2.5e-"), "2.5");
        assert_eq!(fix_partial_json(r#""abc"#), r#""abc""#);
        assert_eq!(
            fix_partial_json(r#""value with \"quoted\" text"#),
            r#""value with \"quoted\" text""#
        );
    }

    #[test]
    fn fix_partial_json_matches_ai_sdk_array_and_object_repairs() {
        assert_eq!(fix_partial_json("["), "[]");
        assert_eq!(fix_partial_json("[[1], [2"), "[[1], [2]]");
        assert_eq!(fix_partial_json("[1, "), "[1]");
        assert_eq!(fix_partial_json(r#"{"key":"#), "{}");
        assert_eq!(
            fix_partial_json(r#"{"a": {"b": 1}, "c": {"d": 2"#),
            r#"{"a": {"b": 1}, "c": {"d": 2}}"#
        );
        assert_eq!(fix_partial_json(r#"{"ke"#), "{}");
        assert_eq!(fix_partial_json(r#"{"k1": 1, "k2":"#), r#"{"k1": 1}"#);
        assert_eq!(
            fix_partial_json(r#"{"key": [1, 2, {"#),
            r#"{"key": [1, 2, {}]}"#
        );
    }

    #[test]
    fn parse_partial_json_reports_state() {
        let undefined = parse_partial_json(None);
        assert_eq!(undefined.state, PartialJsonParseState::UndefinedInput);
        assert_eq!(undefined.value, None);

        let successful = parse_partial_json(Some(r#"{"key":"value"}"#));
        assert_eq!(successful.state, PartialJsonParseState::SuccessfulParse);
        assert_eq!(
            successful.value,
            Some(serde_json::json!({ "key": "value" }))
        );

        let repaired = parse_partial_json(Some(r#"{"key":"value""#));
        assert_eq!(repaired.state, PartialJsonParseState::RepairedParse);
        assert_eq!(repaired.value, Some(serde_json::json!({ "key": "value" })));

        let failed = parse_partial_json(Some("not json at all"));
        assert_eq!(failed.state, PartialJsonParseState::FailedParse);
        assert_eq!(failed.value, None);
    }

    #[test]
    fn parse_partial_json_handles_unicode_inside_strings() {
        let repaired = parse_partial_json(Some(r#"{"text":"你好"#));
        assert_eq!(repaired.state, PartialJsonParseState::RepairedParse);
        assert_eq!(repaired.value, Some(serde_json::json!({ "text": "你好" })));
    }

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
    async fn extracts_json_from_unknown_stream_end_when_stream_end_response_has_json_text() {
        let response = ChatResponse {
            id: None,
            content: MessageContent::MultiModal(vec![ContentPart::text("{\"value\":\"ok\"}")]),
            model: Some("anthropic.claude-3-haiku-20240307-v1:0".to_string()),
            usage: Some(Usage::new(15, 42)),
            finish_reason: Some(FinishReason::Unknown),
            raw_finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: Some(std::collections::HashMap::from([(
                "bedrock".to_string(),
                serde_json::json!({ "isJsonResponseFromTool": true }),
            )])),
        };

        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "{\"value\":\"ok\"}".to_string(),
                index: Some(0),
            }),
            Ok(ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::TextDelta {
                    id: "0".to_string(),
                    delta: "{\"value\":\"ok\"}".to_string(),
                    provider_metadata: None,
                },
            }),
            Ok(ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::Finish {
                    usage: Usage::new(15, 42),
                    finish_reason: crate::types::ChatStreamFinishInfo {
                        unified: FinishReason::Unknown,
                        raw: None,
                    },
                    provider_metadata: Some(std::collections::HashMap::from([(
                        "bedrock".to_string(),
                        serde_json::json!({ "isJsonResponseFromTool": true }),
                    )])),
                },
            }),
            Ok(ChatStreamEvent::StreamEnd { response }),
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
