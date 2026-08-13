//! Bounded OpenAI-family error-envelope decoding without provider transport concerns.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;
use siumai_core::{
    Error, ErrorKind, MAX_RETRY_AFTER_HINT, PublicDiagnosticText, ResponseDiagnostics,
    SensitiveResponse,
};

const MAX_CLASSIFIER_DEPTH: usize = 8;
const MAX_CLASSIFIER_NODES: usize = 128;
const MAX_CLASSIFIER_TEXT_BYTES: usize = 8 * 1024;
const MAX_IDENTIFIER_BYTES: usize = 256;

/// Matchable metadata carried by an OpenAI-family error envelope.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiErrorMetadata {
    code: Option<String>,
    error_type: Option<String>,
    param: Option<String>,
}

impl OpenAiErrorMetadata {
    pub fn code(&self) -> Option<&str> {
        self.code.as_deref()
    }

    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }
}

/// Decode only bounded, matchable identifiers; raw provider text remains sensitive.
pub fn decode_error_metadata(body: &[u8]) -> Option<OpenAiErrorMetadata> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    let inspection = inspect_error(&value);
    if inspection.exhausted {
        return Some(OpenAiErrorMetadata::default());
    }
    Some(OpenAiErrorMetadata {
        code: unique_string(&inspection.codes),
        error_type: unique_string(&inspection.types),
        param: unique_string(&inspection.params),
    })
}

/// Classify an OpenAI-family HTTP failure using status and exact provider identifiers.
pub fn classify_http_error(
    status: u16,
    provider_code: Option<&str>,
    provider_type: Option<&str>,
) -> ErrorKind {
    provider_code
        .and_then(classify_identifier)
        .or_else(|| provider_type.and_then(classify_identifier))
        .unwrap_or_else(|| classify_status(status))
}

/// Convert a bounded in-band OpenAI-family error envelope into the canonical error contract.
pub fn classify_stream_error(
    envelope: &Value,
    diagnostics: ResponseDiagnostics,
    public_message: &'static str,
) -> Error {
    let inspection = inspect_error(envelope);
    let kind = classified_kind(&inspection).unwrap_or(ErrorKind::Provider);
    let diagnostics = merge_diagnostics(diagnostics, &inspection);
    let body = serde_json::to_vec(envelope).unwrap_or_default();
    let sensitive = SensitiveResponse::new(BTreeMap::new(), body);
    let body_truncated = diagnostics.body_truncated() || sensitive.was_truncated();
    Error::new(kind, public_message)
        .with_diagnostics(diagnostics.with_body_truncated(body_truncated))
        .with_sensitive_response(sensitive)
}

fn classified_kind(inspection: &ErrorInspection) -> Option<ErrorKind> {
    if inspection.exhausted {
        return None;
    }
    let code_kinds = classified_identifiers(&inspection.codes);
    if !code_kinds.is_empty() {
        return match code_kinds.as_slice() {
            [kind] => Some(*kind),
            _ => None,
        };
    }
    let type_kinds = classified_identifiers(&inspection.types);
    match type_kinds.as_slice() {
        [kind] => Some(*kind),
        [] => unique_copy(&inspection.statuses).map(classify_status),
        _ => None,
    }
}

fn classified_identifiers(identifiers: &[String]) -> Vec<ErrorKind> {
    let mut kinds = Vec::new();
    for identifier in identifiers {
        if let Some(kind) = classify_identifier(identifier)
            && !kinds.contains(&kind)
        {
            kinds.push(kind);
        }
    }
    kinds
}

fn classify_identifier(identifier: &str) -> Option<ErrorKind> {
    match identifier.to_ascii_lowercase().as_str() {
        "context_length_exceeded"
        | "context_window_exceeded"
        | "max_context_length_exceeded"
        | "prompt_too_long" => Some(ErrorKind::ContextWindowExceeded),
        "rate_limit_exceeded"
        | "rate_limit_error"
        | "concurrency_limit_exceeded"
        | "concurrency_limit_error"
        | "too_many_requests" => Some(ErrorKind::RateLimited),
        "insufficient_quota"
        | "billing_error"
        | "billing_hard_limit_reached"
        | "quota_exceeded" => Some(ErrorKind::QuotaExceeded),
        "timeout" | "timeout_error" | "request_timeout" => Some(ErrorKind::Timeout),
        "invalid_input" | "invalid_parameter" | "invalid_request_error" => {
            Some(ErrorKind::InvalidInput)
        }
        "authentication_error" | "invalid_api_key" | "unauthorized" => {
            Some(ErrorKind::Authentication)
        }
        "forbidden" | "permission_denied" | "permission_error" => Some(ErrorKind::Authorization),
        "api_error"
        | "overloaded_error"
        | "server_error"
        | "service_unavailable"
        | "temporarily_unavailable" => Some(ErrorKind::Unavailable),
        _ => None,
    }
}

fn classify_status(status: u16) -> ErrorKind {
    match status {
        400 | 413 | 422 => ErrorKind::InvalidInput,
        401 => ErrorKind::Authentication,
        403 => ErrorKind::Authorization,
        408 | 504 => ErrorKind::Timeout,
        402 => ErrorKind::QuotaExceeded,
        429 => ErrorKind::RateLimited,
        500 | 502 | 503 | 529 => ErrorKind::Unavailable,
        _ => ErrorKind::Provider,
    }
}

fn merge_diagnostics(
    mut diagnostics: ResponseDiagnostics,
    inspection: &ErrorInspection,
) -> ResponseDiagnostics {
    if inspection.exhausted {
        return diagnostics;
    }
    if diagnostics.status().is_none()
        && let Some(status) = unique_copy(&inspection.statuses)
    {
        diagnostics = diagnostics.with_status(status);
    }
    if let Some(code) = unique_string(&inspection.codes).and_then(public_identifier) {
        diagnostics = diagnostics.with_provider_code(code);
    }
    if let Some(error_type) = unique_string(&inspection.types).and_then(public_identifier) {
        diagnostics = diagnostics.with_provider_type(error_type);
    }
    if let Some(param) = unique_string(&inspection.params).and_then(public_identifier) {
        diagnostics = diagnostics.with_provider_param(param);
    }
    match unique_copy_result(&inspection.retry_after) {
        Ok(Some(retry_after)) => match diagnostics.retry_after() {
            Some(existing) if existing != retry_after => {
                diagnostics = diagnostics.without_retry_after();
            }
            Some(_) => {}
            None => diagnostics = diagnostics.with_retry_after(retry_after),
        },
        Ok(None) => {}
        Err(()) => diagnostics = diagnostics.without_retry_after(),
    }
    diagnostics
}

fn public_identifier(value: String) -> Option<PublicDiagnosticText> {
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return None;
    }
    PublicDiagnosticText::new(value).ok()
}

#[derive(Default)]
struct ErrorInspection {
    codes: Vec<String>,
    types: Vec<String>,
    params: Vec<String>,
    statuses: Vec<u16>,
    retry_after: Vec<Duration>,
    nodes: usize,
    text_bytes: usize,
    exhausted: bool,
}

fn inspect_error(value: &Value) -> ErrorInspection {
    let mut inspection = ErrorInspection::default();
    inspection.visit(value, 0);
    inspection
}

impl ErrorInspection {
    fn visit(&mut self, value: &Value, depth: usize) {
        if self.exhausted {
            return;
        }
        if depth > MAX_CLASSIFIER_DEPTH {
            self.exhausted = true;
            return;
        }
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > MAX_CLASSIFIER_NODES {
            self.exhausted = true;
            return;
        }
        match value {
            Value::Object(object) => {
                for (field, value) in object {
                    if !self.consume_text(field) {
                        return;
                    }
                    match field.as_str() {
                        "code" | "error_code" => {
                            if let Some(value) = bounded_identifier(value) {
                                self.codes.push(value.to_string());
                            }
                        }
                        "type" => {
                            if let Some(value) = bounded_identifier(value) {
                                self.types.push(value.to_string());
                            }
                        }
                        "param" => {
                            if let Some(value) = bounded_identifier(value) {
                                self.params.push(value.to_string());
                            }
                        }
                        "status" => {
                            if let Some(status) = parse_status(value) {
                                self.statuses.push(status);
                            }
                        }
                        "retry_after" => {
                            if let Some(retry_after) = parse_retry_after(value, false) {
                                self.retry_after.push(retry_after);
                            }
                        }
                        "retry_after_ms" => {
                            if let Some(retry_after) = parse_retry_after(value, true) {
                                self.retry_after.push(retry_after);
                            }
                        }
                        _ => {}
                    }
                    self.visit(value, depth + 1);
                }
            }
            Value::Array(values) => {
                for value in values {
                    self.visit(value, depth + 1);
                }
            }
            Value::String(value) => {
                self.consume_text(value);
            }
            _ => {}
        }
    }

    fn consume_text(&mut self, value: &str) -> bool {
        self.text_bytes = self.text_bytes.saturating_add(value.len());
        if self.text_bytes > MAX_CLASSIFIER_TEXT_BYTES {
            self.exhausted = true;
            false
        } else {
            true
        }
    }
}

fn bounded_identifier(value: &Value) -> Option<&str> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .filter(|value| value.len() <= MAX_IDENTIFIER_BYTES)
        .filter(|value| !value.chars().any(char::is_control))
}

fn parse_status(value: &Value) -> Option<u16> {
    let status = match value {
        Value::Number(value) => value.as_u64().and_then(|value| u16::try_from(value).ok()),
        Value::String(value)
            if value.len() == 3 && value.bytes().all(|byte| byte.is_ascii_digit()) =>
        {
            value.parse::<u16>().ok()
        }
        _ => None,
    }?;
    (100..=599).contains(&status).then_some(status)
}

fn parse_retry_after(value: &Value, milliseconds: bool) -> Option<Duration> {
    let value = match value {
        Value::Number(value) => value.as_u64(),
        Value::String(value)
            if !value.is_empty()
                && value.len() <= 10
                && value.bytes().all(|byte| byte.is_ascii_digit()) =>
        {
            value.parse::<u64>().ok()
        }
        _ => None,
    }?;
    let duration = if milliseconds {
        Duration::from_millis(value)
    } else {
        Duration::from_secs(value)
    };
    (duration <= MAX_RETRY_AFTER_HINT).then_some(duration)
}

fn unique_string(values: &[String]) -> Option<String> {
    let first = values.first()?;
    values
        .iter()
        .all(|value| value == first)
        .then(|| first.clone())
}

fn unique_copy<T>(values: &[T]) -> Option<T>
where
    T: Copy + PartialEq,
{
    unique_copy_result(values).ok().flatten()
}

fn unique_copy_result<T>(values: &[T]) -> Result<Option<T>, ()>
where
    T: Copy + PartialEq,
{
    let Some(first) = values.first().copied() else {
        return Ok(None);
    };
    if values.iter().all(|value| *value == first) {
        Ok(Some(first))
    } else {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decodes_bounded_openai_error_identifiers() {
        let metadata = decode_error_metadata(
            br#"{"error":{"message":"invalid","type":"invalid_request_error","param":"thinking.keep","code":"invalid_parameter"}}"#,
        )
        .unwrap();

        assert_eq!(metadata.code(), Some("invalid_parameter"));
        assert_eq!(metadata.error_type(), Some("invalid_request_error"));
        assert_eq!(metadata.param(), Some("thinking.keep"));
        assert_eq!(
            classify_http_error(400, metadata.code(), metadata.error_type()),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            classify_http_error(400, None, metadata.error_type()),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            classify_http_error(429, Some("insufficient_quota"), None),
            ErrorKind::QuotaExceeded
        );
        assert_eq!(
            classify_http_error(400, None, Some("overloaded_error")),
            ErrorKind::Unavailable
        );
        assert_eq!(classify_http_error(503, None, None), ErrorKind::Unavailable);
    }

    #[test]
    fn stream_classification_is_typed_bounded_and_redacted() {
        let error = classify_stream_error(
            &json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "message": "stream-error-sentinel",
                    "status": "429",
                    "retry_after": 3
                }
            }),
            ResponseDiagnostics::default().with_status(200),
            "provider stream reported an error",
        );

        assert_eq!(error.kind(), ErrorKind::RateLimited);
        assert_eq!(
            error.diagnostics().and_then(ResponseDiagnostics::status),
            Some(200)
        );
        assert_eq!(
            error
                .diagnostics()
                .and_then(ResponseDiagnostics::retry_after),
            Some(Duration::from_secs(3))
        );
        assert!(!format!("{error:?}").contains("stream-error-sentinel"));
        assert!(!error.to_string().contains("stream-error-sentinel"));
    }

    #[test]
    fn context_and_status_classification_fail_closed() {
        let context = classify_stream_error(
            &json!({
                "error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded"
                }
            }),
            ResponseDiagnostics::default(),
            "provider stream reported an error",
        );
        assert_eq!(context.kind(), ErrorKind::ContextWindowExceeded);

        for status in [
            json!("000000000000000000000000000000429"),
            json!("0429"),
            json!(99),
            json!(600),
        ] {
            let error = classify_stream_error(
                &json!({"error": {"status": status, "message": "context_length_exceeded"}}),
                ResponseDiagnostics::default(),
                "provider stream reported an error",
            );
            assert_eq!(error.kind(), ErrorKind::Provider);
        }

        let overlong = "a".repeat(MAX_IDENTIFIER_BYTES + 1) + "rate_limit_exceeded";
        let error = classify_stream_error(
            &json!({"error": {"code": overlong}}),
            ResponseDiagnostics::default(),
            "provider stream reported an error",
        );
        assert_eq!(error.kind(), ErrorKind::Provider);

        let exhausted = classify_stream_error(
            &json!({
                "error": {
                    "code": "rate_limit_exceeded",
                    "zz_payload": "x".repeat(MAX_CLASSIFIER_TEXT_BYTES)
                }
            }),
            ResponseDiagnostics::default(),
            "provider stream reported an error",
        );
        assert_eq!(exhausted.kind(), ErrorKind::Provider);
        assert_eq!(
            exhausted
                .diagnostics()
                .and_then(ResponseDiagnostics::provider_code),
            None
        );
    }

    #[test]
    fn conflicting_retry_hints_are_removed() {
        let diagnostics = ResponseDiagnostics::default().with_retry_after(Duration::from_secs(5));
        let error = classify_stream_error(
            &json!({"error": {"code": "rate_limit_exceeded", "retry_after": 7}}),
            diagnostics,
            "provider stream reported an error",
        );

        assert_eq!(
            error
                .diagnostics()
                .and_then(ResponseDiagnostics::retry_after),
            None
        );
    }

    #[test]
    fn malformed_or_non_string_identifiers_are_not_promoted() {
        assert!(decode_error_metadata(b"not-json").is_none());
        let metadata = decode_error_metadata(br#"{"error":{"code":42,"param":null}}"#).unwrap();
        assert_eq!(metadata, OpenAiErrorMetadata::default());
    }
}
