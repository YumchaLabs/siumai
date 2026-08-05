//! OpenAI-family error-envelope decoding without provider transport concerns.

use serde::Deserialize;
use serde_json::Value;
use siumai_core::ErrorKind;

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

#[derive(Debug, Deserialize)]
struct ErrorEnvelopeWire {
    error: ErrorBodyWire,
}

#[derive(Debug, Deserialize)]
struct ErrorBodyWire {
    #[serde(default)]
    code: Option<Value>,
    #[serde(rename = "type", default)]
    error_type: Option<String>,
    #[serde(default)]
    param: Option<Value>,
}

/// Decode only matchable identifiers; the provider layer retains the bounded raw body.
pub fn decode_error_metadata(body: &[u8]) -> Option<OpenAiErrorMetadata> {
    let wire = serde_json::from_slice::<ErrorEnvelopeWire>(body).ok()?;
    Some(OpenAiErrorMetadata {
        code: wire.error.code.and_then(string_value),
        error_type: wire.error.error_type,
        param: wire.error.param.and_then(string_value),
    })
}

/// Classify an OpenAI-family HTTP failure using status and provider code.
pub fn classify_http_error(status: u16, provider_code: Option<&str>) -> ErrorKind {
    if matches!(
        provider_code,
        Some("insufficient_quota" | "billing_hard_limit_reached")
    ) {
        return ErrorKind::QuotaExceeded;
    }
    match status {
        400 | 422 => ErrorKind::InvalidInput,
        401 => ErrorKind::Authentication,
        403 => ErrorKind::Authorization,
        408 | 504 => ErrorKind::Timeout,
        429 => ErrorKind::RateLimited,
        _ => ErrorKind::Provider,
    }
}

fn string_value(value: Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_kimi_and_openai_error_identifiers() {
        let metadata = decode_error_metadata(
            br#"{"error":{"message":"invalid","type":"invalid_request_error","param":"thinking.keep","code":"invalid_parameter"}}"#,
        )
        .unwrap();

        assert_eq!(metadata.code(), Some("invalid_parameter"));
        assert_eq!(metadata.error_type(), Some("invalid_request_error"));
        assert_eq!(metadata.param(), Some("thinking.keep"));
        assert_eq!(
            classify_http_error(400, metadata.code()),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            classify_http_error(429, Some("insufficient_quota")),
            ErrorKind::QuotaExceeded
        );
    }

    #[test]
    fn malformed_or_non_string_identifiers_are_not_promoted() {
        assert!(decode_error_metadata(b"not-json").is_none());
        let metadata = decode_error_metadata(br#"{"error":{"code":42,"param":null}}"#).unwrap();
        assert_eq!(metadata, OpenAiErrorMetadata::default());
    }
}
