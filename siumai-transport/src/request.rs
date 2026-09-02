//! Immutable, rebuildable provider request plans.

use std::fmt;
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use http::Method;
use http::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;

use crate::replay::{is_common_credential_header, is_transport_controlled};
use crate::{ReplaySafety, RequestBuildError, TransportLimits};

const MAX_TARGET_BYTES: usize = 8 * 1024;

/// Relative request target that cannot select another origin.
#[derive(Clone, PartialEq, Eq)]
pub struct RequestTarget(String);

impl RequestTarget {
    pub fn new(value: impl Into<String>) -> Result<Self, RequestBuildError> {
        let value = value.into();
        if value.len() > MAX_TARGET_BYTES {
            return Err(RequestBuildError::TargetTooLong {
                maximum: MAX_TARGET_BYTES,
            });
        }
        let path = value.split('?').next().unwrap_or_default();
        if value.starts_with("//")
            || value.contains(['\r', '\n', '#', '\\'])
            || path.split('/').any(is_unsafe_path_segment)
            || reqwest::Url::parse(&value).is_ok()
        {
            return Err(RequestBuildError::InvalidTarget);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Append one provider-owned opaque identifier as a single path segment.
    ///
    /// Provider codecs should pass the raw identifier to this method after
    /// applying their provider-specific byte bound. The segment is encoded
    /// exactly once and inserted before any existing query. Generic callers
    /// should continue to use [`RequestTarget::new`], whose traversal checks
    /// remain intentionally strict.
    ///
    /// Exact `.` and `..` segments are rejected because URL implementations
    /// normalize them before dispatch. Other dots remain ordinary identifier
    /// data.
    pub fn with_opaque_path_segment(
        self,
        segment: impl AsRef<str>,
    ) -> Result<Self, RequestBuildError> {
        let segment = segment.as_ref();
        if segment.is_empty()
            || matches!(segment, "." | "..")
            || segment.chars().any(char::is_control)
        {
            return Err(RequestBuildError::InvalidTarget);
        }

        let path_end = self.0.find('?').unwrap_or(self.0.len());
        let path = &self.0[..path_end];
        let query = &self.0[path_end..];
        let separator = usize::from(!path.is_empty() && !path.ends_with('/'));
        let encoded_bytes = segment.as_bytes().iter().fold(0_usize, |length, byte| {
            length.saturating_add(if is_unreserved_path_byte(*byte) { 1 } else { 3 })
        });
        let total_bytes = path
            .len()
            .saturating_add(separator)
            .saturating_add(encoded_bytes)
            .saturating_add(query.len());
        if total_bytes > MAX_TARGET_BYTES {
            return Err(RequestBuildError::TargetTooLong {
                maximum: MAX_TARGET_BYTES,
            });
        }

        let mut value = String::with_capacity(total_bytes);
        value.push_str(path);
        if separator != 0 {
            value.push('/');
        }
        push_encoded_path_segment(&mut value, segment.as_bytes());
        value.push_str(query);
        Ok(Self(value))
    }
}

impl fmt::Debug for RequestTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestTarget")
            .field("bytes", &self.0.len())
            .field("has_query", &self.0.contains('?'))
            .finish()
    }
}

/// Non-credential headers supplied by protocol code.
///
/// Common credential and transport headers are rejected immediately. Any
/// provider-specific header emitted by the selected credential applier is
/// rejected on exact collision before network submission.
#[derive(Clone, Default)]
pub struct RequestHeaders(HeaderMap);

impl RequestHeaders {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_insert(
        mut self,
        name: HeaderName,
        value: HeaderValue,
    ) -> Result<Self, RequestBuildError> {
        if is_transport_controlled(&name) || is_common_credential_header(&name) {
            return Err(RequestBuildError::ProtectedHeader);
        }
        self.0.insert(name, value);
        Ok(self)
    }

    pub fn get(&self, name: &HeaderName) -> Option<&HeaderValue> {
        self.0.get(name)
    }

    pub fn iter(&self) -> http::header::Iter<'_, HeaderValue> {
        self.0.iter()
    }

    pub(crate) fn clone_inner(&self) -> HeaderMap {
        self.0.clone()
    }

    pub(crate) fn validate(&self, limits: &TransportLimits) -> Result<(), RequestBuildError> {
        if self.0.len() > limits.max_header_count {
            return Err(RequestBuildError::TooManyHeaders);
        }
        if self
            .0
            .values()
            .any(|value| value.as_bytes().len() > limits.max_header_value_bytes)
        {
            return Err(RequestBuildError::HeaderValueTooLarge);
        }
        Ok(())
    }
}

impl fmt::Debug for RequestHeaders {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestHeaders")
            .field("count", &self.0.len())
            .field("contents", &"[REDACTED]")
            .finish()
    }
}

/// One deterministic multipart part.
#[derive(Clone)]
pub struct MultipartPart {
    name: String,
    file_name: Option<String>,
    content_type: Option<HeaderValue>,
    data: Bytes,
}

impl MultipartPart {
    pub fn field(
        name: impl Into<String>,
        data: impl Into<Bytes>,
    ) -> Result<Self, RequestBuildError> {
        let name = checked_disposition_value(name.into())?;
        Ok(Self {
            name,
            file_name: None,
            content_type: None,
            data: data.into(),
        })
    }

    pub fn file(
        name: impl Into<String>,
        file_name: impl Into<String>,
        content_type: HeaderValue,
        data: impl Into<Bytes>,
    ) -> Result<Self, RequestBuildError> {
        Ok(Self {
            name: checked_disposition_value(name.into())?,
            file_name: Some(checked_disposition_value(file_name.into())?),
            content_type: Some(content_type),
            data: data.into(),
        })
    }
}

impl fmt::Debug for MultipartPart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MultipartPart")
            .field("name_bytes", &self.name.len())
            .field(
                "file_name_bytes",
                &self.file_name.as_ref().map(|value| value.len()),
            )
            .field("content_type_present", &self.content_type.is_some())
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

/// Deterministic multipart body rebuilt byte-for-byte for every attempt.
#[derive(Clone)]
pub struct MultipartBody {
    boundary: Arc<str>,
    parts: Arc<[MultipartPart]>,
}

impl MultipartBody {
    pub fn new(parts: Vec<MultipartPart>) -> Self {
        Self {
            boundary: format!("siumai-{}", uuid::Uuid::new_v4()).into(),
            parts: parts.into(),
        }
    }

    pub fn parts(&self) -> &[MultipartPart] {
        &self.parts
    }

    fn encode(&self, limits: &TransportLimits) -> Result<EncodedBody, RequestBuildError> {
        if self.parts.len() > limits.max_multipart_parts {
            return Err(RequestBuildError::TooManyMultipartParts {
                maximum: limits.max_multipart_parts,
            });
        }
        let mut output = BytesMut::new();
        for part in self.parts.iter() {
            append_bounded(
                &mut output,
                format!("--{}\r\n", self.boundary).as_bytes(),
                limits.max_request_bytes,
            )?;
            let disposition = match &part.file_name {
                Some(file_name) => format!(
                    "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
                    part.name, file_name
                ),
                None => format!("Content-Disposition: form-data; name=\"{}\"\r\n", part.name),
            };
            append_bounded(
                &mut output,
                disposition.as_bytes(),
                limits.max_request_bytes,
            )?;
            if let Some(content_type) = &part.content_type {
                append_bounded(&mut output, b"Content-Type: ", limits.max_request_bytes)?;
                append_bounded(
                    &mut output,
                    content_type.as_bytes(),
                    limits.max_request_bytes,
                )?;
                append_bounded(&mut output, b"\r\n", limits.max_request_bytes)?;
            }
            append_bounded(&mut output, b"\r\n", limits.max_request_bytes)?;
            append_bounded(&mut output, &part.data, limits.max_request_bytes)?;
            append_bounded(&mut output, b"\r\n", limits.max_request_bytes)?;
        }
        append_bounded(
            &mut output,
            format!("--{}--\r\n", self.boundary).as_bytes(),
            limits.max_request_bytes,
        )?;
        let content_type =
            HeaderValue::from_str(&format!("multipart/form-data; boundary={}", self.boundary))
                .map_err(|_| RequestBuildError::InvalidHeaderValue)?;
        Ok(EncodedBody {
            bytes: output.freeze(),
            content_type: Some(content_type),
        })
    }
}

impl fmt::Debug for MultipartBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MultipartBody")
            .field("part_count", &self.parts.len())
            .field("boundary", &"[REDACTED]")
            .finish()
    }
}

/// Closed set of request bodies that can be rebuilt for every attempt.
#[derive(Clone, Default)]
#[non_exhaustive]
pub enum RequestBody {
    #[default]
    Empty,
    Bytes {
        data: Bytes,
        content_type: Option<HeaderValue>,
    },
    Multipart(MultipartBody),
}

impl RequestBody {
    pub fn bytes(data: impl Into<Bytes>) -> Self {
        Self::Bytes {
            data: data.into(),
            content_type: None,
        }
    }

    pub fn bytes_with_content_type(data: impl Into<Bytes>, content_type: HeaderValue) -> Self {
        Self::Bytes {
            data: data.into(),
            content_type: Some(content_type),
        }
    }

    pub fn json(value: &impl Serialize) -> Result<Self, RequestBuildError> {
        let data = serde_json::to_vec(value).map_err(|_| RequestBuildError::JsonSerialization)?;
        Ok(Self::Bytes {
            data: Bytes::from(data),
            content_type: Some(HeaderValue::from_static("application/json")),
        })
    }

    pub fn multipart(body: MultipartBody) -> Self {
        Self::Multipart(body)
    }

    pub(crate) fn encode(
        &self,
        limits: &TransportLimits,
    ) -> Result<EncodedBody, RequestBuildError> {
        match self {
            Self::Empty => Ok(EncodedBody {
                bytes: Bytes::new(),
                content_type: None,
            }),
            Self::Bytes { data, content_type } => {
                if data.len() > limits.max_request_bytes {
                    return Err(RequestBuildError::BodyTooLarge {
                        maximum: limits.max_request_bytes,
                    });
                }
                Ok(EncodedBody {
                    bytes: data.clone(),
                    content_type: content_type.clone(),
                })
            }
            Self::Multipart(body) => body.encode(limits),
        }
    }
}

impl fmt::Debug for RequestBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("RequestBody::Empty"),
            Self::Bytes { data, content_type } => formatter
                .debug_struct("RequestBody::Bytes")
                .field("bytes", &data.len())
                .field("content_type_present", &content_type.is_some())
                .finish(),
            Self::Multipart(body) => body.fmt(formatter),
        }
    }
}

pub(crate) struct EncodedBody {
    pub(crate) bytes: Bytes,
    pub(crate) content_type: Option<HeaderValue>,
}

/// Complete immutable plan for one provider operation.
#[derive(Clone)]
pub struct RequestPlan {
    method: Method,
    target: RequestTarget,
    headers: RequestHeaders,
    body: RequestBody,
    replay_safety: ReplaySafety,
}

impl RequestPlan {
    pub fn new(method: Method, target: RequestTarget) -> Self {
        Self {
            method,
            target,
            headers: RequestHeaders::new(),
            body: RequestBody::Empty,
            replay_safety: ReplaySafety::Never,
        }
    }

    pub fn method(&self) -> &Method {
        &self.method
    }

    pub fn target(&self) -> &RequestTarget {
        &self.target
    }

    pub fn headers(&self) -> &RequestHeaders {
        &self.headers
    }

    pub fn body(&self) -> &RequestBody {
        &self.body
    }

    pub fn replay_safety(&self) -> &ReplaySafety {
        &self.replay_safety
    }

    pub fn with_headers(mut self, headers: RequestHeaders) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_body(mut self, body: RequestBody) -> Self {
        self.body = body;
        self
    }

    pub fn with_replay_safety(
        mut self,
        replay_safety: ReplaySafety,
    ) -> Result<Self, RequestBuildError> {
        if let ReplaySafety::IdempotencyKey(header) = &replay_safety
            && self.headers.get(header.name()).is_some()
        {
            return Err(RequestBuildError::IdempotencyHeaderConflict);
        }
        self.replay_safety = replay_safety;
        Ok(self)
    }

    pub(crate) fn prepare(
        &self,
        limits: &TransportLimits,
    ) -> Result<PreparedRequest, RequestBuildError> {
        if let ReplaySafety::IdempotencyKey(header) = &self.replay_safety
            && self.headers.get(header.name()).is_some()
        {
            return Err(RequestBuildError::IdempotencyHeaderConflict);
        }
        self.headers.validate(limits)?;
        let body = self.body.encode(limits)?;
        let mut headers = self.headers.clone_inner();
        if let Some(content_type) = &body.content_type
            && !headers.contains_key(CONTENT_TYPE)
        {
            headers.insert(CONTENT_TYPE, content_type.clone());
        }
        Ok(PreparedRequest {
            headers,
            body: body.bytes,
        })
    }
}

impl fmt::Debug for RequestPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestPlan")
            .field("method", &self.method)
            .field("target", &self.target)
            .field("headers", &self.headers)
            .field("body", &self.body)
            .field("replay_safety", &self.replay_safety)
            .finish()
    }
}

pub(crate) struct PreparedRequest {
    pub(crate) headers: HeaderMap,
    pub(crate) body: Bytes,
}

fn checked_disposition_value(value: String) -> Result<String, RequestBuildError> {
    if value.is_empty()
        || value.len() > 1024
        || value
            .chars()
            .any(|character| character.is_control() || matches!(character, '"' | '\\'))
    {
        Err(RequestBuildError::InvalidMultipartMetadata)
    } else {
        Ok(value)
    }
}

fn is_unsafe_path_segment(segment: &str) -> bool {
    let mut decoded = segment.as_bytes().to_vec();
    loop {
        if matches!(decoded.as_slice(), b"." | b"..") {
            return true;
        }
        let Some(next) = percent_decode_once(&decoded) else {
            return false;
        };
        if next.contains(&b'/') || next.contains(&b'\\') {
            return true;
        }
        decoded = next;
    }
}

fn is_unreserved_path_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

fn push_encoded_path_segment(output: &mut String, segment: &[u8]) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";

    for byte in segment {
        if is_unreserved_path_byte(*byte) {
            output.push(char::from(*byte));
        } else {
            output.push('%');
            output.push(char::from(HEX[usize::from(*byte >> 4)]));
            output.push(char::from(HEX[usize::from(*byte & 0x0f)]));
        }
    }
}

fn percent_decode_once(input: &[u8]) -> Option<Vec<u8>> {
    let mut decoded = Vec::with_capacity(input.len());
    let mut changed = false;
    let mut index = 0;
    while index < input.len() {
        if input[index] == b'%'
            && let Some(encoded) = input.get(index + 1..index + 3)
            && let (Some(high), Some(low)) = (hex_value(encoded[0]), hex_value(encoded[1]))
        {
            decoded.push((high << 4) | low);
            changed = true;
            index += 3;
        } else {
            decoded.push(input[index]);
            index += 1;
        }
    }
    changed.then_some(decoded)
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn append_bounded(
    output: &mut BytesMut,
    data: &[u8],
    maximum: usize,
) -> Result<(), RequestBuildError> {
    if output.len().saturating_add(data.len()) > maximum {
        return Err(RequestBuildError::BodyTooLarge { maximum });
    }
    output.put_slice(data);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_and_fragment_targets_are_rejected() {
        for target in [
            "https://attacker.invalid/v1",
            "//attacker.invalid/v1",
            "responses#fragment",
        ] {
            assert_eq!(
                RequestTarget::new(target).unwrap_err(),
                RequestBuildError::InvalidTarget
            );
        }
    }

    #[test]
    fn encoded_path_separators_and_dot_segments_are_rejected_at_any_layer() {
        for target in [
            "v1/%2e%2e/secrets",
            "v1/%252e%252e/secrets",
            "v1/.%252e/secrets",
            "v1/models%2fadmin",
            "v1/models%252Fadmin",
            "v1/models%5cadmin",
            "v1/models%255Cadmin",
        ] {
            assert_eq!(
                RequestTarget::new(target).unwrap_err(),
                RequestBuildError::InvalidTarget,
                "target should be rejected: {target}"
            );
        }

        assert!(RequestTarget::new("responses?cursor=a%2Fb%5Cc").is_ok());
    }

    #[test]
    fn opaque_path_segments_are_encoded_once_without_changing_the_query() {
        let target = RequestTarget::new("responses?include=reasoning.encrypted_content")
            .unwrap()
            .with_opaque_path_segment("resp/\\?#%2F%252F资源")
            .unwrap()
            .with_opaque_path_segment("input_items")
            .unwrap();

        assert_eq!(
            target.as_str(),
            "responses/resp%2F%5C%3F%23%252F%25252F%E8%B5%84%E6%BA%90/input_items?include=reasoning.encrypted_content"
        );
        assert!(!format!("{target:?}").contains("resp/"));
    }

    #[test]
    fn opaque_path_segments_reject_ambiguous_or_unbounded_inputs() {
        for segment in ["", ".", "..", "control\nvalue", "control\u{7f}value"] {
            assert_eq!(
                RequestTarget::new("responses")
                    .unwrap()
                    .with_opaque_path_segment(segment)
                    .unwrap_err(),
                RequestBuildError::InvalidTarget
            );
        }

        assert_eq!(
            RequestTarget::new("responses")
                .unwrap()
                .with_opaque_path_segment("x".repeat(MAX_TARGET_BYTES))
                .unwrap_err(),
            RequestBuildError::TargetTooLong {
                maximum: MAX_TARGET_BYTES,
            }
        );

        assert_eq!(
            RequestTarget::new("responses/%2F").unwrap_err(),
            RequestBuildError::InvalidTarget
        );
    }

    #[test]
    fn request_headers_protect_only_exact_common_credentials() {
        for name in ["Authorization", "API-Key", "X-API-Key"] {
            assert_eq!(
                RequestHeaders::new()
                    .try_insert(
                        HeaderName::from_bytes(name.as_bytes()).unwrap(),
                        HeaderValue::from_static("credential")
                    )
                    .unwrap_err(),
                RequestBuildError::ProtectedHeader
            );
        }

        RequestHeaders::new()
            .try_insert(
                HeaderName::from_static("x-token-count-mode"),
                HeaderValue::from_static("enabled"),
            )
            .unwrap();
    }

    #[test]
    fn multipart_encoding_is_identical_across_attempts() {
        let body = RequestBody::multipart(MultipartBody::new(vec![
            MultipartPart::field("purpose", "assistants").unwrap(),
            MultipartPart::file(
                "file",
                "notes.txt",
                HeaderValue::from_static("text/plain"),
                "hello",
            )
            .unwrap(),
        ]));
        let limits = TransportLimits::default();
        let first = body.encode(&limits).unwrap();
        let second = body.encode(&limits).unwrap();
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.content_type, second.content_type);
    }

    #[test]
    fn request_debug_redacts_target_header_and_body_values() {
        let plan = RequestPlan::new(
            Method::GET,
            RequestTarget::new("canary-path?signed=canary-query").unwrap(),
        )
        .with_headers(
            RequestHeaders::new()
                .try_insert(
                    HeaderName::from_static("x-canary-header-name"),
                    HeaderValue::from_static("canary-header"),
                )
                .unwrap(),
        )
        .with_body(RequestBody::bytes_with_content_type(
            "canary-body",
            HeaderValue::from_static("application/canary-content-type"),
        ));
        let debug = format!("{plan:?}");
        assert!(!debug.contains("canary"));
    }

    #[test]
    fn multipart_debug_redacts_metadata_header_value_and_payload() {
        let part = MultipartPart::file(
            "canary-name",
            "canary-file-name",
            HeaderValue::from_static("application/canary-content-type"),
            "canary-data",
        )
        .unwrap();

        assert!(!format!("{part:?}").contains("canary"));
    }

    #[test]
    fn prepare_rejects_idempotency_header_conflicts_for_any_builder_order() {
        let name = HeaderName::from_static("idempotency-key");
        let replay_safety =
            ReplaySafety::IdempotencyKey(crate::IdempotencyHeader::new(name.clone()).unwrap());
        let plan = RequestPlan::new(Method::POST, RequestTarget::new("responses").unwrap())
            .with_replay_safety(replay_safety)
            .unwrap()
            .with_headers(
                RequestHeaders::new()
                    .try_insert(name, HeaderValue::from_static("canary-key"))
                    .unwrap(),
            );

        assert!(matches!(
            plan.prepare(&TransportLimits::default()),
            Err(RequestBuildError::IdempotencyHeaderConflict)
        ));
    }
}
