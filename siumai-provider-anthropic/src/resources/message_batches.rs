use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::io::{self, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::Stream;
use http::Method;
use http::header::HeaderValue;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Number, Value};
use siumai_anthropic_compatible::{MessagesRequestPolicy, MessagesRequestRequirements};
use siumai_core::{CallOptions, Error, ErrorKind, LanguageRequest, ModelId};
use siumai_protocol_anthropic::messages::encode_request_for_scope_with_resolver;
use siumai_transport::{
    ReplaySafety, RequestBody, RequestPlan, RequestTarget, TransportByteStream, TransportLimits,
};

use crate::{AnthropicMessagesOptions, request_policy::AnthropicRequestPolicy};

use super::NativeRuntime;
use super::common::{collect_stream_body, execute_json, headers, resource_status_error, target};

const MAX_BATCH_REQUESTS: usize = 100_000;
const MAX_BATCH_REQUEST_BYTES: usize = 256 * 1024 * 1024;
const MAX_BATCH_STATUS_BYTES: usize = 128;
const MAX_BATCH_CUSTOM_ID_BYTES: usize = 256;
const MAX_BATCH_RESOURCE_ID_BYTES: usize = 256;
const MAX_BATCH_RESULT_LINE_BYTES: usize = 16 * 1024 * 1024;
const MAX_BATCH_RESULT_DECODED_LINE_BYTES: usize = 32 * 1024 * 1024;
const MAX_BATCH_RESULT_AGGREGATE_BYTES: usize = 1024 * 1024 * 1024;
const MAX_BATCH_RESULT_STRING_BYTES: usize = 8 * 1024 * 1024;
const MAX_BATCH_RESULT_JSON_DEPTH: usize = 64;
const MAX_BATCH_RESULT_JSON_NODES: usize = 1_000_000;

/// One language request submitted under a caller-stable batch identifier.
#[derive(Clone)]
pub struct AnthropicBatchItem {
    custom_id: String,
    model: ModelId,
    request: LanguageRequest,
    options: AnthropicMessagesOptions,
}

impl AnthropicBatchItem {
    pub fn new(
        custom_id: impl Into<String>,
        model: impl Into<String>,
        request: LanguageRequest,
    ) -> Result<Self, Error> {
        let custom_id = custom_id.into();
        validate_custom_id(&custom_id)?;
        let model = ModelId::new(model.into()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic model identifier is invalid",
            )
            .with_source(source)
        })?;
        Ok(Self {
            custom_id,
            model,
            request,
            options: AnthropicMessagesOptions::default(),
        })
    }

    pub fn with_options(mut self, options: AnthropicMessagesOptions) -> Self {
        self.options = options;
        self
    }

    pub fn custom_id(&self) -> &str {
        &self.custom_id
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn request(&self) -> &LanguageRequest {
        &self.request
    }

    pub fn options(&self) -> &AnthropicMessagesOptions {
        &self.options
    }

    fn validate_for_batch(&self) -> Result<(), Error> {
        if self.options.speed().is_some()
            || self.options.fallbacks().is_some_and(|fallbacks| {
                matches!(fallbacks, siumai_protocol_anthropic::messages::ServerFallbacks::Explicit(items) if items.iter().any(|fallback| fallback.speed().is_some()))
            })
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic Message Batches do not support speed or Fast mode",
            ));
        }
        if self.request.generation.max_output_tokens == Some(0) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic Message Batches do not support max_tokens set to zero",
            ));
        }
        for field in [
            "store",
            "previous_thread_event_id",
            "cache_hint",
            "context_hint",
        ] {
            if self.options.extra().contains_key(field) {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic Message Batches contain an unsupported top-level Messages field",
                ));
            }
        }
        if self
            .options
            .extra()
            .get("research_preview_2026_02")
            .and_then(Value::as_str)
            == Some("active")
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic Message Batches do not support active research preview mode",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for AnthropicBatchItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchItem")
            .field("custom_id_bytes", &self.custom_id.len())
            .field("model_bytes", &self.model.as_str().len())
            .field("request", &"<redacted>")
            .field("options", &"configured")
            .finish()
    }
}

#[derive(Clone, Default)]
pub struct AnthropicBatchRequest {
    requests: Vec<AnthropicBatchItem>,
}

impl AnthropicBatchRequest {
    pub fn new(requests: Vec<AnthropicBatchItem>) -> Result<Self, Error> {
        if requests.is_empty() || requests.len() > MAX_BATCH_REQUESTS {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic batch request count is outside the supported bounds",
            ));
        }
        let mut unique = HashSet::with_capacity(requests.len());
        if requests
            .iter()
            .any(|request| !unique.insert(request.custom_id.as_str()))
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic batch custom identifiers must be unique",
            ));
        }
        Ok(Self { requests })
    }

    pub fn requests(&self) -> &[AnthropicBatchItem] {
        &self.requests
    }
}

impl fmt::Debug for AnthropicBatchRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchRequest")
            .field("request_count", &self.requests.len())
            .finish()
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct AnthropicBatchListQuery {
    pub before_id: Option<String>,
    pub after_id: Option<String>,
    pub limit: Option<u16>,
}

impl fmt::Debug for AnthropicBatchListQuery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchListQuery")
            .field("before_id_present", &self.before_id.is_some())
            .field("after_id_present", &self.after_id.is_some())
            .field("limit", &self.limit)
            .finish()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessageBatch {
    pub id: String,
    #[serde(rename = "type", default)]
    pub object_type: Option<String>,
    #[serde(default)]
    pub processing_status: Option<AnthropicBatchProcessingStatus>,
    #[serde(default)]
    pub request_counts: Option<AnthropicBatchRequestCounts>,
    #[serde(default)]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub cancel_initiated_at: Option<String>,
    #[serde(default)]
    pub results_url: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for AnthropicMessageBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicMessageBatch")
            .field("id_bytes", &self.id.len())
            .field("object_type_present", &self.object_type.is_some())
            .field("processing_status", &self.processing_status)
            .field("request_counts", &self.request_counts)
            .field("ended_at_present", &self.ended_at.is_some())
            .field("created_at_present", &self.created_at.is_some())
            .field("expires_at_present", &self.expires_at.is_some())
            .field(
                "cancel_initiated_at_present",
                &self.cancel_initiated_at.is_some(),
            )
            .field("results_url_present", &self.results_url.is_some())
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicBatchRequestCounts {
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicBatchList {
    #[serde(default)]
    pub data: Vec<AnthropicMessageBatch>,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    #[serde(default)]
    pub has_more: bool,
}

impl fmt::Debug for AnthropicBatchList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchList")
            .field("item_count", &self.data.len())
            .field("first_id_present", &self.first_id.is_some())
            .field("last_id_present", &self.last_id.is_some())
            .field("has_more", &self.has_more)
            .finish()
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicBatchDeleteResult {
    #[serde(deserialize_with = "deserialize_batch_resource_id")]
    pub id: String,
    #[serde(rename = "type", deserialize_with = "deserialize_batch_delete_type")]
    pub object_type: String,
}

impl AnthropicBatchDeleteResult {
    pub fn is_deleted(&self) -> bool {
        self.object_type == "message_batch_deleted"
    }
}

impl fmt::Debug for AnthropicBatchDeleteResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchDeleteResult")
            .field("id_bytes", &self.id.len())
            .field("deleted", &self.is_deleted())
            .field("object_type_bytes", &self.object_type.len())
            .finish()
    }
}

/// An open, bounded Message Batch processing status.
///
/// Anthropic may add processing states without a client release. Known values
/// are exposed as predicates while the original bounded value remains
/// inspectable through [`Self::as_str`].
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AnthropicBatchProcessingStatus(String);

impl AnthropicBatchProcessingStatus {
    pub const IN_PROGRESS: &'static str = "in_progress";
    pub const CANCELING: &'static str = "canceling";
    pub const ENDED: &'static str = "ended";

    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_open_batch_value(&value, "processing status")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_in_progress(&self) -> bool {
        self.as_str() == Self::IN_PROGRESS
    }

    pub fn is_canceling(&self) -> bool {
        self.as_str() == Self::CANCELING
    }

    pub fn is_ended(&self) -> bool {
        self.as_str() == Self::ENDED
    }
}

impl fmt::Debug for AnthropicBatchProcessingStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchProcessingStatus")
            .field(
                "known",
                &match self.as_str() {
                    Self::IN_PROGRESS | Self::CANCELING | Self::ENDED => Some(self.as_str()),
                    _ => None,
                },
            )
            .field("value_bytes", &self.0.len())
            .finish()
    }
}

impl Serialize for AnthropicBatchProcessingStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AnthropicBatchProcessingStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(|_| {
            <D::Error as serde::de::Error>::custom("invalid bounded Anthropic batch status")
        })
    }
}

/// An open, bounded result status from one unordered Message Batch record.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AnthropicBatchResultStatus(String);

impl AnthropicBatchResultStatus {
    pub const SUCCEEDED: &'static str = "succeeded";
    pub const ERRORED: &'static str = "errored";
    pub const CANCELED: &'static str = "canceled";
    pub const EXPIRED: &'static str = "expired";

    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_open_batch_value(&value, "result status")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_succeeded(&self) -> bool {
        self.as_str() == Self::SUCCEEDED
    }

    pub fn is_errored(&self) -> bool {
        self.as_str() == Self::ERRORED
    }

    pub fn is_canceled(&self) -> bool {
        self.as_str() == Self::CANCELED
    }

    pub fn is_expired(&self) -> bool {
        self.as_str() == Self::EXPIRED
    }
}

impl fmt::Debug for AnthropicBatchResultStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchResultStatus")
            .field(
                "known",
                &match self.as_str() {
                    Self::SUCCEEDED | Self::ERRORED | Self::CANCELED | Self::EXPIRED => {
                        Some(self.as_str())
                    }
                    _ => None,
                },
            )
            .field("value_bytes", &self.0.len())
            .finish()
    }
}

impl Serialize for AnthropicBatchResultStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AnthropicBatchResultStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(|_| {
            <D::Error as serde::de::Error>::custom("invalid bounded Anthropic batch result status")
        })
    }
}

/// The provider-owned result envelope for one unordered batch item.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicBatchResult {
    #[serde(deserialize_with = "deserialize_batch_custom_id")]
    pub custom_id: String,
    pub result: AnthropicBatchResultBody,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Open result body preserving unknown provider fields and statuses.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicBatchResultBody {
    #[serde(rename = "type")]
    pub status: AnthropicBatchResultStatus,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl AnthropicBatchResult {
    pub fn status(&self) -> &AnthropicBatchResultStatus {
        &self.result.status
    }
}

impl AnthropicBatchResultBody {
    pub fn status(&self) -> &AnthropicBatchResultStatus {
        &self.status
    }
}

impl fmt::Debug for AnthropicBatchResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchResult")
            .field("custom_id", &"[REDACTED]")
            .field("status", self.status())
            .field("extra_field_count", &self.extra.len())
            .field("result_field_count", &self.result.extra.len())
            .finish()
    }
}

impl fmt::Debug for AnthropicBatchResultBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchResultBody")
            .field("status", &self.status)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// One sanitized failure from the bounded Message Batch JSONL decoder.
///
/// The raw-line diagnostic excerpt budget is deliberately zero; variants carry
/// only bounded line numbers and configured limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AnthropicBatchResultsDecodeError {
    #[error("Anthropic batch results exceed the encoded aggregate limit of {maximum} bytes")]
    EncodedAggregateTooLarge { maximum: usize },
    #[error("Anthropic batch result line {line} exceeds the {maximum}-byte limit")]
    LineTooLarge { line: usize, maximum: usize },
    #[error("Anthropic batch result line {line} contains invalid UTF-8")]
    InvalidUtf8 { line: usize },
    #[error("Anthropic batch result line {line} is not valid JSON")]
    InvalidJson { line: usize },
    #[error("Anthropic batch results ended with a truncated line {line}")]
    TruncatedFinalLine { line: usize },
    #[error("Anthropic batch results exceed the record limit of {maximum}")]
    TooManyRecords { maximum: usize },
    #[error("Anthropic batch result line {line} exceeds JSON depth {maximum}")]
    JsonDepthExceeded { line: usize, maximum: usize },
    #[error("Anthropic batch result line {line} exceeds {maximum} JSON nodes")]
    JsonNodeLimitExceeded { line: usize, maximum: usize },
    #[error("Anthropic batch result line {line} contains a string longer than {maximum} bytes")]
    StringTooLarge { line: usize, maximum: usize },
    #[error("Anthropic batch result line {line} exceeds the decoded limit of {maximum} bytes")]
    DecodedLineTooLarge { line: usize, maximum: usize },
    #[error("Anthropic batch results exceed the decoded aggregate limit of {maximum} bytes")]
    DecodedAggregateTooLarge { maximum: usize },
    #[error("Anthropic batch result line {line} does not match the result envelope")]
    InvalidRecord { line: usize },
}

/// One terminal item error from the established batch-results stream.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AnthropicBatchResultsStreamError {
    #[error(transparent)]
    Decode(#[from] AnthropicBatchResultsDecodeError),
    #[error("Anthropic batch results transport failed")]
    Transport(#[source] Box<Error>),
}

#[derive(Debug, Clone, Copy)]
struct BatchResultsDecoderLimits {
    max_encoded_bytes: usize,
    max_line_bytes: usize,
    max_decoded_line_bytes: usize,
    max_decoded_bytes: usize,
    max_records: usize,
    max_depth: usize,
    max_nodes: usize,
    max_string_bytes: usize,
}

impl BatchResultsDecoderLimits {
    fn from_transport(limits: &TransportLimits) -> Self {
        Self {
            max_encoded_bytes: limits
                .max_response_bytes
                .min(MAX_BATCH_RESULT_AGGREGATE_BYTES),
            max_line_bytes: limits.max_frame_bytes.min(MAX_BATCH_RESULT_LINE_BYTES),
            max_decoded_line_bytes: limits
                .max_event_bytes
                .min(MAX_BATCH_RESULT_DECODED_LINE_BYTES),
            max_decoded_bytes: limits
                .max_response_bytes
                .min(MAX_BATCH_RESULT_AGGREGATE_BYTES),
            max_records: limits.max_events_per_stream.min(MAX_BATCH_REQUESTS),
            max_depth: MAX_BATCH_RESULT_JSON_DEPTH,
            max_nodes: MAX_BATCH_RESULT_JSON_NODES,
            max_string_bytes: limits.max_event_bytes.min(MAX_BATCH_RESULT_STRING_BYTES),
        }
    }
}

/// Stateful, incremental decoder for unordered Anthropic Message Batch JSONL.
///
/// The decoder never retains a completed raw line and never includes provider
/// payload bytes in its errors or `Debug` output. A malformed line terminates
/// the decoder: records completed earlier in the same chunk are returned first,
/// followed by exactly one typed error, and all later calls return no items.
/// Decoded-byte accounting includes each `serde_json::Value` node plus UTF-8
/// object-key, string, and scalar text bytes, providing a deterministic upper
/// bound independent of allocator-specific overhead.
pub struct AnthropicBatchResultsDecoder {
    limits: BatchResultsDecoderLimits,
    line: Vec<u8>,
    next_line: usize,
    encoded_bytes: usize,
    decoded_bytes: usize,
    records: usize,
    terminated: bool,
}

impl AnthropicBatchResultsDecoder {
    pub fn new(transport_limits: &TransportLimits) -> Self {
        Self::with_limits(BatchResultsDecoderLimits::from_transport(transport_limits))
    }

    fn with_limits(limits: BatchResultsDecoderLimits) -> Self {
        Self {
            limits,
            line: Vec::new(),
            next_line: 1,
            encoded_bytes: 0,
            decoded_bytes: 0,
            records: 0,
            terminated: false,
        }
    }

    pub fn push(
        &mut self,
        chunk: &[u8],
    ) -> Vec<Result<AnthropicBatchResult, AnthropicBatchResultsDecodeError>> {
        let mut output = Vec::new();
        let mut consumed = 0_usize;
        while consumed < chunk.len() && !self.terminated {
            let (used, item) = self.push_one(&chunk[consumed..]);
            consumed = consumed.saturating_add(used);
            if let Some(item) = item {
                let failed = item.is_err();
                output.push(item);
                if failed {
                    break;
                }
            }
            if used == 0 {
                break;
            }
        }

        output
    }

    pub fn finish(
        &mut self,
    ) -> Vec<Result<AnthropicBatchResult, AnthropicBatchResultsDecodeError>> {
        if self.terminated {
            return Vec::new();
        }
        self.terminated = true;
        if self.line.is_empty() {
            self.line = Vec::new();
            return Vec::new();
        }
        self.line = Vec::new();
        vec![Err(AnthropicBatchResultsDecodeError::TruncatedFinalLine {
            line: self.next_line,
        })]
    }

    pub fn records_decoded(&self) -> usize {
        self.records
    }

    pub fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }

    pub fn decoded_bytes(&self) -> usize {
        self.decoded_bytes
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    fn decode_line(
        &mut self,
        line: &mut Vec<u8>,
        line_number: usize,
    ) -> Result<AnthropicBatchResult, AnthropicBatchResultsDecodeError> {
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        let line = std::str::from_utf8(line)
            .map_err(|_| AnthropicBatchResultsDecodeError::InvalidUtf8 { line: line_number })?;
        if line.trim().is_empty() {
            return Err(AnthropicBatchResultsDecodeError::InvalidJson { line: line_number });
        }
        let (value, decoded_line_bytes) =
            deserialize_bounded_json_value(line, line_number, self.limits)?;
        let decoded_bytes = self.decoded_bytes.checked_add(decoded_line_bytes).ok_or(
            AnthropicBatchResultsDecodeError::DecodedAggregateTooLarge {
                maximum: self.limits.max_decoded_bytes,
            },
        )?;
        if decoded_bytes > self.limits.max_decoded_bytes {
            return Err(AnthropicBatchResultsDecodeError::DecodedAggregateTooLarge {
                maximum: self.limits.max_decoded_bytes,
            });
        }
        if self.records >= self.limits.max_records {
            return Err(AnthropicBatchResultsDecodeError::TooManyRecords {
                maximum: self.limits.max_records,
            });
        }
        let record = serde_json::from_value(value)
            .map_err(|_| AnthropicBatchResultsDecodeError::InvalidRecord { line: line_number })?;
        self.decoded_bytes = decoded_bytes;
        self.records += 1;
        Ok(record)
    }

    fn push_one(
        &mut self,
        chunk: &[u8],
    ) -> (
        usize,
        Option<Result<AnthropicBatchResult, AnthropicBatchResultsDecodeError>>,
    ) {
        if self.terminated {
            return (0, None);
        }

        for (index, byte) in chunk.iter().copied().enumerate() {
            if self.encoded_bytes >= self.limits.max_encoded_bytes {
                let error = AnthropicBatchResultsDecodeError::EncodedAggregateTooLarge {
                    maximum: self.limits.max_encoded_bytes,
                };
                self.abort();
                return (index, Some(Err(error)));
            }
            self.encoded_bytes += 1;

            if byte == b'\n' {
                let line_number = self.next_line;
                self.next_line = self.next_line.saturating_add(1);
                let mut line = std::mem::take(&mut self.line);
                let decoded = self.decode_line(&mut line, line_number);
                return match decoded {
                    Ok(record) => {
                        line.clear();
                        self.line = line;
                        (index + 1, Some(Ok(record)))
                    }
                    Err(error) => {
                        self.abort();
                        (index + 1, Some(Err(error)))
                    }
                };
            }
            if self.line.len() >= self.limits.max_line_bytes {
                let error = AnthropicBatchResultsDecodeError::LineTooLarge {
                    line: self.next_line,
                    maximum: self.limits.max_line_bytes,
                };
                self.abort();
                return (index + 1, Some(Err(error)));
            }
            self.line.push(byte);
        }

        (chunk.len(), None)
    }

    fn abort(&mut self) {
        self.line = Vec::new();
        self.terminated = true;
    }
}

impl fmt::Debug for AnthropicBatchResultsDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchResultsDecoder")
            .field("buffered_line_bytes", &self.line.len())
            .field("next_line", &self.next_line)
            .field("encoded_bytes", &self.encoded_bytes)
            .field("decoded_bytes", &self.decoded_bytes)
            .field("records", &self.records)
            .field("terminated", &self.terminated)
            .finish()
    }
}

struct BatchResultsStream<S> {
    body: Option<Pin<Box<S>>>,
    decoder: AnthropicBatchResultsDecoder,
    chunk: Option<Bytes>,
    chunk_offset: usize,
}

impl<S> BatchResultsStream<S> {
    fn new(body: S, decoder: AnthropicBatchResultsDecoder) -> Self {
        Self {
            body: Some(Box::pin(body)),
            decoder,
            chunk: None,
            chunk_offset: 0,
        }
    }

    fn decoder(&self) -> &AnthropicBatchResultsDecoder {
        &self.decoder
    }

    fn terminate(&mut self) {
        self.body = None;
        self.chunk = None;
        self.chunk_offset = 0;
    }
}

impl<S> Stream for BatchResultsStream<S>
where
    S: Stream<Item = Result<Bytes, Error>>,
{
    type Item = Result<AnthropicBatchResult, AnthropicBatchResultsStreamError>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.body.is_none() {
            return Poll::Ready(None);
        }

        loop {
            if let Some(chunk) = this.chunk.as_ref() {
                let (consumed, item) = this.decoder.push_one(&chunk[this.chunk_offset..]);
                this.chunk_offset = this.chunk_offset.saturating_add(consumed);
                if this.chunk_offset >= chunk.len() {
                    this.chunk = None;
                    this.chunk_offset = 0;
                }
                if let Some(item) = item {
                    let item = item.map_err(AnthropicBatchResultsStreamError::from);
                    if item.is_err() {
                        this.terminate();
                    }
                    return Poll::Ready(Some(item));
                }
                if consumed == 0 && this.chunk.is_some() {
                    this.terminate();
                    return Poll::Ready(None);
                }
                continue;
            }

            let Some(body) = this.body.as_mut() else {
                return Poll::Ready(None);
            };
            match body.as_mut().poll_next(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => {
                    if chunk.is_empty() {
                        continue;
                    }
                    this.chunk = Some(chunk);
                    this.chunk_offset = 0;
                }
                Poll::Ready(Some(Err(error))) => {
                    this.decoder.abort();
                    this.terminate();
                    return Poll::Ready(Some(Err(AnthropicBatchResultsStreamError::Transport(
                        Box::new(error),
                    ))));
                }
                Poll::Ready(None) => {
                    let item = this
                        .decoder
                        .finish()
                        .into_iter()
                        .next()
                        .map(|item| item.map_err(AnthropicBatchResultsStreamError::from));
                    this.terminate();
                    return Poll::Ready(item);
                }
            }
        }
    }
}

/// Established, bounded stream of unordered Anthropic Message Batch results.
pub struct AnthropicBatchResultsStream {
    inner: BatchResultsStream<TransportByteStream>,
}

impl AnthropicBatchResultsStream {
    fn new(body: TransportByteStream, limits: &TransportLimits) -> Self {
        Self {
            inner: BatchResultsStream::new(body, AnthropicBatchResultsDecoder::new(limits)),
        }
    }

    pub fn decoder(&self) -> &AnthropicBatchResultsDecoder {
        self.inner.decoder()
    }
}

impl Stream for AnthropicBatchResultsStream {
    type Item = Result<AnthropicBatchResult, AnthropicBatchResultsStreamError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl fmt::Debug for AnthropicBatchResultsStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicBatchResultsStream")
            .field("decoder", self.inner.decoder())
            .field(
                "retained_chunk_bytes",
                &self
                    .inner
                    .chunk
                    .as_ref()
                    .map(|chunk| chunk.len().saturating_sub(self.inner.chunk_offset)),
            )
            .field("finished", &self.inner.body.is_none())
            .finish()
    }
}

/// Shared, lightweight Message Batches API handle.
#[derive(Clone)]
pub struct AnthropicMessageBatches {
    runtime: Arc<NativeRuntime>,
}

impl AnthropicMessageBatches {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn create(
        &self,
        request: AnthropicBatchRequest,
    ) -> Result<AnthropicMessageBatch, Error> {
        self.create_with_options(request, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        request: AnthropicBatchRequest,
        call_options: CallOptions,
    ) -> Result<AnthropicMessageBatch, Error> {
        let maximum_body_bytes =
            MAX_BATCH_REQUEST_BYTES.min(self.runtime.transport.limits().max_request_bytes);
        let mut body = BatchCreateBodyEncoder::new(maximum_body_bytes)?;
        let mut beta_features = BTreeSet::new();
        for item in request.requests {
            let item = self.encode_item(item, &mut beta_features)?;
            body.push(&item)?;
        }
        let body = body.finish()?;
        let beta_features = beta_features.iter().map(String::as_str).collect::<Vec<_>>();
        execute_json(
            &self.runtime,
            Method::POST,
            target("messages/batches")?,
            body,
            ReplaySafety::Never,
            &beta_features,
            call_options,
        )
        .await
    }

    pub async fn retrieve(&self, batch_id: &str) -> Result<AnthropicMessageBatch, Error> {
        self.retrieve_with_options(batch_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicMessageBatch, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            batch_target(batch_id, &[])?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[],
            options,
        )
        .await
    }

    pub async fn list(&self, query: AnthropicBatchListQuery) -> Result<AnthropicBatchList, Error> {
        self.list_with_options(query, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        query: AnthropicBatchListQuery,
        options: CallOptions,
    ) -> Result<AnthropicBatchList, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            target(list_target(&query)?)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[],
            options,
        )
        .await
    }

    pub async fn cancel(&self, batch_id: &str) -> Result<AnthropicMessageBatch, Error> {
        self.cancel_with_options(batch_id, CallOptions::default())
            .await
    }

    pub async fn cancel_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicMessageBatch, Error> {
        execute_json(
            &self.runtime,
            Method::POST,
            batch_target(batch_id, &["cancel"])?,
            RequestBody::json(&serde_json::json!({})).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic batch cancel body is invalid",
                )
                .with_source(source)
            })?,
            ReplaySafety::Never,
            &[],
            options,
        )
        .await
    }

    pub async fn delete(&self, batch_id: &str) -> Result<AnthropicBatchDeleteResult, Error> {
        self.delete_with_options(batch_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicBatchDeleteResult, Error> {
        execute_json(
            &self.runtime,
            Method::DELETE,
            batch_target(batch_id, &[])?,
            RequestBody::Empty,
            ReplaySafety::Never,
            &[],
            options,
        )
        .await
    }

    /// Establish a bounded incremental stream of unordered JSONL results.
    ///
    /// This intentionally returns decoded records instead of buffering the
    /// complete provider body. Correlate records with
    /// [`AnthropicBatchResult::custom_id`]; provider result order is not the
    /// request order. Import [`futures_util::StreamExt`] to iterate the stream.
    pub async fn results(&self, batch_id: &str) -> Result<AnthropicBatchResultsStream, Error> {
        self.results_with_options(batch_id, CallOptions::default())
            .await
    }

    /// Establish a bounded incremental result stream with call controls.
    pub async fn results_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicBatchResultsStream, Error> {
        let plan = RequestPlan::new(Method::GET, batch_target(batch_id, &["results"])?)
            .with_headers(headers(&self.runtime, &[], "application/x-jsonlines")?)
            .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
            .map_err(|source| {
                Error::new(
                    ErrorKind::Configuration,
                    "Anthropic batch results request is not replay-safe",
                )
                .with_source(source)
            })?;
        let response = self.runtime.transport.execute_stream(plan, options).await?;
        if !response.status().is_success() {
            let (status, response_headers, body) = response.into_parts();
            let body = collect_stream_body(body).await?;
            return Err(resource_status_error(status, response_headers, body));
        }
        Ok(AnthropicBatchResultsStream::new(
            response.into_body(),
            self.runtime.transport.limits(),
        ))
    }

    fn encode_item(
        &self,
        item: AnthropicBatchItem,
        beta_features: &mut BTreeSet<String>,
    ) -> Result<BatchItemWire, Error> {
        item.validate_for_batch()?;
        extend_item_beta_features(&item, beta_features)?;
        let protocol_options = item.options.to_protocol(false);
        let mut params = encode_request_for_scope_with_resolver(
            &self.runtime.scope,
            &item.model,
            &item.request,
            &protocol_options,
            self.runtime.annotation_resolver.as_ref(),
        )
        .map_err(Error::from)?;
        if let Some(params) = params.as_object_mut() {
            params.remove("stream");
        }
        Ok(BatchItemWire {
            custom_id: item.custom_id,
            params,
        })
    }
}

impl std::fmt::Debug for AnthropicMessageBatches {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicMessageBatches")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Serialize)]
struct BatchItemWire {
    custom_id: String,
    params: Value,
}

fn derive_item_requirements(
    item: &AnthropicBatchItem,
) -> Result<MessagesRequestRequirements, Error> {
    let mut options = item.options.to_engine();
    AnthropicRequestPolicy.prepare(&item.model, &item.request, &mut options)
}

fn extend_item_beta_features(
    item: &AnthropicBatchItem,
    beta_features: &mut BTreeSet<String>,
) -> Result<(), Error> {
    let requirements = derive_item_requirements(item)?;
    beta_features.extend(requirements.beta_features().map(str::to_owned));
    Ok(())
}

struct BatchCreateBodyEncoder {
    output: BoundedJsonBuffer,
    has_item: bool,
}

impl BatchCreateBodyEncoder {
    fn new(maximum: usize) -> Result<Self, Error> {
        let mut encoder = Self {
            output: BoundedJsonBuffer::new(maximum),
            has_item: false,
        };
        encoder.write_raw(b"{\"requests\":[")?;
        Ok(encoder)
    }

    fn push(&mut self, item: &BatchItemWire) -> Result<(), Error> {
        if self.has_item {
            self.write_raw(b",")?;
        }
        let result = serde_json::to_writer(&mut self.output, item);
        self.check_limit()?;
        result.map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic message batch could not be encoded",
            )
            .with_source(source)
        })?;
        self.has_item = true;
        Ok(())
    }

    fn finish(mut self) -> Result<RequestBody, Error> {
        self.write_raw(b"]}")?;
        Ok(RequestBody::bytes_with_content_type(
            self.output.into_inner(),
            HeaderValue::from_static("application/json"),
        ))
    }

    fn write_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let result = self.output.write_all(bytes);
        self.check_limit()?;
        result.map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic message batch could not be encoded",
            )
            .with_source(source)
        })
    }

    fn check_limit(&self) -> Result<(), Error> {
        if self.output.exceeded_limit() {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "Anthropic message batch exceeds the Siumai encoded request safety bound",
            ));
        }
        Ok(())
    }
}

struct BoundedJsonBuffer {
    bytes: Vec<u8>,
    maximum: usize,
    exceeded: bool,
}

impl BoundedJsonBuffer {
    fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
            exceeded: false,
        }
    }

    fn exceeded_limit(&self) -> bool {
        self.exceeded
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedJsonBuffer {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let remaining = self.maximum.saturating_sub(self.bytes.len());
        if buffer.len() > remaining {
            self.exceeded = true;
            return Err(io::Error::other("bounded JSON output exceeded its limit"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn deserialize_bounded_json_value(
    input: &str,
    line: usize,
    limits: BatchResultsDecoderLimits,
) -> Result<(Value, usize), AnthropicBatchResultsDecodeError> {
    let mut budget = BoundedJsonBudget::new(line, limits);
    let mut deserializer = serde_json::Deserializer::from_str(input);
    let value = BoundedJsonValueSeed {
        budget: &mut budget,
        depth: 1,
    }
    .deserialize(&mut deserializer)
    .map_err(|_| {
        budget
            .failure
            .unwrap_or(AnthropicBatchResultsDecodeError::InvalidJson { line })
    })?;
    deserializer
        .end()
        .map_err(|_| AnthropicBatchResultsDecodeError::InvalidJson { line })?;
    Ok((value, budget.decoded_bytes))
}

struct BoundedJsonBudget {
    line: usize,
    limits: BatchResultsDecoderLimits,
    nodes: usize,
    decoded_bytes: usize,
    failure: Option<AnthropicBatchResultsDecodeError>,
}

impl BoundedJsonBudget {
    fn new(line: usize, limits: BatchResultsDecoderLimits) -> Self {
        Self {
            line,
            limits,
            nodes: 0,
            decoded_bytes: 0,
            failure: None,
        }
    }

    fn account_node(&mut self, depth: usize) -> Result<(), AnthropicBatchResultsDecodeError> {
        if depth > self.limits.max_depth {
            return Err(AnthropicBatchResultsDecodeError::JsonDepthExceeded {
                line: self.line,
                maximum: self.limits.max_depth,
            });
        }
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > self.limits.max_nodes {
            return Err(AnthropicBatchResultsDecodeError::JsonNodeLimitExceeded {
                line: self.line,
                maximum: self.limits.max_nodes,
            });
        }
        self.account_bytes(std::mem::size_of::<Value>())
    }

    fn account_bytes(&mut self, bytes: usize) -> Result<(), AnthropicBatchResultsDecodeError> {
        self.decoded_bytes = add_decoded_bytes(
            self.decoded_bytes,
            bytes,
            self.line,
            self.limits.max_decoded_line_bytes,
        )?;
        Ok(())
    }

    fn account_string(&mut self, value: &str) -> Result<(), AnthropicBatchResultsDecodeError> {
        validate_decoded_string(value, self.line, self.limits.max_string_bytes)?;
        self.account_bytes(value.len())
    }

    fn reject<E>(&mut self, error: AnthropicBatchResultsDecodeError) -> E
    where
        E: de::Error,
    {
        self.failure = Some(error);
        E::custom("Anthropic batch result exceeded a bounded JSON resource limit")
    }
}

struct BoundedJsonValueSeed<'a> {
    budget: &'a mut BoundedJsonBudget,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedJsonValueSeed<'_> {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        if let Err(error) = self.budget.account_node(self.depth) {
            return Err(self.budget.reject(error));
        }
        deserializer.deserialize_any(BoundedJsonValueVisitor {
            budget: self.budget,
            depth: self.depth,
        })
    }
}

struct BoundedJsonValueVisitor<'a> {
    budget: &'a mut BoundedJsonBudget,
    depth: usize,
}

impl<'de> Visitor<'de> for BoundedJsonValueVisitor<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Err(error) = self.budget.account_bytes(std::mem::size_of::<bool>()) {
            return Err(self.budget.reject(error));
        }
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = Number::from(value);
        if let Err(error) = self.budget.account_bytes(number.to_string().len()) {
            return Err(self.budget.reject(error));
        }
        Ok(Value::Number(number))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = Number::from(value);
        if let Err(error) = self.budget.account_bytes(number.to_string().len()) {
            return Err(self.budget.reject(error));
        }
        Ok(Value::Number(number))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = Number::from_f64(value).ok_or_else(|| E::custom("invalid JSON number"))?;
        if let Err(error) = self.budget.account_bytes(number.to_string().len()) {
            return Err(self.budget.reject(error));
        }
        Ok(Value::Number(number))
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Err(error) = self.budget.account_string(value) {
            return Err(self.budget.reject(error));
        }
        Ok(Value::String(value.to_owned()))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Err(error) = self.budget.account_string(value) {
            return Err(self.budget.reject(error));
        }
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Err(error) = self.budget.account_string(&value) {
            return Err(self.budget.reject(error));
        }
        Ok(Value::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(BoundedJsonValueSeed {
            budget: &mut *self.budget,
            depth: self.depth.saturating_add(1),
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key_seed(BoundedJsonStringSeed {
            budget: &mut *self.budget,
        })? {
            let value = object.next_value_seed(BoundedJsonValueSeed {
                budget: &mut *self.budget,
                depth: self.depth.saturating_add(1),
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

struct BoundedJsonStringSeed<'a> {
    budget: &'a mut BoundedJsonBudget,
}

impl<'de> DeserializeSeed<'de> for BoundedJsonStringSeed<'_> {
    type Value = String;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(BoundedJsonStringVisitor {
            budget: self.budget,
        })
    }
}

struct BoundedJsonStringVisitor<'a> {
    budget: &'a mut BoundedJsonBudget,
}

impl<'de> Visitor<'de> for BoundedJsonStringVisitor<'_> {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded JSON object key")
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Err(error) = self.budget.account_string(value) {
            return Err(self.budget.reject(error));
        }
        Ok(value.to_owned())
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Err(error) = self.budget.account_string(value) {
            return Err(self.budget.reject(error));
        }
        Ok(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Err(error) = self.budget.account_string(&value) {
            return Err(self.budget.reject(error));
        }
        Ok(value)
    }
}

#[cfg(test)]
fn measure_json_value(
    value: &Value,
    line: usize,
    limits: BatchResultsDecoderLimits,
) -> Result<usize, AnthropicBatchResultsDecodeError> {
    let mut stack = vec![(value, 1_usize)];
    let mut nodes = 0_usize;
    let mut decoded_bytes = 0_usize;

    while let Some((value, depth)) = stack.pop() {
        if depth > limits.max_depth {
            return Err(AnthropicBatchResultsDecodeError::JsonDepthExceeded {
                line,
                maximum: limits.max_depth,
            });
        }
        nodes = nodes.saturating_add(1);
        if nodes > limits.max_nodes {
            return Err(AnthropicBatchResultsDecodeError::JsonNodeLimitExceeded {
                line,
                maximum: limits.max_nodes,
            });
        }
        decoded_bytes = add_decoded_bytes(
            decoded_bytes,
            std::mem::size_of::<Value>(),
            line,
            limits.max_decoded_line_bytes,
        )?;

        match value {
            Value::Null => {}
            Value::Bool(_) => {
                decoded_bytes = add_decoded_bytes(
                    decoded_bytes,
                    std::mem::size_of::<bool>(),
                    line,
                    limits.max_decoded_line_bytes,
                )?;
            }
            Value::Number(number) => {
                decoded_bytes = add_decoded_bytes(
                    decoded_bytes,
                    number.to_string().len(),
                    line,
                    limits.max_decoded_line_bytes,
                )?;
            }
            Value::String(string) => {
                validate_decoded_string(string, line, limits.max_string_bytes)?;
                decoded_bytes = add_decoded_bytes(
                    decoded_bytes,
                    string.len(),
                    line,
                    limits.max_decoded_line_bytes,
                )?;
            }
            Value::Array(items) => {
                for item in items.iter().rev() {
                    stack.push((item, depth.saturating_add(1)));
                }
            }
            Value::Object(object) => {
                for (name, item) in object.iter().rev() {
                    validate_decoded_string(name, line, limits.max_string_bytes)?;
                    decoded_bytes = add_decoded_bytes(
                        decoded_bytes,
                        name.len(),
                        line,
                        limits.max_decoded_line_bytes,
                    )?;
                    stack.push((item, depth.saturating_add(1)));
                }
            }
        }
    }

    Ok(decoded_bytes)
}

fn add_decoded_bytes(
    current: usize,
    additional: usize,
    line: usize,
    maximum: usize,
) -> Result<usize, AnthropicBatchResultsDecodeError> {
    let total = current
        .checked_add(additional)
        .ok_or(AnthropicBatchResultsDecodeError::DecodedLineTooLarge { line, maximum })?;
    if total > maximum {
        return Err(AnthropicBatchResultsDecodeError::DecodedLineTooLarge { line, maximum });
    }
    Ok(total)
}

fn validate_decoded_string(
    value: &str,
    line: usize,
    maximum: usize,
) -> Result<(), AnthropicBatchResultsDecodeError> {
    if value.len() > maximum {
        return Err(AnthropicBatchResultsDecodeError::StringTooLarge { line, maximum });
    }
    Ok(())
}

fn validate_open_batch_value(value: &str, field: &'static str) -> Result<(), Error> {
    if value.is_empty()
        || value.len() > MAX_BATCH_STATUS_BYTES
        || value.chars().any(char::is_control)
    {
        let message = match field {
            "processing status" => "Anthropic batch processing status is invalid",
            _ => "Anthropic batch result status is invalid",
        };
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

fn deserialize_batch_custom_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    validate_custom_id(&value).map_err(|_| {
        <D::Error as serde::de::Error>::custom("invalid bounded Anthropic batch custom identifier")
    })?;
    Ok(value)
}

fn deserialize_batch_resource_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    validate_batch_resource_id(&value).map_err(|_| {
        <D::Error as serde::de::Error>::custom(
            "invalid bounded Anthropic batch resource identifier",
        )
    })?;
    Ok(value)
}

fn deserialize_batch_delete_type<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    validate_open_batch_value(&value, "delete response type").map_err(|_| {
        <D::Error as serde::de::Error>::custom(
            "invalid bounded Anthropic batch delete response type",
        )
    })?;
    Ok(value)
}

fn validate_custom_id(id: &str) -> Result<(), Error> {
    if id.is_empty() || id.len() > MAX_BATCH_CUSTOM_ID_BYTES || id.chars().any(char::is_control) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic batch custom identifier is invalid",
        ));
    }
    Ok(())
}

fn validate_batch_resource_id(id: &str) -> Result<(), Error> {
    if id.is_empty() || id.len() > MAX_BATCH_RESOURCE_ID_BYTES || id.chars().any(char::is_control) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic batch resource identifier is invalid",
        ));
    }
    Ok(())
}

fn batch_target(batch_id: &str, suffix: &[&str]) -> Result<RequestTarget, Error> {
    validate_batch_resource_id(batch_id)?;
    let mut resource_target = target("messages/batches")?
        .with_opaque_path_segment(batch_id)
        .map_err(batch_target_error)?;
    for segment in suffix {
        resource_target = resource_target
            .with_opaque_path_segment(segment)
            .map_err(batch_target_error)?;
    }
    Ok(resource_target)
}

fn batch_target_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Anthropic batch resource target is invalid",
    )
    .with_source(source)
}

fn list_target(query: &AnthropicBatchListQuery) -> Result<String, Error> {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    if let Some(before) = &query.before_id {
        validate_batch_resource_id(before)?;
        serializer.append_pair("before_id", before);
    }
    if let Some(after) = &query.after_id {
        validate_batch_resource_id(after)?;
        serializer.append_pair("after_id", after);
    }
    if let Some(limit) = query.limit {
        if limit == 0 || limit > 1_000 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic list limit must be between 1 and 1000",
            ));
        }
        serializer.append_pair("limit", &limit.to_string());
    }
    let query = serializer.finish();
    Ok(if query.is_empty() {
        "messages/batches".to_string()
    } else {
        format!("messages/batches?{query}")
    })
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, stream};
    use serde_json::json;
    use siumai_core::{Message, ProviderId, ProviderScope, ReplayDomain, ReplayDomainId};
    use siumai_protocol_anthropic::messages::{
        AnthropicTool, ClearToolUsesEdit, ContainerSkill, ContextManagement, InferenceSpeed,
        McpServer, McpToolsetOptions, MessagesContainer, ServerFallback, ServerFallbacks,
        TokenTaskBudget,
    };
    use siumai_transport::EndpointConfig;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::{
        AnthropicAnnotationResolver, AnthropicCacheTtl, AnthropicMessageCache, AnthropicToolOptions,
    };
    use crate::{AnthropicCredential, AnthropicProvider};

    fn language_request(max_tokens: u64) -> LanguageRequest {
        let mut request = LanguageRequest::new(vec![Message::user("batch fixture")]);
        request.generation.max_output_tokens = Some(max_tokens);
        request
    }

    fn batch_item(options: AnthropicMessagesOptions, max_tokens: u64) -> AnthropicBatchItem {
        AnthropicBatchItem::new("request_1", "future-claude", language_request(max_tokens))
            .expect("valid batch item")
            .with_options(options)
    }

    fn result_line(custom_id: &str, status: &str) -> Vec<u8> {
        let mut line = serde_json::to_vec(&json!({
            "custom_id": custom_id,
            "result": {
                "type": status,
                "message": {"id": format!("message-{custom_id}")},
                "future_field": true
            }
        }))
        .expect("result JSON");
        line.push(b'\n');
        line
    }

    fn decoder_limits() -> BatchResultsDecoderLimits {
        BatchResultsDecoderLimits {
            max_encoded_bytes: 1024 * 1024,
            max_line_bytes: 64 * 1024,
            max_decoded_line_bytes: 1024 * 1024,
            max_decoded_bytes: 4 * 1024 * 1024,
            max_records: 16,
            max_depth: 16,
            max_nodes: 1024,
            max_string_bytes: 64 * 1024,
        }
    }

    fn local_provider(server: &MockServer) -> AnthropicProvider {
        AnthropicProvider::builder(AnthropicCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit(format!("{}/v1/", server.uri()))
                    .expect("local endpoint"),
            )
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("message-batch-tests").expect("replay domain"),
            ))
            .build()
            .expect("provider")
    }

    #[test]
    fn batch_item_allows_caching_and_fallbacks_but_rejects_current_exclusions() {
        AnthropicBatchItem::new("request/资源:1", "future-claude", language_request(64))
            .expect("custom identifiers remain opaque bounded strings");

        let supported = batch_item(
            AnthropicMessagesOptions::new()
                .with_automatic_cache(AnthropicCacheTtl::OneHour)
                .with_fallbacks(ServerFallbacks::Default),
            64,
        );
        supported
            .validate_for_batch()
            .expect("prompt caching and fallbacks are supported in batches");

        let speed = batch_item(
            AnthropicMessagesOptions::new().with_speed(InferenceSpeed::Fast),
            64,
        );
        assert_eq!(
            speed
                .validate_for_batch()
                .expect_err("speed must be rejected")
                .message(),
            "Anthropic Message Batches do not support speed or Fast mode"
        );

        let fallback = ServerFallback::new("fallback-model")
            .expect("fallback")
            .with_speed(InferenceSpeed::Fast);
        let fallback_speed = batch_item(
            AnthropicMessagesOptions::new()
                .with_fallbacks(ServerFallbacks::explicit(vec![fallback]).expect("fallback chain")),
            64,
        );
        assert!(fallback_speed.validate_for_batch().is_err());

        let zero = batch_item(AnthropicMessagesOptions::new(), 0);
        assert_eq!(
            zero.validate_for_batch()
                .expect_err("zero max_tokens must be rejected")
                .message(),
            "Anthropic Message Batches do not support max_tokens set to zero"
        );

        for (field, value) in [
            ("store", json!(true)),
            ("previous_thread_event_id", json!("thread-event")),
            ("cache_hint", json!({"type": "ephemeral"})),
            ("context_hint", json!({"type": "ephemeral"})),
        ] {
            let item = batch_item(
                AnthropicMessagesOptions::new()
                    .try_with_extra(field, value)
                    .expect("reachable raw option"),
                64,
            );
            assert_eq!(
                item.validate_for_batch()
                    .expect_err("documented top-level exclusion must fail")
                    .message(),
                "Anthropic Message Batches contain an unsupported top-level Messages field"
            );
        }

        let research = batch_item(
            AnthropicMessagesOptions::new()
                .try_with_extra("research_preview_2026_02", json!("active"))
                .expect("research preview option"),
            64,
        );
        assert!(research.validate_for_batch().is_err());
        batch_item(
            AnthropicMessagesOptions::new()
                .try_with_extra("research_preview_2026_02", json!("inactive"))
                .expect("future non-active value"),
            64,
        )
        .validate_for_batch()
        .expect("only the documented active value is excluded");
    }

    #[tokio::test]
    async fn create_uses_one_deduplicated_beta_union_and_preserves_item_options() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/batches"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msgbatch_fixture",
                "type": "message_batch",
                "processing_status": "in_progress"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = local_provider(&server);
        let task_budget = TokenTaskBudget::new(128).expect("task budget");
        let container = MessagesContainer::configured()
            .with_skill(ContainerSkill::custom("skill_fixture").expect("Skill reference"))
            .expect("container Skill");
        let context_management = ContextManagement::new().with_edit(
            ClearToolUsesEdit::new()
                .with_clear_at_least_input_tokens(8_000)
                .with_keep_tool_uses(2),
        );
        let mut feature_request = language_request(64);
        feature_request.tools = vec![
            AnthropicToolOptions::for_tool(AnthropicTool::CodeExecution20260521)
                .into_tool_spec()
                .expect("code execution tool"),
            AnthropicToolOptions::for_tool(AnthropicTool::mcp_toolset(
                McpToolsetOptions::new("docs").expect("MCP toolset"),
            ))
            .into_tool_spec()
            .expect("MCP tool"),
        ];
        let request = AnthropicBatchRequest::new(vec![
            AnthropicBatchItem::new(
                "request_fallback",
                "future-fallback-model",
                language_request(64),
            )
            .expect("fallback item")
            .with_options(
                AnthropicMessagesOptions::new()
                    .with_automatic_cache(AnthropicCacheTtl::OneHour)
                    .with_fallbacks(ServerFallbacks::Default)
                    .with_top_k(7),
            ),
            AnthropicBatchItem::new("request_budget_a", "future-budget-model-a", feature_request)
                .expect("budget item")
                .with_options(
                    AnthropicMessagesOptions::new()
                        .with_task_budget(task_budget)
                        .with_container(container)
                        .with_context_management(context_management)
                        .with_mcp_servers([
                            McpServer::new("docs", "https://mcp.example.test").expect("MCP server")
                        ])
                        .with_top_k(8),
                ),
            AnthropicBatchItem::new(
                "request_budget_b",
                "future-budget-model-b",
                language_request(64),
            )
            .expect("budget item")
            .with_options(
                AnthropicMessagesOptions::new()
                    .with_task_budget(task_budget)
                    .with_top_k(9),
            ),
        ])
        .expect("batch request");

        let batch = provider
            .message_batches()
            .create(request)
            .await
            .expect("batch create");
        assert_eq!(batch.id, "msgbatch_fixture");

        let requests = server.received_requests().await.expect("recorded request");
        assert_eq!(requests.len(), 1);
        let beta = requests[0]
            .headers
            .get("anthropic-beta")
            .expect("feature beta header")
            .to_str()
            .expect("ASCII beta header")
            .split(',')
            .collect::<BTreeSet<_>>();
        assert_eq!(
            beta,
            BTreeSet::from([
                "context-management-2025-06-27",
                "mcp-client-2025-11-20",
                "server-side-fallback-2026-07-01",
                "skills-2025-10-02",
                "task-budgets-2026-03-13",
            ])
        );
        let body: Value = serde_json::from_slice(&requests[0].body).expect("batch request body");
        assert_eq!(body["requests"].as_array().map(Vec::len), Some(3));
        assert_eq!(body["requests"][0]["custom_id"], "request_fallback");
        assert_eq!(
            body["requests"][0]["params"]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(body["requests"][0]["params"]["fallbacks"], "default");
        assert_eq!(body["requests"][0]["params"]["top_k"], 7);
        assert_eq!(
            body["requests"][1]["params"]["output_config"]["task_budget"],
            json!({"type": "tokens", "total": 128})
        );
        assert_eq!(
            body["requests"][1]["params"]["container"]["skills"][0]["skill_id"],
            "skill_fixture"
        );
        assert_eq!(
            body["requests"][1]["params"]["context_management"]["edits"][0]["type"],
            "clear_tool_uses_20250919"
        );
        assert_eq!(
            body["requests"][1]["params"]["mcp_servers"][0]["name"],
            "docs"
        );
        assert_eq!(body["requests"][1]["params"]["top_k"], 8);
        assert_eq!(body["requests"][2]["params"]["top_k"], 9);
    }

    #[tokio::test]
    async fn results_endpoint_uses_opaque_path_and_decodes_incrementally() {
        let server = MockServer::start().await;
        let mut body = result_line("request_2", "succeeded");
        body.extend(result_line("request_1", "future_result"));
        Mock::given(method("GET"))
            .and(path(
                "/v1/messages/batches/msg%2Fopaque%3F%23%252F%E8%B5%84%E6%BA%90/results",
            ))
            .and(header("accept", "application/x-jsonlines"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/x-jsonlines"))
            .expect(1)
            .mount(&server)
            .await;

        let mut results = local_provider(&server)
            .message_batches()
            .results("msg/opaque?#%2F资源")
            .await
            .expect("established results stream");
        let first = results
            .next()
            .await
            .expect("first record")
            .expect("valid first record");
        let second = results
            .next()
            .await
            .expect("second record")
            .expect("valid second record");

        assert_eq!(first.custom_id, "request_2");
        assert!(first.status().is_succeeded());
        assert_eq!(second.custom_id, "request_1");
        assert_eq!(second.status().as_str(), "future_result");
        assert!(results.next().await.is_none());
        assert_eq!(results.decoder().records_decoded(), 2);
    }

    #[tokio::test]
    async fn cancel_and_delete_never_replay_after_submission() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/batches/msg%2Fcancel/cancel"))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/v1/messages/batches/msg%2Fdelete"))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&server)
            .await;

        let batches = local_provider(&server).message_batches();
        assert!(batches.cancel("msg/cancel").await.is_err());
        assert!(batches.delete("msg/delete").await.is_err());
    }

    #[test]
    fn prompt_cache_controls_survive_batch_item_encoding() {
        let annotated_message = Message::user("stable prefix")
            .with_provider_annotation(&AnthropicMessageCache::one_hour())
            .expect("cache annotation");
        let mut request = LanguageRequest::new(vec![annotated_message]);
        request.generation.max_output_tokens = Some(64);
        let scope = ProviderScope::new(ProviderId::new("anthropic").expect("provider"));
        let encoded = encode_request_for_scope_with_resolver(
            &scope,
            &ModelId::new("future-model").expect("model"),
            &request,
            &AnthropicMessagesOptions::new()
                .with_automatic_cache(AnthropicCacheTtl::OneHour)
                .to_protocol(false),
            &AnthropicAnnotationResolver,
        )
        .expect("batch item encoding");

        assert_eq!(encoded["cache_control"]["ttl"], "1h");
        assert_eq!(
            encoded["messages"][0]["content"][0]["cache_control"]["ttl"],
            "1h"
        );
    }

    #[test]
    fn create_body_serialization_is_bounded_before_transport() {
        let item = BatchItemWire {
            custom_id: "request_1".to_string(),
            params: json!({"model": "future-model", "messages": []}),
        };
        let mut body = BatchCreateBodyEncoder::new(16).expect("bounded encoder prefix");
        let error = body
            .push(&item)
            .expect_err("small Siumai safety bound must stop serialization");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        assert_eq!(
            error.message(),
            "Anthropic message batch exceeds the Siumai encoded request safety bound"
        );
    }

    #[tokio::test]
    async fn create_honors_the_smaller_configured_transport_limit() {
        let server = MockServer::start().await;
        let limits = TransportLimits {
            max_request_bytes: 256,
            ..TransportLimits::default()
        };
        let provider = AnthropicProvider::builder(AnthropicCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit(format!("{}/v1/", server.uri()))
                    .expect("local endpoint"),
            )
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("small-batch-request-limit").expect("replay domain"),
            ))
            .with_transport_limits(limits)
            .build()
            .expect("provider");
        let mut request = language_request(64);
        request.messages = vec![Message::user("x".repeat(2_048))];
        let request = AnthropicBatchRequest::new(vec![
            AnthropicBatchItem::new("request_1", "future-model", request).expect("batch item"),
        ])
        .expect("batch request");

        let error = provider
            .message_batches()
            .create(request)
            .await
            .expect_err("configured transport limit must stop encoding before submission");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        assert!(
            server
                .received_requests()
                .await
                .expect("recorded requests")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn results_http_errors_share_resource_classification_and_sanitization() {
        let server = MockServer::start().await;
        let sentinel = "private-batch-error-body";
        Mock::given(method("GET"))
            .and(path("/v1/messages/batches/msg_error/results"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": sentinel
                },
                "request_id": "req_batch_results_42"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let error = local_provider(&server)
            .message_batches()
            .results("msg_error")
            .await
            .expect_err("provider error");
        assert_eq!(error.kind(), ErrorKind::RateLimited);
        let diagnostics = error.diagnostics().expect("response diagnostics");
        assert_eq!(diagnostics.status(), Some(400));
        assert_eq!(diagnostics.provider_type(), Some("rate_limit_error"));
        assert_eq!(diagnostics.request_id(), Some("req_batch_results_42"));
        assert!(!format!("{error}").contains(sentinel));
        assert!(!format!("{error:?}").contains(sentinel));
        let (_, body) = error
            .sensitive_response()
            .expect("bounded sensitive response")
            .expose();
        assert!(
            std::str::from_utf8(body)
                .expect("JSON body")
                .contains(sentinel)
        );
    }

    #[test]
    fn open_status_wrappers_preserve_future_values_with_bounds() {
        let batch: AnthropicMessageBatch = serde_json::from_value(json!({
            "id": "msgbatch_1",
            "processing_status": "waiting_for_future_capacity"
        }))
        .expect("open processing status");
        assert_eq!(
            batch.processing_status.as_ref().expect("status").as_str(),
            "waiting_for_future_capacity"
        );

        let record: AnthropicBatchResult = serde_json::from_value(json!({
            "custom_id": "request:external",
            "result": {"type": "future_result", "payload": {"ok": true}}
        }))
        .expect("open result status");
        assert_eq!(record.status().as_str(), "future_result");
        assert_eq!(record.result.extra["payload"]["ok"], true);
        let debug = format!("{record:?}");
        assert!(!debug.contains("request:external"));
        assert!(!debug.contains("payload"));

        assert!(AnthropicBatchResultStatus::new("bad\nstatus").is_err());
        assert!(AnthropicBatchProcessingStatus::new("x".repeat(129)).is_err());
    }

    #[test]
    fn resource_debug_redacts_request_and_provider_identifiers() {
        let item = batch_item(AnthropicMessagesOptions::new(), 64);
        let request_debug = format!("{:?}", AnthropicBatchRequest::new(vec![item]).unwrap());
        assert!(!request_debug.contains("batch fixture"));
        assert!(!request_debug.contains("request_1"));

        let batch: AnthropicMessageBatch = serde_json::from_value(json!({
            "id": "msgbatch-private-canary",
            "type": "private-type-canary",
            "processing_status": "private-status-canary",
            "results_url": "https://private.invalid/result",
            "private-extra": "private-body-canary"
        }))
        .expect("batch response");
        let batch_debug = format!("{batch:?}");
        for sentinel in [
            "msgbatch-private-canary",
            "private-type-canary",
            "private-status-canary",
            "private.invalid",
            "private-body-canary",
        ] {
            assert!(!batch_debug.contains(sentinel));
        }

        let list = AnthropicBatchList {
            data: vec![batch],
            first_id: Some("first-private-canary".to_owned()),
            last_id: Some("last-private-canary".to_owned()),
            has_more: true,
        };
        let list_debug = format!("{list:?}");
        assert!(!list_debug.contains("first-private-canary"));
        assert!(!list_debug.contains("last-private-canary"));

        let deleted = AnthropicBatchDeleteResult {
            id: "delete-private-canary".to_owned(),
            object_type: "delete-type-private-canary".to_owned(),
        };
        let deleted_debug = format!("{deleted:?}");
        assert!(!deleted_debug.contains("delete-private-canary"));
        assert!(!deleted_debug.contains("delete-type-private-canary"));
    }

    #[test]
    fn delete_response_matches_the_open_provider_wire_shape() {
        let deleted: AnthropicBatchDeleteResult = serde_json::from_value(json!({
            "id": "msgbatch_123",
            "type": "message_batch_deleted",
        }))
        .expect("delete response");

        assert!(deleted.is_deleted());
        assert!(
            serde_json::from_value::<AnthropicBatchDeleteResult>(json!({
                "id": "msgbatch_123",
            }))
            .is_err()
        );
    }

    #[test]
    fn terminal_decoder_states_release_the_large_line_buffer() {
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(decoder_limits());
        decoder.line = Vec::with_capacity(1024 * 1024);
        decoder.finish();
        assert_eq!(decoder.line.capacity(), 0);

        let mut decoder = AnthropicBatchResultsDecoder::with_limits(decoder_limits());
        decoder.line = Vec::with_capacity(1024 * 1024);
        decoder.abort();
        assert_eq!(decoder.line.capacity(), 0);
    }

    #[test]
    fn decoder_handles_fragmented_unordered_records_and_future_results() {
        let mut input = result_line("request_2", "succeeded");
        input.extend(result_line("request_1", "future_result"));
        let split = input
            .iter()
            .position(|byte| *byte == b'2')
            .expect("split byte")
            + 1;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(decoder_limits());

        let mut output = decoder.push(&input[..split]);
        output.extend(decoder.push(&input[split..]));
        output.extend(decoder.finish());

        let records = output
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("valid JSONL records");
        assert_eq!(
            records
                .iter()
                .map(|record| record.custom_id.as_str())
                .collect::<Vec<_>>(),
            vec!["request_2", "request_1"]
        );
        assert!(records[0].status().is_succeeded());
        assert_eq!(records[1].status().as_str(), "future_result");
        assert_eq!(decoder.records_decoded(), 2);
    }

    #[test]
    fn decoder_emits_prior_records_then_one_sanitized_error() {
        let mut input = result_line("request_1", "succeeded");
        input.extend_from_slice(br#"{"secret":"raw-canary" not-json}"#);
        input.push(b'\n');
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(decoder_limits());

        let output = decoder.push(&input);
        assert_eq!(output.len(), 2);
        assert!(output[0].is_ok());
        let error = output[1].as_ref().expect_err("malformed record");
        assert_eq!(
            error,
            &AnthropicBatchResultsDecodeError::InvalidJson { line: 2 }
        );
        assert!(!format!("{error}").contains("raw-canary"));
        assert!(!format!("{error:?}").contains("raw-canary"));
        assert!(decoder.push(b"{}\n").is_empty());
        assert!(decoder.finish().is_empty());
    }

    #[test]
    fn decoder_rejects_blank_jsonl_records_once() {
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(decoder_limits());
        let output = decoder.push(b" \t\r\n");
        assert_eq!(
            output,
            vec![Err(AnthropicBatchResultsDecodeError::InvalidJson {
                line: 1
            })]
        );
        assert!(
            decoder
                .push(&result_line("request_1", "succeeded"))
                .is_empty()
        );
        assert!(decoder.finish().is_empty());
    }

    #[test]
    fn decoder_reports_invalid_utf8_and_truncated_final_line_once() {
        let mut invalid_utf8 = AnthropicBatchResultsDecoder::with_limits(decoder_limits());
        assert_eq!(
            invalid_utf8.push(&[0xff, b'\n']),
            vec![Err(AnthropicBatchResultsDecodeError::InvalidUtf8 {
                line: 1
            })]
        );
        assert!(invalid_utf8.finish().is_empty());

        let mut truncated = AnthropicBatchResultsDecoder::with_limits(decoder_limits());
        let mut input = result_line("request_1", "succeeded");
        input.extend_from_slice(br#"{"custom_id":"request_2"}"#);
        let output = truncated.push(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_ok());
        assert_eq!(
            truncated.finish(),
            vec![Err(AnthropicBatchResultsDecodeError::TruncatedFinalLine {
                line: 2
            })]
        );
        assert!(truncated.finish().is_empty());
    }

    #[tokio::test]
    async fn established_results_stream_orders_records_and_one_terminal_error() {
        let mut chunk = result_line("request_1", "succeeded");
        chunk.extend_from_slice(br#"{"secret":"stream-canary" broken}"#);
        chunk.push(b'\n');
        let source = stream::iter([Ok::<Bytes, Error>(Bytes::from(chunk))]);
        let mut results = BatchResultsStream::new(
            source,
            AnthropicBatchResultsDecoder::with_limits(decoder_limits()),
        );

        let record = results
            .next()
            .await
            .expect("record")
            .expect("successful record");
        assert_eq!(record.custom_id, "request_1");
        let error = results
            .next()
            .await
            .expect("decode error")
            .expect_err("malformed second line");
        assert!(matches!(
            &error,
            AnthropicBatchResultsStreamError::Decode(
                AnthropicBatchResultsDecodeError::InvalidJson { line: 2 }
            )
        ));
        assert!(!format!("{error}").contains("stream-canary"));
        assert!(results.next().await.is_none());

        let source = stream::iter([Ok::<Bytes, Error>(Bytes::from_static(
            br#"{"custom_id":"truncated"}"#,
        ))]);
        let mut truncated = BatchResultsStream::new(
            source,
            AnthropicBatchResultsDecoder::with_limits(decoder_limits()),
        );
        assert!(matches!(
            truncated.next().await,
            Some(Err(AnthropicBatchResultsStreamError::Decode(
                AnthropicBatchResultsDecodeError::TruncatedFinalLine { line: 1 }
            )))
        ));
        assert!(truncated.next().await.is_none());

        let source = stream::iter([
            Ok::<Bytes, Error>(Bytes::from_static(br#"{"custom_id":"partial"}"#)),
            Err(Error::new(
                ErrorKind::Unavailable,
                "bounded transport fixture failed",
            )),
        ]);
        let mut transport_failure = BatchResultsStream::new(
            source,
            AnthropicBatchResultsDecoder::with_limits(decoder_limits()),
        );
        assert!(matches!(
            transport_failure.next().await,
            Some(Err(AnthropicBatchResultsStreamError::Transport(_)))
        ));
        assert!(transport_failure.next().await.is_none());
    }

    #[tokio::test]
    async fn established_stream_applies_backpressure_within_one_transport_chunk() {
        let mut chunk = result_line("request_1", "succeeded");
        chunk.extend(result_line("request_2", "succeeded"));
        chunk.extend(result_line("request_3", "succeeded"));
        let source = stream::iter([Ok::<Bytes, Error>(Bytes::from(chunk))]);
        let mut results = BatchResultsStream::new(
            source,
            AnthropicBatchResultsDecoder::with_limits(decoder_limits()),
        );

        let first = results.next().await.expect("first result").expect("record");
        assert_eq!(first.custom_id, "request_1");
        assert_eq!(results.decoder.records_decoded(), 1);
        assert!(results.chunk.is_some());

        let second = results
            .next()
            .await
            .expect("second result")
            .expect("record");
        assert_eq!(second.custom_id, "request_2");
        assert_eq!(results.decoder.records_decoded(), 2);
        assert!(results.chunk.is_some());

        let third = results.next().await.expect("third result").expect("record");
        assert_eq!(third.custom_id, "request_3");
        assert_eq!(results.decoder.records_decoded(), 3);
        assert!(results.next().await.is_none());
    }

    #[test]
    fn decoder_enforces_each_resource_bound() {
        let valid = result_line("request_1", "succeeded");

        let mut limits = decoder_limits();
        limits.max_encoded_bytes = 2;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        assert_eq!(
            decoder.push(b"{}x"),
            vec![Err(
                AnthropicBatchResultsDecodeError::EncodedAggregateTooLarge { maximum: 2 }
            )]
        );

        let mut limits = decoder_limits();
        limits.max_line_bytes = 8;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        assert_eq!(
            decoder.push(b"123456789"),
            vec![Err(AnthropicBatchResultsDecodeError::LineTooLarge {
                line: 1,
                maximum: 8
            })]
        );

        let mut limits = decoder_limits();
        limits.max_records = 1;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        let mut two = valid.clone();
        two.extend(valid.clone());
        let output = decoder.push(&two);
        assert!(output[0].is_ok());
        assert_eq!(
            output[1],
            Err(AnthropicBatchResultsDecodeError::TooManyRecords { maximum: 1 })
        );

        let mut limits = decoder_limits();
        limits.max_depth = 2;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        assert!(matches!(
            decoder.push(&valid).as_slice(),
            [Err(AnthropicBatchResultsDecodeError::JsonDepthExceeded {
                maximum: 2,
                ..
            })]
        ));

        let mut limits = decoder_limits();
        limits.max_nodes = 2;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        assert!(matches!(
            decoder.push(&valid).as_slice(),
            [Err(
                AnthropicBatchResultsDecodeError::JsonNodeLimitExceeded { maximum: 2, .. }
            )]
        ));

        let mut limits = decoder_limits();
        limits.max_string_bytes = 8;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        assert!(matches!(
            decoder.push(&valid).as_slice(),
            [Err(AnthropicBatchResultsDecodeError::StringTooLarge {
                maximum: 8,
                ..
            })]
        ));

        let mut limits = decoder_limits();
        limits.max_decoded_line_bytes = 1;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        assert!(matches!(
            decoder.push(&valid).as_slice(),
            [Err(AnthropicBatchResultsDecodeError::DecodedLineTooLarge {
                maximum: 1,
                ..
            })]
        ));

        let value: Value = serde_json::from_slice(&valid[..valid.len() - 1]).expect("value");
        let decoded_size = measure_json_value(&value, 1, decoder_limits()).expect("measurement");
        let mut limits = decoder_limits();
        limits.max_decoded_bytes = decoded_size;
        let mut decoder = AnthropicBatchResultsDecoder::with_limits(limits);
        let mut two = valid.clone();
        two.extend(valid);
        let output = decoder.push(&two);
        assert!(output[0].is_ok());
        assert_eq!(
            output[1],
            Err(AnthropicBatchResultsDecodeError::DecodedAggregateTooLarge {
                maximum: decoded_size
            })
        );
    }

    #[test]
    fn decoder_maps_each_transport_limit_to_the_owned_jsonl_budget() {
        let transport = TransportLimits {
            max_response_bytes: 91,
            max_frame_bytes: 83,
            max_event_bytes: 79,
            max_events_per_stream: 71,
            ..TransportLimits::default()
        };
        let limits = BatchResultsDecoderLimits::from_transport(&transport);
        assert_eq!(limits.max_encoded_bytes, 91);
        assert_eq!(limits.max_decoded_bytes, 91);
        assert_eq!(limits.max_line_bytes, 83);
        assert_eq!(limits.max_decoded_line_bytes, 79);
        assert_eq!(limits.max_string_bytes, 79);
        assert_eq!(limits.max_records, 71);
    }

    #[test]
    fn bounded_json_seed_stops_high_node_input_during_construction() {
        let mut limits = decoder_limits();
        limits.max_nodes = 4;
        let error =
            deserialize_bounded_json_value(r#"[null,null,null,null,null,null,null]"#, 7, limits)
                .expect_err("node budget must stop the visitor before the whole tree is built");
        assert_eq!(
            error,
            AnthropicBatchResultsDecodeError::JsonNodeLimitExceeded {
                line: 7,
                maximum: 4
            }
        );
    }

    #[test]
    fn resource_identifiers_are_encoded_once_and_list_cursors_are_opaque() {
        let target = batch_target("msg/\\?#%2F资源", &["results"]).expect("batch target");
        assert_eq!(
            target.as_str(),
            "messages/batches/msg%2F%5C%3F%23%252F%E8%B5%84%E6%BA%90/results"
        );

        let query = list_target(&AnthropicBatchListQuery {
            before_id: Some("before/?#%2F资源".to_string()),
            after_id: None,
            limit: Some(10),
        })
        .expect("list target");
        assert_eq!(
            query,
            "messages/batches?before_id=before%2F%3F%23%252F%E8%B5%84%E6%BA%90&limit=10"
        );
    }
}
