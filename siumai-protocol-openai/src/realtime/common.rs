//! Shared OpenAI Realtime wire primitives.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use serde_json::{Map, Value};
use std::fmt;

use super::{
    error::{RealtimeCodecError, RealtimeCodecResult},
    frame::{JsonTextFrame, RealtimeInputFrame},
};

/// Resource limits enforced before or while decoding Realtime events.
///
/// These limits bound all state retained by the codec. Individual transports
/// should additionally enforce their own socket and queue limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RealtimeCodecLimits {
    pub max_text_frame_bytes: usize,
    pub max_decoded_audio_bytes_per_event: usize,
    pub max_active_function_calls: usize,
    pub max_function_arguments_bytes_per_call: usize,
    pub max_function_arguments_bytes_total: usize,
}

impl RealtimeCodecLimits {
    pub(crate) fn validate(self) -> RealtimeCodecResult<Self> {
        for (field, value) in [
            ("max_text_frame_bytes", self.max_text_frame_bytes),
            (
                "max_decoded_audio_bytes_per_event",
                self.max_decoded_audio_bytes_per_event,
            ),
            ("max_active_function_calls", self.max_active_function_calls),
            (
                "max_function_arguments_bytes_per_call",
                self.max_function_arguments_bytes_per_call,
            ),
            (
                "max_function_arguments_bytes_total",
                self.max_function_arguments_bytes_total,
            ),
        ] {
            if value == 0 {
                return Err(RealtimeCodecError::InvalidLimit { field });
            }
        }

        Ok(self)
    }
}

impl Default for RealtimeCodecLimits {
    fn default() -> Self {
        Self {
            max_text_frame_bytes: 4 * 1024 * 1024,
            max_decoded_audio_bytes_per_event: 2 * 1024 * 1024,
            max_active_function_calls: 64,
            max_function_arguments_bytes_per_call: 1024 * 1024,
            max_function_arguments_bytes_total: 8 * 1024 * 1024,
        }
    }
}

/// A typed event paired with its exact source text and parsed JSON value.
///
/// Keeping both forms makes unknown future events replayable without losing
/// whitespace, numeric spelling, field order, or fields unknown to this crate.
#[derive(Clone, PartialEq)]
pub struct DecodedRealtimeEvent<T> {
    pub event: T,
    pub raw: Value,
    pub raw_json: String,
}

impl<T: fmt::Debug> fmt::Debug for DecodedRealtimeEvent<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodedRealtimeEvent")
            .field("event", &self.event)
            .field("event_type", &self.event_type())
            .field("raw_json_bytes", &self.raw_json.len())
            .finish_non_exhaustive()
    }
}

impl<T> DecodedRealtimeEvent<T> {
    pub(crate) fn new(event: T, raw: Value, raw_json: String) -> Self {
        Self {
            event,
            raw,
            raw_json,
        }
    }

    /// Transforms the typed event while preserving its lossless wire payload.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> DecodedRealtimeEvent<U> {
        DecodedRealtimeEvent {
            event: map(self.event),
            raw: self.raw,
            raw_json: self.raw_json,
        }
    }

    /// Returns the original wire event type.
    pub fn event_type(&self) -> Option<&str> {
        self.raw.get("type").and_then(Value::as_str)
    }
}

/// Metadata for a server event type that the current crate does not recognize.
///
/// The complete event remains available through [`DecodedRealtimeEvent::raw`]
/// and [`DecodedRealtimeEvent::raw_json`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownRealtimeEvent {
    pub event_type: String,
}

/// An OpenAI `error` event returned as recoverable protocol data.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiRealtimeServerError {
    pub event_id: Option<String>,
    pub message: String,
    pub kind: Option<String>,
    pub code: Option<String>,
    pub related_event_id: Option<String>,
    pub param: Option<Value>,
    pub details: Map<String, Value>,
}

impl OpenAiRealtimeServerError {
    /// Server `error` events do not, by themselves, terminate a session.
    pub const fn is_session_terminal(&self) -> bool {
        false
    }
}

pub(crate) struct ParsedEvent {
    pub event_type: String,
    object: Map<String, Value>,
    pub raw_json: String,
}

impl ParsedEvent {
    pub fn object(&self) -> &Map<String, Value> {
        &self.object
    }

    pub fn into_decoded<T>(self, event: T) -> DecodedRealtimeEvent<T> {
        DecodedRealtimeEvent::new(event, Value::Object(self.object), self.raw_json)
    }
}

pub(crate) fn parse_frame(
    frame: RealtimeInputFrame<'_>,
    limits: RealtimeCodecLimits,
) -> RealtimeCodecResult<ParsedEvent> {
    let text = match frame {
        RealtimeInputFrame::Text(text) => text,
        RealtimeInputFrame::Binary(bytes) => {
            return Err(RealtimeCodecError::RawBinaryFrameUnsupported { bytes: bytes.len() });
        }
    };

    if text.is_empty() {
        return Err(RealtimeCodecError::EmptyTextFrame);
    }
    if text.len() > limits.max_text_frame_bytes {
        return Err(RealtimeCodecError::TextFrameTooLarge {
            actual: text.len(),
            maximum: limits.max_text_frame_bytes,
        });
    }

    let raw =
        serde_json::from_str::<Value>(text).map_err(|error| RealtimeCodecError::InvalidJson {
            message: error.to_string(),
        })?;
    let Value::Object(object) = raw else {
        return Err(RealtimeCodecError::EventMustBeObject);
    };
    let event_type = object
        .get("type")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(RealtimeCodecError::MissingEventType)?
        .to_owned();

    Ok(ParsedEvent {
        event_type,
        object,
        raw_json: text.to_owned(),
    })
}

pub(crate) fn encode_object(object: Map<String, Value>) -> RealtimeCodecResult<JsonTextFrame> {
    serde_json::to_string(&Value::Object(object))
        .map(JsonTextFrame::new)
        .map_err(|error| RealtimeCodecError::JsonEncoding {
            message: error.to_string(),
        })
}

pub(crate) fn event_object(
    event_type: &'static str,
    event_id: &Option<String>,
) -> Map<String, Value> {
    let mut object = Map::new();
    object.insert("type".to_owned(), Value::String(event_type.to_owned()));
    if let Some(event_id) = event_id {
        object.insert("event_id".to_owned(), Value::String(event_id.clone()));
    }
    object
}

pub(crate) fn required_string(
    object: &Map<String, Value>,
    event_type: &str,
    field: &'static str,
) -> RealtimeCodecResult<String> {
    match object.get(field) {
        None => Err(RealtimeCodecError::MissingField {
            event_type: event_type.to_owned(),
            field,
        }),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(RealtimeCodecError::InvalidField {
            event_type: event_type.to_owned(),
            field,
            reason: "expected a string".to_owned(),
        }),
    }
}

pub(crate) fn optional_string(
    object: &Map<String, Value>,
    event_type: &str,
    field: &'static str,
) -> RealtimeCodecResult<Option<String>> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(RealtimeCodecError::InvalidField {
            event_type: event_type.to_owned(),
            field,
            reason: "expected a string or null".to_owned(),
        }),
    }
}

pub(crate) fn optional_u64(
    object: &Map<String, Value>,
    event_type: &str,
    field: &'static str,
) -> RealtimeCodecResult<Option<u64>> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => {
            value
                .as_u64()
                .map(Some)
                .ok_or_else(|| RealtimeCodecError::InvalidField {
                    event_type: event_type.to_owned(),
                    field,
                    reason: "expected a non-negative integer".to_owned(),
                })
        }
        Some(_) => Err(RealtimeCodecError::InvalidField {
            event_type: event_type.to_owned(),
            field,
            reason: "expected a non-negative integer or null".to_owned(),
        }),
    }
}

pub(crate) fn event_id(
    object: &Map<String, Value>,
    event_type: &str,
) -> RealtimeCodecResult<Option<String>> {
    optional_string(object, event_type, "event_id")
}

pub(crate) fn required_value(
    object: &Map<String, Value>,
    event_type: &str,
    field: &'static str,
) -> RealtimeCodecResult<Value> {
    object
        .get(field)
        .cloned()
        .ok_or_else(|| RealtimeCodecError::MissingField {
            event_type: event_type.to_owned(),
            field,
        })
}

pub(crate) fn decode_audio(
    object: &Map<String, Value>,
    event_type: &str,
    field: &'static str,
    limits: RealtimeCodecLimits,
) -> RealtimeCodecResult<Bytes> {
    let encoded = required_string(object, event_type, field)?;
    let decoded =
        STANDARD
            .decode(encoded.as_bytes())
            .map_err(|error| RealtimeCodecError::InvalidBase64 {
                event_type: event_type.to_owned(),
                field,
                message: error.to_string(),
            })?;

    if decoded.len() > limits.max_decoded_audio_bytes_per_event {
        return Err(RealtimeCodecError::DecodedAudioTooLarge {
            event_type: event_type.to_owned(),
            actual: decoded.len(),
            maximum: limits.max_decoded_audio_bytes_per_event,
        });
    }

    Ok(Bytes::from(decoded))
}

pub(crate) fn encode_audio(audio: &Bytes) -> String {
    STANDARD.encode(audio.as_ref())
}

pub(crate) fn parse_server_error(
    object: &Map<String, Value>,
    event_type: &str,
) -> RealtimeCodecResult<OpenAiRealtimeServerError> {
    let event_id = event_id(object, event_type)?;
    let details = match object.get("error") {
        Some(Value::Object(error)) => error.clone(),
        Some(_) => {
            return Err(RealtimeCodecError::InvalidField {
                event_type: event_type.to_owned(),
                field: "error",
                reason: "expected an object".to_owned(),
            });
        }
        None => Map::new(),
    };

    let message = details
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| object.get("message").and_then(Value::as_str))
        .unwrap_or("Unknown OpenAI Realtime error")
        .to_owned();
    let kind = details
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let code = details
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| object.get("code").and_then(Value::as_str))
        .map(str::to_owned);
    let related_event_id = details
        .get("event_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let param = details.get("param").cloned();

    Ok(OpenAiRealtimeServerError {
        event_id,
        message,
        kind,
        code,
        related_event_id,
        param,
        details,
    })
}
