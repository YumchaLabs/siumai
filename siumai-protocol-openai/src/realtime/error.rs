//! Errors produced by the OpenAI Realtime wire codecs.

use thiserror::Error;

/// Result type used by the OpenAI Realtime wire codecs.
pub type RealtimeCodecResult<T> = Result<T, RealtimeCodecError>;

/// A local protocol, state, or resource-limit violation.
///
/// OpenAI `error` server events are intentionally not represented here. They
/// are valid wire events and are returned as typed server events so callers may
/// recover without losing the session.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RealtimeCodecError {
    #[error("Realtime text frame must not be empty")]
    EmptyTextFrame,

    #[error("Realtime text frame is {actual} bytes, exceeding the {maximum}-byte limit")]
    TextFrameTooLarge { actual: usize, maximum: usize },

    #[error(
        "raw WebSocket binary frames are unsupported by the OpenAI Realtime JSON protocol ({bytes} bytes received)"
    )]
    RawBinaryFrameUnsupported { bytes: usize },

    #[error("invalid Realtime JSON: {message}")]
    InvalidJson { message: String },

    #[error("Realtime event must be a JSON object")]
    EventMustBeObject,

    #[error("Realtime event is missing a non-empty string `type` field")]
    MissingEventType,

    #[error("event `{event_type}` is missing required field `{field}`")]
    MissingField {
        event_type: String,
        field: &'static str,
    },

    #[error("event `{event_type}` has invalid field `{field}`: {reason}")]
    InvalidField {
        event_type: String,
        field: &'static str,
        reason: String,
    },

    #[error("event `{event_type}` field `{field}` is not valid base64: {message}")]
    InvalidBase64 {
        event_type: String,
        field: &'static str,
        message: String,
    },

    #[error(
        "event `{event_type}` decoded {actual} audio bytes, exceeding the {maximum}-byte per-event limit"
    )]
    DecodedAudioTooLarge {
        event_type: String,
        actual: usize,
        maximum: usize,
    },

    #[error("active function-call accumulation exceeded the configured limit of {maximum}")]
    ActiveFunctionCallLimitExceeded { maximum: usize },

    #[error(
        "function call `{call_id}` accumulated {actual} argument bytes, exceeding the {maximum}-byte per-call limit"
    )]
    FunctionArgumentsTooLarge {
        call_id: String,
        actual: usize,
        maximum: usize,
    },

    #[error(
        "all active function calls accumulated {actual} argument bytes, exceeding the {maximum}-byte total limit"
    )]
    TotalFunctionArgumentsTooLarge { actual: usize, maximum: usize },

    #[error("invalid Realtime codec limit `{field}`: it must be greater than zero")]
    InvalidLimit { field: &'static str },

    #[error("cannot encode Realtime JSON: {message}")]
    JsonEncoding { message: String },

    #[error("translation client event `{event_type}` is invalid while the session is {state}")]
    TranslationClientStateViolation {
        event_type: String,
        state: &'static str,
    },

    #[error("translation server event `{event_type}` is invalid while the session is {state}")]
    TranslationServerStateViolation {
        event_type: String,
        state: &'static str,
    },

    #[error("translation transport ended before the required `session.closed` event")]
    TranslationClosedBeforeSessionClosed,
}
