//! Transport-neutral Realtime frame types.

use bytes::Bytes;
use std::fmt;

/// A borrowed WebSocket frame presented to a Realtime decoder.
///
/// OpenAI currently defines all Realtime application messages as JSON text
/// frames. A binary variant is retained so transports can report a typed
/// protocol violation instead of attempting UTF-8 recovery implicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RealtimeInputFrame<'a> {
    Text(&'a str),
    Binary(&'a [u8]),
}

impl<'a> From<&'a str> for RealtimeInputFrame<'a> {
    fn from(value: &'a str) -> Self {
        Self::Text(value)
    }
}

impl<'a> From<&'a String> for RealtimeInputFrame<'a> {
    fn from(value: &'a String) -> Self {
        Self::Text(value.as_str())
    }
}

impl<'a> From<&'a [u8]> for RealtimeInputFrame<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self::Binary(value)
    }
}

impl<'a> From<&'a Bytes> for RealtimeInputFrame<'a> {
    fn from(value: &'a Bytes) -> Self {
        Self::Binary(value.as_ref())
    }
}

/// A validated outbound JSON text frame.
#[derive(Clone, PartialEq, Eq)]
pub struct JsonTextFrame(String);

impl JsonTextFrame {
    pub(crate) fn new(json: String) -> Self {
        Self(json)
    }

    /// Returns the serialized JSON payload.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Consumes the frame and returns its serialized JSON payload.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for JsonTextFrame {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Debug for JsonTextFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsonTextFrame")
            .field("json_bytes", &self.0.len())
            .finish_non_exhaustive()
    }
}

impl From<JsonTextFrame> for String {
    fn from(value: JsonTextFrame) -> Self {
        value.into_string()
    }
}
