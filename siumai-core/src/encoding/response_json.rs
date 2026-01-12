//! Non-streaming JSON response encoders.
//!
//! This complements the streaming encoder pipeline (`ChatStreamEvent` -> bytes).
//! For non-streaming gateways, we need to encode a final `ChatResponse` into a
//! provider-native JSON response body.
//!
//! English-only comments in code as requested.

use crate::error::LlmError;
use crate::types::ChatResponse;

/// JSON encoding options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonEncodeOptions {
    /// Pretty-print JSON output.
    pub pretty: bool,
}

impl Default for JsonEncodeOptions {
    fn default() -> Self {
        Self { pretty: false }
    }
}

/// A protocol-level encoder that serializes a unified `ChatResponse` into a provider-native JSON
/// response body.
///
/// Design goals:
/// - Avoid intermediate `serde_json::Value` allocations for the common case.
/// - Allow protocol crates to own the exact response shape and keep it in sync with fixtures/docs.
pub trait JsonResponseConverter: Send + Sync {
    /// MIME type for the encoded body.
    fn content_type(&self) -> &'static str {
        "application/json"
    }

    /// Serialize `response` into `out`.
    ///
    /// Implementations should treat `out` as an append buffer and not assume it is empty.
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError>;
}

/// Encode a unified `ChatResponse` into a provider-native JSON body.
pub fn encode_chat_response_as_json<C: JsonResponseConverter>(
    response: &ChatResponse,
    converter: C,
    opts: JsonEncodeOptions,
) -> Result<Vec<u8>, LlmError> {
    let mut out = Vec::new();
    converter.serialize_response(response, &mut out, opts)?;
    Ok(out)
}
