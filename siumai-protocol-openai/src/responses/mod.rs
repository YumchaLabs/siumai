//! Native OpenAI Responses protocol codec.
//!
//! This module deliberately does not reuse the Chat Completions converter.
//! Responses output items are retained as native opaque history alongside their
//! portable projection so stateless reasoning and programmatic tool calling can
//! be replayed without identity loss.

mod request;
mod response;
mod stream;
mod wire;

pub use request::{
    FunctionToolCaller, FunctionToolEncodingOptions, PromptCacheBlock, RequestEncodingOptions,
    ResponsesMediaDialect, TEXT_VERBOSITY_OPTION, encode_request, encode_request_with_options,
    is_protected_option_field,
};
pub use response::{DecodedResponse, decode_response, decode_response_resource};
pub use stream::{
    DecodedResponsesStreamFrame, ResponsesStreamDecoder, ResponsesStreamEvent,
    ResponsesStreamEventKind, ResponsesTerminalPolicy,
};
pub use wire::{
    AnnotationWire, CustomToolCallItemWire, FunctionCallItemWire, IncompleteDetailsWire,
    InputTokenDetailsWire, ItemStatus, MessageItemWire, OutputContentPart, OutputItem,
    OutputRefusalWire, OutputTextWire, OutputTokenDetailsWire, ProgramItemWire,
    ProgramOutputItemWire, ProviderToolItemWire, ReasoningItemWire, ReasoningTextWire,
    ResponseErrorWire, ResponseReasoningConfigWire, ResponseStatus, ResponseUsageWire,
    ResponseWire, StreamEventWire, ToolCallerWire, UnknownContentPartWire, UnknownOutputItemWire,
};

/// Stable provenance marker for same-protocol history replay.
pub const OPENAI_RESPONSES_PROTOCOL: &str = "openai.responses";

/// Stable API-mode identity for the Responses operation.
pub const API_MODE_ID: &str = "responses";

/// Relative OpenAI transport target for Responses generation.
pub const RESPONSES_TARGET: &str = "responses";

/// Opaque item kind used for full native output item envelopes.
pub const OPENAI_RESPONSES_OPAQUE_KIND: &str = "response.output_item";

#[cfg(test)]
mod tests;
