//! Axum-specific server adapters
//!
//! This module provides utilities to convert `ChatStream` into Axum-compatible responses.
//!
//! ## Features
//!
//! - **SSE Response**: `to_sse_response()` converts `ChatStream` to `Sse<impl Stream>`
//! - **Text Response**: `to_text_stream()` converts `ChatStream` to plain text stream
//! - **Gateway Helpers**: provider-native SSE/JSON transcoding helpers
//! - **Runtime Helpers**: policy-aware request/upstream body reads
//! - **Error Handling**: automatic error masking for production environments
//! - **Type Safety**: strong typing with Axum SSE primitives

mod runtime;
mod sse;
mod transcode_json;
mod transcode_sse;

pub use crate::bridge::{
    ClosureBridgeCustomization, ClosurePrimitiveRemapper, ClosureRequestBridgeHook,
    ClosureResponseBridgeHook, ClosureStreamBridgeHook, request_bridge_hook, response_bridge_hook,
    stream_bridge_hook,
};
pub use crate::server::GatewayBridgePolicy;
pub use runtime::{
    GatewayBodyReadError, GatewayBodyRole, read_request_body_with_policy,
    read_request_json_with_policy, read_upstream_body_with_policy, read_upstream_json_with_policy,
};
pub use sse::{SseOptions, to_sse_response, to_text_stream};
pub use transcode_json::{
    TargetJsonFormat, TranscodeJsonOptions, to_transcoded_json_response,
    to_transcoded_json_response_with_response_transform,
    to_transcoded_json_response_with_transform, transcode_chat_response_to_json,
};

pub use transcode_sse::{
    TargetSseFormat, TranscodeSseOptions, to_transcoded_sse_response,
    to_transcoded_sse_response_with_transform,
};

#[cfg(feature = "openai")]
pub use transcode_sse::{
    to_openai_chat_completions_sse_response, to_openai_chat_completions_sse_response_with_options,
    to_openai_responses_sse_response, to_openai_responses_sse_response_with_options,
    to_openai_responses_sse_stream, to_openai_responses_sse_stream_with_options,
};

#[cfg(feature = "anthropic")]
pub use transcode_sse::{
    to_anthropic_messages_sse_response, to_anthropic_messages_sse_response_with_options,
};

#[cfg(feature = "google")]
pub use transcode_sse::{
    to_gemini_generate_content_sse_response, to_gemini_generate_content_sse_response_with_options,
};
