//! Transformers for Google Gemini (protocol layer)
//!
//! Centralizes request/response transformation for Gemini to reduce duplication
//! between non-streaming and streaming paths.

use crate::error::LlmError;
use crate::execution::transformers::files::{FilesHttpBody, FilesTransformer};
use crate::execution::transformers::request::{
    GenericRequestTransformer, MappingProfile, ProviderRequestHooks, RangeMode, Rule,
};
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::streaming::SseEventConverter;
use crate::types::EmbeddingRequest;
use crate::types::ImageGenerationRequest;
use crate::types::{ChatRequest, ChatResponse, ContentPart, FinishReason, MessageContent, Usage};
use eventsource_stream::Event;
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;

use super::types::{CreateFileResponse, GeminiFile, GeminiFileState, ListFilesResponse};
use super::types::{GeminiConfig, GenerateContentRequest, GenerateContentResponse, Part};
// No longer depend on chat capability for request construction

use super::convert;
use super::streaming;
use super::types;

use options::gemini_options_from_request;

mod files;
mod options;
mod request;
mod response;
mod stream;

pub use files::GeminiFilesTransformer;
pub use request::GeminiRequestTransformer;
pub use response::GeminiResponseTransformer;
pub use stream::GeminiStreamChunkTransformer;

#[cfg(test)]
mod tests;
