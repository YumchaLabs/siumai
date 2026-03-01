//! Text model family (V3).
//!
//! This module provides a Rust-first, family-oriented abstraction for text generation
//! and streaming. In V3-M2 it is intentionally implemented as an adapter over the
//! existing `ChatCapability` so we can ship the new surface quickly, then iterate
//! towards a fully decoupled “text-first” foundation.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::traits::ChatCapability;
use crate::types::{ChatRequest, ChatResponse};

/// Canonical request type for the text family.
///
/// For now this is an alias of `ChatRequest`. It will evolve as the “text family”
/// becomes independent from chat-centric naming.
pub type TextRequest = ChatRequest;

/// Canonical response type for the text family.
pub type TextResponse = ChatResponse;

/// Canonical stream type for the text family.
pub type TextStream = ChatStream;

/// Canonical stream handle type for the text family.
pub type TextStreamHandle = ChatStreamHandle;

/// V3 interface for text generation models.
#[async_trait]
pub trait TextModelV3: Send + Sync {
    /// Generate a non-streaming response.
    async fn generate(&self, request: TextRequest) -> Result<TextResponse, LlmError>;

    /// Generate a streaming response.
    async fn stream(&self, request: TextRequest) -> Result<TextStream, LlmError>;

    /// Generate a streaming response with a first-class cancellation handle.
    async fn stream_with_cancel(&self, request: TextRequest) -> Result<TextStreamHandle, LlmError>;
}

/// Adapter: any `ChatCapability` can be used as a `TextModelV3`.
#[async_trait]
impl<T> TextModelV3 for T
where
    T: ChatCapability + Send + Sync + ?Sized,
{
    async fn generate(&self, request: TextRequest) -> Result<TextResponse, LlmError> {
        self.chat_request(request).await
    }

    async fn stream(&self, request: TextRequest) -> Result<TextStream, LlmError> {
        self.chat_stream_request(request).await
    }

    async fn stream_with_cancel(&self, request: TextRequest) -> Result<TextStreamHandle, LlmError> {
        self.chat_stream_request_with_cancel(request).await
    }
}
