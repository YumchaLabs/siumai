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
use crate::traits::ModelMetadata;
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

/// Stable language-model contract for the V4 refactor spike.
///
/// This trait intentionally builds on the existing text-family interface while adding
/// shared model metadata. It is a minimal bridge toward a family-model-centered design.
pub trait LanguageModel: TextModelV3 + ModelMetadata + Send + Sync {}

impl<T> LanguageModel for T where T: TextModelV3 + ModelMetadata + Send + Sync + ?Sized {}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::ChatStream;
    use crate::traits::ModelSpecVersion;
    use crate::types::{ChatMessage, ChatStreamEvent, MessageContent};
    use async_trait::async_trait;
    use futures::StreamExt;

    struct FakeChat;

    impl crate::traits::ModelMetadata for FakeChat {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-chat"
        }
    }

    #[async_trait]
    impl ChatCapability for FakeChat {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            Ok(crate::types::ChatResponse::new(MessageContent::Text(
                "ok".to_string(),
            )))
        }

        async fn chat_stream(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<ChatStream, LlmError> {
            let end = crate::types::ChatResponse::new(MessageContent::Text("ok".to_string()));
            let events = vec![
                Ok(ChatStreamEvent::ContentDelta {
                    delta: "o".to_string(),
                    index: None,
                }),
                Ok(ChatStreamEvent::ContentDelta {
                    delta: "k".to_string(),
                    index: None,
                }),
                Ok(ChatStreamEvent::StreamEnd { response: end }),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn adapter_generate_calls_chat_request() {
        let model = FakeChat;
        let resp = TextModelV3::generate(
            &model,
            ChatRequest::new(vec![ChatMessage::user("hi").build()]),
        )
        .await
        .unwrap();
        assert_eq!(resp.content_text(), Some("ok"));
    }

    #[tokio::test]
    async fn adapter_stream_calls_chat_stream_request() {
        let model = FakeChat;
        let mut stream = TextModelV3::stream(
            &model,
            ChatRequest::new(vec![ChatMessage::user("hi").build()]),
        )
        .await
        .unwrap();

        let mut acc = String::new();
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                ChatStreamEvent::ContentDelta { delta, .. } => acc.push_str(&delta),
                ChatStreamEvent::StreamEnd { response } => {
                    assert_eq!(response.content_text(), Some("ok"));
                }
                _ => {}
            }
        }
        assert_eq!(acc, "ok");
    }

    #[tokio::test]
    async fn adapter_stream_with_cancel_wraps_stream() {
        let model = FakeChat;
        let handle = TextModelV3::stream_with_cancel(
            &model,
            ChatRequest::new(vec![ChatMessage::user("hi").build()]),
        )
        .await
        .unwrap();

        let items: Vec<_> = handle.stream.collect().await;
        assert!(!items.is_empty());
    }

    #[test]
    fn language_model_marker_exposes_metadata() {
        fn assert_language_model<M: LanguageModel + ?Sized>(model: &M) {
            assert_eq!(model.provider_id(), "fake");
            assert_eq!(model.model_id(), "fake-chat");
            assert_eq!(model.specification_version(), ModelSpecVersion::V1);
        }

        let model = FakeChat;
        assert_language_model(&model);
    }
}
