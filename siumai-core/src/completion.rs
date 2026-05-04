//! Completion model family (V3).
//!
//! This module provides a Rust-first, family-oriented abstraction for completion endpoints such as
//! OpenAI/OpenAI-compatible `/completions`. The runtime streaming lane intentionally reuses the
//! shared language-model stream carrier rather than inventing a second incompatible stream model.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::traits::{CompletionCapability, ModelMetadata};
use crate::types::{CompletionRequest, CompletionResponse};

/// Canonical stream type for the completion family.
pub type CompletionStream = ChatStream;

/// Canonical stream handle type for the completion family.
pub type CompletionStreamHandle = ChatStreamHandle;

/// V3 interface for completion models.
#[async_trait]
pub trait CompletionModelV3: Send + Sync {
    /// Execute a non-streaming completion request.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    /// Execute a streaming completion request.
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, LlmError>;

    /// Execute a streaming completion request with cancellation support.
    async fn stream_with_cancel(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStreamHandle, LlmError>;
}

/// Stable completion-model contract for the V4 refactor spike.
pub trait CompletionModel: CompletionModelV3 + ModelMetadata + Send + Sync {}

impl<T> CompletionModel for T where T: CompletionModelV3 + ModelMetadata + Send + Sync + ?Sized {}

/// Adapter: any `CompletionCapability` can be used as a `CompletionModelV3`.
#[async_trait]
impl<T> CompletionModelV3 for T
where
    T: CompletionCapability + Send + Sync + ?Sized,
{
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        CompletionCapability::complete(self, request).await
    }

    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, LlmError> {
        CompletionCapability::complete_stream(self, request).await
    }

    async fn stream_with_cancel(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStreamHandle, LlmError> {
        CompletionCapability::complete_stream_with_cancel(self, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelSpecVersion;
    use crate::types::{ChatResponse, ChatStreamEvent, FinishReason};
    use futures::StreamExt;
    use std::collections::HashMap;

    struct FakeCompletion;

    impl crate::traits::ModelMetadata for FakeCompletion {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-completion"
        }
    }

    #[async_trait]
    impl CompletionCapability for FakeCompletion {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                text: request
                    .prompt
                    .first()
                    .and_then(|message| message.content_text())
                    .unwrap_or_default()
                    .to_string(),
                finish_reason: Some(FinishReason::Stop),
                raw_finish_reason: Some("stop".to_string()),
                usage: None,
                response_metadata: None,
                warnings: None,
                provider_metadata: Some(HashMap::new()),
            })
        }

        async fn complete_stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionStream, LlmError> {
            let events = vec![
                Ok(ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::TextDelta {
                        id: "0".to_string(),
                        delta: "o".to_string(),
                        provider_metadata: None,
                    },
                }),
                Ok(ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::TextDelta {
                        id: "0".to_string(),
                        delta: "k".to_string(),
                        provider_metadata: None,
                    },
                }),
                Ok(ChatStreamEvent::StreamEnd {
                    response: ChatResponse::empty_with_finish_reason(FinishReason::Stop),
                }),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn adapter_complete_uses_capability() {
        let model = FakeCompletion;
        let response = CompletionModelV3::complete(&model, CompletionRequest::new("hello"))
            .await
            .unwrap();
        assert_eq!(response.text(), "hello");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn adapter_stream_uses_capability() {
        let model = FakeCompletion;
        let mut stream = CompletionModelV3::stream(&model, CompletionRequest::new("hello"))
            .await
            .unwrap();

        let mut text = String::new();
        while let Some(event) = stream.next().await {
            let event = event.unwrap();
            if let Some(delta) = event.text_delta() {
                text.push_str(delta);
            }
            if let ChatStreamEvent::StreamEnd { response } = event {
                assert_eq!(response.finish_reason, Some(FinishReason::Stop));
            }
        }

        assert_eq!(text, "ok");
    }

    #[tokio::test]
    async fn adapter_stream_with_cancel_wraps_stream() {
        let model = FakeCompletion;
        let handle = CompletionModelV3::stream_with_cancel(&model, CompletionRequest::new("hello"))
            .await
            .unwrap();

        let items: Vec<_> = handle.stream.collect().await;
        assert!(!items.is_empty());
    }

    #[test]
    fn completion_model_trait_includes_metadata() {
        let model = FakeCompletion;

        fn assert_completion_model<M>(model: &M)
        where
            M: CompletionModel + ?Sized,
        {
            assert_eq!(crate::traits::ModelMetadata::provider_id(model), "fake");
            assert_eq!(
                crate::traits::ModelMetadata::model_id(model),
                "fake-completion"
            );
            assert_eq!(
                crate::traits::ModelMetadata::specification_version(model),
                ModelSpecVersion::V1
            );
        }

        assert_completion_model(&model);
    }
}
