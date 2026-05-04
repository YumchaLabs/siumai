//! Text model family contracts.
//!
//! This module provides a Rust-first, family-oriented abstraction for text generation
//! and streaming. In V3-M2 it is intentionally implemented as an adapter over the
//! existing `ChatCapability` so we can ship the new surface quickly, then iterate
//! towards a fully decoupled “text-first” foundation.

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle, LanguageModelV4StreamPart};
use crate::traits::ChatCapability;
use crate::traits::ModelMetadata;
use crate::types::{
    ChatRequest, ChatResponse, LanguageModelV4CallOptions, LanguageModelV4GenerateResult,
    LanguageModelV4StreamResult,
};
use crate::utils::SupportedUrlMap;

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

/// Canonical Rust stream carrier for AI SDK V4 provider stream parts.
///
/// AI SDK models return a JavaScript `ReadableStream<LanguageModelV4StreamPart>`. Rust keeps the
/// stream transport explicit and uses `Result` for transport/runtime errors while still allowing
/// providers to emit V4 `error` parts as data when they are part of the model stream.
pub type LanguageModelV4Stream =
    Pin<Box<dyn Stream<Item = Result<LanguageModelV4StreamPart, LlmError>> + Send + 'static>>;

/// Concrete Rust return type for `LanguageModelV4::do_stream`.
pub type LanguageModelV4DoStreamResult = LanguageModelV4StreamResult<LanguageModelV4Stream>;

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

/// AI SDK V4 provider-facing language-model contract.
///
/// This mirrors `LanguageModelV4` from `@ai-sdk/provider`: model identity plus supported URL
/// patterns and the raw provider `doGenerate` / `doStream` calls over V4 call/result structures.
/// It intentionally does not replace the stable Rust `LanguageModel` helper trait, which remains
/// the ergonomic text-family runtime used by high-level helpers.
#[async_trait]
pub trait LanguageModelV4: ModelMetadata + Send + Sync {
    /// AI SDK provider interface version implemented by this model.
    ///
    /// This method avoids the `ModelMetadata::specification_version()` name because that existing
    /// Rust method tracks Siumai's model-family trait version, not the upstream provider contract.
    fn language_model_v4_specification_version(&self) -> &'static str {
        "v4"
    }

    /// Supported URL patterns by media type for native provider-side fetch support.
    ///
    /// Providers that do not advertise native URL handling should return an empty map. The map uses
    /// the same media-type wildcard semantics as AI SDK `supportedUrls` and is checked with
    /// `is_url_supported(...)`.
    async fn supported_urls(&self) -> Result<SupportedUrlMap, LlmError> {
        Ok(SupportedUrlMap::new())
    }

    /// Generate a non-streaming V4 provider result.
    async fn do_generate(
        &self,
        options: LanguageModelV4CallOptions,
    ) -> Result<LanguageModelV4GenerateResult, LlmError>;

    /// Generate a streaming V4 provider result.
    async fn do_stream(
        &self,
        options: LanguageModelV4CallOptions,
    ) -> Result<LanguageModelV4DoStreamResult, LlmError>;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::ChatStream;
    use crate::traits::ModelSpecVersion;
    use crate::types::{
        ChatMessage, ChatStreamEvent, FinishReason, LanguageModelV4Text, LanguageModelV4Usage,
        MessageContent,
    };
    use crate::utils::UrlSupportRegex;
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
            let event = ev.unwrap();
            if let Some(delta) = event.text_delta() {
                acc.push_str(delta);
            }
            if let ChatStreamEvent::StreamEnd { response } = event {
                assert_eq!(response.content_text(), Some("ok"));
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

    struct FakeLanguageModelV4;

    impl crate::traits::ModelMetadata for FakeLanguageModelV4 {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-v4"
        }
    }

    #[async_trait]
    impl LanguageModelV4 for FakeLanguageModelV4 {
        async fn supported_urls(&self) -> Result<SupportedUrlMap, LlmError> {
            Ok(SupportedUrlMap::from([(
                "image/*".to_string(),
                vec![UrlSupportRegex::new(r"^https://images\.example\.com/").unwrap()],
            )]))
        }

        async fn do_generate(
            &self,
            _options: LanguageModelV4CallOptions,
        ) -> Result<LanguageModelV4GenerateResult, LlmError> {
            Ok(LanguageModelV4GenerateResult::new(
                vec![LanguageModelV4Text::new("ok").into()],
                FinishReason::Stop,
                LanguageModelV4Usage::default(),
            ))
        }

        async fn do_stream(
            &self,
            _options: LanguageModelV4CallOptions,
        ) -> Result<LanguageModelV4DoStreamResult, LlmError> {
            Ok(LanguageModelV4StreamResult::new(
                Box::pin(futures::stream::empty()) as LanguageModelV4Stream,
            ))
        }
    }

    #[tokio::test]
    async fn language_model_v4_trait_exposes_provider_contract() {
        let model = FakeLanguageModelV4;
        let dyn_model: &dyn LanguageModelV4 = &model;

        assert_eq!(dyn_model.language_model_v4_specification_version(), "v4");
        assert_eq!(dyn_model.provider_id(), "fake");
        assert_eq!(dyn_model.model_id(), "fake-v4");
        assert_eq!(
            dyn_model
                .supported_urls()
                .await
                .expect("supported urls")
                .len(),
            1
        );

        let options = LanguageModelV4CallOptions::from_model_messages(vec![
            ChatMessage::user("hi")
                .build()
                .try_into()
                .expect("model message"),
        ]);
        let result = dyn_model
            .do_generate(options.clone())
            .await
            .expect("generate");
        assert_eq!(result.content.len(), 1);

        let stream_result = dyn_model.do_stream(options).await.expect("stream");
        let parts = stream_result.stream.collect::<Vec<_>>().await;
        assert!(parts.is_empty());
    }
}
