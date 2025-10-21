//! Core Trait Definitions (modular)
//!
//! Traits are organized under `traits/*` and re-exported here for a stable API.

// Re-export modular traits
mod chat;
pub use chat::{ChatCapability, ChatExtensions};

mod embedding;
pub use embedding::{EmbeddingCapability, EmbeddingExtensions};

mod image;
pub use image::ImageGenerationCapability;

mod vision;
pub use vision::VisionCapability;

mod files;
pub use files::FileManagementCapability;

mod moderation;
pub use moderation::ModerationCapability;

mod model_listing;
pub use model_listing::ModelListingCapability;

mod timeout;
pub use timeout::TimeoutCapability;

mod rerank;
pub use rerank::RerankCapability;

mod capabilities;
pub use capabilities::ProviderCapabilities;

mod provider_specific;
pub use provider_specific::{
    AnthropicCapability, GeminiCapability, GeminiEmbeddingCapability, OllamaEmbeddingCapability,
    OpenAiCapability, OpenAiEmbeddingCapability,
};

mod audio;
pub use audio::AudioCapability;

/// Core provider trait for capability discovery and metadata
pub trait LlmProvider: Send + Sync {
    fn provider_name(&self) -> &'static str;
    fn supported_models(&self) -> Vec<String>;
    fn capabilities(&self) -> ProviderCapabilities;
    fn http_client(&self) -> &reqwest::Client;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_capabilities() {
        let caps = ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_custom_feature("custom_feature", true);

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("custom_feature"));
        assert!(!caps.supports("audio"));
    }

    // Test that all capability traits are Send + Sync
    #[test]
    fn test_capability_traits_are_send_sync() {
        use std::sync::Arc;

        fn test_arc_usage() {
            let _: Option<Arc<dyn ChatCapability>> = None;
            let _: Option<Arc<dyn AudioCapability>> = None;
            let _: Option<Arc<dyn VisionCapability>> = None;
            let _: Option<Arc<dyn EmbeddingCapability>> = None;
            let _: Option<Arc<dyn ImageGenerationCapability>> = None;
            let _: Option<Arc<dyn FileManagementCapability>> = None;
            let _: Option<Arc<dyn ModerationCapability>> = None;
            let _: Option<Arc<dyn ModelListingCapability>> = None;
            let _: Option<Arc<dyn OpenAiCapability>> = None;
            let _: Option<Arc<dyn AnthropicCapability>> = None;
            let _: Option<Arc<dyn GeminiCapability>> = None;
        }

        test_arc_usage();
    }

    // Test actual multi-threading with capability traits
    #[tokio::test]
    async fn test_capability_traits_multithreading() {
        use crate::types::{ChatMessage, ChatResponse, MessageContent, Tool};
        use std::sync::Arc;
        use tokio::task;

        struct MockCapability;

        #[async_trait::async_trait]
        impl ChatCapability for MockCapability {
            async fn chat_with_tools(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, crate::error::LlmError> {
                Ok(ChatResponse {
                    id: Some("mock-id".to_string()),
                    content: MessageContent::Text("Mock response".to_string()),
                    model: Some("mock-model".to_string()),
                    usage: None,
                    finish_reason: Some(crate::types::FinishReason::Stop),
                    tool_calls: None,
                    thinking: None,
                    metadata: std::collections::HashMap::new(),
                })
            }

            async fn chat_stream(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<crate::streaming::ChatStream, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "Mock streaming not implemented".to_string(),
                ))
            }
        }

        let capability: Arc<dyn ChatCapability> = Arc::new(MockCapability);

        let mut handles = Vec::new();
        for i in 0..5 {
            let capability_clone = capability.clone();
            let handle = task::spawn(async move {
                let messages = vec![ChatMessage::user("Test message").build()];
                let result = capability_clone.chat_with_tools(messages, None).await;
                assert!(result.is_ok());
                i
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        assert_eq!(results.len(), 5);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i);
        }
    }
}
