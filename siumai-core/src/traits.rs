//! Core Trait Definitions
//!
//! This module defines all capability traits for LLM providers, organized by functionality.
//!
//! ## Module Organization
//!
//! Traits are organized in separate files under `traits/` and re-exported here for a stable API:
//!
//! - **`chat`** - Chat completion capabilities (`ChatCapability`, `ChatExtensions`)
//! - **`embedding`** - Embedding generation capabilities (`EmbeddingCapability`, `EmbeddingExtensions`)
//! - **`vision`** - Vision/image analysis capabilities (`VisionCapability`)
//! - **`image`** - Image generation capabilities (`ImageGenerationCapability`)
//! - **`audio`** - Audio transcription/generation capabilities (`AudioCapability`)
//! - **`files`** - File management capabilities (`FileManagementCapability`)
//! - **`moderation`** - Content moderation capabilities (`ModerationCapability`)
//! - **`rerank`** - Document reranking capabilities (`RerankCapability`)
//!
//! ## Usage Guidelines
//!
//! ### For Application Developers
//!
//! Import traits from the prelude (recommended) or directly from this module:
//!
//! ```rust
//! use siumai::prelude::*;  // Recommended - imports all common traits
//! ```
//!
//! Or import specific traits:
//!
//! ```rust
//! use siumai::traits::{ChatCapability, EmbeddingCapability};
//! ```
//!
//! Traits are also re-exported from `core` for convenience:
//!
//! ```rust
//! use siumai::core::{ChatCapability, EmbeddingCapability};
//! ```
//!
//! ### Trait Hierarchy
//!
//! #### Core Capability Traits
//! - `ChatCapability` - Basic chat completion
//! - `EmbeddingCapability` - Vector embeddings
//! - `VisionCapability` - Image analysis and generation
//! - `AudioCapability` - Audio processing
//! - `ImageGenerationCapability` - Image generation
//!
//! #### Extension Traits
//! - `ChatExtensions` - Convenience methods for chat (auto-implemented)
//! - `EmbeddingExtensions` - Convenience methods for embeddings (auto-implemented)
//!
//! ## Design Principles
//!
//! 1. **Capability-based**: Each trait represents a specific capability
//! 2. **Composable**: Providers can implement multiple traits
//! 3. **Async-first**: All methods are async for non-blocking I/O
//! 4. **Send + Sync**: All traits are thread-safe for concurrent usage
//! 5. **Extension traits**: Provide convenience methods without breaking changes
//!
//! ## Example: Using Capabilities
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//!
//! async fn example(client: impl ChatCapability + EmbeddingCapability) -> Result<(), LlmError> {
//!     // Use chat capability
//!     let response = client.chat(vec![user!("Hello!")]).await?;
//!
//!     // Use embedding capability
//!     let embeddings = client.embed(vec!["Hello world".to_string()]).await?;
//!
//!     Ok(())
//! }
//! ```

// Re-export modular traits
mod chat;
pub use chat::{ChatCapability, ChatExtensions};

mod embedding;
pub use embedding::{EmbeddingCapability, EmbeddingExtensions};

mod image;
pub use image::ImageGenerationCapability;

mod vision;
#[allow(deprecated)]
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

mod audio;
pub use audio::AudioCapability;

mod speech;
pub use speech::SpeechCapability;

mod transcription;
pub use transcription::TranscriptionCapability;

mod video;
pub use video::VideoGenerationCapability;

mod music;
pub use music::MusicGenerationCapability;

/// Core provider trait for capability discovery and metadata
pub trait LlmProvider: Send + Sync {
    /// Canonical provider id (e.g., "openai")
    fn provider_id(&self) -> std::borrow::Cow<'static, str>;
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
    #[allow(deprecated)]
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
                    system_fingerprint: None,
                    service_tier: None,
                    audio: None,
                    warnings: None,
                    provider_metadata: None,
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
