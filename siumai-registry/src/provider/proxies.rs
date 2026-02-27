use super::Siumai;
use crate::client::LlmClient;
use crate::core::EmbeddingCapability;
use crate::error::LlmError;
use crate::types::EmbeddingResponse;

/// Type-safe proxy for audio capabilities
pub struct AudioCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> AudioCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports audio support (for reference only)
    ///
    /// Note: This is based on static capability information and may not reflect
    /// the actual capabilities of the current model. Use as a hint, not a restriction.
    /// The library will never block operations based on this information.
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider id for debugging
    pub fn provider_id(&self) -> String {
        self.provider.provider_id().into_owned()
    }

    /// Get a support status message (optional, for user-controlled warnings)
    ///
    /// Returns a message about support status that you can choose to display or ignore.
    /// The library itself will not automatically warn or log anything.
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports audio support", self.provider_id())
        } else {
            format!(
                "Provider {} does not report audio support, but this may still work depending on the model",
                self.provider_id()
            )
        }
    }

    /// Placeholder for future audio operations
    ///
    /// This will attempt the operation regardless of reported support.
    /// Actual errors will come from the API if the model doesn't support it.
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Audio operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

/// Type-safe proxy for embedding capabilities
pub struct EmbeddingCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> EmbeddingCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports embedding support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider id for debugging
    pub fn provider_id(&self) -> String {
        self.provider.provider_id().into_owned()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports embedding support", self.provider_id())
        } else {
            format!(
                "Provider {} does not report embedding support, but this may still work depending on the model",
                self.provider_id()
            )
        }
    }

    /// Generate embeddings for the given input texts
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.provider.embed(texts).await
    }

    /// Get the dimension of embeddings produced by this provider
    pub fn embedding_dimension(&self) -> usize {
        self.provider.embedding_dimension()
    }

    /// Get the maximum number of tokens that can be embedded at once
    pub fn max_tokens_per_embedding(&self) -> usize {
        self.provider.max_tokens_per_embedding()
    }

    /// Get supported embedding models for this provider
    pub fn supported_embedding_models(&self) -> Vec<String> {
        self.provider.supported_embedding_models()
    }

    // removed deprecated placeholder_operation; use embed() instead
}

/// Type-safe proxy for vision capabilities
#[deprecated(
    since = "0.11.0-beta.5",
    note = "VisionCapabilityProxy is deprecated; use multimodal Chat messages for image understanding, and ImageGenerationCapability for image generation."
)]
pub struct VisionCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

#[allow(deprecated)]
impl<'a> VisionCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports vision support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider id for debugging
    pub fn provider_id(&self) -> String {
        self.provider.provider_id().into_owned()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports vision support", self.provider_id())
        } else {
            format!(
                "Provider {} does not report vision support, but this may still work depending on the model",
                self.provider_id()
            )
        }
    }

    /// Placeholder for future vision operations
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Vision operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

// Debug/Default impl moved to provider/siumai_builder.rs

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::client::LlmClient;
    use crate::streaming::ChatStream;
    use crate::traits::*;
    use crate::types::*;
    use async_trait::async_trait;
    use std::borrow::Cow;
    use std::sync::Arc;

    // Mock provider for testing that doesn't support embedding
    #[derive(Debug)]
    struct MockProvider;

    #[async_trait]
    impl ChatCapability for MockProvider {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                id: Some("mock-123".to_string()),
                content: MessageContent::Text("Mock response".to_string()),
                model: Some("mock-model".to_string()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
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
        ) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "Streaming not supported in mock".to_string(),
            ))
        }
    }

    impl LlmClient for MockProvider {
        fn provider_id(&self) -> Cow<'static, str> {
            Cow::Borrowed("mock")
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["mock-model".to_string()]
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
            // Note: not adding .with_embedding() to test unsupported case
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(MockProvider)
        }
    }

    #[tokio::test]
    async fn test_siumai_embedding_unsupported_provider() {
        // Create a mock provider that doesn't support embedding
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Arc::new(mock_provider));

        // Test that embedding returns an error for unsupported provider
        let result = siumai.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_embedding_capability_proxy() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Arc::new(mock_provider));

        let proxy = siumai.embedding_capability();
        assert_eq!(proxy.provider_id(), "mock");
        assert!(!proxy.is_reported_as_supported()); // Mock provider doesn't report embedding support
    }

    #[tokio::test]
    async fn test_embedding_capability_proxy_embed() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Arc::new(mock_provider));

        let proxy = siumai.embedding_capability();
        let result = proxy.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    async fn test_ollama_build_without_api_key() {
        // Test that Ollama can be built without API key
        let result = SiumaiBuilder::new()
            .ollama()
            .model("llama3.2")
            .build()
            .await;

        // This should not fail due to missing API key
        // Note: It might fail for other reasons (like Ollama not running), but not API key
        match result {
            Ok(_) => {
                // Success - Ollama client was created without API key
            }
            Err(LlmError::ConfigurationError(msg)) => {
                // Should not be an API key error
                assert!(
                    !msg.contains("API key not specified"),
                    "Ollama should not require API key, but got: {}",
                    msg
                );
            }
            Err(_) => {
                // Other errors are acceptable (e.g., network issues)
            }
        }
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_openai_requires_api_key() {
        // Temporarily remove API key from environment
        let original_key = std::env::var("OPENAI_API_KEY").ok();
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
        }

        // Test that OpenAI still requires API key
        let result = SiumaiBuilder::new().openai().model("gpt-4o").build().await;

        // Restore original key if it existed
        if let Some(key) = original_key {
            unsafe {
                std::env::set_var("OPENAI_API_KEY", key);
            }
        }

        // This should fail due to missing API key
        assert!(result.is_err());
        if let Err(LlmError::ConfigurationError(msg)) = result {
            assert!(msg.contains("API key not specified"));
        } else {
            panic!("Expected ConfigurationError for missing API key");
        }
    }
}
