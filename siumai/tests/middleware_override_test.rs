use siumai::LlmClient;
use siumai::error::LlmError;
use siumai::middleware::language_model::LanguageModelMiddleware;
use siumai::registry::{ProviderFactory, RegistryOptions, create_provider_registry};
use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, ChatResponse, MessageContent};
use std::collections::HashMap;
use std::sync::Arc;

/// Mock provider factory for testing
struct MockProviderFactory {
    provider_name: &'static str,
}

#[async_trait::async_trait]
impl ProviderFactory for MockProviderFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Return a mock client that echoes the provider and model in the response
        Ok(Arc::new(MockClient {
            provider: self.provider_name.to_string(),
            model: model_id.to_string(),
        }))
    }

    async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "embedding not supported in mock".to_string(),
        ))
    }

    async fn image_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "image not supported in mock".to_string(),
        ))
    }

    async fn speech_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "speech not supported in mock".to_string(),
        ))
    }

    async fn transcription_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "transcription not supported in mock".to_string(),
        ))
    }

    fn provider_name(&self) -> &'static str {
        self.provider_name
    }
}

/// Mock client that echoes provider and model in response
#[derive(Clone)]
struct MockClient {
    provider: String,
    model: String,
}

#[async_trait::async_trait]
impl ChatCapability for MockClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<siumai::types::Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse::new(MessageContent::Text(format!(
            "provider={}, model={}",
            self.provider, self.model
        ))))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<siumai::types::Tool>>,
    ) -> Result<siumai::stream::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "stream not supported in mock".to_string(),
        ))
    }
}

impl LlmClient for MockClient {
    fn provider_name(&self) -> &'static str {
        "mock"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["mock-model".to_string()]
    }

    fn capabilities(&self) -> siumai::traits::ProviderCapabilities {
        siumai::traits::ProviderCapabilities::new().with_chat()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

/// Middleware that overrides model ID to "gpt-4"
struct OverrideModelMiddleware;

impl LanguageModelMiddleware for OverrideModelMiddleware {
    fn override_model_id(&self, _current: &str) -> Option<String> {
        Some("gpt-4".to_string())
    }
}

/// Middleware that overrides provider ID to "test-provider"
struct OverrideProviderMiddleware;

impl LanguageModelMiddleware for OverrideProviderMiddleware {
    fn override_provider_id(&self, _current: &str) -> Option<String> {
        Some("test-provider".to_string())
    }
}

#[tokio::test]
async fn test_model_id_override() {
    // Setup registry with mock provider
    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(MockProviderFactory {
            provider_name: "openai",
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: vec![Arc::new(OverrideModelMiddleware)],
            max_cache_entries: None,
            client_ttl: None,
        }),
    );

    // Request "openai:gpt-3.5-turbo" but middleware should override to "gpt-4"
    let handle = registry.language_model("openai:gpt-3.5-turbo").unwrap();

    let messages = vec![ChatMessage::user("test").build()];

    let response = handle.chat_with_tools(messages, None).await.unwrap();
    let text = response.content_text().unwrap();

    // Should use overridden model "gpt-4"
    assert_eq!(text, "provider=openai, model=gpt-4");
}

#[tokio::test]
async fn test_no_override() {
    // Setup registry without middleware
    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(MockProviderFactory {
            provider_name: "openai",
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(providers, None);

    // Request "openai:gpt-3.5-turbo" without override
    let handle = registry.language_model("openai:gpt-3.5-turbo").unwrap();

    let messages = vec![ChatMessage::user("test").build()];

    let response = handle.chat_with_tools(messages, None).await.unwrap();
    let text = response.content_text().unwrap();

    // Should use original model "gpt-3.5-turbo"
    assert_eq!(text, "provider=openai, model=gpt-3.5-turbo");
}

#[tokio::test]
async fn test_multiple_middlewares_first_override_wins() {
    // Setup registry with multiple middlewares
    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(MockProviderFactory {
            provider_name: "openai",
        }) as Arc<dyn ProviderFactory>,
    );

    struct FirstOverride;
    impl LanguageModelMiddleware for FirstOverride {
        fn override_model_id(&self, _current: &str) -> Option<String> {
            Some("first-model".to_string())
        }
    }

    struct SecondOverride;
    impl LanguageModelMiddleware for SecondOverride {
        fn override_model_id(&self, _current: &str) -> Option<String> {
            Some("second-model".to_string())
        }
    }

    let registry = create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: vec![
                Arc::new(FirstOverride),
                Arc::new(SecondOverride), // This should be ignored
            ],
            max_cache_entries: None,
            client_ttl: None,
        }),
    );

    let handle = registry.language_model("openai:original-model").unwrap();

    let messages = vec![ChatMessage::user("test").build()];

    let response = handle.chat_with_tools(messages, None).await.unwrap();
    let text = response.content_text().unwrap();

    // First middleware's override should win
    assert_eq!(text, "provider=openai, model=first-model");
}
