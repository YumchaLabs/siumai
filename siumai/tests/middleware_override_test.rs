use siumai::experimental::client::LlmClient;
use siumai::experimental::execution::middleware::language_model::LanguageModelMiddleware;
use siumai::prelude::unified::registry::{
    ProviderFactory, RegistryOptions, create_provider_registry,
};
use siumai::prelude::unified::{
    ChatCapability, ChatMessage, ChatResponse, ChatStream, LlmError, MessageContent,
    ProviderCapabilities, Tool,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Mock provider factory for testing
struct MockProviderFactory {
    provider_id: &'static str,
}

#[async_trait::async_trait]
impl ProviderFactory for MockProviderFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Return a mock client that echoes the provider and model in the response
        Ok(Arc::new(MockClient {
            provider: self.provider_id.to_string(),
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

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(self.provider_id)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }
}

/// Counting provider factory for cache/TTL invariant tests
struct CountingProviderFactory {
    provider_id: &'static str,
    build_count: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl ProviderFactory for CountingProviderFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.build_count.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(MockClient {
            provider: self.provider_id.to_string(),
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

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(self.provider_id)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
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
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse::new(MessageContent::Text(format!(
            "provider={}, model={}",
            self.provider, self.model
        ))))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "stream not supported in mock".to_string(),
        ))
    }
}

impl LlmClient for MockClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("mock")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["mock-model".to_string()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
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
#[allow(dead_code)]
struct OverrideProviderMiddleware;

impl LanguageModelMiddleware for OverrideProviderMiddleware {
    fn override_provider_id(&self, _current: &str) -> Option<String> {
        Some("test-provider".to_string())
    }
}

fn registry_options(
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    auto_middleware: bool,
    client_ttl: Option<std::time::Duration>,
) -> RegistryOptions {
    RegistryOptions {
        separator: ':',
        language_model_middleware: middlewares,
        http_interceptors: Vec::new(),
        http_client: None,
        http_transport: None,
        http_config: None,
        api_key: None,
        base_url: None,
        reasoning_enabled: None,
        reasoning_budget: None,
        provider_build_overrides: HashMap::new(),
        retry_options: None,
        max_cache_entries: None,
        client_ttl,
        auto_middleware,
    }
}

#[tokio::test]
async fn test_model_id_override() {
    // Setup registry with mock provider
    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(MockProviderFactory {
            provider_id: "openai",
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(
        providers,
        Some(registry_options(
            vec![Arc::new(OverrideModelMiddleware)],
            true,
            None,
        )),
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
            provider_id: "openai",
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
            provider_id: "openai",
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
        Some(registry_options(
            vec![
                Arc::new(FirstOverride),
                Arc::new(SecondOverride), // This should be ignored
            ],
            true,
            None,
        )),
    );

    let handle = registry.language_model("openai:original-model").unwrap();

    let messages = vec![ChatMessage::user("test").build()];

    let response = handle.chat_with_tools(messages, None).await.unwrap();
    let text = response.content_text().unwrap();

    // First middleware's override should win
    assert_eq!(text, "provider=openai, model=first-model");
}

#[tokio::test]
async fn test_model_override_reuses_cache_key_for_overridden_model() {
    let openai_builds = Arc::new(AtomicUsize::new(0));

    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(CountingProviderFactory {
            provider_id: "openai",
            build_count: openai_builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(
        providers,
        Some(registry_options(
            vec![Arc::new(OverrideModelMiddleware)],
            false,
            None,
        )),
    );

    let handle_a = registry.language_model("openai:gpt-3.5-turbo").unwrap();
    let handle_b = registry.language_model("openai:gpt-4o-mini").unwrap();
    let messages = vec![ChatMessage::user("cache").build()];

    let text_a = handle_a
        .chat_with_tools(messages.clone(), None)
        .await
        .unwrap()
        .content_text()
        .unwrap()
        .to_string();
    let text_b = handle_b
        .chat_with_tools(messages, None)
        .await
        .unwrap()
        .content_text()
        .unwrap()
        .to_string();

    assert_eq!(text_a, "provider=openai, model=gpt-4");
    assert_eq!(text_b, "provider=openai, model=gpt-4");
    assert_eq!(openai_builds.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_provider_override_routes_and_reuses_overridden_provider_cache_key() {
    let openai_builds = Arc::new(AtomicUsize::new(0));
    let target_builds = Arc::new(AtomicUsize::new(0));

    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(CountingProviderFactory {
            provider_id: "openai",
            build_count: openai_builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "test-provider".to_string(),
        Arc::new(CountingProviderFactory {
            provider_id: "test-provider",
            build_count: target_builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(
        providers,
        Some(registry_options(
            vec![Arc::new(OverrideProviderMiddleware)],
            false,
            None,
        )),
    );

    let overridden = registry.language_model("openai:gpt-4").unwrap();
    let direct = registry.language_model("test-provider:gpt-4").unwrap();
    let messages = vec![ChatMessage::user("provider override").build()];

    let overridden_text = overridden
        .chat_with_tools(messages.clone(), None)
        .await
        .unwrap()
        .content_text()
        .unwrap()
        .to_string();
    let direct_text = direct
        .chat_with_tools(messages, None)
        .await
        .unwrap()
        .content_text()
        .unwrap()
        .to_string();

    assert_eq!(overridden_text, "provider=test-provider, model=gpt-4");
    assert_eq!(direct_text, "provider=test-provider, model=gpt-4");
    assert_eq!(openai_builds.load(Ordering::SeqCst), 0);
    assert_eq!(target_builds.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_model_override_ttl_rebuilds_after_expiration() {
    let openai_builds = Arc::new(AtomicUsize::new(0));

    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(CountingProviderFactory {
            provider_id: "openai",
            build_count: openai_builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(
        providers,
        Some(registry_options(
            vec![Arc::new(OverrideModelMiddleware)],
            false,
            Some(std::time::Duration::from_millis(50)),
        )),
    );

    let handle = registry.language_model("openai:gpt-3.5-turbo").unwrap();
    let messages = vec![ChatMessage::user("ttl").build()];

    let first = handle
        .chat_with_tools(messages.clone(), None)
        .await
        .unwrap()
        .content_text()
        .unwrap()
        .to_string();
    let second = handle
        .chat_with_tools(messages.clone(), None)
        .await
        .unwrap()
        .content_text()
        .unwrap()
        .to_string();
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let third = handle
        .chat_with_tools(messages, None)
        .await
        .unwrap()
        .content_text()
        .unwrap()
        .to_string();

    assert_eq!(first, "provider=openai, model=gpt-4");
    assert_eq!(second, "provider=openai, model=gpt-4");
    assert_eq!(third, "provider=openai, model=gpt-4");
    assert_eq!(openai_builds.load(Ordering::SeqCst), 2);
}
