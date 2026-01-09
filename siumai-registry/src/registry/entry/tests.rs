#![allow(clippy::await_holding_lock)]

use super::*;
use crate::execution::http::interceptor::LoggingInterceptor;
use std::sync::{Mutex, OnceLock};

// Serialize registry tests that mutate global TEST_BUILD_COUNT to avoid flakiness under parallel runs
static REG_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn reg_test_guard() -> std::sync::MutexGuard<'static, ()> {
    REG_TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

#[derive(Clone)]
#[allow(dead_code)]
struct MockClient(std::sync::Arc<std::sync::Mutex<usize>>);

#[async_trait::async_trait]
impl ChatCapability for MockClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        *self.0.lock().unwrap() += 1;
        Ok(crate::types::ChatResponse::new(
            crate::types::MessageContent::Text("ok".to_string()),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation("mock stream".into()))
    }
}

impl LlmClient for MockClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("mock")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["mock-model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_chat()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct TestProvRerankClient;

#[async_trait::async_trait]
impl ChatCapability for TestProvRerankClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat not supported in TestProvRerankClient".into(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat stream not supported in TestProvRerankClient".into(),
        ))
    }
}

#[async_trait::async_trait]
impl crate::traits::RerankCapability for TestProvRerankClient {
    async fn rerank(
        &self,
        request: crate::types::RerankRequest,
    ) -> Result<crate::types::RerankResponse, LlmError> {
        Ok(crate::types::RerankResponse {
            id: "rerank-test".to_string(),
            results: request
                .documents
                .iter()
                .enumerate()
                .map(|(i, _)| crate::types::RerankResult {
                    document: None,
                    index: i as u32,
                    relevance_score: (request.documents.len() - i) as f64,
                })
                .collect(),
            tokens: crate::types::RerankTokenUsage {
                input_tokens: 1,
                output_tokens: 1,
            },
        })
    }
}

impl LlmClient for TestProvRerankClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_rerank")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["rerank-model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_rerank()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
    fn as_rerank_capability(&self) -> Option<&dyn crate::traits::RerankCapability> {
        Some(self)
    }
}

struct TestRerankProviderFactory;

#[async_trait::async_trait]
impl ProviderFactory for TestRerankProviderFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvRerankClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_rerank")
    }
}

#[tokio::test]
async fn language_model_handle_builds_client() {
    let _g = reg_test_guard();
    // Create registry with test provider factory
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov:model").unwrap();

    // First call builds a new client
    TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
    let resp = handle.chat(vec![]).await.unwrap();
    assert_eq!(resp.content_text().unwrap_or_default(), "ok");
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "First call should build a new client"
    );

    // Second call uses cached client (LRU cache)
    let resp = handle.chat(vec![]).await.unwrap();
    assert_eq!(resp.content_text().unwrap_or_default(), "ok");
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "Second call should use cached client"
    );
}

#[tokio::test]
async fn reranking_model_handle_builds_and_calls() {
    let _g = reg_test_guard();

    let mut providers = HashMap::new();
    providers.insert(
        "testprov_rerank".to_string(),
        Arc::new(TestRerankProviderFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: vec![Arc::new(LoggingInterceptor)],
            http_config: None,
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: false,
        }),
    );

    let handle = reg.reranking_model("testprov_rerank:rerank-model").unwrap();
    assert_eq!(handle.http_interceptors.len(), 1);

    let req = crate::types::RerankRequest::new(
        "rerank-model".to_string(),
        "query".to_string(),
        vec!["a".into(), "b".into(), "c".into()],
    );
    let resp = handle.rerank(req).await.unwrap();
    assert_eq!(resp.id, "rerank-test");
    assert_eq!(resp.results.len(), 3);
    assert_eq!(resp.top_result_index(), Some(0));
}

#[tokio::test]
async fn lru_cache_eviction() {
    let _g = reg_test_guard();
    // Create registry with small cache (2 entries)
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: Vec::new(),
            http_config: None,
            retry_options: None,
            max_cache_entries: Some(2),
            client_ttl: None,
            auto_middleware: false, // Disable for testing
        }),
    );

    TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);

    // Create 3 different handles
    let handle1 = reg.language_model("testprov:model1").unwrap();
    let handle2 = reg.language_model("testprov:model2").unwrap();
    let handle3 = reg.language_model("testprov:model3").unwrap();

    // Use handle1 and handle2 (cache: [model1, model2])
    handle1.chat(vec![]).await.unwrap();
    handle2.chat(vec![]).await.unwrap();
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        2
    );

    // Use handle3 (cache: [model2, model3], model1 evicted)
    handle3.chat(vec![]).await.unwrap();
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        3
    );

    // Use handle2 again (cache hit)
    handle2.chat(vec![]).await.unwrap();
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        3
    );

    // Use handle1 again (cache miss, model1 was evicted)
    handle1.chat(vec![]).await.unwrap();
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        4
    );
}

#[tokio::test]
async fn ttl_expiration() {
    let _g = reg_test_guard();
    use std::time::Duration;

    // Create registry with TTL of 100ms
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: Vec::new(),
            http_config: None,
            retry_options: None,
            max_cache_entries: None,
            client_ttl: Some(Duration::from_millis(100)),
            auto_middleware: false, // Disable for testing
        }),
    );

    TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);

    let handle = reg.language_model("testprov:model").unwrap();

    // First call builds client
    handle.chat(vec![]).await.unwrap();
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    // Second call uses cached client (within TTL)
    handle.chat(vec![]).await.unwrap();
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    // Wait for TTL to expire
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Third call rebuilds client (TTL expired)
    handle.chat(vec![]).await.unwrap();
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn language_model_inherits_registry_interceptors() {
    let _g = reg_test_guard();
    // Registry with a logging interceptor at registry level
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: vec![Arc::new(LoggingInterceptor)],
            http_config: None,
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: false,
        }),
    );
    let handle = reg.language_model("testprov:model").unwrap();
    // Assert handle picked up interceptors
    assert_eq!(handle.http_interceptors.len(), 1);
    // Ensure chat still works
    let _ = handle.chat(vec![]).await.unwrap();
}

#[tokio::test]
async fn embedding_and_image_handles_inherit_interceptors() {
    let _g = reg_test_guard();
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "testprov_embed".to_string(),
        Arc::new(TestProviderFactory::new("testprov_embed")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: vec![Arc::new(LoggingInterceptor)],
            http_config: None,
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: false,
        }),
    );
    let eh = reg.embedding_model("testprov_embed:model").unwrap();
    assert_eq!(eh.http_interceptors.len(), 1);
    // Call embed to ensure path runs without panic
    let _ = eh.embed(vec!["hello".into()]).await.unwrap();

    // Image handle still inherits vector even if TestProviderFactory returns chat-only client
    let ih = reg.image_model("testprov:model").unwrap();
    assert_eq!(ih.http_interceptors.len(), 1);
}

#[cfg(feature = "builtins")]
#[test]
fn create_registry_with_defaults_registers_native_factories() {
    let _g = reg_test_guard();
    let _reg = crate::registry::helpers::create_registry_with_defaults();

    // These checks validate that the default handle-level registry wiring
    // actually registers factory entries for the common native providers.
    // We deliberately stop at handle creation, so no API keys or network
    // access are required for this test.
    #[cfg(feature = "openai")]
    {
        assert!(_reg.language_model("openai:any-model").is_ok());
    }
    #[cfg(feature = "azure")]
    {
        assert!(_reg.language_model("azure:any-model").is_ok());
        assert!(_reg.language_model("azure-chat:any-model").is_ok());
    }
    #[cfg(feature = "google-vertex")]
    {
        assert!(_reg.language_model("anthropic-vertex:any-model").is_ok());
    }
    #[cfg(feature = "google")]
    {
        assert!(_reg.language_model("gemini:any-model").is_ok());
    }
    #[cfg(feature = "groq")]
    {
        assert!(_reg.language_model("groq:any-model").is_ok());
    }
    #[cfg(feature = "xai")]
    {
        assert!(_reg.language_model("xai:any-model").is_ok());
    }
    #[cfg(feature = "ollama")]
    {
        assert!(_reg.language_model("ollama:any-model").is_ok());
    }
    #[cfg(feature = "minimaxi")]
    {
        assert!(_reg.language_model("minimaxi:any-model").is_ok());
    }
}
