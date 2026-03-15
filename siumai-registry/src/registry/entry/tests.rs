#![allow(clippy::await_holding_lock)]

use super::*;
use crate::execution::http::interceptor::LoggingInterceptor;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// Serialize registry tests that mutate global TEST_BUILD_COUNT to avoid flakiness under parallel runs
static REG_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn reg_test_guard() -> std::sync::MutexGuard<'static, ()> {
    REG_TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedBuildContext {
    provider_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    has_http_client: bool,
    has_http_transport: bool,
    user_agent: Option<String>,
    has_retry_options: bool,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
}

#[derive(Clone, Default)]
struct NoopTransport;

#[async_trait::async_trait]
impl crate::execution::http::transport::HttpTransport for NoopTransport {
    async fn execute_json(
        &self,
        _request: crate::execution::http::transport::HttpTransportRequest,
    ) -> Result<crate::execution::http::transport::HttpTransportResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "noop transport should not execute requests".to_string(),
        ))
    }
}

struct ContextCapturingFactory {
    id: &'static str,
    seen: Arc<Mutex<Option<ObservedBuildContext>>>,
}

#[async_trait::async_trait]
impl ProviderFactory for ContextCapturingFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvClient))
    }

    async fn language_model_with_ctx(
        &self,
        _model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        *self.seen.lock().unwrap() = Some(ObservedBuildContext {
            provider_id: ctx.provider_id.clone(),
            api_key: ctx.api_key.clone(),
            base_url: ctx.base_url.clone(),
            has_http_client: ctx.http_client.is_some(),
            has_http_transport: ctx.http_transport.is_some(),
            user_agent: ctx
                .http_config
                .as_ref()
                .and_then(|config| config.user_agent.clone()),
            has_retry_options: ctx.retry_options.is_some(),
            reasoning_enabled: ctx.reasoning_enabled,
            reasoning_budget: ctx.reasoning_budget,
        });
        Ok(Arc::new(TestProvClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(self.id)
    }
}

#[cfg(any(feature = "google", feature = "google-vertex"))]
#[test]
fn build_context_resolves_google_token_provider_with_backward_compatibility() {
    let google: std::sync::Arc<dyn crate::auth::TokenProvider> =
        std::sync::Arc::new(crate::auth::StaticTokenProvider::new("google"));
    let gemini: std::sync::Arc<dyn crate::auth::TokenProvider> =
        std::sync::Arc::new(crate::auth::StaticTokenProvider::new("gemini"));

    let ctx = BuildContext {
        google_token_provider: Some(google.clone()),
        gemini_token_provider: Some(gemini.clone()),
        ..Default::default()
    };
    let resolved = ctx
        .resolved_google_token_provider()
        .expect("resolved provider");
    assert!(std::sync::Arc::ptr_eq(&resolved, &google));

    let legacy_ctx = BuildContext {
        google_token_provider: None,
        gemini_token_provider: Some(gemini.clone()),
        ..Default::default()
    };
    let legacy_resolved = legacy_ctx
        .resolved_google_token_provider()
        .expect("legacy resolved provider");
    assert!(std::sync::Arc::ptr_eq(&legacy_resolved, &gemini));
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

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
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
        let doc_len = request.documents_len();
        Ok(crate::types::RerankResponse {
            id: "rerank-test".to_string(),
            results: (0..doc_len)
                .map(|i| crate::types::RerankResult {
                    document: None,
                    index: i as u32,
                    relevance_score: (doc_len - i) as f64,
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

#[test]
fn language_model_handle_rejects_provider_without_chat_capability() {
    let _g = reg_test_guard();
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_embed".to_string(),
        Arc::new(TestProviderFactory::new("testprov_embed")) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let err = match reg.language_model("testprov_embed:model") {
        Ok(_) => panic!("language model handle should be rejected without chat capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("language_model/chat handles"))
    );
}

#[test]
fn language_model_handle_implements_model_metadata() {
    let _g = reg_test_guard();
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov:model").unwrap();

    fn assert_language_model<M>(model: &M)
    where
        M: siumai_core::text::LanguageModel + ?Sized,
    {
        assert_eq!(crate::traits::ModelMetadata::provider_id(model), "testprov");
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_language_model(&handle);
}

#[tokio::test]
async fn provider_factory_text_family_bridge_works() {
    let _g = reg_test_guard();
    TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);

    let factory = TestProviderFactory::new("testprov");
    let model = factory.language_model_text("bridged-model").await.unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-model"
    );

    let response = model
        .generate(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
        .await
        .unwrap();
    assert_eq!(response.content_text(), Some("ok"));
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "Bridge should reuse the existing language_model construction path"
    );
}

#[tokio::test]
async fn provider_factory_native_text_family_path_works() {
    let _g = reg_test_guard();

    #[derive(Clone)]
    struct NativeLanguageModel;

    impl crate::traits::ModelMetadata for NativeLanguageModel {
        fn provider_id(&self) -> &str {
            "native"
        }

        fn model_id(&self) -> &str {
            "native-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::text::TextModelV3 for NativeLanguageModel {
        async fn generate(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse::new(crate::types::MessageContent::Text(
                "native-ok".to_string(),
            )))
        }

        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "native stream not implemented in test".to_string(),
            ))
        }

        async fn stream_with_cancel(
            &self,
            _request: ChatRequest,
        ) -> Result<ChatStreamHandle, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "native stream_with_cancel not implemented in test".to_string(),
            ))
        }
    }

    struct NativeOnlyFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native text-family test")
        }

        async fn language_model_text_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::text::LanguageModel>, LlmError> {
            Ok(Arc::new(NativeLanguageModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
        }
    }

    let factory = NativeOnlyFactory;
    let model = factory.language_model_text("native-model").await.unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-model"
    );

    let response = model.generate(ChatRequest::new(vec![])).await.unwrap();
    assert_eq!(response.content_text(), Some("native-ok"));
}

#[tokio::test]
async fn language_model_handle_preserves_chat_request_fields() {
    let _g = reg_test_guard();

    #[derive(Clone)]
    struct CapturingClient(std::sync::Arc<std::sync::Mutex<Option<ChatRequest>>>);

    #[async_trait::async_trait]
    impl ChatCapability for CapturingClient {
        async fn chat_with_tools(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            Err(LlmError::UnsupportedOperation("use chat_request".into()))
        }

        async fn chat_stream(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::streaming::ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "use chat_stream_request".into(),
            ))
        }

        async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
            *self.0.lock().unwrap() = Some(request);
            Ok(ChatResponse::new(crate::types::MessageContent::Text(
                "ok".to_string(),
            )))
        }

        async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
            *self.0.lock().unwrap() = Some(request);
            let end = ChatResponse::new(crate::types::MessageContent::Text("ok".to_string()));
            let events = vec![Ok(crate::types::ChatStreamEvent::StreamEnd {
                response: end,
            })];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    impl LlmClient for CapturingClient {
        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cap")
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["bound-model".into()]
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

        fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
            Some(self)
        }
    }

    struct CaptureFactory {
        seen: std::sync::Arc<std::sync::Mutex<Option<ChatRequest>>>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for CaptureFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            Ok(Arc::new(CapturingClient(self.seen.clone())))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cap")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
        }
    }

    let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut providers = HashMap::new();
    providers.insert(
        "cap".to_string(),
        Arc::new(CaptureFactory { seen: seen.clone() }) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("cap:bound-model").unwrap();

    let request = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .temperature(0.7)
        .max_tokens(123)
        .build()
        .with_provider_option(
            "openai",
            serde_json::json!({ "responsesApi": { "enabled": true } }),
        );

    let _ = handle.chat_request(request.clone()).await.unwrap();

    let captured = seen.lock().unwrap().clone().unwrap();
    assert_eq!(
        captured.common_params.temperature,
        request.common_params.temperature
    );
    assert_eq!(
        captured.common_params.max_tokens,
        request.common_params.max_tokens
    );
    assert_eq!(
        captured.provider_options_map.get("openai"),
        request.provider_options_map.get("openai")
    );
    assert_eq!(captured.common_params.model, "bound-model");
    assert!(!captured.stream);

    let _ = handle
        .chat_stream_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
        .await
        .unwrap();
    let captured = seen.lock().unwrap().clone().unwrap();
    assert!(captured.stream);
    assert_eq!(captured.common_params.model, "bound-model");
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

#[test]
fn language_model_normalizes_common_provider_aliases() {
    let _g = reg_test_guard();

    let mut providers = HashMap::new();
    providers.insert(
        "gemini".to_string(),
        Arc::new(TestProviderFactory::new("gemini")) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "vertex".to_string(),
        Arc::new(TestProviderFactory::new("vertex")) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "anthropic-vertex".to_string(),
        Arc::new(TestProviderFactory::new("anthropic-vertex")) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    assert!(reg.language_model("google:any-model").is_ok());
    assert!(reg.language_model("google-vertex:any-model").is_ok());
    assert!(
        reg.language_model("google-vertex-anthropic:any-model")
            .is_ok()
    );
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
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides: HashMap::new(),
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

#[tokio::test]
async fn registry_builder_propagates_provider_build_overrides_to_language_model_handle() {
    let _g = reg_test_guard();

    let seen_specific = Arc::new(Mutex::new(None));
    let seen_global = Arc::new(Mutex::new(None));
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_ctx".to_string(),
        Arc::new(ContextCapturingFactory {
            id: "testprov_ctx",
            seen: seen_specific.clone(),
        }) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "testprov_global".to_string(),
        Arc::new(ContextCapturingFactory {
            id: "testprov_global",
            seen: seen_global.clone(),
        }) as Arc<dyn ProviderFactory>,
    );

    let mut http_config = crate::types::HttpConfig::default();
    http_config.user_agent = Some("registry-test-agent".to_string());

    let reg = crate::registry::builder::RegistryBuilder::new(providers)
        .with_http_client(reqwest::Client::new())
        .with_http_config(http_config)
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .with_reasoning(true)
        .with_reasoning_budget(1024)
        .with_provider_build_overrides(
            "testprov_ctx",
            crate::registry::ProviderBuildOverrides::default()
                .with_api_key("ctx-key")
                .with_base_url("https://example.com/custom")
                .with_reasoning(false)
                .with_reasoning_budget(2048)
                .fetch(Arc::new(NoopTransport)),
        )
        .with_retry_options(crate::retry_api::RetryOptions::default())
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let specific_handle = reg
        .language_model("testprov_ctx:model")
        .expect("build language handle");
    let global_handle = reg
        .language_model("testprov_global:model")
        .expect("build global language handle");

    let specific_response = specific_handle.chat(vec![]).await.expect("chat response");
    let global_response = global_handle.chat(vec![]).await.expect("chat response");
    assert_eq!(specific_response.content_text(), Some("ok"));
    assert_eq!(global_response.content_text(), Some("ok"));

    let specific_observed = seen_specific
        .lock()
        .unwrap()
        .clone()
        .expect("captured provider-specific build context");
    let global_observed = seen_global
        .lock()
        .unwrap()
        .clone()
        .expect("captured global build context");
    assert_eq!(
        specific_observed,
        ObservedBuildContext {
            provider_id: Some("testprov_ctx".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            has_http_client: true,
            has_http_transport: true,
            user_agent: Some("registry-test-agent".to_string()),
            has_retry_options: true,
            reasoning_enabled: Some(false),
            reasoning_budget: Some(2048),
        }
    );
    assert_eq!(
        global_observed,
        ObservedBuildContext {
            provider_id: Some("testprov_global".to_string()),
            api_key: Some("global-key".to_string()),
            base_url: Some("https://example.com/global".to_string()),
            has_http_client: true,
            has_http_transport: false,
            user_agent: Some("registry-test-agent".to_string()),
            has_retry_options: true,
            reasoning_enabled: Some(true),
            reasoning_budget: Some(1024),
        }
    );
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

#[tokio::test]
async fn language_model_handle_uses_native_family_path_when_available() {
    let _g = reg_test_guard();

    #[derive(Clone)]
    struct NativeLanguageModel;

    impl crate::traits::ModelMetadata for NativeLanguageModel {
        fn provider_id(&self) -> &str {
            "native-handle"
        }

        fn model_id(&self) -> &str {
            "native-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::text::TextModelV3 for NativeLanguageModel {
        async fn generate(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse::new(crate::types::MessageContent::Text(
                "native-handle-ok".to_string(),
            )))
        }

        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "stream not needed".to_string(),
            ))
        }

        async fn stream_with_cancel(
            &self,
            _request: ChatRequest,
        ) -> Result<ChatStreamHandle, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "stream_with_cancel not needed".to_string(),
            ))
        }
    }

    struct NativeHandleFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeHandleFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by language handle")
        }

        async fn language_model_text_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::text::LanguageModel>, LlmError> {
            Ok(Arc::new(NativeLanguageModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-handle".to_string(),
        Arc::new(NativeHandleFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("native-handle:model").unwrap();

    let response = handle.chat_request(ChatRequest::new(vec![])).await.unwrap();
    assert_eq!(response.content_text(), Some("native-handle-ok"));
}
