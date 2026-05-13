use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn lru_cache_eviction() {
    let _g = reg_test_guard();
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
            auto_middleware: false,
        }),
    );

    TEST_BUILD_COUNT.store(0, Ordering::SeqCst);

    let handle1 = reg.language_model("testprov:model1").unwrap();
    let handle2 = reg.language_model("testprov:model2").unwrap();
    let handle3 = reg.language_model("testprov:model3").unwrap();

    handle1.chat(vec![]).await.unwrap();
    handle2.chat(vec![]).await.unwrap();
    assert_eq!(TEST_BUILD_COUNT.load(Ordering::SeqCst), 2);

    handle3.chat(vec![]).await.unwrap();
    assert_eq!(TEST_BUILD_COUNT.load(Ordering::SeqCst), 3);

    handle2.chat(vec![]).await.unwrap();
    assert_eq!(TEST_BUILD_COUNT.load(Ordering::SeqCst), 3);

    handle1.chat(vec![]).await.unwrap();
    assert_eq!(TEST_BUILD_COUNT.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn ttl_expiration() {
    let _g = reg_test_guard();
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
            client_ttl: Some(std::time::Duration::from_millis(100)),
            auto_middleware: false,
        }),
    );

    TEST_BUILD_COUNT.store(0, Ordering::SeqCst);

    let handle = reg.language_model("testprov:model").unwrap();

    handle.chat(vec![]).await.unwrap();
    assert_eq!(TEST_BUILD_COUNT.load(Ordering::SeqCst), 1);

    handle.chat(vec![]).await.unwrap();
    assert_eq!(TEST_BUILD_COUNT.load(Ordering::SeqCst), 1);

    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    handle.chat(vec![]).await.unwrap();
    assert_eq!(TEST_BUILD_COUNT.load(Ordering::SeqCst), 2);
}
