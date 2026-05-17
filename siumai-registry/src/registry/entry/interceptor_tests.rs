#![allow(clippy::await_holding_lock)]

use super::*;
use crate::execution::http::interceptor::LoggingInterceptor;
use std::collections::HashMap;
use std::sync::Arc;

fn registry_with_logging_interceptor(
    providers: HashMap<String, Arc<dyn ProviderFactory>>,
) -> ProviderRegistryHandle {
    create_provider_registry(
        providers,
        Some(RegistryOptions {
            http_interceptors: vec![Arc::new(LoggingInterceptor)],
            auto_middleware: false,
            ..Default::default()
        }),
    )
}

#[tokio::test]
async fn language_model_inherits_registry_interceptors() {
    let _g = reg_test_guard();
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );

    let reg = registry_with_logging_interceptor(providers);
    let handle = reg.language_model("testprov:model").unwrap();

    assert_eq!(handle.http_interceptors.len(), 1);
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
    providers.insert(
        "testprov_image".to_string(),
        Arc::new(TestImageProviderFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = registry_with_logging_interceptor(providers);
    let eh = reg.embedding_model("testprov_embed:model").unwrap();
    assert_eq!(eh.http_interceptors.len(), 1);
    let _ = siumai_core::embedding::EmbeddingModel::embed(
        &eh,
        crate::types::EmbeddingRequest::single("hello"),
    )
    .await
    .unwrap();

    let ih = reg.image_model("testprov_image:model").unwrap();
    assert_eq!(ih.http_interceptors.len(), 1);
}
