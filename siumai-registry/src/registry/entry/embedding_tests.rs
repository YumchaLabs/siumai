use super::*;

#[tokio::test]
async fn embedding_model_handle_builds_client() {
    // Create registry with test provider factory
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_embed".to_string(),
        Arc::new(crate::registry::factories::TestProviderFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.embedding_model("testprov_embed:model").unwrap();

    // Client is built on each call (no caching)
    let out = handle.embed(vec!["a".into(), "b".into()]).await.unwrap();
    assert_eq!(out.embeddings[0][0], 2.0);
}
