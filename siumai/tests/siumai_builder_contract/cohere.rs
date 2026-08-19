use siumai::providers::cohere::{
    CohereEmbeddingModel, CohereProvider, CohereProviderBuilder, CohereRerankModel,
};
use siumai::transport::EndpointConfig;
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "cohere-facade-credential-canary";

#[test]
fn api_key_path_binds_embedding_and_rerank_models() -> Result<(), Box<dyn std::error::Error>> {
    let custom_endpoint = EndpointConfig::public_custom("https://cohere.example.test/v2")?;
    let hub = Siumai::builder()
        .cohere()
        .api_key("test-api-key")
        .configure_provider(|builder: CohereProviderBuilder| builder.with_endpoint(custom_endpoint))
        .build()?;

    let embedding = hub.embedding("future-cohere-embedding-model")?;
    let rerank = hub.rerank("future-cohere-rerank-model")?;
    let _: &CohereProvider = hub.provider();
    let _: &CohereEmbeddingModel = embedding.model();
    let _: &CohereRerankModel = rerank.model();
    assert!(std::ptr::eq(hub.provider(), embedding.provider()));
    assert!(std::ptr::eq(hub.provider(), rerank.provider()));
    assert_eq!(embedding.family(), ModelFamily::Embedding);
    assert_eq!(rerank.family(), ModelFamily::Rerank);
    assert_eq!(embedding.descriptor().api_mode(), Some("v2"));
    assert_eq!(rerank.descriptor().api_mode(), Some("v2"));
    assert!(
        hub.provider()
            .profile()
            .provider_profile()
            .verified_claims()
            .is_none()
    );
    assert_eq!(
        embedding.descriptor().instance_id(),
        rerank.descriptor().instance_id()
    );
    assert_eq!(
        embedding.descriptor().model().as_str(),
        "future-cohere-embedding-model"
    );
    assert_eq!(
        rerank.descriptor().model().as_str(),
        "future-cohere-rerank-model"
    );

    Ok(())
}

#[test]
fn stages_hubs_clients_and_error_chains_redact_cohere_credentials()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().cohere().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.embedding("future-cohere-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .cohere()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned API-key validator must reject control characters");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}
