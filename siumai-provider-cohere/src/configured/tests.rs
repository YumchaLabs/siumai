use std::time::Instant;

use siumai_core::{
    ApiStability, CallOptions, Cancellation, EmbeddingModel, EmbeddingRequest, ErrorDetail,
    ErrorKind, Model, ModelFamily, ModelId, RerankCandidate, RerankModel, RerankRequest,
    ResourceKind, UsageValue, VerifiedFidelity,
};
use siumai_transport::{EndpointConfig, RetryPolicy};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::provider_options::{
    CohereEmbeddingInputType, CohereEmbeddingOptions, CohereEmbeddingTruncate, CohereRerankOptions,
};

use super::CohereProvider;

fn test_provider(server: &MockServer) -> CohereProvider {
    test_provider_with_retry(server, RetryPolicy::default())
}

fn test_provider_with_retry(server: &MockServer, retry_policy: RetryPolicy) -> CohereProvider {
    let endpoint = EndpointConfig::local_explicit(format!("{}/v2", server.uri()))
        .expect("loopback Cohere endpoint");
    CohereProvider::builder("test-api-key")
        .with_endpoint(endpoint)
        .with_retry_policy(retry_policy)
        .build()
        .expect("configured Cohere provider")
}

fn candidates() -> Vec<RerankCandidate> {
    vec![
        RerankCandidate::new("sunny day at the beach")
            .expect("candidate")
            .with_id("sunny")
            .expect("candidate ID"),
        RerankCandidate::new("rainy day in the city")
            .expect("candidate")
            .with_id("rainy")
            .expect("candidate ID"),
    ]
}

#[test]
fn static_configuration_is_validated_and_debug_is_redacted() {
    let error = CohereProvider::builder("").build().unwrap_err();
    assert!(matches!(error, super::CohereConfigError::InvalidApiKey));

    let debug = format!("{:?}", CohereProvider::builder("canary-api-key"));
    assert!(!debug.contains("canary-api-key"));
}

#[test]
fn official_profile_covers_current_embedding_and_rerank_models() {
    let provider = CohereProvider::builder("test-api-key")
        .build()
        .expect("configured Cohere provider");
    let profile = provider.profile().provider_profile();
    let claims = profile.verified_claims().expect("verified claims");
    assert_eq!(claims.len(), 2);
    assert!(claims.iter().all(|claim| {
        claim.fidelity() == VerifiedFidelity::Native && claim.stability() == ApiStability::Stable
    }));
    assert_eq!(
        profile.catalog().expect("model catalog").iter().count(),
        crate::models::CURRENT_EMBEDDING_MODELS.len() + crate::models::CURRENT_RERANK_MODELS.len()
    );
}

#[test]
fn canonical_requests_reject_empty_embedding_and_rerank_inputs() {
    assert!(EmbeddingRequest::new(Vec::<String>::new()).is_err());
    assert!(EmbeddingRequest::single("   ").is_err());
    assert!(RerankRequest::new("query", Vec::new()).is_err());
    assert!(RerankRequest::new("   ", candidates()).is_err());
}

#[test]
fn models_share_one_runtime_and_registration_exposes_only_native_families() {
    let provider = CohereProvider::builder("test-api-key")
        .build()
        .expect("configured Cohere provider");
    let embedding = provider.embedding("embed-v4.0").expect("embedding model");
    let rerank = provider.reranker("rerank-v3.5").expect("rerank model");
    assert_eq!(
        std::sync::Arc::as_ptr(&embedding.runtime),
        std::sync::Arc::as_ptr(&rerank.runtime)
    );

    let registration = provider.registration();
    assert_eq!(
        registration.families().collect::<Vec<_>>(),
        vec![ModelFamily::Embedding, ModelFamily::Rerank]
    );
    let erased_embedding = registration
        .embedding_model(ModelId::new("embed-v4.0").expect("model ID"))
        .expect("erased embedding model");
    let erased_rerank = registration
        .rerank_model(ModelId::new("rerank-v3.5").expect("model ID"))
        .expect("erased rerank model");
    assert_eq!(embedding.descriptor(), erased_embedding.descriptor());
    assert_eq!(rerank.descriptor(), erased_rerank.descriptor());
    assert_eq!(
        embedding.descriptor().instance_id(),
        rerank.descriptor().instance_id()
    );
    assert_eq!(
        embedding.descriptor().instance_id(),
        erased_embedding.descriptor().instance_id()
    );

    let separately_built = CohereProvider::builder("test-api-key")
        .build()
        .expect("separately configured Cohere provider");
    let separate_embedding = separately_built
        .embedding("embed-v4.0")
        .expect("embedding model");
    assert_ne!(
        embedding.descriptor().instance_id(),
        separate_embedding.descriptor().instance_id()
    );
    assert!(
        registration
            .embedding_model(ModelId::new("future-embed-model").expect("future model ID"))
            .is_ok()
    );
}

#[tokio::test]
async fn direct_and_erased_models_share_embedding_and_rerank_wire_contracts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/embed"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(header("accept", "application/json"))
        .and(body_json(serde_json::json!({
            "model": "embed-v4.0",
            "embedding_types": ["float"],
            "texts": ["sunny day", "rainy day"],
            "input_type": "search_document",
            "truncate": "END"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "embed-request")
                .set_body_json(serde_json::json!({
                    "id": "embed-response",
                    "embeddings": { "float": [[0.1, 0.2], [0.3, 0.4]] },
                    "meta": {
                        "api_version": { "version": "2" },
                        "billed_units": { "input_tokens": 10 }
                    }
                })),
        )
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_json(serde_json::json!({
            "model": "rerank-v3.5",
            "query": "rainy day",
            "documents": ["sunny day at the beach", "rainy day in the city"],
            "top_n": 2,
            "max_tokens_per_doc": 1000,
            "priority": 1
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "rerank-request")
                .set_body_json(serde_json::json!({
                    "id": "rerank-response",
                    "results": [
                        { "index": 1, "relevance_score": 0.9 },
                        { "index": 0, "relevance_score": 0.1 }
                    ],
                    "meta": {
                        "api_version": { "version": "2" },
                        "billed_units": { "search_units": 1 }
                    }
                })),
        )
        .expect(2)
        .mount(&server)
        .await;

    let provider = test_provider(&server);
    let registration = provider.registration();
    let embedding_request =
        EmbeddingRequest::new(["sunny day", "rainy day"]).expect("embedding request");
    let embedding = provider.embedding("embed-v4.0").expect("embedding model");
    let typed_embedding_options = CohereEmbeddingOptions::new()
        .with_input_type(CohereEmbeddingInputType::SearchDocument)
        .with_truncate(CohereEmbeddingTruncate::End);
    let embedding_options = CallOptions::default()
        .with_provider_options_for(&embedding, &typed_embedding_options)
        .expect("Cohere embedding call options");
    let direct_embedding = embedding
        .embed(embedding_request.clone(), embedding_options.clone())
        .await
        .expect("direct embedding response");
    let erased_embedding = registration
        .embedding_model(ModelId::new("embed-v4.0").expect("model ID"))
        .expect("erased embedding model")
        .embed(embedding_request, embedding_options)
        .await
        .expect("erased embedding response");
    assert_eq!(direct_embedding, erased_embedding);
    assert_eq!(direct_embedding.usage.input_tokens, UsageValue::Known(10));
    assert_eq!(
        direct_embedding.metadata.request_id.as_deref(),
        Some("embed-request")
    );

    let rerank_request = RerankRequest::new("rainy day", candidates())
        .expect("rerank request")
        .with_top_n(2)
        .expect("top n");
    let reranker = provider.reranker("rerank-v3.5").expect("rerank model");
    let typed_rerank_options = CohereRerankOptions::new()
        .with_max_tokens_per_doc(1000)
        .with_priority(1);
    let rerank_options = CallOptions::default()
        .with_provider_options_for(&reranker, &typed_rerank_options)
        .expect("Cohere rerank call options");
    let direct_rerank = reranker
        .rerank(rerank_request.clone(), rerank_options.clone())
        .await
        .expect("direct rerank response");
    let erased_rerank = registration
        .rerank_model(ModelId::new("rerank-v3.5").expect("model ID"))
        .expect("erased rerank model")
        .rerank(rerank_request, rerank_options)
        .await
        .expect("erased rerank response");
    assert_eq!(direct_rerank, erased_rerank);
    assert_eq!(direct_rerank.results[0].candidate_id(), Some("rainy"));
    assert_eq!(
        direct_rerank.usage.provider.get("search_units"),
        Some(&serde_json::json!(1))
    );
    assert_eq!(direct_rerank.usage.input_tokens, UsageValue::Unknown);
}

#[tokio::test]
async fn embedding_usage_preserves_known_zero_and_absence() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/embed"))
        .and(body_json(serde_json::json!({
            "model": "embed-v4.0",
            "embedding_types": ["float"],
            "texts": ["known zero"],
            "input_type": "search_query"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": { "float": [[0.0]] },
            "meta": { "billed_units": { "input_tokens": 0 } }
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/embed"))
        .and(body_json(serde_json::json!({
            "model": "embed-v4.0",
            "embedding_types": ["float"],
            "texts": ["unknown"],
            "input_type": "search_query"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": { "float": [[0.0]] },
            "meta": { "billed_units": {} }
        })))
        .expect(1)
        .mount(&server)
        .await;
    let model = test_provider(&server)
        .embedding("embed-v4.0")
        .expect("embedding model");

    let zero = model
        .embed(
            EmbeddingRequest::single("known zero").expect("embedding request"),
            CallOptions::default(),
        )
        .await
        .expect("known-zero response");
    let unknown = model
        .embed(
            EmbeddingRequest::single("unknown").expect("embedding request"),
            CallOptions::default(),
        )
        .await
        .expect("unknown-usage response");
    assert_eq!(zero.usage.input_tokens, UsageValue::Known(0));
    assert_eq!(unknown.usage.input_tokens, UsageValue::Unknown);
}

#[tokio::test]
async fn provider_limits_and_dimension_conflicts_fail_before_network_io() {
    let provider = CohereProvider::builder("test-api-key")
        .build()
        .expect("configured Cohere provider");
    let embedding = provider.embedding("embed-v4.0").expect("embedding model");
    let embedding_error = embedding
        .embed(
            EmbeddingRequest::new((0..97).map(|index| format!("input {index}")))
                .expect("embedding request"),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(embedding_error.kind(), ErrorKind::LimitExceeded);
    assert!(matches!(
        embedding_error.detail(),
        Some(ErrorDetail::LimitExceeded {
            resource: ResourceKind::EmbeddingInputs,
            actual: 97,
            maximum: 96
        })
    ));

    let conflict = embedding
        .embed(
            EmbeddingRequest::single("hello")
                .expect("embedding request")
                .with_dimensions(256)
                .expect("dimensions"),
            CallOptions::default()
                .with_provider_options_for(
                    &embedding,
                    &CohereEmbeddingOptions::new().with_output_dimension(512),
                )
                .expect("Cohere embedding call options"),
        )
        .await
        .unwrap_err();
    assert_eq!(conflict.kind(), ErrorKind::InvalidInput);

    let rerank = provider.reranker("rerank-v3.5").expect("rerank model");
    let rerank_error = rerank
        .rerank(
            RerankRequest::new(
                "query",
                (0..1001)
                    .map(|index| RerankCandidate::new(format!("candidate {index}")).unwrap())
                    .collect(),
            )
            .expect("rerank request"),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(rerank_error.kind(), ErrorKind::LimitExceeded);
    assert!(matches!(
        rerank_error.detail(),
        Some(ErrorDetail::LimitExceeded {
            resource: ResourceKind::RerankCandidates,
            actual: 1001,
            maximum: 1000
        })
    ));
}

#[tokio::test]
async fn invalid_response_cardinality_and_identity_are_protocol_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": { "float": [[0.1]] },
            "meta": {}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .and(body_json(serde_json::json!({
            "model": "rerank-v3.5",
            "query": "identity query",
            "documents": ["sunny day at the beach", "rainy day in the city"],
            "top_n": 2
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": [
                { "index": 2, "relevance_score": 0.9 },
                { "index": 0, "relevance_score": 0.1 }
            ],
            "meta": {}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .and(body_json(serde_json::json!({
            "model": "rerank-v3.5",
            "query": "partial query",
            "documents": ["sunny day at the beach", "rainy day in the city"],
            "top_n": 2
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": [
                { "index": 1, "relevance_score": 0.9 }
            ],
            "meta": {}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let provider = test_provider(&server);

    let embedding_error = provider
        .embedding("embed-v4.0")
        .expect("embedding model")
        .embed(
            EmbeddingRequest::new(["one", "two"]).expect("embedding request"),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(embedding_error.kind(), ErrorKind::ProtocolViolation);

    let rerank_error = provider
        .reranker("rerank-v3.5")
        .expect("rerank model")
        .rerank(
            RerankRequest::new("identity query", candidates())
                .expect("rerank request")
                .with_top_n(2)
                .expect("top n"),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(rerank_error.kind(), ErrorKind::ProtocolViolation);

    let partial_error = provider
        .reranker("rerank-v3.5")
        .expect("rerank model")
        .rerank(
            RerankRequest::new("partial query", candidates())
                .expect("rerank request")
                .with_top_n(2)
                .expect("top n"),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(partial_error.kind(), ErrorKind::PartialResult);
    assert!(matches!(
        partial_error.detail(),
        Some(ErrorDetail::PartialResult {
            resource: ResourceKind::RerankCandidates,
            expected: 2,
            actual: 1
        })
    ));
}

#[tokio::test]
async fn cancellation_and_deadline_are_propagated_by_call_options() {
    let server = MockServer::start().await;
    let model = test_provider(&server)
        .embedding("embed-v4.0")
        .expect("embedding model");
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let cancelled = model
        .embed(
            EmbeddingRequest::single("cancelled").expect("embedding request"),
            CallOptions::default().with_cancellation(cancellation),
        )
        .await
        .unwrap_err();
    assert_eq!(cancelled.kind(), ErrorKind::Cancelled);

    let timed_out = model
        .embed(
            EmbeddingRequest::single("timed out").expect("embedding request"),
            CallOptions::default().with_deadline(Instant::now()),
        )
        .await
        .unwrap_err();
    assert_eq!(timed_out.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn non_idempotent_posts_are_never_replayed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/embed"))
        .respond_with(ResponseTemplate::new(500).set_body_string("retry canary"))
        .expect(1)
        .mount(&server)
        .await;
    let retry_policy = RetryPolicy::new(3).expect("retry policy");
    let error = test_provider_with_retry(&server, retry_policy)
        .embedding("embed-v4.0")
        .expect("embedding model")
        .embed(
            EmbeddingRequest::single("hello").expect("embedding request"),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Provider);
}

#[tokio::test]
async fn provider_errors_keep_remote_material_out_of_default_diagnostics() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .respond_with(
            ResponseTemplate::new(400)
                .insert_header("x-request-id", "safe-request-id")
                .insert_header("x-private-canary", "canary-header-secret")
                .set_body_string("canary-body-secret"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let error = test_provider(&server)
        .reranker("rerank-v3.5")
        .expect("rerank model")
        .rerank(
            RerankRequest::new("query", candidates()).expect("rerank request"),
            CallOptions::default(),
        )
        .await
        .unwrap_err();

    let debug = format!("{error:?}");
    assert!(!debug.contains("canary-header-secret"));
    assert!(!debug.contains("canary-body-secret"));
    assert_eq!(
        error.diagnostics().and_then(|value| value.status()),
        Some(400)
    );
    assert_eq!(
        error.diagnostics().and_then(|value| value.request_id()),
        Some("safe-request-id")
    );
    let (_, sensitive_body) = error
        .sensitive_response()
        .expect("explicit sensitive response")
        .expose();
    assert_eq!(sensitive_body, b"canary-body-secret");
}
