use serde_json::{Value, json};
use siumai_core::{
    CallOptions, EmbeddingModel, EmbeddingRequest, ErrorKind, Model, ModelFamily, ModelId,
    ModelLookupError, ProviderOptionError, ProviderOptions, ReplayDomain, ReplayDomainId,
    UsageValue,
};
use siumai_provider_alibaba::{
    AlibabaChatOptions, AlibabaConfigError, AlibabaCredential, AlibabaEmbeddingOptions,
    AlibabaEmbeddingOutputType, AlibabaEmbeddingTextType, AlibabaProvider,
    AlibabaWorkspaceEndpoint,
};
use siumai_transport::{EndpointConfig, OfficialOrigin};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const EMBEDDING_PATH: &str = "/api/v1/services/embeddings/text-embedding/text-embedding";

fn provider(server: &MockServer) -> AlibabaProvider {
    AlibabaProvider::builder(AlibabaCredential::api_key("test-key"))
        .with_embedding_endpoint(
            EndpointConfig::local_explicit(format!("{}/api/v1", server.uri())).unwrap(),
        )
        .build()
        .unwrap()
}

#[test]
fn construction_requires_an_explicit_family_endpoint() {
    let error = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .build()
        .unwrap_err();

    assert!(matches!(error, AlibabaConfigError::MissingEndpoint));
}

#[test]
fn workspace_origin_derives_family_endpoints_without_modeling_regions() {
    let workspace = AlibabaWorkspaceEndpoint::public_origin(
        "https://workspace-id.ap-southeast-1.maas.aliyuncs.com",
    )
    .unwrap();

    assert_eq!(
        workspace.language_endpoint().expose_base_url().as_str(),
        "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    );
    assert_eq!(
        workspace.embedding_endpoint().expose_base_url().as_str(),
        "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/api/v1"
    );
    assert_eq!(
        workspace.messages_endpoint().expose_base_url().as_str(),
        "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/apps/anthropic"
    );
}

#[test]
fn direct_and_registered_embedding_models_share_the_exact_scope() {
    let endpoint = EndpointConfig::local_explicit("http://127.0.0.1:9/api/v1").unwrap();
    let provider = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_embedding_endpoint(endpoint)
        .build()
        .unwrap();

    let direct = provider.embedding("future-embedding-model").unwrap();
    let registration = provider.embedding_registration().unwrap();
    let registered = registration
        .embedding_model(ModelId::new("future-embedding-model").unwrap())
        .unwrap();

    assert_eq!(direct.descriptor(), registered.descriptor());
    assert_eq!(
        direct.descriptor().instance_id(),
        registered.descriptor().instance_id()
    );

    let separately_built = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_embedding_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/api/v1").unwrap(),
        )
        .build()
        .unwrap();
    let separate = separately_built
        .embedding("future-embedding-model")
        .unwrap();
    assert_ne!(
        direct.descriptor().instance_id(),
        separate.descriptor().instance_id()
    );
    assert_eq!(direct.provider_id().as_str(), "alibaba");
    assert_eq!(direct.descriptor().platform(), Some("local"));
    assert_eq!(direct.descriptor().protocol(), Some("alibaba-native"));
    assert_eq!(direct.descriptor().api_mode(), Some("text-embedding"));
    assert!(registration.supports_family(ModelFamily::Embedding));
    assert!(!registration.supports_family(ModelFamily::Language));
    assert!(matches!(
        provider.language("qwen-future").unwrap_err(),
        ModelLookupError::UnsupportedFamily {
            family: ModelFamily::Language,
            ..
        }
    ));
}

#[test]
fn default_registration_keeps_language_and_embedding_bindings_distinct() {
    let provider = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_language_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/compatible-mode/v1").unwrap(),
        )
        .with_embedding_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/api/v1").unwrap(),
        )
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("test-endpoint").unwrap(),
        ))
        .build()
        .unwrap();
    let registration = provider.registration().expect("default registration");

    assert!(registration.supports_family(ModelFamily::Language));
    assert!(registration.supports_family(ModelFamily::Embedding));
    assert_eq!(
        registration
            .api_mode(ModelFamily::Language)
            .map(siumai_core::ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(
        registration
            .api_mode(ModelFamily::Embedding)
            .map(siumai_core::ApiModeId::as_str),
        Some("text-embedding")
    );
    assert_ne!(
        registration.scope(ModelFamily::Language),
        registration.scope(ModelFamily::Embedding)
    );
}

#[test]
fn embedding_support_evidence_is_verified_only_for_the_provider_owned_endpoint() {
    let caller_declared_origin = OfficialOrigin::new("https://workspace-id.example.com").unwrap();
    let caller_declared_endpoint = EndpointConfig::official(
        "https://workspace-id.example.com/api/v1",
        caller_declared_origin,
    )
    .unwrap();
    let custom = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_embedding_endpoint(caller_declared_endpoint)
        .build()
        .unwrap();
    assert!(
        custom
            .support_manifest()
            .profiles()
            .iter()
            .any(|profile| profile.generic_claim().is_some())
    );
    assert_eq!(
        custom
            .embedding("text-embedding-v4")
            .unwrap()
            .descriptor()
            .platform(),
        Some("custom-endpoint")
    );

    let legacy = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_legacy_singapore_embedding()
        .build()
        .unwrap();
    assert!(
        legacy
            .support_manifest()
            .profiles()
            .iter()
            .any(|profile| profile.verified_claims().is_some())
    );
}

#[tokio::test]
async fn native_embedding_maps_options_orders_results_and_preserves_sparse_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDING_PATH))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "embedding-request-42",
            "output": {
                "embeddings": [
                    {
                        "text_index": 1,
                        "embedding": vec![0.2_f32; 128],
                        "sparse_embedding": [{"index": 9, "value": 0.75}]
                    },
                    {
                        "text_index": 0,
                        "embedding": vec![0.1_f32; 128],
                        "sparse_embedding": [{"index": 3, "value": 0.5}]
                    }
                ]
            },
            "usage": {"total_tokens": 0}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let request = EmbeddingRequest::new(["first", "second"])
        .unwrap()
        .with_dimensions(128)
        .unwrap();
    let provider_options = AlibabaEmbeddingOptions::new()
        .with_text_type(AlibabaEmbeddingTextType::Query)
        .with_output_type(AlibabaEmbeddingOutputType::DenseAndSparse)
        .with_instruct("Represent the query for retrieving relevant documents");
    let model = provider(&server).embedding("text-embedding-v4").unwrap();
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &provider_options)
        .unwrap();
    let response = model.embed(request, call_options).await.unwrap();

    assert_eq!(response.embeddings[0], vec![0.1_f32; 128]);
    assert_eq!(response.embeddings[1], vec![0.2_f32; 128]);
    assert_eq!(
        response.metadata.request_id.as_deref(),
        Some("embedding-request-42")
    );
    assert_eq!(response.usage.input_tokens, UsageValue::Known(0));
    assert_eq!(response.usage.total_tokens, UsageValue::Known(0));
    assert!(response.warnings.is_empty());
    assert_eq!(
        response.provider["alibaba"]["sparse_embeddings"][0],
        json!([{"index": 3, "value": 0.5}])
    );
    assert_eq!(
        response.provider["alibaba"]["sparse_embeddings"][1],
        json!([{"index": 9, "value": 0.75}])
    );

    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], json!("text-embedding-v4"));
    assert_eq!(body["input"]["texts"], json!(["first", "second"]));
    assert_eq!(body["parameters"]["text_type"], json!("query"));
    assert_eq!(body["parameters"]["dimension"], json!(128));
    assert_eq!(body["parameters"]["output_type"], json!("dense&sparse"));
    assert_eq!(
        body["parameters"]["instruct"],
        json!("Represent the query for retrieving relevant documents")
    );
}

#[tokio::test]
async fn text_embedding_v3_accepts_its_documented_dense_and_sparse_mode() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDING_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "embeddings": [{
                    "text_index": 0,
                    "embedding": vec![0.5_f32; 512],
                    "sparse_embedding": [{"index": 1, "value": 0.25}]
                }]
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let output =
        AlibabaEmbeddingOptions::new().with_output_type(AlibabaEmbeddingOutputType::DenseAndSparse);
    let model = provider(&server).embedding("text-embedding-v3").unwrap();
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &output)
        .unwrap();
    let response = model
        .embed(
            EmbeddingRequest::single("hello")
                .unwrap()
                .with_dimensions(512)
                .unwrap(),
            call_options,
        )
        .await
        .unwrap();

    assert_eq!(response.embeddings[0].len(), 512);
}

#[tokio::test]
async fn unknown_embedding_models_remain_callable_with_unknown_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDING_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "embeddings": [{"text_index": 0, "embedding": [0.25, 0.75]}]
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let model = provider(&server).embedding("future-embedding-v5").unwrap();
    assert_eq!(model.limits().max_inputs, Some(10));
    let response = model
        .embed(
            EmbeddingRequest::single("hello").unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(response.usage.input_tokens, UsageValue::Unknown);
    assert_eq!(response.usage.total_tokens, UsageValue::Unknown);
    assert!(response.warnings.is_empty());
}

#[tokio::test]
async fn known_limits_dimensions_output_modes_and_option_context_fail_before_wire() {
    let server = MockServer::start().await;
    let provider = provider(&server);

    let too_many = (0..11).map(|index| format!("input-{index}"));
    let error = provider
        .embedding("text-embedding-v3")
        .unwrap()
        .embed(
            EmbeddingRequest::new(too_many).unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);

    let error = provider
        .embedding("text-embedding-v4")
        .unwrap()
        .embed(
            EmbeddingRequest::single("hello")
                .unwrap()
                .with_dimensions(192)
                .unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    let sparse_only =
        AlibabaEmbeddingOptions::new().with_output_type(AlibabaEmbeddingOutputType::SparseOnly);
    let error = ProviderOptions::typed(&sparse_only).unwrap_err();
    assert!(error.to_string().contains("sparse-only"));

    let chat_options = AlibabaChatOptions::new().with_enable_search(true);
    let embedding = provider.embedding("text-embedding-v4").unwrap();
    let error = CallOptions::default()
        .with_provider_options_for(&embedding, &chat_options)
        .unwrap_err();
    assert!(matches!(
        error,
        ProviderOptionError::TargetMismatch {
            expected_family: ModelFamily::Embedding,
            actual_family: ModelFamily::Language,
            ..
        }
    ));

    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn native_embedding_errors_expose_safe_code_and_request_id_only() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDING_PATH))
        .respond_with(
            ResponseTemplate::new(400)
                .insert_header("x-private-canary", "canary-header-secret")
                .set_body_json(json!({
                    "code": "InvalidParameter",
                    "request_id": "embedding-error-42",
                    "message": "canary-body-secret"
                })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let error = provider(&server)
        .embedding("text-embedding-v4")
        .unwrap()
        .embed(
            EmbeddingRequest::single("hello").unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        error.diagnostics().and_then(|value| value.provider_code()),
        Some("InvalidParameter")
    );
    assert_eq!(
        error.diagnostics().and_then(|value| value.request_id()),
        Some("embedding-error-42")
    );
    let debug = format!("{error:?}");
    let display = error.to_string();
    assert!(!debug.contains("canary-header-secret"));
    assert!(!debug.contains("canary-body-secret"));
    assert!(!display.contains("canary-header-secret"));
    assert!(!display.contains("canary-body-secret"));
}
