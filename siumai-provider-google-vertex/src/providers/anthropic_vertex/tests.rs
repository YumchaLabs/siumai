use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use siumai_anthropic_compatible::{MessagesCallOptions, MessagesRequestPolicy};
use siumai_core::{
    CallOptions, ContentAnnotationTarget, ContentPart, Error, ErrorKind, LanguageModel,
    LanguageRequest, LanguageStreamEvent, MediaData, MediaPart, Message, MessagePart, MessageRole,
    ModelAdvisory, ModelId, ModelOperation, Provider, StreamTerminal, StructuredOutputSpec,
    SupportState, ToolSpec, TypedProviderAnnotation, VerifiedFidelity,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, MessagesRequestOptions, OutputEffort, ServerFallback, ServerFallbacks,
    ThinkingConfig, encode_request_with_resolver,
};
use siumai_transport::{EndpointConfig, NoAuth};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::annotations::{
    GoogleVertexAnthropicAnnotationResolver, GoogleVertexAnthropicCacheTtl,
    GoogleVertexAnthropicContentCache, GoogleVertexAnthropicMessageCache,
    GoogleVertexAnthropicTool, GoogleVertexAnthropicToolOptions,
};
use super::auth::{GoogleVertexCredential, GoogleVertexTokenSource};
use super::endpoint::{endpoint_host, official_endpoint};
use super::models::{
    CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_5_20251101, CLAUDE_OPUS_5, CLAUDE_SONNET_5,
    current_models,
};
use super::provider::GoogleVertexAnthropicProvider;
use super::request_policy::GoogleVertexAnthropicRequestPolicy;

fn request(text: &str, max_tokens: u64) -> LanguageRequest {
    let mut request = LanguageRequest::new(vec![Message::text(MessageRole::User, text)]);
    request.generation.max_output_tokens = Some(max_tokens);
    request
}

fn response(model: &str, id: &str, text: &str) -> serde_json::Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
}

fn projected_body(model: &str, text: &str, max_tokens: u64, stream: bool) -> serde_json::Value {
    let _ = model;
    json!({
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }],
        "stream": stream
    })
}

fn local_provider(
    server: &MockServer,
    auth: Arc<dyn siumai_transport::AuthApplier>,
) -> GoogleVertexAnthropicProvider {
    let endpoint = EndpointConfig::local_explicit(format!(
        "{}/v1/projects/test-project/locations/us-central1/publishers/anthropic/",
        server.uri()
    ))
    .expect("local endpoint");
    GoogleVertexAnthropicProvider::builder_with_auth("test-project", "us-central1", auth)
        .with_endpoint(endpoint)
        .build()
        .expect("provider")
}

#[test]
fn official_endpoint_supports_global_multi_region_and_regional_hosts() {
    assert_eq!(endpoint_host("global"), "aiplatform.googleapis.com");
    assert_eq!(endpoint_host("eu"), "aiplatform.eu.rep.googleapis.com");
    assert_eq!(endpoint_host("us"), "aiplatform.us.rep.googleapis.com");
    assert_eq!(
        endpoint_host("us-central1"),
        "us-central1-aiplatform.googleapis.com"
    );

    for (location, host) in [
        ("global", "aiplatform.googleapis.com"),
        ("eu", "aiplatform.eu.rep.googleapis.com"),
        ("us", "aiplatform.us.rep.googleapis.com"),
        ("us-central1", "us-central1-aiplatform.googleapis.com"),
    ] {
        let endpoint = official_endpoint("test-project", location).expect("official endpoint");
        assert_eq!(endpoint.audience().host(), host);
        assert_eq!(
            endpoint.expose_base_url().as_str(),
            format!(
                "https://{host}/v1/projects/test-project/locations/{location}/publishers/anthropic/"
            )
        );
        assert!(endpoint.policy().official_origin().is_some());
    }

    assert!(official_endpoint("unsafe/project", "global").is_err());
    assert!(official_endpoint("test-project", "us/central1").is_err());
    assert!(official_endpoint("test-project", "-us-central1").is_err());
}

#[test]
fn provider_identity_catalog_and_custom_profile_are_explicit() {
    let provider = GoogleVertexAnthropicProvider::builder(
        "test-project",
        "global",
        GoogleVertexCredential::access_token("offline-token"),
    )
    .build()
    .expect("network-free provider construction");
    assert_eq!(provider.provider_id().as_str(), "google");
    let claim = &provider
        .profile()
        .provider_profile()
        .verified_claims()
        .expect("verified claim")[0];
    assert_eq!(claim.scope().platform().as_str(), "vertex-ai");
    assert_eq!(claim.scope().api_mode().as_str(), "messages");
    assert_eq!(claim.fidelity(), VerifiedFidelity::Compatible);
    assert_eq!(current_models()[0], CLAUDE_OPUS_5);
    for model in current_models() {
        assert!(provider.language(model).is_ok());
    }
    let unknown = ModelId::new("claude-future-2030").expect("model");
    assert!(matches!(
        provider
            .registration()
            .evaluate(unknown, ModelOperation::Generate)
            .state(),
        SupportState::Unknown
    ));
    let current = ModelId::new(CLAUDE_OPUS_4_5_20251101).expect("model");
    assert!(matches!(
        provider
            .registration()
            .evaluate(current, ModelOperation::Generate)
            .state(),
        SupportState::Supported
    ));
    let deprecated = provider.registration().evaluate(
        ModelId::new(CLAUDE_OPUS_4_1_20250805).expect("model"),
        ModelOperation::Generate,
    );
    assert!(matches!(deprecated.state(), SupportState::Supported));
    assert!(matches!(
        deprecated.advisories(),
        [ModelAdvisory::Deprecated { .. }]
    ));

    let endpoint =
        EndpointConfig::local_explicit("http://127.0.0.1:9/v1/models/").expect("local endpoint");
    let custom = GoogleVertexAnthropicProvider::builder_with_auth(
        "test-project",
        "global",
        Arc::new(NoAuth),
    )
    .with_endpoint(endpoint)
    .build()
    .expect("custom provider");
    assert!(
        custom
            .profile()
            .provider_profile()
            .verified_claims()
            .is_none()
    );
    assert!(
        custom
            .profile()
            .provider_profile()
            .generic_claim()
            .is_some()
    );
}

#[test]
fn credentials_and_model_paths_are_secret_safe_and_fail_closed() {
    let credential = GoogleVertexCredential::access_token("vertex-canary-secret");
    assert!(!format!("{credential:?}").contains("vertex-canary-secret"));
    let provider = GoogleVertexAnthropicProvider::builder("test-project", "global", credential)
        .build()
        .expect("provider");
    assert!(!format!("{provider:?}").contains("vertex-canary-secret"));
    for model in [
        "../escape",
        "model/child",
        "model:rawPredict",
        "model%2fchild",
    ] {
        assert!(provider.language(model).is_err());
    }
}

#[tokio::test]
async fn generate_projects_vertex_target_body_and_bearer_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!(
            "/v1/projects/test-project/locations/us-central1/publishers/anthropic/models/{CLAUDE_SONNET_5}:rawPredict"
        )))
        .and(header("authorization", "Bearer test-google-token"))
        .and(header("accept", "application/json"))
        .and(body_json(projected_body(
            CLAUDE_SONNET_5,
            "hello",
            64,
            false,
        )))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            CLAUDE_SONNET_5,
            "msg_vertex",
            "ok",
        )))
        .mount(&server)
        .await;
    let auth = GoogleVertexCredential::access_token("test-google-token")
        .into_auth()
        .expect("auth");
    let provider = local_provider(&server, auth);
    let generated = provider
        .language(CLAUDE_SONNET_5)
        .expect("model")
        .generate(request("hello", 64), CallOptions::default())
        .await
        .expect("generate");
    assert_eq!(generated.id(), Some("msg_vertex"));

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    assert!(requests[0].headers.get("anthropic-version").is_none());
    assert!(requests[0].headers.get("anthropic-beta").is_none());
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("body");
    assert!(body.get("model").is_none());
    assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
}

#[tokio::test]
async fn stream_uses_stream_raw_predict_and_one_terminal_outcome() {
    let server = MockServer::start().await;
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": CLAUDE_SONNET_5,
                "usage": {"input_tokens": 2}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 1}
        }),
        json!({"type": "message_stop"}),
    ];
    let sse = frames
        .into_iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path(format!(
            "/v1/projects/test-project/locations/us-central1/publishers/anthropic/models/{CLAUDE_SONNET_5}:streamRawPredict"
        )))
        .and(header("accept", "text/event-stream"))
        .and(body_json(projected_body(
            CLAUDE_SONNET_5,
            "hello",
            64,
            true,
        )))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;
    let provider = local_provider(&server, Arc::new(NoAuth));
    let events = provider
        .language(CLAUDE_SONNET_5)
        .expect("model")
        .stream(request("hello", 64), CallOptions::default())
        .await
        .expect("stream")
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::TextDelta { delta, .. } if delta == "hello"
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LanguageStreamEvent::Terminal(_)))
            .count(),
        1
    );
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        ))
    ));
}

#[derive(Debug)]
struct FailingTokenSource;

#[async_trait]
impl GoogleVertexTokenSource for FailingTokenSource {
    async fn token(&self) -> Result<String, Error> {
        Err(Error::new(
            ErrorKind::Configuration,
            "vertex-token-canary-secret",
        ))
    }
}

#[tokio::test]
async fn token_provider_failures_are_sanitized_before_network() {
    let server = MockServer::start().await;
    let auth = GoogleVertexCredential::dynamic(Arc::new(FailingTokenSource))
        .into_auth()
        .expect("auth");
    let provider = local_provider(&server, auth);
    let error = provider
        .language(CLAUDE_SONNET_5)
        .expect("model")
        .generate(request("hello", 64), CallOptions::default())
        .await
        .expect_err("token failure");
    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert!(!format!("{error:?}").contains("vertex-token-canary-secret"));
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForeignAnthropicCache {
    enabled: bool,
}

impl TypedProviderAnnotation for ForeignAnthropicCache {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "anthropic";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[test]
fn google_annotations_encode_cache_and_ignore_foreign_namespaces() {
    let google = MessagePart::text("google cache")
        .with_provider_annotation(&GoogleVertexAnthropicContentCache::five_minutes())
        .expect("annotation");
    let foreign = MessagePart::text("foreign cache")
        .with_provider_annotation(&ForeignAnthropicCache { enabled: true })
        .expect("annotation");
    let tool = ToolSpec::new("lookup", None, json!({"type": "object"}))
        .expect("tool")
        .with_provider_annotation(
            &GoogleVertexAnthropicToolOptions::new()
                .with_cache_ttl(GoogleVertexAnthropicCacheTtl::OneHour)
                .with_strict(true),
        )
        .expect("tool annotation");
    let memory =
        GoogleVertexAnthropicToolOptions::for_tool(GoogleVertexAnthropicTool::memory_20250818())
            .into_tool_spec()
            .expect("memory tool");
    let mut request =
        LanguageRequest::new(vec![Message::new(MessageRole::User, [google, foreign])]);
    request.generation.max_output_tokens = Some(64);
    request.tools.push(tool);
    request.tools.push(memory);
    let body = encode_request_with_resolver(
        &ModelId::new(CLAUDE_SONNET_5).expect("model"),
        &request,
        &MessagesRequestOptions::new(false),
        &GoogleVertexAnthropicAnnotationResolver,
    )
    .expect("encoded");
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["ttl"],
        "5m"
    );
    assert!(
        body["messages"][0]["content"][1]
            .get("cache_control")
            .is_none()
    );
    assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(body["tools"][1]["type"], "memory_20250818");
}

#[test]
fn request_policy_tracks_current_vertex_feature_boundaries() {
    let policy = GoogleVertexAnthropicRequestPolicy;
    let mut options = MessagesCallOptions::new();
    let mut structured = request("hello", 64);
    structured.structured_output = Some(StructuredOutputSpec {
        name: "answer".to_string(),
        description: None,
        schema: json!({"type": "object"}),
        strict: true,
    });
    assert!(
        policy
            .prepare(
                &ModelId::new(CLAUDE_SONNET_5).expect("model"),
                &structured,
                &mut options,
            )
            .is_ok()
    );
    assert!(
        policy
            .prepare(
                &ModelId::new("claude-future-2030").expect("model"),
                &structured,
                &mut options,
            )
            .is_err()
    );

    let url_media = MessagePart::new(ContentPart::Media(MediaPart {
        media_type: "image/png".to_string(),
        data: MediaData::Url("https://example.com/private.png".to_string()),
        name: None,
    }));
    let mut url_request = LanguageRequest::new(vec![Message::new(MessageRole::User, [url_media])]);
    url_request.generation.max_output_tokens = Some(64);
    let error = policy
        .prepare(
            &ModelId::new(CLAUDE_SONNET_5).expect("model"),
            &url_request,
            &mut MessagesCallOptions::new(),
        )
        .expect_err("URL media must fail locally");
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    let mut effort = MessagesCallOptions::new().with_output_effort(OutputEffort::Max);
    assert!(
        policy
            .prepare(
                &ModelId::new(CLAUDE_OPUS_5).expect("model"),
                &request("effort", 64),
                &mut effort,
            )
            .is_ok()
    );
    let mut unsupported_effort = MessagesCallOptions::new().with_output_effort(OutputEffort::High);
    assert!(
        policy
            .prepare(
                &ModelId::new("claude-haiku-4-5@20251001").expect("model"),
                &request("effort", 64),
                &mut unsupported_effort,
            )
            .is_err()
    );

    let mut top_k = MessagesCallOptions::new().with_top_k(10);
    assert!(
        policy
            .prepare(
                &ModelId::new(CLAUDE_OPUS_5).expect("model"),
                &request("sampling", 64),
                &mut top_k,
            )
            .is_err()
    );

    let fallback = ServerFallback::new(CLAUDE_SONNET_5).expect("fallback");
    let mut fallbacks = MessagesCallOptions::new()
        .with_fallbacks(ServerFallbacks::explicit(vec![fallback]).expect("fallback chain"));
    assert!(
        policy
            .prepare(
                &ModelId::new(CLAUDE_OPUS_5).expect("model"),
                &request("fallback", 64),
                &mut fallbacks,
            )
            .is_err()
    );

    let mut manual_thinking =
        MessagesCallOptions::new().with_thinking(ThinkingConfig::enabled(1_024));
    assert!(
        policy
            .prepare(
                &ModelId::new(CLAUDE_OPUS_5).expect("model"),
                &request("thinking", 4_096),
                &mut manual_thinking,
            )
            .is_err()
    );

    let mut mid_conversation = request("hello", 64);
    mid_conversation
        .messages
        .push(Message::text(MessageRole::System, "late system"));
    assert!(
        policy
            .prepare(
                &ModelId::new(CLAUDE_SONNET_5).expect("model"),
                &mid_conversation,
                &mut MessagesCallOptions::new(),
            )
            .is_err()
    );

    assert!(
        serde_json::from_value::<GoogleVertexAnthropicToolOptions>(json!({
            "anthropic_tool": {"type": "code_execution_20260521"}
        }))
        .is_err()
    );
}

#[test]
fn one_hour_cache_is_model_gated_and_message_annotations_are_google_owned() {
    let message = Message::text(MessageRole::User, "cache me")
        .with_provider_annotation(&GoogleVertexAnthropicMessageCache::one_hour())
        .expect("annotation");
    let mut request = LanguageRequest::new(vec![message]);
    request.generation.max_output_tokens = Some(64);
    let policy = GoogleVertexAnthropicRequestPolicy;
    assert!(
        policy
            .prepare(
                &ModelId::new(CLAUDE_OPUS_5).expect("model"),
                &request,
                &mut MessagesCallOptions::new(),
            )
            .is_ok()
    );
    assert!(
        policy
            .prepare(
                &ModelId::new("claude-future-2030").expect("model"),
                &request,
                &mut MessagesCallOptions::new(),
            )
            .is_err()
    );
}
